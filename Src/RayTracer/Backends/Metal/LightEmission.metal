// Independent Metal emitted-ray sampling for future light subpaths / photons.
// PBRT 3e, "The Path-Space Measurement Equation", section 16.1.2.
// https://pbr-book.org/3ed-2018/Light_Transport_III_Bidirectional_Methods/The_Path-Space_Measurement_Equation
// Values follow demo units: rectangle radiance = Strength / area, point radiant
// intensity = Strength, directional irradiance = Strength. Environment is the
// existing linear RGB gradient. Source-selection probability is external.
constant float EmissionPi = 3.14159265358979323846f;
constant float EmissionParallelThreshold = 0.999f;
constant uint EmissionDirectional = 0, EmissionPoint = 1, EmissionRectangle = 2, EmissionEnvironment = 3;
constant uint EmissionValid = 1, EmissionDeltaPosition = 2, EmissionDeltaDirection = 4, EmissionInfinite = 8;
struct EmissionSource
{
    uint Kind;
    float3 Position, Direction, Strength, BackgroundTop, BackgroundBottom;
    float2 Size;
};
struct EmissionDomain { float3 Center; float Radius; };
struct EmissionSample
{
    float3 Origin, Direction, Normal, Value, Weight;
    float PositionPdf, DirectionPdf;
    uint Flags;
};
void EmissionFrame(float3 n, thread float3& tangent, thread float3& bitangent)
{
    float3 helper = fabs(n.y) < EmissionParallelThreshold ? float3(0,1,0) : float3(1,0,0);
    tangent = normalize(cross(helper,n)); bitangent = cross(n,tangent);
}
float2 EmissionDisk(float2 variates)
{
    float angle = 2*EmissionPi*variates.y, radius = sqrt(variates.x);
    return float2(radius*cos(angle),radius*sin(angle));
}
float3 EmissionLatitude(float y, float variate)
{
    float angle = 2*EmissionPi*variate, radius = sqrt(max(0.0f,1-y*y));
    return float3(radius*cos(angle),y,radius*sin(angle));
}
float2 EmissionGradientWeights(EmissionSource source)
{
    const float3 luminance = float3(0.2126f,0.7152f,0.0722f);
    float3 peak = max(source.BackgroundTop,source.BackgroundBottom);
    float scale = max(peak.x,max(peak.y,peak.z));
    if (!(scale > 0)) return float2(0);
    float bottom = dot(source.BackgroundBottom/scale,luminance), top = dot(source.BackgroundTop/scale,luminance);
    return float2(bottom/(bottom+top),top/(bottom+top));
}
float EmissionGradientPdf(float2 weights, float outwardY)
{
    return ((1-outwardY)*weights.x+(1+outwardY)*weights.y)/(4*EmissionPi);
}
// Query only continuous measures, assuming position on support and unit
// direction. Delta sample masses are one, marked explicitly in sample.Flags.
float2 LightEmissionPdf(EmissionSource source, EmissionDomain domain, float3 direction)
{
    if (source.Kind == EmissionPoint) return float2(0,1/(4*EmissionPi));
    if (source.Kind == EmissionRectangle)
    {
        if (!(source.Size.x > 0 && source.Size.y > 0)) return float2(0);
        return float2(1/(source.Size.x*source.Size.y),max(0.0f,dot(normalize(source.Direction),direction))/EmissionPi);
    }
    if (!(domain.Radius > 0)) return float2(0);
    float diskPdf = 1/(EmissionPi*domain.Radius*domain.Radius);
    if (source.Kind == EmissionDirectional) return float2(diskPdf,0);
    if (source.Kind == EmissionEnvironment)
        return float2(diskPdf,EmissionGradientPdf(EmissionGradientWeights(source),-direction.y));
    return float2(0);
}
// Validated scene inputs: finite/nonnegative colors, nonzero axes, u in [0,1).
// Infinite emitters require a positive sphere strictly enclosing world geometry.
EmissionSample SampleLightEmission(EmissionSource source, EmissionDomain domain, float2 positionU, float2 directionU)
{
    EmissionSample result{};
    if (source.Kind == EmissionPoint)
    {
        result.Origin = source.Position;
        result.Direction = EmissionLatitude(1-2*directionU.x,directionU.y);
        result.Normal = result.Direction; result.Value = source.Strength;
        result.PositionPdf = 1; result.DirectionPdf = 1/(4*EmissionPi);
        result.Weight = source.Strength*(4*EmissionPi);
        result.Flags = EmissionValid|EmissionDeltaPosition;
        return result;
    }
    if (source.Kind == EmissionRectangle)
    {
        if (!(source.Size.x > 0 && source.Size.y > 0)) return result;
        float3 tangent,bitangent;
        result.Normal = normalize(source.Direction);
        EmissionFrame(result.Normal,tangent,bitangent);
        result.Origin = source.Position+tangent*((positionU.x-0.5f)*source.Size.x)
                       +bitangent*((positionU.y-0.5f)*source.Size.y);
        float2 disk = EmissionDisk(directionU);
        float cosine = sqrt(max(0.0f,1-directionU.x));
        if (!(cosine > 0)) return EmissionSample{};
        result.Direction = tangent*disk.x+bitangent*disk.y+result.Normal*cosine;
        result.PositionPdf = 1/(source.Size.x*source.Size.y); result.DirectionPdf = cosine/EmissionPi;
        result.Value = source.Strength*result.PositionPdf;
        result.Weight = source.Strength*EmissionPi;
        result.Flags = EmissionValid;
        return result;
    }
    if (!(domain.Radius > 0)) return result;
    float3 outward;
    if (source.Kind == EmissionDirectional)
    {
        outward = normalize(source.Direction);
        result.Value = source.Strength; result.DirectionPdf = 1;
        result.Flags = EmissionValid|EmissionDeltaDirection|EmissionInfinite;
    }
    else if (source.Kind == EmissionEnvironment)
    {
        float2 weights = EmissionGradientWeights(source);
        if (!(weights.x+weights.y > 0)) return result;
        float u = directionU.x;
        // Rationalize the quadratic root; equal endpoint luminance needs no branch.
        float denominator = weights.x+sqrt((1-u)*weights.x*weights.x+u*weights.y*weights.y);
        float q = denominator > 0 ? clamp(u/denominator,0.0f,1.0f) : 0;
        outward = EmissionLatitude(2*q-1,directionU.y);
        result.DirectionPdf = EmissionGradientPdf(weights,outward.y);
        if (!(result.DirectionPdf > 0)) return EmissionSample{};
        result.Value = source.BackgroundBottom*(1-q)+source.BackgroundTop*q;
        result.Flags = EmissionValid|EmissionInfinite;
    }
    else return result;
    float3 tangent,bitangent; EmissionFrame(outward,tangent,bitangent);
    float2 disk = EmissionDisk(positionU);
    result.Origin = domain.Center+(outward+tangent*disk.x+bitangent*disk.y)*domain.Radius;
    result.Direction = -outward; result.Normal = result.Direction;
    result.PositionPdf = 1/(EmissionPi*domain.Radius*domain.Radius);
    result.Weight = result.Value/(result.PositionPdf*result.DirectionPdf);
    return result;
}
