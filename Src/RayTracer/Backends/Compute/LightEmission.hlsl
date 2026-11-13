// Independent compute emitted-ray sampler. PBRT 3e, section 16.1.2,
// "The Path-Space Measurement Equation". Source selection is external.
// Delta position/direction entries in a sample are probability masses; PDF()
// queries contain only continuous densities. Never compare the two measures.
static const float EmissionPi = 3.14159265358979323846f;
static const float EmissionParallelThreshold = 0.999f;
static const float3 EmissionLuminance = float3(0.2126f, 0.7152f, 0.0722f);
static const uint EmissionDirectional = 0, EmissionPoint = 1, EmissionRectangle = 2, EmissionEnvironment = 3;
static const uint EmissionValid = 1, EmissionDeltaPosition = 2, EmissionDeltaDirection = 4, EmissionInfinite = 8;
struct EmissionDomain { float3 Center; float Radius; };
struct EmissionSource
{
    uint Kind;
    float3 Position, Direction, Strength, BackgroundTop, BackgroundBottom;
    float2 Size;
};
struct EmissionSample
{
    float3 Origin, Direction, Normal, Value, Weight;
    float PositionPdf, DirectionPdf;
    uint Flags;
};
struct LightEmissionFrame { float3 Tangent, Bitangent; };
LightEmissionFrame EmissionFrame(float3 n)
{
    float3 axis = abs(n.y) < EmissionParallelThreshold ? float3(0,1,0) : float3(1,0,0);
    LightEmissionFrame frame;
    frame.Tangent = normalize(cross(axis,n));
    frame.Bitangent = cross(n,frame.Tangent);
    return frame;
}
EmissionSample EmptyLightEmission()
{
    EmissionSample sample;
    sample.Origin=float3(0,0,0); sample.Direction=float3(0,0,0); sample.Normal=float3(0,0,0);
    sample.Value=float3(0,0,0); sample.Weight=float3(0,0,0);
    sample.PositionPdf=0; sample.DirectionPdf=0; sample.Flags=0;
    return sample;
}
float2 EmissionDisk(float2 sample)
{
    float angle = 2*EmissionPi*sample.y;
    return sqrt(sample.x)*float2(cos(angle),sin(angle));
}
float3 EmissionLatitude(float y, float azimuth)
{
    float r = sqrt(max(0,1-y*y)), angle = 2*EmissionPi*azimuth;
    return float3(r*cos(angle),y,r*sin(angle));
}
float2 EmissionGradientWeights(EmissionSource source)
{
    float3 largest = max(source.BackgroundTop,source.BackgroundBottom);
    float scale = max(largest.x,max(largest.y,largest.z));
    if (!(scale > 0)) return float2(0,0);
    float2 luminance = float2(dot(source.BackgroundBottom/scale,EmissionLuminance),
                             dot(source.BackgroundTop/scale,EmissionLuminance));
    return luminance/(luminance.x+luminance.y);
}
float EmissionGradientPdf(float2 weights, float outwardY)
{
    return dot(float2(1-outwardY,1+outwardY),weights)/(4*EmissionPi);
}
// Position lies on the emitter support; direction is normalized. Rectangle
// direction points OUT of the emitter; directional direction points TO light.
float2 LightEmissionPdf(EmissionSource source, EmissionDomain domain, float3 direction)
{
    if (source.Kind == EmissionPoint) return float2(0,1/(4*EmissionPi));
    if (source.Kind == EmissionRectangle)
    {
        if (!(source.Size.x > 0 && source.Size.y > 0)) return float2(0,0);
        return float2(1/(source.Size.x*source.Size.y),max(0,dot(normalize(source.Direction),direction))/EmissionPi);
    }
    if (!(domain.Radius > 0)) return float2(0,0);
    float areaPdf = 1/(EmissionPi*domain.Radius*domain.Radius);
    if (source.Kind == EmissionDirectional) return float2(areaPdf,0);
    if (source.Kind == EmissionEnvironment)
        return float2(areaPdf,EmissionGradientPdf(EmissionGradientWeights(source),-direction.y));
    return float2(0,0);
}
// Finite/nonnegative scene inputs, nonzero axes, random pairs in [0,1).
// Domain strictly bounds scene geometry, including transformed instances.
EmissionSample SampleLightEmission(EmissionSource source, EmissionDomain domain, float2 positionU, float2 directionU)
{
    EmissionSample result = EmptyLightEmission();
    if (source.Kind == EmissionPoint)
    {
        result.Origin = source.Position;
        result.Direction = EmissionLatitude(1-2*directionU.x,directionU.y);
        result.Normal = result.Direction;
        result.Value = source.Strength;
        result.Weight = source.Strength*(4*EmissionPi);
        result.PositionPdf = 1; result.DirectionPdf = 1/(4*EmissionPi);
        result.Flags = EmissionValid|EmissionDeltaPosition;
        return result;
    }
    if (source.Kind == EmissionRectangle)
    {
        if (!(source.Size.x > 0 && source.Size.y > 0)) return result;
        result.Normal = normalize(source.Direction);
        LightEmissionFrame frame=EmissionFrame(result.Normal);
        float3 tangent=frame.Tangent,bitangent=frame.Bitangent;
        float2 offset = (positionU-0.5f)*source.Size;
        result.Origin = source.Position+tangent*offset.x+bitangent*offset.y;
        float2 disk = EmissionDisk(directionU);
        float cosine = sqrt(max(0,1-directionU.x));
        if (!(cosine > 0)) return EmptyLightEmission();
        result.Direction = tangent*disk.x+bitangent*disk.y+result.Normal*cosine;
        result.PositionPdf = 1/(source.Size.x*source.Size.y); result.DirectionPdf = cosine/EmissionPi;
        result.Value = source.Strength*result.PositionPdf;
        // Demo intensity is radiance times area. Total one-sided power = pi*Strength.
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
        // Rationalized inverse of the gradient CDF, including equal endpoint luminance.
        float denominator = weights.x+sqrt((1-u)*weights.x*weights.x+u*weights.y*weights.y);
        float q = denominator > 0 ? saturate(u/denominator) : 0;
        outward = EmissionLatitude(2*q-1,directionU.y);
        result.DirectionPdf = EmissionGradientPdf(weights,outward.y);
        if (!(result.DirectionPdf > 0)) return EmptyLightEmission();
        result.Value = (1-q)*source.BackgroundBottom+q*source.BackgroundTop;
        result.Flags = EmissionValid|EmissionInfinite;
    }
    else return result;
    LightEmissionFrame frame=EmissionFrame(outward);
    float3 tangent=frame.Tangent,bitangent=frame.Bitangent;
    float2 disk = EmissionDisk(positionU);
    result.Origin = domain.Center+domain.Radius*(outward+tangent*disk.x+bitangent*disk.y);
    result.Direction = -outward; result.Normal = result.Direction;
    result.PositionPdf = 1/(EmissionPi*domain.Radius*domain.Radius);
    result.Weight = result.Value/(result.PositionPdf*result.DirectionPdf);
    return result;
}
