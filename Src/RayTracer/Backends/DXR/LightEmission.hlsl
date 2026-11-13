// DXR light endpoints, independently implemented for ray-generation programs.
// PBRT 3e, "The Path-Space Measurement Equation", section 16.1.2.
// No TraceRay is needed until this endpoint starts an importance subpath.
// Value is radiance (area/environment), intensity (point), or irradiance (sun).
// Weight integrates emitted flux; the emitter-selection PMF is applied outside.
static const float EmissionPi = 3.14159265358979323846f;
static const float EmissionAxisThreshold = 0.999f;
static const float3 EmissionRgbWeights = float3(0.2126f,0.7152f,0.0722f);
static const uint EmissionDirectional = 0, EmissionPoint = 1, EmissionRectangle = 2, EmissionEnvironment = 3;
static const uint EmissionValid = 1, EmissionDeltaPosition = 2, EmissionDeltaDirection = 4, EmissionInfinite = 8;
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
float3 EmissionSphereDirection(float vertical, float azimuthSample)
{
    float circle = sqrt(max(0,1-vertical*vertical)), azimuth = 2*EmissionPi*azimuthSample;
    return float3(circle*cos(azimuth),vertical,circle*sin(azimuth));
}
struct LightEmissionAxes { float3 Right, Up; };
LightEmissionAxes EmissionAxes(float3 n)
{
    float3 reference = abs(n.y) < EmissionAxisThreshold ? float3(0,1,0) : float3(1,0,0);
    LightEmissionAxes axes;
    axes.Right = normalize(cross(reference,n)); axes.Up = cross(n,axes.Right);
    return axes;
}
EmissionSample EmptyLightEmission()
{
    EmissionSample value;
    value.Origin=float3(0,0,0); value.Direction=float3(0,0,0); value.Normal=float3(0,0,0);
    value.Value=float3(0,0,0); value.Weight=float3(0,0,0);
    value.PositionPdf=0; value.DirectionPdf=0; value.Flags=0;
    return value;
}
float2 EmissionDiskCoordinates(float2 u)
{
    float distance = sqrt(u.x), angle = 2*EmissionPi*u.y;
    return float2(distance*cos(angle),distance*sin(angle));
}
float2 EmissionSkyWeights(EmissionSource source)
{
    float3 maximum = max(source.BackgroundTop,source.BackgroundBottom);
    float scale = max(maximum.x,max(maximum.y,maximum.z));
    if (!(scale > 0)) return float2(0,0);
    float bottom = dot(source.BackgroundBottom/scale,EmissionRgbWeights);
    float top = dot(source.BackgroundTop/scale,EmissionRgbWeights);
    return float2(bottom,top)/(bottom+top);
}
float EmissionSkyDensity(float2 weights, float y)
{
    return ((1-y)*weights.x+(1+y)*weights.y)/(4*EmissionPi);
}
// Continuous densities on known source support; discrete dimensions are zero.
// The sampled delta masses are one, so Flags must travel with the endpoint.
float2 LightEmissionPdf(EmissionSource source, EmissionDomain domain, float3 direction)
{
    switch (source.Kind)
    {
    case EmissionPoint: return float2(0,1/(4*EmissionPi));
    case EmissionRectangle:
        if (!(source.Size.x > 0 && source.Size.y > 0)) return float2(0,0);
        return float2(1/(source.Size.x*source.Size.y),max(0,dot(normalize(source.Direction),direction))/EmissionPi);
    case EmissionDirectional:
        return domain.Radius > 0 ? float2(1/(EmissionPi*domain.Radius*domain.Radius),0) : float2(0,0);
    case EmissionEnvironment:
        if (!(domain.Radius > 0)) return float2(0,0);
        return float2(1/(EmissionPi*domain.Radius*domain.Radius),EmissionSkyDensity(EmissionSkyWeights(source),-direction.y));
    }
    return float2(0,0);
}
// Directions are nonzero, RGB is finite/nonnegative, both variate pairs [0,1).
// Domain is a positive, strictly enclosing world-space scene sphere.
EmissionSample SampleLightEmission(EmissionSource source, EmissionDomain domain, float2 positionU, float2 directionU)
{
    EmissionSample sample = EmptyLightEmission();
    switch (source.Kind)
    {
    case EmissionPoint:
        sample.Origin = source.Position;
        sample.Direction = EmissionSphereDirection(1-2*directionU.x,directionU.y);
        sample.Normal = sample.Direction;
        sample.Value = source.Strength;
        sample.PositionPdf = 1; sample.DirectionPdf = 1/(4*EmissionPi);
        sample.Weight = source.Strength*(4*EmissionPi);
        sample.Flags = EmissionValid|EmissionDeltaPosition;
        return sample;
    case EmissionRectangle:
    {
        if (!(source.Size.x > 0 && source.Size.y > 0)) return sample;
        float3 n = normalize(source.Direction);
        LightEmissionAxes axes=EmissionAxes(n);
        float3 right=axes.Right,up=axes.Up;
        sample.Normal = n;
        sample.Origin = source.Position + right*((positionU.x-0.5f)*source.Size.x)
                       + up*((positionU.y-0.5f)*source.Size.y);
        float2 disk = EmissionDiskCoordinates(directionU);
        float cosTheta = sqrt(max(0,1-directionU.x));
        if (!(cosTheta > 0)) return EmptyLightEmission();
        sample.Direction = right*disk.x+up*disk.y+n*cosTheta;
        sample.PositionPdf = 1/(source.Size.x*source.Size.y); sample.DirectionPdf = cosTheta/EmissionPi;
        sample.Value = source.Strength*sample.PositionPdf;
        sample.Weight = source.Strength*EmissionPi;
        sample.Flags = EmissionValid;
        return sample;
    }
    }
    if (!(domain.Radius > 0)) return sample;
    float3 outward;
    if (source.Kind == EmissionDirectional)
    {
        outward = normalize(source.Direction); // Scene stores direction TO light.
        sample.Value = source.Strength;
        sample.DirectionPdf = 1;
        sample.Flags = EmissionValid|EmissionDeltaDirection|EmissionInfinite;
    }
    else if (source.Kind == EmissionEnvironment)
    {
        float2 weights = EmissionSkyWeights(source);
        if (!(weights.x+weights.y > 0)) return sample;
        float u = directionU.x;
        // Stable inverse of 2*b*q+(t-b)*q^2, with q=(outward.y+1)/2.
        float root = sqrt((1-u)*weights.x*weights.x+u*weights.y*weights.y);
        float q = weights.x+root > 0 ? saturate(u/(weights.x+root)) : 0;
        outward = EmissionSphereDirection(2*q-1,directionU.y);
        sample.DirectionPdf = EmissionSkyDensity(weights,outward.y);
        if (!(sample.DirectionPdf > 0)) return EmptyLightEmission();
        sample.Value = source.BackgroundTop*q+source.BackgroundBottom*(1-q);
        sample.Flags = EmissionValid|EmissionInfinite;
    }
    else return sample;
    LightEmissionAxes axes=EmissionAxes(outward);
    float3 right=axes.Right,up=axes.Up;
    float2 disk = EmissionDiskCoordinates(positionU);
    sample.Origin = domain.Center+(outward+right*disk.x+up*disk.y)*domain.Radius;
    sample.Direction = -outward; sample.Normal = sample.Direction;
    sample.PositionPdf = 1/(EmissionPi*domain.Radius*domain.Radius);
    sample.Weight = sample.Value/(sample.PositionPdf*sample.DirectionPdf);
    return sample;
}
