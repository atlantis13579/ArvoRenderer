// Independent MSL interior free flights and HG phase sampling. Actual closed
// surfaces terminate the random walk; a nonlocal exit emerges from transport.
// Chiang et al., "Practical and Controllable Subsurface Scattering" (2016).
#include <metal_stdlib>
using namespace metal;
struct SubsurfaceFlight { float3 Weight; float Distance; uint Scattered; };
struct SubsurfaceDirection { float3 Direction; float Pdf; float Weight; };
constant float SubsurfacePi = 3.14159265358979323846f;
constant float SubsurfaceOpenUniform = 1 - 1.0f / float(1u << 24);
constant float SubsurfaceParallelCosine = .999f;
SubsurfaceFlight SubsurfaceFreeFlight(float meanFreePath, float3 albedo, float3 direction,
                                      float endpoint, float variate)
{
    const float speed = length(direction);
    const float u = clamp(variate, 0.0f, SubsurfaceOpenUniform);
    const float LogPolynomialThreshold = .01f;
    const float optical = u < LogPolynomialThreshold ? u*(1 + u/2 + u*u/3 + u*u*u/4) : -log(1-u);
    const float distance = optical * meanFreePath / speed;
    SubsurfaceFlight result;
    result.Scattered = distance < endpoint;
    result.Distance = result.Scattered ? distance : endpoint;
    result.Weight = result.Scattered ? albedo : float3(1);
    return result;
}
float SubsurfacePhase(float cosine, float g)
{
    const float magnitude = abs(g), alignedCosine = g < 0 ? -cosine : cosine;
    const float distanceSquared = (1-magnitude)*(1-magnitude) + 2*magnitude*(1-clamp(alignedCosine, -1.0f, 1.0f));
    return (1-g*g)/(4*SubsurfacePi*distanceSquared*sqrt(distanceSquared));
}
SubsurfaceDirection SubsurfaceSampleDirection(float3 incoming, float g, float2 variates, bool importance)
{
    const float centered = 2*variates.x - 1;
    const float divisor = 1 + g*centered;
    const float cubic = centered*(1+g*g) + g/2*(centered*centered+3+g*g*(centered*centered-1));
    const float cosine = clamp(importance ? cubic/(divisor*divisor) : centered, -1.0f, 1.0f);
    const float radius = sqrt(max(0.0f, 1-cosine*cosine)), angle = 2*SubsurfacePi*variates.y;
    const float3 axis = normalize(incoming);
    const float3 reference = abs(axis.y) < SubsurfaceParallelCosine ? float3(0,1,0) : float3(1,0,0);
    const float3 u = normalize(cross(reference, axis)), v = cross(axis, u);
    SubsurfaceDirection result;
    result.Direction = normalize(axis*cosine + u*(radius*cos(angle)) + v*(radius*sin(angle)));
    const float phase = SubsurfacePhase(dot(axis, result.Direction), g);
    result.Pdf = importance ? phase : 1/(4*SubsurfacePi);
    result.Weight = phase/result.Pdf;
    return result;
}
