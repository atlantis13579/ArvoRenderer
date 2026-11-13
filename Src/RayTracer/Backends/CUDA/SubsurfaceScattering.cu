// CUDA-native random-walk subsurface transport (Chiang et al., SIGGRAPH 2016).
// Boundary survival has unit weight after its exponential proposal cancels
// transmittance. Interior collisions retain the material's scattering albedo.
#include <cuda_runtime.h>
#include <cmath>
struct SubsurfaceFlight { float3 Weight; float Distance; unsigned Scattered; };
struct SubsurfaceDirection { float3 Direction; float Pdf; float Weight; };
constexpr float SubsurfacePi = 3.14159265358979323846f;
constexpr float SubsurfaceOpenUniform = 1 - 1.0f / float(1u << 24);
constexpr float SubsurfaceBasisParallel = .999f;
__device__ float SubsurfaceDot(float3 a, float3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
__device__ float3 SubsurfaceCross(float3 a, float3 b)
{ return make_float3(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x); }
__device__ float3 SubsurfaceUnit(float3 v)
{
    const float inverse = 1 / sqrtf(SubsurfaceDot(v, v));
    return make_float3(v.x*inverse, v.y*inverse, v.z*inverse);
}
__device__ SubsurfaceFlight SubsurfaceFreeFlight(float meanFreePath, float3 albedo, float3 direction,
                                                 float endpoint, float variate)
{
    SubsurfaceFlight result{make_float3(1,1,1), endpoint, 0};
    const float pathLength = -log1pf(-fminf(fmaxf(variate, 0), SubsurfaceOpenUniform)) * meanFreePath;
    const float speed = sqrtf(SubsurfaceDot(direction, direction));
    if (pathLength < speed * endpoint)
    {
        result.Distance = pathLength / speed;
        result.Weight = albedo;
        result.Scattered = 1;
    }
    return result;
}
__device__ float SubsurfacePhase(float cosine, float g)
{
    const float magnitude = fabsf(g), aligned = g < 0 ? -cosine : cosine;
    const float denominator = (1-magnitude)*(1-magnitude) + 2*magnitude*(1-fminf(1, fmaxf(-1, aligned)));
    return (1-g*g) / (4*SubsurfacePi*denominator*sqrtf(denominator));
}
__device__ SubsurfaceDirection SubsurfaceSampleDirection(float3 incoming, float g, float2 variates,
                                                         bool importance)
{
    const float v = 2*variates.x - 1, square = v*v, d = 1 + g*v;
    float cosine = importance ? ((1+g*g)*v + g/2*(square+3+g*g*(square-1)))/(d*d) : v;
    cosine = fminf(1, fmaxf(-1, cosine));
    const float sine = sqrtf(fmaxf(0, 1-cosine*cosine)), angle = 2*SubsurfacePi*variates.y;
    const float3 forward = SubsurfaceUnit(incoming);
    const float3 helper = fabsf(forward.y) < SubsurfaceBasisParallel ? make_float3(0,1,0) : make_float3(1,0,0);
    const float3 tangent = SubsurfaceUnit(SubsurfaceCross(helper, forward));
    const float3 bitangent = SubsurfaceCross(forward, tangent);
    const float x = sine*cosf(angle), y = sine*sinf(angle);
    SubsurfaceDirection result;
    result.Direction = SubsurfaceUnit(make_float3(forward.x*cosine+tangent.x*x+bitangent.x*y,
        forward.y*cosine+tangent.y*x+bitangent.y*y, forward.z*cosine+tangent.z*x+bitangent.z*y));
    const float phase = SubsurfacePhase(SubsurfaceDot(forward, result.Direction), g);
    result.Pdf = importance ? phase : 1/(4*SubsurfacePi);
    result.Weight = phase/result.Pdf;
    return result;
}
