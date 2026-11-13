#include <cuda_runtime.h>
namespace ArvoCudaBidirectional
{
// Independent CUDA complete-path MIS. Veach (1997), chapter 10; PBRT 3e,
// "Bidirectional Path Tracing". Log PDFs retain tiny area densities. Internal
// vertex reference factors cancel from every strategy before any accumulation.
// Volume destinations retain inverse-square spreading without a surface cosine.
constexpr unsigned Zero = 0, Area = 1, SolidAngle = 2, Dirac = 3, SurfaceDestination = 0,
                   InfiniteDestination = 1, Volume = 4, VolumeDestination = 2;
constexpr unsigned DeltaScattering = 1, EndpointNotIntersectable = 2;
constexpr unsigned MaximumReflections = 32, MaximumScattering = MaximumReflections + 1,
                   MaximumVertices = MaximumScattering + 2,
                   MaximumStrategies = MaximumVertices + 1;
constexpr unsigned NoMis = 0, BalanceMis = 1, PowerMis = 2;
struct Density
{
    float Log2Value;
    unsigned Measure;
};
struct Vertex
{
    Density Forward, Reverse;
    unsigned Flags, Reserved0, Reserved1, Reserved2;
};
struct Strategy
{
    Density LightEndpoint, CameraEndpoint;
    float LightSelection, SampleCount;
    unsigned Enabled, Reserved;
};
struct Weights
{
    float Values[MaximumStrategies];
    unsigned ActiveStrategies;
};
struct Product
{
    float Log;
    bool Exists;
};
__device__ bool Exists(Density p)
{
    return p.Measure >= Area && p.Measure <= Volume && isfinite(p.Log2Value);
}
__device__ float Comparison(Density p)
{
    return p.Measure == Dirac ? 0 : p.Log2Value;
}
__device__ float Dot(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ Density Make(float value, unsigned measure)
{
    if (!(value > 0) || !isfinite(value) || measure < Area || measure > Volume)
        return {};
    return {log2f(value), measure};
}
__device__ Density Convert(Density p, float3 displacement, float3 geometricNormal, unsigned destination)
{
    if (!Exists(p) || (p.Measure != SolidAngle && p.Measure != Dirac))
        return {};
    if (destination == InfiniteDestination)
        return p;
    if (destination != SurfaceDestination && destination != VolumeDestination)
        return {};
    float scale = fmaxf(fabsf(displacement.x), fmaxf(fabsf(displacement.y), fabsf(displacement.z)));
    if (!(scale > 0) || !isfinite(scale))
        return {};
    if (p.Measure == Dirac)
        return p;
    float3 v = make_float3(displacement.x / scale, displacement.y / scale, displacement.z / scale);
    float length = sqrtf(Dot(v, v));
    float3 direction = make_float3(v.x / length, v.y / length, v.z / length);
    float cosine = destination == VolumeDestination ? 1 : fabsf(Dot(geometricNormal, direction));
    if (!(cosine > 0) || !isfinite(cosine))
        return {};
    return {p.Log2Value + log2f(cosine) - 2 * (log2f(scale) + log2f(length)),
            destination == VolumeDestination ? Volume : Area};
}
// An infinite emitter's perpendicular disk projects onto the first surface.
// It has no inverse-distance-square conversion, unlike angular sampling.
__device__ Density ProjectDisk(Density p, float3 normal, float3 direction)
{
    if (!Exists(p) || p.Measure != Area)
        return {};
    float cosine = fabsf(Dot(normal, direction));
    if (!(cosine > 0) || !isfinite(cosine))
        return {};
    return {p.Log2Value + log2f(cosine), Area};
}
__device__ Product Append(Product product, Density density, float baseline)
{
    if (!product.Exists || !Exists(density))
        return {0, false};
    float value = product.Log + (Comparison(density) - baseline);
    return {value, isfinite(value)};
}
__device__ bool Allowed(const Vertex* vertices, unsigned count, unsigned cut)
{
    if (cut == 0)
        return !(vertices[0].Flags & EndpointNotIntersectable);
    if (cut == count)
        return !(vertices[count - 1].Flags & EndpointNotIntersectable);
    if (cut > 1 && (vertices[cut - 1].Flags & DeltaScattering))
        return false;
    if (cut < count - 1 && (vertices[cut].Flags & DeltaScattering))
        return false;
    return true;
}
// Forward[i] samples x[i] from x[i-1], Reverse[i] from x[i+1], ordered light
// to camera. Re-evaluate PDFs adjacent to the connection on the completed path.
// Dirac branch masses belong in throughput; their comparison factors cancel
// to one. Explicit source-selection PMFs and sample counts are never discarded.
__device__ Weights Evaluate(const Vertex* vertices, unsigned count, const Strategy* strategies,
                            unsigned heuristic)
{
    Weights result{};
    if (count < 2 || count > MaximumVertices || !vertices || !strategies || heuristic > PowerMis)
        return result;
    float common[MaximumVertices]{};
    for (unsigned i = 1; i + 1 < count; ++i)
    {
        const Vertex& v = vertices[i];
        bool f = Exists(v.Forward), r = Exists(v.Reverse);
        common[i] = f ? Comparison(v.Forward) : (r ? Comparison(v.Reverse) : 0);
        if (f && r)
            common[i] = fmaxf(common[i], Comparison(v.Reverse));
    }
    Product prefix[MaximumStrategies], suffix[MaximumStrategies];
    for (unsigned i = 0; i <= count; ++i)
    {
        prefix[i] = {0, true};
        suffix[i] = {0, true};
    }
    for (unsigned s = 2; s <= count; ++s)
        prefix[s] = Append(prefix[s - 1], vertices[s - 1].Forward, common[s - 1]);
    for (unsigned s = count - 1; s > 0; --s)
        suffix[s - 1] = Append(suffix[s], vertices[s - 1].Reverse, common[s - 1]);
    float score[MaximumStrategies]{};
    bool active[MaximumStrategies]{};
    float maximum = 0;
    for (unsigned s = 0; s <= count; ++s)
    {
        const Strategy& strategy = strategies[s];
        if (!strategy.Enabled || !Allowed(vertices, count, s) || !prefix[s].Exists || !suffix[s].Exists ||
            !(strategy.LightSelection > 0 && strategy.SampleCount > 0) ||
            !isfinite(strategy.LightSelection) || !isfinite(strategy.SampleCount) ||
            !Exists(strategy.LightEndpoint) || !Exists(strategy.CameraEndpoint))
            continue;
        score[s] = prefix[s].Log + suffix[s].Log + Comparison(strategy.LightEndpoint) +
                   Comparison(strategy.CameraEndpoint) + log2f(strategy.LightSelection) +
                   log2f(strategy.SampleCount);
        if (!isfinite(score[s]))
            continue;
        active[s] = true;
        maximum = result.ActiveStrategies ? fmaxf(maximum, score[s]) : score[s];
        ++result.ActiveStrategies;
    }
    if (!result.ActiveStrategies)
        return result;
    float sum = 0;
    for (unsigned s = 0; s <= count; ++s)
        if (active[s])
        {
            float exponent = heuristic == PowerMis ? 2.0f : 1.0f;
            result.Values[s] = heuristic == NoMis ? 1 : exp2f(exponent * (score[s] - maximum));
            sum += result.Values[s];
        }
    for (unsigned s = 0; s <= count; ++s)
        result.Values[s] /= sum;
    return result;
}
} // namespace ArvoCudaBidirectional
