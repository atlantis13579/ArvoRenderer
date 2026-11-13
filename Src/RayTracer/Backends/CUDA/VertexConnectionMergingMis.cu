#include <cuda_runtime.h>
#include "VertexConnectionMergingLayout.h"

namespace Arvo::RayTracing::CudaVcm
{
// Independent CUDA joint VC/VM weighting. Georgiev et al. (2012), "Light
// Transport Simulation with Vertex Connection and Merging", and Georgiev's
// "Implementing Vertex Connection and Merging", revision 2, equations 10-12:
// https://www.iliyan.com/publications/ImplementingVCM/ImplementingVCM_TechRep2012_rev2.pdf
//
// A connection samples each interior vertex once. Merging samples x[s-1]
// from both ends; the second arrival density times a disk or sphere kernel
// measure is its acceptance probability. Compare attempted path masses jointly.
// Delta events retain branch masses and geometric factors, while eligibility
// excludes connection cuts beside them and merging at the event itself.
//
// One forward scan caches normalized prefix products. A reverse scan carries
// the camera product and emits both strategy families, without a suffix array.
// Shared per-vertex reference factors cancel BEFORE summation, preserving small
// differences when paths have large common log densities. The extra VM arrival
// is not shared by VC: subtracting its reference would change the estimator.
constexpr double VcmPi = 3.14159265358979323846;
struct LogMass
{
    double Value;
    unsigned Zeros;
};

__device__ bool HasDensity(Density density)
{
    return density.Positive != 0 && isfinite(density.Log2Value);
}
__device__ Density DensityFromValue(double value)
{
    return value > 0 && isfinite(value) ? Density{log2(value), 1, 0} : Density{};
}
__device__ Density AreaArrival(Density directional, float3 displacement, float3 normal)
{
    // Double geometry avoids squaring tiny/huge float distances in float.
    if (!HasDensity(directional))
        return {};
    const double length = norm3d(double(displacement.x), double(displacement.y), double(displacement.z));
    if (!(length > 0) || !isfinite(length))
        return {};
    const double cosine =
        fabs(double(normal.x) * (displacement.x / length) + double(normal.y) * (displacement.y / length) +
             double(normal.z) * (displacement.z / length));
    if (!(cosine > 0) || !isfinite(cosine))
        return {};
    const double value = directional.Log2Value + log2(cosine) - 2 * log2(length);
    return isfinite(value) ? Density{value, 1, 0} : Density{};
}
__device__ Density DiskProjection(Density disk, float3 normal, float3 direction)
{
    if (!HasDensity(disk))
        return {};
    const double cosine = fabs(double(normal.x) * direction.x + double(normal.y) * direction.y +
                               double(normal.z) * direction.z);
    if (!(cosine > 0) || !isfinite(cosine))
        return {};
    const double value = disk.Log2Value + log2(cosine);
    return isfinite(value) ? Density{value, 1, 0} : Density{};
}
__device__ Density DiskArea(double radius)
{
    if (!(radius > 0) || !isfinite(radius))
        return {};
    return {log2(VcmPi) + 2 * log2(radius), 1, 0};
}
__device__ Density SphereVolume(double radius)
{
    if (!(radius > 0) || !isfinite(radius))
        return {};
    return {log2(4 * VcmPi / 3) + 3 * log2(radius), 1, 0};
}
__device__ double VertexReference(const VertexPdf* vertices, unsigned count, unsigned index)
{
    if (index == 0 || index + 1 == count)
        return 0;
    const auto& vertex = vertices[index];
    const bool forward = HasDensity(vertex.Forward), reverse = HasDensity(vertex.Reverse);
    if (forward && reverse)
        return fmax(vertex.Forward.Log2Value, vertex.Reverse.Log2Value);
    return forward ? vertex.Forward.Log2Value : reverse ? vertex.Reverse.Log2Value : 0;
}
__device__ LogMass Accumulate(LogMass mass, Density factor, double reference = 0)
{
    if (HasDensity(factor))
        mass.Value += factor.Log2Value - reference;
    else
        ++mass.Zeros;
    return mass;
}
__device__ bool ValidConnection(const VertexPdf* vertices, unsigned count, unsigned split)
{
    if (split == 0)
        return !(vertices[0].Flags & EndpointNotIntersectable);
    if (split == count)
        return !(vertices[count - 1].Flags & EndpointNotIntersectable);
    const unsigned leftFlags = split > 1 ? vertices[split - 1].Flags : 0;
    const unsigned rightFlags = split + 1 < count ? vertices[split].Flags : 0;
    return !((leftFlags | rightFlags) & DeltaScattering);
}
__device__ bool StrategyLogScore(LogMass mass, const Strategy& strategy, double& score)
{
    if (!strategy.Enabled || !(strategy.LightSelection > 0 && strategy.LightSelection <= 1) ||
        !(strategy.SampleCount > 0) || !isfinite(strategy.SampleCount))
        return false;
    mass = Accumulate(mass, strategy.LightEndpoint);
    mass = Accumulate(mass, strategy.CameraEndpoint);
    if (mass.Zeros)
        return false;
    score = mass.Value + log2(strategy.LightSelection) + log2(strategy.SampleCount);
    return isfinite(score);
}
__device__ Weights EvaluateStrategies(const VertexPdf* vertices, unsigned count, const Strategy* connections,
                                      const Strategy* mergings, double radius, unsigned heuristic)
{
    Weights result{};
    if (!vertices || (!connections && !mergings) || count < 2 || count > MaximumVertices ||
        heuristic > PowerMis || !(radius >= 0) || !isfinite(radius))
        return result;

    LogMass prefix[MaximumStrategies];
    prefix[0] = prefix[1] = {};
    for (unsigned vertex = 1; vertex < count; ++vertex)
        prefix[vertex + 1] =
            Accumulate(prefix[vertex], vertices[vertex].Forward, VertexReference(vertices, count, vertex));
    LogMass camera{};
    unsigned long long activeConnection = 0, activeMerging = 0;
    double maximum = 0;
    const Density area = DiskArea(radius), volume = SphereVolume(radius);
    for (unsigned cursor = count + 1; cursor > 0; --cursor)
    {
        const unsigned split = cursor - 1;
        const LogMass base{prefix[split].Value + camera.Value, prefix[split].Zeros + camera.Zeros};
        double value;
        if (connections && ValidConnection(vertices, count, split) &&
            StrategyLogScore(base, connections[split], value))
        {
            maximum = (activeConnection || activeMerging) ? fmax(maximum, value) : value;
            activeConnection |= 1ull << split;
            result.Connection[split] = value;
            ++result.ActiveConnections;
        }
        if (mergings && split >= 2 && split < count && !(vertices[split - 1].Flags & DeltaScattering))
        {
            // VM eligibility is independent of VC at the same split index.
            const Density measure = vertices[split - 1].Flags & VolumeScattering ? volume : area;
            LogMass joined = Accumulate(Accumulate(base, vertices[split - 1].Reverse), measure);
            if (StrategyLogScore(joined, mergings[split], value))
            {
                maximum = (activeConnection || activeMerging) ? fmax(maximum, value) : value;
                activeMerging |= 1ull << split;
                result.Merging[split] = value;
                ++result.ActiveMergings;
            }
        }
        if (split > 0 && split < count)
            camera =
                Accumulate(camera, vertices[split - 1].Reverse, VertexReference(vertices, count, split - 1));
    }
    if (!(activeConnection || activeMerging))
        return result;
    const double exponent = heuristic == PowerMis ? 2 : 1;
    double denominator = 0;
    for (unsigned split = 0; split <= count; ++split)
    {
        const auto bit = 1ull << split;
        result.Connection[split] =
            (activeConnection & bit)
                ? (heuristic == EqualMis ? 1 : exp2(exponent * (result.Connection[split] - maximum)))
                : 0;
        result.Merging[split] =
            (activeMerging & bit)
                ? (heuristic == EqualMis ? 1 : exp2(exponent * (result.Merging[split] - maximum)))
                : 0;
        denominator += result.Connection[split] + result.Merging[split];
    }
    for (unsigned split = 0; split <= count; ++split)
    {
        result.Connection[split] /= denominator;
        result.Merging[split] /= denominator;
    }
    return result;
}
} // namespace Arvo::RayTracing::CudaVcm
