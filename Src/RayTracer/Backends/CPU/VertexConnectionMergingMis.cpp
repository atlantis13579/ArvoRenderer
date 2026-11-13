#include "RayTracer/Backends/CPU/VertexConnectionMergingMis.h"
#include <algorithm>
#include <cmath>

namespace Arvo::RayTracing::Cpu
{
// Georgiev et al. (2012), "Light Transport Simulation with Vertex Connection
// and Merging"; Georgiev (2013 revision), "Implementing Vertex Connection and
// Merging", equations 10-12 and sections 5.1/5.4:
// https://www.iliyan.com/publications/ImplementingVCM/ImplementingVCM_TechRep2012_rev2.pdf
//
// For n unique path vertices, a connection cut s contributes
//   endpoint * Forward[1..s-1] * Reverse[s..n-2].
// The merging strategy s samples x[s-1] from BOTH walks, so its density has
// one extra Reverse[s-1] factor and a surface disk or volume sphere measure.
// MIS compares (sampleCount * density)^beta across BOTH method families.
// The merging count is the attempted light population, not accepted photons.
//
// This explicit enumeration favors inspectable path measures over the paper's
// recursive optimization. Per-vertex references cancel common factors before
// summation. The EXTRA merged density must not subtract that reference again:
// doing so would incorrectly remove the relative scale of connection/merging.
// A specular event disables its two adjacent cuts and its own merge. Its delta
// component PMF remains in generalized densities; it is not replaced by one.
namespace
{
constexpr double Pi = 3.14159265358979323846;
bool Positive(VcmDensity density)
{
    return density.Positive && std::isfinite(density.Log2Value);
}
struct LogProduct
{
    double Value = 0;
    bool Valid = true;
};
LogProduct Append(LogProduct product, VcmDensity factor, double reference = 0)
{
    if (!product.Valid || !Positive(factor))
        return {0, false};
    double value = product.Value + (factor.Log2Value - reference);
    return {value, std::isfinite(value)};
}
bool ConnectionAllowed(const VcmVertexPdf* vertices, uint32_t count, uint32_t cut)
{
    if (!cut)
        return !(vertices[0].Flags & VcmEndpointNotIntersectable);
    if (cut == count)
        return !(vertices[count - 1].Flags & VcmEndpointNotIntersectable);
    bool lightDelta = cut > 1 && (vertices[cut - 1].Flags & VcmDeltaScattering);
    bool cameraDelta = cut + 1 < count && (vertices[cut].Flags & VcmDeltaScattering);
    return !lightDelta && !cameraDelta;
}
LogProduct StrategyScore(LogProduct path, const VcmStrategy& strategy)
{
    if (!strategy.Enabled || !(strategy.LightSelection > 0 && strategy.LightSelection <= 1) ||
        !(strategy.SampleCount > 0) || !std::isfinite(strategy.SampleCount))
        return {0, false};
    path = Append(Append(path, strategy.LightEndpoint), strategy.CameraEndpoint);
    return Append(Append(path, MakeVcmDensity(strategy.LightSelection)),
                  MakeVcmDensity(strategy.SampleCount));
}
double UnnormalizedWeight(LogProduct score, double maximum, MisHeuristic heuristic)
{
    if (!score.Valid)
        return 0;
    if (heuristic == MisHeuristic::None)
        return 1;
    double power = heuristic == MisHeuristic::Power ? 2 : 1;
    return std::exp2(power * (score.Value - maximum));
}
} // namespace
VcmDensity MakeVcmDensity(double value)
{
    return value > 0 && std::isfinite(value) ? VcmDensity{std::log2(value), 1} : VcmDensity{};
}
VcmDensity VcmSurfaceDensity(VcmDensity directional, Float3 displacement, Float3 geometricNormal)
{
    if (!Positive(directional))
        return {};
    double distance = std::hypot(double(displacement.X), double(displacement.Y), double(displacement.Z));
    if (!(distance > 0) || !std::isfinite(distance))
        return {};
    double cosine = std::abs(double(geometricNormal.X) * (displacement.X / distance) +
                             double(geometricNormal.Y) * (displacement.Y / distance) +
                             double(geometricNormal.Z) * (displacement.Z / distance));
    if (!(cosine > 0) || !std::isfinite(cosine))
        return {};
    double log = directional.Log2Value + std::log2(cosine) - 2 * std::log2(distance);
    return std::isfinite(log) ? VcmDensity{log, 1} : VcmDensity{};
}
VcmDensity VcmProjectedDiskDensity(VcmDensity disk, Float3 normal, Float3 direction)
{
    if (!Positive(disk))
        return {};
    double cosine = std::abs(double(normal.X) * direction.X + double(normal.Y) * direction.Y +
                             double(normal.Z) * direction.Z);
    if (!(cosine > 0) || !std::isfinite(cosine))
        return {};
    double log = disk.Log2Value + std::log2(cosine);
    return std::isfinite(log) ? VcmDensity{log, 1} : VcmDensity{};
}
VcmDensity VcmMergeArea(double radius)
{
    if (!(radius > 0) || !std::isfinite(radius))
        return {};
    return {std::log2(Pi) + 2 * std::log2(radius), 1};
}
VcmDensity VcmMergeVolume(double radius)
{
    if (!(radius > 0) || !std::isfinite(radius))
        return {};
    return {std::log2(4 * Pi / 3) + 3 * std::log2(radius), 1};
}
VcmWeights VertexConnectionMergingMisWeights(const VcmVertexPdf* vertices, uint32_t count,
                                             const VcmStrategy* connections, const VcmStrategy* mergings,
                                             double radius, MisHeuristic heuristic, double volumeRadius)
{
    VcmWeights result;
    if (!vertices || (!connections && !mergings) || count < 2 || count > MaximumVcmVertices ||
        !(radius >= 0) || !std::isfinite(radius) || !(volumeRadius >= 0) ||
        !std::isfinite(volumeRadius) ||
        (heuristic != MisHeuristic::None && heuristic != MisHeuristic::Balance &&
         heuristic != MisHeuristic::Power))
        return result;

    std::array<double, MaximumVcmVertices> reference{};
    for (uint32_t i = 1; i + 1 < count; ++i)
    {
        bool forward = Positive(vertices[i].Forward), reverse = Positive(vertices[i].Reverse);
        if (forward && reverse)
            reference[i] = std::max(vertices[i].Forward.Log2Value, vertices[i].Reverse.Log2Value);
        else if (forward || reverse)
            reference[i] = forward ? vertices[i].Forward.Log2Value : vertices[i].Reverse.Log2Value;
    }
    std::array<LogProduct, MaximumVcmStrategies> forward{}, reverse{};
    for (uint32_t s = 2; s <= count; ++s)
        forward[s] = Append(forward[s - 1], vertices[s - 1].Forward, reference[s - 1]);
    for (uint32_t s = count - 1; s > 0; --s)
        reverse[s - 1] = Append(reverse[s], vertices[s - 1].Reverse, reference[s - 1]);

    std::array<LogProduct, MaximumVcmStrategies> connectionScore{}, mergingScore{};
    const VcmDensity area = VcmMergeArea(radius);
    const VcmDensity volume = VcmMergeVolume(volumeRadius > 0 ? volumeRadius : radius);
    double maximum = 0;
    bool any = false;
    auto record = [&](LogProduct score, uint32_t& active) {
        if (score.Valid)
        {
            maximum = any ? std::max(maximum, score.Value) : score.Value;
            any = true;
            ++active;
        }
    };
    for (uint32_t s = 0; s <= count; ++s)
    {
        LogProduct path{forward[s].Value + reverse[s].Value, forward[s].Valid && reverse[s].Valid};
        connectionScore[s] = {0, false};
        mergingScore[s] = {0, false};
        if (connections && ConnectionAllowed(vertices, count, s))
            connectionScore[s] = StrategyScore(path, connections[s]);
        if (mergings && s >= 2 && s < count && !(vertices[s - 1].Flags & VcmDeltaScattering))
        {
            // Merging has its own endpoint factors/enable state. It does not
            // inherit the eligibility of the connection at this array slot.
            const auto measure = vertices[s - 1].Flags & VcmVolumeScattering ? volume : area;
            auto mergedPath = Append(Append(path, vertices[s - 1].Reverse), measure);
            mergingScore[s] = StrategyScore(mergedPath, mergings[s]);
        }
        record(connectionScore[s], result.ActiveConnections);
        record(mergingScore[s], result.ActiveMergings);
    }
    if (!any)
        return result;
    double sum = 0;
    for (uint32_t s = 0; s <= count; ++s)
    {
        result.Connection[s] = UnnormalizedWeight(connectionScore[s], maximum, heuristic);
        result.Merging[s] = UnnormalizedWeight(mergingScore[s], maximum, heuristic);
        sum += result.Connection[s] + result.Merging[s];
    }
    for (uint32_t s = 0; s <= count; ++s)
    {
        result.Connection[s] /= sum;
        result.Merging[s] /= sum;
    }
    return result;
}
} // namespace Arvo::RayTracing::Cpu
