#include "RayTracer/Backends/CPU/BidirectionalMis.h"
#include <algorithm>
#include <cmath>

namespace Arvo::RayTracing::Cpu
{
// Veach (1997), "Robust Monte Carlo Methods for Light Transport Simulation",
// chapter 10; PBRT 3e, "Bidirectional Path Tracing", MISWeight / ConvertDensity:
// https://pbr-book.org/3ed-2018/Light_Transport_III_Bidirectional_Methods/Bidirectional_Path_Tracing
// A cut s has endpoint factors times Forward[1..s-1]*Reverse[s..n-2]. Cut 0
// also samples the light hit via Reverse[0]; cut n samples the camera hit via
// Forward[n-1]. Every internal vertex contributes exactly once. Subtracting the
// same reference log-density from both directions at each vertex leaves all MIS
// ratios unchanged, while removing enormous common factors before summation.
// Dirac comparison factors are one (PBRT's cancellation convention), but true
// zero continuous PDFs remain zero. Delta-adjacent deterministic cuts are absent.
namespace
{
bool Valid(PathDensity p)
{
    return p.Measure >= PathMeasure::Area && p.Measure <= PathMeasure::Volume && std::isfinite(p.Log2Value);
}
float ComparisonLog(PathDensity p)
{
    return p.Measure == PathMeasure::Dirac ? 0 : p.Log2Value;
}
struct Product
{
    float Log = 0;
    bool Valid = true;
};
Product Append(Product product, PathDensity factor, float reference)
{
    if (!product.Valid || !Valid(factor))
        return {0, false};
    float log = product.Log + (ComparisonLog(factor) - reference);
    return {log, std::isfinite(log)};
}
bool Allowed(const BidirectionalVertexPdf* vertices, uint32_t count, uint32_t cut)
{
    if (cut == 0)
        return !(vertices[0].Flags & PathEndpointNotIntersectable);
    if (cut == count)
        return !(vertices[count - 1].Flags & PathEndpointNotIntersectable);
    return !(cut > 1 && (vertices[cut - 1].Flags & PathDeltaScattering)) &&
           !(cut + 1 < count && (vertices[cut].Flags & PathDeltaScattering));
}
} // namespace
PathDensity MakePathDensity(float value, PathMeasure measure)
{
    if (!(value > 0) || !std::isfinite(value) || measure < PathMeasure::Area || measure > PathMeasure::Volume)
        return {};
    return {std::log2(value), measure};
}
PathDensity ConvertPathDensity(PathDensity p, Float3 displacement, Float3 geometricNormal,
                               PathDestination target)
{
    if (!Valid(p) || (p.Measure != PathMeasure::SolidAngle && p.Measure != PathMeasure::Dirac))
        return {};
    if (target == PathDestination::Infinite)
        return p;
    if (target != PathDestination::Surface && target != PathDestination::Volume)
        return {};
    float scale =
        std::max(std::abs(displacement.X), std::max(std::abs(displacement.Y), std::abs(displacement.Z)));
    if (!(scale > 0) || !std::isfinite(scale))
        return {};
    if (p.Measure == PathMeasure::Dirac)
        return p;
    Float3 scaled = displacement / scale;
    float length = std::sqrt(Dot(scaled, scaled));
    float cosine = target == PathDestination::Volume ? 1 :
        std::abs(Dot(geometricNormal, scaled / length));
    if (!(cosine > 0) || !std::isfinite(cosine)) return {};
    float logDistance = std::log2(scale) + std::log2(length);
    return {p.Log2Value + std::log2(cosine) - 2 * logDistance,
            target == PathDestination::Volume ? PathMeasure::Volume : PathMeasure::Area};
}
PathDensity ProjectedDiskDensity(PathDensity disk, Float3 normal, Float3 direction)
{
    if (!Valid(disk) || disk.Measure != PathMeasure::Area)
        return {};
    float cosine = std::abs(Dot(normal, direction));
    if (!(cosine > 0) || !std::isfinite(cosine))
        return {};
    return {disk.Log2Value + std::log2(cosine), PathMeasure::Area};
}
BidirectionalWeights BidirectionalMisWeights(const BidirectionalVertexPdf* vertices, uint32_t count,
                                             const BidirectionalStrategy* strategies, MisHeuristic heuristic)
{
    BidirectionalWeights result;
    if (count < 2 || count > MaximumBidirectionalVertices || !vertices || !strategies ||
        (heuristic != MisHeuristic::None && heuristic != MisHeuristic::Balance &&
         heuristic != MisHeuristic::Power))
        return result;
    std::array<float, MaximumBidirectionalVertices> reference{};
    for (uint32_t i = 1; i + 1 < count; ++i)
    {
        bool f = Valid(vertices[i].Forward), r = Valid(vertices[i].Reverse);
        reference[i] = f ? ComparisonLog(vertices[i].Forward) : (r ? ComparisonLog(vertices[i].Reverse) : 0);
        if (f && r)
            reference[i] = std::max(reference[i], ComparisonLog(vertices[i].Reverse));
    }
    std::array<Product, MaximumBidirectionalStrategies> prefix{}, suffix{};
    for (uint32_t s = 2; s <= count; ++s)
        prefix[s] = Append(prefix[s - 1], vertices[s - 1].Forward, reference[s - 1]);
    for (uint32_t s = count - 1; s > 0; --s)
        suffix[s - 1] = Append(suffix[s], vertices[s - 1].Reverse, reference[s - 1]);
    std::array<float, MaximumBidirectionalStrategies> score{};
    std::array<bool, MaximumBidirectionalStrategies> active{};
    float maximum = 0;
    for (uint32_t s = 0; s <= count; ++s)
    {
        const auto& strategy = strategies[s];
        if (!strategy.Enabled || !(strategy.SampleCount > 0 && strategy.LightSelection > 0) ||
            !std::isfinite(strategy.SampleCount) || !std::isfinite(strategy.LightSelection) ||
            !Allowed(vertices, count, s) || !prefix[s].Valid || !suffix[s].Valid ||
            !Valid(strategy.LightEndpoint) || !Valid(strategy.CameraEndpoint))
            continue;
        score[s] = prefix[s].Log + suffix[s].Log + ComparisonLog(strategy.LightEndpoint) +
                   ComparisonLog(strategy.CameraEndpoint) + std::log2(strategy.LightSelection) +
                   std::log2(strategy.SampleCount);
        if (!std::isfinite(score[s]))
            continue;
        active[s] = true;
        maximum = result.ActiveStrategies ? std::max(maximum, score[s]) : score[s];
        ++result.ActiveStrategies;
    }
    if (!result.ActiveStrategies)
        return result;
    float sum = 0;
    for (uint32_t s = 0; s <= count; ++s)
        if (active[s])
        {
            float exponent = heuristic == MisHeuristic::Power ? 2.0f : 1.0f;
            result.Values[s] =
                heuristic == MisHeuristic::None ? 1 : std::exp2(exponent * (score[s] - maximum));
            sum += result.Values[s];
        }
    for (uint32_t s = 0; s <= count; ++s)
        result.Values[s] /= sum;
    return result;
}
} // namespace Arvo::RayTracing::Cpu
