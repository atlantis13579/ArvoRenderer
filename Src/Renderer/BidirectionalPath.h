#pragma once
#include "Renderer/RayTracing.h"

namespace Arvo::RayTracing
{
// Data only. Native implementations independently evaluate path-space densities
// and MIS. Log representation preserves positive densities below float range.
enum class PathMeasure : uint32_t
{
    Zero,
    Area,
    SolidAngle,
    Dirac,
    Volume
};
enum class PathDestination : uint32_t
{
    Surface,
    Infinite,
    Volume
};
struct PathDensity
{
    float Log2Value = 0;
    PathMeasure Measure = PathMeasure::Zero;
};
inline constexpr uint32_t MaximumBidirectionalReflections = 32;
inline constexpr uint32_t MaximumBidirectionalScattering = MaximumBidirectionalReflections + 1;
inline constexpr uint32_t MaximumBidirectionalVertices = MaximumBidirectionalScattering + 2;
inline constexpr uint32_t MaximumBidirectionalStrategies = MaximumBidirectionalVertices + 1;
inline constexpr uint32_t PathDeltaScattering = 1, PathEndpointNotIntersectable = 2;
// Ordered from light x[0] to camera x[n-1]. Forward[i] is the density of
// sampling x[i] from x[i-1]; Reverse[i] samples x[i] from x[i+1]. Evaluate both
// on the completed path, including the connection's changed adjacent BSDF PDFs.
// DeltaScattering describes this path's event, not a discarded continuation
// sample beyond a connection endpoint. Source selection is never this flag.
struct alignas(16) BidirectionalVertexPdf
{
    PathDensity Forward, Reverse;
    uint32_t Flags = 0, Reserved0 = 0, Reserved1 = 0, Reserved2 = 0;
};
// Slot s uses s light vertices and n-s camera vertices. Endpoint factors may
// differ for NEE / camera connections. Unused endpoints have factor one; absent
// strategies have Enabled=0. LightSelection is a discrete PMF, NOT a Dirac PDF.
struct alignas(16) BidirectionalStrategy
{
    PathDensity LightEndpoint, CameraEndpoint;
    float LightSelection = 1, SampleCount = 1;
    uint32_t Enabled = 1, Reserved = 0;
};
struct BidirectionalWeights
{
    std::array<float, MaximumBidirectionalStrategies> Values{};
    uint32_t ActiveStrategies = 0;
};
static_assert(sizeof(BidirectionalVertexPdf) == 2 * 4 * sizeof(float));
static_assert(sizeof(BidirectionalStrategy) == 2 * 4 * sizeof(float));
} // namespace Arvo::RayTracing
