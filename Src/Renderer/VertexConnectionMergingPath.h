#pragma once
#include "Renderer/RayTracing.h"

namespace Arvo::RayTracing
{
// Data only. The backend owns density conversion and VCM weight evaluation.
inline constexpr uint32_t MaximumVcmReflections = 32;
inline constexpr uint32_t MaximumVcmScattering = MaximumVcmReflections + 1;
inline constexpr uint32_t MaximumVcmVertices = MaximumVcmScattering + 2;
inline constexpr uint32_t MaximumVcmStrategies = MaximumVcmVertices + 1;
inline constexpr uint32_t VcmDeltaScattering = 1, VcmEndpointNotIntersectable = 2,
                          VcmVolumeScattering = 4;

// A generalized path density: directional delta events retain their component
// probability and the geometric conversion factor. This is NOT an ordinary
// area PDF at a specular vertex. Eligibility is recorded separately below.
// Positive distinguishes zero probability from log2(1), without infinities.
struct VcmDensity
{
    double Log2Value = 0;
    uint32_t Positive = 0;
};
struct VcmVertexPdf
{
    // Ordered from emitter x[0] to camera x[n-1]. Forward[i] samples x[i]
    // from x[i-1]; Reverse[i] samples it from x[i+1]. The caller evaluates
    // the completed path, including changed directions near its connection
    // or merge. Infinite endpoints use angular instead of area measure.
    VcmDensity Forward, Reverse;
    // Delta describes this completed scattering event, not an unused sampled
    // continuation beyond a connection/merge endpoint of a mixed BSDF.
    uint32_t Flags = 0;
};
struct VcmStrategy
{
    VcmDensity LightEndpoint, CameraEndpoint;
    double LightSelection = 1, SampleCount = 1;
    uint32_t Enabled = 0;
};
struct VcmWeights
{
    // Connection slot s: s light vertices, n-s camera vertices.
    // Merging slot s: s light vertices, n+1-s camera vertices; x[s-1]
    // occurs in BOTH walks. Only slots 2..n-1 may merge finite vertices.
    std::array<double, MaximumVcmStrategies> Connection{}, Merging{};
    uint32_t ActiveConnections = 0, ActiveMergings = 0;
};
} // namespace Arvo::RayTracing
