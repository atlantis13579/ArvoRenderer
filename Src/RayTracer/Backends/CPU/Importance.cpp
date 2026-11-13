#include "RayTracer/Backends/CPU/CpuTransport.h"
#include <algorithm>
#include <cmath>

namespace Arvo::RayTracing::Cpu
{
// Inverse-transform hemisphere sampling. The uniform alternative exposes the
// f*cos/pdf correction for comparison with cosine-weighted importance sampling.
Float3 SampleDiffuseDirection(const Float3& normal, float u, float v, bool importance)
{
    const float radius = importance ? std::sqrt(u) : std::sqrt(std::max(0.0f, 1.0f - (1.0f - u) * (1.0f - u)));
    const float cosine = importance ? std::sqrt(std::max(0.0f, 1.0f - radius * radius)) : 1.0f - u;
    const float phi = 2.0f * Pi * v;
    const Float3 helper = std::fabs(normal.Y) < BasisParallelThreshold
        ? Float3(0, 1, 0) : Float3(1, 0, 0);
    const Float3 tangent = Normalize(Cross(helper, normal));
    const Float3 bitangent = Cross(normal, tangent);
    return Normalize(tangent * (radius * std::cos(phi)) + bitangent * (radius * std::sin(phi)) + normal * cosine);
}

float DiffusePdf(float cosine, bool importance)
{
    if (cosine <= 0.0f) return 0.0f;
    return importance ? cosine / Pi : 1.0f / (2.0f * Pi);
}
}
