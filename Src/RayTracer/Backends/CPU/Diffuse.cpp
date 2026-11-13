#include "RayTracer/Backends/CPU/Bsdf.h"
#include <algorithm>
#include <cmath>

namespace Arvo::RayTracing::Cpu
{
// Lambert's f=rho/pi with either cosine-weighted or uniform hemisphere sampling.
// The alternative sampler changes PDFs, never the BRDF. Variates lie in [0,1).
namespace
{
constexpr float DiffusePi = 3.14159265358979323846f;
}
BsdfEvaluation EvaluateDiffuse(Float3 view, Float3 light, float reflectance, bool importance)
{
    if (view.Z <= 0 || light.Z <= 0)
        return {};
    return {reflectance / DiffusePi, importance ? light.Z / DiffusePi : 1 / (2 * DiffusePi),
            importance ? view.Z / DiffusePi : 1 / (2 * DiffusePi)};
}
BsdfSample SampleDiffuse(Float3 view, float reflectance, Float2 variates, bool importance)
{
    if (view.Z <= 0)
        return {};
    const float cosine = importance ? std::sqrt(1 - variates.X) : 1 - variates.X;
    const float radius = std::sqrt(std::max(0.0f, 1 - cosine * cosine));
    const float azimuth = 2 * DiffusePi * variates.Y;
    const Float3 light(radius * std::cos(azimuth), radius * std::sin(azimuth), cosine);
    const auto evaluation = EvaluateDiffuse(view, light, reflectance, importance);
    return {light,
            evaluation.Value,
            evaluation.Pdf,
            evaluation.ReversePdf,
            1,
            ScatteringReflection | ScatteringDiffuse};
}
} // namespace Arvo::RayTracing::Cpu
