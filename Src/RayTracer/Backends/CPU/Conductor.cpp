#include "RayTracer/Backends/CPU/Bsdf.h"
#include "RayTracer/Backends/CPU/Microfacet.h"

namespace Arvo::RayTracing::Cpu
{
// Single-scattering GGX conductor, or an exact mirror when both slopes are zero.
// eta and absorption are relative to the incident dielectric's index.
// PBRT v4, "Conductor BRDF": https://pbr-book.org/4ed/Reflection_Models/Conductor_BRDF
BsdfEvaluation EvaluateConductor(Float3 view, Float3 light, Float2 alpha, float eta, float absorption)
{
    if (view.Z <= 0 || light.Z <= 0 || alpha.X <= 0 || alpha.Y <= 0)
        return {};
    const Float3 normal = Normalize(view + light);
    const float fresnel = ConductorFresnel(Dot(view, normal), eta, absorption);
    return {fresnel * GgxReflectionFactor(view, light, alpha), GgxReflectionPdf(view, light, alpha),
            GgxReflectionPdf(light, view, alpha)};
}
BsdfSample SampleConductor(Float3 view, Float2 alpha, float eta, float absorption, Float2 variates)
{
    if (view.Z <= 0)
        return {};
    if (alpha.X == 0 && alpha.Y == 0)
        return {{-view.X, -view.Y, view.Z},
                ConductorFresnel(view.Z, eta, absorption) / view.Z,
                1,
                1,
                1,
                ScatteringReflection | ScatteringDelta};
    if (alpha.X <= 0 || alpha.Y <= 0)
        return {};
    const Float3 normal = SampleGgxVisibleNormal(view, alpha, variates);
    const Float3 light = normal * (2 * Dot(view, normal)) - view;
    if (light.Z <= 0)
        return {}; // The null mass is part of the sampler; never retry.
    const auto evaluation = EvaluateConductor(view, light, alpha, eta, absorption);
    return {light,
            evaluation.Value,
            evaluation.Pdf,
            evaluation.ReversePdf,
            1,
            ScatteringReflection | ScatteringGlossy};
}
} // namespace Arvo::RayTracing::Cpu
