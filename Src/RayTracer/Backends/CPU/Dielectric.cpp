#include "RayTracer/Backends/CPU/Bsdf.h"
#include "RayTracer/Backends/CPU/Microfacet.h"
#include <cmath>

namespace Arvo::RayTracing::Cpu
{
// Walter et al. (2007), "Microfacet Models for Refraction through Rough Surfaces".
// https://pbr-book.org/4ed/Reflection_Models/Dielectric_BSDF
// Reflection and transmission share a visible-normal proposal and Fresnel split.
// Radiance transmission includes (eta_i/eta_t)^2; importance transport does not.
// Rejected macro-hemisphere events remain null samples, not resampled events.
namespace
{
BsdfEvaluation Transmission(Float3 view, Float3 light, Float3 normal, Float2 alpha, float incidentIndex,
                            float transmittedIndex, TransportMode mode, float denominatorSquared,
                            bool sampledFacet)
{
    const float viewDot = Dot(view, normal), lightDot = Dot(light, normal);
    if (viewDot <= 0 || lightDot >= 0 || denominatorSquared <= 0)
        return {};
    const float ratio = incidentIndex / transmittedIndex;
    float transmission = 1 - DielectricFresnel(viewDot, incidentIndex, transmittedIndex);
    float reverseTransmission = 1 - DielectricFresnel(-lightDot, transmittedIndex, incidentIndex);
    // For a known direction pair, use the lower-index side of Fresnel reciprocity:
    // reconstructing a near-critical transmitted cosine from the other side is ill-conditioned.
    // A sampled facet must instead retain the probabilities used to choose its branch.
    if (!sampledFacet)
        transmission = reverseTransmission =
            incidentIndex > transmittedIndex ? reverseTransmission : transmission;
    const float jacobian = -lightDot / denominatorSquared;
    float value = transmission * GgxDistribution(normal, alpha) * GgxMaskingShadowing(view, light, alpha) *
                  (viewDot * jacobian / (view.Z * -light.Z));
    if (mode == TransportMode::Radiance)
        value *= ratio * ratio;
    return {value, transmission * GgxVisibleNormalPdf(view, normal, alpha) * jacobian,
            reverseTransmission * GgxVisibleNormalPdf(-light, normal, alpha) * viewDot * ratio * ratio /
                denominatorSquared};
}
} // namespace
BsdfEvaluation EvaluateDielectric(Float3 view, Float3 light, Float2 alpha, float incidentIndex,
                                  float transmittedIndex, TransportMode mode)
{
    if (view.Z <= 0 || light.Z == 0 || alpha.X <= 0 || alpha.Y <= 0 || incidentIndex == transmittedIndex)
        return {};
    if (light.Z > 0)
    {
        const Float3 normal = Normalize(view + light);
        const float fresnel = DielectricFresnel(Dot(view, normal), incidentIndex, transmittedIndex);
        return {fresnel * GgxReflectionFactor(view, light, alpha),
                fresnel * GgxReflectionPdf(view, light, alpha),
                fresnel * GgxReflectionPdf(light, view, alpha)};
    }
    const Float3 half = view * (incidentIndex / transmittedIndex) + light;
    const float squaredLength = Dot(half, half);
    if (squaredLength == 0)
        return {};
    Float3 normal = half / std::sqrt(squaredLength);
    if (normal.Z < 0)
        normal = -normal;
    // |rho*wo+wi|^2 is the transmission denominator, avoiding a second subtraction
    // of nearly equal dot products when reconstructing the half vector.
    return Transmission(view, light, normal, alpha, incidentIndex, transmittedIndex, mode, squaredLength,
                        false);
}
BsdfSample SampleDielectric(Float3 view, Float2 alpha, float incidentIndex, float transmittedIndex,
                            TransportMode mode, Float2 variates, float lobeVariate)
{
    if (view.Z <= 0)
        return {};
    const bool matched = incidentIndex == transmittedIndex;
    const bool delta = matched || (alpha.X == 0 && alpha.Y == 0);
    if (!delta && (alpha.X <= 0 || alpha.Y <= 0))
        return {};
    const Float3 normal = delta ? Float3(0, 0, 1) : SampleGgxVisibleNormal(view, alpha, variates);
    const float viewDot = Dot(view, normal);
    const float fresnel = DielectricFresnel(viewDot, incidentIndex, transmittedIndex);
    if (lobeVariate < fresnel)
    {
        const Float3 light = normal * (2 * viewDot) - view;
        if (light.Z <= 0)
            return {};
        if (delta)
            return {light, fresnel / light.Z, fresnel, fresnel, 1, ScatteringReflection | ScatteringDelta};
        const auto evaluation = EvaluateDielectric(view, light, alpha, incidentIndex, transmittedIndex, mode);
        return {light,
                evaluation.Value,
                evaluation.Pdf,
                evaluation.ReversePdf,
                1,
                ScatteringReflection | ScatteringGlossy};
    }
    const float ratio = incidentIndex / transmittedIndex;
    Float3 light;
    if (!RefractDirection(view, normal, ratio, light) || light.Z >= 0)
        return {};
    if (delta)
    {
        const float mass = 1 - fresnel;
        const float reverseMass = 1 - DielectricFresnel(-light.Z, transmittedIndex, incidentIndex);
        float value = mass / -light.Z;
        if (mode == TransportMode::Radiance)
            value *= ratio * ratio;
        return {light,
                value,
                mass,
                reverseMass,
                transmittedIndex / incidentIndex,
                ScatteringTransmission | ScatteringDelta};
    }
    // Rationalize rho*cos(theta_i)-cos(theta_t) before squaring. The ordinary
    // subtraction loses precision near index matching; no roughness/IOR cutoff.
    const float transmittedCosine = std::sqrt((1 - ratio) * (1 + ratio) + ratio * ratio * viewDot * viewDot);
    const float denominator = (ratio - 1) * (ratio + 1) / (ratio * viewDot + transmittedCosine);
    const auto evaluation = Transmission(view, light, normal, alpha, incidentIndex, transmittedIndex, mode,
                                         denominator * denominator, true);
    return {light,
            evaluation.Value,
            evaluation.Pdf,
            evaluation.ReversePdf,
            transmittedIndex / incidentIndex,
            ScatteringTransmission | ScatteringGlossy};
}
} // namespace Arvo::RayTracing::Cpu
