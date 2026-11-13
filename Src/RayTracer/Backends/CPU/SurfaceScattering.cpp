#include "RayTracer/Backends/CPU/SurfaceScattering.h"
#include <algorithm>
#include <cmath>

namespace Arvo::RayTracing::Cpu
{
// PBRT v4, "BSDF Representation" and "Surface Interactions".
// https://pbr-book.org/4ed/Reflection_Models/BSDF_Representation
// Shading and geometry have different roles. Geometry selects the medium and rejects paths
// that cross the wrong side; the shading frame defines the BSDF and its PDFs.
// Uniform proposals deliberately retain zero-contribution samples. Rejection
// sampling would require a different normalization in both continuation and MIS.
SurfaceBsdf::SurfaceBsdf(const Material& material, const SurfaceHit& hit, Float3 view, Float3 surface,
                         TransportMode mode)
    : Material_(material), Normal_(hit.Normal), GeometricNormal_(hit.GeometricNormal),
      IncidentIndex_(hit.FrontFace ? material.ExteriorIndex : material.InteriorIndex),
      TransmittedIndex_(hit.FrontFace ? material.InteriorIndex : material.ExteriorIndex),
      ViewGeometryCosine_(std::abs(Dot(Normalize(view), hit.GeometricNormal))), Mode_(mode),
      FrontFace_(hit.FrontFace)
{
    const Float3 helper = std::abs(Normal_.Y) < BasisParallelThreshold ? Float3(0, 1, 0) : Float3(1, 0, 0);
    Tangent_ = Normalize(Cross(helper, Normal_));
    Bitangent_ = Cross(Normal_, Tangent_);
    View_ = ToLocal(Normalize(view));
    const Float3 color = Multiply(material.Albedo, surface);
    Reflectance_ = {std::clamp(color.X, 0.0f, 1.0f), std::clamp(color.Y, 0.0f, 1.0f),
                    std::clamp(color.Z, 0.0f, 1.0f)};
}
Float3 SurfaceBsdf::ToLocal(Float3 v) const
{
    return {Dot(v, Tangent_), Dot(v, Bitangent_), Dot(v, Normal_)};
}
Float3 SurfaceBsdf::ToWorld(Float3 v) const
{
    return Normalize(Tangent_ * v.X + Bitangent_ * v.Y + Normal_ * v.Z);
}
bool SurfaceBsdf::SupportedHemisphere(Float3 world, bool transmission) const
{
    const float cosine = Dot(world, GeometricNormal_);
    if (cosine == 0)
        return false;
    return ((cosine > 0) == FrontFace_) != transmission;
}
bool SurfaceBsdf::IsDelta() const
{
    return Material_.Model != MaterialModel::Diffuse &&
           (Material_.MicrofacetSlope == 0 ||
            (Material_.Model == MaterialModel::Dielectric && IncidentIndex_ == TransmittedIndex_));
}
float SurfaceBsdf::Cosine(Float3 direction) const
{
    return std::abs(Dot(Normal_, direction));
}
Float3 SurfaceBsdf::Offset(Float3 position, Float3 direction) const
{
    const float side = Dot(GeometricNormal_, direction) >= 0 ? 1.0f : -1.0f;
    return position + GeometricNormal_ * (side * RayEpsilon);
}
float SurfaceBsdf::ConnectionMeanFreePath(Float3 direction, float currentInterior) const
{
    if (Material_.SubsurfaceMeanFreePath == 0) return currentInterior;
    return Dot(GeometricNormal_, direction) < 0 ? Material_.SubsurfaceMeanFreePath : 0;
}
float SurfaceBsdf::AdjointCorrection(Float3 light) const
{
    // Veach (1997), chapter 5: adjoint shading-normal correction. The local
    // dielectric already handles radiance eta^-2; this factor changes f, not
    // the sampling density. Do not cap it or renormalize clipped directions.
    // https://pbr-book.org/3ed-2018/Light_Transport_III_Bidirectional_Methods/Bidirectional_Path_Tracing
    if (Mode_ == TransportMode::Radiance)
        return 1;
    const float denominator = ViewGeometryCosine_ * Cosine(light);
    return denominator > 0 ? std::abs(View_.Z * Dot(light, GeometricNormal_)) / denominator : 0;
}
SurfaceEvaluation SurfaceBsdf::Evaluate(Float3 light, bool importance) const
{
    const Float3 local = ToLocal(light);
    const bool transmission = local.Z < 0;
    if (View_.Z <= 0 || !SupportedHemisphere(light, transmission))
        return {};
    const Float2 alpha{Material_.MicrofacetSlope, Material_.MicrofacetSlope};
    BsdfEvaluation red, green, blue;
    switch (Material_.Model)
    {
    case MaterialModel::Diffuse:
        red = EvaluateDiffuse(View_, local, Reflectance_.X, importance);
        green = EvaluateDiffuse(View_, local, Reflectance_.Y, importance);
        blue = EvaluateDiffuse(View_, local, Reflectance_.Z, importance);
        break;
    case MaterialModel::Conductor:
        red = EvaluateConductor(View_, local, alpha, Material_.ConductorIndex.X,
                                Material_.ConductorExtinction.X);
        green = EvaluateConductor(View_, local, alpha, Material_.ConductorIndex.Y,
                                  Material_.ConductorExtinction.Y);
        blue = EvaluateConductor(View_, local, alpha, Material_.ConductorIndex.Z,
                                 Material_.ConductorExtinction.Z);
        if (!importance && !IsDelta() && local.Z > 0)
            red.Pdf = red.ReversePdf = 1 / (2 * Pi);
        break;
    case MaterialModel::Dielectric:
        red = EvaluateDielectric(View_, local, alpha, IncidentIndex_, TransmittedIndex_, Mode_);
        green = blue = red;
        if (!importance && !IsDelta() && local.Z != 0)
            red.Pdf = red.ReversePdf = 1 / (4 * Pi);
        break;
    default:
        return {};
    }
    return {Float3(red.Value, green.Value, blue.Value) * AdjointCorrection(light), red.Pdf, red.ReversePdf};
}
SurfaceSample SurfaceBsdf::Sample(Float2 variates, float lobeVariate, bool importance) const
{
    if (View_.Z <= 0)
        return {};
    if (!importance && !IsDelta() && Material_.Model != MaterialModel::Diffuse)
    {
        const bool dielectric = Material_.Model == MaterialModel::Dielectric;
        const float cosine = dielectric ? 1 - 2 * variates.X : variates.X;
        // Reject the exact local horizon before frame roundoff can give it a
        // tiny cosine whose inverse is amplified by the shading-normal adjoint.
        if (cosine == 0)
            return {};
        const float sine = std::sqrt(std::max(0.0f, 1 - cosine * cosine));
        const float azimuth = 2 * Pi * variates.Y;
        const Float3 direction = ToWorld({sine * std::cos(azimuth), sine * std::sin(azimuth), cosine});
        const SurfaceEvaluation evaluation = Evaluate(direction, false);
        if (evaluation.Pdf <= 0)
            return {};
        return {direction,
                evaluation.Value * (Cosine(direction) / evaluation.Pdf),
                evaluation.Pdf,
                evaluation.ReversePdf,
                cosine < 0 ? TransmittedIndex_ / IncidentIndex_ : 1,
                ScatteringGlossy | (cosine < 0 ? ScatteringTransmission : ScatteringReflection)};
    }
    const Float2 alpha{Material_.MicrofacetSlope, Material_.MicrofacetSlope};
    BsdfSample red, green, blue;
    switch (Material_.Model)
    {
    case MaterialModel::Diffuse:
        red = SampleDiffuse(View_, Reflectance_.X, variates, importance);
        green = SampleDiffuse(View_, Reflectance_.Y, variates, importance);
        blue = SampleDiffuse(View_, Reflectance_.Z, variates, importance);
        break;
    case MaterialModel::Conductor:
        red = SampleConductor(View_, alpha, Material_.ConductorIndex.X, Material_.ConductorExtinction.X,
                              variates);
        green = SampleConductor(View_, alpha, Material_.ConductorIndex.Y, Material_.ConductorExtinction.Y,
                                variates);
        blue = SampleConductor(View_, alpha, Material_.ConductorIndex.Z, Material_.ConductorExtinction.Z,
                               variates);
        break;
    case MaterialModel::Dielectric:
        red = SampleDielectric(View_, alpha, IncidentIndex_, TransmittedIndex_, Mode_, variates, lobeVariate);
        green = blue = red;
        break;
    default:
        return {};
    }
    if (red.Event == ScatteringNone || red.Pdf <= 0)
        return {};
    const Float3 world = ToWorld(red.Direction);
    if (!SupportedHemisphere(world, (red.Event & ScatteringTransmission) != 0))
        return {};
    return {world,
            Float3(red.Value, green.Value, blue.Value) *
                (std::abs(red.Direction.Z) / red.Pdf * AdjointCorrection(world)),
            red.Pdf,
            red.ReversePdf,
            red.RelativeIndex,
            red.Event};
}
std::array<SurfaceSample, 2> SurfaceBsdf::DeltaBranches() const
{
    if (!IsDelta())
        return {};
    // Enumerate both ends of the discrete lobe CDF, without random selection.
    // Multiplying each conditional Weight by its Pdf recovers f*abs(cos).
    // Matching indices and TIR have only one event, which must not be duplicated.
    const float lastVariate = std::nextafter(1.0f, 0.0f);
    std::array<SurfaceSample, 2> branches{Sample({}, 0, true), Sample({}, lastVariate, true)};
    if (branches[0].Event == branches[1].Event)
        branches[1] = {};
    return branches;
}
} // namespace Arvo::RayTracing::Cpu
