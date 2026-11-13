#include "RayTracer/Backends/CPU/SpectralTransport.h"
#include <algorithm>
#include <stdexcept>

namespace Arvo::RayTracing::Cpu
{
namespace
{
constexpr uint32_t WavelengthDimension = 0x57415645u; // "WAVE", independent of lens and walk dimensions.
Float3 Monochromatic(float value)
{
    return {value, value, value};
}
} // namespace

RenderStatus PrepareSpectralFrame(const PreparedSpectralScene& scene, const MediumOptions& medium,
                                  PreparedSpectralFrame& output, std::string& error)
{
    PreparedSpectralFrame candidate;
    candidate.Scene = &scene;
    candidate.Medium = medium;
    if (medium.Enabled)
    {
        RgbFitTable response;
        for (unsigned i = 0; i < response.size(); ++i)
            response[i] = MakeRgbFitResponse(i);
        candidate.Absorption = FitRgbIlluminant(medium.Absorption, response);
        candidate.Scattering = FitRgbIlluminant(medium.Scattering, response);
        if (candidate.Absorption.Status != SpectrumFitConverged ||
            candidate.Scattering.Status != SpectrumFitConverged)
        {
            error = "CPU spectral medium coefficient preparation did not converge.";
            return RenderStatus::Failed;
        }
    }
    output = candidate;
    error.clear();
    return RenderStatus::Completed;
}

// A single hero wavelength travels through the complete camera walk, including
// explicit light connections. PBRT v4, "A Better Path Tracer": wavelength PDFs
// belong in the film estimator, not in each BSDF factor. Replicated scalars let
// the existing RGB BSDF/free-flight code operate on one wavelength without
// multiplying RGB colors before their spectral reconstruction.
SpectralPath::SpectralPath(const SampleContext& samples, bool importance) : Frame(samples.SpectralFrame)
{
    if (Frame)
        Wavelengths = TerminateSecondaryWavelengths(
            SampleWavelengths(Sample2D(samples, WavelengthDimension).X, importance));
}

float SpectralPath::Value(uint32_t entry, SpectralSceneData::Role role) const
{
    return EvaluateSceneSpectrum(Frame->Scene->Spectra.at(entry), role, Wavelengths.Lambda[0]);
}

Material SpectralPath::EvaluateMaterial(const Material& original, uint32_t index) const
{
    if (!Frame)
        return original;
    const auto& input = Frame->Scene->Inputs.Materials.at(index);
    using Role = SpectralSceneData::Role;
    Material material = original;
    material.Albedo = Monochromatic(Value(input.Albedo, Role::Reflectance));
    material.AmbientColor = Monochromatic(Value(input.Ambient, Role::Reflectance));
    material.SpecularColor = Monochromatic(Value(input.Specular, Role::Reflectance));
    material.Emission = Monochromatic(Value(input.Emission, Role::Illuminant));
    material.ConductorIndex = Monochromatic(Value(input.ConductorIndex, Role::Nonnegative));
    material.ConductorExtinction = Monochromatic(Value(input.ConductorExtinction, Role::Nonnegative));
    return material;
}

Float3 SpectralPath::Surface(const Scene& original, const SurfaceHit& hit, uint32_t textureIndex) const
{
    if (!Frame)
        return hit.HasTexCoord
                   ? Multiply(hit.Color, SampleDiffuseTexture(original, textureIndex, hit.TexCoord))
                   : hit.Color;
    const auto& scene = *Frame->Scene;
    float color = 1;
    if (hit.SourceTriangle)
    {
        const auto source = scene.TriangleIndices.find(hit.SourceTriangle);
        if (source == scene.TriangleIndices.end())
            throw std::runtime_error("CPU spectral hit references a different scene snapshot.");
        const auto& triangle = scene.Inputs.Triangles.at(source->second);
        color = InterpolateSpectralVertex(
            scene.Spectra[triangle.Color[0]], scene.Spectra[triangle.Color[1]],
            scene.Spectra[triangle.Color[2]],
            {1 - hit.Barycentric.X - hit.Barycentric.Y, hit.Barycentric.X, hit.Barycentric.Y},
            Wavelengths.Lambda[0]);
    }
    if (hit.HasTexCoord)
        color *= SampleSpectralTexture(scene, textureIndex, hit.TexCoord, Wavelengths.Lambda[0]);
    return Monochromatic(color);
}

Float3 SpectralPath::Environment(const Scene& scene, const Ray& ray) const
{
    if (!Frame)
        return Background(scene, ray);
    const float blend = std::clamp(ray.Direction.Y * 0.5f + 0.5f, 0.0f, 1.0f);
    const auto& inputs = Frame->Scene->Inputs;
    return Monochromatic(Value(inputs.BackgroundBottom, SpectralSceneData::Role::Illuminant) * (1 - blend) +
                         Value(inputs.BackgroundTop, SpectralSceneData::Role::Illuminant) * blend);
}

Float3 SpectralPath::DirectionalColor(const Scene& scene, const DirectionalLight& light) const
{
    return Frame ? Monochromatic(Value(Frame->Scene->Inputs.DirectionalLights.at(
                                           size_t(&light - scene.DirectionalLights.data())),
                                       SpectralSceneData::Role::Illuminant))
                 : light.Color;
}
Float3 SpectralPath::PointColor(const Scene& scene, const PointLight& light) const
{
    return Frame ? Monochromatic(
                       Value(Frame->Scene->Inputs.PointLights.at(size_t(&light - scene.PointLights.data())),
                             SpectralSceneData::Role::Illuminant))
                 : light.Color;
}
Float3 SpectralPath::RectColor(const Scene& scene, const RectAreaLight& light) const
{
    return Frame ? Monochromatic(Value(
                       Frame->Scene->Inputs.RectAreaLights.at(size_t(&light - scene.RectAreaLights.data())),
                       SpectralSceneData::Role::Illuminant))
                 : light.Color;
}

MediumOptions SpectralPath::Medium() const
{
    MediumOptions medium = Frame->Medium;
    if (medium.Enabled)
    {
        // Extinction is an optical coefficient, not emitted radiance: no white SPD.
        medium.Absorption = Monochromatic(EvaluateRgbReflectance(Frame->Absorption, Wavelengths.Lambda[0]));
        medium.Scattering = Monochromatic(EvaluateRgbReflectance(Frame->Scattering, Wavelengths.Lambda[0]));
    }
    return medium;
}

Float3 SpectralPath::Film(Float3 radiance) const
{
    if (!Frame)
        return radiance;
    SampledSpectrum spectrum;
    spectrum.Value[0] = radiance.X;
    // Retain negative RGB components until image reconstruction/display. Clamping
    // individual wavelength samples biases saturated colors and film variance.
    return XYZToLinearSRGB(SpectrumToXYZ(Wavelengths, spectrum));
}
} // namespace Arvo::RayTracing::Cpu
