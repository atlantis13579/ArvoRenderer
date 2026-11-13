#pragma once
#include "RayTracer/Backends/CPU/Spectral.h"
#include <unordered_map>

namespace Arvo::RayTracing::Detail { class PersistentExecutor; }
namespace Arvo::RayTracing::Cpu
{
struct PreparedSpectralScene
{
    SpectralSceneData::Scene Inputs;
    std::vector<RgbSpectrum> Spectra;
    std::unordered_map<const Triangle*, uint32_t> TriangleIndices;
};
RenderStatus PrepareSpectralScene(const Scene&, const SpectralSceneData::Cancellation&,
                                  PreparedSpectralScene&, std::string&, Detail::PersistentExecutor* = nullptr);
float EvaluateSceneSpectrum(RgbSpectrum, SpectralSceneData::Role, float wavelengthNm);
float SampleSpectralTexture(const PreparedSpectralScene&, uint32_t textureIndex, Float2 uv,
                            float wavelengthNm);
float InterpolateSpectralVertex(RgbSpectrum a, RgbSpectrum b, RgbSpectrum c, Float3 weights,
                                float wavelengthNm);
} // namespace Arvo::RayTracing::Cpu
