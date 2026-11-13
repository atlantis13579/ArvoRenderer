#pragma once
#include "RayTracer/Backends/CPU/SpectralScene.h"
#include "RayTracer/Backends/CPU/CpuTransport.h"

namespace Arvo::RayTracing::Cpu
{
struct PreparedSpectralFrame
{
    const PreparedSpectralScene* Scene = nullptr;
    MediumOptions Medium;
    RgbSpectrum Absorption{}, Scattering{};
};
RenderStatus PrepareSpectralFrame(const PreparedSpectralScene&, const MediumOptions&, PreparedSpectralFrame&,
                                  std::string&);

class SpectralPath
{
public:
    SpectralPath(const SampleContext&, bool importance);
    bool Enabled() const { return Frame != nullptr; }
    Material EvaluateMaterial(const Material&, uint32_t materialIndex) const;
    Float3 Surface(const Scene&, const SurfaceHit&, uint32_t textureIndex) const;
    Float3 Environment(const Scene&, const Ray&) const;
    Float3 DirectionalColor(const Scene&, const DirectionalLight&) const;
    Float3 PointColor(const Scene&, const PointLight&) const;
    Float3 RectColor(const Scene&, const RectAreaLight&) const;
    MediumOptions Medium() const;
    Float3 Film(Float3 radiance) const;

private:
    float Value(uint32_t entry, SpectralSceneData::Role) const;
    const PreparedSpectralFrame* Frame = nullptr;
    WavelengthPacket Wavelengths;
};
} // namespace Arvo::RayTracing::Cpu
