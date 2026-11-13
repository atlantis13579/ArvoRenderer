#pragma once
#include "RayTracer/Backends/CPU/Bsdf.h"
#include "RayTracer/Backends/CPU/CpuTransport.h"
#include <array>

namespace Arvo::RayTracing::Cpu
{
struct SurfaceEvaluation
{
    Float3 Value;
    float Pdf = 0, ReversePdf = 0;
};
struct SurfaceSample
{
    Float3 Direction, Weight;
    float Pdf = 0, ReversePdf = 0, RelativeIndex = 1;
    uint32_t Event = ScatteringNone;
};

// World-space adapter for the backend's scalar BSDFs. Only diffuse reflectance
// uses vertex color / the diffuse texture; optical constants remain independent.
class SurfaceBsdf
{
  public:
    // Hit normals face view. Reverse queries flip the shading normal and medium
    // side on transmission; PDFs remain solid-angle densities or delta masses.
    SurfaceBsdf(const Material& material, const SurfaceHit& hit, Float3 view, Float3 surface,
                TransportMode mode = TransportMode::Radiance);
    bool IsDelta() const;
    SurfaceEvaluation Evaluate(Float3 light, bool importance) const;
    SurfaceSample Sample(Float2 variates, float lobeVariate, bool importance) const;
    std::array<SurfaceSample, 2> DeltaBranches() const;
    float Cosine(Float3 direction) const;
    Float3 Offset(Float3 position, Float3 direction) const;
    float ConnectionMeanFreePath(Float3 direction, float currentInterior) const;

  private:
    Float3 ToLocal(Float3 direction) const;
    Float3 ToWorld(Float3 direction) const;
    bool SupportedHemisphere(Float3 world, bool transmission) const;
    float AdjointCorrection(Float3 light) const;
    const Material& Material_;
    Float3 Normal_, GeometricNormal_, Tangent_, Bitangent_, View_, Reflectance_;
    float IncidentIndex_, TransmittedIndex_;
    float ViewGeometryCosine_;
    TransportMode Mode_;
    bool FrontFace_;
};

Float3 PhysicalDirectLighting(const Scene& scene, const SceneAccelerator& accelerator, Float3 position,
                              const SurfaceBsdf& bsdf, const RenderSettings& settings,
                              AtomicStatistics& statistics, const CancellationProbe& cancellation,
                              const SampleContext& samples, uint32_t bounce,
                              const SpectralPath* spectral = nullptr, float interiorMeanFreePath = 0);
} // namespace Arvo::RayTracing::Cpu
