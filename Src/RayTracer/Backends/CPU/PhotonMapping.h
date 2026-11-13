#pragma once
#include "RayTracer/Backends/CPU/CpuTransport.h"

namespace Arvo::RayTracing::Cpu
{
inline constexpr uint32_t PhotonMaximumReflections = 32;
struct PhotonMappingDiagnostics
{
    uint64_t EmittedPaths = 0, DirectPhotons = 0, CausticPhotons = 0, IndirectPhotons = 0;
    double Radius = 0;
};
using PhotonDispatcher = std::function<void(uint32_t, const std::function<void(uint32_t)>&)>;
// One global seeded photon population is reused for every camera sample.
// Crops regenerate that population, then evaluate only their own camera pixels.
bool RenderPhotonFrame(const Scene&, const SceneAccelerator&, const CameraFrame&, const RenderSettings&,
                       LinearFilm&, AtomicStatistics&, const CancellationProbe&, const PhotonDispatcher&,
                       std::string& error, PhotonMappingDiagnostics* diagnostics = nullptr);
} // namespace Arvo::RayTracing::Cpu
