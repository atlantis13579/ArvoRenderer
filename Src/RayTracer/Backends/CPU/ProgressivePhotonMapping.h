#pragma once
#include "RayTracer/Backends/CPU/CpuTransport.h"
#include <array>

namespace Arvo::RayTracing::Cpu
{
inline constexpr uint32_t ProgressivePhotonMaximumReflections = 32;
struct ProgressivePhotonPixel
{
    double RadiusSquared = 0, EffectiveCount = 0;
    std::array<double, 3> Flux{};
};
struct ProgressivePhotonDiagnostics
{
    uint64_t EmittedPaths = 0, AcceptedArrivals = 0;
    uint32_t CompletedPasses = 0;
    std::vector<ProgressivePhotonPixel> Pixels;
    std::vector<ProgressivePhotonPixel> VolumePixels;
};
// The independent estimator owns this recurrence; it is also exposed for
// analytic checks of radius/flux rescaling without a stochastic image test.
void AdvanceProgressivePhotonPixel(ProgressivePhotonPixel&, uint64_t arrivals,
                                   const std::array<double, 3>& weightedFlux, double alpha);
using ProgressivePhotonDispatcher = std::function<void(uint32_t, const std::function<void(uint32_t)>&)>;
bool RenderProgressivePhotonFrame(const Scene&, const SceneAccelerator&, const CameraFrame&,
                                  const RenderSettings&, LinearFilm&, AtomicStatistics&,
                                  const CancellationProbe&, const ProgressivePhotonDispatcher&,
                                  std::string& error, ProgressivePhotonDiagnostics* = nullptr);
} // namespace Arvo::RayTracing::Cpu
