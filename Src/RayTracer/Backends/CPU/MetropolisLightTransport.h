#pragma once
#include "RayTracer/Backends/CPU/CpuTransport.h"
#include <array>

namespace Arvo::RayTracing::Cpu
{
inline constexpr uint32_t MetropolisMaximumReflections = 32;
inline constexpr uint32_t MetropolisCameraDimensions = 4;
inline constexpr uint32_t MetropolisVertexDimensions = 12;
inline constexpr uint32_t MetropolisDimensions =
    MetropolisCameraDimensions + (MetropolisMaximumReflections + 2) * MetropolisVertexDimensions;
struct MetropolisPrimaryState
{
    std::array<double, MetropolisDimensions> Values{};
};
struct MetropolisRandom
{
    uint64_t State;
    double Uniform();
};
MetropolisPrimaryState ProposeMetropolis(const MetropolisPrimaryState&, MetropolisRandom&, bool largeStep,
                                         double mutationScale);
struct MetropolisPath
{
    Float2 Raster;
    Float3 Radiance;
    double Importance = 0;
};
struct MetropolisDiagnostics
{
    double Normalization = 0;
    uint64_t Proposals = 0, Accepted = 0, LargeSteps = 0, RejectedPositive = 0;
};
using MetropolisDispatcher = std::function<void(uint32_t, const std::function<void(uint32_t)>&)>;
// Each pass advances all chains by a total of FullWidth*FullHeight mutations.
// Crop jobs replay the SAME full-image chains and retain only local splats.
bool RenderMetropolisFrame(const Scene&, const SceneAccelerator&, const CameraFrame&, const RenderSettings&,
                           LinearFilm&, AtomicStatistics&, const CancellationProbe&,
                           const MetropolisDispatcher&, std::string& error,
                           MetropolisDiagnostics* diagnostics = nullptr);
} // namespace Arvo::RayTracing::Cpu
