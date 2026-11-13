#pragma once
#include "RayTracer/Backends/CPU/CpuTransport.h"

namespace Arvo::RayTracing::Cpu
{
Float3 VolumeDirectLighting(const Scene&, const SceneAccelerator&, Float3 position,
                            Float3 incidentPropagation, const RenderSettings&, AtomicStatistics&,
                            const CancellationProbe&, const SampleContext&, uint32_t bounce,
                            const SpectralPath* spectral = nullptr, const MediumOptions* activeMedium = nullptr);
} // namespace Arvo::RayTracing::Cpu
