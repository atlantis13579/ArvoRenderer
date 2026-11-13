#include "RayTracer/Backends/CPU/CpuTransport.h"

namespace Arvo::RayTracing::Cpu
{
RenderStatistics AtomicStatistics::Resolve(double buildMilliseconds, double renderMilliseconds) const
{
    RenderStatistics statistics;
    statistics.PrimaryRays = PrimaryRays.load(std::memory_order_relaxed);
    statistics.PilotRays = PilotRays.load(std::memory_order_relaxed);
    statistics.ShadowRays = ShadowRays.load(std::memory_order_relaxed);
    statistics.ReflectionRays = ReflectionRays.load(std::memory_order_relaxed);
    statistics.PrimitiveTests = PrimitiveTests.load(std::memory_order_relaxed);
    statistics.CompletedPixels = CompletedPixels.load(std::memory_order_relaxed);
    statistics.BuildMilliseconds = buildMilliseconds;
    statistics.RenderMilliseconds = renderMilliseconds;
    return statistics;
}
}
