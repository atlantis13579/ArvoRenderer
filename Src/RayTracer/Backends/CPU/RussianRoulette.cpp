#include "RayTracer/Backends/CPU/CpuTransport.h"
#include <algorithm>

namespace Arvo::RayTracing::Cpu
{
// Survival changes work, not expected radiance: callers divide the surviving
// throughput by this probability. Bounds avoid extreme weights and long tails.
float RouletteProbability(const Float3& throughput)
{
    constexpr float MinimumSurvival = 0.05f;
    constexpr float MaximumSurvival = 0.95f;
    const float energy = std::max({throughput.X, throughput.Y, throughput.Z});
    return energy > 0.0f ? std::clamp(energy, MinimumSurvival, MaximumSurvival) : 0.0f;
}
}
