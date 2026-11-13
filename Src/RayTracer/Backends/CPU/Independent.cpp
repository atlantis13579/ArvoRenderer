#include "RayTracer/Backends/CPU/CpuTransport.h"

namespace Arvo::RayTracing::Cpu
{
// A fixed integer avalanche maps a counter to reproducible uniform variates.
// The 24-bit mantissa conversion produces [0,1), including on tile boundaries.
constexpr uint32_t AvalancheMultiplierA = 0x7feb352du;
constexpr uint32_t AvalancheMultiplierB = 0x846ca68bu;
constexpr uint32_t MantissaMask = 0x00ffffffu;
constexpr uint32_t MantissaRange = 0x01000000u;
uint32_t Hash(uint32_t value)
{
    value ^= value >> 16;
    value *= AvalancheMultiplierA;
    value ^= value >> 15;
    value *= AvalancheMultiplierB;
    value ^= value >> 16;
    return value;
}

float Random01(uint32_t seed)
{
    return static_cast<float>(Hash(seed) & MantissaMask) / static_cast<float>(MantissaRange);
}

Float2 IndependentSample(const SampleContext& context, uint32_t dimension)
{
    const uint32_t seed = (context.PixelSeed + context.SampleIndex * SampleSeedMultiplier) ^ dimension;
    return {Random01(seed), Random01(seed ^ SecondDimensionScramble)};
}
}
