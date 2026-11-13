#include "RayTracer/Backends/CPU/CpuTransport.h"

namespace Arvo::RayTracing::Cpu
{
namespace
{
constexpr uint32_t LeadingBit = 0x80000000u;
constexpr uint32_t AlternatingBits = 0x55555555u, BitPairs = 0x33333333u;
constexpr uint32_t Nibbles = 0x0f0f0f0fu, Bytes = 0x00ff00ffu;
constexpr uint32_t PermutationRounds = 3, PermutationRoundKey = 0x9e3779b9u;
constexpr uint32_t PermutationShift = 5, DiscardedBits = 8;
constexpr float MantissaRange = 16777216.0f;

uint32_t ReverseBits(uint32_t value)
{
    value = ((value & AlternatingBits) << 1u) | ((value >> 1u) & AlternatingBits);
    value = ((value & BitPairs) << 2u) | ((value >> 2u) & BitPairs);
    value = ((value & Nibbles) << 4u) | ((value >> 4u) & Nibbles);
    value = ((value & Bytes) << 8u) | ((value >> 8u) & Bytes);
    return (value << 16u) | (value >> 16u);
}

uint32_t Scramble(uint32_t value, uint32_t seed)
{
    // Each binary-tree node chooses its own swap. The terminating bit uniquely
    // identifies the level; hashing avoids storing the nested permutation tree.
    uint32_t result = 0, prefix = 0;
    for (uint32_t bit = LeadingBit; bit >= (1u << DiscardedBits); bit >>= 1u)
    {
        const bool swap = (Hash(seed ^ prefix ^ bit) & 1u) != 0u;
        result |= (value & bit) ^ (swap ? bit : 0u);
        prefix |= value & bit;
    }
    return result;
}

uint32_t Permute(uint32_t index, uint32_t count, uint32_t seed)
{
    uint32_t mask = 1u;
    while (mask < count) mask <<= 1u;
    --mask;
    // Each masked odd multiply and triangular XOR is bijective. Cycle walking
    // restricts the permutation to [0,count), including non-power-of-two counts.
    do
    {
        for (uint32_t round = 0; round < PermutationRounds; ++round)
        {
            const uint32_t key = Hash(seed + round * PermutationRoundKey);
            index = (index * (key | 1u) + Hash(key)) & mask;
            index ^= index >> PermutationShift;
        }
    } while (index >= count);
    return index;
}
}

// Padded two-dimensional Sobol nets, independently permuted per path domain.
// The second direction sequence follows v[j+1]=v[j] XOR (v[j]>>1).
// Owen (1995); PBRT 4e, "Sobol' Samplers", section on padded sampling.
Float2 SobolSample(const SampleContext& context, uint32_t dimension)
{
    const uint32_t seed = Hash(context.PixelSeed ^ dimension);
    const uint32_t index = Permute(context.SampleIndex, context.SampleCount, seed);
    uint32_t second = 0, direction = LeadingBit;
    for (uint32_t bits = index; bits; bits >>= 1u)
    {
        if (bits & 1u) second ^= direction;
        direction ^= direction >> 1u;
    }
    return {static_cast<float>(Scramble(ReverseBits(index), seed) >> DiscardedBits) / MantissaRange,
        static_cast<float>(Scramble(second, seed ^ SecondDimensionScramble) >> DiscardedBits) / MantissaRange};
}
}
