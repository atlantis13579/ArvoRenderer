#include "RayTracer/Backends/CPU/CpuTransport.h"
#include <algorithm>

namespace Arvo::RayTracing::Cpu
{
namespace
{
constexpr uint32_t PixelCoordinateBits = 14;
constexpr uint32_t LeadingBit = 0x80000000u;
constexpr uint32_t DigitMask = 3u, PermutationCount = 24u, DigitSum = 6u;
constexpr uint32_t LevelScramble = 0x632be5abu, DimensionScramble = 0x02e5be93u;
constexpr uint32_t AlternatingBits = 0x55555555u, BitPairs = 0x33333333u;
constexpr uint32_t Nibbles = 0x0f0f0f0fu, Bytes = 0x00ff00ffu, DiscardedBits = 8u;
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
    uint32_t result = 0, prefix = 0;
    for (uint32_t bit = LeadingBit; bit >= (1u << DiscardedBits); bit >>= 1u)
    {
        const bool swap = (Hash(seed ^ prefix ^ bit) & 1u) != 0u;
        result |= (value & bit) ^ (swap ? bit : 0u);
        prefix |= value & bit;
    }
    return result;
}

uint32_t PermuteDigit(uint32_t digit, uint32_t rank)
{
    // Factoradic unranking supplies all 4! child permutations without a table.
    const uint32_t first = rank % 4u;
    rank /= 4u;
    uint32_t second = rank % 3u;
    if (second >= first) ++second;
    uint32_t third = rank / 3u;
    if (third >= std::min(first, second)) ++third;
    if (third >= std::max(first, second)) ++third;
    switch (digit)
    {
    case 0u: return first;
    case 1u: return second;
    case 2u: return third;
    default: return DigitSum - first - second - third;
    }
}
}

// Ahmed and Wonka (2020), "Screen-Space Blue-Noise Diffusion of Monte Carlo
// Sampling Error via Hierarchical Ordering of Pixels". This independent variant
// hashes the hierarchy rather than loading a grammar or a blue-noise texture.
// https://abdallagafar.com/publications/zsampler/
// A scrambled Morton order assigns contiguous Sobol blocks to nearby pixels.
// Power-of-two sample counts retain complete nets; arbitrary counts use a prefix
// of the next block. The output scramble is shared across pixels intentionally.
Float2 BlueNoiseSample(const SampleContext& context, uint32_t dimension)
{
    uint32_t sampleBits = 0, blockSize = 1;
    while (blockSize < context.SampleCount) { blockSize <<= 1u; ++sampleBits; }
    uint64_t morton = 0;
    for (uint32_t bit = 0; bit < PixelCoordinateBits; ++bit)
    {
        morton |= uint64_t((context.PixelX >> bit) & 1u) << (2u * bit);
        morton |= uint64_t((context.PixelY >> bit) & 1u) << (2u * bit + 1u);
    }
    const uint64_t original = (morton << sampleBits) | context.SampleIndex;
    uint64_t ordered = 0;
    const uint32_t seed = Hash(dimension ^ DimensionScramble ^ context.SamplingSeed);
    const uint32_t oddBit = sampleBits & 1u;
    const uint32_t levels = PixelCoordinateBits + sampleBits / 2u;
    for (uint32_t level = 0; level < levels; ++level)
    {
        const uint32_t shift = 2u * level + oddBit;
        const uint64_t parent = original >> (shift + 2u);
        const uint32_t key = Hash(uint32_t(parent) ^ Hash(uint32_t(parent >> 32u)) ^ seed ^ shift * LevelScramble);
        const uint32_t digit = uint32_t(original >> shift) & DigitMask;
        ordered |= uint64_t(PermuteDigit(digit, key % PermutationCount)) << shift;
    }
    if (oddBit)
    {
        const uint64_t parent = original >> 1u;
        const uint32_t key = Hash(uint32_t(parent) ^ Hash(uint32_t(parent >> 32u)) ^ seed);
        ordered |= (uint32_t(original) ^ key) & 1u;
    }
    uint32_t second = 0, direction = LeadingBit;
    for (uint64_t bits = ordered; bits; bits >>= 1u)
    {
        if (bits & 1u) second ^= direction;
        direction ^= direction >> 1u;
    }
    return {static_cast<float>(Scramble(ReverseBits(uint32_t(ordered)), seed) >> DiscardedBits) / MantissaRange,
        static_cast<float>(Scramble(second, seed ^ SecondDimensionScramble) >> DiscardedBits) / MantissaRange};
}
}
