constexpr unsigned int SobolLeadingBit = 0x80000000u;
constexpr unsigned int SobolRounds = 3u, SobolRoundKey = 0x9e3779b9u;
constexpr unsigned int SobolPermutationShift = 5u, SobolDiscardedBits = 8u;
constexpr float SobolMantissaRange = 16777216.0f;

__device__ unsigned int SobolScramble(unsigned int value, unsigned int seed)
{
    unsigned int result = 0u, prefix = 0u;
    for (unsigned int bit = SobolLeadingBit; bit >= (1u << SobolDiscardedBits); bit >>= 1u)
    {
        const bool swap = (Hash(seed ^ prefix ^ bit) & 1u) != 0u;
        result |= (value & bit) ^ (swap ? bit : 0u);
        prefix |= value & bit;
    }
    return result;
}

__device__ unsigned int SobolPermute(unsigned int index, unsigned int count, unsigned int seed)
{
    unsigned int mask = 1u;
    while (mask < count) mask <<= 1u;
    --mask;
    do
    {
        for (unsigned int round = 0u; round < SobolRounds; ++round)
        {
            const unsigned int key = Hash(seed + round * SobolRoundKey);
            index = (index * (key | 1u) + Hash(key)) & mask;
            index ^= index >> SobolPermutationShift;
        }
    } while (index >= count);
    return index;
}

// Hashed nested Owen scrambling retains dyadic nets. Padding independently
// permutes 2D domains; this is not a claim of an arbitrary-dimensional net.
// Owen (1995); PBRT 4e, "Sobol' Samplers".
__device__ float2 SobolSample(SampleContext context, unsigned int dimension)
{
    const unsigned int seed = Hash(context.PixelSeed ^ dimension);
    const unsigned int index = SobolPermute(context.SampleIndex, context.SampleCount, seed);
    unsigned int second = 0u, direction = SobolLeadingBit;
    for (unsigned int bits = index; bits != 0u; bits >>= 1u)
    {
        if (bits & 1u) second ^= direction;
        direction ^= direction >> 1u;
    }
    return make_float2(static_cast<float>(SobolScramble(__brev(index), seed) >> SobolDiscardedBits) / SobolMantissaRange,
        static_cast<float>(SobolScramble(second, seed ^ SecondDimensionScramble) >> SobolDiscardedBits) / SobolMantissaRange);
}
