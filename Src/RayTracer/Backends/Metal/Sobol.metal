constant uint SobolLeadingBit = 0x80000000u;
constant uint SobolAlternatingBits = 0x55555555u, SobolBitPairs = 0x33333333u;
constant uint SobolNibbles = 0x0f0f0f0fu, SobolBytes = 0x00ff00ffu;
constant uint SobolRounds = 3u, SobolRoundKey = 0x9e3779b9u;
constant uint SobolPermutationShift = 5u, SobolDiscardedBits = 8u;
constant float SobolMantissaRange = 16777216.0f;

uint SobolReverseBits(uint value)
{
    value = ((value & SobolAlternatingBits) << 1u) | ((value >> 1u) & SobolAlternatingBits);
    value = ((value & SobolBitPairs) << 2u) | ((value >> 2u) & SobolBitPairs);
    value = ((value & SobolNibbles) << 4u) | ((value >> 4u) & SobolNibbles);
    value = ((value & SobolBytes) << 8u) | ((value >> 8u) & SobolBytes);
    return (value << 16u) | (value >> 16u);
}

uint SobolScramble(uint value, uint seed)
{
    uint result = 0u, prefix = 0u;
    for (uint bit = SobolLeadingBit; bit >= (1u << SobolDiscardedBits); bit >>= 1u)
    {
        const bool swap = (Hash(seed ^ prefix ^ bit) & 1u) != 0u;
        result |= (value & bit) ^ (swap ? bit : 0u);
        prefix |= value & bit;
    }
    return result;
}

uint SobolPermute(uint index, uint count, uint seed)
{
    uint mask = 1u;
    while (mask < count) mask <<= 1u;
    --mask;
    do
    {
        for (uint round = 0u; round < SobolRounds; ++round)
        {
            const uint key = Hash(seed + round * SobolRoundKey);
            index = (index * (key | 1u) + Hash(key)) & mask;
            index ^= index >> SobolPermutationShift;
        }
    } while (index >= count);
    return index;
}

// Padded 2D Sobol nets in native MSL. Hashed nested Owen scrambling retains
// elementary-interval coverage; shuffled indices associate the path domains.
// Owen (1995); PBRT 4e, "Sobol' Samplers".
float2 SobolSample(SampleContext context, uint dimension)
{
    const uint seed = Hash(context.PixelSeed ^ dimension);
    const uint index = SobolPermute(context.SampleIndex, context.SampleCount, seed);
    uint second = 0u, direction = SobolLeadingBit;
    for (uint bits = index; bits != 0u; bits >>= 1u)
    {
        if ((bits & 1u) != 0u) second ^= direction;
        direction ^= direction >> 1u;
    }
    return float2(float(SobolScramble(SobolReverseBits(index), seed) >> SobolDiscardedBits),
        float(SobolScramble(second, seed ^ SecondDimensionScramble) >> SobolDiscardedBits)) / SobolMantissaRange;
}
