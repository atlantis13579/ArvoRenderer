static const uint SobolLeadingBit = 0x80000000u;
static const uint SobolPermutationRounds = 3u, SobolRoundKey = 0x9e3779b9u;
static const uint SobolPermutationShift = 5u, SobolDiscardedBits = 8u;
static const float SobolMantissaRange = 16777216.0f;

uint SobolScramble(uint value, uint seed)
{
    uint result = 0u, prefix = 0u;
    [loop] for (uint bit = SobolLeadingBit; bit >= (1u << SobolDiscardedBits); bit >>= 1u)
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
    [loop] while (mask < count) mask <<= 1u;
    --mask;
    [loop] do
    {
        for (uint round = 0; round < SobolPermutationRounds; ++round)
        {
            const uint key = Hash(seed + round * SobolRoundKey);
            index = (index * (key | 1u) + Hash(key)) & mask;
            index ^= index >> SobolPermutationShift;
        }
    } while (index >= count);
    return index;
}

// Padded Sobol nets with a hashed nested Owen scramble. A separate
// permutation associates each 2D domain with the path's sample index.
// Owen (1995); PBRT 4e, "Sobol' Samplers".
float2 SobolSample(SampleContext context, uint dimension)
{
    const uint seed = Hash(context.PixelSeed ^ dimension);
    const uint index = SobolPermute(context.SampleIndex, context.SampleCount, seed);
    uint second = 0u, direction = SobolLeadingBit;
    [loop] for (uint bits = index; bits != 0u; bits >>= 1u)
    {
        if ((bits & 1u) != 0u) second ^= direction;
        direction ^= direction >> 1u;
    }
    return float2(float(SobolScramble(reversebits(index), seed) >> SobolDiscardedBits),
        float(SobolScramble(second, seed ^ SecondDimensionScramble) >> SobolDiscardedBits)) / SobolMantissaRange;
}
