constant uint BlueCoordinateBits = 14u, BlueLeadingBit = 0x80000000u;
constant uint BlueDigitMask = 3u, BluePermutationCount = 24u, BlueDigitSum = 6u;
constant uint BlueLevelScramble = 0x632be5abu, BlueDimensionScramble = 0x02e5be93u;
constant uint BlueAlternatingBits = 0x55555555u, BlueBitPairs = 0x33333333u;
constant uint BlueNibbles = 0x0f0f0f0fu, BlueBytes = 0x00ff00ffu, BlueDiscardedBits = 8u;
constant float BlueMantissaRange = 16777216.0f;

uint BlueReverseBits(uint value)
{
    value = ((value & BlueAlternatingBits) << 1u) | ((value >> 1u) & BlueAlternatingBits);
    value = ((value & BlueBitPairs) << 2u) | ((value >> 2u) & BlueBitPairs);
    value = ((value & BlueNibbles) << 4u) | ((value >> 4u) & BlueNibbles);
    value = ((value & BlueBytes) << 8u) | ((value >> 8u) & BlueBytes);
    return (value << 16u) | (value >> 16u);
}

uint BlueScramble(uint value, uint seed)
{
    uint result = 0u, prefix = 0u;
    for (uint bit = BlueLeadingBit; bit >= (1u << BlueDiscardedBits); bit >>= 1u)
    {
        const bool swap = (Hash(seed ^ prefix ^ bit) & 1u) != 0u;
        result |= (value & bit) ^ (swap ? bit : 0u);
        prefix |= value & bit;
    }
    return result;
}

uint BluePermuteDigit(uint digit, uint rank)
{
    const uint first = rank % 4u;
    rank /= 4u;
    uint second = rank % 3u;
    if (second >= first) ++second;
    uint third = rank / 3u;
    if (third >= min(first, second)) ++third;
    if (third >= max(first, second)) ++third;
    if (digit == 0u) return first;
    if (digit == 1u) return second;
    if (digit == 2u) return third;
    return BlueDigitSum - first - second - third;
}

// Native Metal hierarchy with no downloaded lookup resources. Neighbors share
// contiguous Sobol blocks through a scrambled Morton order; output scrambling
// remains global for each domain, preserving screen-space decorrelation.
// Ahmed & Wonka (2020), https://abdallagafar.com/publications/zsampler/
float2 BlueNoiseSample(SampleContext context, uint dimension)
{
    uint sampleBits = 0u, blockSize = 1u;
    while (blockSize < context.SampleCount) { blockSize <<= 1u; ++sampleBits; }
    ulong morton = 0u;
    for (uint bit = 0u; bit < BlueCoordinateBits; ++bit)
    {
        morton |= ulong((context.PixelX >> bit) & 1u) << (2u * bit);
        morton |= ulong((context.PixelY >> bit) & 1u) << (2u * bit + 1u);
    }
    const ulong original = (morton << sampleBits) | context.SampleIndex;
    ulong ordered = 0u;
    const uint seed = Hash(dimension ^ BlueDimensionScramble ^ context.SamplingSeed);
    const uint oddBit = sampleBits & 1u;
    const uint levels = BlueCoordinateBits + sampleBits / 2u;
    for (uint level = 0u; level < levels; ++level)
    {
        const uint shift = 2u * level + oddBit;
        const ulong parent = original >> (shift + 2u);
        const uint key = Hash(uint(parent) ^ Hash(uint(parent >> 32u)) ^ seed ^ shift * BlueLevelScramble);
        const uint digit = uint(original >> shift) & BlueDigitMask;
        ordered |= ulong(BluePermuteDigit(digit, key % BluePermutationCount)) << shift;
    }
    if (oddBit != 0u)
    {
        const ulong parent = original >> 1u;
        const uint key = Hash(uint(parent) ^ Hash(uint(parent >> 32u)) ^ seed);
        ordered |= (uint(original) ^ key) & 1u;
    }
    uint second = 0u, direction = BlueLeadingBit;
    for (ulong bits = ordered; bits != 0u; bits >>= 1u)
    {
        if ((bits & 1u) != 0u) second ^= direction;
        direction ^= direction >> 1u;
    }
    return float2(float(BlueScramble(BlueReverseBits(uint(ordered)), seed) >> BlueDiscardedBits),
        float(BlueScramble(second, seed ^ SecondDimensionScramble) >> BlueDiscardedBits)) / BlueMantissaRange;
}
