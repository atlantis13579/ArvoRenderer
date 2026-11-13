static const uint BlueCoordinateBits = 14u, BlueLeadingBit = 0x80000000u;
static const uint BlueDigitMask = 3u, BluePermutationCount = 24u, BlueDigitSum = 6u;
static const uint BlueLevelScramble = 0x632be5abu, BlueDimensionScramble = 0x02e5be93u;
static const uint BlueWordBits = 32u, BlueDiscardedBits = 8u;
static const float BlueMantissaRange = 16777216.0f;

uint BlueScramble(uint value, uint seed)
{
    uint result = 0u, prefix = 0u;
    [loop] for (uint bit = BlueLeadingBit; bit >= (1u << BlueDiscardedBits); bit >>= 1u)
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
    switch (digit)
    {
    case 0u: return first;
    case 1u: return second;
    case 2u: return third;
    default: return BlueDigitSum - first - second - third;
    }
}

uint2 BlueShiftRight(uint2 value, uint shift)
{
    if (shift == 0u) return value;
    if (shift >= BlueWordBits) return uint2(value.y >> (shift - BlueWordBits), 0u);
    return uint2((value.x >> shift) | (value.y << (BlueWordBits - shift)), value.y >> shift);
}

void BlueInsert(inout uint2 value, uint digit, uint shift)
{
    if (shift >= BlueWordBits) value.y |= digit << (shift - BlueWordBits);
    else
    {
        value.x |= digit << shift;
        if (shift != 0u) value.y |= digit >> (BlueWordBits - shift);
    }
}

// A native DXR 40-bit hierarchy distributes consecutive Sobol blocks over a
// scrambled Morton pixel order. No other sampler or backend supplies samples.
// Ahmed & Wonka (2020), https://abdallagafar.com/publications/zsampler/
float2 BlueNoiseSample(SampleContext context, uint dimension)
{
    uint sampleBits = 0u, blockSize = 1u;
    [loop] while (blockSize < context.SampleCount) { blockSize <<= 1u; ++sampleBits; }
    uint2 original = uint2(context.SampleIndex, 0u);
    for (uint bit = 0u; bit < BlueCoordinateBits; ++bit)
    {
        BlueInsert(original, (context.PixelX >> bit) & 1u, sampleBits + 2u * bit);
        BlueInsert(original, (context.PixelY >> bit) & 1u, sampleBits + 2u * bit + 1u);
    }
    uint2 ordered = uint2(0u, 0u);
    const uint seed = Hash(dimension ^ BlueDimensionScramble ^ context.SamplingSeed);
    const uint oddBit = sampleBits & 1u;
    const uint levels = BlueCoordinateBits + sampleBits / 2u;
    [loop] for (uint level = 0u; level < levels; ++level)
    {
        const uint shift = 2u * level + oddBit;
        const uint2 parent = BlueShiftRight(original, shift + 2u);
        const uint key = Hash(parent.x ^ Hash(parent.y) ^ seed ^ shift * BlueLevelScramble);
        const uint digit = BlueShiftRight(original, shift).x & BlueDigitMask;
        BlueInsert(ordered, BluePermuteDigit(digit, key % BluePermutationCount), shift);
    }
    if (oddBit != 0u)
    {
        const uint2 parent = BlueShiftRight(original, 1u);
        const uint key = Hash(parent.x ^ Hash(parent.y) ^ seed);
        ordered.x |= (original.x ^ key) & 1u;
    }
    uint second = 0u, direction = BlueLeadingBit;
    uint2 bits = ordered;
    [loop] while ((bits.x | bits.y) != 0u)
    {
        if ((bits.x & 1u) != 0u) second ^= direction;
        direction ^= direction >> 1u;
        bits = BlueShiftRight(bits, 1u);
    }
    const uint x = BlueScramble(reversebits(ordered.x), seed);
    const uint y = BlueScramble(second, seed ^ SecondDimensionScramble);
    return float2(float(x >> BlueDiscardedBits), float(y >> BlueDiscardedBits)) / BlueMantissaRange;
}
