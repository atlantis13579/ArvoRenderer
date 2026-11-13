constexpr unsigned int BlueCoordinateBits = 14u, BlueLeadingBit = 0x80000000u;
constexpr unsigned int BlueDigitMask = 3u, BluePermutationCount = 24u, BlueDigitSum = 6u;
constexpr unsigned int BlueLevelScramble = 0x632be5abu, BlueDimensionScramble = 0x02e5be93u;
constexpr unsigned int BlueDiscardedBits = 8u;
constexpr float BlueMantissaRange = 16777216.0f;

__device__ unsigned int BlueScramble(unsigned int value, unsigned int seed)
{
    unsigned int result = 0u, prefix = 0u;
    for (unsigned int bit = BlueLeadingBit; bit >= (1u << BlueDiscardedBits); bit >>= 1u)
    {
        const bool swap = (Hash(seed ^ prefix ^ bit) & 1u) != 0u;
        result |= (value & bit) ^ (swap ? bit : 0u);
        prefix |= value & bit;
    }
    return result;
}

__device__ unsigned int BluePermuteDigit(unsigned int digit, unsigned int rank)
{
    const unsigned int first = rank % 4u;
    rank /= 4u;
    unsigned int second = rank % 3u;
    if (second >= first) ++second;
    unsigned int third = rank / 3u;
    if (third >= (first < second ? first : second)) ++third;
    if (third >= (first > second ? first : second)) ++third;
    switch (digit)
    {
    case 0u: return first;
    case 1u: return second;
    case 2u: return third;
    default: return BlueDigitSum - first - second - third;
    }
}

// CUDA's native 64-bit index preserves the hierarchy at large image coordinates.
// Hash-based child permutations replace external lookup assets. Output digit
// scrambling is shared across the image to preserve neighboring-pixel coverage.
// Ahmed & Wonka (2020), https://abdallagafar.com/publications/zsampler/
__device__ float2 BlueNoiseSample(SampleContext context, unsigned int dimension)
{
    unsigned int sampleBits = 0u, block = 1u;
    while (block < context.SampleCount) { block <<= 1u; ++sampleBits; }
    unsigned long long morton = 0u;
    for (unsigned int bit = 0u; bit < BlueCoordinateBits; ++bit)
    {
        morton |= static_cast<unsigned long long>((context.PixelX >> bit) & 1u) << (2u * bit);
        morton |= static_cast<unsigned long long>((context.PixelY >> bit) & 1u) << (2u * bit + 1u);
    }
    const unsigned long long original = (morton << sampleBits) | context.SampleIndex;
    unsigned long long ordered = 0u;
    const unsigned int seed = Hash(dimension ^ BlueDimensionScramble ^ context.SamplingSeed);
    const unsigned int oddBit = sampleBits & 1u;
    const unsigned int levels = BlueCoordinateBits + sampleBits / 2u;
    for (unsigned int level = 0u; level < levels; ++level)
    {
        const unsigned int shift = 2u * level + oddBit;
        const unsigned long long parent = original >> (shift + 2u);
        const unsigned int key = Hash(static_cast<unsigned int>(parent)
            ^ Hash(static_cast<unsigned int>(parent >> 32u)) ^ seed ^ shift * BlueLevelScramble);
        const unsigned int digit = static_cast<unsigned int>(original >> shift) & BlueDigitMask;
        ordered |= static_cast<unsigned long long>(BluePermuteDigit(digit, key % BluePermutationCount)) << shift;
    }
    if (oddBit)
    {
        const unsigned long long parent = original >> 1u;
        const unsigned int key = Hash(static_cast<unsigned int>(parent)
            ^ Hash(static_cast<unsigned int>(parent >> 32u)) ^ seed);
        ordered |= (static_cast<unsigned int>(original) ^ key) & 1u;
    }
    unsigned int second = 0u, direction = BlueLeadingBit;
    for (unsigned long long bits = ordered; bits != 0u; bits >>= 1u)
    {
        if (bits & 1u) second ^= direction;
        direction ^= direction >> 1u;
    }
    return make_float2(static_cast<float>(BlueScramble(__brev(static_cast<unsigned int>(ordered)), seed)
            >> BlueDiscardedBits) / BlueMantissaRange,
        static_cast<float>(BlueScramble(second, seed ^ SecondDimensionScramble) >> BlueDiscardedBits) / BlueMantissaRange);
}
