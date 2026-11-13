// CUDA exact nonnegative binary32 sum. A channel is a 320-bit unsigned integer
// with quantum 2^-149. Atomic integer additions and their carries commute;
// rounding to nearest/even happens once, after every path in the pass finishes.
// Capacity covers UINT_MAX paths * 35 terms at the largest finite float.
// Related: Neal (2015), "Fast exact summation using small and large
// superaccumulators", https://arxiv.org/abs/1505.05571.
namespace ArvoCudaRadiance
{
constexpr unsigned WordCount = 10, WordBits = 32;
constexpr unsigned FractionBits = 23, FractionMask = 0x007fffffu;
constexpr unsigned ImplicitBit = 0x00800000u, ExponentLimit = 0xffu, SignMask = 0x80000000u;
__device__ bool AddDigit(unsigned* sum, unsigned digit, unsigned increment)
{
    while (increment && digit < WordCount)
    {
        unsigned previous = atomicAdd(sum + digit, increment);
        increment = previous + increment < previous;
        ++digit;
    }
    return increment == 0;
}
__device__ bool Add(unsigned* sum, float value)
{
    unsigned bits = __float_as_uint(value), exponent = (bits >> FractionBits) & ExponentLimit;
    if (exponent == ExponentLimit || ((bits & SignMask) && (bits & ~SignMask))) return false;
    unsigned mantissa = bits & FractionMask;
    if (exponent) mantissa |= ImplicitBit;
    unsigned shift = exponent ? exponent - 1 : 0;
    unsigned digit = shift / WordBits, offset = shift % WordBits;
    bool valid = AddDigit(sum, digit, mantissa << offset);
    if (offset)
        valid = AddDigit(sum, digit + 1, mantissa >> (WordBits - offset)) && valid;
    return valid;
}
__device__ unsigned Digit(const unsigned* sum, unsigned index)
{
    return index < WordCount ? sum[index] : 0;
}
__device__ float Round(const unsigned* sum)
{
    unsigned last = WordCount - 1;
    while (last && sum[last] == 0) --last;
    unsigned top = sum[last];
    if (last == 0 && top < ImplicitBit) return __uint_as_float(top);
    unsigned leading = last * WordBits;
    for (; top > 1; top >>= 1) ++leading;
    unsigned shift = leading - FractionBits, first = shift / WordBits, offset = shift % WordBits;
    unsigned mantissa = Digit(sum, first) >> offset;
    if (offset) mantissa |= Digit(sum, first + 1) << (WordBits - offset);
    if (shift)
    {
        unsigned guardIndex = (shift - 1) / WordBits, guardOffset = (shift - 1) % WordBits;
        unsigned guard = Digit(sum, guardIndex);
        bool sticky = (guard & ((1u << guardOffset) - 1)) != 0;
        for (unsigned i = 0; i < guardIndex; ++i) sticky |= sum[i] != 0;
        if (((guard >> guardOffset) & 1) && (sticky || (mantissa & 1))) ++mantissa;
    }
    unsigned exponent = leading - FractionBits + 1;
    if (mantissa == ImplicitBit * 2) { mantissa >>= 1; ++exponent; }
    if (exponent >= ExponentLimit) return __uint_as_float(ExponentLimit << FractionBits);
    return __uint_as_float((exponent << FractionBits) | (mantissa & FractionMask));
}
} // namespace ArvoCudaRadiance
