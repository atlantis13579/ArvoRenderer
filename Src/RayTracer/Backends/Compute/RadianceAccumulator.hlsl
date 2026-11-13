// Exact nonnegative binary32 sums, rounded once at film commit. Integer atomic
// carries commute, so light splats survive crop/retry scheduling bit-for-bit.
// Each float is an integer multiple of 2^-149. Ten 32-bit words accommodate
// UINT_MAX paths times 35 contributions even at the largest finite float.
// Related: Neal (2015), "Fast exact summation using small and large
// superaccumulators", https://arxiv.org/abs/1505.05571. This is a compact
// unsigned accumulator for radiance, not Neal's general signed algorithm.
static const uint RadianceSumWords = 10, RadianceWordBits = 32;
static const uint RadianceFractionBits = 23, RadianceFractionMask = 0x007fffffu;
static const uint RadianceHiddenBit = 0x00800000u, RadianceExponentMask = 0xffu;
static const uint RadianceSignBit = 0x80000000u, RadianceWordBytes = 4;
bool AddRadianceWord(RWByteAddressBuffer words, uint base, uint word, uint increment)
{
    [allow_uav_condition] while (increment != 0 && word < RadianceSumWords)
    {
        uint previous;
        words.InterlockedAdd(base + word * RadianceWordBytes, increment, previous);
        increment = previous + increment < previous ? 1 : 0;
        ++word;
    }
    return increment == 0;
}
bool AccumulateRadiance(RWByteAddressBuffer words, uint base, float value)
{
    uint bits = asuint(value), exponent = (bits >> RadianceFractionBits) & RadianceExponentMask;
    if (exponent == RadianceExponentMask ||
        ((bits & RadianceSignBit) != 0 && (bits & ~RadianceSignBit) != 0))
        return false;
    uint significand = bits & RadianceFractionMask;
    if (exponent != 0) significand |= RadianceHiddenBit;
    uint shift = exponent == 0 ? 0 : exponent - 1;
    uint word = shift / RadianceWordBits, offset = shift % RadianceWordBits;
    bool valid = AddRadianceWord(words, base, word, significand << offset);
    if (offset != 0)
        valid = AddRadianceWord(words, base, word + 1, significand >> (RadianceWordBits - offset)) && valid;
    return valid;
}
uint LoadRadianceWord(RWByteAddressBuffer words, uint base, uint word)
{
    return word < RadianceSumWords ? words.Load(base + word * RadianceWordBytes) : 0;
}
float ResolveRadiance(RWByteAddressBuffer words, uint base)
{
    int highest = int(RadianceSumWords) - 1;
    while (highest > 0 && LoadRadianceWord(words, base, uint(highest)) == 0) --highest;
    uint top = LoadRadianceWord(words, base, uint(highest));
    if (highest == 0 && top < RadianceHiddenBit) return asfloat(top);
    uint leading = uint(highest) * RadianceWordBits;
    while (top > 1) { top >>= 1; ++leading; }
    uint discarded = leading - RadianceFractionBits;
    uint word = discarded / RadianceWordBits, offset = discarded % RadianceWordBits;
    uint significand = LoadRadianceWord(words, base, word) >> offset;
    if (offset != 0)
        significand |= LoadRadianceWord(words, base, word + 1) << (RadianceWordBits - offset);
    if (discarded != 0)
    {
        uint roundWord = (discarded - 1) / RadianceWordBits;
        uint roundOffset = (discarded - 1) % RadianceWordBits;
        uint tail = LoadRadianceWord(words, base, roundWord);
        bool sticky = (tail & ((1u << roundOffset) - 1)) != 0;
        for (uint lower = 0; lower < roundWord; ++lower)
            sticky = sticky || LoadRadianceWord(words, base, lower) != 0;
        if (((tail >> roundOffset) & 1) != 0 && (sticky || (significand & 1) != 0)) ++significand;
    }
    uint exponent = leading - RadianceFractionBits + 1;
    if (significand == RadianceHiddenBit * 2) { significand >>= 1; ++exponent; }
    if (exponent >= RadianceExponentMask) return asfloat(RadianceExponentMask << RadianceFractionBits);
    return asfloat((exponent << RadianceFractionBits) | (significand & RadianceFractionMask));
}
