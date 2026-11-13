// Metal-owned exact positive radiance sum. Store binary32 values as integers
// in units of the smallest subnormal, then round once at pass completion.
// Ten atomic uint words per channel cover UINT_MAX paths * 35 contributions;
// integer carries remove any dependency on thread or tile scheduling.
// Background: Neal (2015), "Fast exact summation using small and large
// superaccumulators", https://arxiv.org/abs/1505.05571.
constant uint RadianceWords = 10, RadianceWordBits = 32;
constant uint RadianceFractionBits = 23, RadianceFractionMask = 0x007fffffu;
constant uint RadianceImplicitBit = 0x00800000u, RadianceExponentLimit = 0xffu;
constant uint RadianceSignBit = 0x80000000u;
bool RadianceAddWord(device atomic_uint* sum, uint index, uint value)
{
    while (value != 0 && index < RadianceWords)
    {
        uint old = atomic_fetch_add_explicit(sum + index, value, memory_order_relaxed);
        value = old + value < old ? 1u : 0u;
        ++index;
    }
    return value == 0;
}
bool RadianceAdd(device atomic_uint* sum, float value)
{
    uint raw = as_type<uint>(value), exponent = (raw >> RadianceFractionBits) & RadianceExponentLimit;
    if (exponent == RadianceExponentLimit ||
        ((raw & RadianceSignBit) != 0 && (raw & ~RadianceSignBit) != 0)) return false;
    uint fraction = raw & RadianceFractionMask;
    if (exponent != 0) fraction |= RadianceImplicitBit;
    uint shift = exponent > 0 ? exponent - 1 : 0;
    uint index = shift / RadianceWordBits, bit = shift % RadianceWordBits;
    bool valid = RadianceAddWord(sum, index, fraction << bit);
    if (bit != 0)
        valid = RadianceAddWord(sum, index + 1, fraction >> (RadianceWordBits - bit)) && valid;
    return valid;
}
uint RadianceDigit(device atomic_uint* sum, uint index)
{
    return index < RadianceWords ? atomic_load_explicit(sum + index, memory_order_relaxed) : 0;
}
float RadianceRound(device atomic_uint* sum)
{
    uint last = RadianceWords - 1;
    while (last > 0 && RadianceDigit(sum, last) == 0) --last;
    uint head = RadianceDigit(sum, last);
    if (last == 0 && head < RadianceImplicitBit) return as_type<float>(head);
    uint highBit = last * RadianceWordBits;
    while (head > 1) { head >>= 1; ++highBit; }
    uint shift = highBit - RadianceFractionBits, first = shift / RadianceWordBits;
    uint bit = shift % RadianceWordBits;
    uint significand = RadianceDigit(sum, first) >> bit;
    if (bit != 0) significand |= RadianceDigit(sum, first + 1) << (RadianceWordBits - bit);
    if (shift > 0)
    {
        uint guardIndex = (shift - 1) / RadianceWordBits, guardBit = (shift - 1) % RadianceWordBits;
        uint guard = RadianceDigit(sum, guardIndex);
        bool tail = (guard & ((1u << guardBit) - 1)) != 0;
        for (uint i = 0; i < guardIndex; ++i) tail = tail || RadianceDigit(sum, i) != 0;
        if (((guard >> guardBit) & 1) != 0 && (tail || (significand & 1) != 0)) ++significand;
    }
    uint exponent = highBit - RadianceFractionBits + 1;
    if (significand == RadianceImplicitBit * 2) { significand >>= 1; ++exponent; }
    if (exponent >= RadianceExponentLimit) return as_type<float>(RadianceExponentLimit << RadianceFractionBits);
    return as_type<float>((exponent << RadianceFractionBits) | (significand & RadianceFractionMask));
}
