// DXR-owned exact positive radiance accumulation. Binary32 values become
// integers in units of 2^-149; atomic word additions propagate unsigned carries.
// A 320-bit channel covers the complete binary32 range and UINT_MAX * 35
// contributions. No quantization scale or order-dependent float atomic is used.
// Superaccumulator background: Neal (2015), "Fast exact summation using small
// and large superaccumulators", https://arxiv.org/abs/1505.05571.
static const uint SplatSumWordCount = 10, SplatWordBits = 32, SplatWordBytes = 4;
static const uint SplatFractionBits = 23, SplatFractionMask = 0x007fffffu;
static const uint SplatImplicitBit = 0x00800000u, SplatExponentLimit = 0xffu;
static const uint SplatSignMask = 0x80000000u;
bool SplatAddInteger(RWByteAddressBuffer storage, uint address, uint digit, uint value)
{
    [allow_uav_condition] while (value != 0 && digit < SplatSumWordCount)
    {
        uint old;
        storage.InterlockedAdd(address + digit * SplatWordBytes, value, old);
        value = old + value < old ? 1u : 0u;
        ++digit;
    }
    return value == 0;
}
bool SplatAccumulate(RWByteAddressBuffer storage, uint address, float radiance)
{
    uint encoding = asuint(radiance);
    uint exponent = (encoding >> SplatFractionBits) & SplatExponentLimit;
    if (exponent == SplatExponentLimit ||
        ((encoding & SplatSignMask) != 0 && (encoding & ~SplatSignMask) != 0)) return false;
    uint mantissa = encoding & SplatFractionMask;
    uint shift = 0;
    if (exponent > 0) { mantissa |= SplatImplicitBit; shift = exponent - 1; }
    uint digit = shift / SplatWordBits, bit = shift % SplatWordBits;
    bool valid = SplatAddInteger(storage, address, digit, mantissa << bit);
    if (bit > 0)
        valid = SplatAddInteger(storage, address, digit + 1, mantissa >> (SplatWordBits - bit)) && valid;
    return valid;
}
uint SplatDigit(RWByteAddressBuffer storage, uint address, uint digit)
{
    return digit < SplatSumWordCount ? storage.Load(address + digit * SplatWordBytes) : 0;
}
float SplatRound(RWByteAddressBuffer storage, uint address)
{
    uint digit = SplatSumWordCount - 1;
    while (digit > 0 && SplatDigit(storage, address, digit) == 0) --digit;
    uint head = SplatDigit(storage, address, digit);
    if (digit == 0 && head < SplatImplicitBit) return asfloat(head);
    uint highestBit = digit * SplatWordBits;
    while (head > 1) { head >>= 1; ++highestBit; }
    uint shift = highestBit - SplatFractionBits;
    uint first = shift / SplatWordBits, bit = shift % SplatWordBits;
    uint mantissa = SplatDigit(storage, address, first) >> bit;
    if (bit > 0) mantissa |= SplatDigit(storage, address, first + 1) << (SplatWordBits - bit);
    if (shift > 0)
    {
        uint guardDigit = (shift - 1) / SplatWordBits, guardBit = (shift - 1) % SplatWordBits;
        uint guard = SplatDigit(storage, address, guardDigit);
        bool remainder = (guard & ((1u << guardBit) - 1)) != 0;
        for (uint i = 0; i < guardDigit; ++i) remainder = remainder || SplatDigit(storage, address, i) != 0;
        if (((guard >> guardBit) & 1) != 0 && (remainder || (mantissa & 1) != 0)) ++mantissa;
    }
    uint exponent = highestBit - SplatFractionBits + 1;
    if (mantissa >= SplatImplicitBit * 2) { mantissa >>= 1; ++exponent; }
    if (exponent >= SplatExponentLimit) return asfloat(SplatExponentLimit << SplatFractionBits);
    return asfloat((exponent << SplatFractionBits) | (mantissa & SplatFractionMask));
}
