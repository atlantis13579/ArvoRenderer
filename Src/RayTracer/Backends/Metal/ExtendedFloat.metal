// Scaled float expansions for affine ray transforms. MSL has no double type;
// two significands keep about 48 bits while a separate exponent carries range.
// Dekker (1971), "A Floating-Point Technique for Extending the Available Precision".
// FMA supplies the product residual; compilation must disable fast math.
// IEEE binary32 inputs/outputs are decoded as integers so subnormal endpoints
// survive devices that flush native float arithmetic. This is not IEEE binary64.
namespace WideMath
{
#if defined(__METAL_VERSION__)
#pragma STDC FP_CONTRACT OFF
#endif
constant uint FractionBits = 23u;
constant uint FractionMask = 0x007fffffu;
constant uint SignMask = 0x80000000u;
constant uint InfinityBits = 0x7f800000u;
constant uint HalfExponentBits = 126u << FractionBits;
constant int HalfExponentBias = 126;
constant int SubnormalUnitExponent = -149;
constant int MinimumNormalExponent = -125;
constant int MaximumNormalExponent = 128;
constant int AlignmentLimit = 64;

struct Number
{
    float High, Low;
    int Exponent;
    uint Reserved;
};

Number Normalize(float high, float low, int exponent)
{
    const float sum = high + low;
    const float recovered = sum - high;
    const float residual = (high - (sum - recovered)) + (low - recovered);
    if (sum == 0.0f)
        return {0.0f, 0.0f, 0, 0u};
    int shift = 0;
    const float significand = frexp(sum, shift);
    return {significand, ldexp(residual, -shift), exponent + shift, 0u};
}

Number Decode(uint bits)
{
    const uint fraction = bits & FractionMask;
    const uint storedExponent = (bits & ~SignMask) >> FractionBits;
    if (storedExponent != 0u)
        return {as_type<float>((bits & SignMask) | HalfExponentBits | fraction), 0.0f,
                int(storedExponent) - HalfExponentBias, 0u};
    if (fraction == 0u)
        return {0.0f, 0.0f, 0, 0u};
    const uint highest = 31u - clz(fraction);
    const uint normalized = (fraction << (FractionBits - highest)) & FractionMask;
    return {as_type<float>((bits & SignMask) | HalfExponentBits | normalized), 0.0f,
            int(highest) + 1 + SubnormalUnitExponent, 0u};
}

Number FromFloat(float value)
{
    return Decode(as_type<uint>(value));
}
Number Negate(Number value)
{
    value.High = -value.High;
    value.Low = -value.Low;
    return value;
}
Number Shift(Number value, int exponent)
{
    value.Exponent += exponent;
    return value;
}
Number Absolute(Number value)
{
    return value.High < 0.0f ? Negate(value) : value;
}

Number Add(Number a, Number b)
{
    if (a.High == 0.0f)
        return b;
    if (b.High == 0.0f)
        return a;
    if (a.Exponent < b.Exponent)
    {
        const Number temporary = a;
        a = b;
        b = temporary;
    }
    const int difference = a.Exponent - b.Exponent;
    if (difference > AlignmentLimit)
        return a;
    const float high = ldexp(b.High, -difference), low = ldexp(b.Low, -difference);
    const float sum = a.High + high;
    const float recovered = sum - a.High;
    const float residual = (a.High - (sum - recovered)) + (high - recovered);
    return Normalize(sum, (a.Low + low) + residual, a.Exponent);
}
Number Subtract(Number a, Number b)
{
    return Add(a, Negate(b));
}

Number Multiply(Number a, Number b)
{
    const float product = a.High * b.High;
    const float residual = fma(a.High, b.High, -product) + (a.High * b.Low + a.Low * b.High);
    return Normalize(product, residual, a.Exponent + b.Exponent);
}

Number Divide(Number a, Number b)
{
    // Callers test a zero denominator before division (parallel slabs/rays).
    const float quotient = a.High / b.High;
    const float residual = fma(-quotient, b.High, a.High) + (a.Low - quotient * b.Low);
    return Normalize(quotient, residual / b.High, a.Exponent - b.Exponent);
}

int Compare(Number a, Number b)
{
    if (a.High == 0.0f)
        return b.High > 0.0f ? -1 : (b.High < 0.0f ? 1 : 0);
    if (b.High == 0.0f)
        return a.High > 0.0f ? 1 : -1;
    if ((a.High < 0.0f) != (b.High < 0.0f))
        return a.High < 0.0f ? -1 : 1;
    if (a.Exponent != b.Exponent)
        return (a.Exponent < b.Exponent) == (a.High > 0.0f) ? -1 : 1;
    if (a.High != b.High)
        return a.High < b.High ? -1 : 1;
    return a.Low < b.Low ? -1 : (a.Low > b.Low ? 1 : 0);
}
Number Minimum(Number a, Number b)
{
    return Compare(a, b) <= 0 ? a : b;
}
Number Maximum(Number a, Number b)
{
    return Compare(a, b) >= 0 ? a : b;
}

uint Encode(Number value)
{
    value = Normalize(value.High, value.Low, value.Exponent);
    const uint sign = as_type<uint>(value.High) & SignMask;
    if (value.High == 0.0f)
        return 0u;
    if (value.Exponent > MaximumNormalExponent)
        return sign | InfinityBits;
    if (value.Exponent >= MinimumNormalExponent)
        return sign | (uint(value.Exponent + HalfExponentBias) << FractionBits) |
               (as_type<uint>(value.High) & FractionMask);
    if (value.Exponent < SubnormalUnitExponent)
        return sign;
    value = Absolute(value);
    const int shift = value.Exponent - SubnormalUnitExponent;
    const float units = ldexp(value.High, shift);
    const float integral = floor(units);
    uint fraction = uint(integral);
    const float midpointDistance = (units - integral - 0.5f) + ldexp(value.Low, shift);
    if (midpointDistance > 0.0f || (midpointDistance == 0.0f && (fraction & 1u)))
        ++fraction;
    // Rounding up to 1<<23 naturally produces the smallest normal encoding.
    return sign | fraction;
}
#if defined(__METAL_VERSION__)
#pragma STDC FP_CONTRACT ON
#endif
} // namespace WideMath
