#include "MotionData.hlsli"
bool MotionTimeRepresentable(float value)
{
    static const uint MagnitudeMask = 0x7fffffff, MinimumNormalBits = 0x00800000;
    uint bits = asuint(value) & MagnitudeMask;
    return bits == 0 || bits >= MinimumNormalBits;
}
// Native compute box-shutter sampling, Cook et al., Distributed Ray Tracing
// (1984). Response/PDF is one for a normalized exposure. A closed shutter is
// a delta measure, not a box with an arbitrary minimum exposure width.
bool ValidShutter(MotionShutter shutter)
{
    if (!isfinite(shutter.Open) || !isfinite(shutter.Close) || shutter.Open < 0 || shutter.Close > 1 ||
        shutter.Open > shutter.Close)
        return false;
    if (!MotionTimeRepresentable(shutter.Open) || !MotionTimeRepresentable(shutter.Close))
        return false;
    float width = shutter.Close - shutter.Open;
    return shutter.Open == shutter.Close || (width >= MotionMinimumNormal && isfinite(1 / width));
}
MotionTimeSample SampleShutter(MotionShutter shutter, float variate)
{
    MotionTimeSample result;
    result.Time = result.Density = 0;
    result.Flags = MotionInvalidSample;
    result.Reserved = 0;
    if (!ValidShutter(shutter))
        return result;
    if (shutter.Open == shutter.Close)
    {
        result.Time = shutter.Open;
        result.Density = 1;
        result.Flags = MotionDeltaTime;
        return result;
    }
    if (!isfinite(variate) || variate < 0 || variate >= 1)
        return result;
    result.Time = shutter.Open * (1 - variate) + shutter.Close * variate;
    result.Density = 1 / (shutter.Close - shutter.Open);
    result.Flags = 0;
    return result;
}
