#include "MotionData.h"
bool MotionTimeRepresentable(float value)
{
    constexpr uint MagnitudeMask = 0x7fffffff, MinimumNormalBits = 0x00800000;
    uint magnitude = as_type<uint>(value) & MagnitudeMask;
    return magnitude == 0 || magnitude >= MinimumNormalBits;
}
bool ValidShutter(MotionShutter shutter)
{
    if (!isfinite(shutter.Open) || !isfinite(shutter.Close) || shutter.Open < 0 || shutter.Close > 1 ||
        shutter.Close < shutter.Open)
        return false;
    if (!MotionTimeRepresentable(shutter.Open) || !MotionTimeRepresentable(shutter.Close))
        return false;
    float duration = shutter.Close - shutter.Open;
    return shutter.Open == shutter.Close || (duration >= MotionMinimumNormal && isfinite(1 / duration));
}
MotionTimeSample SampleShutter(MotionShutter shutter, float variate)
{
    // Normalized exposure (Cook et al., 1984): density and shutter response
    // cancel. The delta case has unit discrete mass and consumes no variate.
    if (!ValidShutter(shutter))
        return {0, 0, MotionInvalidSample, 0};
    if (shutter.Open == shutter.Close)
        return {shutter.Open, 1, MotionDeltaTime, 0};
    if (!isfinite(variate) || variate < 0 || variate >= 1)
        return {0, 0, MotionInvalidSample, 0};
    float time = (1 - variate) * shutter.Open + variate * shutter.Close;
    return {time, 1 / (shutter.Close - shutter.Open), 0, 0};
}
