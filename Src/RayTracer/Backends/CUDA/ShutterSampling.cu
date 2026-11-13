#include "MotionData.cuh"
__device__ bool MotionTimeRepresentable(float value)
{
    constexpr unsigned MagnitudeMask = 0x7fffffffU, MinimumNormalBits = 0x00800000U;
    const unsigned magnitude = __float_as_uint(value) & MagnitudeMask;
    return magnitude == 0 || magnitude >= MinimumNormalBits;
}
__device__ bool ValidShutter(MotionShutter shutter)
{
    if (!isfinite(shutter.Open) || !isfinite(shutter.Close) || shutter.Open < 0 || shutter.Close > 1 ||
        shutter.Open > shutter.Close)
        return false;
    if (!MotionTimeRepresentable(shutter.Open) || !MotionTimeRepresentable(shutter.Close))
        return false;
    const float width = shutter.Close - shutter.Open;
    return shutter.Open == shutter.Close || (width >= FLT_MIN && isfinite(1.0f / width));
}
__device__ MotionTimeSample SampleShutter(MotionShutter shutter, float uniform)
{
    // A single time describes the entire light path (Cook et al., 1984).
    // Box shutter response is normalized: response / density equals one.
    if (!ValidShutter(shutter))
        return {0, 0, MotionInvalidSample, 0};
    if (shutter.Open == shutter.Close)
        return {shutter.Open, 1, MotionDeltaTime, 0};
    if (!isfinite(uniform) || uniform < 0 || uniform >= 1)
        return {0, 0, MotionInvalidSample, 0};
    return {shutter.Open * (1 - uniform) + shutter.Close * uniform, 1 / (shutter.Close - shutter.Open), 0, 0};
}
