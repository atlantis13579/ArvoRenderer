#include "MotionData.hlsli"
bool MotionTimeRepresentable(float value)
{
    static const uint MinimumNormalBits = 0x00800000, MagnitudeMask = 0x7fffffff;
    uint magnitude = asuint(value) & MagnitudeMask;
    return magnitude == 0 || magnitude >= MinimumNormalBits;
}
bool ValidShutter(MotionShutter shutter)
{
    if (!isfinite(shutter.Open) || !isfinite(shutter.Close) || shutter.Open < 0 || shutter.Close > 1 ||
        shutter.Close < shutter.Open)
        return false;
    if (!MotionTimeRepresentable(shutter.Open) || !MotionTimeRepresentable(shutter.Close))
        return false;
    float width = shutter.Close - shutter.Open;
    return shutter.Open == shutter.Close || (width >= MotionMinimumNormal && isfinite(1 / width));
}
MotionTimeSample SampleShutter(MotionShutter shutter, float variate)
{
    // Cook et al., Distributed Ray Tracing (1984). The normalized box response
    // cancels its time PDF; changing exposure duration does not brighten film.
    MotionTimeSample sample;
    sample.Time = 0;
    sample.Density = 0;
    sample.Flags = MotionInvalidSample;
    sample.Reserved = 0;
    if (!ValidShutter(shutter))
        return sample;
    if (shutter.Open == shutter.Close)
    {
        sample.Time = shutter.Open;
        sample.Density = 1;
        sample.Flags = MotionDeltaTime;
        return sample;
    }
    if (!(variate >= 0 && variate < 1) || !isfinite(variate))
        return sample;
    sample.Time = (1 - variate) * shutter.Open + variate * shutter.Close;
    sample.Density = 1 / (shutter.Close - shutter.Open);
    sample.Flags = 0;
    return sample;
}
