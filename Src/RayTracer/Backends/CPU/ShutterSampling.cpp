#include "RayTracer/Backends/CPU/Motion.h"
#include <cmath>
#include <limits>
#include <cstring>

namespace Arvo::RayTracing::Cpu
{
namespace
{
bool NormalOrZeroTime(float value)
{
    constexpr uint32_t MagnitudeMask = 0x7fffffffU, MinimumNormalBits = 0x00800000U;
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    bits &= MagnitudeMask;
    return bits == 0 || bits >= MinimumNormalBits;
}
} // namespace
// Cook, Porter and Carpenter, "Distributed Ray Tracing" (1984). Time is a
// camera-sample dimension, held fixed through all connections of that path.
// A normalized box shutter has response 1/width and PDF 1/width, hence unit
// throughput weight. Exposure duration is not an additional brightness gain.
bool ValidShutter(MotionData::Shutter shutter)
{
    if (!std::isfinite(shutter.Open) || !std::isfinite(shutter.Close) || shutter.Open < 0 ||
        shutter.Close > 1 || shutter.Open > shutter.Close)
        return false;
    constexpr float MinimumNormal = std::numeric_limits<float>::min();
    if (!NormalOrZeroTime(shutter.Open) || !NormalOrZeroTime(shutter.Close))
        return false;
    const float width = shutter.Close - shutter.Open;
    return shutter.Open == shutter.Close || (width >= MinimumNormal && std::isfinite(1 / width));
}
MotionData::TimeSample SampleShutter(MotionData::Shutter shutter, float variate)
{
    if (!ValidShutter(shutter))
        return {};
    if (shutter.Open == shutter.Close)
        return {shutter.Open, 1, MotionData::DeltaTime, 0};
    if (!std::isfinite(variate) || variate < 0 || variate >= 1)
        return {};
    return {shutter.Open * (1 - variate) + shutter.Close * variate, 1 / (shutter.Close - shutter.Open), 0, 0};
}
} // namespace Arvo::RayTracing::Cpu
