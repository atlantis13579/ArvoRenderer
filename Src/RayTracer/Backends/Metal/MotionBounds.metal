#include "MotionData.h"
bool MotionBoundsValid(MotionBounds b)
{
    float3 low = float3(b.Minimum), high = float3(b.Maximum);
    return b.Valid != 0 && all(isfinite(low)) && all(isfinite(high)) && all(low <= high);
}
float MotionPeakCoordinate(float3 v)
{
    return max(abs(v.x), max(abs(v.y), abs(v.z)));
}
MotionBounds MotionExpandBounds(float3 low, float3 high, float magnitude)
{
    constexpr float ArithmeticSteps = 64;
    const float Gamma = ArithmeticSteps * MotionFloatEpsilon / (1 - ArithmeticSteps * MotionFloatEpsilon);
    float pad = max(magnitude * Gamma, ArithmeticSteps * MotionMinimumNormal);
    MotionBounds result = {packed_float3(low - pad), 1, packed_float3(high + pad), 0};
    result.Valid = MotionBoundsValid(result) ? 1 : 0;
    return result;
}
MotionBounds LinearMotionBounds(MotionBounds start, MotionBounds end, MotionShutter shutter)
{
    if (!MotionBoundsValid(start) || !MotionBoundsValid(end) || !ValidShutter(shutter))
        return {{}, 0, {}, 0};
    float3 low = min(InterpolateMotionPoint(float3(start.Minimum), float3(end.Minimum), shutter.Open),
                     InterpolateMotionPoint(float3(start.Minimum), float3(end.Minimum), shutter.Close));
    float3 high = max(InterpolateMotionPoint(float3(start.Maximum), float3(end.Maximum), shutter.Open),
                      InterpolateMotionPoint(float3(start.Maximum), float3(end.Maximum), shutter.Close));
    float scale =
        max(max(MotionPeakCoordinate(float3(start.Minimum)), MotionPeakCoordinate(float3(start.Maximum))),
            max(MotionPeakCoordinate(float3(end.Minimum)), MotionPeakCoordinate(float3(end.Maximum))));
    return MotionExpandBounds(low, high, scale);
}
MotionBounds RigidMotionBounds(MotionBounds box, MotionTrack track, MotionShutter shutter)
{
    // Endpoint boxes do not bound a rotating object. The L1 radius is a loose
    // enclosing sphere, translated over the shutter and padded for roundoff.
    if (!MotionBoundsValid(box) || !ValidMotionTrack(track) || !ValidShutter(shutter))
        return {{}, 0, {}, 0};
    float3 extent = max(abs(float3(box.Minimum)), abs(float3(box.Maximum)));
    float radius = extent.x + extent.y + extent.z;
    float3 start = float3(track.Start.Translation), end = float3(track.End.Translation);
    float3 first = InterpolateMotionPoint(start, end, shutter.Open),
           last = InterpolateMotionPoint(start, end, shutter.Close);
    return MotionExpandBounds(min(first, last) - radius, max(first, last) + radius,
                              radius + max(MotionPeakCoordinate(start), MotionPeakCoordinate(end)));
}
