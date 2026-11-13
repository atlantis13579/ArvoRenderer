#include "MotionData.hlsli"
MotionBounds MotionMakeBounds(float3 low, float3 high, uint valid)
{
    MotionBounds box;
    box.Minimum = low;
    box.Maximum = high;
    box.Valid = valid;
    box.Reserved = 0;
    return box;
}
bool MotionBoundsValid(MotionBounds box)
{
    return box.Valid != 0 && all(isfinite(box.Minimum)) && all(isfinite(box.Maximum)) &&
           all(box.Minimum <= box.Maximum);
}
float MotionMaxComponent(float3 value)
{
    return max(abs(value.x), max(abs(value.y), abs(value.z)));
}
MotionBounds MotionPadBounds(float3 low, float3 high, float magnitude)
{
    static const float RoundoffOperations = 64;
    static const float Gamma =
        RoundoffOperations * MotionFloatEpsilon / (1 - RoundoffOperations * MotionFloatEpsilon);
    float margin = max(Gamma * magnitude, RoundoffOperations * MotionMinimumNormal);
    MotionBounds result = MotionMakeBounds(low - margin, high + margin, 1);
    result.Valid = MotionBoundsValid(result) ? 1 : 0;
    return result;
}
MotionBounds LinearMotionBounds(MotionBounds start, MotionBounds end, MotionShutter shutter)
{
    if (!MotionBoundsValid(start) || !MotionBoundsValid(end) || !ValidShutter(shutter))
        return MotionMakeBounds(float3(0, 0, 0), float3(0, 0, 0), 0);
    float3 lowStart = InterpolateMotionPoint(start.Minimum, end.Minimum, shutter.Open);
    float3 lowEnd = InterpolateMotionPoint(start.Minimum, end.Minimum, shutter.Close);
    float3 highStart = InterpolateMotionPoint(start.Maximum, end.Maximum, shutter.Open);
    float3 highEnd = InterpolateMotionPoint(start.Maximum, end.Maximum, shutter.Close);
    float magnitude = max(max(MotionMaxComponent(start.Minimum), MotionMaxComponent(start.Maximum)),
                          max(MotionMaxComponent(end.Minimum), MotionMaxComponent(end.Maximum)));
    return MotionPadBounds(min(lowStart, lowEnd), max(highStart, highEnd), magnitude);
}
MotionBounds RigidMotionBounds(MotionBounds local, MotionTrack track, MotionShutter shutter)
{
    // The L1 enclosing sphere covers intermediate rotations missed by endpoint
    // AABBs. This envelope is conservative, deliberately trading traversal cost
    // for a simple bound; it is not a tight rotational-extrema solver.
    if (!MotionBoundsValid(local) || !ValidMotionTrack(track) || !ValidShutter(shutter))
        return MotionMakeBounds(float3(0, 0, 0), float3(0, 0, 0), 0);
    float3 extent = max(abs(local.Minimum), abs(local.Maximum));
    float radius = extent.x + extent.y + extent.z;
    float3 begin = InterpolateMotionPoint(track.Start.Translation, track.End.Translation, shutter.Open);
    float3 finish = InterpolateMotionPoint(track.Start.Translation, track.End.Translation, shutter.Close);
    float magnitude =
        radius + max(MotionMaxComponent(track.Start.Translation), MotionMaxComponent(track.End.Translation));
    return MotionPadBounds(min(begin, finish) - radius, max(begin, finish) + radius, magnitude);
}
