#include "MotionData.hlsli"
bool MotionBoxValid(MotionBounds box)
{
    return box.Valid != 0 && all(isfinite(box.Minimum)) && all(isfinite(box.Maximum)) &&
           all(box.Minimum <= box.Maximum);
}
float MotionMagnitude(float3 value)
{
    return max(abs(value.x), max(abs(value.y), abs(value.z)));
}
MotionBounds MotionBox(float3 low, float3 high, uint valid)
{
    MotionBounds result;
    result.Minimum = low;
    result.Maximum = high;
    result.Valid = valid;
    result.Reserved = 0;
    return result;
}
MotionBounds InflateMotionBox(float3 low, float3 high, float magnitude)
{
    static const float ArithmeticSteps = 64;
    static const float Gamma =
        ArithmeticSteps * MotionFloatEpsilon / (1 - ArithmeticSteps * MotionFloatEpsilon);
    float pad = max(magnitude * Gamma, ArithmeticSteps * MotionMinimumNormal);
    MotionBounds result = MotionBox(low - pad, high + pad, 1);
    result.Valid = MotionBoxValid(result) ? 1 : 0;
    return result;
}
MotionBounds LinearMotionBounds(MotionBounds start, MotionBounds end, MotionShutter shutter)
{
    if (!MotionBoxValid(start) || !MotionBoxValid(end) || !ValidShutter(shutter))
        return MotionBox(float3(0, 0, 0), float3(0, 0, 0), 0);
    float3 low = min(InterpolateMotionPoint(start.Minimum, end.Minimum, shutter.Open),
                     InterpolateMotionPoint(start.Minimum, end.Minimum, shutter.Close));
    float3 high = max(InterpolateMotionPoint(start.Maximum, end.Maximum, shutter.Open),
                      InterpolateMotionPoint(start.Maximum, end.Maximum, shutter.Close));
    float scale = max(max(MotionMagnitude(start.Minimum), MotionMagnitude(start.Maximum)),
                      max(MotionMagnitude(end.Minimum), MotionMagnitude(end.Maximum)));
    return InflateMotionBox(low, high, scale);
}
MotionBounds RigidMotionBounds(MotionBounds box, MotionTrack track, MotionShutter shutter)
{
    // A rotating box can leave both endpoint boxes. Enclose every orientation
    // by its L1 radius about the rigid origin, swept along the translation.
    if (!MotionBoxValid(box) || !ValidMotionTrack(track) || !ValidShutter(shutter))
        return MotionBox(float3(0, 0, 0), float3(0, 0, 0), 0);
    float3 extent = max(abs(box.Minimum), abs(box.Maximum));
    float radius = extent.x + extent.y + extent.z;
    float3 a = InterpolateMotionPoint(track.Start.Translation, track.End.Translation, shutter.Open);
    float3 b = InterpolateMotionPoint(track.Start.Translation, track.End.Translation, shutter.Close);
    float scale =
        radius + max(MotionMagnitude(track.Start.Translation), MotionMagnitude(track.End.Translation));
    return InflateMotionBox(min(a, b) - radius, max(a, b) + radius, scale);
}
