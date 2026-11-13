#include "MotionData.cuh"
__device__ bool MotionBoxValid(MotionBounds b)
{
    return b.Valid && isfinite(b.Minimum.x) && isfinite(b.Minimum.y) && isfinite(b.Minimum.z) &&
           isfinite(b.Maximum.x) && isfinite(b.Maximum.y) && isfinite(b.Maximum.z) &&
           b.Minimum.x <= b.Maximum.x && b.Minimum.y <= b.Maximum.y && b.Minimum.z <= b.Maximum.z;
}
__device__ float MotionMagnitude(float3 v)
{
    return fmaxf(fabsf(v.x), fmaxf(fabsf(v.y), fabsf(v.z)));
}
__device__ float3 MotionMin(float3 a, float3 b)
{
    return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}
__device__ float3 MotionMax(float3 a, float3 b)
{
    return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}
__device__ MotionBounds MotionPadded(float3 low, float3 high, float magnitude)
{
    constexpr float ArithmeticSteps = 64,
                    Gamma = ArithmeticSteps * FLT_EPSILON / (1 - ArithmeticSteps * FLT_EPSILON);
    const float pad = fmaxf(magnitude * Gamma, ArithmeticSteps * FLT_MIN);
    MotionBounds result{make_float3(low.x - pad, low.y - pad, low.z - pad), 1,
                        make_float3(high.x + pad, high.y + pad, high.z + pad), 0};
    result.Valid = MotionBoxValid(result);
    return result;
}
__device__ MotionBounds LinearMotionBounds(MotionBounds a, MotionBounds b, MotionShutter shutter)
{
    if (!MotionBoxValid(a) || !MotionBoxValid(b) || !ValidShutter(shutter))
        return {{}, 0, {}, 0};
    const float3 low = MotionMin(InterpolateMotionPoint(a.Minimum, b.Minimum, shutter.Open),
                                 InterpolateMotionPoint(a.Minimum, b.Minimum, shutter.Close));
    const float3 high = MotionMax(InterpolateMotionPoint(a.Maximum, b.Maximum, shutter.Open),
                                  InterpolateMotionPoint(a.Maximum, b.Maximum, shutter.Close));
    return MotionPadded(low, high,
                        fmaxf(fmaxf(MotionMagnitude(a.Minimum), MotionMagnitude(a.Maximum)),
                              fmaxf(MotionMagnitude(b.Minimum), MotionMagnitude(b.Maximum))));
}
__device__ MotionBounds RigidMotionBounds(MotionBounds box, MotionTrack track, MotionShutter shutter)
{
    // A rotating point stays within its L1 radius about the rigid origin.
    // Sweep this envelope along translation; endpoint box union alone is unsafe.
    if (!MotionBoxValid(box) || !ValidMotionTrack(track) || !ValidShutter(shutter))
        return {{}, 0, {}, 0};
    const float radius = fmaxf(fabsf(box.Minimum.x), fabsf(box.Maximum.x)) +
                         fmaxf(fabsf(box.Minimum.y), fabsf(box.Maximum.y)) +
                         fmaxf(fabsf(box.Minimum.z), fabsf(box.Maximum.z));
    const float3 begin = InterpolateMotionPoint(track.Start.Translation, track.End.Translation, shutter.Open);
    const float3 end = InterpolateMotionPoint(track.Start.Translation, track.End.Translation, shutter.Close);
    const float3 low = MotionMin(begin, end), high = MotionMax(begin, end);
    return MotionPadded(
        make_float3(low.x - radius, low.y - radius, low.z - radius),
        make_float3(high.x + radius, high.y + radius, high.z + radius),
        radius + fmaxf(MotionMagnitude(track.Start.Translation), MotionMagnitude(track.End.Translation)));
}
