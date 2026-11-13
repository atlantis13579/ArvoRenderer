#ifndef ARVO_CUDA_MOTION_INTERPOLATION
#define ARVO_CUDA_MOTION_INTERPOLATION
#include "MotionData.cuh"
__device__ float MotionPeak(MotionQuaternion q)
{
    return fmaxf(fmaxf(fabsf(q.X), fabsf(q.Y)), fmaxf(fabsf(q.Z), fabsf(q.W)));
}
__device__ MotionQuaternion MotionUnit(MotionQuaternion q)
{
    const float scale = MotionPeak(q);
    q = {q.X / scale, q.Y / scale, q.Z / scale, q.W / scale};
    const float inverse = 1 / sqrtf(q.X * q.X + q.Y * q.Y + q.Z * q.Z + q.W * q.W);
    return {q.X * inverse, q.Y * inverse, q.Z * inverse, q.W * inverse};
}
__device__ bool MotionValidPose(MotionPose p)
{
    return isfinite(p.Rotation.X) && isfinite(p.Rotation.Y) && isfinite(p.Rotation.Z) &&
           isfinite(p.Rotation.W) && MotionPeak(p.Rotation) >= FLT_MIN && isfinite(p.Translation.x) &&
           isfinite(p.Translation.y) && isfinite(p.Translation.z);
}
__device__ bool ValidMotionTrack(MotionTrack t)
{
    return MotionValidPose(t.Start) && MotionValidPose(t.End);
}
__device__ float3 InterpolateMotionPoint(float3 a, float3 b, float time)
{
    if (time == 0)
        return a;
    if (time == 1)
        return b;
    return make_float3(a.x * (1 - time) + b.x * time, a.y * (1 - time) + b.y * time,
                       a.z * (1 - time) + b.z * time);
}
__device__ MotionPose InterpolateMotion(MotionTrack track, float time)
{
    // Quaternion SLERP follows the shorter arc. The scaled normalization keeps
    // finite large/small input quaternions away from squared-length overflow.
    MotionQuaternion a = MotionUnit(track.Start.Rotation), b = MotionUnit(track.End.Rotation);
    float cosine = a.X * b.X + a.Y * b.Y + a.Z * b.Z + a.W * b.W;
    if (cosine < 0)
    {
        b = {-b.X, -b.Y, -b.Z, -b.W};
        cosine = -cosine;
    }
    cosine = fminf(cosine, 1.0f);
    constexpr float ParallelThreshold = 1 - 32 * FLT_EPSILON;
    float wa = 1 - time, wb = time;
    if (cosine < ParallelThreshold)
    {
        const float theta = acosf(cosine), denominator = sinf(theta);
        wa = sinf((1 - time) * theta) / denominator;
        wb = sinf(time * theta) / denominator;
    }
    const MotionQuaternion q =
        MotionUnit({wa * a.X + wb * b.X, wa * a.Y + wb * b.Y, wa * a.Z + wb * b.Z, wa * a.W + wb * b.W});
    return {q, InterpolateMotionPoint(track.Start.Translation, track.End.Translation, time), 0};
}
__device__ float3 TransformMotionVector(MotionPose pose, float3 value)
{
    const auto q = pose.Rotation;
    const float x = 2 * (q.Y * value.z - q.Z * value.y), y = 2 * (q.Z * value.x - q.X * value.z),
                z = 2 * (q.X * value.y - q.Y * value.x);
    return make_float3(value.x + q.W * x + q.Y * z - q.Z * y, value.y + q.W * y + q.Z * x - q.X * z,
                       value.z + q.W * z + q.X * y - q.Y * x);
}
__device__ float3 TransformMotionPoint(MotionPose pose, float3 value)
{
    const float3 rotated = TransformMotionVector(pose, value);
    return make_float3(rotated.x + pose.Translation.x, rotated.y + pose.Translation.y,
                       rotated.z + pose.Translation.z);
}
__device__ float3 InverseMotionVector(MotionPose pose, float3 value)
{
    pose.Rotation.X = -pose.Rotation.X;
    pose.Rotation.Y = -pose.Rotation.Y;
    pose.Rotation.Z = -pose.Rotation.Z;
    // Keep direction length: normalizing here would change the ray parameter.
    return TransformMotionVector(pose, value);
}
__device__ float3 InverseMotionPoint(MotionPose pose, float3 value)
{
    return InverseMotionVector(pose, make_float3(value.x - pose.Translation.x, value.y - pose.Translation.y,
                                                 value.z - pose.Translation.z));
}

#endif
