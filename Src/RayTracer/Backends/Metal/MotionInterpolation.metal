#ifndef ARVO_METAL_MOTION_INTERPOLATION
#define ARVO_METAL_MOTION_INTERPOLATION
#include "MotionData.h"
float MotionRotationScale(MotionQuaternion q)
{
    return max(max(abs(q.X), abs(q.Y)), max(abs(q.Z), abs(q.W)));
}
MotionQuaternion MotionUnitRotation(MotionQuaternion q)
{
    float scale = MotionRotationScale(q);
    q.X /= scale;
    q.Y /= scale;
    q.Z /= scale;
    q.W /= scale;
    float norm = sqrt(q.X * q.X + q.Y * q.Y + q.Z * q.Z + q.W * q.W);
    return {q.X / norm, q.Y / norm, q.Z / norm, q.W / norm};
}
bool MotionPoseValid(MotionPose pose)
{
    MotionQuaternion q = pose.Rotation;
    return isfinite(q.X) && isfinite(q.Y) && isfinite(q.Z) && isfinite(q.W) &&
           MotionRotationScale(q) >= MotionMinimumNormal && all(isfinite(float3(pose.Translation)));
}
bool ValidMotionTrack(MotionTrack track)
{
    return MotionPoseValid(track.Start) && MotionPoseValid(track.End);
}
float3 InterpolateMotionPoint(float3 start, float3 end, float time)
{
    if (time == 0)
        return start;
    if (time == 1)
        return end;
    return start * (1 - time) + end * time;
}
MotionPose InterpolateMotion(MotionTrack track, float time)
{
    // Shoemake, Animating Rotation with Quaternion Curves (1985). Normalized
    // endpoint quaternions select the shortest arc; q and -q denote one pose.
    MotionQuaternion a = MotionUnitRotation(track.Start.Rotation), b = MotionUnitRotation(track.End.Rotation);
    float cosine = a.X * b.X + a.Y * b.Y + a.Z * b.Z + a.W * b.W;
    if (cosine < 0)
    {
        b = {-b.X, -b.Y, -b.Z, -b.W};
        cosine = -cosine;
    }
    cosine = min(1.0f, cosine);
    const float ParallelThreshold = 1 - 32 * MotionFloatEpsilon;
    float first = 1 - time, second = time;
    if (cosine < ParallelThreshold)
    {
        float angle = acos(cosine), sine = sin(angle);
        first = sin((1 - time) * angle) / sine;
        second = sin(time * angle) / sine;
    }
    MotionQuaternion q = {first * a.X + second * b.X, first * a.Y + second * b.Y, first * a.Z + second * b.Z,
                          first * a.W + second * b.W};
    MotionPose pose;
    pose.Rotation = MotionUnitRotation(q);
    pose.Translation = packed_float3(
        InterpolateMotionPoint(float3(track.Start.Translation), float3(track.End.Translation), time));
    pose.Reserved = 0;
    return pose;
}
float3 TransformMotionVector(MotionPose pose, float3 value)
{
    float3 axis = float3(pose.Rotation.X, pose.Rotation.Y, pose.Rotation.Z);
    float3 turn = cross(axis, value) * 2;
    return value + turn * pose.Rotation.W + cross(axis, turn);
}
float3 TransformMotionPoint(MotionPose pose, float3 value)
{
    return TransformMotionVector(pose, value) + float3(pose.Translation);
}
float3 InverseMotionVector(MotionPose pose, float3 value)
{
    pose.Rotation.X = -pose.Rotation.X;
    pose.Rotation.Y = -pose.Rotation.Y;
    pose.Rotation.Z = -pose.Rotation.Z;
    return TransformMotionVector(pose, value);
}
float3 InverseMotionPoint(MotionPose pose, float3 value)
{
    return InverseMotionVector(pose, value - float3(pose.Translation));
}

#endif
