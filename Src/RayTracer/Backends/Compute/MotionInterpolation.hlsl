#ifndef ARVO_COMPUTE_MOTION_INTERPOLATION
#define ARVO_COMPUTE_MOTION_INTERPOLATION
#include "MotionData.hlsli"
float MotionQuaternionPeak(MotionQuaternion q)
{
    return max(max(abs(q.X), abs(q.Y)), max(abs(q.Z), abs(q.W)));
}
MotionQuaternion MotionUnitQuaternion(MotionQuaternion q)
{
    float scale = MotionQuaternionPeak(q);
    q.X /= scale;
    q.Y /= scale;
    q.Z /= scale;
    q.W /= scale;
    float length = sqrt(q.X * q.X + q.Y * q.Y + q.Z * q.Z + q.W * q.W);
    q.X /= length;
    q.Y /= length;
    q.Z /= length;
    q.W /= length;
    return q;
}
bool ValidMotionPose(MotionPose p)
{
    return isfinite(p.Rotation.X) && isfinite(p.Rotation.Y) && isfinite(p.Rotation.Z) &&
           isfinite(p.Rotation.W) && MotionQuaternionPeak(p.Rotation) >= MotionMinimumNormal &&
           all(isfinite(p.Translation));
}
bool ValidMotionTrack(MotionTrack track)
{
    return ValidMotionPose(track.Start) && ValidMotionPose(track.End);
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
    // Shortest-arc SLERP: Shoemake (1985). The validated quaternion endpoints
    // cannot produce the singular frames of componentwise matrix interpolation.
    MotionQuaternion a = MotionUnitQuaternion(track.Start.Rotation);
    MotionQuaternion b = MotionUnitQuaternion(track.End.Rotation);
    float cosine = a.X * b.X + a.Y * b.Y + a.Z * b.Z + a.W * b.W;
    if (cosine < 0)
    {
        b.X = -b.X;
        b.Y = -b.Y;
        b.Z = -b.Z;
        b.W = -b.W;
        cosine = -cosine;
    }
    cosine = min(cosine, 1.0f);
    static const float ParallelThreshold = 1 - 32 * MotionFloatEpsilon;
    float first = 1 - time, second = time;
    if (cosine < ParallelThreshold)
    {
        float angle = acos(cosine), denominator = sin(angle);
        first = sin((1 - time) * angle) / denominator;
        second = sin(time * angle) / denominator;
    }
    MotionQuaternion blended;
    blended.X = first * a.X + second * b.X;
    blended.Y = first * a.Y + second * b.Y;
    blended.Z = first * a.Z + second * b.Z;
    blended.W = first * a.W + second * b.W;
    MotionPose pose;
    pose.Rotation = MotionUnitQuaternion(blended);
    pose.Translation = InterpolateMotionPoint(track.Start.Translation, track.End.Translation, time);
    pose.Reserved = 0;
    return pose;
}
float3 TransformMotionVector(MotionPose pose, float3 value)
{
    float3 axis = float3(pose.Rotation.X, pose.Rotation.Y, pose.Rotation.Z);
    float3 turn = 2 * cross(axis, value);
    return value + pose.Rotation.W * turn + cross(axis, turn);
}
float3 TransformMotionPoint(MotionPose pose, float3 value)
{
    return TransformMotionVector(pose, value) + pose.Translation;
}
float3 InverseMotionVector(MotionPose pose, float3 value)
{
    pose.Rotation.X = -pose.Rotation.X;
    pose.Rotation.Y = -pose.Rotation.Y;
    pose.Rotation.Z = -pose.Rotation.Z;
    // A direction is transformed without normalization, retaining ray t.
    return TransformMotionVector(pose, value);
}
float3 InverseMotionPoint(MotionPose pose, float3 value)
{
    return InverseMotionVector(pose, value - pose.Translation);
}

#endif
