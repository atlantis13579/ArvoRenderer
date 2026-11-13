#ifndef ARVO_DXR_MOTION_INTERPOLATION
#define ARVO_DXR_MOTION_INTERPOLATION
#include "MotionData.hlsli"
float MotionPeak(MotionQuaternion value)
{
    return max(max(abs(value.X), abs(value.Y)), max(abs(value.Z), abs(value.W)));
}
MotionQuaternion MotionNormalized(MotionQuaternion value)
{
    float peak = MotionPeak(value);
    value.X /= peak;
    value.Y /= peak;
    value.Z /= peak;
    value.W /= peak;
    float inverseLength =
        1 / sqrt(value.X * value.X + value.Y * value.Y + value.Z * value.Z + value.W * value.W);
    value.X *= inverseLength;
    value.Y *= inverseLength;
    value.Z *= inverseLength;
    value.W *= inverseLength;
    return value;
}
bool MotionPoseValid(MotionPose pose)
{
    MotionQuaternion q = pose.Rotation;
    return isfinite(q.X) && isfinite(q.Y) && isfinite(q.Z) && isfinite(q.W) &&
           MotionPeak(q) >= MotionMinimumNormal && all(isfinite(pose.Translation));
}
bool ValidMotionTrack(MotionTrack track)
{
    return MotionPoseValid(track.Start) && MotionPoseValid(track.End);
}
float3 InterpolateMotionPoint(float3 a, float3 b, float time)
{
    if (time == 0)
        return a;
    if (time == 1)
        return b;
    return (1 - time) * a + time * b;
}
MotionPose InterpolateMotion(MotionTrack track, float time)
{
    // Shoemake (1985), shortest-arc quaternion interpolation. A rigid frame
    // remains invertible even for a half-turn; interpolating matrices does not.
    MotionQuaternion from = MotionNormalized(track.Start.Rotation), to = MotionNormalized(track.End.Rotation);
    float cosine = from.X * to.X + from.Y * to.Y + from.Z * to.Z + from.W * to.W;
    if (cosine < 0)
    {
        to.X = -to.X;
        to.Y = -to.Y;
        to.Z = -to.Z;
        to.W = -to.W;
        cosine = -cosine;
    }
    cosine = min(1.0f, cosine);
    static const float ParallelTolerance = 32 * MotionFloatEpsilon;
    float fromWeight = 1 - time, toWeight = time;
    if (1 - cosine > ParallelTolerance)
    {
        float angle = acos(cosine), inverseSine = 1 / sin(angle);
        fromWeight = sin((1 - time) * angle) * inverseSine;
        toWeight = sin(time * angle) * inverseSine;
    }
    MotionQuaternion q;
    q.X = fromWeight * from.X + toWeight * to.X;
    q.Y = fromWeight * from.Y + toWeight * to.Y;
    q.Z = fromWeight * from.Z + toWeight * to.Z;
    q.W = fromWeight * from.W + toWeight * to.W;
    MotionPose pose;
    pose.Rotation = MotionNormalized(q);
    pose.Translation = InterpolateMotionPoint(track.Start.Translation, track.End.Translation, time);
    pose.Reserved = 0;
    return pose;
}
float3 TransformMotionVector(MotionPose pose, float3 value)
{
    float3 axis = float3(pose.Rotation.X, pose.Rotation.Y, pose.Rotation.Z);
    float3 crossTerm = 2 * cross(axis, value);
    return value + crossTerm * pose.Rotation.W + cross(axis, crossTerm);
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
    return TransformMotionVector(pose, value);
}
float3 InverseMotionPoint(MotionPose pose, float3 value)
{
    return InverseMotionVector(pose, value - pose.Translation);
}

#endif
