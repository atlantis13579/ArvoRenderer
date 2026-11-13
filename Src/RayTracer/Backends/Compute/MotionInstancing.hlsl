#include "MotionData.hlsli"
// Ray coordinates stay wide until the existing BLAS conditioning step. Rigid
// motion changes neither the affine ray parameter nor its open interval.
double3 InverseMotionWideVector(MotionPose pose, double3 value)
{
    const double3 q = -double3(pose.Rotation.X, pose.Rotation.Y, pose.Rotation.Z);
    const double3 tangent = 2 * (q.yzx * value.zxy - q.zxy * value.yzx);
    return value + double(pose.Rotation.W) * tangent + q.yzx * tangent.zxy - q.zxy * tangent.yzx;
}
void InverseInstanceMotion(MotionPose pose, inout double3 origin, inout double3 direction)
{
    origin = InverseMotionWideVector(pose, origin - double3(pose.Translation));
    direction = InverseMotionWideVector(pose, direction);
}
