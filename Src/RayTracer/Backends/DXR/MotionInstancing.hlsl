#include "MotionData.hlsli"
// The hardware instance supplies the fixed affine map; procedural motion
// intersects the rest primitive in its time-dependent local rigid frame.
double3 InverseMotionCoordinates(MotionPose pose, double3 coordinates)
{
    double x = -double(pose.Rotation.X), y = -double(pose.Rotation.Y), z = -double(pose.Rotation.Z);
    double3 tangent = double3(y * coordinates.z - z * coordinates.y,
        z * coordinates.x - x * coordinates.z, x * coordinates.y - y * coordinates.x) * 2;
    return coordinates + double(pose.Rotation.W) * tangent +
        double3(y * tangent.z - z * tangent.y, z * tangent.x - x * tangent.z, x * tangent.y - y * tangent.x);
}
void InverseInstanceMotion(MotionPose pose, inout double3 origin, inout double3 direction)
{
    origin -= double3(pose.Translation);
    origin = InverseMotionCoordinates(pose, origin);
    direction = InverseMotionCoordinates(pose, direction);
}
