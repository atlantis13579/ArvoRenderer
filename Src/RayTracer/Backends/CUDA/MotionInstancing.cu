#include "MotionData.cuh"
// This conversion crosses only a data ABI: the interpolation and ray transform
// below are CUDA-owned and never call another backend's numerical code.
__device__ MotionTrack InstanceMotionTrack(const Arvo::RayTracing::MotionData::Track& source)
{
    MotionTrack result;
    result.Start.Rotation = {source.Start.Rotation.X, source.Start.Rotation.Y, source.Start.Rotation.Z, source.Start.Rotation.W};
    result.End.Rotation = {source.End.Rotation.X, source.End.Rotation.Y, source.End.Rotation.Z, source.End.Rotation.W};
    result.Start.Translation = make_float3(source.Start.Translation.X, source.Start.Translation.Y, source.Start.Translation.Z);
    result.End.Translation = make_float3(source.End.Translation.X, source.End.Translation.Y, source.End.Translation.Z);
    result.Start.Reserved = result.End.Reserved = 0;
    return result;
}
__device__ void InverseMotionCoordinates(MotionPose pose, double value[3])
{
    const double x = -double(pose.Rotation.X), y = -double(pose.Rotation.Y), z = -double(pose.Rotation.Z);
    const double tx = 2 * (y * value[2] - z * value[1]);
    const double ty = 2 * (z * value[0] - x * value[2]);
    const double tz = 2 * (x * value[1] - y * value[0]);
    value[0] += double(pose.Rotation.W) * tx + y * tz - z * ty;
    value[1] += double(pose.Rotation.W) * ty + z * tx - x * tz;
    value[2] += double(pose.Rotation.W) * tz + x * ty - y * tx;
}
// Evaluate the inverse local rigid map in double before the float BLAS handoff.
__device__ void InverseInstanceMotion(MotionPose pose, double origin[3], double direction[3])
{
    origin[0] -= pose.Translation.x;
    origin[1] -= pose.Translation.y;
    origin[2] -= pose.Translation.z;
    InverseMotionCoordinates(pose, origin);
    InverseMotionCoordinates(pose, direction);
}
