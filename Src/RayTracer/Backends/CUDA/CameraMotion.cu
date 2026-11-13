#include "MotionData.cuh"
static constexpr unsigned TimeSampleDimension = 0x54494d45u;
struct MotionCameraRay
{
    float3 Origin, Direction;
    float SceneTime;
};
__device__ MotionCameraRay SampleCameraMotion(float3 origin, float3 direction, float3 pivot,
                                              MotionExposure exposure, float variate)
{
    if (!exposure.Enabled && !exposure.SceneEnabled)
        return {origin, direction, 0};
    const auto time = SampleShutter({exposure.Open, exposure.Close}, variate);
    if (!exposure.Enabled)
        return {origin, direction, time.Time};
    const auto pose = InterpolateMotion(exposure.Motion, time.Time);
    const float3 relative = make_float3(origin.x - pivot.x, origin.y - pivot.y, origin.z - pivot.z);
    const float3 moved = TransformMotionPoint(pose, relative);
    // Preserve the lens offset and ray interval under the same rigid pose.
    return {make_float3(moved.x + pivot.x, moved.y + pivot.y, moved.z + pivot.z),
            TransformMotionVector(pose, direction), time.Time};
}
