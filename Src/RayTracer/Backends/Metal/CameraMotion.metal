#include "MotionData.h"
constant uint TimeSampleDimension = 0x54494d45u;
struct MotionCameraRay
{
    float3 Origin, Direction;
    float SceneTime;
};
MotionCameraRay SampleCameraMotion(float3 origin, float3 direction, float3 pivot, MotionExposure exposure,
                                   float variate)
{
    MotionCameraRay result = {origin, direction, 0};
    if (!exposure.Enabled && !exposure.SceneEnabled)
        return result;
    MotionTimeSample time = SampleShutter({exposure.Open, exposure.Close}, variate);
    result.SceneTime = time.Time;
    if (!exposure.Enabled)
        return result;
    MotionPose pose = InterpolateMotion(exposure.Motion, time.Time);
    // Aperture and sensor frame move together; shutter response/PDF is unity.
    result.Origin = pivot + TransformMotionPoint(pose, origin - pivot);
    result.Direction = TransformMotionVector(pose, direction);
    return result;
}
