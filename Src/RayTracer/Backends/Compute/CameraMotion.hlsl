#include "MotionData.hlsli"
static const uint TimeSampleDimension = 0x54494d45u; // TIME, separate from lens and wavelength streams.
struct MotionCameraRay
{
    float3 Origin, Direction;
    float SceneTime;
};
MotionCameraRay SampleCameraMotion(float3 origin, float3 direction, float3 pivot, MotionExposure exposure,
                                   float variate)
{
    MotionCameraRay ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.SceneTime = 0;
    if (exposure.Enabled == 0 && exposure.SceneEnabled == 0)
        return ray;
    MotionShutter shutter;
    shutter.Open = exposure.Open;
    shutter.Close = exposure.Close;
    MotionTimeSample time = SampleShutter(shutter, variate);
    ray.SceneTime = time.Time;
    if (exposure.Enabled == 0)
        return ray;
    MotionPose pose = InterpolateMotion(exposure.Motion, time.Time);
    // The lens origin and orientation belong to the same rigid camera frame.
    ray.Origin = pivot + TransformMotionPoint(pose, origin - pivot);
    ray.Direction = TransformMotionVector(pose, direction);
    return ray;
}
