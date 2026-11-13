#include "MotionData.hlsli"
static const uint TimeSampleDimension = 0x54494d45u;
struct MotionCameraRay
{
    float3 Origin, Direction;
    float SceneTime;
};
MotionCameraRay SampleCameraMotion(float3 origin, float3 direction, float3 pivot, MotionExposure exposure,
                                   float variate)
{
    MotionCameraRay result;
    result.Origin = origin;
    result.Direction = direction;
    result.SceneTime = 0;
    if (exposure.Enabled != 0 || exposure.SceneEnabled != 0)
    {
        // Preview supplies a zero-width midpoint shutter to this shared estimator.
        MotionShutter shutter;
        shutter.Open = exposure.Open;
        shutter.Close = exposure.Close;
        MotionTimeSample observation = SampleShutter(shutter, variate);
        result.SceneTime = observation.Time;
        if (exposure.Enabled == 0)
            return result;
        MotionPose pose = InterpolateMotion(exposure.Motion, observation.Time);
        result.Origin = TransformMotionPoint(pose, origin - pivot) + pivot;
        result.Direction = TransformMotionVector(pose, direction);
    }
    return result;
}
