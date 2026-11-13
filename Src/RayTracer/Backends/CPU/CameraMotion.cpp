#include "RayTracer/Backends/CPU/Motion.h"

namespace Arvo::RayTracing::Cpu
{
// Cook et al. (1984): time is independent of raster and lens dimensions.
// Rotate the already sampled lens ray about the base camera origin. Rotating
// only its direction would leave the aperture behind during camera rotation.
Ray SampleCameraMotion(const Ray& input, Float3 pivot, const MotionData::Exposure& exposure, float variate,
                       float* sceneTime, bool sceneMotion)
{
    if (sceneTime)
        *sceneTime = 0;
    if (!exposure.Enabled && !sceneMotion)
        return input;
    const auto time = SampleShutter(exposure.Interval, variate);
    if (sceneTime)
        *sceneTime = time.Time;
    if (!exposure.Enabled)
        return input;
    const auto pose = InterpolateMotion(exposure.Motion, time.Time);
    Ray ray = input;
    ray.Origin = pivot + TransformMotionPoint(pose, input.Origin - pivot);
    ray.Direction = TransformMotionVector(pose, input.Direction);
    return ray;
}
} // namespace Arvo::RayTracing::Cpu
