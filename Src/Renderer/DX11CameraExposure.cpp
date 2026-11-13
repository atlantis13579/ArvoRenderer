#include "Renderer/DX11CameraExposure.h"
#include "Maths/Quaternion.h"
#include <algorithm>
#include <cmath>

namespace Arvo::Renderer
{
RayTracing::Camera CameraAtExposureMidpoint(const RayTracing::Camera& camera)
{
    if (!camera.Motion.Enabled)
        return camera;
    // Raster preview uses the shutter centre pose. It does not claim to integrate
    // visibility over exposure. Offline paths integrate time; DXR preview also
    // supplies a zero-width midpoint shutter to its shared camera estimator.
    const float time = (camera.Motion.Interval.Open + camera.Motion.Interval.Close) / 2;
    const auto unit = [](RayTracing::MotionData::Quaternion q) {
        const float scale = std::max({std::abs(q.X), std::abs(q.Y), std::abs(q.Z), std::abs(q.W)});
        return Maths::Quaternion(q.X / scale, q.Y / scale, q.Z / scale, q.W / scale).Normalized();
    };
    const auto rotation = Maths::Quaternion::Slerp(unit(camera.Motion.Motion.Start.Rotation),
                                                   unit(camera.Motion.Motion.End.Rotation), time);
    const auto rotate = [&](RayTracing::Float3 value) {
        const auto result = rotation.Rotate({value.X, value.Y, value.Z});
        return RayTracing::Float3{result.x, result.y, result.z};
    };
    auto result = camera;
    result.Position = camera.Position + camera.Motion.Motion.Start.Translation * (1 - time) +
                      camera.Motion.Motion.End.Translation * time;
    result.Target = result.Position + rotate(camera.Target - camera.Position);
    result.Up = rotate(camera.Up);
    result.Motion.Enabled = 0;
    return result;
}
} // namespace Arvo::Renderer
