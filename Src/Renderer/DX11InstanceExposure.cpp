#include "DX11InstanceExposure.h"
#include <algorithm>
#include <cmath>

namespace Arvo::Renderer
{
// Raster visibility observes the shutter midpoint. Composition matches the
// offline instance contract: fixed affine * local rigid pose. Keep a matrix,
// since a rotated object below nonuniform scale is not a single TRS transform.
Maths::Matrix4 InstanceAtExposureMidpoint(const Maths::Matrix4& fixed,
                                          const RayTracing::MotionData::RigidMotion& motion,
                                          RayTracing::MotionData::Shutter shutter)
{
    if (!motion.Enabled)
        return fixed;
    const float time = (shutter.Open + shutter.Close) / 2;
    const auto normalized = [](RayTracing::MotionData::Quaternion value) {
        const float peak =
            std::max({std::abs(value.X), std::abs(value.Y), std::abs(value.Z), std::abs(value.W)});
        return Maths::Quaternion(value.X / peak, value.Y / peak, value.Z / peak, value.W / peak).Normalized();
    };
    const auto& track = motion.Transform;
    const auto orientation =
        Maths::Quaternion::Slerp(normalized(track.Start.Rotation), normalized(track.End.Rotation), time);
    const auto translation = track.Start.Translation * (1 - time) + track.End.Translation * time;
    return fixed * Maths::Matrix4::Translation({translation.X, translation.Y, translation.Z}) *
           Maths::Matrix4::Rotation(orientation);
}
} // namespace Arvo::Renderer
