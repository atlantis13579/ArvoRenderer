#include "MotionInstancing.h"

namespace Arvo::RayTracing::Cpu
{
// Apply the inverse rigid pose after the static affine inverse. Keeping the
// intermediate ray in double avoids overflow before direction conditioning.
void InverseInstanceMotion(const MotionData::Pose& pose, std::array<double, 3>& origin,
                            std::array<double, 3>& direction)
{
    const std::array<double, 3> q{-double(pose.Rotation.X), -double(pose.Rotation.Y), -double(pose.Rotation.Z)};
    const auto rotate = [&](std::array<double, 3> v) {
        const std::array<double, 3> twiceCross{2 * (q[1] * v[2] - q[2] * v[1]),
            2 * (q[2] * v[0] - q[0] * v[2]), 2 * (q[0] * v[1] - q[1] * v[0])};
        return std::array<double, 3>{v[0] + pose.Rotation.W * twiceCross[0] + q[1] * twiceCross[2] - q[2] * twiceCross[1],
            v[1] + pose.Rotation.W * twiceCross[1] + q[2] * twiceCross[0] - q[0] * twiceCross[2],
            v[2] + pose.Rotation.W * twiceCross[2] + q[0] * twiceCross[1] - q[1] * twiceCross[0]};
    };
    origin = rotate({origin[0] - pose.Translation.X, origin[1] - pose.Translation.Y, origin[2] - pose.Translation.Z});
    direction = rotate(direction);
}
}
