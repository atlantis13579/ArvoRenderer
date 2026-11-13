#include "RayTracer/Backends/CPU/Motion.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace Arvo::RayTracing::Cpu
{
namespace
{
using Q = MotionData::Quaternion;
float Peak(Q q)
{
    return std::max({std::abs(q.X), std::abs(q.Y), std::abs(q.Z), std::abs(q.W)});
}
Q Unit(Q q)
{
    const double scale = Peak(q);
    const double x = q.X / scale, y = q.Y / scale, z = q.Z / scale, w = q.W / scale;
    const double inverse = 1 / std::sqrt(x * x + y * y + z * z + w * w);
    return {float(x * inverse), float(y * inverse), float(z * inverse), float(w * inverse)};
}
bool ValidPose(const MotionData::Pose& pose)
{
    const auto q = pose.Rotation;
    const auto t = pose.Translation;
    return std::isfinite(q.X) && std::isfinite(q.Y) && std::isfinite(q.Z) && std::isfinite(q.W) &&
           Peak(q) >= std::numeric_limits<float>::min() && std::isfinite(t.X) && std::isfinite(t.Y) &&
           std::isfinite(t.Z);
}
} // namespace
bool ValidMotionTrack(const MotionData::Track& track)
{
    return ValidPose(track.Start) && ValidPose(track.End);
}
Float3 InterpolateMotionPoint(Float3 a, Float3 b, float time)
{
    if (time == 0)
        return a;
    if (time == 1)
        return b;
    // Convex products avoid overflowing b-a for opposite finite endpoints.
    return a * (1 - time) + b * time;
}
MotionData::Pose InterpolateMotion(const MotionData::Track& track, float time)
{
    // Shoemake, "Animating Rotation with Quaternion Curves" (1985).
    // The caller validates the track and time. No elementwise matrix lerp:
    // singular intermediate frames would destroy ray parameter invariance.
    Q a = Unit(track.Start.Rotation), b = Unit(track.End.Rotation);
    double cosine = double(a.X) * b.X + double(a.Y) * b.Y + double(a.Z) * b.Z + double(a.W) * b.W;
    if (cosine < 0)
    {
        b = {-b.X, -b.Y, -b.Z, -b.W};
        cosine = -cosine;
    }
    cosine = std::min(cosine, 1.0);
    constexpr double ParallelThreshold = 1 - 32 * std::numeric_limits<float>::epsilon();
    double first = 1 - double(time), second = time;
    if (cosine < ParallelThreshold)
    {
        const double angle = std::acos(cosine), denominator = std::sin(angle);
        first = std::sin((1 - time) * angle) / denominator;
        second = std::sin(time * angle) / denominator;
    }
    const Q rotation = Unit({float(first * a.X + second * b.X), float(first * a.Y + second * b.Y),
                             float(first * a.Z + second * b.Z), float(first * a.W + second * b.W)});
    return {rotation, InterpolateMotionPoint(track.Start.Translation, track.End.Translation, time), 0};
}
Float3 TransformMotionVector(const MotionData::Pose& pose, Float3 vector)
{
    const Float3 axis{pose.Rotation.X, pose.Rotation.Y, pose.Rotation.Z};
    const Float3 twiceCross = Cross(axis, vector) * 2;
    return vector + twiceCross * pose.Rotation.W + Cross(axis, twiceCross);
}
Float3 TransformMotionPoint(const MotionData::Pose& pose, Float3 point)
{
    return TransformMotionVector(pose, point) + pose.Translation;
}
Float3 InverseMotionVector(const MotionData::Pose& pose, Float3 vector)
{
    auto inverse = pose;
    inverse.Rotation.X = -inverse.Rotation.X;
    inverse.Rotation.Y = -inverse.Rotation.Y;
    inverse.Rotation.Z = -inverse.Rotation.Z;
    // Do not normalize the ray direction: its world-space t must survive.
    return TransformMotionVector(inverse, vector);
}
Float3 InverseMotionPoint(const MotionData::Pose& pose, Float3 point)
{
    return InverseMotionVector(pose, point - pose.Translation);
}
} // namespace Arvo::RayTracing::Cpu
