#include "RayTracer/Backends/CPU/Motion.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace Arvo::RayTracing::Cpu
{
namespace
{
using Box = MotionData::Bounds;
bool ValidBox(const Box& b)
{
    return b.Valid && std::isfinite(b.Minimum.X) && std::isfinite(b.Minimum.Y) &&
           std::isfinite(b.Minimum.Z) && std::isfinite(b.Maximum.X) && std::isfinite(b.Maximum.Y) &&
           std::isfinite(b.Maximum.Z) && b.Minimum.X <= b.Maximum.X && b.Minimum.Y <= b.Maximum.Y &&
           b.Minimum.Z <= b.Maximum.Z;
}
Float3 Minimum(Float3 a, Float3 b)
{
    return {std::min(a.X, b.X), std::min(a.Y, b.Y), std::min(a.Z, b.Z)};
}
Float3 Maximum(Float3 a, Float3 b)
{
    return {std::max(a.X, b.X), std::max(a.Y, b.Y), std::max(a.Z, b.Z)};
}
Box Inflated(Float3 low, Float3 high, float magnitude)
{
    constexpr float ArithmeticSteps = 64, Epsilon = std::numeric_limits<float>::epsilon();
    constexpr float Gamma = ArithmeticSteps * Epsilon / (1 - ArithmeticSteps * Epsilon);
    const float margin = std::max(magnitude * Gamma, ArithmeticSteps * std::numeric_limits<float>::min());
    const Float3 padding{margin, margin, margin};
    Box result{low - padding, 1, high + padding, 0};
    result.Valid = ValidBox(result);
    return result;
}
float Magnitude(Float3 p)
{
    return std::max({std::abs(p.X), std::abs(p.Y), std::abs(p.Z)});
}
} // namespace
Box LinearMotionBounds(const Box& start, const Box& end, MotionData::Shutter shutter)
{
    if (!ValidBox(start) || !ValidBox(end) || !ValidShutter(shutter))
        return {{}, 0, {}, 0};
    const auto low = Minimum(InterpolateMotionPoint(start.Minimum, end.Minimum, shutter.Open),
                             InterpolateMotionPoint(start.Minimum, end.Minimum, shutter.Close));
    const auto high = Maximum(InterpolateMotionPoint(start.Maximum, end.Maximum, shutter.Open),
                              InterpolateMotionPoint(start.Maximum, end.Maximum, shutter.Close));
    return Inflated(low, high,
                    std::max({Magnitude(start.Minimum), Magnitude(start.Maximum), Magnitude(end.Minimum),
                              Magnitude(end.Maximum)}));
}
Box RigidMotionBounds(const Box& box, const MotionData::Track& track, MotionData::Shutter shutter)
{
    // PBRT, "Animating Transformations", motion bounds. An endpoint union is
    // unsafe under rotation. This deliberately loose envelope encloses every
    // orientation using an L1 radius about the transform origin, then sweeps
    // the translating origin. The margin covers interpolation/rotation roundoff.
    // https://pbr-book.org/3ed-2018/Geometry_and_Transformations/Animating_Transformations
    if (!ValidBox(box) || !ValidMotionTrack(track) || !ValidShutter(shutter))
        return {{}, 0, {}, 0};
    const float radius = std::max(std::abs(box.Minimum.X), std::abs(box.Maximum.X)) +
                         std::max(std::abs(box.Minimum.Y), std::abs(box.Maximum.Y)) +
                         std::max(std::abs(box.Minimum.Z), std::abs(box.Maximum.Z));
    const auto a = InterpolateMotionPoint(track.Start.Translation, track.End.Translation, shutter.Open);
    const auto b = InterpolateMotionPoint(track.Start.Translation, track.End.Translation, shutter.Close);
    const Float3 extent{radius, radius, radius};
    return Inflated(Minimum(a, b) - extent, Maximum(a, b) + extent,
                    radius + std::max(Magnitude(track.Start.Translation), Magnitude(track.End.Translation)));
}
} // namespace Arvo::RayTracing::Cpu
