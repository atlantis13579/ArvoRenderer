#include "Renderer/RayTracing.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace Arvo::RayTracing
{
bool SupportsMotionBlur(Algorithm algorithm)
{
    return algorithm == Algorithm::PathTracing || algorithm == Algorithm::DistributedRayTracing;
}
bool ValidateCameraMotion(const MotionData::Exposure& exposure, std::string& error)
{
    constexpr float MaximumCameraCoordinate = 1000000;
    const auto finite = [](float value) {
        return std::isfinite(value) && std::fpclassify(value) != FP_SUBNORMAL &&
               std::abs(value) <= MaximumCameraCoordinate;
    };
    const auto validPose = [&](const MotionData::Pose& pose) {
        const auto q = pose.Rotation;
        return finite(q.X) && finite(q.Y) && finite(q.Z) && finite(q.W) &&
               std::max({std::abs(q.X), std::abs(q.Y), std::abs(q.Z), std::abs(q.W)}) >=
                   std::numeric_limits<float>::min() &&
               finite(pose.Translation.X) && finite(pose.Translation.Y) && finite(pose.Translation.Z);
    };
    const auto s = exposure.Interval;
    const float width = s.Close - s.Open;
    if (exposure.Enabled > 1 || !std::isfinite(s.Open) || !std::isfinite(s.Close) || s.Open < 0 ||
        s.Close > 1 || s.Open > s.Close || std::fpclassify(s.Open) == FP_SUBNORMAL ||
        std::fpclassify(s.Close) == FP_SUBNORMAL ||
        (s.Open != s.Close && width < std::numeric_limits<float>::min()) ||
        !validPose(exposure.Motion.Start) || !validPose(exposure.Motion.End))
    {
        error = "Camera exposure requires an ordered shutter in [0,1], finite nonzero rotations and "
                "translations within +/-1000000 world units.";
        return false;
    }
    return true;
}
bool ValidateCameraMotionSupport(Algorithm algorithm, const MotionData::Exposure& exposure,
                                 std::string& error)
{
    if (!ValidateCameraMotion(exposure, error))
        return false;
    if (exposure.Enabled && !SupportsMotionBlur(algorithm))
    {
        error = "Camera exposure requires path or distributed ray tracing; connected-path and cache "
                "integrators need a common time measure.";
        return false;
    }
    return true;
}
} // namespace Arvo::RayTracing
