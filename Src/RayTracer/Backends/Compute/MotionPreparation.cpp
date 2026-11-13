#include "MotionPreparation.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace Arvo::RayTracing::ComputeInstances
{
// Native Compute preparation bounds the entire normalized frame. A rotational
// envelope is necessary even when the two endpoint boxes coincide.
bool SweptBounds(const MotionData::RigidMotion& motion, const std::array<double, 3>& minimum,
                 const std::array<double, 3>& maximum, std::array<double, 3>& resultMinimum,
                 std::array<double, 3>& resultMaximum)
{
    resultMinimum = minimum;
    resultMaximum = maximum;
    if (!motion.Enabled)
        return true;
    double reach = 0;
    for (unsigned coordinate = 0; coordinate < minimum.size(); ++coordinate)
    {
        const double low = minimum[coordinate], high = maximum[coordinate];
        if (!std::isfinite(low) || !std::isfinite(high) || low > high)
            return false;
        reach += std::max(std::abs(low), std::abs(high));
    }
    const auto& first = motion.Transform.Start.Translation;
    const auto& last = motion.Transform.End.Translation;
    const std::array<double, 3> start{first.X, first.Y, first.Z}, finish{last.X, last.Y, last.Z};
    constexpr double RoundoffSteps = 64;
    constexpr double ProductError = RoundoffSteps * std::numeric_limits<float>::epsilon();
    for (unsigned coordinate = 0; coordinate < start.size(); ++coordinate)
    {
        const double scale = reach + std::max(std::abs(start[coordinate]), std::abs(finish[coordinate]));
        const double padding = std::max(scale * ProductError / (1 - ProductError),
                                        RoundoffSteps * std::numeric_limits<float>::min());
        resultMinimum[coordinate] = std::min(start[coordinate], finish[coordinate]) - reach - padding;
        resultMaximum[coordinate] = std::max(start[coordinate], finish[coordinate]) + reach + padding;
        if (!std::isfinite(resultMinimum[coordinate]) || !std::isfinite(resultMaximum[coordinate]))
            return false;
    }
    return true;
}
} // namespace Arvo::RayTracing::ComputeInstances
