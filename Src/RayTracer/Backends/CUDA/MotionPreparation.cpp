#include "MotionPreparation.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace Arvo::RayTracing::CudaInstances
{
// Rotation preserves distance from the object origin. The L1 corner radius
// bounds that distance without a square/sqrt overflow before affine contraction.
// This host envelope covers the full frame, independent of the camera shutter.
bool MotionBounds(const MotionData::RigidMotion& motion, const std::array<double, 3>& lower,
                  const std::array<double, 3>& upper, std::array<double, 3>& sweptLower,
                  std::array<double, 3>& sweptUpper)
{
    if (motion.Enabled == 0)
    {
        sweptLower = lower;
        sweptUpper = upper;
        return true;
    }
    double envelope = 0;
    for (size_t axis = 0; axis != lower.size(); ++axis)
    {
        if (!(std::isfinite(lower[axis]) && std::isfinite(upper[axis]) && lower[axis] <= upper[axis]))
            return false;
        envelope += std::max(std::abs(lower[axis]), std::abs(upper[axis]));
    }
    const auto& p = motion.Transform.Start.Translation;
    const auto& q = motion.Transform.End.Translation;
    const std::array<double, 3> first{p.X, p.Y, p.Z}, last{q.X, q.Y, q.Z};
    // Budget quaternion normalization, interpolation, rotation and translation.
    constexpr double RoundoffOperations = 64;
    constexpr double UnitRoundoff = std::numeric_limits<float>::epsilon();
    constexpr double RelativeSlack = RoundoffOperations * UnitRoundoff / (1 - RoundoffOperations * UnitRoundoff);
    constexpr double UnderflowSlack = RoundoffOperations * std::numeric_limits<float>::min();
    for (size_t axis = 0; axis != first.size(); ++axis)
    {
        const double low = std::min(first[axis], last[axis]), high = std::max(first[axis], last[axis]);
        const double slack = std::max(UnderflowSlack, RelativeSlack * (envelope + std::max(std::abs(low), std::abs(high))));
        sweptLower[axis] = low - envelope - slack;
        sweptUpper[axis] = high + envelope + slack;
        if (!std::isfinite(sweptLower[axis]) || !std::isfinite(sweptUpper[axis]))
            return false;
    }
    return true;
}
}
