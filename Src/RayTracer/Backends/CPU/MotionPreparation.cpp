#include "MotionPreparation.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace Arvo::RayTracing::Cpu
{
// A rotation may leave the union of its endpoint boxes. An L1 radius encloses
// every intermediate orientation, then translation sweeps that envelope.
// PBRT, "Animating Transformations". Double host bounds survive a later affine
// contraction; they need not be representable in float until world-space upload.
bool SweptInstanceBounds(const MotionData::RigidMotion& motion, const std::array<double, 3>& lower,
                         const std::array<double, 3>& upper, std::array<double, 3>& sweptLower,
                         std::array<double, 3>& sweptUpper)
{
    if (!motion.Enabled)
    {
        sweptLower = lower;
        sweptUpper = upper;
        return true;
    }
    double radius = 0;
    for (size_t axis = 0; axis < lower.size(); ++axis)
    {
        if (!std::isfinite(lower[axis]) || !std::isfinite(upper[axis]) || lower[axis] > upper[axis])
            return false;
        radius += std::max(std::abs(lower[axis]), std::abs(upper[axis]));
    }
    const auto start = motion.Transform.Start.Translation, end = motion.Transform.End.Translation;
    const double a[]{start.X, start.Y, start.Z}, b[]{end.X, end.Y, end.Z};
    constexpr double ArithmeticSteps = 64, Epsilon = std::numeric_limits<float>::epsilon();
    constexpr double Gamma = ArithmeticSteps * Epsilon / (1 - ArithmeticSteps * Epsilon);
    for (size_t axis = 0; axis < lower.size(); ++axis)
    {
        const double magnitude = radius + std::max(std::abs(a[axis]), std::abs(b[axis]));
        const double margin = std::max(magnitude * Gamma, ArithmeticSteps * std::numeric_limits<float>::min());
        sweptLower[axis] = std::min(a[axis], b[axis]) - radius - margin;
        sweptUpper[axis] = std::max(a[axis], b[axis]) + radius + margin;
        if (!std::isfinite(sweptLower[axis]) || !std::isfinite(sweptUpper[axis]))
            return false;
    }
    return true;
}
} // namespace Arvo::RayTracing::Cpu
