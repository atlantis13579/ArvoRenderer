#include "MotionPreparation.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace Arvo::RayTracing::MetalInstances
{
// Metal's host prepares conservative world acceleration bounds in double, even
// though native traversal uses scaled float expansions. A full-frame rotation
// envelope also covers a collapsed shutter without rebuilding the hierarchy.
bool BoundMotion(const MotionData::RigidMotion& motion, const std::array<double, 3>& minimum,
                 const std::array<double, 3>& maximum, std::array<double, 3>& sweptMinimum,
                 std::array<double, 3>& sweptMaximum)
{
    sweptMinimum = minimum;
    sweptMaximum = maximum;
    if (motion.Enabled == 0)
        return true;
    std::array<double, 3> absoluteCorner{};
    for (size_t component = 0; component < minimum.size(); ++component)
    {
        if (!std::isfinite(minimum[component]) || !std::isfinite(maximum[component]) || minimum[component] > maximum[component])
            return false;
        absoluteCorner[component] = std::max(std::abs(minimum[component]), std::abs(maximum[component]));
    }
    const double radius = absoluteCorner[0] + absoluteCorner[1] + absoluteCorner[2];
    const auto& first = motion.Transform.Start.Translation;
    const auto& second = motion.Transform.End.Translation;
    const std::array<double, 3> begin{first.X, first.Y, first.Z}, end{second.X, second.Y, second.Z};
    constexpr double ArithmeticBudget = 64;
    constexpr double EpsilonProduct = ArithmeticBudget * std::numeric_limits<float>::epsilon();
    for (size_t component = 0; component < begin.size(); ++component)
    {
        const double scale = std::max(std::abs(begin[component]), std::abs(end[component])) + radius;
        const double padding = std::max(scale * EpsilonProduct / (1 - EpsilonProduct),
                                        ArithmeticBudget * std::numeric_limits<float>::min());
        sweptMinimum[component] = std::min(begin[component], end[component]) - radius - padding;
        sweptMaximum[component] = std::max(begin[component], end[component]) + radius + padding;
        if (!std::isfinite(sweptMinimum[component]) || !std::isfinite(sweptMaximum[component]))
            return false;
    }
    return true;
}
}
