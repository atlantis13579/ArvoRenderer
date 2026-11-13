#include "MotionPreparation.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace Arvo::RayTracing::DxrInstances
{
namespace
{
constexpr unsigned AxisCount = 3, CornerCount = 1u << AxisCount;
constexpr double Pi = 3.14159265358979323846;
using Vector = std::array<double, AxisCount>;
using Quaternion = std::array<double, 4>;
Vector Cross(const Vector& a, const Vector& b)
{
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
double Dot(const Vector& a, const Vector& b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
bool Normalize(const MotionData::Quaternion& source, Quaternion& result)
{
    result = {source.X, source.Y, source.Z, source.W};
    double peak = 0;
    for (double value : result)
    {
        if (!std::isfinite(value)) return false;
        peak = std::max(peak, std::abs(value));
    }
    if (peak < std::numeric_limits<float>::min()) return false;
    double square = 0;
    for (double& value : result)
    {
        value /= peak;
        square += value * value;
    }
    const double length = std::sqrt(square);
    for (double& value : result) value /= length;
    return true;
}
Vector Rotate(const Quaternion& rotation, const Vector& point)
{
    const Vector axis{rotation[0], rotation[1], rotation[2]};
    const Vector first = Cross(axis, point), second = Cross(axis, first);
    Vector result{};
    for (unsigned coordinate = 0; coordinate < AxisCount; ++coordinate)
        result[coordinate] = point[coordinate] + 2 * (rotation[3] * first[coordinate] + second[coordinate]);
    return result;
}
}

// Bound the entire shortest quaternion arc, including interior extrema. For a
// rotated corner, each coordinate is A + B*cos(angle) + C*sin(angle); stationary
// points are separated by pi. Linear translation is bounded independently.
// Normalized lerp for nearly parallel keys follows the same arc, so its time
// parameterization does not affect containment. An all-rotation sphere would
// make procedural triangle BLAS boxes overlap across an entire moving mesh.
bool SweptLocalBounds(const MotionData::RigidMotion& motion, const std::array<double, 3>& low,
                      const std::array<double, 3>& high, std::array<double, 3>& resultLow,
                      std::array<double, 3>& resultHigh)
{
    resultLow = low;
    resultHigh = high;
    if (!motion.Enabled)
        return true;
    Quaternion from{}, to{};
    if (!Normalize(motion.Transform.Start.Rotation, from) || !Normalize(motion.Transform.End.Rotation, to))
        return false;
    double cosine = 0;
    for (unsigned component = 0; component < from.size(); ++component)
        cosine += from[component] * to[component];
    if (cosine < 0)
    {
        for (double& value : to) value = -value;
        cosine = -cosine;
    }
    const Vector startAxis{from[0], from[1], from[2]}, endAxis{to[0], to[1], to[2]};
    Vector axis = Cross(startAxis, endAxis);
    for (unsigned coordinate = 0; coordinate < AxisCount; ++coordinate)
        axis[coordinate] += from[3] * endAxis[coordinate] - to[3] * startAxis[coordinate];
    const double sine = std::hypot(axis[0], axis[1], axis[2]);
    const double angle = 2 * std::atan2(sine, cosine);
    if (sine > 0)
        for (double& coordinate : axis) coordinate /= sine;
    double radius = 0;
    for (unsigned coordinate = 0; coordinate < low.size(); ++coordinate)
    {
        if (low[coordinate] > high[coordinate] || !std::isfinite(low[coordinate]) || !std::isfinite(high[coordinate]))
            return false;
        radius += std::max(std::abs(low[coordinate]), std::abs(high[coordinate]));
    }
    resultLow.fill(INFINITY);
    resultHigh.fill(-INFINITY);
    for (unsigned corner = 0; corner < CornerCount; ++corner)
    {
        Vector point{};
        for (unsigned coordinate = 0; coordinate < AxisCount; ++coordinate)
            point[coordinate] = (corner & (1u << coordinate)) ? high[coordinate] : low[coordinate];
        const Vector first = Rotate(from, point), last = Rotate(to, point);
        const Vector tangent = Cross(axis, first);
        const double projection = Dot(axis, first);
        for (unsigned coordinate = 0; coordinate < AxisCount; ++coordinate)
        {
            double minimum = std::min(first[coordinate], last[coordinate]);
            double maximum = std::max(first[coordinate], last[coordinate]);
            if (sine > 0)
            {
                const double parallel = axis[coordinate] * projection;
                const double radial = first[coordinate] - parallel;
                double extremum = std::atan2(tangent[coordinate], radial);
                if (extremum < 0) extremum += Pi;
                if (extremum > 0 && extremum < angle)
                {
                    const double value = parallel + radial * std::cos(extremum) +
                                         tangent[coordinate] * std::sin(extremum);
                    minimum = std::min(minimum, value);
                    maximum = std::max(maximum, value);
                }
            }
            resultLow[coordinate] = std::min(resultLow[coordinate], minimum);
            resultHigh[coordinate] = std::max(resultHigh[coordinate], maximum);
        }
    }
    const auto a = motion.Transform.Start.Translation, b = motion.Transform.End.Translation;
    const double start[]{a.X, a.Y, a.Z}, end[]{b.X, b.Y, b.Z};
    constexpr double FloatOperationBudget = 64;
    constexpr double ErrorProduct = FloatOperationBudget * std::numeric_limits<float>::epsilon();
    constexpr double RelativeMargin = ErrorProduct / (1 - ErrorProduct);
    for (unsigned coordinate = 0; coordinate < resultLow.size(); ++coordinate)
    {
        if (!std::isfinite(start[coordinate]) || !std::isfinite(end[coordinate])) return false;
        const double motionMagnitude = std::max(std::abs(start[coordinate]), std::abs(end[coordinate]));
        const double margin = std::max((radius + motionMagnitude) * RelativeMargin,
                                      FloatOperationBudget * std::numeric_limits<float>::min());
        resultLow[coordinate] += std::min(start[coordinate], end[coordinate]) - margin;
        resultHigh[coordinate] += std::max(start[coordinate], end[coordinate]) + margin;
        if (!std::isfinite(resultLow[coordinate]) || !std::isfinite(resultHigh[coordinate]))
            return false;
    }
    return true;
}
}
