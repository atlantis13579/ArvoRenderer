#include "RayTracer/Backends/CPU/HomogeneousMedium.h"
#include <algorithm>
#include <cmath>

namespace Arvo::RayTracing::Cpu
{
namespace
{
constexpr double FourPi = 4 * 3.14159265358979323846;
constexpr float LargestVariate = 0.9999999403953552f;
constexpr float BasisParallelThreshold = 0.999f;
bool Interval(const MediumOptions& medium, const Ray& ray, float endpoint, float& begin, float& end)
{
    begin = std::max(0.0f, ray.MinDistance);
    end = std::min(endpoint, ray.MaxDistance);
    if (!medium.Enabled)
        return false;
    const float origin[]{ray.Origin.X, ray.Origin.Y, ray.Origin.Z};
    const float direction[]{ray.Direction.X, ray.Direction.Y, ray.Direction.Z};
    const float lower[]{medium.Minimum.X, medium.Minimum.Y, medium.Minimum.Z};
    const float upper[]{medium.Maximum.X, medium.Maximum.Y, medium.Maximum.Z};
    for (unsigned axis = 0; axis < 3; ++axis)
    {
        if (direction[axis] == 0)
        {
            if (origin[axis] < lower[axis] || origin[axis] > upper[axis])
                return false;
            continue;
        }
        const float a = (lower[axis] - origin[axis]) / direction[axis];
        const float b = (upper[axis] - origin[axis]) / direction[axis];
        begin = std::max(begin, std::min(a, b));
        end = std::min(end, std::max(a, b));
    }
    return end > begin;
}
Float3 Beer(Float3 extinction, float distance)
{
    return {float(std::exp(-double(extinction.X) * distance)),
            float(std::exp(-double(extinction.Y) * distance)),
            float(std::exp(-double(extinction.Z) * distance))};
}
} // namespace
// PBRT v3, "Sampling Volume Scattering": RGB channel-mixture free paths.
// Distances are ray parameters; multiplying by |direction| gives world length.
Float3 MediumTransmittance(const MediumOptions& medium, const Ray& ray, float endpoint)
{
    float begin, end;
    if (!Interval(medium, ray, endpoint, begin, end))
        return {1, 1, 1};
    return Beer(medium.Absorption + medium.Scattering,
                (end - begin) * std::sqrt(Dot(ray.Direction, ray.Direction)));
}
MediumSegmentSample SampleMedium(const MediumOptions& medium, const Ray& ray, float endpoint, Float2 variates)
{
    MediumSegmentSample result;
    float begin, end;
    if (!Interval(medium, ray, endpoint, begin, end))
        return result;
    const float speed = std::sqrt(Dot(ray.Direction, ray.Direction));
    if (!(speed > 0))
        return result;
    const Float3 extinction = medium.Absorption + medium.Scattering;
    const float length = (end - begin) * speed;
    if (medium.Scattering.X == 0 && medium.Scattering.Y == 0 && medium.Scattering.Z == 0)
    {
        result.Weight = Beer(extinction, length);
        return result;
    }
    constexpr unsigned ColorChannels = 3;
    const unsigned channel =
        std::min(unsigned(std::clamp(variates.X, 0.0f, LargestVariate) * ColorChannels), ColorChannels - 1);
    const float coefficients[]{extinction.X, extinction.Y, extinction.Z};
    const float coefficient = coefficients[channel];
    const double opticalDepth = -std::log1p(-double(std::clamp(variates.Y, 0.0f, LargestVariate)));
    const double distance = coefficient > 0 ? opticalDepth / coefficient : double(length);
    result.Scattered = distance < length;
    const float travelled = result.Scattered ? float(distance) : length;
    result.Distance = begin + travelled / speed;
    const Float3 transmission = Beer(extinction, travelled);
    const double pdf = result.Scattered
                           ? (double(extinction.X) * transmission.X + double(extinction.Y) * transmission.Y +
                              double(extinction.Z) * transmission.Z) /
                                 ColorChannels
                           : (double(transmission.X) + transmission.Y + transmission.Z) / ColorChannels;
    if (!(pdf > 0))
    {
        result.Weight = {};
        return result;
    }
    const Float3 coefficientWeight = result.Scattered ? medium.Scattering : Float3(1, 1, 1);
    result.Weight = {float(double(transmission.X) * coefficientWeight.X / pdf),
                     float(double(transmission.Y) * coefficientWeight.Y / pdf),
                     float(double(transmission.Z) * coefficientWeight.Z / pdf)};
    return result;
}
float MediumPhase(float cosine, float anisotropy)
{
    const double g = anisotropy;
    const double denominator = 1 + g * g - 2 * g * std::clamp(double(cosine), -1.0, 1.0);
    return float((1 - g) * (1 + g) / (FourPi * denominator * std::sqrt(denominator)));
}
Float3 SampleMediumDirection(Float3 incoming, float anisotropy, Float2 variates, bool importance, float& pdf)
{
    const double g = importance ? anisotropy : 0;
    const double v = 2 * double(std::clamp(variates.X, 0.0f, LargestVariate)) - 1;
    const double divisor = 1 + g * v;
    // Algebraically expanded HG inversion avoids division by g at isotropy.
    const double cosine = std::clamp(
        (2 * v * (1 + g * g) + g * (3 - g * g + v * v * (1 + g * g))) / (2 * divisor * divisor), -1.0, 1.0);
    const double phi = FourPi / 2 * std::clamp(variates.Y, 0.0f, LargestVariate);
    const float sine = float(std::sqrt(std::max(0.0, 1 - cosine * cosine)));
    incoming = Normalize(incoming);
    const Float3 helper = std::abs(incoming.Y) < BasisParallelThreshold ? Float3(0, 1, 0) : Float3(1, 0, 0);
    const Float3 tangent = Normalize(Cross(helper, incoming));
    const Float3 direction = Normalize(incoming * float(cosine) + tangent * (sine * float(std::cos(phi))) +
                                       Cross(incoming, tangent) * (sine * float(std::sin(phi))));
    pdf = importance ? MediumPhase(Dot(incoming, direction), anisotropy) : float(1 / FourPi);
    return direction;
}
} // namespace Arvo::RayTracing::Cpu
