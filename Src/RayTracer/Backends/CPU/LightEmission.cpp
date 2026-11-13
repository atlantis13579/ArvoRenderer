#include "RayTracer/Backends/CPU/LightEmission.h"
#include <algorithm>
#include <cmath>

namespace Arvo::RayTracing::Cpu
{
// PBRT, "The Path-Space Measurement Equation", section 16.1.2:
// https://pbr-book.org/3ed-2018/Light_Transport_III_Bidirectional_Methods/The_Path-Space_Measurement_Equation
// Emission samples seed importance paths / photons. Weight is emitted flux per
// conditional sample. Value is radiance for rectangles/environment, radiant
// intensity for points, and irradiance for directional lights. Delta flags are
// essential when comparing densities in bidirectional MIS.
namespace
{
constexpr float EmissionPi = 3.14159265358979323846f;
constexpr float FrameParallelThreshold = 0.999f;
const Float3 LinearRgbLuminance{0.2126f, 0.7152f, 0.0722f};
void Frame(Float3 normal, Float3& tangent, Float3& bitangent)
{
    const Float3 reference = std::abs(normal.Y) < FrameParallelThreshold ? Float3{0, 1, 0} : Float3{1, 0, 0};
    tangent = Normalize(Cross(reference, normal));
    bitangent = Cross(normal, tangent);
}
Float2 Disk(Float2 u)
{
    const float radius = std::sqrt(u.X), angle = 2 * EmissionPi * u.Y;
    return {radius * std::cos(angle), radius * std::sin(angle)};
}
Float3 Latitude(float y, float azimuth)
{
    const float radial = std::sqrt(std::max(0.0f, 1 - y * y)), angle = 2 * EmissionPi * azimuth;
    return {radial * std::cos(angle), y, radial * std::sin(angle)};
}
Float2 GradientWeights(const EmissionSource& source)
{
    // Scale colors before luminance / addition to avoid overflowing bright HDR.
    const float scale =
        std::max({source.BackgroundTop.X, source.BackgroundTop.Y, source.BackgroundTop.Z,
                  source.BackgroundBottom.X, source.BackgroundBottom.Y, source.BackgroundBottom.Z});
    if (!(scale > 0))
        return {};
    const float bottom = Dot(source.BackgroundBottom / scale, LinearRgbLuminance);
    const float top = Dot(source.BackgroundTop / scale, LinearRgbLuminance);
    return {bottom / (bottom + top), top / (bottom + top)};
}
float GradientPdf(Float2 weights, float y)
{
    return ((1 - y) * weights.X + (1 + y) * weights.Y) / (4 * EmissionPi);
}
} // namespace

EmissionDensity LightEmissionPdf(const EmissionSource& source, const EmissionDomain& domain, Float3 direction)
{
    switch (source.Kind)
    {
    case EmissionKind::Point:
        return {0, 1 / (4 * EmissionPi)};
    case EmissionKind::Rectangle:
        if (!(source.Size.X > 0 && source.Size.Y > 0))
            return {};
        return {1 / (source.Size.X * source.Size.Y),
                std::max(0.0f, Dot(Normalize(source.Direction), direction)) / EmissionPi};
    case EmissionKind::Directional:
        return domain.Radius > 0 ? EmissionDensity{1 / (EmissionPi * domain.Radius * domain.Radius), 0}
                                 : EmissionDensity{};
    case EmissionKind::Environment:
        if (!(domain.Radius > 0))
            return {};
        return {1 / (EmissionPi * domain.Radius * domain.Radius),
                GradientPdf(GradientWeights(source), -direction.Y)};
    }
    return {};
}

EmissionSample SampleLightEmission(const EmissionSource& source, const EmissionDomain& domain,
                                   Float2 positionVariates, Float2 directionVariates)
{
    EmissionSample sample;
    if (source.Kind == EmissionKind::Point)
    {
        sample.Origin = source.Position;
        sample.Direction = Latitude(1 - 2 * directionVariates.X, directionVariates.Y);
        sample.Normal = sample.Direction;
        sample.Value = source.Strength;
        sample.PositionPdf = 1;
        sample.DirectionPdf = 1 / (4 * EmissionPi);
        sample.Weight = source.Strength * (4 * EmissionPi);
        sample.Flags = EmissionValid | EmissionDeltaPosition;
        return sample;
    }
    if (source.Kind == EmissionKind::Rectangle)
    {
        if (!(source.Size.X > 0 && source.Size.Y > 0))
            return sample;
        Float3 tangent, bitangent;
        sample.Normal = Normalize(source.Direction);
        Frame(sample.Normal, tangent, bitangent);
        sample.Origin = source.Position + tangent * ((positionVariates.X - 0.5f) * source.Size.X) +
                        bitangent * ((positionVariates.Y - 0.5f) * source.Size.Y);
        const Float2 disk = Disk(directionVariates);
        const float cosine = std::sqrt(std::max(0.0f, 1 - directionVariates.X));
        if (!(cosine > 0))
            return {};
        sample.Direction = tangent * disk.X + bitangent * disk.Y + sample.Normal * cosine;
        sample.PositionPdf = 1 / (source.Size.X * source.Size.Y);
        sample.DirectionPdf = cosine / EmissionPi;
        sample.Value = source.Strength * sample.PositionPdf;
        // The demo's intensity convention is radiance * area, not total power.
        sample.Weight = source.Strength * EmissionPi;
        sample.Flags = EmissionValid;
        return sample;
    }
    if (!(domain.Radius > 0))
        return sample;
    Float3 outward;
    if (source.Kind == EmissionKind::Directional)
    {
        outward = Normalize(source.Direction);
        sample.Value = source.Strength;
        sample.DirectionPdf = 1;
        sample.Flags = EmissionValid | EmissionInfinite | EmissionDeltaDirection;
    }
    else if (source.Kind == EmissionKind::Environment)
    {
        const Float2 weights = GradientWeights(source);
        if (!(weights.X + weights.Y > 0))
            return sample;
        // Invert F(q)=2*b*q+(t-b)*q*q without cancellation, q=(y+1)/2.
        const float u = directionVariates.X;
        const float denominator =
            weights.X + std::sqrt((1 - u) * weights.X * weights.X + u * weights.Y * weights.Y);
        const float latitude = denominator > 0 ? std::clamp(u / denominator, 0.0f, 1.0f) : 0;
        outward = Latitude(2 * latitude - 1, directionVariates.Y);
        sample.DirectionPdf = GradientPdf(weights, outward.Y);
        if (!(sample.DirectionPdf > 0))
            return {};
        sample.Value = source.BackgroundBottom * (1 - latitude) + source.BackgroundTop * latitude;
        sample.Flags = EmissionValid | EmissionInfinite;
    }
    else
        return sample;
    Float3 tangent, bitangent;
    Frame(outward, tangent, bitangent);
    const Float2 disk = Disk(positionVariates);
    sample.Origin = domain.Center + (outward + tangent * disk.X + bitangent * disk.Y) * domain.Radius;
    sample.Direction = -outward;
    sample.Normal = sample.Direction;
    sample.PositionPdf = 1 / (EmissionPi * domain.Radius * domain.Radius);
    sample.Weight = sample.Value / (sample.PositionPdf * sample.DirectionPdf);
    return sample;
}
} // namespace Arvo::RayTracing::Cpu
