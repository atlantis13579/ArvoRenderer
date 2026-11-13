#include "TemporalDenoise.h"
#include "Maths/ColorSpace.h"
#include <algorithm>
#include <array>
#include <cmath>

namespace Arvo::RayTracing::Cpu
{
namespace
{
// Geometry-validated temporal reconstruction, independently authored for CPU.
// Reproject bilinear taps, reject incompatible visibility, clip accepted history
// to compatible current neighbors, then blend with bounded frame age. This is
// a biased radiance filter, not a replacement for transport moments or SVGF.
constexpr double Pi = 3.14159265358979323846, HalfTurn = 180, PixelCenter = 0.5;
constexpr int NeighborhoodRadius = 1, BilinearWidth = 2;
Float3 Unit(Float3 v)
{
    const float scale = std::max({std::abs(v.X), std::abs(v.Y), std::abs(v.Z)});
    if (!(scale > 0))
        return {};
    v = v / scale;
    return v / std::sqrt(Dot(v, v));
}
bool Compatible(const GeometryGuide& center, const GeometryGuide& other, float expectedDepth,
                const TemporalData::Controls& c)
{
    if (center.Flags & GeometryGuide::Background)
        return (other.Flags & GeometryGuide::Background) != 0;
    if (!(other.Flags & GeometryGuide::Surface) || center.PrimitiveIndex != other.PrimitiveIndex ||
        center.MaterialIndex != other.MaterialIndex || center.Flags != other.Flags ||
        Dot(center.GeometricNormal, other.GeometricNormal) < c.MinimumNormalCosine)
        return false;
    const double tolerance =
        c.AbsoluteDepthTolerance +
        double(c.RelativeDepthTolerance) * std::max(double(expectedDepth), double(other.CameraDepth));
    const Float3 delta = other.Position - center.Position;
    const double plane = std::max(std::abs(double(Dot(delta, center.GeometricNormal))),
                                  std::abs(double(Dot(delta, other.GeometricNormal))));
    return std::isfinite(plane) && std::abs(double(other.CameraDepth) - expectedDepth) <= tolerance &&
           plane <= tolerance;
}
bool Project(const GeometryGuide& guide, const TemporalData::Controls& c, Float3 forward, Float3 right,
             Float3 up, double& x, double& y, float& depth)
{
    const Float3 relative = guide.Flags & GeometryGuide::Background
                                ? guide.Direction
                                : guide.Position - Float3{c.EyeX, c.EyeY, c.EyeZ};
    depth = Dot(relative, forward);
    if (!(depth > 0) || !std::isfinite(depth))
        return false;
    const double halfY = std::tan(double(c.FieldOfView) * Pi / (2 * HalfTurn));
    const double halfX = halfY * c.Width / c.Height;
    x = (double(Dot(relative, right)) / depth / halfX + 1) * c.Width / 2 - PixelCenter;
    y = (1 - double(Dot(relative, up)) / depth / halfY) * c.Height / 2 - PixelCenter;
    return std::isfinite(x) && std::isfinite(y) && x > -1 && x < c.Width && y > -1 && y < c.Height;
}
using Color = std::array<double, 3>;
Color Wide(Float3 p)
{
    return {p.X, p.Y, p.Z};
}
} // namespace
bool TemporalDenoise(const TemporalData::Controls& c, const std::vector<TemporalPixel>& current,
                     const std::vector<GeometryGuide>& geometry, const std::vector<TemporalPixel>& previous,
                     const std::vector<GeometryGuide>& oldGeometry,
                     const Detail::CancellationProbe& cancelled, std::vector<TemporalPixel>& result,
                     Image& display, std::string& error)
{
    const size_t count = size_t(c.Width) * c.Height;
    if (!count || current.size() != count || geometry.size() != count || !c.MaximumHistory ||
        (c.HasHistory && (previous.size() != count || oldGeometry.size() != count)))
    {
        error = "CPU temporal reconstruction extent mismatch";
        return false;
    }
    const Float3 forward = Unit({c.TargetX - c.EyeX, c.TargetY - c.EyeY, c.TargetZ - c.EyeZ});
    const Float3 right = Unit(Cross(forward, {c.UpX, c.UpY, c.UpZ})), up = Cross(right, forward);
    std::vector<TemporalPixel> pixels(count);
    Image encoded(c.Width, c.Height);
    constexpr uint32_t Channels = 4, Opaque = 255;
    for (uint32_t y = 0; y < c.Height; ++y)
        for (uint32_t x = 0; x < c.Width; ++x)
        {
            if (cancelled && cancelled())
            {
                error = "CPU temporal reconstruction cancelled";
                return false;
            }
            const size_t index = size_t(y) * c.Width + x;
            const auto& guide = geometry[index];
            const Color center = Wide(current[index].Radiance);
            Color color = center;
            uint32_t age = 1;
            double px = 0, py = 0;
            float depth = 0;
            if (c.HasHistory && Project(guide, c, forward, right, up, px, py, depth))
            {
                const int left = int(std::floor(px)), top = int(std::floor(py));
                const double fx = px - left, fy = py - top;
                Color history{};
                double accepted = 0;
                uint32_t youngest = c.MaximumHistory;
                for (int dy = 0; dy < BilinearWidth; ++dy)
                    for (int dx = 0; dx < BilinearWidth; ++dx)
                    {
                        const int sx = left + dx, sy = top + dy;
                        const double weight = (dx ? fx : 1 - fx) * (dy ? fy : 1 - fy);
                        if (!(weight > 0) || sx < 0 || sy < 0 || sx >= int(c.Width) || sy >= int(c.Height))
                            continue;
                        const size_t tap = size_t(sy) * c.Width + sx;
                        if (!Compatible(guide, oldGeometry[tap], depth, c))
                            continue;
                        const auto value = Wide(previous[tap].Radiance);
                        for (size_t channel = 0; channel < history.size(); ++channel)
                            history[channel] += weight * value[channel];
                        accepted += weight;
                        youngest = std::min(youngest, previous[tap].Age);
                    }
                if (accepted >= c.MinimumHistoryWeight)
                {
                    Color low = center, high = center;
                    for (int dy = -NeighborhoodRadius; dy <= NeighborhoodRadius; ++dy)
                        for (int dx = -NeighborhoodRadius; dx <= NeighborhoodRadius; ++dx)
                        {
                            const int sx = int(x) + dx, sy = int(y) + dy;
                            if (sx < 0 || sy < 0 || sx >= int(c.Width) || sy >= int(c.Height))
                                continue;
                            const size_t tap = size_t(sy) * c.Width + sx;
                            if (!Compatible(guide, geometry[tap], guide.CameraDepth, c))
                                continue;
                            const auto value = Wide(current[tap].Radiance);
                            for (size_t channel = 0; channel < low.size(); ++channel)
                            {
                                low[channel] = std::min(low[channel], value[channel]);
                                high[channel] = std::max(high[channel], value[channel]);
                            }
                        }
                    age = std::min(c.MaximumHistory, youngest + 1);
                    const double fresh = std::max(double(c.MinimumCurrentWeight), 1.0 / age);
                    for (size_t channel = 0; channel < color.size(); ++channel)
                    {
                        const double expansion = c.NeighborhoodExpansion * (high[channel] - low[channel]);
                        const double clipped =
                            std::clamp(history[channel] / accepted, low[channel] - expansion,
                                       high[channel] + expansion);
                        color[channel] = clipped * (1 - fresh) + center[channel] * fresh;
                    }
                }
            }
            const Float3 value{float(color[0]), float(color[1]), float(color[2])};
            if (!std::isfinite(value.X) || !std::isfinite(value.Y) || !std::isfinite(value.Z))
            {
                error = "CPU temporal reconstruction produced invalid radiance";
                return false;
            }
            pixels[index] = {value, age};
            const float scale = Maths::DisplayLinearScale(value.X, value.Y, value.Z);
            const auto encode = [&](float v) {
                return c.GammaCorrection ? Maths::LinearToSRGB8(v * scale) : Maths::LinearToUNorm8(v * scale);
            };
            auto* rgba = encoded.GetPixelsRGBA8().data() + index * Channels;
            rgba[0] = encode(value.X);
            rgba[1] = encode(value.Y);
            rgba[2] = encode(value.Z);
            rgba[3] = Opaque;
        }
    result = std::move(pixels);
    display = std::move(encoded);
    return true;
}
} // namespace Arvo::RayTracing::Cpu
