#include "GeometryGuides.h"
#include <algorithm>
#include <cmath>

namespace Arvo::RayTracing::Cpu
{
namespace
{
constexpr double Pi = 3.14159265358979323846, DegreesPerHalfTurn = 180, PixelCenter = 0.5;
Float3 Unit(Float3 value)
{
    const float scale = std::max({std::abs(value.X), std::abs(value.Y), std::abs(value.Z)});
    if (!(scale > 0))
        return {};
    value = value / scale;
    return value / std::sqrt(Dot(value, value));
}
Float3 Forward(const GeometryGuideData::Controls& c)
{
    return Unit(Float3{c.TargetX - c.PositionX, c.TargetY - c.PositionY, c.TargetZ - c.PositionZ});
}
bool Finite(Float3 p)
{
    return std::isfinite(p.X) && std::isfinite(p.Y) && std::isfinite(p.Z);
}
} // namespace
// Independent central camera probes. They establish geometric correspondence,
// not a sampled optical path, visibility probability or transport contribution.
bool GenerateGeometryGuideRays(const GeometryGuideData::Controls& c,
                               const Detail::CancellationProbe& cancelled,
                               std::vector<GeometryGuideData::EncodedRay>& result, std::string& error)
{
    const Float3 forward = Forward(c), right = Unit(Cross(forward, {c.UpX, c.UpY, c.UpZ}));
    const Float3 up = Cross(right, forward);
    const double halfHeight = std::tan(double(c.FieldOfView) * Pi / (2 * DegreesPerHalfTurn));
    const double halfWidth = halfHeight * c.FullWidth / c.FullHeight;
    std::vector<GeometryGuideData::EncodedRay> rays(size_t(c.Width) * c.Height);
    for (uint32_t y = 0; y < c.Height; ++y)
        for (uint32_t x = 0; x < c.Width; ++x)
        {
            if (cancelled && cancelled())
            {
                error = "CPU geometry guide generation cancelled";
                return false;
            }
            const double screenX = (2 * (double(c.TileX + x) + PixelCenter) / c.FullWidth - 1) * halfWidth;
            const double screenY = (1 - 2 * (double(c.TileY + y) + PixelCenter) / c.FullHeight) * halfHeight;
            const Float3 direction = Unit(forward + right * float(screenX) + up * float(screenY));
            if (!Finite(direction) || !(Dot(direction, direction) > 0))
            {
                error = "CPU geometry guide direction is invalid";
                return false;
            }
            rays[size_t(y) * c.Width + x] = {c.PositionX, c.PositionY, c.PositionZ, c.MinimumDistance,
                                             direction.X, direction.Y, direction.Z, c.MaximumDistance};
        }
    result = std::move(rays);
    return true;
}
bool ResolveGeometryGuideHits(const GeometryGuideData::Controls& c,
                              const std::vector<GeometryGuideData::EncodedRay>& rays,
                              const std::vector<RayQueryHit>& hits,
                              const Detail::CancellationProbe& cancelled, std::vector<GeometryGuide>& result,
                              std::string& error)
{
    if (rays.size() != size_t(c.Width) * c.Height || hits.size() != rays.size())
    {
        error = "CPU geometry guide resolve extent mismatch";
        return false;
    }
    const Float3 forward = Forward(c);
    std::vector<GeometryGuide> guides(rays.size());
    for (size_t i = 0; i < rays.size(); ++i)
    {
        if (cancelled && cancelled())
        {
            error = "CPU geometry guide resolve cancelled";
            return false;
        }
        const auto& ray = rays[i];
        const auto& hit = hits[i];
        auto& guide = guides[i];
        guide.Direction = {ray.DirectionX, ray.DirectionY, ray.DirectionZ};
        guide.Flags = GeometryGuide::Background;
        if (!hit.HasHit())
            continue;
        guide.Position = {float(double(ray.OriginX) + double(ray.DirectionX) * hit.Distance),
                          float(double(ray.OriginY) + double(ray.DirectionY) * hit.Distance),
                          float(double(ray.OriginZ) + double(ray.DirectionZ) * hit.Distance)};
        guide.CameraDepth = hit.Distance * Dot(guide.Direction, forward);
        guide.RayDistance = hit.Distance;
        guide.GeometricNormal = Unit(hit.GeometricNormal);
        if (!std::isfinite(hit.Distance) || hit.Distance <= ray.Minimum || hit.Distance >= ray.Maximum ||
            !Finite(guide.Position) || !Finite(guide.GeometricNormal) ||
            !(Dot(guide.GeometricNormal, guide.GeometricNormal) > 0) || !(guide.CameraDepth > 0) ||
            !std::isfinite(guide.CameraDepth) || hit.PrimitiveIndex == InvalidRayPrimitiveIndex)
        {
            error = "CPU geometry guide hit is invalid";
            return false;
        }
        guide.PrimitiveIndex = hit.PrimitiveIndex;
        guide.MaterialIndex = hit.MaterialIndex;
        guide.Flags = GeometryGuide::Surface | (hit.IsFrontFace() ? GeometryGuide::FrontFace : 0);
    }
    result = std::move(guides);
    return true;
}
} // namespace Arvo::RayTracing::Cpu
