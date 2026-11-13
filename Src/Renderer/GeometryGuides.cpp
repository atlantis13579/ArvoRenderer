#include "Renderer/GeometryGuides.h"
#include <cmath>

namespace Arvo::RayTracing
{
// Geometry guides precede temporal reconstruction. Schied et al.,
// "Spatiotemporal Variance-Guided Filtering" (HPG 2017):
// https://research.nvidia.com/labs/rtr/publication/schied2017spatiotemporal/
// The current contract deliberately preserves world position, oriented normal,
// primitive identity and the background direction. It is neither motion vectors
// for deforming geometry nor SVGF itself. Every backend generates central rays
// and resolves its own native traversal results in a separate source unit.
//
// Temporal dependency plan:
// 1. Capture these guides at absolute full-film pixel centers, without jitter.
// 2. Reproject into the previous camera; validate depth, normal and visibility.
//    Background reprojection uses direction and ignores camera translation.
// 3. Reject missing/disoccluded taps, clamp compatible history to a current
//    neighborhood and track history age separately from Monte Carlo sample count.
// 4. Reset on scene/material/lighting/transport/lens/resolution changes. A changed
//    camera pose alone permits geometric reprojection. Cancellation publishes
//    neither partially reconstructed pixels nor partially updated history.
// 5. A sequence supplies explicit frame/sample identities and bounded history.
//    Distributed crops need a frame barrier and access to the previous full film;
//    an arbitrary worker's last tile is never a valid historical frame.
// Finite-aperture footprints and moving/deforming geometry need additional
// correspondence data. A central pinhole probe must not silently approve them.
bool ValidateGeometryGuideRequest(const Camera& camera, const RenderSettings& request, std::string& error)
{
    constexpr uint32_t MaximumExtent = 16384;
    const auto width = FullImageWidth(request), height = FullImageHeight(request);
    if (!request.Width || !request.Height || !width || !height || width > MaximumExtent ||
        height > MaximumExtent || request.TileX > width || request.Width > width - request.TileX ||
        request.TileY > height || request.Height > height - request.TileY)
    {
        error = "Geometry guide crop lies outside the full image";
        return false;
    }
    const auto finite = [](Float3 p) {
        return std::isfinite(p.X) && std::isfinite(p.Y) && std::isfinite(p.Z);
    };
    const Float3 forward = camera.Target - camera.Position;
    const Float3 side = Cross(forward, camera.Up);
    constexpr float MinimumFov = 1, MaximumFov = 179;
    if (!finite(camera.Position) || !finite(camera.Target) || !finite(camera.Up) || !finite(forward) ||
        !finite(side) || !(Dot(forward, forward) > 0) || !std::isfinite(Dot(forward, forward)) ||
        !(Dot(side, side) > 0) || !std::isfinite(Dot(side, side)) ||
        !std::isfinite(camera.VerticalFieldOfViewDegrees) || camera.VerticalFieldOfViewDegrees < MinimumFov ||
        camera.VerticalFieldOfViewDegrees > MaximumFov)
    {
        error = "Geometry guides require a finite, nondegenerate perspective camera";
        return false;
    }
    return ValidateCameraLens(camera.Lens, error) && ValidateCameraMotion(camera.Motion, error);
}
namespace GeometryGuideData
{
Controls PackControls(const Camera& camera, const RenderSettings& request)
{
    const Ray interval;
    return {request.Width,
            request.Height,
            FullImageWidth(request),
            FullImageHeight(request),
            request.TileX,
            request.TileY,
            0,
            request.Width * request.Height,
            camera.Position.X,
            camera.Position.Y,
            camera.Position.Z,
            camera.VerticalFieldOfViewDegrees,
            camera.Target.X,
            camera.Target.Y,
            camera.Target.Z,
            interval.MinDistance,
            camera.Up.X,
            camera.Up.Y,
            camera.Up.Z,
            interval.MaxDistance,
            {}};
}
} // namespace GeometryGuideData
namespace Detail
{
GeometryGuideResult CaptureGeometryGuides(const Scene& scene, const Camera& camera, const RenderSettings& settings,
                                          const CancellationProbe& cancellation,
                                          const GenerateGuideRays& generate, const TraceGuideRays& trace,
                                          const ResolveGeometryGuides& resolve)
{
    GeometryGuideResult result;
    if (!ValidateGeometryGuideRequest(camera, settings, result.Error))
        return result;
    const auto stopped = [&] {
        if (!cancellation || !cancellation())
            return false;
        result.Status = RenderStatus::Cancelled;
        result.Error.clear();
        return true;
    };
    if (stopped())
        return result;
    const auto controls = GeometryGuideData::PackControls(camera, settings);
    std::vector<GeometryGuideData::EncodedRay> encoded;
    if (!generate(controls, encoded, result.Error))
    {
        stopped();
        return result;
    }
    if (encoded.size() != controls.Count)
    {
        result.Error = "Native geometry guide generation returned an incomplete ray array";
        return result;
    }
    std::vector<Ray> rays;
    rays.reserve(encoded.size());
    for (const auto& ray : encoded)
    {
        if (stopped())
            return result;
        rays.push_back({{ray.OriginX, ray.OriginY, ray.OriginZ},
                        {ray.DirectionX, ray.DirectionY, ray.DirectionZ},
                        ray.Minimum,
                        ray.Maximum});
    }
    if (!ValidateRayQueries(rays, result.Error))
        return result;
    auto hits = trace(rays);
    if (hits.Status != RenderStatus::Completed)
    {
        result.Status = hits.Status;
        result.Error = std::move(hits.Error);
        return result;
    }
    if (stopped())
        return result;
    if (hits.Hits.size() != encoded.size())
    {
        result.Error = "Native geometry guide traversal returned an incomplete hit array";
        return result;
    }
    std::vector<GeometryGuide> guides;
    if (!resolve(controls, encoded, hits.Hits, guides, result.Error))
    {
        stopped();
        return result;
    }
    if (stopped())
        return result;
    if (guides.size() != encoded.size())
    {
        result.Error = "Native geometry guide resolve returned an incomplete image";
        return result;
    }
    result.Image = {settings.Width,
                    settings.Height,
                    controls.FullWidth,
                    controls.FullHeight,
                    settings.TileX,
                    settings.TileY,
                    camera.Lens.Radius == 0 && !camera.Motion.Enabled && !HasSceneMotion(scene),
                    std::move(guides)};
    result.Status = RenderStatus::Completed;
    return result;
}
} // namespace Detail
} // namespace Arvo::RayTracing
