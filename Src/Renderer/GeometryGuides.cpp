#include "Renderer/GeometryGuides.h"
#include <algorithm>
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
namespace
{
bool DiffuseOnly(const Material& material)
{
    return material.Model == MaterialModel::Diffuse && !material.SampledEmitter &&
           material.Emission.X == 0 && material.Emission.Y == 0 && material.Emission.Z == 0;
}
Float3 FootprintAlbedo(const Scene& scene, const GeometryGuide& center, const RayQueryHit* taps,
                      size_t stride)
{
    const Float3 unavailable{-1, -1, -1};
    if (!(center.Flags & GeometryGuide::Surface) || center.MaterialIndex >= scene.Materials.size())
        return unavailable;
    const auto& material = scene.Materials[center.MaterialIndex];
    if (!DiffuseOnly(material))
        return unavailable;
    Float3 sum{};
    for (size_t offset : {size_t(0), size_t(1), stride, stride + 1})
    {
        const auto& hit = taps[offset];
        if (!hit.HasHit() || hit.MaterialIndex != center.MaterialIndex)
            return unavailable;
        const Float3 texture = hit.HasTexCoord()
            ? SampleDiffuseTexture(scene, material.DiffuseTextureIndex, hit.TexCoord)
            : Float3{1, 1, 1};
        sum = sum + Float3{std::clamp(material.Albedo.X * hit.Color.X * texture.X, 0.f, 1.f),
                           std::clamp(material.Albedo.Y * hit.Color.Y * texture.Y, 0.f, 1.f),
                           std::clamp(material.Albedo.Z * hit.Color.Z * texture.Z, 0.f, 1.f)};
    }
    return sum * 0.25f;
}
} // namespace
GeometryGuideResult CaptureGeometryGuides(const Scene& scene, const Camera& camera, const RenderSettings& settings,
                                          bool spatialAlbedo, const CancellationProbe& cancellation,
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
    auto controls = GeometryGuideData::PackControls(camera, settings);
    const bool pinhole = camera.Lens.Radius == 0 && !camera.Motion.Enabled && !HasSceneMotion(scene);
    GeometryGuideImage image{settings.Width, settings.Height, controls.FullWidth, controls.FullHeight,
                             settings.TileX, settings.TileY, pinhole, {}, {}};
    const bool captureAlbedo = spatialAlbedo && pinhole && !settings.Medium.Enabled &&
                               settings.Radiometry == RadiometryMode::RGB &&
                               std::any_of(scene.Materials.begin(), scene.Materials.end(), DiffuseOnly);
    image.Pixels.reserve(controls.Count);
    if (captureAlbedo)
        image.DiffuseAlbedo.reserve(controls.Count);

    // Trace pixel centers and subpixel albedo probes together so the native backend
    // builds the acceleration structure only once per band.
    const uint32_t raysPerPixel = captureAlbedo ? 5 : 1;
    const uint32_t bandHeight = MaximumRayQueryBatch / (raysPerPixel * controls.Width);
    for (uint32_t row = 0; row < settings.Height; row += bandHeight)
    {
        controls.TileY = settings.TileY + row;
        controls.Height = std::min(bandHeight, settings.Height - row);
        controls.Count = controls.Width * controls.Height;
        std::vector<Ray> rays;
        rays.reserve(size_t(controls.Count) * raysPerPixel);
        const auto append = [&](const GeometryGuideData::Controls& window,
                                std::vector<GeometryGuideData::EncodedRay>& encoded) {
            if (stopped() || !generate(window, encoded, result.Error))
                return false;
            if (encoded.size() != window.Count)
            {
                result.Error = "Native geometry guide generation returned an incomplete ray array";
                return false;
            }
            for (const auto& ray : encoded)
                rays.push_back({{ray.OriginX, ray.OriginY, ray.OriginZ},
                                {ray.DirectionX, ray.DirectionY, ray.DirectionZ}, ray.Minimum, ray.Maximum});
            return true;
        };

        std::vector<GeometryGuideData::EncodedRay> centers;
        if (!append(controls, centers))
        {
            stopped();
            return result;
        }
        if (captureAlbedo)
        {
            // The 2x2 footprint follows absolute film coordinates, including tile halos.
            auto fine = controls;
            fine.Width *= 2;
            fine.Height *= 2;
            fine.FullWidth *= 2;
            fine.FullHeight *= 2;
            fine.TileX *= 2;
            fine.TileY *= 2;
            fine.Count = fine.Width * fine.Height;
            std::vector<GeometryGuideData::EncodedRay> subpixels;
            if (!append(fine, subpixels))
            {
                stopped();
                return result;
            }
        }
        if (stopped() || !ValidateRayQueries(rays, result.Error))
            return result;
        auto query = trace(rays);
        if (query.Status != RenderStatus::Completed)
        {
            result.Status = query.Status;
            result.Error = std::move(query.Error);
            return result;
        }
        if (query.Hits.size() != rays.size())
        {
            result.Error = "Native geometry guide traversal returned an incomplete hit array";
            return result;
        }
        const auto firstSubpixel = query.Hits.begin() + controls.Count;
        std::vector<RayQueryHit> centerHits;
        if (captureAlbedo)
            centerHits.assign(query.Hits.begin(), firstSubpixel);
        std::vector<GeometryGuide> guides;
        if (!resolve(controls, centers, captureAlbedo ? centerHits : query.Hits, guides, result.Error))
        {
            stopped();
            return result;
        }
        if (stopped())
            return result;
        if (guides.size() != controls.Count)
        {
            result.Error = "Native geometry guide resolve returned an incomplete image";
            return result;
        }
        image.Pixels.insert(image.Pixels.end(), guides.begin(), guides.end());
        if (!captureAlbedo)
            continue;
        const size_t stride = 2 * controls.Width;
        for (uint32_t y = 0; y < controls.Height; ++y)
        {
            if (stopped())
                return result;
            for (uint32_t x = 0; x < controls.Width; ++x)
            {
                const auto& center = guides[size_t(y) * controls.Width + x];
                const size_t offset = controls.Count + 2 * y * stride + 2 * x;
                image.DiffuseAlbedo.push_back(FootprintAlbedo(scene, center, &query.Hits[offset], stride));
            }
        }
    }
    result.Image = std::move(image);
    result.Status = RenderStatus::Completed;
    return result;
}
} // namespace Detail
} // namespace Arvo::RayTracing
