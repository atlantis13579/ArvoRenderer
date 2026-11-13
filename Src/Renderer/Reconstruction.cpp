#include "Renderer/Reconstruction.h"
#include <cmath>
#include <chrono>
#include <algorithm>

namespace Arvo::RayTracing
{
namespace
{
constexpr uint32_t MaximumPasses = 5;
constexpr float MaximumNoiseScale = 16, MaximumRelativeScale = 4;
constexpr float MinimumAbsoluteScale = 1e-6f, MaximumAbsoluteScale = 10000;
constexpr float MinimumGeometryScale = 1e-6f, MaximumNormalScale = 2, MaximumPlaneScale = 1;
constexpr uint32_t StencilRadius = 2;
constexpr uint32_t MaximumExtent = 16384;
} // namespace
void ReconstructionData::PackGuide(Pixel &target, const GeometryGuide &source)
{
    target.GuideFlags = source.Flags;
    target.GuideMaterial = source.MaterialIndex;
    target.GuidePosition = source.Position;
    target.GuideDepth = source.CameraDepth;
    target.GuideNormal = source.GeometricNormal;
}
const char* ReconstructionFilterKey(ReconstructionFilter filter)
{
    switch (filter)
    {
    case ReconstructionFilter::None:
        return "none";
    case ReconstructionFilter::SpatialAtrous:
        return "spatial";
    case ReconstructionFilter::Temporal:
        return "temporal";
    case ReconstructionFilter::Ai:
        return "ai";
    default:
        return nullptr;
    }
}
bool ParseReconstructionFilter(std::string_view key, ReconstructionFilter& filter)
{
    for (auto candidate : {ReconstructionFilter::None, ReconstructionFilter::SpatialAtrous,
                           ReconstructionFilter::Temporal, ReconstructionFilter::Ai})
        if (key == ReconstructionFilterKey(candidate))
        {
            filter = candidate;
            return true;
        }
    return false;
}
bool ValidateReconstructionOptions(const ReconstructionOptions& options, std::string& error)
{
    if (options.Filter >= ReconstructionFilter::Count)
        error = "Unknown reconstruction filter";
    else if (options.Filter == ReconstructionFilter::Temporal)
        error = "Temporal denoising requires reprojection and validated frame history";
    else if (options.Filter == ReconstructionFilter::Ai)
        error = "AI denoising requires a supplied model and native runtime";
    else if (!options.Passes || options.Passes > MaximumPasses)
        error = "Spatial denoising requires between 1 and 5 passes";
    else if (!std::isfinite(options.NoiseScale) || options.NoiseScale < 0 ||
             options.NoiseScale > MaximumNoiseScale || !std::isfinite(options.RelativeColorScale) ||
             options.RelativeColorScale < 0 || options.RelativeColorScale > MaximumRelativeScale ||
             !std::isfinite(options.AbsoluteColorScale) ||
             options.AbsoluteColorScale < MinimumAbsoluteScale ||
             options.AbsoluteColorScale > MaximumAbsoluteScale)
        error = "Invalid spatial denoising noise or color scale";
    else if (!std::isfinite(options.NormalScale) || options.NormalScale < MinimumGeometryScale ||
             options.NormalScale > MaximumNormalScale || !std::isfinite(options.RelativePlaneScale) ||
             options.RelativePlaneScale < MinimumGeometryScale ||
             options.RelativePlaneScale > MaximumPlaneScale)
        error = "Invalid spatial denoising normal or relative plane scale";
    else
        return true;
    return false;
}
uint32_t ReconstructionHalo(const ReconstructionOptions& options)
{
    // Validation precedes scheduling; an invalid shift must never overflow.
    if (options.Filter != ReconstructionFilter::SpatialAtrous || !options.Passes ||
        options.Passes > MaximumPasses)
        return 0;
    return StencilRadius * ((uint32_t(1) << options.Passes) - 1);
}
bool ReconstructionUsesMoments(Algorithm algorithm)
{
    return algorithm != Algorithm::MetropolisLightTransport &&
           algorithm != Algorithm::ProgressivePhotonMapping &&
           algorithm != Algorithm::VertexConnectionMerging && algorithm < Algorithm::Count;
}
bool ValidateReconstructionInput(const LinearFilm& film, const ReconstructionOptions& options,
                                 std::string& error)
{
    if (!ValidateReconstructionOptions(options, error))
        return false;
    if (!film.Width || !film.Height || film.Width > MaximumExtent || film.Height > MaximumExtent ||
        uint64_t(film.Width) * film.Height != film.Pixels.size())
    {
        error = "Reconstruction requires a complete rectangular linear film";
        return false;
    }
    for (const auto& pixel : film.Pixels)
        if (!pixel.SampleCount || !std::isfinite(pixel.LuminanceM2) || pixel.LuminanceM2 < 0 ||
            !std::isfinite(pixel.MeanRadiance.X) || !std::isfinite(pixel.MeanRadiance.Y) ||
            !std::isfinite(pixel.MeanRadiance.Z))
        {
            error = "Reconstruction input contains incomplete samples or invalid raw moments/radiance";
            return false;
        }
    return true;
}
bool ValidateReconstructionSupport(const Camera &camera, bool sceneMotion,
                                   const ReconstructionOptions &options, const MediumOptions &medium,
                                   std::string &error)
{
    if (!ValidateReconstructionOptions(options, error))
        return false;
    if (options.Filter == ReconstructionFilter::SpatialAtrous && options.GeometryGuidance &&
        (camera.Lens.Radius != 0 || camera.Motion.Enabled || sceneMotion || medium.Enabled))
    {
        error =
            "Geometry-guided spatial denoising requires static pinhole surfaces without participating media";
        return false;
    }
    return true;
}
bool ValidateReconstructionGuides(const LinearFilm &film, const ReconstructionOptions &options,
                                  const GeometryGuideImage *guides, std::string &error)
{
    if (!options.GeometryGuidance || options.Filter != ReconstructionFilter::SpatialAtrous)
        return true;
    if (!guides || !guides->PinholeCorrespondence || guides->Width != film.Width ||
        guides->Height != film.Height || guides->Pixels.size() != film.Pixels.size() ||
        guides->TileX > guides->FullWidth || film.Width > guides->FullWidth - guides->TileX ||
        guides->TileY > guides->FullHeight || film.Height > guides->FullHeight - guides->TileY)
    {
        error = "Spatial reconstruction requires matching static pinhole geometry guides";
        return false;
    }
    constexpr double NormalLengthTolerance = 0.001;
    constexpr uint32_t Classification = GeometryGuide::Surface | GeometryGuide::Background;
    constexpr uint32_t KnownFlags = Classification | GeometryGuide::FrontFace;
    const auto finite = [](Float3 v)
    { return std::isfinite(v.X) && std::isfinite(v.Y) && std::isfinite(v.Z); };
    for (const auto &guide : guides->Pixels)
    {
        const auto kind = guide.Flags & Classification;
        const auto n = guide.GeometricNormal;
        const double lengthSquared = double(n.X) * n.X + double(n.Y) * n.Y + double(n.Z) * n.Z;
        if ((guide.Flags & ~KnownFlags) ||
            (kind != GeometryGuide::Surface && kind != GeometryGuide::Background) ||
            (kind == GeometryGuide::Surface &&
             (!finite(guide.Position) || !finite(n) || !std::isfinite(guide.CameraDepth) ||
              guide.CameraDepth <= 0 || std::abs(lengthSquared - 1) > NormalLengthTolerance)))
        {
            error = "Spatial geometry guides contain an invalid classification, depth, position or normal";
            return false;
        }
    }
    return true;
}
namespace Detail
{
const GeometryGuideImage *SpatialGuideCache::Get(const RenderSettings &raster,
                                                 const std::function<GeometryGuideResult()> &capture,
                                                 std::string &error)
{
    if (Image.Width != raster.Width || Image.Height != raster.Height ||
        Image.FullWidth != FullImageWidth(raster) || Image.FullHeight != FullImageHeight(raster) ||
        Image.TileX != raster.TileX || Image.TileY != raster.TileY || Image.Pixels.empty())
    {
        auto result = capture();
        if (result.Status != RenderStatus::Completed)
        {
            error = result.Error.empty() ? "Native spatial guide capture did not complete" : result.Error;
            return nullptr;
        }
        Image = std::move(result.Image);
        if (Image.Width != raster.Width || Image.Height != raster.Height ||
            Image.FullWidth != FullImageWidth(raster) || Image.FullHeight != FullImageHeight(raster) ||
            Image.TileX != raster.TileX || Image.TileY != raster.TileY)
        {
            Image = {};
            error = "Native spatial guides do not describe the reconstruction raster";
            return nullptr;
        }
    }
    return &Image;
}
RenderResult RenderWithReconstruction(const RenderSettings& request, const CancellationProbe& cancelled,
                                      const RawRenderFunction& render,
                                      const ReconstructionFunction& reconstruct)
{
    std::string error;
    if (!ValidateReconstructionOptions(request.Reconstruction, error))
        return MakeFailedResult(error);
    if (request.Reconstruction.Filter == ReconstructionFilter::None)
        return render(request);
    const uint32_t fullWidth = FullImageWidth(request), fullHeight = FullImageHeight(request);
    if (!request.Width || !request.Height || !fullWidth || !fullHeight || fullWidth > MaximumExtent ||
        fullHeight > MaximumExtent || request.TileX > fullWidth ||
        request.Width > fullWidth - request.TileX || request.TileY > fullHeight ||
        request.Height > fullHeight - request.TileY)
        return MakeFailedResult("Reconstruction crop lies outside the full image");
    if (cancelled && cancelled())
        return MakeCancelledResult();
    const uint32_t halo = ReconstructionHalo(request.Reconstruction);
    RenderSettings expanded = request;
    expanded.FullWidth = fullWidth;
    expanded.FullHeight = fullHeight;
    expanded.TileX = request.TileX > halo ? request.TileX - halo : 0;
    expanded.TileY = request.TileY > halo ? request.TileY - halo : 0;
    expanded.Width = std::min(fullWidth, request.TileX + request.Width + halo) - expanded.TileX;
    expanded.Height = std::min(fullHeight, request.TileY + request.Height + halo) - expanded.TileY;
    // Reconstruction is scheduled exactly once outside the transport invocation.
    expanded.Reconstruction.Filter = ReconstructionFilter::None;
    auto raw = render(expanded);
    if (raw.Status != RenderStatus::Completed)
        return raw;
    if (cancelled && cancelled())
        return MakeCancelledResult();
    if (raw.Film.Width != expanded.Width || raw.Film.Height != expanded.Height ||
        !ValidateReconstructionInput(raw.Film, request.Reconstruction, error))
        return MakeFailedResult(error.empty() ? "Transport returned incomplete reconstruction film" : error);
    const auto begin = std::chrono::steady_clock::now();
    ReconstructedImage image;
    if (!reconstruct(expanded, raw.Film, image, error))
        return cancelled && cancelled() ? MakeCancelledResult() : MakeFailedResult(error);
    if (image.Width != expanded.Width || image.Height != expanded.Height ||
        image.Pixels.size() != raw.Film.Pixels.size() || image.Display.GetWidth() != expanded.Width ||
        image.Display.GetHeight() != expanded.Height)
        return MakeFailedResult("Native reconstruction returned incomplete image data");
    RenderResult result;
    result.Film = LinearFilm(request.Width, request.Height);
    result.ReconstructedRadiance.resize(result.Film.Pixels.size());
    result.Output = Image(request.Width, request.Height);
    const uint32_t offsetX = request.TileX - expanded.TileX, offsetY = request.TileY - expanded.TileY;
    constexpr size_t Channels = 4;
    for (uint32_t row = 0; row < request.Height; ++row)
    {
        if (cancelled && cancelled())
            return MakeCancelledResult();
        const size_t from = size_t(row + offsetY) * expanded.Width + offsetX;
        const size_t to = size_t(row) * request.Width;
        std::copy_n(raw.Film.Pixels.begin() + from, request.Width, result.Film.Pixels.begin() + to);
        std::copy_n(image.Pixels.begin() + from, request.Width, result.ReconstructedRadiance.begin() + to);
        std::copy_n(image.Display.GetPixelsRGBA8().data() + from * Channels, size_t(request.Width) * Channels,
                    result.Output.GetPixelsRGBA8().data() + to * Channels);
    }
    // Actual ray counters include halo work; CompletedPixels describes delivered pixels.
    result.Statistics = raw.Statistics;
    result.Statistics.CompletedPixels = uint64_t(request.Width) * request.Height;
    result.Statistics.RenderMilliseconds +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
    result.Status = RenderStatus::Completed;
    return result;
}
} // namespace Detail
} // namespace Arvo::RayTracing
