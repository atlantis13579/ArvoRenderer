#include "Renderer/TemporalReconstruction.h"
#include "Renderer/Reconstruction.h"
#include <cmath>

namespace Arvo::RayTracing
{
// Temporal reconstruction depends on completed film, deterministic geometric
// guides and an explicitly owned sequence. Schied et al., "Spatiotemporal
// Variance-Guided Filtering" (HPG 2017) motivates geometric reprojection:
// https://research.nvidia.com/labs/rtr/publication/schied2017spatiotemporal/
// This component implements bounded, neighborhood-clipped temporal radiance;
// it does not implement SVGF's variance estimation or hierarchical wavelets.
// Transport moments remain immutable and no independence claim is attached to
// frame age. The caller publishes History only with the accepted output frame.
// A distributed owner must first assemble the full film; workers cannot invent
// history from arbitrary tile order. Deformation and finite apertures need more
// correspondence data than the static, pinhole guides available here.
bool ValidateTemporalOptions(const TemporalOptions& o, std::string& error)
{
    constexpr uint32_t MaximumFrames = 256;
    constexpr float MaximumWorldTolerance = 10000, MaximumRelativeTolerance = 0.25f;
    const auto unit = [](float v) { return std::isfinite(v) && v >= 0 && v <= 1; };
    if (!o.MaximumHistory || o.MaximumHistory > MaximumFrames || !unit(o.MinimumCurrentWeight) ||
        !unit(o.MinimumNormalCosine) || !unit(o.MinimumHistoryWeight) || o.MinimumHistoryWeight == 0 ||
        !unit(o.NeighborhoodExpansion) || !std::isfinite(o.AbsoluteDepthTolerance) ||
        o.AbsoluteDepthTolerance < 0 || o.AbsoluteDepthTolerance > MaximumWorldTolerance ||
        !std::isfinite(o.RelativeDepthTolerance) || o.RelativeDepthTolerance < 0 ||
        o.RelativeDepthTolerance > MaximumRelativeTolerance)
    {
        error = "Invalid temporal history, geometry tolerance or clipping controls";
        return false;
    }
    return true;
}
TemporalResult IRayTracingRHI::ReconstructTemporal(const TemporalFrame&, const TemporalHistory*,
                                                   const TemporalOptions&,
                                                   const CancellationToken& cancellation) const
{
    TemporalResult result;
    if (cancellation.IsCancellationRequested())
        result.Status = RenderStatus::Cancelled;
    else
        result.Error = "Temporal reconstruction is not implemented by this backend";
    return result;
}
namespace
{
bool Finite(Float3 p)
{
    return std::isfinite(p.X) && std::isfinite(p.Y) && std::isfinite(p.Z);
}
bool GuidesValid(const GeometryGuideImage& image, const Camera& view, std::string& error)
{
    RenderSettings settings;
    settings.Width = image.Width;
    settings.Height = image.Height;
    if (!image.PinholeCorrespondence || view.Lens.Radius != 0 || view.Motion.Enabled || image.TileX ||
        image.TileY || image.Width != image.FullWidth || image.Height != image.FullHeight ||
        uint64_t(image.Width) * image.Height != image.Pixels.size() ||
        !ValidateGeometryGuideRequest(view, settings, error))
    {
        if (error.empty())
            error = "Temporal reconstruction requires complete instantaneous pinhole geometry guides";
        return false;
    }
    constexpr float UnitTolerance = 0.001f;
    for (const auto& p : image.Pixels)
    {
        const bool surface = (p.Flags & GeometryGuide::Surface) != 0;
        const uint32_t allowed =
            surface ? GeometryGuide::Surface | GeometryGuide::FrontFace : GeometryGuide::Background;
        if ((p.Flags & ~allowed) || !(p.Flags & allowed) || !Finite(p.Direction) ||
            std::abs(Dot(p.Direction, p.Direction) - 1) > UnitTolerance ||
            (surface &&
             (!Finite(p.Position) || !Finite(p.GeometricNormal) ||
              std::abs(Dot(p.GeometricNormal, p.GeometricNormal) - 1) > UnitTolerance ||
              !std::isfinite(p.CameraDepth) || !(p.CameraDepth > 0) || !std::isfinite(p.RayDistance) ||
              !(p.RayDistance > 0) || p.PrimitiveIndex == InvalidRayPrimitiveIndex)) ||
            (!surface &&
             (p.PrimitiveIndex != InvalidRayPrimitiveIndex || p.CameraDepth != 0 || p.RayDistance != 0)))
        {
            error = "Invalid geometric correspondence in temporal input";
            return false;
        }
    }
    return true;
}
TemporalReset ResetReason(BackendType backend, const TemporalFrame& current, const TemporalHistory* old,
                          const TemporalOptions& options)
{
    if (!old)
        return TemporalReset::NoHistory;
    if (old->Identity.Sequence != current.Identity.Sequence)
        return TemporalReset::Sequence;
    if (old->Identity.Content != current.Identity.Content)
        return TemporalReset::Content;
    if (old->Identity.Frame + 1 != current.Identity.Frame)
        return TemporalReset::FrameGap;
    if (old->Identity.Samples == current.Identity.Samples)
        return TemporalReset::RepeatedSamples;
    if (old->Backend != backend)
        return TemporalReset::Backend;
    if (!(old->Options == options))
        return TemporalReset::Options;
    if (old->View.VerticalFieldOfViewDegrees != current.View.VerticalFieldOfViewDegrees ||
        old->View.Lens != current.View.Lens || old->View.Motion != current.View.Motion)
        return TemporalReset::Projection;
    if (old->Geometry.Width != current.Film.Width || old->Geometry.Height != current.Film.Height)
        return TemporalReset::Resolution;
    return TemporalReset::None;
}
} // namespace
namespace Detail
{
bool ValidateTemporalRequest(BackendType backend, const TemporalFrame& frame, const TemporalHistory* old,
                             const TemporalOptions& options, TemporalReset& reset, std::string& error)
{
    if (!ValidateTemporalOptions(options, error) ||
        !ValidateReconstructionInput(frame.Film, ReconstructionOptions{}, error) ||
        !GuidesValid(frame.Geometry, frame.View, error))
        return false;
    if (!frame.Identity.Sequence || frame.Geometry.Width != frame.Film.Width ||
        frame.Geometry.Height != frame.Film.Height)
    {
        error = "Temporal input requires a sequence identity and matching full-film guides";
        return false;
    }
    if (old && old->Identity.Sequence == frame.Identity.Sequence &&
        old->Identity.Frame >= frame.Identity.Frame)
    {
        error = "Temporal frames must advance monotonically within their sequence";
        return false;
    }
    reset = ResetReason(backend, frame, old, options);
    const bool reuse = reset == TemporalReset::None;
    if (reuse)
    {
        if (!GuidesValid(old->Geometry, old->View, error) ||
            old->Pixels.size() != frame.Film.Pixels.size())
        {
            if (error.empty())
                error = "Temporal history extent mismatch";
            return false;
        }
        for (const auto& p : old->Pixels)
            if (!Finite(p.Radiance) || !p.Age ||
                p.Age > options.MaximumHistory)
            {
                error = "Temporal history contains invalid radiance or frame age";
                return false;
            }
    }
    return true;
}
TemporalResult ReconstructTemporal(BackendType backend, const TemporalFrame& frame,
                                   const TemporalHistory* old, const TemporalOptions& options,
                                   const CancellationProbe& cancelled,
                                   const NativeTemporalReconstruction& run)
{
    TemporalResult result;
    const auto stopped = [&] {
        if (!cancelled || !cancelled()) return false;
        result = {};
        result.Status = RenderStatus::Cancelled;
        return true;
    };
    if (stopped() || !ValidateTemporalRequest(backend, frame, old, options, result.Reset, result.Error))
        return result;
    const bool reuse = result.Reset == TemporalReset::None;
    std::vector<TemporalPixel> raw;
    raw.reserve(frame.Film.Pixels.size());
    for (const auto& p : frame.Film.Pixels)
    {
        if (stopped())
            return result;
        raw.push_back({p.MeanRadiance, 0});
    }
    const Camera& previous = reuse ? old->View : frame.View;
    TemporalData::Controls c{frame.Film.Width,
                             frame.Film.Height,
                             0,
                             uint32_t(raw.size()),
                             uint32_t(reuse),
                             options.MaximumHistory,
                             uint32_t(options.GammaCorrection),
                             0,
                             options.MinimumCurrentWeight,
                             options.AbsoluteDepthTolerance,
                             options.RelativeDepthTolerance,
                             options.MinimumNormalCosine,
                             options.MinimumHistoryWeight,
                             options.NeighborhoodExpansion,
                             {},
                             previous.Position.X,
                             previous.Position.Y,
                             previous.Position.Z,
                             previous.VerticalFieldOfViewDegrees,
                             previous.Target.X,
                             previous.Target.Y,
                             previous.Target.Z,
                             0,
                             previous.Up.X,
                             previous.Up.Y,
                             previous.Up.Z,
                             0};
    const std::vector<TemporalPixel> noPixels;
    const std::vector<GeometryGuide> noGuides;
    std::vector<TemporalPixel> filtered;
    Image display;
    if (!run(c, raw, frame.Geometry.Pixels, reuse ? old->Pixels : noPixels,
             reuse ? old->Geometry.Pixels : noGuides, filtered, display, result.Error))
    {
        stopped();
        return result;
    }
    if (stopped())
        return result;
    if (filtered.size() != raw.size() || display.GetWidth() != c.Width || display.GetHeight() != c.Height)
    {
        result.Error = "Native temporal reconstruction returned an incomplete frame";
        return result;
    }
    uint64_t accepted = 0;
    for (const auto& p : filtered)
    {
        if (stopped())
            return result;
        if (!Finite(p.Radiance) || !p.Age ||
            p.Age > c.MaximumHistory)
        {
            result.Error = "Native temporal reconstruction returned invalid radiance or age";
            return result;
        }
        accepted += p.Age > 1;
    }
    TemporalHistory next{frame.Identity, frame.View, options, backend, frame.Geometry, std::move(filtered)};
    if (stopped())
        return result;
    result.History = std::move(next);
    result.Display = std::move(display);
    result.AcceptedHistoryPixels = accepted;
    result.Status = RenderStatus::Completed;
    return result;
}
} // namespace Detail
} // namespace Arvo::RayTracing
