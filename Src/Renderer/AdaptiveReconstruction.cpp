#include "Renderer/AdaptiveReconstruction.h"
#include <algorithm>
#include <chrono>

namespace Arvo::RayTracing
{
bool ValidateCameraSamplePlan(const Camera& camera, const RenderSettings& settings, std::string& error)
{
    if (!settings.ProductionBudgets)
        return true;
    const auto& original = settings.ProductionBudgets->View;
    const auto same = [](Float3 a, Float3 b) { return a.X == b.X && a.Y == b.Y && a.Z == b.Z; };
    if (!same(camera.Position, original.Position) || !same(camera.Target, original.Target) ||
        !same(camera.Up, original.Up) ||
        camera.VerticalFieldOfViewDegrees != original.VerticalFieldOfViewDegrees ||
        camera.Lens != original.Lens || camera.Motion != original.Motion)
    {
        error = "Production budget map belongs to a different camera or exposure";
        return false;
    }
    return true;
}
bool ValidateSamplePlan(const RenderSettings& settings, std::string& error)
{
    const bool camera = settings.Integrator == Algorithm::RayTracing ||
                        settings.Integrator == Algorithm::PathTracing ||
                        settings.Integrator == Algorithm::DistributedRayTracing;
    const bool pilot = settings.SampleStream == CameraSampleStream::DenoisePilotA ||
                       settings.SampleStream == CameraSampleStream::DenoisePilotB;
    if (settings.SampleStream != CameraSampleStream::Production && !pilot)
        error = "Unknown camera sampling stream";
    else if (pilot && (!camera || settings.Adaptive.Enabled ||
                       settings.Sampling != SampleSequence::Independent || settings.SamplesPerPixel < 2))
        error = "Denoising pilots require fixed independent camera samples";
    else if (settings.Adaptive.DenoiserAware &&
             (!settings.Adaptive.Enabled ||
              settings.Reconstruction.Filter != ReconstructionFilter::SpatialAtrous))
        error = "Denoiser-aware allocation requires adaptive sampling and spatial reconstruction";
    else if (settings.ProductionBudgets)
    {
        const auto& map = *settings.ProductionBudgets;
        if (!camera || pilot || !settings.Adaptive.Enabled || settings.Adaptive.DenoiserAware ||
            settings.Reconstruction.Filter != ReconstructionFilter::None || map.Width != settings.Width ||
            map.Height != settings.Height || map.FullWidth != FullImageWidth(settings) ||
            map.FullHeight != FullImageHeight(settings) || map.TileX != settings.TileX ||
            map.TileY != settings.TileY || map.SamplingSeed != settings.SamplingSeed ||
            map.Integrator != settings.Integrator || map.Radiometry != settings.Radiometry ||
            map.Counts.size() != size_t(settings.Width) * settings.Height)
            error = "Production budget map belongs to another camera observation";
        else if (std::any_of(map.Counts.begin(), map.Counts.end(), [&](uint32_t count) {
                     return count < settings.Adaptive.MinimumSamples || count > settings.SamplesPerPixel;
                 }))
            error = "Production budget map exceeds the requested sample limits";
        else
            return true;
    }
    else
        return true;
    return false;
}
namespace Detail
{
RenderResult RenderWithDenoisedSampling(const Scene &scene, const Camera &camera,
                                        const RenderSettings &request, const CancellationProbe &cancelled,
                                        const RawRenderFunction &render,
                                        const ReconstructionFunction &reconstruct,
                                        const AdaptiveAllocationFunction &allocate)
{
    std::string error;
    if (!ValidateTileSettings(request, error) ||
        !ValidateReconstructionSupport(camera, HasSceneMotion(scene), request.Reconstruction, request.Medium,
                                       error))
        return MakeFailedResult(error);
    if (!request.Adaptive.DenoiserAware)
        return RenderWithReconstruction(request, cancelled, render, reconstruct);
    if (!allocate)
        return MakeFailedResult("Native denoiser-aware allocation is unavailable");

    // The outer halo supports final reconstruction. Each pilot gets another
    // filter halo, so its error estimate is invariant under tile boundaries.
    return RenderWithReconstruction(
        request, cancelled,
        [&](const RenderSettings& expanded) {
            RenderResult pilots[2];
            const CameraSampleStream streams[]{CameraSampleStream::DenoisePilotA,
                                               CameraSampleStream::DenoisePilotB};
            for (size_t index = 0; index < std::size(pilots); ++index)
            {
                if (cancelled && cancelled())
                    return MakeCancelledResult();
                auto pilot = expanded;
                pilot.Adaptive.Enabled = pilot.Adaptive.DenoiserAware = false;
                pilot.ProductionBudgets.reset();
                pilot.SampleStream = streams[index];
                pilot.Sampling = SampleSequence::Independent;
                pilot.SamplesPerPixel = request.Adaptive.PilotSamples;
                pilot.Reconstruction = request.Reconstruction;
                pilots[index] = RenderWithReconstruction(pilot, cancelled, render, reconstruct);
                if (pilots[index].Status != RenderStatus::Completed)
                    return std::move(pilots[index]);
            }
            const auto begin = std::chrono::steady_clock::now();
            const size_t pixels = size_t(expanded.Width) * expanded.Height;
            std::vector<AdaptiveReconstructionData::Pixel> pairs;
            pairs.reserve(pixels);
            for (size_t index = 0; index < pixels; ++index)
            {
                if (cancelled && cancelled())
                    return MakeCancelledResult();
                const auto& a = pilots[0].Film.Pixels[index];
                const auto& b = pilots[1].Film.Pixels[index];
                if (a.SampleCount != request.Adaptive.PilotSamples || b.SampleCount != a.SampleCount)
                    return MakeFailedResult("Native transport returned an incomplete allocation pilot");
                pairs.push_back({a.MeanRadiance, a.LuminanceM2, b.MeanRadiance, b.LuminanceM2,
                                 pilots[0].ReconstructedRadiance[index], a.SampleCount,
                                 pilots[1].ReconstructedRadiance[index], b.SampleCount});
            }
            // Packed pairs now own the allocation inputs. Drop both full pilot
            // images before the native upload to limit peak host memory.
            RenderStatistics pilotStatistics;
            for (auto& pilot : pilots)
            {
                pilotStatistics.PrimaryRays += pilot.Statistics.PrimaryRays;
                pilotStatistics.ShadowRays += pilot.Statistics.ShadowRays;
                pilotStatistics.ReflectionRays += pilot.Statistics.ReflectionRays;
                pilotStatistics.PrimitiveTests += pilot.Statistics.PrimitiveTests;
                pilotStatistics.BuildMilliseconds += pilot.Statistics.BuildMilliseconds;
                pilotStatistics.RenderMilliseconds += pilot.Statistics.RenderMilliseconds;
                pilot = {};
            }
            const auto& adaptive = request.Adaptive;
            AdaptiveReconstructionData::Controls controls{0,
                                                          uint32_t(pixels),
                                                          adaptive.MinimumSamples,
                                                          request.SamplesPerPixel,
                                                          adaptive.RelativeError,
                                                          adaptive.AbsoluteError,
                                                          adaptive.MinimumVarianceFraction,
                                                          uint32_t(request.Sampling)};
            auto budgets = std::make_shared<SampleBudgetImage>();
            budgets->View = camera;
            budgets->Width = expanded.Width;
            budgets->Height = expanded.Height;
            budgets->FullWidth = FullImageWidth(expanded);
            budgets->FullHeight = FullImageHeight(expanded);
            budgets->TileX = expanded.TileX;
            budgets->TileY = expanded.TileY;
            budgets->Integrator = expanded.Integrator;
            budgets->Radiometry = expanded.Radiometry;
            budgets->SamplingSeed = expanded.SamplingSeed;
            if (!allocate(pairs, controls, budgets->Counts, error))
                return cancelled && cancelled() ? MakeCancelledResult() : MakeFailedResult(error);
            if (cancelled && cancelled())
                return MakeCancelledResult();
            auto production = expanded;
            production.Adaptive.DenoiserAware = false;
            production.ProductionBudgets = std::move(budgets);
            if (!ValidateSamplePlan(production, error))
                return MakeFailedResult(error);
            const double allocationMilliseconds =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
            // Pilot moments are deliberately absent from the production film.
            std::vector<AdaptiveReconstructionData::Pixel>().swap(pairs);
            auto result = render(production);
            result.Statistics.PrimaryRays += pilotStatistics.PrimaryRays;
            result.Statistics.PilotRays += pilotStatistics.PrimaryRays;
            result.Statistics.ShadowRays += pilotStatistics.ShadowRays;
            result.Statistics.ReflectionRays += pilotStatistics.ReflectionRays;
            result.Statistics.PrimitiveTests += pilotStatistics.PrimitiveTests;
            result.Statistics.BuildMilliseconds += pilotStatistics.BuildMilliseconds;
            result.Statistics.RenderMilliseconds +=
                pilotStatistics.RenderMilliseconds + allocationMilliseconds;
            return result;
        },
        reconstruct);
}
} // namespace Detail
} // namespace Arvo::RayTracing
