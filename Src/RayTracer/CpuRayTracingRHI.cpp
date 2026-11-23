#include "CpuRayTracingRHI.h"

#include "Renderer/RayTracingBackendCommon.h"
#include "TracingCore.h"
#include "RayTracer/Backends/CPU/CpuTransport.h"
#include "RayTracer/Backends/CPU/SpatialDenoise.h"
#include "RayTracer/Backends/CPU/GeometryGuides.h"
#include "RayTracer/Backends/CPU/TemporalDenoise.h"
#include "RayTracer/Backends/CPU/AdaptiveDenoise.h"
#include "RayTracer/Backends/CPU/SpectralTransport.h"
#include "RayTracer/Backends/CPU/MicropolygonDicing.h"
#include "Renderer/MicropolygonPreparation.h"

#include <utility>

namespace Arvo
{
namespace RayTracing
{
    struct CpuRayTracingRHI::Impl
    {
        Impl()
            : Executor(Detail::GetBoundedWorkerCount(16))
        {
        }

        RenderJob Submit(
            const Scene* scene,
            std::shared_ptr<const Scene> owner,
            Camera camera,
            RenderSettings settings,
            CancellationToken external = CancellationToken()) const
        {
            return Queue.Submit(
                [this, scene, owner = std::move(owner), camera, settings, external]
                (const CancellationToken& queued) mutable
                {
                    if (!scene)
                    {
                        return Detail::MakeFailedResult(
                            "The CPU ray tracing submission has no scene.");
                    }
                    ++TileDispatchSequence;
                    const auto cancelled = Detail::CombineCancellation(queued, external);
                    if (settings.Micropolygons.Enabled)
                    {
                        auto diced = Detail::PrepareMicropolygons(
                            *scene, camera, settings, cancelled,
                            [&](const auto& controls, const auto& patches, const auto& ranges, auto& words,
                                auto&) {
                                return Cpu::MicropolygonPass(controls, patches, ranges, words, cancelled);
                            });
                        if (diced.Status != RenderStatus::Completed)
                            return diced.Status == RenderStatus::Cancelled
                                       ? Detail::MakeCancelledResult()
                                       : Detail::MakeFailedResult(diced.Error);
                        owner = std::move(diced.Geometry);
                        scene = owner.get();
                        settings.Micropolygons.Enabled = false;
                        settings.DicingPlan.reset();
                    }
                    std::shared_ptr<const Cpu::PreparedSpectralScene> spectralScene;
                    Cpu::PreparedSpectralFrame spectralFrame;
                    if (settings.Radiometry == RadiometryMode::Spectral)
                    {
                        RenderResult prepared;
                        if (!Detail::ValidateRenderInput(*scene, camera, settings, prepared.Error))
                            return prepared;
                        if (owner && SpectralOwner.lock() == owner)
                            spectralScene = SpectralCache;
                        if (!spectralScene)
                        {
                            auto candidate = std::make_shared<Cpu::PreparedSpectralScene>();
                            prepared.Status = Cpu::PrepareSpectralScene(*scene, cancelled, *candidate, prepared.Error, &Executor);
                            if (prepared.Status != RenderStatus::Completed) return prepared;
                            spectralScene = std::move(candidate);
                            if (owner)
                            {
                                SpectralCache = spectralScene;
                                SpectralOwner = owner;
                            }
                        }
                        prepared.Status = Cpu::PrepareSpectralFrame(*spectralScene, settings.Medium,
                                                                    spectralFrame, prepared.Error);
                        if (prepared.Status != RenderStatus::Completed) return prepared;
                    }
                    Detail::SpatialGuideCache spatialGuides;
                    return Detail::RenderWithDenoisedSampling(
                        *scene, camera, settings, cancelled,
                        [&](const RenderSettings &expanded)
                        {
                            return Detail::RenderCpuTiles(*scene, camera, expanded, Executor, cancelled,
                                                          spectralScene ? &spectralFrame : nullptr);
                        },
                        [&](const RenderSettings &raster, const LinearFilm &film, ReconstructedImage &image,
                            std::string &error)
                        {
                            const GeometryGuideImage *guides = nullptr;
                            if (settings.Reconstruction.GeometryGuidance)
                            {
                                guides = spatialGuides.Get(
                                    raster,
                                    [&] { return CaptureGuidesNative(*scene, camera, raster, cancelled, true); },
                                    error);
                                if (!guides)
                                    return false;
                            }
                            return Cpu::SpatialDenoise(
                                film, settings.Reconstruction, ReconstructionUsesMoments(settings.Integrator),
                                settings.GammaCorrection, cancelled, image, error, guides);
                        },
                        [&](const std::vector<AdaptiveReconstructionData::Pixel> &input,
                            AdaptiveReconstructionData::Controls controls, std::vector<uint32_t> &counts,
                            std::string &)
                        {
                            std::vector<uint32_t> candidate;
                            candidate.reserve(input.size());
                            for (const auto& pixel : input)
                            {
                                if (cancelled())
                                    return false;
                                candidate.push_back(Cpu::DenoisedSampleBudget(pixel, controls));
                            }
                            counts = std::move(candidate);
                            return true;
                        });
                });
        }

        GeometryGuideResult CaptureGuidesNative(const Scene &scene, const Camera &camera,
                                                const RenderSettings &settings,
                                                const Detail::CancellationProbe &stopped, bool spatialAlbedo = false) const
        {
            return Detail::CaptureGeometryGuides(
                scene, camera, settings, spatialAlbedo, stopped, [&](const auto &controls, auto &rays, auto &error)
                { return Cpu::GenerateGeometryGuideRays(controls, stopped, rays, error); },
                [&](const auto &rays)
                { return QueryNative(scene, rays, {settings.Acceleration, settings.BvhPolicy}, stopped); },
                [&](const auto &controls, const auto &rays, const auto &hits, auto &guides, auto &error)
                { return Cpu::ResolveGeometryGuideHits(controls, rays, hits, stopped, guides, error); });
        }

        RayQueryResult QueryNative(const Scene& scene, const std::vector<Ray>& rays,
                                   AccelerationOptions acceleration,
                                   const Detail::CancellationProbe& cancelled) const
        {
            RayQueryResult result;
            if (!ValidateAcceleration(acceleration, result.Error) ||
                !ValidateRayQueries(rays, result.Error) ||
                !Detail::ValidateRenderInput(scene, Camera{}, RenderSettings{}, result.Error, false))
                return result;
            if (cancelled())
            {
                result.Status = RenderStatus::Cancelled;
                return result;
            }
            if (rays.empty())
            {
                result.Status = RenderStatus::Completed;
                return result;
            }
            const auto accelerator = Cpu::BuildAccelerator(scene, acceleration, cancelled);
            if (accelerator->WasCancelled())
            {
                result.Status = RenderStatus::Cancelled;
                return result;
            }
            result.Hits.resize(rays.size());
            for (size_t index = 0; index < rays.size(); ++index)
            {
                if (cancelled())
                {
                    result.Status = RenderStatus::Cancelled;
                    result.Hits.clear();
                    return result;
                }
                Cpu::SurfaceHit hit;
                hit.Distance = rays[index].MaxDistance;
                uint64_t primitiveTests = 0;
                RayQueryHit& output = result.Hits[index];
                output.Distance = rays[index].MaxDistance;
                if (accelerator->Intersect(rays[index], hit, primitiveTests))
                {
                    output.Normal = hit.Normal;
                    output.GeometricNormal = hit.GeometricNormal;
                    output.Distance = hit.Distance;
                    output.Color = hit.Color;
                    output.PrimitiveIndex = hit.PrimitiveIndex;
                    output.TexCoord = hit.TexCoord;
                    output.MaterialIndex = hit.MaterialIndex;
                    output.Flags = RayQueryHit::HitFlag | (hit.HasTexCoord ? RayQueryHit::TexCoordFlag : 0) |
                                   (hit.FrontFace ? RayQueryHit::FrontFaceFlag : 0);
                }
            }
            result.Status = RenderStatus::Completed;
            return result;
        }

        mutable Detail::PersistentExecutor Executor;
        mutable std::weak_ptr<const Scene> SpectralOwner;
        mutable std::shared_ptr<const Cpu::PreparedSpectralScene> SpectralCache;
        mutable Detail::PersistentRenderQueue Queue;
        mutable uint64_t TileDispatchSequence = 0;
    };

    CpuRayTracingRHI::CpuRayTracingRHI()
        : m_Impl(std::make_unique<Impl>())
    {
    }

    CpuRayTracingRHI::~CpuRayTracingRHI() = default;

    BackendInfo CpuRayTracingRHI::GetBackendInfo() const
    {
        return {
            BackendType::CPU,
            "CPU",
            "Native CPU transport with independent acceleration and a persistent tile worker pool",
            true,
            false,
            true};
    }

    RenderResult CpuRayTracingRHI::Render(
        const Scene& scene,
        const Camera& camera,
        const RenderSettings& settings,
        const CancellationToken& cancellation) const
    {
        return m_Impl->Submit(&scene, nullptr, camera, settings, cancellation).GetResult();
    }

    RayQueryResult CpuRayTracingRHI::QueryRays(const Scene& scene, const std::vector<Ray>& rays,
        AccelerationOptions acceleration,
        const CancellationToken& cancellation) const
    {
        RayQueryResult result;
        if (!ValidateAcceleration(acceleration, result.Error) || !ValidateRayQueries(rays, result.Error)
            || !Detail::ValidateRenderInput(scene, Camera{}, RenderSettings{}, result.Error, false)) return result;
        const RenderResult task = m_Impl->Queue.Submit([&](const CancellationToken& queued)
        {
            const auto cancelled = Detail::CombineCancellation(queued, cancellation);
            result = m_Impl->QueryNative(scene, rays, acceleration, cancelled);
            RenderResult completed;
            completed.Status = result.Status;
            completed.Error = result.Error;
            return completed;
        }).GetResult();
        result.Status = task.Status;
        result.Error = task.Error;
        if (result.Status != RenderStatus::Completed) result.Hits.clear();
        return result;
    }

    MicropolygonResult CpuRayTracingRHI::PrepareMicropolygons(const Scene& scene, const Camera& camera,
                                                              const RenderSettings& settings,
                                                              const CancellationToken& cancellation) const
    {
        MicropolygonResult result;
        const auto task =
            m_Impl->Queue
                .Submit([&](const CancellationToken& queued) {
                    const auto stopped = Detail::CombineCancellation(queued, cancellation);
                    result = Detail::PrepareMicropolygons(scene, camera, settings, stopped,
                                                          [&](const auto& controls, const auto& patches,
                                                              const auto& ranges, auto& words, auto&) {
                                                              return Cpu::MicropolygonPass(
                                                                  controls, patches, ranges, words, stopped);
                                                          });
                    RenderResult completion;
                    completion.Status = result.Status;
                    completion.Error = result.Error;
                    return completion;
                })
                .GetResult();
        result.Status = task.Status;
        result.Error = task.Error;
        return result;
    }

    TemporalResult CpuRayTracingRHI::ReconstructTemporal(const TemporalFrame& frame,
                                                         const TemporalHistory* history,
                                                         const TemporalOptions& options,
                                                         const CancellationToken& cancellation) const
    {
        TemporalResult result;
        const auto task =
            m_Impl->Queue
                .Submit([&](const CancellationToken& queued) {
                    const auto stopped = Detail::CombineCancellation(queued, cancellation);
                    result = Detail::ReconstructTemporal(
                        BackendType::CPU, frame, history, options, stopped,
                        [&](const auto& controls, const auto& current, const auto& guides,
                            const auto& previous, const auto& oldGuides, auto& filtered, auto& display,
                            auto& error) {
                            return Cpu::TemporalDenoise(controls, current, guides, previous, oldGuides,
                                                        stopped, filtered, display, error);
                        });
                    RenderResult completion;
                    completion.Status = result.Status;
                    completion.Error = result.Error;
                    return completion;
                })
                .GetResult();
        result.Status = task.Status;
        result.Error = task.Error;
        if (result.Status != RenderStatus::Completed)
        {
            result.History = {};
            result.Display = {};
            result.AcceptedHistoryPixels = 0;
        }
        return result;
    }

    GeometryGuideResult CpuRayTracingRHI::CaptureGeometryGuides(const Scene& scene, const Camera& camera,
                                                                const RenderSettings& settings,
                                                                const CancellationToken& cancellation) const
    {
        GeometryGuideResult result;
        if (settings.Micropolygons.Enabled)
        {
            auto diced = PrepareMicropolygons(scene, camera, settings, cancellation);
            if (diced.Status != RenderStatus::Completed)
                return {diced.Status, {}, diced.Error};
            auto prepared = settings;
            prepared.Micropolygons.Enabled = false;
            prepared.DicingPlan.reset();
            return CaptureGeometryGuides(*diced.Geometry, camera, prepared, cancellation);
        }
        const auto task =
            m_Impl->Queue
                .Submit([&](const CancellationToken& queued) {
                    const auto stopped = Detail::CombineCancellation(queued, cancellation);
                    result = m_Impl->CaptureGuidesNative(scene, camera, settings, stopped);
                    RenderResult completion;
                    completion.Status = result.Status;
                    completion.Error = result.Error;
                    return completion;
                })
                .GetResult();
        result.Status = task.Status;
        result.Error = task.Error;
        if (result.Status != RenderStatus::Completed)
            result.Image = {};
        return result;
    }

    RenderJob CpuRayTracingRHI::RenderAsync(
        Scene scene,
        Camera camera,
        RenderSettings settings) const
    {
        return RenderAsync(std::make_shared<Scene>(std::move(scene)), camera, settings);
    }

    RenderJob CpuRayTracingRHI::RenderAsync(
        std::shared_ptr<const Scene> scene,
        Camera camera,
        RenderSettings settings) const
    {
        const Scene* pointer = scene.get();
        return m_Impl->Submit(pointer, std::move(scene), camera, settings);
    }

    void CpuRayTracingRHI::WaitIdle() const
    {
        m_Impl->Queue.WaitIdle();
    }
}
}
