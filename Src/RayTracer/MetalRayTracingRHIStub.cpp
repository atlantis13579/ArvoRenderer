#include "MetalRayTracingRHI.h"
#include "Renderer/TemporalReconstruction.h"
#include "Renderer/MicropolygonPreparation.h"

#if !defined(__APPLE__)

#include "Renderer/RayTracingBackendCommon.h"

#include <utility>

namespace Arvo
{
namespace RayTracing
{
    struct MetalRayTracingRHI::Impl
    {
        RenderJob Submit() const
        {
            return Queue.Submit([](const CancellationToken& cancellation)
            {
                if (cancellation.IsCancellationRequested())
                {
                    return Detail::MakeCancelledResult();
                }
                return Detail::MakeFailedResult(
                    "Metal compute ray tracing is available only on Apple platforms.");
            });
        }

        mutable Detail::PersistentRenderQueue Queue;
    };

    MetalRayTracingRHI::MetalRayTracingRHI()
        : m_Impl(std::make_unique<Impl>())
    {
    }

    MetalRayTracingRHI::~MetalRayTracingRHI() = default;

    BackendInfo MetalRayTracingRHI::GetBackendInfo() const
    {
        return {
            BackendType::Metal,
            "Metal Compute",
            "Metal compute ray tracing is available only on Apple platforms",
            false,
            false,
            false};
    }

    RayQueryResult MetalRayTracingRHI::QueryRays(const Scene&, const std::vector<Ray>&, AccelerationOptions,
        const CancellationToken& cancellation) const
    {
        RayQueryResult result;
        if (cancellation.IsCancellationRequested()) result.Status = RenderStatus::Cancelled;
        else result.Error = "Metal compute ray tracing is available only on Apple platforms.";
        return result;
    }

    MicropolygonResult MetalRayTracingRHI::PrepareMicropolygons(const Scene&, const Camera&,
                                                                const RenderSettings&,
                                                                const CancellationToken& cancellation) const
    {
        MicropolygonResult result;
        if (cancellation.IsCancellationRequested())
            result.Status = RenderStatus::Cancelled;
        else
            result.Error = "Metal micropolygons require an Apple platform";
        return result;
    }

    TemporalResult MetalRayTracingRHI::ReconstructTemporal(const TemporalFrame&, const TemporalHistory*,
                                                           const TemporalOptions&,
                                                           const CancellationToken& cancellation) const
    {
        TemporalResult result;
        if (cancellation.IsCancellationRequested())
            result.Status = RenderStatus::Cancelled;
        else
            result.Error = "Metal temporal reconstruction requires an Apple device";
        return result;
    }

    GeometryGuideResult MetalRayTracingRHI::CaptureGeometryGuides(const Scene&, const Camera&,
                                                                  const RenderSettings&,
                                                                  const CancellationToken& cancellation) const
    {
        if (cancellation.IsCancellationRequested())
            return {RenderStatus::Cancelled, {}, {}};
        return {RenderStatus::Failed, {}, "Metal geometry guides require an Apple Metal device"};
    }

    RenderResult MetalRayTracingRHI::Render(
        const Scene&,
        const Camera&,
        const RenderSettings&,
        const CancellationToken& cancellation) const
    {
        if (cancellation.IsCancellationRequested())
        {
            return Detail::MakeCancelledResult();
        }
        return Detail::MakeFailedResult(
            "Metal compute ray tracing is available only on Apple platforms.");
    }

    RenderJob MetalRayTracingRHI::RenderAsync(
        Scene,
        Camera,
        RenderSettings) const
    {
        return m_Impl->Submit();
    }

    RenderJob MetalRayTracingRHI::RenderAsync(
        std::shared_ptr<const Scene>,
        Camera,
        RenderSettings) const
    {
        return m_Impl->Submit();
    }

    void MetalRayTracingRHI::WaitIdle() const
    {
        m_Impl->Queue.WaitIdle();
    }
}
}

#endif
