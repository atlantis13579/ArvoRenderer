#include "MetalRayTracingRHI.h"

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
