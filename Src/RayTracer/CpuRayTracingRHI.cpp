#include "CpuRayTracingRHI.h"

#include "Renderer/RayTracingBackendCommon.h"
#include "TracingCore.h"

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
                    (void)owner;
                    if (!scene)
                    {
                        return Detail::MakeFailedResult(
                            "The CPU ray tracing submission has no scene.");
                    }
                    ++TileDispatchSequence;
                    return Detail::RenderCpuTiles(
                        *scene,
                        camera,
                        settings,
                        Executor,
                        Detail::CombineCancellation(queued, external));
                });
        }

        mutable Detail::PersistentExecutor Executor;
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
            "Native CPU BVH backend with a persistent tile worker pool",
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
