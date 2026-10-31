#pragma once

#include <RayTracer/RayTracer.h>

namespace Arvo
{
namespace RayTracing
{
    class ARVO_RAYTRACER_API MetalRayTracingRHI final : public IRayTracingRHI
    {
    public:
        MetalRayTracingRHI();
        ~MetalRayTracingRHI() override;

        BackendInfo GetBackendInfo() const override;
        RenderResult Render(
            const Scene& scene,
            const Camera& camera,
            const RenderSettings& settings,
            const CancellationToken& cancellation = CancellationToken()) const override;
        RenderJob RenderAsync(Scene scene, Camera camera, RenderSettings settings) const override;
        RenderJob RenderAsync(
            std::shared_ptr<const Scene> scene,
            Camera camera,
            RenderSettings settings) const override;
        void WaitIdle() const override;

    private:
        struct Impl;
        std::unique_ptr<Impl> m_Impl;
    };
}
}
