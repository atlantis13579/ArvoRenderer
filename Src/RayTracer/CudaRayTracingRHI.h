#pragma once

#include <RayTracer/RayTracer.h>

namespace Arvo
{
namespace RayTracing
{
    class ARVO_RAYTRACER_API CudaRayTracingRHI final : public IRayTracingRHI
    {
    public:
        CudaRayTracingRHI();
        ~CudaRayTracingRHI() override;

        BackendInfo GetBackendInfo() const override;
        MicropolygonResult PrepareMicropolygons(const Scene&, const Camera&, const RenderSettings&,
            const CancellationToken& = CancellationToken()) const override;
        TemporalResult ReconstructTemporal(const TemporalFrame&, const TemporalHistory*,
                                           const TemporalOptions&,
                                           const CancellationToken& = CancellationToken()) const override;
        GeometryGuideResult CaptureGeometryGuides(
            const Scene&, const Camera&, const RenderSettings&,
            const CancellationToken& = CancellationToken()) const override;
        RayQueryResult QueryRays(const Scene& scene, const std::vector<Ray>& rays,
            AccelerationOptions acceleration = AccelerationType::Bvh,
            const CancellationToken& cancellation = CancellationToken()) const override;
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
