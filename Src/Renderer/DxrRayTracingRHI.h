#pragma once

#include "Renderer/RayTracing.h"

namespace Arvo
{
namespace RayTracing
{
    class ARVO_RENDERER_API DxrRayTracingRHI final : public IRayTracingRHI
    {
    public:
        DxrRayTracingRHI();
        ~DxrRayTracingRHI() override;

        BackendInfo GetBackendInfo() const override;
        MicropolygonResult PrepareMicropolygons(const Scene&, const Camera&, const RenderSettings&,
            const CancellationToken& = CancellationToken()) const override;
        TemporalResult ReconstructTemporal(const TemporalFrame&, const TemporalHistory*,
                                           const TemporalOptions&,
                                           const CancellationToken& = CancellationToken()) const override;
        GeometryGuideResult CaptureGeometryGuides(
            const Scene&, const Camera&, const RenderSettings&,
            const CancellationToken& = CancellationToken()) const override;
        RayQueryResult QueryRays(
            const Scene& scene,
            const std::vector<Ray>& rays,
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

namespace Arvo::Renderer
{
// Shared DXR entry point for offline workers and realtime preview. Both use
// this implementation. Preview selects native-resolution ideal reflection and
// refraction from the same Whitted HLSL used by offline jobs.
std::shared_ptr<RayTracing::IRayTracingRHI> CreateDxrRHI();
}
