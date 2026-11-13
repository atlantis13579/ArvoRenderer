#include "RayTracer/Backends/CPU/CpuTransport.h"

namespace Arvo::RayTracing::Cpu
{
// Pinhole projection uses absolute image coordinates, independent of tile layout.
Ray MakeCameraRay(
    const CameraFrame& camera,
    const RenderSettings& settings,
    const SampleContext& samples)
{
    float jitterX = 0.5f;
    float jitterY = 0.5f;
    if (samples.SampleCount > 1 || samples.Sequence != SampleSequence::Independent || samples.SamplingSeed != 0)
    {
        const Float2 jitter = Sample2D(samples, 0u);
        jitterX = jitter.X;
        jitterY = jitter.Y;
    }
    const float u = (static_cast<float>(samples.PixelX) + jitterX) / static_cast<float>(FullImageWidth(settings));
    const float v = (static_cast<float>(samples.PixelY) + jitterY) / static_cast<float>(FullImageHeight(settings));
    const float screenX = (2.0f * u - 1.0f) * camera.HalfWidth;
    const float screenY = (1.0f - 2.0f * v) * camera.HalfHeight;
    Ray ray;
    ray.Origin = camera.Position;
    ray.Direction = Normalize(camera.Forward + camera.Right * screenX + camera.Up * screenY);
    ray.MinDistance = RayEpsilon;
    return ray;
}
}
