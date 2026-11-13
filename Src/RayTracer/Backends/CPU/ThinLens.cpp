#include "RayTracer/Backends/CPU/CpuTransport.h"
#include <cmath>

namespace Arvo::RayTracing::Cpu
{
// PBRT v4, "Projective Camera Models", thin-lens depth of field:
// https://pbr-book.org/4ed/Cameras_and_Film/Projective_Camera_Models
// The concentric square-to-disk map preserves area without rejection. Every
// lens ray for a raster sample meets its original ray on the axial focus plane.
// We average aperture radiance: radius changes blur, not exposure or vignetting.
void ApplyThinLens(Ray& ray, const CameraFrame& camera, const Float2& lensSample)
{
    if (camera.Lens.Radius == 0)
        return;
    const float squareX = 2 * lensSample.X - 1;
    const float squareY = 2 * lensSample.Y - 1;
    float diskX = 0, diskY = 0;
    if (squareX != 0 || squareY != 0)
    {
        const bool horizontal = std::abs(squareX) > std::abs(squareY);
        const float radius = horizontal ? squareX : squareY;
        const float angle =
            horizontal ? (Pi / 4) * (squareY / squareX) : Pi / 2 - (Pi / 4) * (squareX / squareY);
        diskX = radius * std::cos(angle);
        diskY = radius * std::sin(angle);
    }
    const Float3 offset = (camera.Right * diskX + camera.Up * diskY) * camera.Lens.Radius;
    const Float3 focus = ray.Direction * (camera.Lens.FocusDistance / Dot(ray.Direction, camera.Forward));
    ray.Origin = camera.Position + offset;
    ray.Direction = Normalize(focus - offset);
}
} // namespace Arvo::RayTracing::Cpu
