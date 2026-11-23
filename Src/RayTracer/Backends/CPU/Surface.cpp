#include "RayTracer/Backends/CPU/CpuTransport.h"
#include <algorithm>
#include <cmath>

namespace Arvo::RayTracing::Cpu
{
Float3 Background(const Scene& scene, const Ray& ray)
{
    const float blend = std::max(0.0f, std::min(1.0f, ray.Direction.Y * 0.5f + 0.5f));
    return scene.BackgroundBottom * (1.0f - blend) + scene.BackgroundTop * blend;
}
}
