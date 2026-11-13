#include "RayTracer/Backends/CPU/CpuTransport.h"

namespace Arvo::RayTracing::Cpu
{
// Cook et al., "Distributed Ray Tracing" (1984): a camera observation freezes
// scene time for its entire transport path, including visibility connections.
// Time is separate from ray distance; transforming a ray cannot rescale it.
bool SceneAccelerator::IntersectAtTime(const Ray& ray, SurfaceHit& hit, uint64_t& tests,
                                      float, bool anyHit) const
{
    return Intersect(ray, hit, tests, anyHit);
}
bool SceneAccelerator::OccludedAtTime(const Ray& ray, uint64_t& tests, float sceneTime) const
{
    SurfaceHit hit;
    hit.Distance = ray.MaxDistance;
    return IntersectAtTime(ray, hit, tests, sceneTime, true);
}
} // namespace Arvo::RayTracing::Cpu
