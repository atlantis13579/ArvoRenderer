#include "RayTracer/Backends/CPU/CpuTransport.h"
#include <stdexcept>

namespace Arvo::RayTracing::Cpu
{
std::unique_ptr<SceneAccelerator> BuildAccelerator(const Scene &scene, AccelerationOptions options,
                                                   CancellationProbe cancellation)
{
    if (!scene.Instances.empty())
        return std::make_unique<InstancedScene>(scene, options, std::move(cancellation));
    return BuildGeometryAccelerator(scene, options, std::move(cancellation));
}

std::unique_ptr<SceneAccelerator> BuildGeometryAccelerator(const SceneGeometry &scene,
    AccelerationOptions options, CancellationProbe cancellation)
{
    std::string error;
    if (!ValidateAcceleration(options, error))
        throw std::invalid_argument(error);
    switch (options.Type)
    {
    case AccelerationType::Bvh:
        if (options.BvhPolicy == BvhBuildPolicy::SpatialSplits)
            return std::make_unique<SpatialBvh>(scene, std::move(cancellation));
        return std::make_unique<BVH>(scene, std::move(cancellation));
    case AccelerationType::KdTree:
        return std::make_unique<KdTree>(scene, std::move(cancellation));
    case AccelerationType::Bih:
        return std::make_unique<Bih>(scene, std::move(cancellation));
    default:
        throw std::invalid_argument("The CPU accelerator has no implementation");
    }
}
} // namespace Arvo::RayTracing::Cpu
