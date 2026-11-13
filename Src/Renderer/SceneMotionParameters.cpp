#include "Renderer/RayTracing.h"
#include <algorithm>

namespace Arvo::RayTracing
{
bool ValidateRigidMotion(const MotionData::RigidMotion& motion, std::string& error)
{
    // Parameter validation is shared; pose interpolation remains backend-owned.
    MotionData::Exposure parameters;
    parameters.Motion = motion.Transform;
    parameters.Enabled = motion.Enabled;
    if (!ValidateCameraMotion(parameters, error))
    {
        error = "Invalid rigid instance motion: " + error;
        return false;
    }
    return true;
}
bool HasSceneMotion(const Scene& scene)
{
    return std::any_of(scene.Instances.begin(), scene.Instances.end(),
                       [](const GeometryInstance& instance) { return instance.Motion.Enabled != 0; });
}
bool ValidateSceneMotionSupport(Algorithm integrator, const Scene& scene, std::string& error)
{
    for (const auto& instance : scene.Instances)
    {
        if (!ValidateRigidMotion(instance.Motion, error))
            return false;
        if (!instance.Motion.Enabled)
            continue;
        if (!SupportsMotionBlur(integrator))
        {
            error = "Animated instances require path or distributed ray tracing; connected paths and caches "
                    "must first share the same scene-time measure.";
            return false;
        }
        if (instance.GeometryIndex >= scene.Geometries.size() || !scene.Geometries[instance.GeometryIndex])
        {
            error = "Animated instance references missing geometry.";
            return false;
        }
        const auto registered = [&](uint32_t material) {
            return material < scene.Materials.size() && scene.Materials[material].SampledEmitter;
        };
        const auto& object = *scene.Geometries[instance.GeometryIndex];
        for (const auto& triangle : object.Triangles)
            if (registered(triangle.MaterialIndex))
            {
                error = "A moving registered emitter also needs a synchronized light trajectory.";
                return false;
            }
        for (const auto& sphere : object.Spheres)
            if (registered(sphere.MaterialIndex))
            {
                error = "A moving registered emitter also needs a synchronized light trajectory.";
                return false;
            }
    }
    return true;
}
} // namespace Arvo::RayTracing
