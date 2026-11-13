#pragma once
#include "Renderer/RayTracing.h"

namespace Arvo::RayTracing
{
// Shared scene metadata only. Each backend owns its catalog construction and
// selection arithmetic; this header contains no transport implementation.
enum class EmitterKind : uint32_t
{
    Directional,
    Point,
    Rectangle,
    Environment,
    Triangle,
    Sphere
};
inline constexpr uint32_t InvalidEmitterIndex = std::numeric_limits<uint32_t>::max();
inline constexpr uint32_t MaximumUniformEmitters = uint32_t{1} << std::numeric_limits<float>::digits;
struct EmitterReference
{
    EmitterKind Kind = EmitterKind::Point;
    // Analytic-light index, or triangle/sphere index in the referenced geometry.
    uint32_t Index = 0;
    // Invalid means legacy world-space geometry. Otherwise index Scene::Instances.
    uint32_t Instance = InvalidEmitterIndex;
    uint32_t Primitive = InvalidRayPrimitiveIndex;
};
struct EmissionCatalog
{
    std::vector<EmitterReference> Sources;
    Float3 Center;
    // Strictly enclosing geometry sphere. Zero denotes an empty geometry scene;
    // infinite emission then yields no light path, but camera background remains.
    float Radius = 0;
};
struct EmitterSelection
{
    uint32_t Index = InvalidEmitterIndex;
    float Probability = 0;
};

// Builders require ordinary scene validation first and also check the indices,
// transforms, source parameters and numerical domain they themselves consume.
// They throw on invalid input; they never return an incomplete light catalog.
namespace Cpu
{
EmissionCatalog BuildEmissionCatalog(const Scene& scene);
EmitterSelection SelectEmitter(uint32_t count, float variate);
float EmitterProbability(uint32_t count, uint32_t index);
} // namespace Cpu
namespace CudaEmissionHost
{
EmissionCatalog BuildCatalog(const Scene& scene);
}
namespace ComputeEmission
{
EmissionCatalog BuildCatalog(const Scene& scene);
}
namespace DxrEmission
{
EmissionCatalog BuildCatalog(const Scene& scene);
}
namespace MetalEmission
{
EmissionCatalog BuildCatalog(const Scene& scene);
}
} // namespace Arvo::RayTracing
