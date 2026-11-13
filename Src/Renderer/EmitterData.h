#pragma once
#include "Renderer/EmissionCatalog.h"
#include <cstddef>

namespace Arvo::RayTracing
{
// Shared upload schema only; every backend prepares and samples its own data.
// Each row is a 16-byte lane. MSL uses packed_float3; CUDA/HLSL use float3.
inline constexpr uint32_t EmitterCastsShadows = 1, EmitterTwoSided = 2;
struct alignas(16) PackedEmitter
{
    Float3 Origin;
    EmitterKind Kind = EmitterKind::Point;
    Float3 Direction;
    float Width = 0;
    // Analytic Color*Intensity, or material radiance for geometric emitters.
    Float3 Strength;
    float Height = 0;
    Float3 BackgroundTop;
    uint32_t Primitive = InvalidRayPrimitiveIndex;
    Float3 BackgroundBottom;
    uint32_t SourceIndex = 0;
    Float3 AxisX;
    uint32_t Instance = InvalidEmitterIndex;
    Float3 AxisY;
    uint32_t Flags = 0;
    Float3 AxisZ;
    uint32_t Reserved = 0;
};
inline constexpr size_t EmitterLaneBytes = 4 * sizeof(float), EmitterLaneCount = 8;
static_assert(sizeof(PackedEmitter) == EmitterLaneCount * EmitterLaneBytes);
static_assert(offsetof(PackedEmitter, Direction) == EmitterLaneBytes);
static_assert(offsetof(PackedEmitter, Strength) == 2 * EmitterLaneBytes);
static_assert(offsetof(PackedEmitter, BackgroundTop) == 3 * EmitterLaneBytes);
static_assert(offsetof(PackedEmitter, BackgroundBottom) == 4 * EmitterLaneBytes);
static_assert(offsetof(PackedEmitter, AxisX) == 5 * EmitterLaneBytes);
static_assert(offsetof(PackedEmitter, AxisY) == 6 * EmitterLaneBytes);
static_assert(offsetof(PackedEmitter, AxisZ) == 7 * EmitterLaneBytes);
struct PreparedEmissionScene
{
    Float3 Center;
    float Radius = 0;
    std::vector<PackedEmitter> Sources;
};
// Prepare from the same immutable scene snapshot as traversal/material uploads.
// Returned source slots retain catalog order; Primitive is the traversal ID.
namespace Cpu
{
PreparedEmissionScene PrepareEmitters(const Scene& scene);
}
namespace ComputeEmission
{
PreparedEmissionScene PrepareEmitters(const Scene& scene);
}
namespace CudaEmissionHost
{
PreparedEmissionScene PrepareEmitters(const Scene& scene);
}
namespace DxrEmission
{
PreparedEmissionScene PrepareEmitters(const Scene& scene);
}
namespace MetalEmission
{
PreparedEmissionScene PrepareEmitters(const Scene& scene);
}
} // namespace Arvo::RayTracing
