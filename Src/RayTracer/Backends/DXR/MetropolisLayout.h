#pragma once
#include <cstdint>
namespace Arvo::RayTracing::DxrMlt
{
// Raw storage contract only. DXR MLT owns all arithmetic in its HLSL file.
constexpr uint32_t MaximumReflections = 32, CameraDimensions = 4, VertexDimensions = 12;
constexpr uint32_t Dimensions = CameraDimensions + (MaximumReflections + 2) * VertexDimensions;
constexpr uint32_t BootstrapBatch = 1024, MutationBatch = 64;
constexpr uint32_t EmitterRoot = 19, ArenaRoot = 20, PassRoot = 21, ErrorRoot = 22, ControlsRoot = 23;
constexpr uint32_t Clear = 0, Bootstrap = 1, Distribution = 2, Initialize = 3, Mutate = 4, Commit = 5;
constexpr uint32_t DistributionBytes = 16, ChainHeaderBytes = 80, ChainStride = 1728;
constexpr uint32_t BootstrapStride = 48, PathBytes = 32;
constexpr uint32_t ChainPathOffset = 16, ChainCountersOffset = 48, ChainPrimaryOffset = ChainHeaderBytes;
constexpr uint32_t BootstrapCdfOffset = PathBytes;
struct Controls
{
    float Center[3], Radius;
    uint32_t EmitterCount, BootstrapCount, ChainCount, Seed;
    uint32_t Begin, Count, Stage, PassIndex;
    float LargeStepProbability, MutationScale;
    uint32_t Reserved[2];
};
constexpr uint32_t BootstrapOffset(uint32_t chains)
{
    return DistributionBytes + chains * ChainStride;
}
constexpr uint64_t ArenaBytes(uint32_t chains, uint32_t bootstrap)
{
    return uint64_t(BootstrapOffset(chains)) + uint64_t(bootstrap) * BootstrapStride;
}
static_assert(sizeof(Controls) == 64 && ChainPrimaryOffset + Dimensions * sizeof(uint32_t) <= ChainStride);
} // namespace Arvo::RayTracing::DxrMlt
