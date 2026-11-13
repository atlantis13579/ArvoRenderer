#pragma once
#include <cstdint>

namespace Arvo::RayTracing::MetalMlt
{
// Data ABI only; numerical algorithms belong to MetropolisLightTransport.metal.
constexpr uint32_t MaximumReflections = 32, CameraDimensions = 4, VertexDimensions = 12;
constexpr uint32_t Dimensions = CameraDimensions + (MaximumReflections + 2) * VertexDimensions;
constexpr uint32_t BootstrapBatch = 1024, MutationBatch = 64, ThreadsPerGroup = 64;
constexpr uint32_t ChainBinding = 16, PassBinding = 17, ControlsBinding = 18, ErrorBinding = 19;
constexpr uint32_t BootstrapBinding = 20, DistributionBinding = 21;
constexpr uint32_t BootstrapStage = 0, DistributionStage = 1, InitializeStage = 2;
constexpr uint32_t MutateStage = 3, ClearStage = 4, CommitStage = 5, StageCount = 6;
struct WideNumber
{
    float High, Low;
    int32_t Exponent;
    uint32_t Reserved;
};
struct Primary
{
    uint32_t Values[Dimensions];
};
struct Random
{
    uint64_t State, Stream;
};
struct alignas(8) Path
{
    float Raster[2], Color[3];
    uint32_t Reserved;
    WideNumber Importance;
};
struct Chain
{
    Random Generator;
    Path Current;
    uint64_t Proposals, Accepted, LargeSteps, PositiveRejected;
    Primary Coordinates;
};
struct Bootstrap
{
    Path Value;
    WideNumber Cumulative;
};
struct Distribution
{
    WideNumber Total, Mean;
};
struct Controls
{
    float Center[3], Radius;
    uint32_t Sources, BootstrapCount, ChainCount, Seed;
    uint32_t First, Count;
    float LargeStepProbability, MutationScale;
};
static_assert(sizeof(Path) == 40 && sizeof(Chain) == 1736 && sizeof(Bootstrap) == 56);
static_assert(sizeof(Controls) == 48 && sizeof(Distribution) == 32);
} // namespace Arvo::RayTracing::MetalMlt
