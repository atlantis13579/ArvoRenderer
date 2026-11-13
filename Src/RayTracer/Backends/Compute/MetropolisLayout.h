#pragma once
#include <cstdint>
namespace Arvo::RayTracing::ComputeMlt
{
// Upload/storage ABI only. The independently handwritten HLSL owns all transport.
constexpr uint32_t MaximumReflections = 32, CameraDimensions = 4, VertexDimensions = 12;
constexpr uint32_t PrimaryDimensions = CameraDimensions + (MaximumReflections + 2) * VertexDimensions;
constexpr uint32_t BootstrapBatch = 1024, ThreadsPerGroup = 64, FilmGroupWidth = 8;
// Publish one complete Markov transition per lane at each dispatch boundary.
// The RHI still executes every full-film mutation; expensive small chain sets
// cannot monopolize one long shader loop between fences/cancellation checks.
constexpr uint32_t MutationBatch = 1;
constexpr uint32_t SceneDescriptors = 10, FrameDescriptors = 12;
constexpr uint32_t DescriptorCount = SceneDescriptors + FrameDescriptors;
struct Primary
{
    uint32_t Values[PrimaryDimensions];
};
struct PathValue
{
    float Raster[2], Radiance[3], Reserved;
    double Importance;
};
struct Chain
{
    PathValue Current;
    uint32_t Random[4];
    uint32_t Proposals[2], Accepted[2], LargeSteps[2], RejectedPositive[2];
};
struct Controls
{
    float Center[3], Radius;
    uint32_t EmitterCount, BootstrapCount, ChainCount, Seed;
    uint32_t Begin, Count;
    float LargeStepProbability, MutationScale;
};
static_assert(sizeof(Primary) == 1648 && sizeof(PathValue) == 32);
static_assert(sizeof(Chain) == 80 && sizeof(Controls) == 48);
} // namespace Arvo::RayTracing::ComputeMlt
