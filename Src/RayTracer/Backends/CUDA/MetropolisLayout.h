#pragma once
#include <cstdint>

namespace Arvo::RayTracing::CudaMlt
{
// CUDA MLT upload/storage ABI only. Numerical operations live in its .cu file.
inline constexpr unsigned MaximumReflections = 32;
inline constexpr unsigned CameraDimensions = 4;
inline constexpr unsigned VertexDimensions = 12;
inline constexpr unsigned PrimaryDimensions = CameraDimensions + (MaximumReflections + 2) * VertexDimensions;
inline constexpr unsigned ThreadsPerBlock = 64;
inline constexpr unsigned MutationsPerBatch = 64;
inline constexpr unsigned BootstrapBatch = 1024;
inline constexpr unsigned NumericalFailure = 1, InvalidConfiguration = 2;
struct Vector
{
    float X, Y, Z;
};
struct PrimaryState
{
    double Values[PrimaryDimensions];
};
struct PathValue
{
    float RasterX, RasterY;
    Vector Radiance;
    double Importance;
};
struct alignas(16) Chain
{
    PrimaryState Primary;
    PathValue Current;
    unsigned long long RandomState;
    unsigned long long Proposals, Accepted, LargeSteps, RejectedPositive;
};
struct alignas(16) Controls
{
    Vector WorldCenter;
    float WorldRadius;
    unsigned EmitterCount, Seed, BootstrapCount, ChainCount;
    unsigned Begin, Count;
    double LargeStepProbability, MutationScale;
    unsigned Reserved[2];
};
struct Distribution
{
    double Normalization, TotalImportance;
};
// Double accumulation reduces unordered GPU splat rounding before conversion
// to the float film. Native atomic scheduling need not be bitwise reproducible.
struct PassPixel
{
    double Red, Green, Blue;
};
static_assert(sizeof(PathValue) == 32 && sizeof(Controls) == 64);
static_assert(sizeof(Chain) == 3376 && sizeof(PassPixel) == 24);
} // namespace Arvo::RayTracing::CudaMlt
