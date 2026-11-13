#pragma once
#include <cstdint>

namespace Arvo::RayTracing::ComputePhoton
{
// Upload contracts only. Photon transport is implemented in PhotonMapping.hlsl.
inline constexpr uint32_t MaximumReflections = 32, ThreadsPerGroup = 64;
inline constexpr uint32_t TransportBatch = 1024, IndexBatch = 65536;
inline constexpr uint32_t SceneDescriptors = 10, FrameDescriptors = 8;
inline constexpr uint32_t DescriptorCount = SceneDescriptors + FrameDescriptors;
struct Vector
{
    float X, Y, Z;
};
struct alignas(16) Photon
{
    Vector Position;
    uint32_t Next;
    Vector Incoming;
    uint32_t LightDepth;
    Vector Flux;
    uint32_t Kind;
    Vector Normal;
    uint32_t Reserved; // Volume photons store their material index here.
};
struct alignas(16) Controls
{
    Vector Center;
    float WorldRadius;
    double Radius;
    uint32_t EmitterCount, PhotonPaths;
    uint32_t SlotsPerPath, BucketCount, Seed, Classes;
    uint32_t Begin, Count, Sample, Reserved;
};
static_assert(sizeof(Photon) == 64 && sizeof(Controls) == 64);
} // namespace Arvo::RayTracing::ComputePhoton
