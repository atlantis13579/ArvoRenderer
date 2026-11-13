#pragma once
#include <cstdint>
namespace Arvo::RayTracing::MetalPhoton
{
// Data contracts only. All Metal transport is in PhotonMapping.metal.
constexpr uint32_t MaximumReflections = 32, ThreadsPerGroup = 64;
constexpr uint32_t TransportBatch = 1024, IndexBatch = 65536;
constexpr uint32_t RecordsBinding = 16, HeadsBinding = 17, ControlsBinding = 18, ErrorBinding = 19;
constexpr uint32_t ClearStage = 0, EmitStage = 1, IndexStage = 2, CameraStage = 3, StageCount = 4;
struct Vector
{
    float X, Y, Z;
};
struct Photon
{
    Vector Position;
    uint32_t Next;
    Vector Incoming;
    uint32_t Depth;
    Vector Flux;
    uint32_t Kind;
    Vector Normal;
    uint32_t Reserved; // Volume photons store their material index here.
};
struct Controls
{
    Vector Center;
    float WorldRadius;
    uint32_t Sources, PhotonPaths, SlotsPerPath, Buckets;
    uint32_t Seed, Classes, First, Count;
    float RadiusScale;
    uint32_t Sample, Reserved[2];
};
static_assert(sizeof(Photon) == 64 && sizeof(Controls) == 64);
} // namespace Arvo::RayTracing::MetalPhoton
