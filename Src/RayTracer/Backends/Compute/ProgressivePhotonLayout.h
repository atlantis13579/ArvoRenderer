#pragma once
#include <cstdint>
namespace Arvo::RayTracing::ComputePpm
{
// Data ABI only. The independent HLSL owns transport and progressive updates.
constexpr uint32_t MaximumReflections = 32, ThreadsPerGroup = 64;
constexpr uint32_t LightBatch = 1024, CameraBatch = 1024, LinearBatch = 65536;
constexpr uint32_t SceneDescriptors = 10, FrameDescriptors = 10;
constexpr uint32_t DescriptorCount = SceneDescriptors + FrameDescriptors;
enum VisibleKind : uint32_t { VisibleNone, VisibleSurface, VisibleVolume };
struct Vector
{
    float X, Y, Z;
};
struct alignas(16) VisiblePoint
{
    Vector Position;
    uint32_t Active;
    Vector Normal;
    uint32_t FrontFace;
    Vector Geometry;
    uint32_t Depth;
    Vector View;
    uint32_t Material;
    Vector Color;
    uint32_t Reserved;
    Vector Throughput;
    uint32_t Padding;
};
struct alignas(16) Pixel
{
    double RadiusSquared, EffectiveCount;
    double Flux[3], Direct[3], Incoming[3];
    uint32_t Arrivals, Reserved;
    double VolumeRadiusSquared, VolumeEffectiveCount;
    double VolumeFlux[3], VolumeIncoming[3];
    uint32_t VolumeArrivals, VolumeReserved[3];
};
struct alignas(16) Photon
{
    Vector Position;
    uint32_t Next;
    Vector Incoming;
    uint32_t Depth;
    Vector Flux;
    uint32_t Kind;
    Vector Normal;
    uint32_t Reserved;
};
struct alignas(16) Controls
{
    Vector Center;
    float WorldRadius;
    double InitialRadius, Alpha;
    uint32_t Sources, PathsPerPass, Seed, Classes;
    uint32_t SlotsPerPath, Buckets, Begin, Count;
    uint32_t Pass, LightBegin, LightCount, Reserved;
};
static_assert(sizeof(VisiblePoint) == 96 && sizeof(Pixel) == 176);
static_assert(sizeof(Photon) == 64 && sizeof(Controls) == 80);
} // namespace Arvo::RayTracing::ComputePpm
