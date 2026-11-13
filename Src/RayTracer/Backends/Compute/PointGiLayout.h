#pragma once
#include <cstddef>
#include <cstdint>

namespace Arvo::RayTracing::ComputePointGi
{
// Native upload ABI only. Transport is owned by PointBasedGlobalIllumination.hlsl.
constexpr uint32_t Invalid = 0xffffffffu, MaximumReflections = 32;
constexpr uint32_t Threads = 64, WorkBatch = 1024, SampleBatch = 32;
constexpr uint32_t SceneDescriptors = 13, FrameDescriptors = 9;
constexpr uint32_t DescriptorCount = SceneDescriptors + FrameDescriptors;
struct Vector
{
    float X, Y, Z;
};
struct alignas(16) Surface
{
    Vector Position;
    uint32_t Material;
    Vector Geometry;
    uint32_t Primitive;
    Vector Normal;
    uint32_t HasTexCoord;
    Vector Color;
    uint32_t Patch;
    float TexCoord[2];
    double Area, Radius;
    uint32_t Padding[2]{};
};
struct alignas(16) Bounds
{
    double Minimum[3];
    uint32_t Begin, Count;
    double Maximum[3];
    uint32_t Escape, Padding = 0;
};
struct alignas(16) Radiance
{
    double Front[3]{}, Back[3]{};
};
struct alignas(16) Controls
{
    uint32_t SurfaceCount, BoundsCount, SourceCount, Seed;
    uint32_t DirectSamples, GatherSamples, Begin, Count;
    uint32_t Order, FirstSample, Samples, Reserved = 0;
};
static_assert(sizeof(Surface) == 96 && offsetof(Surface, Area) == 72);
static_assert(sizeof(Bounds) == 64 && offsetof(Bounds, Maximum) == 32);
static_assert(sizeof(Radiance) == 48 && sizeof(Controls) == 48);
} // namespace Arvo::RayTracing::ComputePointGi
