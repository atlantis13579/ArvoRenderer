#pragma once
#include <cstddef>
#include <cstdint>

namespace Arvo::RayTracing::MetalPointGi
{
constexpr uint32_t Invalid = 0xffffffffu, MaximumReflections = 32;
constexpr uint32_t Threads = 64, WorkBatch = 1024, SampleBatch = 32;
constexpr uint32_t Initialize = 0, Bake = 1, Camera = 2, StageCount = 3;
constexpr uint32_t SurfacesBinding = 16, NodesBinding = 17, IndicesBinding = 18;
constexpr uint32_t PreviousBinding = 19, NextBinding = 20, TotalBinding = 21;
constexpr uint32_t ControlsBinding = 22, ErrorsBinding = 23;
// Upload data only; scaled two-float arithmetic is implemented by Metal leaves.
struct Number
{
    float High, Low;
    int32_t Exponent;
    uint32_t Reserved = 0;
};
struct Vector
{
    float X, Y, Z;
};
struct alignas(16) Surface
{
    Vector Position;
    uint32_t Material;
    Vector GeometricNormal;
    uint32_t Primitive;
    Vector ShadingNormal;
    uint32_t HasTexCoord;
    Vector Color;
    uint32_t Patch;
    float TexCoord[2];
    uint32_t Reserved[2]{};
    Number Area, RadiusSquared;
};
struct alignas(16) Node
{
    Number Minimum[3], Maximum[3];
    uint32_t Begin, Count, Escape, Reserved = 0;
};
struct alignas(16) Radiance
{
    Number Front[3]{}, Back[3]{};
};
struct alignas(16) Controls
{
    uint32_t Surfaces, Nodes, Sources, Seed;
    uint32_t DirectSamples, GatherSamples, Begin, Count;
    uint32_t Order, FirstSample, Samples, Reserved = 0;
};
static_assert(sizeof(Number) == 16 && sizeof(Surface) == 112 && offsetof(Surface, Area) == 80);
static_assert(sizeof(Node) == 112 && offsetof(Node, Begin) == 96);
static_assert(sizeof(Radiance) == 96 && sizeof(Controls) == 48);
} // namespace Arvo::RayTracing::MetalPointGi
