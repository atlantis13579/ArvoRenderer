#pragma once
#include <cstddef>
#include <cstdint>

namespace Arvo::RayTracing::DxrPointGi
{
// Data ABI for the single DXR library shared by offline jobs and viewport jobs.
constexpr uint32_t Invalid = 0xffffffffu, MaximumDepth = 32;
constexpr uint32_t PopulationBatch = 1024, SampleBatch = 32;
constexpr uint32_t InitializeStage = 0, BakeStage = 1, CameraStage = 2;
constexpr uint32_t SourceRoot = 19, SurfaceRoot = 20, BoundsRoot = 21, IndexRoot = 22;
constexpr uint32_t PreviousRoot = 23, NextRoot = 24, TotalRoot = 25, ErrorRoot = 26, ControlsRoot = 27;
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
    Vector Shading;
    uint32_t HasTexCoord;
    Vector Color;
    uint32_t Patch;
    float TexCoord[2];
    uint32_t Reserved[2]{};
    double RadiusSquared, Area;
};
struct alignas(16) Node
{
    double Minimum[3];
    uint32_t Left = Invalid, Right = Invalid;
    double Maximum[3];
    uint32_t Begin = 0, Count = 0;
};
struct alignas(16) Radiance
{
    double Front[3]{}, Back[3]{};
};
struct alignas(16) Controls
{
    uint32_t Stage, Begin, Count, Order;
    uint32_t Surfaces, Nodes, Sources, Seed;
    uint32_t DirectSamples, GatherSamples, SampleBegin, SampleCount;
};
static_assert(sizeof(Surface) == 96 && offsetof(Surface, RadiusSquared) == 80);
static_assert(sizeof(Node) == 64 && offsetof(Node, Maximum) == 32);
static_assert(sizeof(Radiance) == 48 && sizeof(Controls) == 48);
} // namespace Arvo::RayTracing::DxrPointGi
