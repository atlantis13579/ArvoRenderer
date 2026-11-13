#pragma once
#include <cstddef>
#include <cstdint>

namespace Arvo::RayTracing::CudaPointGi
{
// CUDA host/device data only. Geometry construction and light transport are
// owned by this backend; no CPU integrator or generated backend code is used.
inline constexpr uint32_t Invalid = 0xffffffffu, MaximumReflections = 32;
inline constexpr uint32_t ThreadsPerBlock = 64, TransportBatch = 1024;
inline constexpr uint32_t SamplesPerDispatch = 32;
inline constexpr uint32_t NumericalError = 1, BoundsError = 2, ConfigurationError = 4;
struct Vector
{
    float X, Y, Z;
};
struct alignas(16) Surfel
{
    Vector Position;
    uint32_t Material;
    Vector GeometryNormal;
    uint32_t Primitive;
    Vector ShadingNormal;
    uint32_t HasTexCoord;
    Vector Color;
    uint32_t Patch;
    float TexCoord[2];
    double Area, Radius;
    uint32_t Reserved[2]{};
};
struct alignas(16) Node
{
    double Low[3];
    uint32_t Begin, Count;
    double High[3];
    uint32_t Left = Invalid, Right = Invalid;
};
struct alignas(16) Radiance
{
    double Front[3]{}, Back[3]{};
};
struct alignas(16) Controls
{
    uint32_t Surfels, Nodes, Sources, Seed;
    uint32_t DirectSamples, GatherSamples, Begin, Count;
    uint32_t Order, SampleBegin, SampleCount, Reserved = 0;
};
static_assert(sizeof(Surfel) == 96 && offsetof(Surfel, Area) == 72 && offsetof(Surfel, Radius) == 80);
static_assert(sizeof(Node) == 64 && offsetof(Node, Begin) == 24 && offsetof(Node, High) == 32);
static_assert(sizeof(Radiance) == 48 && sizeof(Controls) == 48);
} // namespace Arvo::RayTracing::CudaPointGi
