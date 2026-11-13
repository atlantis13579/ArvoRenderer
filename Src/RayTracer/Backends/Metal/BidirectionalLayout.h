#pragma once
#include <cstdint>

namespace Arvo::RayTracing::MetalBdpt
{
// Data ABI only. Packed triples share a 16-byte lane with a scalar in MSL.
constexpr uint32_t MaximumReflections = 32, MaximumVertices = MaximumReflections + 1 + 2;
constexpr uint32_t PairSubpaths = 2, BatchCapacity = 1024, ThreadsPerGroup = 64;
constexpr uint32_t RadianceWordsPerChannel = 10, ColorChannels = 3;
constexpr uint32_t PassPixelBytes = RadianceWordsPerChannel * ColorChannels * sizeof(uint32_t);
constexpr uint32_t VertexBinding = 16, PassBinding = 17, ControlsBinding = 18, ErrorBinding = 19;
struct Vector
{
    float X, Y, Z;
};
struct Vertex
{
    Vector Position;
    uint32_t Kind;
    Vector View;
    uint32_t Material;
    Vector Normal;
    uint32_t Source;
    Vector Geometry;
    uint32_t Event;
    Vector Color;
    float Forward;
    Vector Beta;
    float Reverse;
};
struct Controls
{
    Vector Center;
    float Radius;
    uint32_t Sources, FirstPath, PathCount, Reserved;
};
static_assert(sizeof(Vertex) == 96 && sizeof(Controls) == 32);
} // namespace Arvo::RayTracing::MetalBdpt
