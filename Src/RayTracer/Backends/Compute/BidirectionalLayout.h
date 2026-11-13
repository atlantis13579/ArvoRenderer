#pragma once
#include <cstdint>
namespace Arvo::RayTracing::ComputeBdpt
{
// Data ABI only. Each thread owns two fixed-capacity subpaths in a bounded UAV.
constexpr uint32_t MaximumReflections = 32, MaximumScattering = MaximumReflections + 1;
constexpr uint32_t EndpointCount = 2, MaximumVertices = MaximumScattering + EndpointCount;
constexpr uint32_t BatchCapacity = 1024, ThreadsPerGroup = 64, FilmGroupWidth = 8;
constexpr uint32_t RadianceWordsPerChannel = 10, ColorChannels = 3;
constexpr uint32_t PassPixelBytes = RadianceWordsPerChannel * ColorChannels * sizeof(uint32_t);
constexpr uint32_t SceneDescriptors = 10, FrameDescriptors = 8,
                   DescriptorCount = SceneDescriptors + FrameDescriptors;
struct Vector
{
    float X, Y, Z;
};
struct Vertex
{
    Vector Position;
    uint32_t Kind;
    Vector Previous;
    uint32_t Material;
    Vector Normal;
    uint32_t FrontFace;
    Vector Geometry;
    uint32_t Emitter;
    Vector Color;
    float Pdf;
    Vector Throughput;
    float ReversePdf;
    uint32_t Event, Reserved0, Reserved1, Reserved2;
};
struct Controls
{
    Vector Center;
    float Radius;
    uint32_t EmitterCount, PathBegin, PathCount, Reserved;
};
static_assert(sizeof(Vertex) == 112 && sizeof(Controls) == 32);
} // namespace Arvo::RayTracing::ComputeBdpt
