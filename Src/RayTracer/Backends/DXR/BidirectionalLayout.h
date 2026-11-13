#pragma once
#include <cstdint>

namespace Arvo::RayTracing::DxrBdpt
{
// Host/shader storage contract only. Two subpaths belong to each ray generation
// invocation; a bounded batch keeps path vertices out of the DXR call stack.
constexpr uint32_t MaximumReflections = 32, MaximumVertices = MaximumReflections + 1 + 2;
constexpr uint32_t SubpathsPerPair = 2, BatchCapacity = 1024;
constexpr uint32_t RadianceWordsPerChannel = 10, ColorChannels = 3;
constexpr uint32_t PassPixelBytes = RadianceWordsPerChannel * ColorChannels * sizeof(uint32_t);
constexpr uint32_t EmitterRoot = 19, VerticesRoot = 20, PassRoot = 21, ErrorRoot = 22, ControlsRoot = 23;
constexpr uint32_t ClearPass = 0, ConnectPaths = 1, CommitPass = 2;
struct Vector
{
    float X, Y, Z;
};
struct Vertex
{
    Vector Position;
    uint32_t Kind;
    Vector Incoming;
    uint32_t Material;
    Vector ShadingNormal;
    uint32_t Emitter;
    Vector GeometricNormal;
    uint32_t Event;
    Vector Color;
    float ForwardPdf;
    Vector Throughput;
    float ReversePdf;
};
struct Controls
{
    Vector Center;
    float Radius;
    uint32_t EmitterCount, FirstPath, SampleIndex, Stage;
};
static_assert(sizeof(Vertex) == 96 && sizeof(Controls) == 32);
} // namespace Arvo::RayTracing::DxrBdpt
