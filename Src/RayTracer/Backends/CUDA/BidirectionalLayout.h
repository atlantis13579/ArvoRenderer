#pragma once
#include <cstdint>

namespace Arvo::RayTracing::CudaBdpt
{
// Host/device data ABI only. Paths live in bounded global-memory batches;
// putting two maximum-depth paths on each CUDA thread's stack is unnecessary.
constexpr unsigned MaximumReflections = 32;
constexpr unsigned MaximumVertices = MaximumReflections + 3;
constexpr unsigned BatchCapacity = 1024;
constexpr unsigned ThreadsPerBlock = 64;
constexpr unsigned RadianceWordsPerChannel = 10, ColorChannels = 3;
constexpr unsigned PassWords = RadianceWordsPerChannel * ColorChannels;
constexpr unsigned PassPixelBytes = PassWords * sizeof(unsigned);
constexpr unsigned CameraVertex = 0, EmitterVertex = 1, SurfaceVertex = 2, InfiniteVertex = 3,
                   VolumeVertex = 4;
constexpr unsigned NumericalFailure = 1;
struct Vector
{
    float X, Y, Z;
};
struct alignas(16) Vertex
{
    Vector Position;
    unsigned Type;
    Vector Previous;
    unsigned Material;
    Vector Normal;
    unsigned FrontFace;
    Vector Geometry;
    unsigned Emitter;
    Vector Color;
    float Pdf;
    Vector Beta;
    float ReversePdf;
    unsigned Event, Reserved0, Reserved1, Reserved2;
};
struct alignas(16) Path
{
    Vertex Vertices[MaximumVertices];
    Vector LaunchDirection;
    float PositionPdf;
    Vector LaunchWeight;
    float DirectionPdf;
    Vector EndpointValue;
    float SelectionProbability;
    float LensX, LensY, RasterX, RasterY;
    unsigned EndpointFlags, Count, Limit, Mode;
};
struct Pair
{
    Path Light, Camera;
};
struct alignas(16) Controls
{
    Vector WorldCenter;
    float WorldRadius;
    unsigned EmitterCount, PathBegin, PathCount, Reserved;
};
static_assert(sizeof(Vertex) == 112 && sizeof(Path) == 4000 && sizeof(Pair) == 8000);
static_assert(sizeof(Controls) == 32);
} // namespace Arvo::RayTracing::CudaBdpt
