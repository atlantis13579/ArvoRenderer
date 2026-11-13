#pragma once
#include <cstddef>
#include <cstdint>
namespace Arvo::RayTracing::MetalVcm
{
// Data only. Two binary32 significands plus an integer exponent; not binary64.
constexpr uint32_t MaximumReflections = 32, MaximumVertices = MaximumReflections + 3;
constexpr uint32_t MaximumStrategies = MaximumVertices + 1;
constexpr uint32_t VolumeScattering = 4, VolumeVertex = 5;
struct Number
{
    float High, Low;
    int32_t Exponent;
    uint32_t Reserved;
};
static_assert(sizeof(Number) == 16);
struct Vector
{
    Number X, Y, Z;
};
struct SceneControls
{
    float CameraPosition[3];
    uint32_t FullWidth;
    float CameraForward[3];
    uint32_t FullHeight;
    float CameraRight[3];
    uint32_t TotalSamples;
    float CameraUp[3];
    uint32_t MaxReflections;
    float HalfWidth, HalfHeight, LensRadius, FocusDistance;
    uint32_t MaterialCount, NodeCount, TriangleCount, Acceleration;
    uint32_t InstanceNodeCount, TextureCount, TexelCount, TransportFlags;
    float EmissionCenter[3];
    float EmissionRadius;
    uint32_t SourceCount, Sequence, Seed, SamplingSeed;
};
struct WalkDispatch
{
    uint32_t FirstPath, PathCount, Pass, OutputPathBegin;
    uint32_t OutputVertexBegin, VertexStride, PathCapacity, VertexCapacity;
    uint32_t StatisticsCapacity, SourceCapacity, LightPopulation, Reserved;
};
struct WalkVertex
{
    float Position[3];
    uint32_t Kind;
    float Normal[3];
    uint32_t FrontFace;
    float Geometry[3];
    uint32_t Material;
    float Incoming[3];
    uint32_t Primitive;
    float Color[3];
    uint32_t Source;
    float Outgoing[3];
    uint32_t Event;
    Vector Throughput;
    Number Arrival, ReverseArrival, ForwardAngular, ReverseAngular;
    float TexCoord[2];
    uint32_t HasTexCoord, Reserved;
};
struct Walk
{
    uint32_t Begin, Count, Mode, End;
    uint32_t Identity, Pass, Source, EndpointFlags;
    float Raster[2], LensPoint[2], LaunchDirection[3];
    uint32_t Reserved;
    Number PositionPdf, DirectionPdf, Selection;
};
struct WalkStatistics
{
    uint32_t Camera[2], Transport[2], Tests[2], Shadow[2];
};
static_assert(sizeof(SceneControls) == 144 && sizeof(WalkDispatch) == 48 && sizeof(WalkStatistics) == 32);
static_assert(sizeof(WalkVertex) == 224 && sizeof(Walk) == 112);
static_assert(offsetof(WalkVertex, Throughput) == 96 && offsetof(WalkVertex, Arrival) == 144);
static_assert(offsetof(Walk, PositionPdf) == 64 && offsetof(SceneControls, SourceCount) == 128);
struct CouplingControls
{
    Number Radius, MergingSamples, LightTracingSamples;
    uint32_t MaximumVertices, LightVertexLimit, CameraVertexLimit, Heuristic;
    uint32_t Families, NextEventEstimation, PathCapacity, VertexCapacity;
    uint32_t SourceCapacity, QueryCount, QueryBegin, OutputCapacity;
    uint32_t QueryCapacity, StatisticsCapacity, Reserved0, Reserved1;
};
struct CouplingQuery
{
    uint32_t LightPath, CameraPath, LightCount, CameraCount, Merge, Reserved;
    float LensVariates[2];
};
struct Contribution
{
    Vector Radiance;
    Number Weight;
    float Raster[2];
    uint32_t Splat, Accepted;
};
static_assert(sizeof(CouplingControls) == 112 && sizeof(CouplingQuery) == 32 && sizeof(Contribution) == 80);
static_assert(offsetof(CouplingControls, MaximumVertices) == 48 && offsetof(Contribution, Weight) == 48);
struct FrameControls
{
    uint32_t TileX, TileY, Width, Height;
    uint32_t Pass, LightBegin, LightCount, SlotsPerLight;
    uint32_t CameraBegin, CameraCount, IndexCapacity, PixelCapacity;
    uint32_t SplatCapacity, StatisticsCapacity, WorkBegin, WorkCount;
    uint32_t SortWidth, SortStride, SortKind, Reserved;
};
struct IndexEntry
{
    int32_t CellX, CellY, CellZ;
    uint32_t Path, Vertex, Identity, Key, Valid;
};
struct Splat
{
    Vector Radiance;
    uint32_t Pixel, Identity, Vertex, Valid;
};
struct PixelState
{
    Vector Sum, Estimate;
    uint32_t CameraPath, LightPath, ConnectionS, ConnectionT;
    uint32_t MergeT, Neighbor, Cursor, Phase;
    uint32_t SplatCursor, CompletedPasses, Pass, Reserved;
};
struct FilmPixel
{
    float Mean[3];
    uint32_t Samples;
};
static_assert(sizeof(FrameControls) == 80 && sizeof(IndexEntry) == 32 && sizeof(Splat) == 64);
static_assert(sizeof(PixelState) == 144 && sizeof(FilmPixel) == 16);
static_assert(offsetof(PixelState, CameraPath) == 96 && offsetof(Splat, Pixel) == 48);
} // namespace Arvo::RayTracing::MetalVcm
