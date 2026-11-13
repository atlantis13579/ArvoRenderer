#pragma once
#include <cstddef>
#include <cstdint>

namespace Arvo::RayTracing::ComputeVcm
{
// Host upload/readback records only. Compute owns its HLSL density arithmetic.
constexpr unsigned MaximumVertices = 35, MaximumStrategies = MaximumVertices + 1;
constexpr unsigned DeltaScattering = 1, EndpointNotIntersectable = 2, VolumeScattering = 4;
constexpr unsigned EqualMis = 0, BalanceMis = 1, PowerMis = 2;
constexpr unsigned Connections = 1, Mergings = 2;
constexpr unsigned CameraVertex = 0, LightVertex = 1, SurfaceVertex = 2, EnvironmentVertex = 3,
                   VolumeVertex = 5;
constexpr unsigned InvalidInput = 0, NullEndpoint = 1, VertexLimit = 2, Escaped = 3, Absorbed = 4,
                   NumericalFailure = 5;
constexpr unsigned WalkThreads = 64;
constexpr unsigned SplatContribution = 1, InvalidContribution = 2;
struct Vector
{
    float X, Y, Z;
};
struct alignas(16) Density
{
    double Log2Value;
    unsigned Positive, Reserved;
};
struct alignas(16) PathVertex
{
    Vector Position;
    unsigned Kind;
    Vector ToPrevious;
    unsigned Material;
    Vector Normal;
    unsigned FrontFace;
    Vector Geometry;
    unsigned Emitter;
    Vector Color;
    unsigned Primitive;
    Vector Throughput;
    float Pdf;
    Vector Continuation;
    float ReversePdf;
    unsigned Event;
    float RelativeIndex;
    unsigned Reserved[2];
    Density Arrival, ReturnToPrevious;
};
// Vertices use a separate buffer: a complete 35-vertex path exceeds D3D12's
// maximum structured-buffer stride. VertexBase names its fixed-capacity slot.
struct alignas(16) Path
{
    unsigned Count, Limit, Mode, End;
    Vector LaunchDirection;
    float PositionPdf;
    Vector LaunchWeight;
    float DirectionPdf;
    Vector EndpointValue;
    float SelectionProbability;
    Vector EndpointNormal;
    unsigned Emitter;
    float Lens[2], Raster[2];
    unsigned EndpointFlags, VertexBase, Reserved[2];
};
struct alignas(16) WalkControls
{
    Vector EmissionCenter;
    float EmissionRadius;
    unsigned EmitterCount, PathBegin, PathCount, Seed;
    unsigned Pass, VertexLimit, OutputOffset, Reserved;
};
struct alignas(16) Statistics
{
    unsigned Primary[2], Shadow[2], Reflection[2], Tests[2];
};
struct alignas(16) CouplingQuery
{
    unsigned LightPath, CameraPath, S, T;
    float Lens[2];
    unsigned Merge, Reserved;
};
struct alignas(16) CouplingControls
{
    // Attempted global light walks and walks per full-film pixel, respectively.
    double MergeRadius, MergingSamples, LightTracingSamples;
    unsigned MaximumVertices, Heuristic, NextEventEstimation, QueryCount, QueryBegin, OutputOffset;
};
struct alignas(16) Contribution
{
    Vector Radiance;
    unsigned Flags;
    double MisWeight;
    float Raster[2];
};
struct alignas(16) IndexEntry
{
    // Encoded signed cell coordinates, or (full-film pixel, 0, 0) during splats.
    // Identity is the unsorted light-path/surface slot; UINT_MAX is inactive.
    unsigned X, Y, Z, Identity;
};
struct alignas(16) FilmPixel
{
    double Sum[3], Pass[3];
    unsigned CompletedPasses, Reserved[3];
};
struct alignas(16) Splat
{
    Vector Radiance;
    unsigned Pixel;
};
struct alignas(16) FrameControls
{
    double InitialRadius, Alpha;
    // Light paths remain global; CameraPathBase names a reusable camera batch.
    unsigned LightPathBase, LightPathCount, CameraPathBase, LensPathIndex;
    unsigned IndexCapacity, WorkBegin, WorkCount, CameraPixelBegin;
    unsigned SortSpan, SortStride, CameraPrefix, Reserved;
};
static_assert(sizeof(Density) == 16);
static_assert(sizeof(PathVertex) == 160 && sizeof(Path) == 112 && sizeof(WalkControls) == 48);
static_assert(sizeof(Statistics) == 32 && offsetof(PathVertex, Arrival) == 128);
static_assert(sizeof(CouplingQuery) == 32 && sizeof(CouplingControls) == 48 && sizeof(Contribution) == 32);
static_assert(sizeof(IndexEntry) == 16 && sizeof(FilmPixel) == 64 && sizeof(Splat) == 16);
static_assert(sizeof(FrameControls) == 64 && offsetof(FrameControls, LightPathBase) == 16);
static_assert(offsetof(CouplingControls, MaximumVertices) == 24 && offsetof(Contribution, MisWeight) == 16);
static_assert(offsetof(Path, EndpointFlags) == 96 && offsetof(Path, VertexBase) == 100);
} // namespace Arvo::RayTracing::ComputeVcm
