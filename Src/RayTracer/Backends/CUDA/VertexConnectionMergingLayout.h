#pragma once
#include <cstddef>
#include <cstdint>

namespace Arvo::RayTracing::CudaVcm
{
// Host/device ABI only. Each backend evaluates its own generalized densities.
constexpr unsigned MaximumReflections = 32;
constexpr unsigned MaximumVertices = MaximumReflections + 3;
constexpr unsigned MaximumStrategies = MaximumVertices + 1;
constexpr unsigned DeltaScattering = 1, EndpointNotIntersectable = 2, VolumeScattering = 4;
constexpr unsigned EqualMis = 0, BalanceMis = 1, PowerMis = 2;
constexpr unsigned ConnectionFamily = 1, MergingFamily = 2;
constexpr unsigned CameraVertex = 0, LightVertex = 1, SurfaceVertex = 2, EnvironmentVertex = 3,
                   VolumeVertex = 4;
constexpr unsigned InvalidWalk = 0, NullEndpoint = 1, VertexLimitReached = 2, Escaped = 3, Absorbed = 4,
                   NumericalFailure = 5;
constexpr unsigned ConfigurationError = 1, TransportError = 2;
constexpr unsigned InvalidIndex = 0xffffffffu, WalkThreadsPerBlock = 64;
constexpr unsigned TransportBatch = 64, LinearBatch = 65536;
constexpr unsigned long long MaximumPathCacheBytes = 512ull * 1024 * 1024;
struct Vector
{
    float X, Y, Z;
};

struct alignas(16) Density
{
    double Log2Value;
    unsigned Positive, Reserved;
};
struct alignas(16) VertexPdf
{
    // Light-to-camera order: Forward[i] samples x[i] from x[i-1].
    // Reverse[i] samples x[i] from x[i+1], including the geometric conversion.
    Density Forward, Reverse;
    unsigned Flags, Reserved[3];
};
struct alignas(16) Strategy
{
    Density LightEndpoint, CameraEndpoint;
    double LightSelection, SampleCount;
    unsigned Enabled, Reserved[3];
};
struct alignas(16) Weights
{
    // VC slot s has s+t=n; VM slot s has s+t=n+1. The merge at x[s-1]
    // belongs to both subpaths. Slots 2..n-1 are the finite merge candidates.
    double Connection[MaximumStrategies], Merging[MaximumStrategies];
    unsigned ActiveConnections, ActiveMergings, Reserved[2];
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
struct alignas(16) Path
{
    PathVertex Vertices[MaximumVertices];
    Vector LaunchDirection;
    float PositionPdf;
    Vector LaunchWeight;
    float DirectionPdf;
    Vector EndpointValue;
    float SourceSelection;
    Vector EndpointNormal;
    unsigned Emitter;
    float LensX, LensY, RasterX, RasterY;
    unsigned EndpointFlags, Count, Limit, Mode;
    unsigned End, Reserved[3];
};
struct alignas(16) WalkControls
{
    Vector Center;
    float WorldRadius;
    unsigned Sources, Begin, Count, Seed;
    unsigned Pass, VertexLimit, Reserved[2];
};
struct alignas(16) CouplingControls
{
    double MergeRadius, MergingSamples, LightTracingSamples;
    unsigned MaximumPathVertices, Heuristic;
    unsigned NextEventEstimation, Reserved[3];
};
constexpr unsigned SplatContribution = 1, InvalidContribution = 2;
struct alignas(16) Contribution
{
    Vector Radiance;
    unsigned Flags;
    double MisWeight;
    float RasterX, RasterY;
};
struct alignas(16) CouplingQuery
{
    unsigned LightPath, CameraPath, LightVertices, CameraVertices;
    float LensU, LensV;
    unsigned Merge, Reserved;
};
struct alignas(16) FrameControls
{
    double InitialRadius, Alpha;
    unsigned Paths, IndexCapacity, Buckets, Heuristic;
    unsigned NextEventEstimation, Reserved[3];
};
struct IndexEntry
{
    unsigned Key, Reference;
};
struct BucketRange
{
    unsigned Begin, End;
};
struct alignas(16) PixelState
{
    double Sum[3], Pass[3];
    unsigned CompletedPasses, Reserved[3];
};
static_assert(sizeof(Density) == 16 && sizeof(VertexPdf) == 48 && sizeof(Strategy) == 64);
static_assert(offsetof(Strategy, LightSelection) == 32 && offsetof(Strategy, Enabled) == 48);
static_assert(sizeof(Weights) == 592);
static_assert(sizeof(PathVertex) == 160 && sizeof(Path) == 5712 && sizeof(WalkControls) == 48);
static_assert(offsetof(PathVertex, Arrival) == 128 && offsetof(Path, LaunchDirection) == 5600);
static_assert(sizeof(CouplingControls) == 48 && sizeof(Contribution) == 32 && sizeof(CouplingQuery) == 32);
static_assert(sizeof(FrameControls) == 48 && sizeof(IndexEntry) == 8 && sizeof(BucketRange) == 8 &&
              sizeof(PixelState) == 64);
} // namespace Arvo::RayTracing::CudaVcm
