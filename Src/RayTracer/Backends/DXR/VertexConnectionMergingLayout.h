#pragma once
#include <cstddef>
#include <cstdint>
namespace Arvo::RayTracing::DxrVcm
{
// Data only. DXR owns its HLSL evaluation; the host never substitutes CPU MIS.
constexpr uint32_t MaximumReflections = 32, MaximumScattering = MaximumReflections + 1;
constexpr uint32_t MaximumVertices = MaximumScattering + 2;
constexpr uint32_t MaximumStrategies = MaximumVertices + 1;
constexpr uint32_t Connections = 1, Mergings = 2;
constexpr uint32_t VolumeScattering = 4, VolumeVertex = 5;
constexpr uint32_t EqualMis = 0, BalanceMis = 1, PowerMis = 2;
struct Density
{
    double Log2Value;
    uint32_t Positive, Reserved;
};
constexpr uint32_t SourceRoot = 19, VerticesRoot = 20, PathsRoot = 21;
constexpr uint32_t ErrorsRoot = 22, StatisticsRoot = 23, ControlsRoot = 24;
constexpr uint32_t QueriesRoot = 25, ContributionsRoot = 26, CouplingControlsRoot = 27;
constexpr uint32_t IndexRoot = 28, FramePixelsRoot = 29, SplatsRoot = 30, FrameControlsRoot = 31;
constexpr uint32_t RootParameters = FrameControlsRoot + 1;
constexpr uint32_t CameraStage = 0, LightStage = 1, CouplingStage = 2;
constexpr uint32_t InitializeStage = 3, GridStage = 4, SortStage = 5, EstimateStage = 6;
constexpr uint32_t MergeStage = 7, SplatStage = 8, CommitStage = 9;
// The shader table selects a dedicated ray-generation program for each stage.
// Stage numbers still describe the same persistent buffers and dispatch schedule.
inline constexpr const wchar_t* StageEntryPoints[] = {
    L"RayGen", L"VcmLight", L"VcmCoupling", L"VcmInitialize", L"VcmGrid",
    L"VcmSort", L"VcmEstimate", L"VcmMerge", L"VcmSplat", L"VcmCommit"};
constexpr uint32_t MergeWorkBudget = 64, SplatReductionBudget = 1024;
struct IndexEntry
{
    uint32_t Key, Identity;
    int32_t CellX, CellY, CellZ;
    uint32_t Reserved0, Reserved1, Reserved2;
};
struct FramePixel
{
    double SumRed, SumGreen, SumBlue;
    double CameraRed, CameraGreen, CameraBlue;
    double SplatRed, SplatGreen, SplatBlue;
    uint32_t CompletedPasses, MergePrefix, Neighbor, Cursor;
    uint32_t Phase, Reserved0, Reserved1, Reserved2;
};
struct Splat
{
    float Radiance[3];
    uint32_t FullPixel;
};
struct FrameControls
{
    double InitialRadius, Alpha;
    uint32_t LightPathBase, LightPathCount, CameraPathBase, LensPathIndex;
    uint32_t IndexCapacity, BucketCount, WorkBegin, WorkCount;
    uint32_t CameraPixelBegin, SortSpan, SortDistance, Reserved;
};
struct PathVertex
{
    float Position[3];
    uint32_t Kind;
    float Incoming[3];
    uint32_t Material;
    float ShadingNormal[3];
    uint32_t FrontFace;
    float GeometricNormal[3];
    uint32_t Emitter;
    float Color[3];
    uint32_t Primitive;
    float Throughput[3], ForwardPdf;
    float Outgoing[3], ReversePdf;
    uint32_t Event;
    float RelativeIndex;
    uint32_t Reserved0, Reserved1;
    Density Arrival, PreviousArrival;
};
struct Path
{
    uint32_t First, Count, Limit, Mode;
    float LaunchDirection[3], DirectionPdf;
    float LaunchWeight[3], PositionPdf;
    float EndpointValue[3], SourceProbability;
    float EndpointNormal[3];
    uint32_t SourceIndex;
    float Raster[2], LensPoint[2];
    uint32_t EndpointFlags, End, Reserved0, Reserved1;
};
struct WalkStatistics
{
    uint32_t Primary[2], Shadow[2], Reflection[2], Completed[2];
};
struct WalkControls
{
    float EmissionCenter[3], EmissionRadius;
    uint32_t EmitterCount, PathBegin, PathCount, OutputBegin;
    uint32_t PassIndex, PopulationSeed, VertexLimit, Stage;
    uint32_t PathCapacity, VertexCapacity, StatisticsCapacity, SourceCapacity;
};
struct CouplingQuery
{
    uint32_t LightPath, EyePath, S, T;
    float Lens[2];
    uint32_t Merge, Reserved;
};
struct Contribution
{
    float Radiance[3];
    uint32_t Flags;
    double MisWeight;
    float Raster[2];
    uint32_t ShadowRays, Reserved0, Reserved1, Reserved2;
};
struct CouplingControls
{
    double MergeRadius, MergingSamples, LightTracingSamples;
    uint32_t MaximumVertices, Heuristic, NextEventEstimation, Count;
    uint32_t Begin, OutputBegin;
    uint32_t QueryCapacity, ContributionCapacity, Reserved0, Reserved1;
};
static_assert(sizeof(Density) == 16);
static_assert(sizeof(PathVertex) == 160 && offsetof(PathVertex, Arrival) == 128);
static_assert(sizeof(Path) == 112 && offsetof(Path, EndpointFlags) == 96);
static_assert(sizeof(WalkControls) == 64 && offsetof(WalkControls, PathCapacity) == 48);
static_assert(sizeof(WalkStatistics) == 32);
static_assert(sizeof(CouplingQuery) == 32 && sizeof(Contribution) == 48 && sizeof(CouplingControls) == 64);
static_assert(offsetof(Contribution, MisWeight) == 16 && offsetof(Contribution, ShadowRays) == 32);
static_assert(offsetof(CouplingControls, MaximumVertices) == 24 && offsetof(CouplingControls, Begin) == 40);
static_assert(sizeof(IndexEntry) == 32 && offsetof(IndexEntry, CellX) == 8);
static_assert(sizeof(FramePixel) == 104 && offsetof(FramePixel, Phase) == 88);
static_assert(sizeof(Splat) == 16 && sizeof(FrameControls) == 64);
static_assert(offsetof(FrameControls, IndexCapacity) == 32 &&
              offsetof(FrameControls, CameraPixelBegin) == 48);
} // namespace Arvo::RayTracing::DxrVcm
