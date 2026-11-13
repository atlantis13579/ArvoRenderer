#pragma once
#include "VertexConnectionMergingLayout.h"
#include "Renderer/RenderTechniques.h"
#include <array>
#include <functional>
#include <limits>
#include <string>

namespace Arvo::RayTracing::MetalVcm
{
enum class Stage : uint32_t
{
    LightWalks,
    CameraWalks,
    BuildGrid,
    SortIndex,
    ClearFilm,
    CameraFrame,
    ProjectLightPaths,
    CommitFilm,
    Count
};
enum class Storage : uint32_t
{
    Vertices,
    Paths,
    Statistics,
    Index,
    Pixels,
    Splats,
    Film,
    Pending,
    Errors,
    Count
};
inline constexpr uint32_t StageCount = uint32_t(Stage::Count), StorageCount = uint32_t(Storage::Count);
inline constexpr const char* EntryPoints[StageCount] = {
    "ArvoVcmLightWalks", "ArvoVcmCameraWalks", "ArvoVcmBuildGrid",         "ArvoVcmSortIndex",
    "ArvoVcmClearFilm",  "ArvoVcmCameraFrame", "ArvoVcmProjectLightPaths", "ArvoVcmCommitFilm"};
inline constexpr uint32_t TransportBatch = 64, LinearBatch = 16384, ThreadsPerGroup = 32;
enum Binding : uint32_t
{
    NodesBinding = 0,
    ReferencesBinding,
    TrianglesBinding,
    SpheresBinding,
    MaterialsBinding,
    SourcesBinding,
    SceneBinding,
    TexturesBinding,
    TexelsBinding,
    InstancesBinding,
    InstanceNodesBinding,
    ErrorsBinding,
    ControlsBinding,
    VerticesBinding,
    PathsBinding,
    StatisticsBinding,
    FrameBinding,
    IndexBinding,
    PixelsBinding,
    SplatsBinding,
    FilmBinding,
    PendingBinding
};
struct FrameRequest
{
    SceneControls Scene{};
    uint32_t TileX = 0, TileY = 0, Width = 0, Height = 0;
    VertexConnectionMergingOptions Options;
    MisHeuristic Mis = MisHeuristic::Balance;
    bool NextEventEstimation = true;
    uint32_t TransportLimit = TransportBatch, LinearLimit = LinearBatch;
    uint64_t MaximumBufferBytes = std::numeric_limits<size_t>::max();
};
struct FramePlan
{
    SceneControls Scene{};
    FrameControls Frame{};
    CouplingControls Coupling{};
    uint32_t VertexStride = 0, TransportLimit = 0, LinearLimit = 0;
    std::array<size_t, StorageCount> Bytes{};
    double InitialRadius = 0, Alpha = 0;
};
struct Dispatch
{
    Stage Kernel = Stage::ClearFilm;
    WalkDispatch Walks{};
    FrameControls Frame{};
    CouplingControls Coupling{};
    uint32_t Lanes = 0, StatisticsRows = 0;
};
// Execute must retire the command (or retain all resources in a terminal device
// state), read errors/statistics, and reset/read the pending word per dispatch.
// A successful callback guarantees visibility to the next dispatch. Publish is
// called only after every pixel has committed the complete pass.
struct FrameCallbacks
{
    std::function<bool(const Dispatch&, uint32_t& pending, std::string& error)> Execute;
    std::function<bool(uint32_t completedPasses, std::string& error)> Publish;
    std::function<bool()> Cancelled;
};
bool PrepareFrame(const FrameRequest&, FramePlan&, std::string& error);
bool ExecuteFrame(const FramePlan&, const FrameCallbacks&, std::string& error);
} // namespace Arvo::RayTracing::MetalVcm
