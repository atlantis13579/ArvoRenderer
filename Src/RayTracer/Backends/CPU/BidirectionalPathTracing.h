#pragma once
#include "RayTracer/Backends/CPU/CameraEndpoint.h"
#include "RayTracer/Backends/CPU/EmitterEndpoint.h"
#include "RayTracer/Backends/CPU/SurfaceScattering.h"
#include "Renderer/BidirectionalPath.h"

namespace Arvo::RayTracing::Cpu
{
enum class BdptVertexType
{
    Camera,
    Emitter,
    Surface,
    Volume,
    Infinite
};
enum class BdptTermination
{
    VertexLimit,
    Escaped,
    Absorbed,
    NullEndpoint,
    Cancelled,
    InvalidInput,
    NumericalFailure
};

struct BdptVertex
{
    BdptVertexType Type = BdptVertexType::Surface;
    Float3 Position, ToPrevious, SurfaceColor, Throughput;
    SurfaceHit Hit;
    // The continuation is a sampled event, not a property of the material.
    // A connector must replace it when it changes either neighboring direction.
    SurfaceSample Continuation;
    PathDensity ArrivingDensity, ReturningDensity;
    uint32_t EmitterIndex = InvalidEmitterIndex;
};
struct BdptSubpath
{
    std::array<BdptVertex, MaximumBidirectionalVertices> Vertices{};
    uint32_t Count = 0;
    uint32_t VertexLimit = 0;
    TransportMode Mode = TransportMode::Radiance;
    BdptTermination Termination = BdptTermination::InvalidInput;
    CameraEndpointSample Camera;
    EmitterEndpoint Light;
};
struct BdptWalkOptions
{
    // Includes the endpoint. A terminal surface is kept without sampling a
    // discarded continuation. Complete-path depth is bounded by the connector.
    uint32_t MaximumVertices = MaximumBidirectionalVertices;
    bool ImportanceSampling = true;
};

// CPU BDPT execution interfaces. These fixed-depth walks have no roulette;
// adding survival requires strategy-dependent forward AND reverse probabilities.
BdptSubpath GenerateCameraSubpath(const Scene&, const SceneAccelerator&, const CameraFrame&, Float2 raster,
                                  const SampleContext&, BdptWalkOptions, AtomicStatistics&,
                                  const CancellationProbe&, const PreparedEmissionScene* emitters = nullptr);
BdptSubpath GenerateLightSubpath(const Scene&, const SceneAccelerator&, const PreparedEmissionScene&,
                                 const SampleContext&, BdptWalkOptions, AtomicStatistics&,
                                 const CancellationProbe&);

// Reorient shading/medium state for the completed path, retaining the original
// geometric normal. Legacy materials use their authored diffuse/mirror mixture.
SurfaceEvaluation EvaluateBidirectionalScattering(const Scene&, const BdptVertex&, Float3 view,
                                                  Float3 outgoing, TransportMode, bool importance);
SurfaceSample SampleBidirectionalScattering(const Scene&, const BdptVertex&, Float2 variates,
                                            float lobeVariate, TransportMode, bool importance);
struct BdptConnection
{
    Float3 Radiance;
    Float2 Raster;
    float MisWeight = 0;
    bool Splat = false;
};
// Surface connections (s >= 2, t >= 1), including explicit lens sampling at
// t=1. Returned radiance already includes visibility and complete-path MIS.
// Weights include the alternative s=0/1 strategies, evaluated separately by the
// frame integrator using emitter hits and direct endpoint connections. The t=0
// strategy is absent. NEE is enabled and all strategies have equal sample counts.
// A t=1 result is a full-image splat, not an additional sample of its source pixel.
BdptConnection ConnectBidirectionalSubpaths(const Scene&, const SceneAccelerator&,
                                            const PreparedEmissionScene&, const CameraFrame&,
                                            const BdptSubpath& light, uint32_t lightCount,
                                            const BdptSubpath& camera, uint32_t cameraCount,
                                            Float2 lensVariates, bool importance, MisHeuristic,
                                            AtomicStatistics&, const CancellationProbe&);
BdptConnection ConnectBidirectionalEmitter(const Scene&, const SceneAccelerator&,
                                           const PreparedEmissionScene&, const BdptSubpath& light,
                                           const BdptSubpath& camera, uint32_t cameraCount, bool importance,
                                           MisHeuristic, AtomicStatistics&, const CancellationProbe&);
BdptConnection EvaluateBidirectionalEmitterHit(const Scene&, const PreparedEmissionScene&,
                                               const BdptSubpath& camera, uint32_t cameraCount,
                                               uint32_t lightVertexLimit, bool importance, MisHeuristic);
using BdptDispatcher = std::function<void(uint32_t, const std::function<void(uint32_t)>&)>;
// Full-image light samples are also generated for cropped output. Deterministic
// bounded batches keep cross-pixel splats separate until a complete sample pass.
bool RenderBidirectionalFrame(const Scene&, const SceneAccelerator&, const CameraFrame&,
                              const RenderSettings&, LinearFilm&, AtomicStatistics&, const CancellationProbe&,
                              const BdptDispatcher&, std::string& error);
} // namespace Arvo::RayTracing::Cpu
