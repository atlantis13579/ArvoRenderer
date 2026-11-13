#pragma once
#include "RayTracer/Backends/CPU/CameraEndpoint.h"
#include "RayTracer/Backends/CPU/EmitterEndpoint.h"
#include "RayTracer/Backends/CPU/SurfaceScattering.h"
#include "Renderer/VertexConnectionMergingPath.h"

namespace Arvo::RayTracing::Cpu
{
enum class VcmVertexKind
{
    Camera,
    Light,
    Surface,
    Volume,
    Environment
};
enum class VcmWalkEnd
{
    InvalidInput,
    NullEndpoint,
    VertexLimit,
    Escaped,
    Absorbed,
    Cancelled,
    NumericalFailure
};
struct VcmVertex
{
    VcmVertexKind Kind = VcmVertexKind::Surface;
    Float3 Position, ToPrevious, Color, Throughput;
    SurfaceHit Hit;
    SurfaceSample Continuation;
    VcmDensity Arrival, ReturnToPrevious;
    uint32_t Emitter = InvalidEmitterIndex;
    uint32_t EnclosingMaterial = InvalidRayPrimitiveIndex;
};
struct VcmSubpath
{
    std::array<VcmVertex, MaximumVcmVertices> Vertices{};
    uint32_t Count = 0, VertexLimit = 0;
    TransportMode Mode = TransportMode::Radiance;
    VcmWalkEnd End = VcmWalkEnd::InvalidInput;
    CameraEndpointSample Camera;
    EmitterEndpoint Light;
};
struct VcmWalkOptions
{
    // Includes the endpoint; the last surface keeps no unused continuation.
    uint32_t VertexLimit = MaximumVcmVertices;
    bool ImportanceSampling = true;
};
// The VCM estimator owns these walks. No other transport integrator is called.
// Fixed depth avoids an unaccounted survival PMF in alternative path strategies.
VcmSubpath TraceVcmCameraPath(const Scene&, const SceneAccelerator&, const PreparedEmissionScene&,
                              const CameraFrame&, Float2 raster, const SampleContext&, VcmWalkOptions,
                              AtomicStatistics&, const CancellationProbe&);
VcmSubpath TraceVcmLightPath(const Scene&, const SceneAccelerator&, const PreparedEmissionScene&,
                             const SampleContext&, VcmWalkOptions, AtomicStatistics&,
                             const CancellationProbe&);
// Connections and merges replace directions at their endpoints. Re-evaluate the
// mixed BSDF there instead of reusing the walk's discarded continuation event.
SurfaceEvaluation EvaluateVcmScattering(const Scene&, const VcmVertex&, Float3 view, Float3 outgoing,
                                        TransportMode, bool importance);
SurfaceSample SampleVcmScattering(const Scene&, const VcmVertex&, Float2 variates, float lobeVariate,
                                  TransportMode, bool importance);
struct VcmCouplingOptions
{
    uint32_t MaximumVertices = MaximumVcmVertices;
    double MergeRadius = 0, VolumeMergeRadius = 0, MergingSamples = 0;
    // Light paths / full-image pixels. Camera splats use this relative strategy
    // count in MIS; the frame divides their accumulated value by the same ratio.
    double LightTracingSamples = 1;
    bool ImportanceSampling = true, NextEventEstimation = true;
    MisHeuristic Mis = MisHeuristic::Power;
};
struct VcmContribution
{
    Float3 Radiance;
    Float2 Raster;
    double MisWeight = 0;
    bool Splat = false, NumericalFailure = false;
};
// s=0 evaluates an emitter hit; s=1 connects the sampled source; s>=2 connects
// a finite light vertex. t=1 samples the camera lens and returns a full-image splat.
VcmContribution ConnectVcmSubpaths(const Scene&, const SceneAccelerator&, const PreparedEmissionScene&,
                                   const CameraFrame&, const VcmSubpath& light, uint32_t s,
                                   const VcmSubpath& camera, uint32_t t, Float2 lensVariates,
                                   const VcmCouplingOptions&, AtomicStatistics&, const CancellationProbe&);
// The two endpoints represent one path vertex. Surface merges use a disk;
// volume merges use a sphere. The attempted light population enters joint MIS.
VcmContribution MergeVcmSubpaths(const Scene&, const PreparedEmissionScene&, const VcmSubpath& light,
                                 uint32_t s, const VcmSubpath& camera, uint32_t t,
                                 const VcmCouplingOptions&, const SceneAccelerator* = nullptr);
struct VcmDiagnostics
{
    uint64_t EmittedPaths = 0, MergedPairs = 0, CameraSplats = 0;
    uint32_t CompletedPasses = 0;
    double InitialRadius = 0, LastRadius = 0;
};
// Pass zero uses the initial radius. No lower radius clamp changes the limit.
double VcmPassRadius(double initialRadius, double alpha, uint32_t pass);
double VcmPassVolumeRadius(double initialRadius, double alpha, uint32_t pass);
using VcmDispatcher = std::function<void(uint32_t, const std::function<void(uint32_t)>&)>;
bool RenderVertexConnectionMergingFrame(const Scene&, const SceneAccelerator&, const CameraFrame&,
                                        const RenderSettings&, LinearFilm&, AtomicStatistics&,
                                        const CancellationProbe&, const VcmDispatcher&, std::string& error,
                                        VcmDiagnostics* = nullptr);
} // namespace Arvo::RayTracing::Cpu
