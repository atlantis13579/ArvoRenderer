#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace Arvo::RayTracing
{
// Values 0 and 1 are retained for existing scene adapters and shader ABIs.
enum class Algorithm : uint32_t
{
    RayTracing,
    PathTracing,
    DistributedRayTracing,
    BidirectionalPathTracing,
    MetropolisLightTransport,
    VertexConnectionMerging,
    PhotonMapping,
    ProgressivePhotonMapping,
    PointBasedGlobalIllumination,
    Count
};

enum class BackendType : uint8_t;
enum class MisHeuristic : uint32_t { None, Balance, Power };
enum class SampleSequence : uint32_t { Independent, Stratified, Sobol, BlueNoise, Count };
enum class AccelerationType : uint32_t { Bvh, KdTree, Bih, Count };
enum class BvhBuildPolicy : uint32_t { Default, SpatialSplits, Count };
// Scene conversion policy; independent of the per-object acceleration structure.
enum class GeometryMode : uint32_t { Flat, Instanced, Count };
// Reconstruction is independent of transport. Only implemented methods may be selected.
enum class ReconstructionFilter : uint32_t
{
    None,
    SpatialAtrous,
    Temporal,
    Ai,
    Count
};
struct ReconstructionOptions
{
    ReconstructionFilter Filter = ReconstructionFilter::None;
    uint32_t Passes = 3;
    // Linear-radiance range sigma: absolute + relative * max(luminance) +
    // noise * sqrt(raw mean variances). Moment guidance is a heuristic.
    float NoiseScale = 2;
    float RelativeColorScale = 0.05f;
    float AbsoluteColorScale = 0.01f;
    // Deterministic primary geometry is meaningful for static pinhole surfaces.
    // Normal sigma is a unit-vector distance; plane sigma is relative to depth.
    bool GeometryGuidance = false;
    float NormalScale = 0.25f;
    float RelativePlaneScale = 0.01f;
    bool operator==(const ReconstructionOptions& other) const
    {
        return Filter == other.Filter && Passes == other.Passes && NoiseScale == other.NoiseScale &&
               RelativeColorScale == other.RelativeColorScale &&
               AbsoluteColorScale == other.AbsoluteColorScale && GeometryGuidance == other.GeometryGuidance &&
               NormalScale == other.NormalScale && RelativePlaneScale == other.RelativePlaneScale;
    }
    bool operator!=(const ReconstructionOptions& other) const { return !(*this == other); }
};
bool ValidateReconstructionOptions(const ReconstructionOptions&, std::string& error);
const char* ReconstructionFilterKey(ReconstructionFilter filter);
bool ParseReconstructionFilter(std::string_view key, ReconstructionFilter& filter);
uint32_t ReconstructionHalo(const ReconstructionOptions&);
bool ReconstructionUsesMoments(Algorithm);
const char* GeometryModeKey(GeometryMode mode);
bool ParseGeometryMode(std::string_view key, GeometryMode& mode);

struct AccelerationOptions
{
    AccelerationType Type = AccelerationType::Bvh;
    BvhBuildPolicy BvhPolicy = BvhBuildPolicy::Default;
    constexpr AccelerationOptions() = default;
    constexpr AccelerationOptions(AccelerationType type, BvhBuildPolicy policy = BvhBuildPolicy::Default)
        : Type(type), BvhPolicy(policy) {}
    bool operator==(const AccelerationOptions& other) const
    {
        return Type == other.Type && BvhPolicy == other.BvhPolicy;
    }
    bool operator!=(const AccelerationOptions& other) const { return !(*this == other); }
};

struct AccelerationInfo
{
    AccelerationType Id;
    const char* Key;
    const char* Label;
    bool Implemented;
};
const AccelerationInfo* FindAcceleration(AccelerationType type);
bool ParseAcceleration(std::string_view key, AccelerationType& type);
bool ValidateAcceleration(AccelerationType type, std::string& error);
bool ValidateAcceleration(AccelerationOptions options, std::string& error);
const char* BvhBuildKey(BvhBuildPolicy policy);
bool ParseBvhBuildPolicy(std::string_view key, BvhBuildPolicy& policy);

struct SamplerInfo
{
    SampleSequence Id;
    const char* Key;
    const char* Label;
};
const SamplerInfo* FindSampler(SampleSequence sequence);
bool ParseSampler(std::string_view key, SampleSequence& sequence);

struct AdaptiveOptions
{
    bool Enabled = false;
    uint32_t PilotSamples = 16;
    uint32_t MinimumSamples = 16;
    float RelativeError = 0.05f;
    float AbsoluteError = 0.001f;
    // Two fixed independent pilots are filtered before selecting production counts.
    bool DenoiserAware = false;
    float MinimumVarianceFraction = 0.1f;

    bool operator==(const AdaptiveOptions& other) const
    {
        return Enabled == other.Enabled && PilotSamples == other.PilotSamples
            && MinimumSamples == other.MinimumSamples && RelativeError == other.RelativeError
            && AbsoluteError == other.AbsoluteError && DenoiserAware == other.DenoiserAware
            && MinimumVarianceFraction == other.MinimumVarianceFraction;
    }
    bool operator!=(const AdaptiveOptions& other) const { return !(*this == other); }
};
bool ValidateAdaptiveOptions(Algorithm algorithm, const AdaptiveOptions& options,
    uint32_t maximumSamples, std::string& error);

struct TransportOptions
{
    bool ImportanceSampling = true;
    // Area-light strategy. Delta lights always require explicit evaluation.
    bool NextEventEstimation = true;
    MisHeuristic Mis = MisHeuristic::None;
    bool RussianRoulette = false;
    uint32_t RouletteStartDepth = 3;

    bool operator==(const TransportOptions& other) const
    {
        return ImportanceSampling == other.ImportanceSampling
            && NextEventEstimation == other.NextEventEstimation && Mis == other.Mis
            && RussianRoulette == other.RussianRoulette && RouletteStartDepth == other.RouletteStartDepth;
    }
    bool operator!=(const TransportOptions& other) const { return !(*this == other); }
};

// PSSMLT proposals use independent large steps and symmetric wrapped small
// steps. Chain count is an algorithm parameter, independent of CPU threads.
struct MetropolisOptions
{
    uint32_t BootstrapSamples = 4096;
    uint32_t ChainCount = 16;
    uint32_t Seed = 1;
    float LargeStepProbability = 0.3f;
    float MutationScale = 0.01f;
    bool operator==(const MetropolisOptions& other) const
    {
        return BootstrapSamples == other.BootstrapSamples && ChainCount == other.ChainCount &&
               Seed == other.Seed && LargeStepProbability == other.LargeStepProbability &&
               MutationScale == other.MutationScale;
    }
    bool operator!=(const MetropolisOptions& other) const { return !(*this == other); }
};
bool ValidateMetropolisOptions(const MetropolisOptions&, std::string& error);

// Fixed-radius photon mapping. Radius is relative to the emitter domain's
// scene bounding sphere, so existing demos retain comparable density bandwidth.
// These controls do not describe PPM radius updates or VCM merge weights.
struct PhotonMappingOptions
{
    uint32_t PhotonPaths = 65536;
    uint32_t Seed = 1;
    float RadiusScale = 0.02f;
    bool Caustics = true;
    bool Indirect = true;
    bool operator==(const PhotonMappingOptions& other) const
    {
        return PhotonPaths == other.PhotonPaths && Seed == other.Seed && RadiusScale == other.RadiusScale &&
               Caustics == other.Caustics && Indirect == other.Indirect;
    }
    bool operator!=(const PhotonMappingOptions& other) const { return !(*this == other); }
};
bool ValidatePhotonMappingOptions(const PhotonMappingOptions&, std::string& error);

// Stochastic PPM: one new camera visible point per pixel and pass. The render's
// sample count is the pass count; light paths are globally seeded per pass.
struct ProgressivePhotonOptions
{
    uint32_t PathsPerPass = 8192;
    uint32_t Seed = 1;
    float InitialRadiusScale = 0.05f;
    float Alpha = 2.0f / 3.0f;
    bool Caustics = true;
    bool Indirect = true;
    bool operator==(const ProgressivePhotonOptions& other) const
    {
        return PathsPerPass == other.PathsPerPass && Seed == other.Seed &&
               InitialRadiusScale == other.InitialRadiusScale && Alpha == other.Alpha &&
               Caustics == other.Caustics && Indirect == other.Indirect;
    }
    bool operator!=(const ProgressivePhotonOptions& other) const { return !(*this == other); }
};
bool ValidateProgressivePhotonOptions(const ProgressivePhotonOptions&, std::string& error);

// Progressive VCM combines fresh connection/merging estimates each pass.
// Unlike PPM, it does not carry a per-pixel radius or accumulated photon flux.
struct VertexConnectionMergingOptions
{
    uint32_t LightPathsPerPass = 8192, Seed = 1;
    float InitialRadiusScale = 0.05f, Alpha = 2.0f / 3.0f;
    bool operator==(const VertexConnectionMergingOptions& other) const
    {
        return LightPathsPerPass == other.LightPathsPerPass && Seed == other.Seed &&
               InitialRadiusScale == other.InitialRadiusScale && Alpha == other.Alpha;
    }
    bool operator!=(const VertexConnectionMergingOptions& other) const { return !(*this == other); }
};
bool ValidateVertexConnectionMergingOptions(const VertexConnectionMergingOptions&, std::string& error);

// A camera-independent diffuse surface cache. Area is measured in squared world
// units; bake sample counts do not change when the camera is cropped or tiled.
struct PointGiOptions
{
    float MaximumArea = 1;
    uint32_t MaximumSurfels = 1048576, DirectSamples = 16, GatherSamples = 64, Seed = 1;
    bool operator==(const PointGiOptions& other) const
    {
        return MaximumArea == other.MaximumArea && MaximumSurfels == other.MaximumSurfels &&
               DirectSamples == other.DirectSamples && GatherSamples == other.GatherSamples && Seed == other.Seed;
    }
    bool operator!=(const PointGiOptions& other) const { return !(*this == other); }
};
bool ValidatePointGiOptions(const PointGiOptions&, std::string& error);

bool ValidateTransportOptions(Algorithm algorithm, const TransportOptions& options, std::string& error);
uint32_t PackTransportFlags(const TransportOptions& options);
const char* MisKey(MisHeuristic heuristic);
bool ParseMis(std::string_view key, MisHeuristic& heuristic);
enum class TransportSchedule : uint8_t { CameraSamples, BidirectionalPaths, MarkovChains, PhotonPasses, SurfelPasses };

struct IntegratorInfo
{
    Algorithm Id;
    const char* Key;
    const char* Label;
    TransportSchedule Schedule;
    const char* Dependencies;
};

const IntegratorInfo* FindIntegrator(Algorithm algorithm);
bool ParseIntegrator(std::string_view key, Algorithm& algorithm);
bool SupportsIntegrator(BackendType backend, Algorithm algorithm);
bool ValidateIntegrator(BackendType backend, Algorithm algorithm, std::string& error);

}
