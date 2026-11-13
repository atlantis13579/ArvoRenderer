#include "Renderer/RenderTechniques.h"
#include "Renderer/RayTracing.h"
#include <cmath>

// Each execution backend owns its transport, traversal, scattering and sampling
// algorithms. Shared headers define configuration and data contracts only.
// DXR also provides interactive preview; DX11 remains a raster preview backend.
// Preview approximations and offline transport have independent settings/history.
// Sample identity is stable across tiles and retries. Stateful integrators schedule
// complete photon passes, bidirectional populations or Markov chains.

namespace Arvo::RayTracing
{
namespace
{
constexpr AccelerationInfo Accelerators[] = {
    {AccelerationType::Bvh, "bvh", "Bounding volume hierarchy", true},
    {AccelerationType::KdTree, "kd-tree", "k-d tree", true},
    {AccelerationType::Bih, "bih", "Bounding interval hierarchy", true}
};
constexpr IntegratorInfo Integrators[] = {
    {Algorithm::RayTracing, "ray", "Whitted-style ray tracing", TransportSchedule::CameraSamples, "Direct lights and specular recursion"},
    {Algorithm::PathTracing, "path", "Path tracing", TransportSchedule::CameraSamples, "BSDF sampling and direct illumination"},
    {Algorithm::DistributedRayTracing, "distributed-ray", "Distributed ray tracing", TransportSchedule::CameraSamples, "Stochastic area lights and glossy reflection"},
    {Algorithm::BidirectionalPathTracing, "bdpt", "Bidirectional path tracing", TransportSchedule::BidirectionalPaths, "Light/camera paths and strategy MIS"},
    {Algorithm::MetropolisLightTransport, "mlt", "Metropolis light transport", TransportSchedule::MarkovChains, "Bootstrap, mutations and weighted splats"},
    {Algorithm::VertexConnectionMerging, "vcm", "Vertex connection and merging", TransportSchedule::BidirectionalPaths, "Connection/merge MIS and light vertices"},
    {Algorithm::PhotonMapping, "photon", "Photon mapping", TransportSchedule::PhotonPasses, "Emission pass and photon density estimation"},
    {Algorithm::ProgressivePhotonMapping, "ppm", "Progressive photon mapping", TransportSchedule::PhotonPasses, "Visible points and progressive flux/radius"},
    {Algorithm::PointBasedGlobalIllumination, "point-gi", "Point-based global illumination", TransportSchedule::SurfelPasses, "Surfel bake and solid-angle integration"}
};
static_assert(sizeof(Integrators) / sizeof(Integrators[0]) == static_cast<size_t>(Algorithm::Count));
}

const AccelerationInfo* FindAcceleration(AccelerationType type)
{
    for (const auto& info : Accelerators) if (info.Id == type) return &info;
    return nullptr;
}

bool ParseAcceleration(std::string_view key, AccelerationType& type)
{
    for (const auto& info : Accelerators)
        if (key == info.Key) { type = info.Id; return true; }
    return false;
}

bool ValidateAcceleration(AccelerationType type, std::string& error)
{
    const auto* info = FindAcceleration(type);
    if (!info) { error = "Unknown acceleration structure"; return false; }
    if (!info->Implemented) { error = std::string(info->Label) + " is not implemented"; return false; }
    return true;
}

const IntegratorInfo* FindIntegrator(Algorithm algorithm)
{
    for (const auto& info : Integrators) if (info.Id == algorithm) return &info;
    return nullptr;
}

const char* BvhBuildKey(BvhBuildPolicy policy)
{
    switch (policy)
    {
    case BvhBuildPolicy::Default: return "default";
    case BvhBuildPolicy::SpatialSplits: return "spatial";
    default: return nullptr;
    }
}

const char* GeometryModeKey(GeometryMode mode)
{
    switch (mode)
    {
    case GeometryMode::Flat: return "flat";
    case GeometryMode::Instanced: return "instanced";
    default: return nullptr;
    }
}

bool ParseGeometryMode(std::string_view key, GeometryMode& mode)
{
    for (auto candidate : {GeometryMode::Flat, GeometryMode::Instanced})
        if (key == GeometryModeKey(candidate)) { mode = candidate; return true; }
    return false;
}

bool ParseBvhBuildPolicy(std::string_view key, BvhBuildPolicy& policy)
{
    for (auto candidate : {BvhBuildPolicy::Default, BvhBuildPolicy::SpatialSplits})
        if (key == BvhBuildKey(candidate)) { policy = candidate; return true; }
    return false;
}

bool ValidateAcceleration(AccelerationOptions options, std::string& error)
{
    if (!ValidateAcceleration(options.Type, error)) return false;
    if (!BvhBuildKey(options.BvhPolicy)) { error = "Unknown BVH build policy"; return false; }
    if (options.Type != AccelerationType::Bvh && options.BvhPolicy != BvhBuildPolicy::Default)
    {
        error = "Spatial splits require BVH acceleration";
        return false;
    }
    return true;
}

bool ParseIntegrator(std::string_view key, Algorithm& algorithm)
{
    for (const auto& info : Integrators)
        if (key == info.Key) { algorithm = info.Id; return true; }
    return false;
}

bool SupportsIntegrator(BackendType backend, Algorithm algorithm)
{
    if (!FindIntegrator(algorithm)) return false;
    switch (backend)
    {
    case BackendType::CPU:
    case BackendType::CUDA:
    case BackendType::ComputeShader:
    case BackendType::DXR:
    case BackendType::Metal:
        return true;
    default: return false;
    }
}

bool ValidateIntegrator(BackendType backend, Algorithm algorithm, std::string& error)
{
    const auto* info = FindIntegrator(algorithm);
    if (!info) { error = "Unknown rendering integrator"; return false; }
    if (!SupportsIntegrator(backend, algorithm))
    {
        error = std::string(info->Label) + " is not implemented on the selected backend";
        return false;
    }
    return true;
}

bool ValidateAdaptiveOptions(Algorithm algorithm, const AdaptiveOptions& options,
    uint32_t maximumSamples, std::string& error)
{
    constexpr uint32_t MinimumPilotSamples = 2, MaximumPilotSamples = 256;
    constexpr uint32_t MinimumProductionSamples = 2, MaximumProductionSamples = 4096;
    constexpr float MaximumRelativeError = 1.0f, MaximumAbsoluteError = 1000000.0f;
    if (options.PilotSamples < MinimumPilotSamples || options.PilotSamples > MaximumPilotSamples)
        error = "Adaptive pilot samples must be in [2, 256]";
    else if (!std::isfinite(options.MinimumVarianceFraction) || options.MinimumVarianceFraction < 0 ||
             options.MinimumVarianceFraction > 1)
        error = "Denoising variance retention must be in [0, 1]";
    else if (options.DenoiserAware && !options.Enabled)
        error = "Denoiser-aware allocation requires adaptive sampling";
    else if (options.MinimumSamples < MinimumProductionSamples || options.MinimumSamples > MaximumProductionSamples)
        error = "Adaptive minimum samples must be in [2, 4096]";
    else if (!std::isfinite(options.RelativeError) || options.RelativeError < 0.0f
        || options.RelativeError > MaximumRelativeError || !std::isfinite(options.AbsoluteError)
        || options.AbsoluteError <= 0.0f || options.AbsoluteError > MaximumAbsoluteError)
        error = "Adaptive error targets require relative error in [0, 1] and absolute error in (0, 1000000]";
    else if (options.Enabled && (maximumSamples < options.MinimumSamples || maximumSamples > MaximumProductionSamples))
        error = "Adaptive maximum samples must be between the minimum and 4096";
    else if (options.Enabled && algorithm != Algorithm::RayTracing && algorithm != Algorithm::PathTracing
        && algorithm != Algorithm::DistributedRayTracing)
        error = "Adaptive sampling requires an implemented camera-sample integrator";
    else return true;
    return false;
}

bool ValidateTransportOptions(Algorithm algorithm, const TransportOptions& options, std::string& error)
{
    constexpr uint32_t MaximumRouletteDepth = 32;
    if (options.Mis != MisHeuristic::None && options.Mis != MisHeuristic::Balance && options.Mis != MisHeuristic::Power)
        error = "Unknown MIS heuristic";
    else if (options.Mis != MisHeuristic::None && !options.NextEventEstimation &&
             algorithm != Algorithm::VertexConnectionMerging)
        error = "MIS requires both light sampling (NEE) and BSDF sampling";
    else if (options.RouletteStartDepth == 0 || options.RouletteStartDepth > MaximumRouletteDepth)
        error = "Russian roulette start depth must be in [1, 32]";
    else if (algorithm == Algorithm::BidirectionalPathTracing &&
        (options.RussianRoulette || !options.NextEventEstimation))
        error = "BDPT currently requires NEE and fixed depth (roulette is pending)";
    else if (algorithm == Algorithm::VertexConnectionMerging && options.RussianRoulette)
        error = "VCM currently uses fixed-depth walks; roulette needs bidirectional survival probabilities";
    else if (algorithm == Algorithm::PointBasedGlobalIllumination &&
             (!options.NextEventEstimation || options.RussianRoulette || options.Mis != MisHeuristic::None))
        error = "Point GI uses source sampling and fixed diffuse orders, without MIS or roulette";
    else if ((algorithm == Algorithm::PhotonMapping || algorithm == Algorithm::ProgressivePhotonMapping) && options.Mis != MisHeuristic::None)
        error = "Photon mapping partitions direct and merged paths without connection MIS";
    else if (algorithm != Algorithm::PathTracing && algorithm != Algorithm::BidirectionalPathTracing &&
             algorithm != Algorithm::MetropolisLightTransport && algorithm != Algorithm::PhotonMapping &&
             algorithm != Algorithm::ProgressivePhotonMapping &&
             algorithm != Algorithm::PointBasedGlobalIllumination &&
             algorithm != Algorithm::VertexConnectionMerging && options != TransportOptions{})
        error = "The selected integrator does not support path transport options";
    else return true;
    return false;
}

bool ValidateMetropolisOptions(const MetropolisOptions& options, std::string& error)
{
    constexpr uint32_t MinimumBootstrap = 64, MaximumBootstrap = 65536, MaximumChains = 128;
    constexpr float MinimumMutationScale = 0.0001f, MaximumMutationScale = 0.5f;
    if (options.BootstrapSamples < MinimumBootstrap || options.BootstrapSamples > MaximumBootstrap)
        error = "MLT bootstrap samples must be in [64, 65536]";
    else if (!options.ChainCount || options.ChainCount > MaximumChains)
        error = "MLT chain count must be in [1, 128]";
    else if (!std::isfinite(options.LargeStepProbability) || options.LargeStepProbability <= 0 ||
             options.LargeStepProbability > 1)
        error = "MLT large-step probability must be in (0, 1]";
    else if (!std::isfinite(options.MutationScale) || options.MutationScale < MinimumMutationScale ||
             options.MutationScale > MaximumMutationScale)
        error = "MLT mutation scale must be in [0.0001, 0.5]";
    else return true;
    return false;
}

bool ValidatePhotonMappingOptions(const PhotonMappingOptions& options, std::string& error)
{
    constexpr uint32_t MinimumPaths = 64, MaximumPaths = 1048576;
    constexpr float MinimumRadiusScale = 0.0001f, MaximumRadiusScale = 0.5f;
    if (options.PhotonPaths < MinimumPaths || options.PhotonPaths > MaximumPaths)
        error = "Photon path count must be in [64, 1048576]";
    else if (!std::isfinite(options.RadiusScale) || options.RadiusScale < MinimumRadiusScale ||
             options.RadiusScale > MaximumRadiusScale)
        error = "Photon radius scale must be in [0.0001, 0.5]";
    else return true;
    return false;
}

bool ValidateProgressivePhotonOptions(const ProgressivePhotonOptions& options, std::string& error)
{
    constexpr uint32_t MinimumPaths = 64, MaximumPaths = 1048576;
    constexpr float MinimumRadius = 0.0001f, MaximumRadius = 0.5f;
    if (options.PathsPerPass < MinimumPaths || options.PathsPerPass > MaximumPaths)
        error = "PPM paths per pass must be in [64, 1048576]";
    else if (!std::isfinite(options.InitialRadiusScale) || options.InitialRadiusScale < MinimumRadius ||
             options.InitialRadiusScale > MaximumRadius)
        error = "PPM initial radius scale must be in [0.0001, 0.5]";
    else if (!std::isfinite(options.Alpha) || options.Alpha <= 0 || options.Alpha >= 1)
        error = "PPM alpha must be finite and strictly between zero and one";
    else return true;
    return false;
}

bool ValidateVertexConnectionMergingOptions(const VertexConnectionMergingOptions& options, std::string& error)
{
    constexpr uint32_t MinimumPaths = 64, MaximumPaths = 1048576;
    constexpr float MinimumRadius = 0.0001f, MaximumRadius = 0.5f;
    if (options.LightPathsPerPass < MinimumPaths || options.LightPathsPerPass > MaximumPaths)
        error = "VCM light paths per pass must be in [64, 1048576]";
    else if (!std::isfinite(options.InitialRadiusScale) || options.InitialRadiusScale < MinimumRadius ||
             options.InitialRadiusScale > MaximumRadius)
        error = "VCM initial radius scale must be in [0.0001, 0.5]";
    else if (!std::isfinite(options.Alpha) || options.Alpha <= 0 || options.Alpha >= 1)
        error = "VCM alpha must be finite and strictly between zero and one";
    else
        return true;
    return false;
}

bool ValidatePointGiOptions(const PointGiOptions& options, std::string& error)
{
    constexpr uint32_t MaximumSurfelBudget = 1048576, MaximumBakeSamples = 65536;
    constexpr float MaximumPatchArea = 1000000;
    if (!std::isfinite(options.MaximumArea) || options.MaximumArea <= 0 || options.MaximumArea > MaximumPatchArea)
        error = "Point GI maximum patch area must be finite and in (0, 1000000] squared world units";
    else if (!options.MaximumSurfels || options.MaximumSurfels > MaximumSurfelBudget)
        error = "Point GI surface budget must be in [1, 1048576]";
    else if (!options.DirectSamples || options.DirectSamples > MaximumBakeSamples ||
             !options.GatherSamples || options.GatherSamples > MaximumBakeSamples)
        error = "Point GI direct and gather bake samples must be in [1, 65536]";
    else return true;
    return false;
}

uint32_t PackTransportFlags(const TransportOptions& options)
{
    // Native shader ABI: three booleans followed by a two-bit MIS selector.
    constexpr uint32_t ImportanceBit = 1u << 0;
    constexpr uint32_t NeeBit = 1u << 1;
    constexpr uint32_t RouletteBit = 1u << 2;
    constexpr uint32_t MisShift = 3;
    return (options.ImportanceSampling ? ImportanceBit : 0u)
        | (options.NextEventEstimation ? NeeBit : 0u)
        | (options.RussianRoulette ? RouletteBit : 0u)
        | (static_cast<uint32_t>(options.Mis) << MisShift);
}

const char* MisKey(MisHeuristic heuristic)
{
    switch (heuristic)
    {
    case MisHeuristic::None: return "none";
    case MisHeuristic::Balance: return "balance";
    case MisHeuristic::Power: return "power";
    default: return "invalid";
    }
}

bool ParseMis(std::string_view key, MisHeuristic& heuristic)
{
    for (const auto candidate : {MisHeuristic::None, MisHeuristic::Balance, MisHeuristic::Power})
        if (key == MisKey(candidate)) { heuristic = candidate; return true; }
    return false;
}

const SamplerInfo* FindSampler(SampleSequence sequence)
{
    static constexpr SamplerInfo Samplers[] = {
        {SampleSequence::Independent, "independent", "Independent random"},
        {SampleSequence::Stratified, "stratified", "Stratified"},
        {SampleSequence::Sobol, "sobol", "QMC (padded Sobol)"},
        {SampleSequence::BlueNoise, "blue-noise", "Blue noise (screen space)"}
    };
    for (const auto& sampler : Samplers) if (sampler.Id == sequence) return &sampler;
    return nullptr;
}

bool ParseSampler(std::string_view key, SampleSequence& sequence)
{
    for (uint32_t index = 0; index < static_cast<uint32_t>(SampleSequence::Count); ++index)
    {
        const auto* info = FindSampler(static_cast<SampleSequence>(index));
        if (key == info->Key) { sequence = info->Id; return true; }
    }
    return false;
}
}
