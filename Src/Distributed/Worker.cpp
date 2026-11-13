#include "Distributed/Distributed.h"
#include "Distributed/Http.h"
#include "Distributed/Protocol.h"
#include "Distributed/FilmTransfer.h"
#include "Distributed/TemporalTransfer.h"
#include "Distributed/MicropolygonTransfer.h"
#include "Renderer/MicropolygonPreparation.h"
#include "Renderer/Reconstruction.h"
#include "Editor/Demos.hpp"
#include "Editor/ModuleAdapters.h"
#include "RayTracer/RayTracer.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <thread>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#include <sys/resource.h>
#endif

namespace Arvo::Distributed
{
namespace
{
using Clock = std::chrono::steady_clock;

RayTracing::BackendType Backend(const std::string& name)
{
    if (name == "cpu") return RayTracing::BackendType::CPU;
    if (name == "cuda") return RayTracing::BackendType::CUDA;
    if (name == "compute") return RayTracing::BackendType::ComputeShader;
    if (name == "metal") return RayTracing::BackendType::Metal;
    if (name == "dxr") return RayTracing::BackendType::DXR;
    throw std::runtime_error("Unknown offline backend: " + name);
}

double CpuSeconds()
{
#ifdef _WIN32
    FILETIME creation{}, exit{}, kernel{}, user{};
    if (!GetProcessTimes(GetCurrentProcess(), &creation, &exit, &kernel, &user)) return -1.0;
    ULARGE_INTEGER k{}, u{};
    k.LowPart = kernel.dwLowDateTime; k.HighPart = kernel.dwHighDateTime;
    u.LowPart = user.dwLowDateTime; u.HighPart = user.dwHighDateTime;
    return double(k.QuadPart + u.QuadPart) / 10000000.0;
#else
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage)) return -1.0;
    return usage.ru_utime.tv_sec + usage.ru_stime.tv_sec +
        (usage.ru_utime.tv_usec + usage.ru_stime.tv_usec) / 1000000.0;
#endif
}

class GpuMeter
{
public:
    GpuMeter()
    {
#ifdef _WIN32
        Library_ = LoadLibraryW(L"nvml.dll");
#else
        Library_ = dlopen("libnvidia-ml.so.1", RTLD_LAZY | RTLD_LOCAL);
#endif
        if (!Library_) return;
        const auto init = reinterpret_cast<int (*)()>(Symbol("nvmlInit_v2"));
        const auto get = reinterpret_cast<int (*)(unsigned int, void**)>(Symbol("nvmlDeviceGetHandleByIndex_v2"));
        Utilization_ = reinterpret_cast<int (*)(void*, Rates*)>(Symbol("nvmlDeviceGetUtilizationRates"));
        Shutdown_ = reinterpret_cast<int (*)()>(Symbol("nvmlShutdown"));
        if (!init || !get || !Utilization_ || init() != 0) return;
        Initialized_ = true;
        if (get(0, &Device_) != 0) Device_ = nullptr;
    }

    ~GpuMeter()
    {
        if (Initialized_ && Shutdown_) Shutdown_();
#ifdef _WIN32
        if (Library_) FreeLibrary(Library_);
#else
        if (Library_) dlclose(Library_);
#endif
    }

    double Read() const
    {
        Rates rates{};
        if (!Device_ || !Utilization_ || Utilization_(Device_, &rates)) return -1.0;
        return rates.Gpu;
    }

private:
    struct Rates { unsigned int Gpu, Memory; };
    void* Symbol(const char* name)
    {
#ifdef _WIN32
        return reinterpret_cast<void*>(GetProcAddress(Library_, name));
#else
        return dlsym(Library_, name);
#endif
    }
#ifdef _WIN32
    HMODULE Library_ = nullptr;
#else
    void* Library_ = nullptr;
#endif
    void* Device_ = nullptr;
    int (*Utilization_)(void*, Rates*) = nullptr;
    int (*Shutdown_)() = nullptr;
    bool Initialized_ = false;
};

struct WorkerState
{
    explicit WorkerState(WorkerConfig config) : Config(std::move(config)), Catalog(Config.ContentRoot) {}
    WorkerConfig Config;
    Editor::DemoCatalog Catalog;
    std::mutex StatusMutex;
    std::mutex RenderMutex;
    std::string CurrentJob;
    std::string ActiveBackend;
    std::string Status = "Ready";
    RayTracing::CancellationSource Cancellation;
    std::set<std::string> CancelledJobs;
    std::map<std::string, std::shared_ptr<RayTracing::IRayTracingRHI>> Backends;
    using SceneKey = std::pair<std::string, RayTracing::GeometryMode>;
    std::map<SceneKey, std::shared_ptr<const RayTracing::Scene>> Scenes;
    std::map<SceneKey, std::uint64_t> SceneHashes;
    struct EditedVariant
    {
        std::string Key;
        std::shared_ptr<const RayTracing::Scene> Scene;
        uint64_t Hash = 0;
    };
    // Keep only the latest edited variant per demo/geometry, not one scene per slider position.
    std::map<SceneKey, EditedVariant> EditedScenes;
    struct DicedVariant
    {
        std::shared_ptr<const RayTracing::Scene> Source;
        std::string Backend, Key;
        RayTracing::MicropolygonResult Prepared;
    } DicedScene;
    std::uint32_t CompletedTiles = 0;
    double CpuPercent = -1.0, GpuPercent = -1.0;
    std::atomic<unsigned int> Connections{0};
};

std::shared_ptr<RayTracing::IRayTracingRHI> GetBackend(WorkerState& state, const std::string& name)
{
    if (const auto found = state.Backends.find(name); found != state.Backends.end()) return found->second;
    auto rhi = RayTracing::CreateRayTracingRHI(Backend(name));
    if (!rhi || !rhi->GetBackendInfo().IsAvailable)
        throw std::runtime_error("The requested native backend is unavailable: " + name);
    state.Backends.emplace(name, rhi);
    return rhi;
}

void Render(WorkerState& state, Http::Socket socket, const std::map<std::string, std::string>& fields,
            bool planOnly = false)
{
    if (fields.at("version") != Protocol::Version) throw std::runtime_error("Unsupported worker protocol version");
    const auto& sceneName = fields.at("scene");
    Editor::DemoId demo;
    if (!Editor::ParseDemo(sceneName, demo)) throw std::runtime_error("Unknown scene key");
    RayTracing::GeometryMode geometry;
    if (!RayTracing::ParseGeometryMode(fields.at("geometry"), geometry))
        throw std::runtime_error("Unknown geometry mode");
    const WorkerState::SceneKey sceneKey{sceneName, geometry};
    auto backend = fields.at("backend");
    if (backend == "auto") backend = state.Config.Backend;
    Backend(backend);
    RenderOutput output;
    if (!ParseRenderOutput(fields.at("output"), output)) throw std::runtime_error("Unknown worker output mode");
    const auto& jobId = fields.at("job");
    if (jobId.empty() || jobId.size() > 128) throw std::runtime_error("Invalid job identifier");
    RayTracing::RenderSettings settings;
    if (const auto found = fields.find("micropolygons"); found != fields.end())
        settings.Micropolygons = ParseMicropolygons(found->second);
    if (const auto found = fields.find("dicing_plan"); found != fields.end())
        settings.DicingPlan = ParseDicingPlan(found->second);
    if (planOnly && !settings.Micropolygons.Enabled)
        throw std::runtime_error("The planning endpoint requires enabled micropolygons");
    if (!planOnly && settings.Micropolygons.Enabled && !settings.DicingPlan)
        throw std::runtime_error("Tiled micropolygons require the coordinator's shared plan");
    settings.Width = Protocol::UInt(fields, "width", 1, 1024);
    settings.Height = Protocol::UInt(fields, "height", 1, 1024);
    settings.FullWidth = Protocol::UInt(fields, "full_width", 1, 16384);
    settings.FullHeight = Protocol::UInt(fields, "full_height", 1, 16384);
    settings.TileX = Protocol::UInt(fields, "x", 0, 16383);
    settings.TileY = Protocol::UInt(fields, "y", 0, 16383);
    if (settings.TileX + settings.Width > settings.FullWidth || settings.TileY + settings.Height > settings.FullHeight)
        throw std::runtime_error("Tile lies outside the image");
    settings.SamplesPerPixel = Protocol::UInt(fields, "samples", 1, 4096);
    settings.MaxReflectionBounces = Protocol::UInt(fields, "bounces", 0, 32);
    settings.ShadowsEnabled = Protocol::UInt(fields, "shadows", 0, 1) != 0;
    const auto& algorithm = fields.at("algorithm");
    if (!RayTracing::ParseIntegrator(algorithm, settings.Integrator))
        throw std::runtime_error("Unknown rendering algorithm");
    if (!RayTracing::ParseRadiometry(fields.at("radiometry"), settings.Radiometry))
        throw std::runtime_error("Unknown radiometry mode");
    if (!RayTracing::ParseSampler(fields.at("sampler"), settings.Sampling))
        throw std::runtime_error("Unknown sampling sequence");
    settings.SamplingSeed = Protocol::UInt(fields, "sampling_seed", 0, std::numeric_limits<uint32_t>::max());
    settings.Adaptive.Enabled = Protocol::UInt(fields, "adaptive", 0, 1) != 0;
    settings.Medium.Enabled = Protocol::UInt(fields, "medium", 0, 1) != 0;
    settings.Medium.Minimum = Protocol::Vector(fields.at("medium_min"));
    settings.Medium.Maximum = Protocol::Vector(fields.at("medium_max"));
    settings.Medium.Absorption = Protocol::Vector(fields.at("medium_absorption"));
    settings.Medium.Scattering = Protocol::Vector(fields.at("medium_scattering"));
    settings.Medium.Anisotropy = Protocol::Float(fields, "medium_g");
    settings.Adaptive.DenoiserAware = Protocol::UInt(fields, "adaptive_denoise", 0, 1) != 0;
    settings.Adaptive.MinimumVarianceFraction = Protocol::Float(fields, "adaptive_variance_floor");
    if (!RayTracing::ParseReconstructionFilter(fields.at("denoiser"), settings.Reconstruction.Filter))
        throw std::runtime_error("Unknown reconstruction filter");
    settings.Reconstruction.Passes = Protocol::UInt(fields, "denoise_passes", 1, 5);
    settings.Reconstruction.NoiseScale = Protocol::Float(fields, "denoise_noise");
    settings.Reconstruction.RelativeColorScale = Protocol::Float(fields, "denoise_relative");
    settings.Reconstruction.AbsoluteColorScale = Protocol::Float(fields, "denoise_absolute");
    settings.Reconstruction.GeometryGuidance = Protocol::UInt(fields, "denoise_geometry", 0, 1) != 0;
    settings.Reconstruction.NormalScale = Protocol::Float(fields, "denoise_normal");
    settings.Reconstruction.RelativePlaneScale = Protocol::Float(fields, "denoise_plane");
    settings.Metropolis.BootstrapSamples = Protocol::UInt(fields, "mlt_bootstrap", 64, 65536);
    settings.Metropolis.ChainCount = Protocol::UInt(fields, "mlt_chains", 1, 128);
    settings.Metropolis.Seed = Protocol::UInt(fields, "mlt_seed", 0, std::numeric_limits<uint32_t>::max());
    settings.Metropolis.LargeStepProbability = Protocol::Float(fields, "mlt_large_step");
    settings.Metropolis.MutationScale = Protocol::Float(fields, "mlt_scale");
    settings.Photons.PhotonPaths = Protocol::UInt(fields, "photon_paths", 64, 1048576);
    settings.Photons.Seed = Protocol::UInt(fields, "photon_seed", 0, std::numeric_limits<uint32_t>::max());
    settings.Photons.RadiusScale = Protocol::Float(fields, "photon_radius");
    settings.Photons.Caustics = Protocol::UInt(fields, "photon_caustics", 0, 1) != 0;
    settings.Photons.Indirect = Protocol::UInt(fields, "photon_indirect", 0, 1) != 0;
    settings.ProgressivePhotons.PathsPerPass = Protocol::UInt(fields, "ppm_paths", 64, 1048576);
    settings.ProgressivePhotons.Seed = Protocol::UInt(fields, "ppm_seed", 0, std::numeric_limits<uint32_t>::max());
    settings.ProgressivePhotons.InitialRadiusScale = Protocol::Float(fields, "ppm_radius");
    settings.ProgressivePhotons.Alpha = Protocol::Float(fields, "ppm_alpha");
    settings.ProgressivePhotons.Caustics = Protocol::UInt(fields, "ppm_caustics", 0, 1) != 0;
    settings.ProgressivePhotons.Indirect = Protocol::UInt(fields, "ppm_indirect", 0, 1) != 0;
    settings.Vcm.LightPathsPerPass = Protocol::UInt(fields, "vcm_paths", 64, 1048576);
    settings.Vcm.Seed = Protocol::UInt(fields, "vcm_seed", 0, std::numeric_limits<uint32_t>::max());
    settings.Vcm.InitialRadiusScale = Protocol::Float(fields, "vcm_radius");
    settings.Vcm.Alpha = Protocol::Float(fields, "vcm_alpha");
    settings.PointGi.MaximumArea = Protocol::Float(fields, "point_area");
    settings.PointGi.MaximumSurfels = Protocol::UInt(fields, "point_surfels", 1, 1048576);
    settings.PointGi.DirectSamples = Protocol::UInt(fields, "point_direct", 1, 65536);
    settings.PointGi.GatherSamples = Protocol::UInt(fields, "point_gather", 1, 65536);
    settings.PointGi.Seed = Protocol::UInt(fields, "point_seed", 0, std::numeric_limits<uint32_t>::max());
    if (!RayTracing::ParseAcceleration(fields.at("accel"), settings.Acceleration))
        throw std::runtime_error("Unknown acceleration structure");
    if (!RayTracing::ParseBvhBuildPolicy(fields.at("bvh_build"), settings.BvhPolicy))
        throw std::runtime_error("Unknown BVH build policy");
    std::string accelerationError;
    if (!RayTracing::ValidateAcceleration({settings.Acceleration, settings.BvhPolicy}, accelerationError))
        throw std::runtime_error(accelerationError);
    settings.Adaptive.PilotSamples = Protocol::UInt(fields, "pilot_samples", 2, 256);
    settings.Adaptive.MinimumSamples = Protocol::UInt(fields, "min_samples", 2, 4096);
    settings.Adaptive.RelativeError = Protocol::Float(fields, "relative_error");
    settings.Adaptive.AbsoluteError = Protocol::Float(fields, "absolute_error");
    settings.Transport.ImportanceSampling = Protocol::UInt(fields, "importance", 0, 1) != 0;
    settings.Transport.NextEventEstimation = Protocol::UInt(fields, "nee", 0, 1) != 0;
    settings.Transport.RussianRoulette = Protocol::UInt(fields, "roulette", 0, 1) != 0;
    settings.Transport.RouletteStartDepth = Protocol::UInt(fields, "roulette_depth", 1, 32);
    if (!RayTracing::ParseMis(fields.at("mis"), settings.Transport.Mis))
        throw std::runtime_error("Unknown MIS heuristic");
    std::string integratorError;
    if (!RayTracing::ValidateMediumOptions(settings.Integrator, settings.Medium, integratorError))
        throw std::runtime_error(integratorError);
    if (!RayTracing::ValidateMetropolisOptions(settings.Metropolis, integratorError)) throw std::runtime_error(integratorError);
    if (!RayTracing::ValidatePhotonMappingOptions(settings.Photons, integratorError)) throw std::runtime_error(integratorError);
    if (!RayTracing::ValidateProgressivePhotonOptions(settings.ProgressivePhotons, integratorError)) throw std::runtime_error(integratorError);
    if (!RayTracing::ValidateVertexConnectionMergingOptions(settings.Vcm, integratorError))
        throw std::runtime_error(integratorError);
    if (!RayTracing::ValidatePointGiOptions(settings.PointGi, integratorError)) throw std::runtime_error(integratorError);
    if (!RayTracing::ValidateReconstructionOptions(settings.Reconstruction, integratorError))
        throw std::runtime_error(integratorError);
    if (output != RenderOutput::Png && settings.Reconstruction.Filter != RayTracing::ReconstructionFilter::None &&
        !settings.Adaptive.DenoiserAware)
        throw std::runtime_error("Raw film jobs reconstruct only after full-frame assembly");
    if (settings.Integrator == RayTracing::Algorithm::MetropolisLightTransport && settings.Sampling != RayTracing::SampleSequence::Independent)
        throw std::runtime_error("MLT requires independent primary samples");
    if (!RayTracing::ValidateTransportOptions(settings.Integrator, settings.Transport, integratorError))
        throw std::runtime_error(integratorError);
    if (!RayTracing::ValidateRadiometry(settings.Integrator, settings.Radiometry, integratorError))
        throw std::runtime_error(integratorError);
    if (!RayTracing::ValidateAdaptiveOptions(settings.Integrator, settings.Adaptive, settings.SamplesPerPixel, integratorError))
        throw std::runtime_error(integratorError);
    if (!RayTracing::ValidateIntegrator(Backend(backend), settings.Integrator, integratorError))
        throw std::runtime_error(integratorError);
    const auto materials = Protocol::ParseMaterials(fields.at("materials"));
    const auto motions = Protocol::ParseInstanceMotions(fields.at("instance_motion"));
    if (!motions.empty() && geometry != RayTracing::GeometryMode::Instanced)
        throw std::runtime_error("Instance motion requires instanced geometry");
    for (const auto& entry : materials)
        if (!RayTracing::ValidateMaterialSupport(entry.Scattering, Backend(backend), settings.Integrator, integratorError))
            throw std::runtime_error(integratorError);
    RayTracing::Camera camera;
    camera.Lens.Radius = Protocol::Float(fields, "lens_radius");
    camera.Lens.FocusDistance = Protocol::Float(fields, "focus_distance");
    camera.Motion = Protocol::ParseCameraExposure(fields.at("camera_exposure"));
    if (!RayTracing::ValidateCameraMotionSupport(settings.Integrator, camera.Motion, integratorError))
        throw std::runtime_error(integratorError);
    if (!RayTracing::ValidateCameraLens(camera.Lens, integratorError)) throw std::runtime_error(integratorError);
    camera.Position = {Protocol::Float(fields, "px"), Protocol::Float(fields, "py"), Protocol::Float(fields, "pz")};
    camera.Target = {Protocol::Float(fields, "tx"), Protocol::Float(fields, "ty"), Protocol::Float(fields, "tz")};
    camera.Up = {Protocol::Float(fields, "ux"), Protocol::Float(fields, "uy"), Protocol::Float(fields, "uz")};
    camera.VerticalFieldOfViewDegrees = Protocol::Float(fields, "fov");
    if (camera.VerticalFieldOfViewDegrees <= 1.0f || camera.VerticalFieldOfViewDegrees >= 179.0f ||
        RayTracing::Dot(camera.Target - camera.Position, camera.Target - camera.Position) < 1e-12f ||
        RayTracing::Dot(RayTracing::Cross(camera.Target - camera.Position, camera.Up),
                        RayTracing::Cross(camera.Target - camera.Position, camera.Up)) < 1e-12f)
        throw std::runtime_error("Invalid render camera");

    std::unique_lock<std::mutex> rendering(state.RenderMutex, std::try_to_lock);
    if (!rendering.owns_lock()) { Http::Reply(socket, 409, "Worker is busy"); return; }
    RayTracing::CancellationToken cancellation;
    {
        std::lock_guard<std::mutex> lock(state.StatusMutex);
        if (state.CancelledJobs.count(jobId)) { Http::Reply(socket, 409, "Job cancelled"); return; }
        state.CurrentJob = jobId;
        state.ActiveBackend = backend;
        state.Cancellation = RayTracing::CancellationSource{};
        cancellation = state.Cancellation.GetToken();
        state.Status = "Loading " + sceneName;
    }
    try
    {
        if (!state.Scenes.count(sceneKey))
        {
            const auto snapshot = state.Catalog.Load(demo);
            auto scene = Editor::BuildOfflineScene(snapshot, geometry);
            if (!scene) throw std::runtime_error("Cannot prepare worker scene");
            state.SceneHashes[sceneKey] = Protocol::SceneHash(*scene);
            state.Scenes[sceneKey] = std::move(scene);
        }
        auto scene = state.Scenes.at(sceneKey);
        auto sceneHash = state.SceneHashes.at(sceneKey);
        if (!materials.empty() || !motions.empty())
        {
            const auto key = Protocol::EncodeMaterials(materials) + '\n' + Protocol::EncodeInstanceMotions(motions);
            auto& variant = state.EditedScenes[sceneKey];
            if (!variant.Scene || variant.Key != key)
            {
                auto edited = std::make_shared<RayTracing::Scene>(*scene);
                Protocol::ApplyMaterials(*edited, materials);
                Protocol::ApplyInstanceMotions(*edited, motions);
                variant = {key, edited, Protocol::SceneHash(*edited)};
            }
            scene = variant.Scene;
            sceneHash = variant.Hash;
        }
        if (!RayTracing::ValidateSceneMotionSupport(settings.Integrator, *scene, integratorError))
            throw std::runtime_error(integratorError);
        if (!RayTracing::ValidateReconstructionSupport(camera, RayTracing::HasSceneMotion(*scene),
                                                       settings.Reconstruction, settings.Medium,
                                                       integratorError))
            throw std::runtime_error(integratorError);
        const auto rhi = GetBackend(state, backend);
        if (settings.Micropolygons.Enabled)
        {
            std::string key;
            for (const char* field : {"micropolygons", "full_width", "full_height", "px", "py", "pz", "tx",
                                      "ty", "tz", "ux", "uy", "uz", "fov"})
                key += fields.at(field) + '\n';
            auto& cache = state.DicedScene;
            const std::string planKey =
                settings.DicingPlan ? EncodeDicingPlan(*settings.DicingPlan) : std::string{};
            const bool reuse =
                !planOnly && cache.Source == scene && cache.Backend == backend && cache.Key == key + planKey;
            if (!reuse)
            {
                auto diced = rhi->PrepareMicropolygons(*scene, camera, settings, cancellation);
                if (diced.Status != RayTracing::RenderStatus::Completed)
                {
                    if (diced.Status == RayTracing::RenderStatus::Cancelled)
                    {
                        {
                            std::lock_guard<std::mutex> lock(state.StatusMutex);
                            state.CurrentJob.clear();
                            state.Status = "Dicing cancelled";
                        }
                        Http::Reply(socket, 409, "Job cancelled");
                        return;
                    }
                    throw std::runtime_error(diced.Error);
                }
                cache = {scene, backend, key + EncodeDicingPlan(*diced.Plan), std::move(diced)};
            }
            if (planOnly)
            {
                const auto reply = Http::EncodeFields(
                    {{"micropolygon_wire", MicropolygonWireVersion},
                     {"scene_hash", std::to_string(sceneHash)},
                     {"dicing_plan", EncodeDicingPlan(*cache.Prepared.Plan)},
                     {"input_triangles", std::to_string(cache.Prepared.InputTriangles)},
                     {"output_triangles", std::to_string(cache.Prepared.OutputTriangles)},
                     {"clamped_components", std::to_string(cache.Prepared.ClampedComponents)}});
                const Http::Bytes bytes(reply.begin(), reply.end());
                Http::Reply(socket, 200, bytes, "text/plain",
                            [&] { return cancellation.IsCancellationRequested(); });
                std::lock_guard<std::mutex> lock(state.StatusMutex);
                state.CurrentJob.clear();
                state.Status = "Ready";
                return;
            }
            scene = cache.Prepared.Geometry;
            settings.Micropolygons.Enabled = false;
            settings.DicingPlan.reset();
        }
        {
            std::lock_guard<std::mutex> lock(state.StatusMutex);
            state.Status = "Rendering " + std::to_string(settings.TileX) + "," + std::to_string(settings.TileY);
        }
        const auto start = Clock::now();
        auto job = rhi->RenderAsync(scene, camera, settings);
        while (!job.IsReady())
        {
            if (cancellation.IsCancellationRequested()) job.Cancel();
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        auto result = job.GetResult();
        RayTracing::GeometryGuideResult guides;
        if (result.Status == RayTracing::RenderStatus::Completed && output == RenderOutput::GuidedFilm)
        {
            guides = rhi->CaptureGeometryGuides(*scene, camera, settings, cancellation);
            if (guides.Status != RayTracing::RenderStatus::Completed)
            {
                result.Status = guides.Status;
                result.Error = std::move(guides.Error);
            }
        }
        const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - start).count();
        const auto finish = [&](bool completed) {
            std::lock_guard<std::mutex> lock(state.StatusMutex);
            state.CurrentJob.clear();
            state.Status = completed ? "Ready" : "Render stopped";
            if (completed) ++state.CompletedTiles;
        };
        if (result.Status == RayTracing::RenderStatus::Cancelled || cancellation.IsCancellationRequested())
        { finish(false); Http::Reply(socket, 409, "Job cancelled"); return; }
        if (result.Status != RayTracing::RenderStatus::Completed) throw std::runtime_error(result.Error);
        if (result.Output.GetWidth() != settings.Width || result.Output.GetHeight() != settings.Height)
            throw std::runtime_error("Backend returned invalid tile dimensions");
        Http::Bytes bytes;
        if (output != RenderOutput::Png)
        {
            FilmTile tile;
            tile.Film = std::move(result.Film);
            tile.Geometry = std::move(guides.Image);
            tile.FullWidth = settings.FullWidth; tile.FullHeight = settings.FullHeight;
            tile.X = settings.TileX; tile.Y = settings.TileY;
            tile.Backend = Backend(backend);
            tile.Integrator = settings.Integrator; tile.Sampling = settings.Sampling;
            tile.Radiometry = settings.Radiometry;
            tile.SamplingSeed = settings.SamplingSeed;
            tile.RenderMilliseconds = static_cast<uint64_t>(milliseconds); tile.SceneHash = sceneHash;
            tile.RequestHash = FilmRequestHash(Http::EncodeFields(fields));
            bytes = EncodeFilmTile(tile);
        }
        else
        {
            bytes = {'A', 'R', 'V', '1'};
            Protocol::Append32(bytes, settings.Width); Protocol::Append32(bytes, settings.Height);
            Protocol::Append64(bytes, static_cast<std::uint64_t>(milliseconds));
            Protocol::Append64(bytes, sceneHash);
            const auto& pixels = result.Output.GetPixelsRGBA8();
            bytes.insert(bytes.end(), pixels.begin(), pixels.end());
        }
        if (cancellation.IsCancellationRequested()) { finish(false); Http::Reply(socket, 409, "Job cancelled"); return; }
        finish(true);
        Http::Reply(socket, 200, bytes, "application/octet-stream");
    }
    catch (...)
    {
        std::lock_guard<std::mutex> lock(state.StatusMutex);
        state.CurrentJob.clear();
        state.Status = "Render failed";
        throw;
    }
}

struct RequestFailure : std::runtime_error
{
    int Status;
    RequestFailure(int status, const char* message) : std::runtime_error(message), Status(status) {}
};
void Temporal(WorkerState& state, Http::Socket socket, Http::Message& request,
              const RayTracing::CancellationToken& cancellation)
{
    // The handler acquired RenderMutex before receiving the body. No persistent
    // history is kept here: the caller owns publication and supplies every input.
    auto packet = DecodeTemporalRequest(request.Body, cancellation);
    if (packet.Job != request.Headers.at("x-arvo-job")) throw std::runtime_error("Temporal job header mismatch");
    Http::Bytes{}.swap(request.Body);
    const char* name = nullptr;
    using B = RayTracing::BackendType;
    switch (packet.Backend)
    {
    case B::CPU: name = "cpu"; break;
    case B::CUDA: name = "cuda"; break;
    case B::ComputeShader: name = "compute"; break;
    case B::DXR: name = "dxr"; break;
    case B::Metal: name = "metal"; break;
    }
    if (!name) throw std::runtime_error("Unknown temporal backend");
    {
        std::lock_guard<std::mutex> lock(state.StatusMutex);
        state.ActiveBackend = name;
        state.Status = "Reconstructing complete temporal frame";
    }
    if (cancellation.IsCancellationRequested()) throw RequestFailure(409, "Temporal job cancelled");
    const auto rhi = GetBackend(state, name);
    auto result = rhi->ReconstructTemporal(packet.Frame(), packet.Previous ? &*packet.Previous : nullptr,
                                          packet.Options, cancellation);
    if (result.Status == RayTracing::RenderStatus::Cancelled || cancellation.IsCancellationRequested())
        throw RequestFailure(409, "Temporal job cancelled");
    if (result.Status != RayTracing::RenderStatus::Completed) throw std::runtime_error(result.Error);
    auto bytes = EncodeTemporalResponse(packet, result, cancellation);
    Http::Reply(socket, 200, bytes, "application/octet-stream",
                [&] { return cancellation.IsCancellationRequested(); });
}
void Handle(WorkerState& state, Http::Socket socket)
{
    std::unique_lock<std::mutex> nativeLock(state.RenderMutex, std::defer_lock);
    RayTracing::CancellationToken nativeCancellation;
    bool nativeAdmitted = false;
    bool nativeSucceeded = false;
    try
    {
        constexpr size_t MaximumRenderRequestBytes = 4 * 1024 * 1024;
        auto request = Http::ReadRequest(socket, MaximumTemporalRequestBytes,
            [&](const Http::Message& headers, size_t bytes, int& bodyTimeout) {
                if (!state.Config.Token.empty())
                {
                    const auto auth = headers.Headers.find("authorization");
                    if (auth == headers.Headers.end() || auth->second != "Bearer " + state.Config.Token)
                        throw RequestFailure(401, "Worker authorization required");
                }
                if (headers.FirstLine != "POST /temporal HTTP/1.1")
                {
                    if (bytes > MaximumRenderRequestBytes) throw RequestFailure(400, "Worker form exceeds limit");
                    return;
                }
                const auto id = headers.Headers.find("x-arvo-job");
                if (id == headers.Headers.end() || id->second.empty() || id->second.size() > 128)
                    throw RequestFailure(400, "Native upload requires a job identifier");
                if (!nativeLock.try_lock()) throw RequestFailure(409, "Worker is busy");
                constexpr int NativeUploadTimeoutSeconds = 600;
                bodyTimeout = NativeUploadTimeoutSeconds;
                std::lock_guard<std::mutex> status(state.StatusMutex);
                if (state.CancelledJobs.count(id->second)) throw RequestFailure(409, "Native job cancelled");
                state.Cancellation = RayTracing::CancellationSource{};
                nativeCancellation = state.Cancellation.GetToken();
                state.CurrentJob = id->second;
                state.Status = "Receiving complete temporal frame";
                nativeAdmitted = true;
            }, [&] { return nativeCancellation.IsCancellationRequested(); }, 30);
        if (request.FirstLine == "GET /status HTTP/1.1" || request.FirstLine == "GET /status HTTP/1.0")
        {
            std::map<std::string, std::string> fields;
            {
                std::lock_guard<std::mutex> lock(state.StatusMutex);
                fields = {
                    {"version", Protocol::Version},
                    {"temporal_wire", TemporalWireVersion},
                    {"micropolygon_wire", MicropolygonWireVersion},
                    {"status", state.Status},
                    {"backend", state.ActiveBackend.empty() ? state.Config.Backend : state.ActiveBackend},
                    {"job", state.CurrentJob},
                    {"cpu", Protocol::Number(state.CpuPercent)},
                    {"gpu", Protocol::Number(state.GpuPercent)},
                    {"gpu_scope", "device0"},
                    {"cpu_scope", "process_normalized"},
                    {"completed", std::to_string(state.CompletedTiles)}};
            }
            Http::Reply(socket, 200, Http::EncodeFields(fields));
        }
        else if (request.FirstLine == "POST /render HTTP/1.1") Render(state, socket, Http::ParseFields(Http::Text(request.Body)));
        else if (request.FirstLine == "POST /micropolygon-plan HTTP/1.1")
            Render(state, socket, Http::ParseFields(Http::Text(request.Body)), true);
        else if (request.FirstLine == "POST /temporal HTTP/1.1")
        {
            Temporal(state, socket, request, nativeCancellation);
            nativeSucceeded = true;
        }
        else if (request.FirstLine == "POST /cancel HTTP/1.1")
        {
            const auto fields = Http::ParseFields(Http::Text(request.Body));
            const auto& job = fields.at("job");
            if (job.empty() || job.size() > 128) throw std::runtime_error("Invalid cancellation identifier");
            {
                std::lock_guard<std::mutex> lock(state.StatusMutex);
                if (state.CancelledJobs.size() > 2048) state.CancelledJobs.erase(state.CancelledJobs.begin());
                state.CancelledJobs.insert(job);
                if (state.CurrentJob == job) state.Cancellation.Cancel();
            }
            Http::Reply(socket, 200, "Cancelled\n");
        }
        else Http::Reply(socket, 404, "Unknown worker endpoint");
    }
    catch (const RequestFailure& error)
    {
        try { Http::Reply(socket, error.Status, error.what()); } catch (...) {}
    }
    catch (const std::exception& error)
    {
        try { Http::Reply(socket, nativeCancellation.IsCancellationRequested() ? 409 : 400, error.what()); }
        catch (...) {}
    }
    if (nativeAdmitted)
    {
        std::lock_guard<std::mutex> lock(state.StatusMutex);
        state.CurrentJob.clear();
        state.Status = nativeSucceeded ? "Ready" : nativeCancellation.IsCancellationRequested()
            ? "Native job cancelled" : "Native job failed";
    }
    Http::Close(socket);
    --state.Connections;
}
} // namespace

int RunWorker(const WorkerConfig& config)
{
    const auto stateOwner = std::make_shared<WorkerState>(config);
    auto& state = *stateOwner;
    GetBackend(state, config.Backend);
    const auto listener = Http::Listen(config.BindAddress, config.Port);
    std::cout << "Arvo worker ready at http://" << config.BindAddress << ':' << config.Port
              << " backend=" << config.Backend << " assets=" << config.ContentRoot.string() << std::endl;
    std::thread telemetry([stateOwner] {
        auto& state = *stateOwner;
        GpuMeter gpu;
        auto previousTime = Clock::now();
        double previousCpu = CpuSeconds();
        for (;;)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            const auto now = Clock::now();
            const double cpu = CpuSeconds();
            const double elapsed = std::chrono::duration<double>(now - previousTime).count();
            const double utilization = cpu < 0 || previousCpu < 0 ? -1.0 :
                std::clamp((cpu - previousCpu) / elapsed / std::max(1U, std::thread::hardware_concurrency()) * 100.0, 0.0, 100.0);
            {
                std::lock_guard<std::mutex> lock(state.StatusMutex);
                state.CpuPercent = utilization;
                state.GpuPercent = gpu.Read();
            }
            previousCpu = cpu;
            previousTime = now;
        }
    });
    telemetry.detach();
    for (;;)
    {
        const auto connection = Http::Accept(listener);
        if (connection == Http::InvalidSocket) continue;
        if (state.Connections.fetch_add(1) >= 16)
        {
            --state.Connections;
            Http::Close(connection);
            continue;
        }
        try
        {
            std::thread([stateOwner, connection] { Handle(*stateOwner, connection); }).detach();
        }
        catch (...)
        {
            --state.Connections;
            Http::Close(connection);
        }
    }
}
} // namespace Arvo::Distributed
