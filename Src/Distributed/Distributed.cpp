#include "Distributed/Distributed.h"
#include "Distributed/Http.h"
#include "Distributed/Protocol.h"
#include "Distributed/FilmTransfer.h"
#include "Distributed/MicropolygonTransfer.h"
#include "Renderer/Reconstruction.h"
#include "Editor/Demos.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <random>
#include <set>
#include <stdexcept>
#include <thread>
#include <charconv>

namespace Arvo::Distributed
{
namespace
{
using Clock = std::chrono::steady_clock;
struct Tile
{
    std::uint32_t X = 0, Y = 0, Width = 0, Height = 0, Attempts = 0;
};

std::string JobIdentifier()
{
    std::random_device random;
    return std::to_string(Clock::now().time_since_epoch().count()) + "-" +
        std::to_string(random()) + "-" + std::to_string(random());
}

double Utilization(const std::string& text)
{
    std::size_t used = 0;
    const double value = std::stod(text, &used);
    if (used != text.size() || !std::isfinite(value) || value < 0.0 || value > 100.0) return -1.0;
    return value;
}

std::string RequestBody(const RenderConfig& config, const std::string& id, const Tile& tile)
{
    std::map<std::string, std::string> fields{
        {"version", Protocol::Version},
        {"job", id},
        {"scene", config.Scene},
        {"backend", config.Backend},
        {"output", RenderOutputKey(config.Output)},
        {"algorithm", RayTracing::FindIntegrator(config.Integrator)->Key},
        {"radiometry", RayTracing::RadiometryKey(config.Radiometry)},
        {"sampler", RayTracing::FindSampler(config.Sampling)->Key},
        {"sampling_seed", std::to_string(config.SamplingSeed)},
        {"adaptive", config.Adaptive.Enabled ? "1" : "0"},
        {"medium", config.Medium.Enabled ? "1" : "0"},
        {"medium_min", Protocol::VectorText(config.Medium.Minimum)},
        {"medium_max", Protocol::VectorText(config.Medium.Maximum)},
        {"medium_absorption", Protocol::VectorText(config.Medium.Absorption)},
        {"medium_scattering", Protocol::VectorText(config.Medium.Scattering)},
        {"medium_g", Protocol::Number(config.Medium.Anisotropy)},
        {"adaptive_denoise", config.Adaptive.DenoiserAware ? "1" : "0"},
        {"adaptive_variance_floor", Protocol::Number(config.Adaptive.MinimumVarianceFraction)},
        {"denoiser", RayTracing::ReconstructionFilterKey(config.Reconstruction.Filter)},
        {"denoise_passes", std::to_string(config.Reconstruction.Passes)},
        {"denoise_noise", Protocol::Number(config.Reconstruction.NoiseScale)},
        {"denoise_relative", Protocol::Number(config.Reconstruction.RelativeColorScale)},
        {"denoise_absolute", Protocol::Number(config.Reconstruction.AbsoluteColorScale)},
        {"denoise_geometry", config.Reconstruction.GeometryGuidance ? "1" : "0"},
        {"denoise_normal", Protocol::Number(config.Reconstruction.NormalScale)},
        {"denoise_plane", Protocol::Number(config.Reconstruction.RelativePlaneScale)},
        {"pilot_samples", std::to_string(config.Adaptive.PilotSamples)},
        {"min_samples", std::to_string(config.Adaptive.MinimumSamples)},
        {"mlt_bootstrap", std::to_string(config.Metropolis.BootstrapSamples)},
        {"mlt_chains", std::to_string(config.Metropolis.ChainCount)},
        {"mlt_seed", std::to_string(config.Metropolis.Seed)},
        {"mlt_large_step", Protocol::Number(config.Metropolis.LargeStepProbability)},
        {"mlt_scale", Protocol::Number(config.Metropolis.MutationScale)},
        {"photon_paths", std::to_string(config.Photons.PhotonPaths)},
        {"photon_seed", std::to_string(config.Photons.Seed)},
        {"photon_radius", Protocol::Number(config.Photons.RadiusScale)},
        {"photon_caustics", config.Photons.Caustics ? "1" : "0"},
        {"photon_indirect", config.Photons.Indirect ? "1" : "0"},
        {"ppm_paths", std::to_string(config.ProgressivePhotons.PathsPerPass)},
        {"ppm_seed", std::to_string(config.ProgressivePhotons.Seed)},
        {"ppm_radius", Protocol::Number(config.ProgressivePhotons.InitialRadiusScale)},
        {"ppm_alpha", Protocol::Number(config.ProgressivePhotons.Alpha)},
        {"ppm_caustics", config.ProgressivePhotons.Caustics ? "1" : "0"},
        {"ppm_indirect", config.ProgressivePhotons.Indirect ? "1" : "0"},
        {"vcm_paths", std::to_string(config.Vcm.LightPathsPerPass)},
        {"vcm_seed", std::to_string(config.Vcm.Seed)},
        {"vcm_radius", Protocol::Number(config.Vcm.InitialRadiusScale)},
        {"vcm_alpha", Protocol::Number(config.Vcm.Alpha)},
        {"point_area", Protocol::Number(config.PointGi.MaximumArea)},
        {"point_surfels", std::to_string(config.PointGi.MaximumSurfels)},
        {"point_direct", std::to_string(config.PointGi.DirectSamples)},
        {"point_gather", std::to_string(config.PointGi.GatherSamples)},
        {"point_seed", std::to_string(config.PointGi.Seed)},
        {"accel", RayTracing::FindAcceleration(config.Acceleration)->Key},
        {"bvh_build", RayTracing::BvhBuildKey(config.BvhPolicy)},
        {"geometry", RayTracing::GeometryModeKey(config.Geometry)},
        {"materials", Protocol::EncodeMaterials(config.Materials)},
        {"instance_motion", Protocol::EncodeInstanceMotions(config.InstanceMotions)},
        {"lens_radius", Protocol::Number(config.Camera.Lens.Radius)},
        {"focus_distance", Protocol::Number(config.Camera.Lens.FocusDistance)},
        {"camera_exposure", Protocol::EncodeCameraExposure(config.Camera.Motion)},
        {"relative_error", Protocol::Number(config.Adaptive.RelativeError)},
        {"absolute_error", Protocol::Number(config.Adaptive.AbsoluteError)},
        {"importance", config.Transport.ImportanceSampling ? "1" : "0"},
        {"nee", config.Transport.NextEventEstimation ? "1" : "0"},
        {"mis", RayTracing::MisKey(config.Transport.Mis)},
        {"roulette", config.Transport.RussianRoulette ? "1" : "0"},
        {"roulette_depth", std::to_string(config.Transport.RouletteStartDepth)},
        {"width", std::to_string(tile.Width)},
        {"height", std::to_string(tile.Height)},
        {"full_width", std::to_string(config.Width)},
        {"full_height", std::to_string(config.Height)},
        {"x", std::to_string(tile.X)},
        {"y", std::to_string(tile.Y)},
        {"samples", std::to_string(config.SamplesPerPixel)},
        {"bounces", std::to_string(config.MaxBounces)},
        {"shadows", config.ShadowsEnabled ? "1" : "0"},
        {"px", Protocol::Number(config.Camera.Position.X)},
        {"py", Protocol::Number(config.Camera.Position.Y)},
        {"pz", Protocol::Number(config.Camera.Position.Z)},
        {"tx", Protocol::Number(config.Camera.Target.X)},
        {"ty", Protocol::Number(config.Camera.Target.Y)},
        {"tz", Protocol::Number(config.Camera.Target.Z)},
        {"ux", Protocol::Number(config.Camera.Up.X)},
        {"uy", Protocol::Number(config.Camera.Up.Y)},
        {"uz", Protocol::Number(config.Camera.Up.Z)},
        {"fov", Protocol::Number(config.Camera.VerticalFieldOfViewDegrees)}};
    if (config.Micropolygons.Enabled)
    {
        fields["micropolygons"] = EncodeMicropolygons(config.Micropolygons);
        if (config.DicingPlan)
            fields["dicing_plan"] = EncodeDicingPlan(*config.DicingPlan);
    }
    return Http::EncodeFields(fields);
}

void Validate(const RenderConfig& config)
{
    if (config.Servers.empty() || config.Servers.size() > 64) throw std::runtime_error("Configure between 1 and 64 worker URLs");
    std::set<std::string> unique;
    for (const auto& url : config.Servers)
    {
        if (url.rfind("http://", 0) != 0) throw std::runtime_error("Worker URLs must start with http://");
        if (!unique.insert(url).second) throw std::runtime_error("Duplicate worker URL: " + url);
    }
    Editor::DemoId demo;
    if (!Editor::ParseDemo(config.Scene, demo)) throw std::runtime_error("Unknown scene key");
    if (config.Backend != "cpu" && config.Backend != "cuda" && config.Backend != "compute"
        && config.Backend != "metal" && config.Backend != "dxr" && config.Backend != "auto")
        throw std::runtime_error("Unknown worker backend");
    if (!config.Width || !config.Height || config.Width > 16384 || config.Height > 16384 ||
        std::uint64_t(config.Width) * config.Height > 67108864)
        throw std::runtime_error("Output dimensions must be positive and at most 64 megapixels / 16384 per side");
    if (!config.TileSize || config.TileSize > 1024) throw std::runtime_error("Tile size must be between 1 and 1024");
    if (!config.SamplesPerPixel || config.SamplesPerPixel > 4096 || config.MaxBounces > 32 || config.MaxRetries > 16)
        throw std::runtime_error("Render samples, bounces or retries exceed the supported limits");
    if (!RayTracing::FindIntegrator(config.Integrator))
        throw std::runtime_error("Unknown render integrator");
    if (!RayTracing::FindSampler(config.Sampling)) throw std::runtime_error("Unknown sampling sequence");
    if (!RayTracing::GeometryModeKey(config.Geometry)) throw std::runtime_error("Unknown geometry mode");
    (void)Protocol::EncodeMaterials(config.Materials);
    (void)Protocol::EncodeInstanceMotions(config.InstanceMotions);
    if (!config.InstanceMotions.empty() && config.Geometry != RayTracing::GeometryMode::Instanced)
        throw std::runtime_error("Instance motion requires instanced geometry");
    if (HasInstanceMotion(config) && !RayTracing::SupportsMotionBlur(config.Integrator))
        throw std::runtime_error("Instance motion requires Path or Distributed Ray Tracing");
    std::string accelerationError;
    if (!RayTracing::ValidateAcceleration({config.Acceleration, config.BvhPolicy}, accelerationError)) throw std::runtime_error(accelerationError);
    std::string transportError;
    if (!RayTracing::ValidateCameraLens(config.Camera.Lens, transportError) ||
        !RayTracing::ValidateCameraMotionSupport(config.Integrator, config.Camera.Motion, transportError))
        throw std::runtime_error(transportError);
    if (!RayTracing::ValidateTransportOptions(config.Integrator, config.Transport, transportError))
        throw std::runtime_error(transportError);
    if (!RayTracing::ValidateRadiometry(config.Integrator, config.Radiometry, transportError))
        throw std::runtime_error(transportError);
    if (!RayTracing::ValidateAdaptiveOptions(config.Integrator, config.Adaptive, config.SamplesPerPixel, transportError))
        throw std::runtime_error(transportError);
    if (!RayTracing::ValidateMetropolisOptions(config.Metropolis, transportError)) throw std::runtime_error(transportError);
    if (!RayTracing::ValidatePhotonMappingOptions(config.Photons, transportError)) throw std::runtime_error(transportError);
    if (!RayTracing::ValidateProgressivePhotonOptions(config.ProgressivePhotons, transportError)) throw std::runtime_error(transportError);
    if (!RayTracing::ValidateVertexConnectionMergingOptions(config.Vcm, transportError))
        throw std::runtime_error(transportError);
    if (!RayTracing::ValidatePointGiOptions(config.PointGi, transportError)) throw std::runtime_error(transportError);
    if (!RayTracing::ValidateMediumOptions(config.Integrator, config.Medium, transportError)) throw std::runtime_error(transportError);
    if (!RayTracing::ValidateMicropolygonOptions(config.Micropolygons, transportError))
        throw std::runtime_error(transportError);
    if (!RayTracing::ValidateReconstructionSupport(config.Camera, HasInstanceMotion(config),
                                                   config.Reconstruction, config.Medium, transportError))
        throw std::runtime_error(transportError);
    if (config.Adaptive.DenoiserAware && config.Reconstruction.Filter != RayTracing::ReconstructionFilter::SpatialAtrous)
        throw std::runtime_error("Denoiser-aware allocation requires spatial reconstruction");
    if (config.Integrator == RayTracing::Algorithm::MetropolisLightTransport && config.Sampling != RayTracing::SampleSequence::Independent)
        throw std::runtime_error("MLT requires independent primary samples");
    if (!RenderOutputKey(config.Output)) throw std::runtime_error("Unknown worker output mode");
    if (config.Output != RenderOutput::Png && config.Reconstruction.Filter != RayTracing::ReconstructionFilter::None &&
        !config.Adaptive.DenoiserAware)
        throw std::runtime_error("Raw film jobs reconstruct only after full-frame assembly");
    if (config.Output == RenderOutput::Png && config.OutputPath.empty()) throw std::runtime_error("Select a PNG output path");
}
} // namespace

void ValidateRenderConfig(const RenderConfig& config) { Validate(config); }

void PrepareRemoteMicropolygons(RenderConfig& config, const RayTracing::CancellationToken& cancellation,
                                const std::string& jobId)
{
    if (!config.Micropolygons.Enabled || config.DicingPlan)
        return;
    ValidateRenderConfig(config);
    // Planning has its own cancellation identity. A failed planning attempt
    // must not blacklist the later tile job on that worker.
    const auto id = jobId.empty() ? JobIdentifier() : jobId + "-dicing";
    const auto request = RequestBody(config, id, {0, 0, 1, 1, 0});
    const Http::Bytes body(request.begin(), request.end());
    std::string lastError = "No micropolygon planning worker is available";
    for (const auto& endpoint : config.Servers)
    {
        if (cancellation.IsCancellationRequested())
            throw std::runtime_error("Micropolygon planning cancelled");
        try
        {
            const auto status = Http::Request(endpoint, "GET", "/status", {}, config.Token, 2);
            if (status.Status != 200)
                throw std::runtime_error("Cannot inspect micropolygon worker");
            RequireMicropolygonWorker(Http::ParseFields(Http::Text(status.Body)));
            const auto response = Http::RequestBinary(
                endpoint, "POST", "/micropolygon-plan", body, config.Token, 600,
                [&] { return cancellation.IsCancellationRequested(); }, MaximumMicropolygonReplyBytes, id);
            if (response.Status != 200)
                throw std::runtime_error(Http::Text(response.Body));
            const auto fields = Http::ParseFields(Http::Text(response.Body));
            if (fields.at("micropolygon_wire") != MicropolygonWireVersion)
                throw std::runtime_error("Micropolygon plan wire mismatch");
            const auto& identity = fields.at("scene_hash");
            uint64_t sceneHash = 0;
            const auto parsed =
                std::from_chars(identity.data(), identity.data() + identity.size(), sceneHash);
            if (parsed.ec != std::errc{} || parsed.ptr != identity.data() + identity.size() ||
                (config.ExpectedSceneHash && config.ExpectedSceneHash != sceneHash))
                throw std::runtime_error("Planning worker assets differ from the preview scene");
            auto plan = ParseDicingPlan(fields.at("dicing_plan"));
            if (cancellation.IsCancellationRequested())
                throw std::runtime_error("Micropolygon planning cancelled");
            config.DicingPlan = std::move(plan);
            return;
        }
        catch (const std::exception& error)
        {
            lastError = endpoint + ": " + error.what();
            // Cancel the exact planning job when its observation connection
            // fails; otherwise an abandoned native dispatch can hold the worker.
            try
            {
                Http::Request(endpoint, "POST", "/cancel", Http::EncodeFields({{"job", id}}), config.Token,
                              2);
            }
            catch (...)
            {
            }
        }
    }
    throw std::runtime_error(lastError);
}

struct RenderClient::Impl
{
    mutable std::mutex Mutex;
    Progress Current;
    RenderConfig Config;
    std::string Id;
    std::atomic<bool> CancelRequested{false};
    std::atomic<bool> AbortRequests{false};
    std::atomic<bool> Finished{true};
    std::thread Coordinator;
    Clock::time_point Started;
    std::shared_ptr<const FrameData> Result;
    RayTracing::CancellationSource PreparationCancellation;

    void Run()
    {
        try { Render(); }
        catch (const std::exception& error)
        {
            std::lock_guard<std::mutex> lock(Mutex);
            Current.State = CancelRequested.load() ? JobState::Cancelled : JobState::Failed;
            Current.Error = error.what();
        }
        {
            std::lock_guard<std::mutex> lock(Mutex);
            Current.ElapsedSeconds = std::chrono::duration<double>(Clock::now() - Started).count();
        }
        Finished.store(true);
    }

    void Render()
    {
        PrepareRemoteMicropolygons(Config, PreparationCancellation.GetToken(), Id);
        std::deque<Tile> tiles;
        for (std::uint32_t y = 0; y < Config.Height; y += Config.TileSize)
            for (std::uint32_t x = 0; x < Config.Width; x += Config.TileSize)
                tiles.push_back({x, y, std::min(Config.TileSize, Config.Width - x), std::min(Config.TileSize, Config.Height - y), 0});
        const bool rawFilm = Config.Output == RenderOutput::LinearFilm || Config.Output == RenderOutput::GuidedFilm;
        std::unique_ptr<FilmAssembly> assembly;
        std::vector<std::uint8_t> pixels;
        if (rawFilm) assembly = std::make_unique<FilmAssembly>(Config);
        else pixels.resize(static_cast<std::size_t>(Config.Width) * Config.Height * 4);
        std::mutex queueMutex;
        std::condition_variable ready;
        std::uint32_t inFlight = 0;
        std::uint64_t sceneHash = 0;
        bool hasSceneHash = false;
        std::atomic<bool> stopMonitoring{false};
        std::atomic<bool> fatal{false};
        std::string lastError;

        std::vector<std::thread> workers;
        workers.reserve(Config.Servers.size());
        std::thread monitor([&] {
          try {
            while (!stopMonitoring.load() && !AbortRequests.load())
            {
                for (std::size_t index = 0; index < Config.Servers.size() && !stopMonitoring.load(); ++index)
                {
                    if (AbortRequests.load()) break;
                    try
                    {
                        const auto response = Http::Request(Config.Servers[index], "GET", "/status", {}, Config.Token, 2, &AbortRequests);
                        if (response.Status != 200) continue;
                        const auto fields = Http::ParseFields(Http::Text(response.Body));
                        if (fields.at("version") != Protocol::Version) continue;
                        std::lock_guard<std::mutex> lock(Mutex);
                        auto& server = Current.Servers[index];
                        server.CpuPercent = Utilization(fields.at("cpu"));
                        server.GpuPercent = Utilization(fields.at("gpu"));
                        server.Backend = fields.at("backend");
                    }
                    catch (...) {}
                }
                for (int interval = 0; interval < 5 && !stopMonitoring.load(); ++interval)
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
          } catch (...) {}
        });

        try
        {
        for (std::size_t index = 0; index < Config.Servers.size(); ++index)
        {
            workers.emplace_back([&, index] {
              try {
                  bool micropolygonReady = false;
                  for (;;)
                  {
                      Tile tile;
                      {
                          std::unique_lock<std::mutex> lock(queueMutex);
                          ready.wait_for(lock, std::chrono::milliseconds(200), [&] {
                              return !tiles.empty() || inFlight == 0 || AbortRequests.load() || fatal.load();
                          });
                          if (AbortRequests.load() || fatal.load())
                              return;
                          if (tiles.empty())
                          {
                              if (inFlight == 0)
                                  return;
                              continue;
                          }
                          tile = tiles.front();
                          tiles.pop_front();
                          ++inFlight;
                      }
                      {
                          std::lock_guard<std::mutex> lock(Mutex);
                          Current.Servers[index].Status =
                              "Rendering " + std::to_string(tile.X) + "," + std::to_string(tile.Y);
                      }
                      try
                      {
                          if (Config.Micropolygons.Enabled && !micropolygonReady)
                          {
                              const auto status = Http::Request(Config.Servers[index], "GET", "/status", {},
                                                                Config.Token, 2, &AbortRequests);
                              if (status.Status != 200)
                                  throw std::runtime_error("Cannot inspect micropolygon worker");
                              RequireMicropolygonWorker(Http::ParseFields(Http::Text(status.Body)));
                              micropolygonReady = true;
                          }
                          const auto request = RequestBody(Config, Id, tile);
                          constexpr size_t DisplayResponseLimit = 16 * 1024 * 1024;
                          const auto response = Http::Request(Config.Servers[index], "POST", "/render",
                                                              request, Config.Token, 600, &AbortRequests,
                                                              rawFilm ? MaximumFilmTileBytes : DisplayResponseLimit);
                          if (response.Status != 200)
                              throw std::runtime_error("HTTP " + std::to_string(response.Status) + ": " +
                                                       Http::Text(response.Body));
                          const auto& bytes = response.Body;
                          FilmTile film;
                          uint64_t hash = 0, milliseconds = 0;
                          if (rawFilm)
                          {
                              film = DecodeFilmTile(bytes);
                              if (film.X != tile.X || film.Y != tile.Y || film.Film.Width != tile.Width ||
                                  film.Film.Height != tile.Height)
                                  throw std::runtime_error("Worker returned another film rectangle");
                              hash = film.SceneHash;
                              milliseconds = film.RenderMilliseconds;
                          }
                          else
                          {
                              if (bytes.size() !=
                                      28 + static_cast<std::size_t>(tile.Width) * tile.Height * 4 ||
                                  bytes[0] != 'A' || bytes[1] != 'R' || bytes[2] != 'V' || bytes[3] != '1' ||
                                  Protocol::Read32(bytes, 4) != tile.Width ||
                                  Protocol::Read32(bytes, 8) != tile.Height)
                                  throw std::runtime_error("Worker returned a corrupt tile response");
                              hash = Protocol::Read64(bytes, 20);
                              milliseconds = Protocol::Read64(bytes, 12);
                          }
                          if (Config.ExpectedSceneHash && hash != Config.ExpectedSceneHash)
                              throw std::runtime_error(
                                  "Worker assets differ from the client's preview scene");
                          {
                              std::lock_guard<std::mutex> lock(queueMutex);
                              if (hasSceneHash && hash != sceneHash)
                                  throw std::runtime_error(
                                      "Worker scene assets differ from the other workers");
                              if (rawFilm)
                                  assembly->Add(film, FilmRequestHash(request));
                              else
                                  for (std::uint32_t row = 0; row < tile.Height; ++row)
                                      std::copy_n(bytes.data() + 28 +
                                                      static_cast<std::size_t>(row) * tile.Width * 4,
                                                  static_cast<std::size_t>(tile.Width) * 4,
                                                  pixels.data() +
                                                      (static_cast<std::size_t>(tile.Y + row) * Config.Width +
                                                       tile.X) *
                                                          4);
                              sceneHash = hash;
                              hasSceneHash = true;
                              --inFlight;
                          }
                          {
                              std::lock_guard<std::mutex> lock(Mutex);
                              ++Current.CompletedTiles;
                              auto& server = Current.Servers[index];
                              ++server.CompletedTiles;
                              if (rawFilm)
                                  server.NativeBackend = film.Backend;
                              server.RenderMilliseconds += static_cast<double>(milliseconds);
                              server.Status = "Ready";
                          }
                          ready.notify_all();
                      }
                      catch (const std::exception& error)
                      {
                          {
                              std::lock_guard<std::mutex> lock(queueMutex);
                              --inFlight;
                              if (lastError.empty() || !AbortRequests.load())
                                  lastError = Config.Servers[index] + ": " + error.what();
                              if (!AbortRequests.load())
                              {
                                  if (++tile.Attempts <= Config.MaxRetries)
                                      tiles.push_front(tile);
                                  else
                                  {
                                      fatal.store(true);
                                      AbortRequests.store(true);
                                  }
                              }
                          }
                          {
                              std::lock_guard<std::mutex> lock(Mutex);
                              auto& server = Current.Servers[index];
                              ++server.FailedAttempts;
                              server.Status = CancelRequested.load() ? "Cancelled" : error.what();
                          }
                          ready.notify_all();
                          // Cancellation runs on the corresponding endpoint thread, so many
                          // unreachable cloud workers do not serialize the shutdown timeout.
                          try
                          { Http::Request(Config.Servers[index], "POST", "/cancel", Http::EncodeFields({{"job", Id}}), Config.Token, 2); }
                        catch (...) {}
                        // A failed endpoint is retired for this job. Other workers retry its tile.
                        return;
                    }
                }
              } catch (...) { fatal.store(true); AbortRequests.store(true); ready.notify_all(); }
            });
        }
        for (auto& worker : workers) worker.join();
        }
        catch (...)
        {
            AbortRequests.store(true);
            fatal.store(true);
            ready.notify_all();
            for (auto& worker : workers) if (worker.joinable()) worker.join();
            stopMonitoring.store(true);
            if (monitor.joinable()) monitor.join();
            throw;
        }
        stopMonitoring.store(true);
        monitor.join();
        if (CancelRequested.load())
        {
            std::lock_guard<std::mutex> lock(Mutex);
            Current.State = JobState::Cancelled;
            return;
        }
        {
            std::lock_guard<std::mutex> lock(Mutex);
            if (Current.CompletedTiles != Current.TotalTiles)
                throw std::runtime_error(lastError.empty() ? "All workers stopped before completing the image" : lastError);
        }
        std::shared_ptr<const FrameData> frame;
        if (rawFilm) frame = assembly->Finish();
        else WritePng(Config.OutputPath, Config.Width, Config.Height, pixels, [this] { return CancelRequested.load(); });
        std::lock_guard<std::mutex> lock(Mutex);
        if (rawFilm && CancelRequested.load()) { Current.State = JobState::Cancelled; return; }
        Result = std::move(frame);
        Current.State = JobState::Completed;
    }
};

RenderClient::RenderClient() : Impl_(std::make_unique<Impl>()) {}
RenderClient::~RenderClient() { Cancel(); Wait(); }

void RenderClient::Start(RenderConfig config)
{
    Validate(config);
    if (!Impl_->Finished.load()) throw std::runtime_error("An offline render is already running");
    Wait();
    Impl_->Config = std::move(config);
    Impl_->Id = JobIdentifier();
    Impl_->Started = Clock::now();
    {
        std::lock_guard<std::mutex> lock(Impl_->Mutex);
        Impl_->CancelRequested.store(false);
        Impl_->AbortRequests.store(false);
        Impl_->PreparationCancellation = RayTracing::CancellationSource{};
        Impl_->Current = {};
        Impl_->Result.reset();
        Impl_->Current.State = JobState::Running;
        if (Impl_->Config.Output == RenderOutput::Png) Impl_->Current.OutputPath = Impl_->Config.OutputPath;
        Impl_->Current.TotalTiles = ((Impl_->Config.Width + Impl_->Config.TileSize - 1) / Impl_->Config.TileSize) *
                                   ((Impl_->Config.Height + Impl_->Config.TileSize - 1) / Impl_->Config.TileSize);
        for (const auto& url : Impl_->Config.Servers)
        {
            ServerProgress server;
            server.Url = url;
            server.Backend = Impl_->Config.Backend;
            Impl_->Current.Servers.push_back(std::move(server));
        }
    }
    Impl_->Finished.store(false);
    try { Impl_->Coordinator = std::thread([this] { Impl_->Run(); }); }
    catch (...)
    {
        Impl_->Finished.store(true);
        std::lock_guard<std::mutex> lock(Impl_->Mutex);
        Impl_->Current.State = JobState::Failed;
        Impl_->Current.Error = "Cannot create the render coordinator thread";
        throw;
    }
}

void RenderClient::Cancel()
{
    std::lock_guard<std::mutex> lock(Impl_->Mutex);
    Impl_->CancelRequested.store(true);
    Impl_->AbortRequests.store(true);
    Impl_->PreparationCancellation.Cancel();
}
bool RenderClient::IsRunning() const { return !Impl_->Finished.load(); }
void RenderClient::Wait() { if (Impl_->Coordinator.joinable()) Impl_->Coordinator.join(); }

Progress RenderClient::Snapshot() const
{
    std::lock_guard<std::mutex> lock(Impl_->Mutex);
    auto snapshot = Impl_->Current;
    if (snapshot.State == JobState::Running)
        snapshot.ElapsedSeconds = std::chrono::duration<double>(Clock::now() - Impl_->Started).count();
    return snapshot;
}
std::shared_ptr<const FrameData> RenderClient::Result() const
{
    std::lock_guard<std::mutex> lock(Impl_->Mutex);
    return Impl_->Current.State == JobState::Completed ? Impl_->Result : nullptr;
}
} // namespace Arvo::Distributed
