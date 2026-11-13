#include "Distributed/OfflineRenderer.h"
#include "Distributed/TemporalTransfer.h"
#include <atomic>
#include <chrono>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace Arvo::Distributed
{
namespace
{
namespace R = RayTracing;
using Clock = std::chrono::steady_clock;
constexpr auto TilePollInterval = std::chrono::milliseconds(10);

R::RenderSettings Settings(const RenderConfig& config)
{
    R::RenderSettings result;
    result.Width = config.Width;
    result.Height = config.Height;
    result.Integrator = config.Integrator;
    result.Radiometry = config.Radiometry;
    result.Transport = config.Transport;
    result.Sampling = config.Sampling;
    result.SamplingSeed = config.SamplingSeed;
    result.Adaptive = config.Adaptive;
    result.Medium = config.Medium;
    result.Micropolygons = config.Micropolygons;
    result.DicingPlan = config.DicingPlan;
    result.Metropolis = config.Metropolis;
    result.Photons = config.Photons;
    result.ProgressivePhotons = config.ProgressivePhotons;
    result.Vcm = config.Vcm;
    result.PointGi = config.PointGi;
    result.Acceleration = config.Acceleration;
    result.BvhPolicy = config.BvhPolicy;
    result.SamplesPerPixel = config.SamplesPerPixel;
    result.MaxReflectionBounces = config.MaxBounces;
    result.ShadowsEnabled = config.ShadowsEnabled;
    return result;
}
void CheckCancellation(const R::CancellationToken& token)
{
    if (token.IsCancellationRequested())
        throw std::runtime_error("Offline rendering cancelled");
}
} // namespace

void ValidateTemporalRendering(const RenderConfig& config, const TemporalRenderOptions& options)
{
    ValidateRenderConfig(config);
    if (config.Output != RenderOutput::Png)
        throw std::invalid_argument("Offline rendering requires a PNG output request");
    if (!options.FramesPerRender || options.FramesPerRender > MaximumTemporalBatchFrames)
        throw std::invalid_argument("Temporal batch frames must be between 1 and 256");
    std::string error;
    if (!R::ValidateTemporalOptions(options.Filter, error))
        throw std::invalid_argument(error);
    if (options.Enabled &&
        (config.Camera.Lens.Radius != 0 || config.Camera.Motion.Enabled || HasInstanceMotion(config) ||
         config.Reconstruction.Filter != R::ReconstructionFilter::None || config.Medium.Enabled))
        throw std::invalid_argument("Temporal offline rendering requires an instantaneous pinhole camera, no "
                                    "object motion or participating medium, and spatial filtering off");
    if (options.Enabled && !options.Filter.GammaCorrection)
        throw std::invalid_argument("PNG temporal reconstruction requires display-sRGB encoding");
}

struct OfflineRenderClient::Impl
{
    mutable std::mutex Mutex;
    Progress Current;
    Clock::time_point Started;
    std::atomic<bool> Finished{true};
    std::thread Coordinator;
    R::CancellationSource Cancellation;
    std::shared_ptr<RenderClient> ActiveTiles;
    std::unique_ptr<R::TemporalSession> Accepted = std::make_unique<R::TemporalSession>();
    uint64_t AcceptedSceneHash = 0;

    void Stage(std::string stage)
    {
        std::lock_guard<std::mutex> lock(Mutex);
        Current.Stage = std::move(stage);
    }
    Progress Tiles(RenderConfig config, const R::CancellationToken& token)
    {
        auto client = std::make_shared<RenderClient>();
        {
            std::lock_guard<std::mutex> lock(Mutex);
            ActiveTiles = client;
            Current.Stage = "Rendering tiles";
        }
        CheckCancellation(token);
        client->Start(std::move(config));
        while (client->IsRunning())
        {
            // Start clears the tile client's cancellation state. Repeating this
            // bridges cancellation that races its thread creation.
            if (token.IsCancellationRequested())
                client->Cancel();
            std::this_thread::sleep_for(TilePollInterval);
        }
        client->Wait();
        auto progress = client->Snapshot();
        {
            std::lock_guard<std::mutex> lock(Mutex);
            Current.CompletedTiles = progress.CompletedTiles;
            Current.TotalTiles = progress.TotalTiles;
            Current.Servers = progress.Servers;
        }
        if (progress.State != JobState::Completed)
            throw std::runtime_error(progress.Error.empty() ? "Tile rendering did not complete"
                                                            : progress.Error);
        return progress;
    }
    R::TemporalResult Reconstruct(const RenderConfig& config, const FrameData& assembled,
                                  const Progress& progress, const R::TemporalFrame& frame,
                                  const R::TemporalHistory* previous, const R::TemporalOptions& options,
                                  const R::CancellationToken& token)
    {
        if (assembled.Backends.empty())
            throw std::runtime_error("No native film producer");
        // Auto may assemble heterogeneous tiles. Choose the first recorded native
        // type deterministically, then retry only producers of that same type.
        const auto backend = assembled.Backends.front();
        R::TemporalResult result;
        result.Error = "No successful producer can reconstruct this frame";
        for (const auto& server : progress.Servers)
        {
            CheckCancellation(token);
            if (!server.CompletedTiles || server.NativeBackend != backend)
                continue;
            Stage("Temporal reconstruction on " + server.Url);
            result = ReconstructRemoteTemporal({server.Url, config.Token, backend}, frame, previous, options,
                                               token);
            if (result.Status == R::RenderStatus::Completed || result.Status == R::RenderStatus::Cancelled)
                break;
        }
        return result;
    }
    void Render(RenderConfig config, const TemporalRenderOptions& options, const R::CancellationToken& token)
    {
        std::unique_ptr<R::TemporalSession> transaction;
        uint64_t sceneHash;
        {
            std::lock_guard<std::mutex> lock(Mutex);
            transaction = Accepted->Fork();
            sceneHash = config.ExpectedSceneHash ? config.ExpectedSceneHash : AcceptedSceneHash;
        }
        if (!options.Enabled)
        {
            transaction->Reset();
            Tiles(config, token);
            // RenderClient's successful PNG publication is already its commit
            // point. A late cancellation cannot revoke that completed save.
            std::lock_guard<std::mutex> lock(Mutex);
            Accepted.swap(transaction);
            AcceptedSceneHash = 0;
            Current.CompletedFrames = 1;
            Current.State = JobState::Completed;
            return;
        }

        if (config.Micropolygons.Enabled)
        {
            Stage("Preparing shared micropolygon topology");
            PrepareRemoteMicropolygons(config, token);
        }
        const auto settings = Settings(config);
        R::Image display;
        for (uint32_t index = 0; index < options.FramesPerRender; ++index)
        {
            CheckCancellation(token);
            auto ticket = transaction->Begin(settings, sceneHash, options.Filter);
            auto raw = config;
            raw.Output = RenderOutput::GuidedFilm;
            raw.SamplingSeed = ticket.Settings().SamplingSeed;
            const auto progress = Tiles(raw, token);
            CheckCancellation(token);
            const auto assembled = ActiveTiles->Result();
            if (!assembled)
                throw std::runtime_error("Completed tile job has no assembled film");
            if (assembled->SceneHash != sceneHash)
            {
                sceneHash = assembled->SceneHash;
                ticket = transaction->Begin(settings, sceneHash, options.Filter);
            }
            if (assembled->SamplingSeed != ticket.Settings().SamplingSeed)
                throw std::runtime_error("Assembled film has another sampling identity");
            auto result = transaction->Reconstruct(
                [&](const auto& frame, const auto* previous, const auto& filter, const auto& cancellation) {
                    return Reconstruct(config, *assembled, progress, frame, previous, filter, cancellation);
                },
                ticket, assembled->Film, assembled->Geometry, assembled->View, token);
            CheckCancellation(token);
            if (result.Status != R::RenderStatus::Completed || !result.Accept || !result.Accept())
                throw std::runtime_error(result.Error.empty() ? "Temporal reconstruction did not complete"
                                                              : result.Error);
            display = std::move(result.Display);
            std::lock_guard<std::mutex> lock(Mutex);
            ++Current.CompletedFrames;
            ActiveTiles.reset();
        }
        Stage("Saving PNG");
        WritePng(config.OutputPath, config.Width, config.Height, display.GetPixelsRGBA8(),
                 [&] { return token.IsCancellationRequested(); });
        // Publication is the transaction boundary. No cancellation check or
        // fallible history reconstruction may follow the atomic file rename.
        std::lock_guard<std::mutex> lock(Mutex);
        Accepted.swap(transaction);
        AcceptedSceneHash = sceneHash;
        Current.State = JobState::Completed;
    }
    void Run(RenderConfig config, TemporalRenderOptions options, R::CancellationToken token)
    {
        try
        {
            Render(config, options, token);
        }
        catch (const std::exception& error)
        {
            std::lock_guard<std::mutex> lock(Mutex);
            Current.State = token.IsCancellationRequested() ? JobState::Cancelled : JobState::Failed;
            Current.Error = error.what();
        }
        std::lock_guard<std::mutex> lock(Mutex);
        ActiveTiles.reset();
        Current.Stage.clear();
        Current.ElapsedSeconds = std::chrono::duration<double>(Clock::now() - Started).count();
        Finished.store(true);
    }
};

OfflineRenderClient::OfflineRenderClient() : Impl_(std::make_unique<Impl>()) {}
OfflineRenderClient::~OfflineRenderClient()
{
    Cancel();
    Wait();
}
void OfflineRenderClient::Start(RenderConfig config, TemporalRenderOptions options)
{
    ValidateTemporalRendering(config, options);
    if (IsRunning())
        throw std::logic_error("An offline render is already running");
    Wait();
    std::lock_guard<std::mutex> lock(Impl_->Mutex);
    Impl_->Cancellation = R::CancellationSource();
    Impl_->Current = {};
    Impl_->Current.State = JobState::Running;
    Impl_->Current.TotalFrames = options.Enabled ? options.FramesPerRender : 1;
    Impl_->Current.OutputPath = config.OutputPath;
    Impl_->Started = Clock::now();
    Impl_->Finished.store(false);
    try
    {
        Impl_->Coordinator =
            std::thread(&Impl::Run, Impl_.get(), std::move(config), options, Impl_->Cancellation.GetToken());
    }
    catch (...)
    {
        Impl_->Current.State = JobState::Failed;
        Impl_->Finished.store(true);
        throw;
    }
}
void OfflineRenderClient::Cancel()
{
    std::shared_ptr<RenderClient> tiles;
    {
        std::lock_guard<std::mutex> lock(Impl_->Mutex);
        Impl_->Cancellation.Cancel();
        tiles = Impl_->ActiveTiles;
    }
    if (tiles)
        tiles->Cancel();
}
void OfflineRenderClient::Wait()
{
    if (Impl_->Coordinator.joinable())
        Impl_->Coordinator.join();
}
bool OfflineRenderClient::IsRunning() const
{
    return !Impl_->Finished.load();
}
Progress OfflineRenderClient::Snapshot() const
{
    Progress result;
    std::shared_ptr<RenderClient> tiles;
    {
        std::lock_guard<std::mutex> lock(Impl_->Mutex);
        result = Impl_->Current;
        tiles = Impl_->ActiveTiles;
        if (result.State == JobState::Running)
            result.ElapsedSeconds = std::chrono::duration<double>(Clock::now() - Impl_->Started).count();
    }
    if (tiles)
    {
        auto progress = tiles->Snapshot();
        result.CompletedTiles = progress.CompletedTiles;
        result.TotalTiles = progress.TotalTiles;
        result.Servers = std::move(progress.Servers);
    }
    return result;
}
void OfflineRenderClient::ResetHistory()
{
    std::lock_guard<std::mutex> lock(Impl_->Mutex);
    if (IsRunning())
        throw std::logic_error("Cannot reset history during an offline render");
    Impl_->Accepted->Reset();
    Impl_->AcceptedSceneHash = 0;
}
std::shared_ptr<const RayTracing::TemporalHistory> OfflineRenderClient::History() const
{
    std::lock_guard<std::mutex> lock(Impl_->Mutex);
    return Impl_->Accepted->History();
}
} // namespace Arvo::Distributed
