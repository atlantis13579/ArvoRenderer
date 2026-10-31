#include "Distributed/Distributed.h"
#include "Distributed/Http.h"
#include "Distributed/Protocol.h"

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
    return Http::EncodeFields({
        {"version", "1"}, {"job", id}, {"scene", config.Scene}, {"backend", config.Backend},
        {"algorithm", config.Integrator == RayTracing::Algorithm::PathTracing ? "path" : "ray"},
        {"width", std::to_string(tile.Width)}, {"height", std::to_string(tile.Height)},
        {"full_width", std::to_string(config.Width)}, {"full_height", std::to_string(config.Height)},
        {"x", std::to_string(tile.X)}, {"y", std::to_string(tile.Y)},
        {"samples", std::to_string(config.SamplesPerPixel)}, {"bounces", std::to_string(config.MaxBounces)},
        {"shadows", config.ShadowsEnabled ? "1" : "0"},
        {"px", Protocol::Number(config.Camera.Position.X)}, {"py", Protocol::Number(config.Camera.Position.Y)},
        {"pz", Protocol::Number(config.Camera.Position.Z)}, {"tx", Protocol::Number(config.Camera.Target.X)},
        {"ty", Protocol::Number(config.Camera.Target.Y)}, {"tz", Protocol::Number(config.Camera.Target.Z)},
        {"ux", Protocol::Number(config.Camera.Up.X)}, {"uy", Protocol::Number(config.Camera.Up.Y)},
        {"uz", Protocol::Number(config.Camera.Up.Z)}, {"fov", Protocol::Number(config.Camera.VerticalFieldOfViewDegrees)}});
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
    if (config.Scene != "cornell" && config.Scene != "sponza") throw std::runtime_error("Select cornell or sponza");
    if (config.Backend != "cpu" && config.Backend != "cuda" && config.Backend != "compute" && config.Backend != "metal" && config.Backend != "auto")
        throw std::runtime_error("Unknown worker backend");
    if (!config.Width || !config.Height || config.Width > 16384 || config.Height > 16384 ||
        std::uint64_t(config.Width) * config.Height > 67108864)
        throw std::runtime_error("Output dimensions must be positive and at most 64 megapixels / 16384 per side");
    if (!config.TileSize || config.TileSize > 1024) throw std::runtime_error("Tile size must be between 1 and 1024");
    if (!config.SamplesPerPixel || config.SamplesPerPixel > 4096 || config.MaxBounces > 32 || config.MaxRetries > 16)
        throw std::runtime_error("Render samples, bounces or retries exceed the supported limits");
    if (config.Integrator != RayTracing::Algorithm::RayTracing && config.Integrator != RayTracing::Algorithm::PathTracing)
        throw std::runtime_error("Unknown render integrator");
    if (config.OutputPath.empty()) throw std::runtime_error("Select a PNG output path");
}
} // namespace

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
        std::deque<Tile> tiles;
        for (std::uint32_t y = 0; y < Config.Height; y += Config.TileSize)
            for (std::uint32_t x = 0; x < Config.Width; x += Config.TileSize)
                tiles.push_back({x, y, std::min(Config.TileSize, Config.Width - x), std::min(Config.TileSize, Config.Height - y), 0});
        std::vector<std::uint8_t> pixels(static_cast<std::size_t>(Config.Width) * Config.Height * 4, 0);
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
                        if (fields.at("version") != "1") continue;
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
                for (;;)
                {
                    Tile tile;
                    {
                        std::unique_lock<std::mutex> lock(queueMutex);
                        ready.wait_for(lock, std::chrono::milliseconds(200), [&] {
                            return !tiles.empty() || inFlight == 0 || AbortRequests.load() || fatal.load();
                        });
                        if (AbortRequests.load() || fatal.load()) return;
                        if (tiles.empty()) { if (inFlight == 0) return; continue; }
                        tile = tiles.front(); tiles.pop_front(); ++inFlight;
                    }
                    {
                        std::lock_guard<std::mutex> lock(Mutex);
                        Current.Servers[index].Status = "Rendering " + std::to_string(tile.X) + "," + std::to_string(tile.Y);
                    }
                    try
                    {
                        const auto response = Http::Request(Config.Servers[index], "POST", "/render", RequestBody(Config, Id, tile), Config.Token, 600, &AbortRequests);
                        if (response.Status != 200) throw std::runtime_error("HTTP " + std::to_string(response.Status) + ": " + Http::Text(response.Body));
                        const auto& bytes = response.Body;
                        if (bytes.size() != 28 + static_cast<std::size_t>(tile.Width) * tile.Height * 4 ||
                            bytes[0] != 'A' || bytes[1] != 'R' || bytes[2] != 'V' || bytes[3] != '1' ||
                            Protocol::Read32(bytes, 4) != tile.Width || Protocol::Read32(bytes, 8) != tile.Height)
                            throw std::runtime_error("Worker returned a corrupt tile response");
                        const auto hash = Protocol::Read64(bytes, 20);
                        if (Config.ExpectedSceneHash && hash != Config.ExpectedSceneHash)
                            throw std::runtime_error("Worker assets differ from the client's preview scene");
                        {
                            std::lock_guard<std::mutex> lock(queueMutex);
                            if (hasSceneHash && hash != sceneHash) throw std::runtime_error("Worker scene assets differ from the other workers");
                            sceneHash = hash; hasSceneHash = true;
                            for (std::uint32_t row = 0; row < tile.Height; ++row)
                                std::copy_n(bytes.data() + 28 + static_cast<std::size_t>(row) * tile.Width * 4,
                                            static_cast<std::size_t>(tile.Width) * 4,
                                            pixels.data() + (static_cast<std::size_t>(tile.Y + row) * Config.Width + tile.X) * 4);
                            --inFlight;
                        }
                        {
                            std::lock_guard<std::mutex> lock(Mutex);
                            ++Current.CompletedTiles;
                            auto& server = Current.Servers[index];
                            ++server.CompletedTiles;
                            server.RenderMilliseconds += static_cast<double>(Protocol::Read64(bytes, 12));
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
                                if (++tile.Attempts <= Config.MaxRetries) tiles.push_front(tile);
                                else { fatal.store(true); AbortRequests.store(true); }
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
                        try { Http::Request(Config.Servers[index], "POST", "/cancel", Http::EncodeFields({{"job", Id}}), Config.Token, 2); }
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
        WritePng(Config.OutputPath, Config.Width, Config.Height, pixels, [this] { return CancelRequested.load(); });
        std::lock_guard<std::mutex> lock(Mutex);
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
    Impl_->CancelRequested.store(false);
    Impl_->AbortRequests.store(false);
    Impl_->Started = Clock::now();
    {
        std::lock_guard<std::mutex> lock(Impl_->Mutex);
        Impl_->Current = {};
        Impl_->Current.State = JobState::Running;
        Impl_->Current.OutputPath = Impl_->Config.OutputPath;
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
    Impl_->CancelRequested.store(true);
    Impl_->AbortRequests.store(true);
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
} // namespace Arvo::Distributed
