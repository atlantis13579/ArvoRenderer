#include "Distributed/Distributed.h"
#include "Distributed/Http.h"
#include "Distributed/Protocol.h"
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
    std::map<std::string, std::shared_ptr<const RayTracing::Scene>> Scenes;
    std::map<std::string, std::uint64_t> SceneHashes;
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

void Render(WorkerState& state, Http::Socket socket, const std::map<std::string, std::string>& fields)
{
    if (fields.at("version") != "1") throw std::runtime_error("Unsupported worker protocol version");
    const auto& sceneName = fields.at("scene");
    if (sceneName != "cornell" && sceneName != "sponza") throw std::runtime_error("Only cornell and sponza are supported");
    auto backend = fields.at("backend");
    if (backend == "auto") backend = state.Config.Backend;
    Backend(backend);
    const auto& jobId = fields.at("job");
    if (jobId.empty() || jobId.size() > 128) throw std::runtime_error("Invalid job identifier");
    RayTracing::RenderSettings settings;
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
    if (algorithm != "ray" && algorithm != "path") throw std::runtime_error("Unknown rendering algorithm");
    settings.Integrator = algorithm == "path" ? RayTracing::Algorithm::PathTracing : RayTracing::Algorithm::RayTracing;
    RayTracing::Camera camera;
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
        const auto rhi = GetBackend(state, backend);
        if (!state.Scenes.count(sceneName))
        {
            const auto snapshot = state.Catalog.Load(sceneName == "cornell" ? Editor::DemoId::CornellBox : Editor::DemoId::SponzaAtrium);
            auto scene = Editor::BuildOfflineScene(snapshot);
            if (!scene) throw std::runtime_error("Cannot prepare worker scene");
            state.SceneHashes[sceneName] = Protocol::SceneHash(*scene);
            state.Scenes[sceneName] = std::move(scene);
        }
        {
            std::lock_guard<std::mutex> lock(state.StatusMutex);
            state.Status = "Rendering " + std::to_string(settings.TileX) + "," + std::to_string(settings.TileY);
        }
        const auto start = Clock::now();
        auto job = rhi->RenderAsync(state.Scenes.at(sceneName), camera, settings);
        while (!job.IsReady())
        {
            if (cancellation.IsCancellationRequested()) job.Cancel();
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        auto result = job.GetResult();
        const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - start).count();
        {
            std::lock_guard<std::mutex> lock(state.StatusMutex);
            state.CurrentJob.clear();
            state.Status = result.Status == RayTracing::RenderStatus::Completed ? "Ready" : "Render stopped";
            if (result.Status == RayTracing::RenderStatus::Completed) ++state.CompletedTiles;
        }
        if (result.Status == RayTracing::RenderStatus::Cancelled || cancellation.IsCancellationRequested())
        { Http::Reply(socket, 409, "Job cancelled"); return; }
        if (result.Status != RayTracing::RenderStatus::Completed) throw std::runtime_error(result.Error);
        if (result.Output.GetWidth() != settings.Width || result.Output.GetHeight() != settings.Height)
            throw std::runtime_error("Backend returned invalid tile dimensions");
        Http::Bytes bytes{'A', 'R', 'V', '1'};
        Protocol::Append32(bytes, settings.Width); Protocol::Append32(bytes, settings.Height);
        Protocol::Append64(bytes, static_cast<std::uint64_t>(milliseconds));
        Protocol::Append64(bytes, state.SceneHashes.at(sceneName));
        const auto& pixels = result.Output.GetPixelsRGBA8();
        bytes.insert(bytes.end(), pixels.begin(), pixels.end());
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

void Handle(WorkerState& state, Http::Socket socket)
{
    try
    {
        const auto request = Http::Read(socket, 16384, nullptr, 30);
        if (!state.Config.Token.empty())
        {
            const auto authorization = request.Headers.find("authorization");
            if (authorization == request.Headers.end() || authorization->second != "Bearer " + state.Config.Token)
            { Http::Reply(socket, 401, "Worker authorization required"); Http::Close(socket); --state.Connections; return; }
        }
        if (request.FirstLine == "GET /status HTTP/1.1" || request.FirstLine == "GET /status HTTP/1.0")
        {
            std::map<std::string, std::string> fields;
            {
                std::lock_guard<std::mutex> lock(state.StatusMutex);
                fields = {{"version", "1"}, {"status", state.Status}, {"backend", state.ActiveBackend.empty() ? state.Config.Backend : state.ActiveBackend},
                    {"job", state.CurrentJob}, {"cpu", Protocol::Number(state.CpuPercent)}, {"gpu", Protocol::Number(state.GpuPercent)},
                    {"gpu_scope", "device0"}, {"cpu_scope", "process_normalized"}, {"completed", std::to_string(state.CompletedTiles)}};
            }
            Http::Reply(socket, 200, Http::EncodeFields(fields));
        }
        else if (request.FirstLine == "POST /render HTTP/1.1") Render(state, socket, Http::ParseFields(Http::Text(request.Body)));
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
    catch (const std::exception& error)
    {
        try { Http::Reply(socket, 400, error.what()); } catch (...) {}
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
