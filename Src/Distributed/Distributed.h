#pragma once

#include "Renderer/RayTracing.h"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace Arvo::Distributed
{

struct RenderConfig
{
    std::vector<std::string> Servers{"http://127.0.0.1:7831", "http://127.0.0.1:7832"};
    std::string Token;
    std::string Scene = "cornell";
    std::string Backend = "compute";
    RayTracing::Camera Camera;
    RayTracing::Algorithm Integrator = RayTracing::Algorithm::PathTracing;
    std::uint32_t Width = 3840;
    std::uint32_t Height = 2160;
    std::uint32_t TileSize = 256;
    std::uint32_t SamplesPerPixel = 16;
    std::uint32_t MaxBounces = 4;
    std::uint32_t MaxRetries = 2;
    // A nonzero hash also verifies the worker assets against the client's preview scene.
    std::uint64_t ExpectedSceneHash = 0;
    bool ShadowsEnabled = true;
    std::filesystem::path OutputPath = "Binaries/Renders/render.png";
};

enum class JobState { Idle, Running, Completed, Cancelled, Failed };

struct ServerProgress
{
    std::string Url;
    std::string Backend;
    std::string Status = "Waiting";
    std::uint32_t CompletedTiles = 0;
    std::uint32_t FailedAttempts = 0;
    // Negative values mean that a real utilization measurement is unavailable.
    double CpuPercent = -1.0;
    double GpuPercent = -1.0;
    double RenderMilliseconds = 0.0;
};

struct Progress
{
    JobState State = JobState::Idle;
    std::uint32_t CompletedTiles = 0;
    std::uint32_t TotalTiles = 0;
    double ElapsedSeconds = 0.0;
    std::vector<ServerProgress> Servers;
    std::string Error;
    std::filesystem::path OutputPath;
};

class RenderClient
{
public:
    RenderClient();
    ~RenderClient();
    RenderClient(const RenderClient&) = delete;
    RenderClient& operator=(const RenderClient&) = delete;
    // Start returns immediately. A second job cannot replace a running job.
    void Start(RenderConfig config);
    void Cancel();
    [[nodiscard]] Progress Snapshot() const;
    [[nodiscard]] bool IsRunning() const;
    void Wait();

private:
    struct Impl;
    std::unique_ptr<Impl> Impl_;
};

struct WorkerConfig
{
    std::string BindAddress = "127.0.0.1";
    std::uint16_t Port = 7831;
    std::string Backend = "compute";
    std::string Token;
    std::filesystem::path ContentRoot;
};

// Runs until the process receives its normal shutdown signal.
int RunWorker(const WorkerConfig& config);
void WritePng(const std::filesystem::path& path, std::uint32_t width,
              std::uint32_t height, const std::vector<std::uint8_t>& rgba,
              const std::function<bool()>& cancelled = {});

} // namespace Arvo::Distributed
