#pragma once

#include "Renderer/RayTracing.h"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace Arvo::Distributed
{

struct MaterialOverride
{
    std::uint32_t Index = 0;
    RayTracing::MaterialScattering Scattering;
};
struct InstanceMotionOverride
{
    std::uint32_t Index = 0;
    RayTracing::MotionData::RigidMotion Motion;
};

enum class RenderOutput { Png, LinearFilm, GuidedFilm };
const char* RenderOutputKey(RenderOutput output);
bool ParseRenderOutput(const std::string& key, RenderOutput& output);

// Published only after every tile has passed identity and coverage checks.
// Moments retain their native meaning; a correlated estimator does not become
// independent merely because its pixels were transported over the network.
struct FrameData
{
    RayTracing::LinearFilm Film;
    RayTracing::GeometryGuideImage Geometry;
    RayTracing::Camera View;
    RayTracing::Algorithm Integrator = RayTracing::Algorithm::PathTracing;
    RayTracing::RadiometryMode Radiometry = RayTracing::RadiometryMode::RGB;
    RayTracing::SampleSequence Sampling = RayTracing::SampleSequence::Independent;
    uint32_t SamplingSeed = 0;
    uint64_t SceneHash = 0;
    std::vector<RayTracing::BackendType> Backends;
};

struct RenderConfig
{
    std::vector<std::string> Servers{"http://127.0.0.1:7831", "http://127.0.0.1:7832"};
    std::string Token;
    std::string Scene = "cornell";
    std::string Backend = "compute";
    RayTracing::Camera Camera;
    RayTracing::Algorithm Integrator = RayTracing::Algorithm::PathTracing;
    RayTracing::RadiometryMode Radiometry = RayTracing::RadiometryMode::RGB;
    RayTracing::TransportOptions Transport;
    RayTracing::SampleSequence Sampling = RayTracing::SampleSequence::Independent;
    uint32_t SamplingSeed = 0;
    RayTracing::AdaptiveOptions Adaptive;
    RayTracing::MediumOptions Medium;
    RayTracing::MicropolygonOptions Micropolygons;
    std::shared_ptr<const RayTracing::MicropolygonPlan> DicingPlan;
    RayTracing::ReconstructionOptions Reconstruction;
    RayTracing::MetropolisOptions Metropolis;
    RayTracing::PhotonMappingOptions Photons;
    RayTracing::ProgressivePhotonOptions ProgressivePhotons;
    RayTracing::VertexConnectionMergingOptions Vcm;
    RayTracing::PointGiOptions PointGi;
    RayTracing::AccelerationType Acceleration = RayTracing::AccelerationType::Bvh;
    RayTracing::BvhBuildPolicy BvhPolicy = RayTracing::BvhBuildPolicy::Default;
    RayTracing::GeometryMode Geometry = RayTracing::GeometryMode::Flat;
    std::vector<MaterialOverride> Materials;
    std::vector<InstanceMotionOverride> InstanceMotions;
    std::uint32_t Width = 3840;
    std::uint32_t Height = 2160;
    std::uint32_t TileSize = 256;
    std::uint32_t SamplesPerPixel = 16;
    std::uint32_t MaxBounces = 4;
    std::uint32_t MaxRetries = 2;
    // A nonzero hash also verifies the worker assets against the client's preview scene.
    std::uint64_t ExpectedSceneHash = 0;
    bool ShadowsEnabled = true;
    RenderOutput Output = RenderOutput::Png;
    std::filesystem::path OutputPath = "Binaries/Renders/render.png";
};

void ValidateRenderConfig(const RenderConfig& config);
void PrepareRemoteMicropolygons(RenderConfig&, const RayTracing::CancellationToken&,
                                const std::string& jobId = {});
bool HasInstanceMotion(const RenderConfig& config);

enum class JobState { Idle, Running, Completed, Cancelled, Failed };

struct ServerProgress
{
    std::string Url;
    std::string Backend;
    std::string Status = "Waiting";
    // From an accepted raw film tile, independent of status-monitor display text.
    std::optional<RayTracing::BackendType> NativeBackend;
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
    std::uint32_t CompletedFrames = 0;
    std::uint32_t TotalFrames = 0;
    std::string Stage;
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
    // Only completed raw film jobs expose this result; PNG jobs return null.
    [[nodiscard]] std::shared_ptr<const FrameData> Result() const;
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
void WriteBinaryAtomic(const std::filesystem::path& path, const std::vector<std::uint8_t>& bytes,
                       const std::function<bool()>& cancelled = {});
void WritePng(const std::filesystem::path& path, std::uint32_t width,
              std::uint32_t height, const std::vector<std::uint8_t>& rgba,
              const std::function<bool()>& cancelled = {});

} // namespace Arvo::Distributed
