#include "Core/Paths.h"
#include "Editor/RenderPanel.h"
#include "Editor/Imgui.h"
#include "Editor/ModuleAdapters.h"
#include "Distributed/Protocol.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace Arvo::Editor
{
namespace
{
constexpr const char* BackendIds[] = {"cpu", "cuda", "compute", "metal", "auto"};
constexpr std::uint32_t Widths[] = {1280, 1920, 3840};
constexpr std::uint32_t Heights[] = {720, 1080, 2160};
std::string Utilization(double value)
{
    if (value < 0.0) return "N/A";
    return std::to_string(static_cast<int>(value + 0.5)) + "%";
}
}
RenderPanel::RenderPanel()
{
#if defined(__APPLE__)
    Backend_ = 3;
#endif
    OutputDirectory_ = (Arvo::Core::BinaryRoot() / "Renders").string();
    if (const char* token = std::getenv("ARVO_WORKER_TOKEN")) Token_ = token;
}
RenderPanel::~RenderPanel() { Cancel(); }
void RenderPanel::Cancel()
{
    if (Preparation_.valid()) PreparationWasCancelled_ = true;
    if (PreparationCancelled_) PreparationCancelled_->store(true);
    Client_.Cancel();
}
Distributed::Progress RenderPanel::Progress() const
{
    auto progress = Client_.Snapshot();
    if (Preparation_.valid())
    {
        progress = {};
        progress.State = PreparationCancelled_ && PreparationCancelled_->load()
            ? Distributed::JobState::Cancelled : Distributed::JobState::Running;
    }
    if (!Error_.empty()) { progress.State = Distributed::JobState::Failed; progress.Error = Error_; }
    else if (PreparationWasCancelled_) progress.State = Distributed::JobState::Cancelled;
    return progress;
}
void RenderPanel::PollPreparation()
{
    if (!Preparation_.valid() || Preparation_.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return;
    try
    {
        auto config = Preparation_.get();
        if (!PreparationCancelled_->load()) Client_.Start(std::move(config));
    }
    catch (const std::exception& e) { Error_ = e.what(); }
}
void RenderPanel::Start(EditorCore& editor)
{
    try
    {
        if (editor.IsDemoLoading() || Preparation_.valid() || Client_.IsRunning()) return;
        PreparationWasCancelled_ = false;
        Distributed::RenderConfig config;
        config.Servers = Servers_;
        config.Token = Token_;
        config.Scene = editor.SelectedDemo() == DemoId::CornellBox ? "cornell" : "sponza";
        config.Backend = BackendIds[Backend_];
        config.Camera = BuildOfflineCamera(editor.Scene());
        config.Integrator = Algorithm_ == 0 ? RayTracing::Algorithm::RayTracing : RayTracing::Algorithm::PathTracing;
        config.Width = Widths[Resolution_];
        config.Height = Heights[Resolution_];
        config.TileSize = static_cast<std::uint32_t>(TileSize_);
        config.SamplesPerPixel = static_cast<std::uint32_t>(Samples_);
        config.MaxBounces = static_cast<std::uint32_t>(Bounces_);
        config.ShadowsEnabled = editor.Settings().ShadowsEnabled;
        const auto stamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        config.OutputPath = std::filesystem::path(OutputDirectory_) /
            (config.Scene + "-" + (Algorithm_ == 0 ? "ray-" : "path-") + std::to_string(stamp) + ".png");
        PreparationCancelled_ = std::make_shared<std::atomic<bool>>(false);
        Preparation_ = std::async(std::launch::async,
            [config = std::move(config), scene = editor.Scene(), cancelled = PreparationCancelled_]() mutable {
                if (!cancelled->load())
                    config.ExpectedSceneHash = Distributed::Protocol::SceneHash(*BuildOfflineScene(scene));
                return config;
            });
        Error_.clear();
    }
    catch (const std::exception& e) { Error_ = e.what(); }
}
void RenderPanel::Draw(EditorCore& editor, int height)
{
    PollPreparation();
    imguiBeginScrollArea("Arvo Studio", 10, 10, 330, std::max(120, height - 20), &Scroll_);
    const auto backends = editor.Backends();
    std::vector<const char*> labels;
    int selected = 0;
    for (std::size_t i = 0; i < backends.size(); ++i)
    {
        labels.push_back(backends[i].DisplayName.c_str());
        if (backends[i].Id == editor.SelectedBackend()) selected = static_cast<int>(i);
    }
    if (!labels.empty() && imguiCombo("Preview", &selected, labels.data(), static_cast<int>(labels.size())))
        editor.SelectBackend(backends[selected].Id);
    const char* scenes[] = {"Cornell Box", "Sponza"};
    int scene = editor.SelectedDemo() == DemoId::CornellBox ? 0 : 1;
    if (imguiCombo("Scene", &scene, scenes, 2))
        editor.SelectDemo(scene == 0 ? DemoId::CornellBox : DemoId::SponzaAtrium);
    imguiValue("WASD: move; right drag: look; click: pick");
    const char* algorithms[] = {"Ray Tracing", "Path Tracing"};
    imguiCombo("Algorithm", &Algorithm_, algorithms, 2);
    imguiSlider("Preview samples", &PreviewSamples_, 1, 16, 1);
    imguiSlider("Max bounces", &Bounces_, 1, 12, 1);
    editor.SetPreviewIntegrator(Algorithm_ == 1, static_cast<std::uint32_t>(PreviewSamples_), static_cast<std::uint32_t>(Bounces_));
    if (imguiCheck("Shadows", editor.Settings().ShadowsEnabled))
        editor.SetShadowsEnabled(!editor.Settings().ShadowsEnabled);
    imguiValue(editor.StatusText().c_str());
    if (editor.SelectedBackend() == "renderer.dx11") imguiValue("Use DXR for matching light transport");
    imguiSeparatorLine();
    const bool running = Preparation_.valid() || Client_.IsRunning();
    if (running)
    {
        if (imguiButton("Cancel render")) Cancel();
    }
    else if (imguiButton("Render PNG", !editor.IsDemoLoading())) Start(editor);
    if (Preparation_.valid()) imguiValue("Preparing scene...");
    const auto progress = Progress();
    if (progress.TotalTiles)
    {
        const std::string total = "Tiles: " + std::to_string(progress.CompletedTiles) + "/" + std::to_string(progress.TotalTiles);
        imguiValue(total.c_str());
        float percentage = 100.0f * progress.CompletedTiles / progress.TotalTiles;
        imguiSlider("Progress %", &percentage, 0, 100, 1, false);
    }
    if (progress.State == Distributed::JobState::Completed)
    {
        imguiLabel("Saved PNG");
        imguiValue(progress.OutputPath.filename().string().c_str());
    }
    if (progress.State == Distributed::JobState::Cancelled) imguiValue("Render cancelled");
    for (const auto& server : progress.Servers)
    {
        imguiSeparatorLine();
        imguiValue(server.Url.c_str());
        imguiValue((server.Status + " / tiles " + std::to_string(server.CompletedTiles)).c_str());
        imguiValue(("CPU " + Utilization(server.CpuPercent) + " / GPU " + Utilization(server.GpuPercent)).c_str());
    }
    if (!progress.Error.empty()) imguiValue(progress.Error.c_str());
    if (!Error_.empty()) imguiValue(Error_.c_str());
    imguiSeparatorLine();
    const char* backendLabels[] = {"CPU", "CUDA", "Compute Shader", "Metal", "Per server (auto)"};
    const char* resolutions[] = {"1280 x 720", "1920 x 1080", "3840 x 2160 (4K)"};
    imguiCombo("Worker backend", &Backend_, backendLabels, 5, !running);
    imguiCombo("Output resolution", &Resolution_, resolutions, 3, !running);
    imguiSlider("Offline samples", &Samples_, 1, 256, 1, !running);
    imguiSlider("Tile size", &TileSize_, 64, 512, 64, !running);
    imguiInputText("Output directory", OutputDirectory_, !running);
    imguiInputText("Access token (optional)", Token_, !running, true);
    for (std::size_t i = 0; i < Servers_.size(); ++i)
    {
        const auto label = "HTTP worker " + std::to_string(i + 1);
        imguiInputText(label.c_str(), Servers_[i], !running);
    }
    if (imguiButton("Add server", !running && Servers_.size() < 32)) Servers_.push_back("http://127.0.0.1:7833");
    if (imguiButton("Remove last server", !running && Servers_.size() > 1)) Servers_.pop_back();
    imguiEndScrollArea();
}
}
