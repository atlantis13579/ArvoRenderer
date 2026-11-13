#include "Core/Paths.h"
#include "Editor/RenderPanel.h"
#include "Editor/Imgui.h"
#include "Editor/ModuleAdapters.h"
#include "Distributed/Protocol.h"
#include "Renderer/MicropolygonData.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace Arvo::Editor
{
namespace
{
constexpr const char* BackendIds[] = {"cpu", "cuda", "compute", "metal", "dxr", "auto"};
constexpr RayTracing::BackendType WorkerBackends[] = {
    RayTracing::BackendType::CPU, RayTracing::BackendType::CUDA,
    RayTracing::BackendType::ComputeShader, RayTracing::BackendType::Metal, RayTracing::BackendType::DXR};
static_assert(std::size(BackendIds) == std::size(WorkerBackends) + 1);
RayTracing::TransportOptions TransportFor(RayTracing::Algorithm algorithm, RayTracing::TransportOptions options)
{
    if (algorithm == RayTracing::Algorithm::PointBasedGlobalIllumination)
    {
        options.NextEventEstimation = true;
        options.RussianRoulette = false;
        options.Mis = RayTracing::MisHeuristic::None;
        return options;
    }
    if (algorithm == RayTracing::Algorithm::PhotonMapping || algorithm == RayTracing::Algorithm::ProgressivePhotonMapping)
    {
        options.Mis = RayTracing::MisHeuristic::None;
        return options;
    }
    if (algorithm == RayTracing::Algorithm::BidirectionalPathTracing)
    {
        options.NextEventEstimation = true;
        options.RussianRoulette = false;
        return options;
    }
    if (algorithm == RayTracing::Algorithm::VertexConnectionMerging)
    {
        options.RussianRoulette = false;
        return options;
    }
    if (!options.NextEventEstimation) options.Mis = RayTracing::MisHeuristic::None;
    return algorithm == RayTracing::Algorithm::PathTracing || algorithm == RayTracing::Algorithm::MetropolisLightTransport
        ? options : RayTracing::TransportOptions{};
}
bool WorkerSupports(int backend, RayTracing::Algorithm algorithm)
{
    if (backend < static_cast<int>(std::size(WorkerBackends)))
        return RayTracing::SupportsIntegrator(WorkerBackends[backend], algorithm);
    // Auto is safe only for the intersection of worker capabilities until the
    // handshake advertises the actual backend and its feature set.
    for (const auto type : WorkerBackends)
        if (!RayTracing::SupportsIntegrator(type, algorithm)) return false;
    return true;
}
bool MaterialsSupported(const SceneSnapshot& scene, RayTracing::BackendType backend,
    RayTracing::Algorithm algorithm, std::string& error,
    std::vector<std::string>* issues = nullptr)
{
    bool supported = true;
    for (const auto& material : scene.Materials)
    {
        std::string reason;
        if (algorithm == RayTracing::Algorithm::PointBasedGlobalIllumination &&
            material.Scattering.Model == RayTracing::MaterialModel::Legacy &&
            (material.Specular.R != 0 || material.Specular.G != 0 || material.Specular.B != 0))
        {
            reason = "Point GI's view-independent surfel cache cannot represent legacy specular reflectance";
        }
        else RayTracing::ValidateMaterialSupport(material.Scattering, backend, algorithm, reason);
        if (reason.empty()) continue;
        const std::string issue = material.Name + ": " + reason;
        if (error.empty()) error = issue;
        if (issues) issues->push_back(issue);
        supported = false;
    }
    return supported;
}
bool WorkerMaterialsSupported(const SceneSnapshot& scene, int backend,
    RayTracing::Algorithm algorithm, std::string& error,
    std::vector<std::string>* issues = nullptr)
{
    error.clear();
    if (issues) issues->clear();
    if (backend < static_cast<int>(std::size(WorkerBackends)))
        return MaterialsSupported(scene, WorkerBackends[backend], algorithm, error, issues);
    bool supported = true;
    for (const auto type : WorkerBackends)
        supported &= MaterialsSupported(scene, type, algorithm, error, issues);
    if (issues)
    {
        std::sort(issues->begin(), issues->end());
        issues->erase(std::unique(issues->begin(), issues->end()), issues->end());
    }
    return supported;
}
constexpr std::uint32_t Widths[] = {1280, 1920, 3840};
constexpr std::uint32_t Heights[] = {720, 1080, 2160};
void DrawMessage(std::string_view message)
{
    // The fixed-width panel needs short lines for actionable capability errors.
    constexpr size_t MaximumLineCharacters = 40;
    while (!message.empty())
    {
        size_t length = std::min(message.size(), MaximumLineCharacters);
        if (length < message.size())
        {
            const auto space = message.rfind(' ', length);
            if (space != std::string_view::npos && space != 0) length = space;
        }
        imguiLabel(std::string(message.substr(0, length)).c_str());
        message.remove_prefix(length);
        while (!message.empty() && message.front() == ' ') message.remove_prefix(1);
    }
}
std::string Utilization(double value)
{
    if (value < 0.0) return "N/A";
    return std::to_string(static_cast<int>(value + 0.5)) + "%";
}
}
RenderPanel::RenderPanel()
{
#if defined(_WIN32)
    Backend_ = 1; // CUDA
#elif defined(__APPLE__)
    Backend_ = 3;
#endif
    OutputDirectory_ = (Arvo::Core::ContentRoot().parent_path() / "Output").string();
    if (const char* token = std::getenv("ARVO_WORKER_TOKEN")) Token_ = token;
}
RenderPanel::~RenderPanel() { Cancel(); }
void RenderPanel::StartLocalWorkers()
{
    if (!LocalWorkers_) LocalWorkers_ = std::make_shared<Distributed::LocalWorkers>(Core::ContentRoot(), Token_);
}
void RenderPanel::StopLocalWorkers()
{
    Cancel();
    if (Preparation_.valid()) Preparation_.wait();
    Client_.Wait();
    // Release the completed preparation's capture before stopping its workers.
    Preparation_ = {};
    LocalWorkers_.reset();
}
void RenderPanel::SetAcceleration(RayTracing::AccelerationType type)
{
    std::string error;
    if (!RayTracing::ValidateAcceleration(type, error)) throw std::invalid_argument(error);
    Acceleration_ = static_cast<int>(type);
}
void RenderPanel::SetBvhPolicy(RayTracing::BvhBuildPolicy policy)
{
    std::string error;
    if (!RayTracing::ValidateAcceleration({RayTracing::AccelerationType::Bvh, policy}, error))
        throw std::invalid_argument(error);
    BvhPolicy_ = static_cast<int>(policy);
}
void RenderPanel::SetAdaptiveSampling(RayTracing::AdaptiveOptions options, std::uint32_t previewLimit)
{
    std::string error;
    if (previewLimit == 0 || previewLimit > 64
        || !RayTracing::ValidateAdaptiveOptions(static_cast<RayTracing::Algorithm>(Algorithm_), options, previewLimit, error))
        throw std::invalid_argument(error.empty() ? "Preview sample limit must be in [1, 64]" : error);
    Adaptive_ = options;
    PreviewSamples_ = static_cast<float>(previewLimit);
    if (Adaptive_.Enabled) Samples_ = std::max(Samples_, static_cast<float>(Adaptive_.MinimumSamples));
}
void RenderPanel::SetGeometryMode(RayTracing::GeometryMode mode)
{
    if (!RayTracing::GeometryModeKey(mode)) throw std::invalid_argument("Unknown geometry mode");
    Geometry_ = static_cast<int>(mode);
}
void RenderPanel::Cancel()
{
    if (Preparation_.valid()) PreparationWasCancelled_ = true;
    if (PreparationCancelled_) PreparationCancelled_->store(true);
    Client_.Cancel();
}
void RenderPanel::SetTemporalRendering(Distributed::TemporalRenderOptions options)
{
    Distributed::ValidateTemporalRendering({}, options);
    Temporal_ = options;
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
        if (!PreparationCancelled_->load()) Client_.Start(std::move(config.Config), config.Temporal);
    }
    catch (const std::exception& e) { Error_ = e.what(); }
}
void RenderPanel::Start(EditorCore& editor)
{
    try
    {
        if (editor.IsDemoLoading() || Preparation_.valid() || Client_.IsRunning()) return;
        if (!WorkerSupports(Backend_, static_cast<RayTracing::Algorithm>(Algorithm_)))
        {
            Error_ = "The selected integrator is not implemented on this worker backend";
            return;
        }
        if (!WorkerMaterialsSupported(editor.Scene(), Backend_,
            static_cast<RayTracing::Algorithm>(Algorithm_), Error_)) return;
        PreparationWasCancelled_ = false;
        Distributed::RenderConfig config;
        config.Servers = Servers_;
        config.Token = Token_;
        config.Scene = DemoKey(editor.SelectedDemo());
        config.Backend = BackendIds[Backend_];
        config.Camera = BuildOfflineCamera(editor.Scene());
        config.Integrator = static_cast<RayTracing::Algorithm>(Algorithm_);
        config.Radiometry = config.Integrator == RayTracing::Algorithm::PathTracing
            ? static_cast<RayTracing::RadiometryMode>(Radiometry_) : RayTracing::RadiometryMode::RGB;
        config.Transport = TransportFor(config.Integrator, Transport_);
        config.Sampling = static_cast<RayTracing::SampleSequence>(Sampling_);
        config.SamplingSeed = SamplingSeed_;
        config.Adaptive = Adaptive_;
        config.Medium = Medium_;
        config.Micropolygons = Micropolygons_;
        if (config.Integrator != RayTracing::Algorithm::PathTracing) config.Medium.Enabled = false;
        config.Reconstruction = Reconstruction_;
        config.Metropolis = Metropolis_;
        config.Photons = Photons_;
        config.ProgressivePhotons = ProgressivePhotons_;
        config.Vcm = Vcm_;
        config.PointGi = PointGi_;
        const bool pointGi = config.Integrator == RayTracing::Algorithm::PointBasedGlobalIllumination;
        const bool vcm = config.Integrator == RayTracing::Algorithm::VertexConnectionMerging;
        const bool photon = config.Integrator == RayTracing::Algorithm::PhotonMapping ||
            config.Integrator == RayTracing::Algorithm::ProgressivePhotonMapping;
        const bool markov = config.Integrator == RayTracing::Algorithm::MetropolisLightTransport;
        if (config.Integrator == RayTracing::Algorithm::BidirectionalPathTracing || markov || photon || vcm || pointGi)
            config.Adaptive.Enabled = false;
        if (!config.Adaptive.Enabled) config.Adaptive.DenoiserAware = false;
        if (markov) config.Sampling = RayTracing::SampleSequence::Independent;
        config.Acceleration = static_cast<RayTracing::AccelerationType>(Acceleration_);
        config.Geometry = static_cast<RayTracing::GeometryMode>(Geometry_);
        uint32_t instanceIndex = 0;
        for (const auto& node : editor.Scene().Nodes)
        {
            if (!node.Visible || !node.Mesh) continue;
            if (node.Motion.Enabled) config.InstanceMotions.push_back({instanceIndex, node.Motion});
            ++instanceIndex;
        }
        for (uint32_t index = 0; index < editor.Scene().Materials.size(); ++index)
        {
            const auto& scattering = editor.Scene().Materials[index].Scattering;
            if (scattering != RayTracing::MaterialScattering{}) config.Materials.push_back({index, scattering});
        }
        config.BvhPolicy = config.Acceleration == RayTracing::AccelerationType::Bvh
            ? static_cast<RayTracing::BvhBuildPolicy>(BvhPolicy_) : RayTracing::BvhBuildPolicy::Default;
        if (!RayTracing::ValidateAcceleration({config.Acceleration, config.BvhPolicy}, Error_)) return;
        config.Width = Widths[Resolution_];
        config.Height = Heights[Resolution_];
        config.TileSize = static_cast<std::uint32_t>(TileSize_);
        config.SamplesPerPixel = static_cast<std::uint32_t>(Samples_);
        config.MaxBounces = static_cast<std::uint32_t>(Bounces_);
        config.ShadowsEnabled = config.Integrator == RayTracing::Algorithm::BidirectionalPathTracing ||
                                markov || photon || vcm || pointGi || editor.Settings().ShadowsEnabled;
        const auto stamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        config.OutputPath = std::filesystem::path(OutputDirectory_) /
            (config.Scene + "-" + RayTracing::FindIntegrator(config.Integrator)->Key + "-"
                + RayTracing::FindAcceleration(config.Acceleration)->Key + "-"
                + RayTracing::BvhBuildKey(config.BvhPolicy) + "-" + RayTracing::GeometryModeKey(config.Geometry)
                + "-" + std::to_string(stamp) + ".png");
        PreparationCancelled_ = std::make_shared<std::atomic<bool>>(false);
        Preparation_ = std::async(std::launch::async,
            [config = std::move(config), temporal = Temporal_, scene = editor.Scene(), cancelled = PreparationCancelled_,
             workers = LocalWorkers_]() mutable {
                if (workers) workers->EnsureReady(config.Servers, cancelled.get());
                if (!cancelled->load())
                    config.ExpectedSceneHash = Distributed::Protocol::SceneHash(*BuildOfflineScene(scene, config.Geometry));
                return PreparedRender{std::move(config), temporal};
            });
        Error_.clear();
    }
    catch (const std::exception& e) { Error_ = e.what(); }
}
void RenderPanel::DrawMaterials(EditorCore& editor)
{
    if (imguiCollapse("Surface materials", "", MaterialsOpen_)) MaterialsOpen_ = !MaterialsOpen_;
    if (!MaterialsOpen_ || editor.Scene().Materials.empty()) return;
    std::vector<const char*> names;
    for (const auto& material : editor.Scene().Materials) names.push_back(material.Name.c_str());
    Material_ = std::clamp(Material_, 0, static_cast<int>(names.size()) - 1);
    const bool editable = !editor.IsDemoLoading();
    imguiCombo("Material", &Material_, names.data(), static_cast<int>(names.size()), editable);
    auto value = editor.Scene().Materials[Material_].Scattering;
    const char* models[] = {"Authored (legacy)", "Diffuse", "Conductor", "Dielectric"};
    int model = static_cast<int>(value.Model);
    bool changed = imguiCombo("Scattering", &model, models, static_cast<int>(std::size(models)), editable);
    value.Model = static_cast<RayTracing::MaterialModel>(model);
    if (value.Model != RayTracing::MaterialModel::Dielectric && value.SubsurfaceMeanFreePath != 0)
    {
        value.SubsurfaceMeanFreePath = 0;
        changed = true;
    }
    const bool specular = value.Model == RayTracing::MaterialModel::Conductor
        || value.Model == RayTracing::MaterialModel::Dielectric;
    if (specular)
    {
        const bool ideal = value.MicrofacetSlope == 0;
        if (imguiCheck("Ideal surface", ideal, editable))
        {
            value.MicrofacetSlope = ideal ? RayTracing::MaterialScattering{}.MicrofacetSlope : 0;
            changed = true;
        }
        if (value.MicrofacetSlope != 0)
            changed |= imguiSlider("GGX slope", &value.MicrofacetSlope, RayTracing::MinimumMicrofacetSlope,
                                   1, RayTracing::MinimumMicrofacetSlope, editable);
        if (static_cast<RayTracing::Algorithm>(Algorithm_) == RayTracing::Algorithm::RayTracing)
            imguiValue("Whitted needs ideal specular surfaces");
    }
    constexpr float MinimumIndex = 0.001f, MaximumOpticalValue = 100, OpticalStep = 0.001f;
    if (value.Model == RayTracing::MaterialModel::Dielectric)
    {
        changed |= imguiSlider("Exterior IOR", &value.ExteriorIndex, MinimumIndex, MaximumOpticalValue, OpticalStep, editable);
        changed |= imguiSlider("Interior IOR", &value.InteriorIndex, MinimumIndex, MaximumOpticalValue, OpticalStep, editable);
        const bool volumeVcmWorker = Backend_ >= 0 &&
            Backend_ <= int(std::size(WorkerBackends)) &&
            WorkerSupports(Backend_, RayTracing::Algorithm::VertexConnectionMerging);
        const bool supportsSubsurface = Algorithm_ == int(RayTracing::Algorithm::PathTracing) ||
            Algorithm_ == int(RayTracing::Algorithm::DistributedRayTracing) ||
            Algorithm_ == int(RayTracing::Algorithm::BidirectionalPathTracing) ||
            Algorithm_ == int(RayTracing::Algorithm::MetropolisLightTransport) ||
            Algorithm_ == int(RayTracing::Algorithm::PhotonMapping) ||
            Algorithm_ == int(RayTracing::Algorithm::ProgressivePhotonMapping) ||
            (Algorithm_ == int(RayTracing::Algorithm::VertexConnectionMerging) && volumeVcmWorker) ||
            (Algorithm_ == int(RayTracing::Algorithm::PointBasedGlobalIllumination) &&
             Backend_ >= 0 && Backend_ <= int(std::size(WorkerBackends)) &&
             WorkerSupports(Backend_, RayTracing::Algorithm::PointBasedGlobalIllumination));
        const bool subsurface = value.SubsurfaceMeanFreePath > 0;
        if (imguiCheck("Subsurface random walk", subsurface, editable && (supportsSubsurface || subsurface)))
        {
            const auto camera = BuildOfflineCamera(editor.Scene());
            constexpr float DefaultRelativeMeanFreePath = 0.01f;
            const auto view = camera.Target - camera.Position;
            value.SubsurfaceMeanFreePath = subsurface ? 0 : std::clamp(
                std::sqrt(RayTracing::Dot(view, view)) * DefaultRelativeMeanFreePath,
                RayTracing::MinimumSubsurfaceMeanFreePath, RayTracing::MaximumSubsurfaceMeanFreePath);
            changed = true;
        }
        if (value.SubsurfaceMeanFreePath > 0)
        {
            constexpr float MeanFreePathStep = RayTracing::MinimumSubsurfaceMeanFreePath;
            constexpr float AlbedoStep = 0.001f, AnisotropyStep = 0.001f;
            const auto camera = BuildOfflineCamera(editor.Scene());
            const auto view = camera.Target - camera.Position;
            constexpr float RelativeMeanFreePathRange = .1f;
            const float maximumPath = std::clamp(std::max(value.SubsurfaceMeanFreePath,
                std::sqrt(RayTracing::Dot(view, view)) * RelativeMeanFreePathRange), MeanFreePathStep,
                RayTracing::MaximumSubsurfaceMeanFreePath);
            changed |= imguiSlider("Mean free path", &value.SubsurfaceMeanFreePath, MeanFreePathStep,
                maximumPath, MeanFreePathStep, editable);
            changed |= imguiSlider("Scattering albedo scale", &value.SubsurfaceAlbedoScale, 0, 1, AlbedoStep, editable);
            changed |= imguiSlider("Interior anisotropy", &value.SubsurfaceAnisotropy,
                -RayTracing::MaximumSubsurfaceAnisotropy, RayTracing::MaximumSubsurfaceAnisotropy,
                AnisotropyStep, editable);
            DrawMessage("World units; albedo uses authored material color");
            DrawMessage("Closed, outward solid; keep a gap to other geometry");
            DrawMessage("Static solids and full shadows required");
            DrawMessage("Interior scatter events consume the bounce budget");
            DrawMessage("DX11: thick-body single-scattering approximation");
        }
        if (!supportsSubsurface)
        {
            if (Algorithm_ == int(RayTracing::Algorithm::RayTracing))
                DrawMessage("Whitted recursion has no interior scattering events");
            else if (Algorithm_ == int(RayTracing::Algorithm::PointBasedGlobalIllumination))
                DrawMessage("Point GI subsurface transport is unavailable on this backend");
            else
                DrawMessage("This integrator/backend has no interior scattering implementation");
        }
    }
    if (value.Model == RayTracing::MaterialModel::Conductor)
    {
        changed |= imguiSlider("Index R", &value.ConductorIndex.X, MinimumIndex, MaximumOpticalValue, OpticalStep, editable);
        changed |= imguiSlider("Index G", &value.ConductorIndex.Y, MinimumIndex, MaximumOpticalValue, OpticalStep, editable);
        changed |= imguiSlider("Index B", &value.ConductorIndex.Z, MinimumIndex, MaximumOpticalValue, OpticalStep, editable);
        changed |= imguiSlider("Extinction R", &value.ConductorExtinction.X, 0, MaximumOpticalValue, OpticalStep, editable);
        changed |= imguiSlider("Extinction G", &value.ConductorExtinction.Y, 0, MaximumOpticalValue, OpticalStep, editable);
        changed |= imguiSlider("Extinction B", &value.ConductorExtinction.Z, 0, MaximumOpticalValue, OpticalStep, editable);
    }
    if (imguiButton("Restore authored material", editable)) { value = {}; changed = true; }
    if (changed)
    {
        try { editor.SetMaterialScattering(static_cast<size_t>(Material_), value); Error_.clear(); }
        catch (const std::exception& error) { Error_ = error.what(); }
    }
}

void RenderPanel::DrawCamera(EditorCore& editor)
{
    if (imguiCollapse("Camera", "", CameraOpen_))
        CameraOpen_ = !CameraOpen_;
    if (!CameraOpen_) return;
    const auto camera = BuildOfflineCamera(editor.Scene());
    auto lens = camera.Lens;
    const auto view = camera.Target - camera.Position;
    const float targetDistance = std::max(RayTracing::MinimumFocusDistance, std::sqrt(RayTracing::Dot(view, view)));
    const bool editable = !editor.IsDemoLoading();
    bool changed = false;
    constexpr float DefaultRelativeRadius = 0.02f, MaximumRelativeRadius = 0.25f;
    constexpr float FocusRangeScale = 4, SliderResolution = 10000;
    if (imguiCheck("Depth of field", lens.Radius > 0, editable))
    {
        if (lens.Radius > 0) lens.Radius = 0;
        else { lens.FocusDistance = targetDistance; lens.Radius = targetDistance * DefaultRelativeRadius; }
        changed = true;
    }
    if (lens.Radius > 0)
    {
        const float step = std::max(RayTracing::MinimumFocusDistance, targetDistance / SliderResolution);
        changed |= imguiSlider("Lens radius", &lens.Radius, step,
            std::max(lens.Radius, targetDistance * MaximumRelativeRadius), step, editable);
        changed |= imguiSlider("Focus distance", &lens.FocusDistance, RayTracing::MinimumFocusDistance,
            std::max(lens.FocusDistance, targetDistance * FocusRangeScale), step, editable);
        if (imguiButton("Focus on camera target", editable)) { lens.FocusDistance = targetDistance; changed = true; }
        imguiValue("World units; aperture-average exposure");
        imguiValue("More camera samples resolve lens blur");
    }
    if (changed)
    {
        // Slider rounding can undershoot a strictly positive focus endpoint.
        lens.FocusDistance = std::max(lens.FocusDistance, RayTracing::MinimumFocusDistance);
        try { editor.SetCameraLens(lens); Error_.clear(); }
        catch (const std::exception& error) { Error_ = error.what(); }
    }
    auto exposure = camera.Motion;
    bool motionChanged = false;
    const bool motionAvailable = RayTracing::SupportsMotionBlur(static_cast<RayTracing::Algorithm>(Algorithm_));
    if (imguiCheck("Camera motion blur", exposure.Enabled != 0, editable && motionAvailable))
    {
        const bool initialize = exposure == RayTracing::MotionData::Exposure{};
        exposure.Enabled = !exposure.Enabled;
        if (exposure.Enabled && initialize)
        {
            exposure.Interval = {0, 1};
            exposure.Motion.End.Translation.X = targetDistance * DefaultRelativeRadius;
        }
        motionChanged = true;
    }
    if (!motionAvailable)
        DrawMessage("Camera exposure requires Path or Distributed Ray Tracing");
    if (exposure.Enabled || HasObjectMotion(editor.Scene()))
    {
        constexpr float ShutterStep = 0.001f, DegreesToHalfRadians = 3.14159265358979323846f / 360;
        constexpr float MaximumYaw = 180, AngularStep = 0.1f;
        motionChanged |= imguiSlider("Shutter open", &exposure.Interval.Open, 0, exposure.Interval.Close,
                                     ShutterStep, editable);
        motionChanged |= imguiSlider("Shutter close", &exposure.Interval.Close, exposure.Interval.Open, 1,
                                     ShutterStep, editable);
        if (exposure.Enabled)
        {
            const float range = std::max(1.0f, targetDistance), step = range / SliderResolution;
            motionChanged |=
                imguiSlider("End shift X", &exposure.Motion.End.Translation.X, -range, range, step, editable);
            motionChanged |=
                imguiSlider("End shift Y", &exposure.Motion.End.Translation.Y, -range, range, step, editable);
            motionChanged |=
                imguiSlider("End shift Z", &exposure.Motion.End.Translation.Z, -range, range, step, editable);
            float yaw =
                std::atan2(exposure.Motion.End.Rotation.Y, exposure.Motion.End.Rotation.W) / DegreesToHalfRadians;
            if (imguiSlider("End yaw (degrees)", &yaw, -MaximumYaw, MaximumYaw, AngularStep, editable))
            {
                exposure.Motion.End.Rotation = {0, std::sin(yaw * DegreesToHalfRadians), 0,
                                                std::cos(yaw * DegreesToHalfRadians)};
                motionChanged = true;
            }
        }
        DrawMessage("Normalized exposure; camera shifts use world units");
        DrawMessage("Offline: sampled exposure; previews: midpoint pose");
        DrawMessage("The shutter also controls enabled object motion");
    }
    if (motionChanged)
    {
        try
        {
            editor.SetCameraMotion(exposure);
            Error_.clear();
        }
        catch (const std::exception& error)
        {
            Error_ = error.what();
        }
    }
}

void RenderPanel::Draw(EditorCore& editor, int height)
{
    PollPreparation();
    const bool running = Preparation_.valid() || Client_.IsRunning();
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
    const auto& demos = DemoCatalog::Descriptors();
    std::vector<const char*> scenes;
    int scene = 0;
    for (size_t index = 0; index < demos.size(); ++index)
    {
        scenes.push_back(demos[index].Name.data());
        if (demos[index].Id == editor.SelectedDemo()) scene = static_cast<int>(index);
    }
    if (imguiCombo("Scene", &scene, scenes.data(), static_cast<int>(scenes.size())))
        editor.SelectDemo(demos.at(scene).Id);
    DrawMaterials(editor);
    DrawCamera(editor);
    DrawObjectMotion(editor);
    std::vector<const char*> algorithms;
    for (uint32_t index = 0; index < static_cast<uint32_t>(RayTracing::Algorithm::Count); ++index)
        algorithms.push_back(RayTracing::FindIntegrator(static_cast<RayTracing::Algorithm>(index))->Label);
    if (imguiCombo("Integrator", &Algorithm_, algorithms.data(), static_cast<int>(algorithms.size()),
                   !editor.IsDemoLoading()) &&
        !RayTracing::SupportsMotionBlur(static_cast<RayTracing::Algorithm>(Algorithm_)))
    {
        if (editor.Scene().SceneCamera.Motion.Enabled)
        {
            auto exposure = editor.Scene().SceneCamera.Motion;
            exposure.Enabled = 0;
            editor.SetCameraMotion(exposure);
        }
        editor.DisableObjectMotion();
    }
    const auto algorithm = static_cast<RayTracing::Algorithm>(Algorithm_);
    const bool spectralSupported = algorithm == RayTracing::Algorithm::PathTracing;
    const char* radiometryLabels[]{"RGB", "Spectral"};
    int displayedRadiometry = spectralSupported ? Radiometry_ : 0;
    if (imguiCombo("Radiometry", &displayedRadiometry, radiometryLabels, 2, spectralSupported))
        Radiometry_ = displayedRadiometry;
    if (!spectralSupported) imguiValue("Spectral transport requires Path Tracing");
    const bool bidirectional = algorithm == RayTracing::Algorithm::BidirectionalPathTracing;
    const bool markov = algorithm == RayTracing::Algorithm::MetropolisLightTransport;
    const bool progressive = algorithm == RayTracing::Algorithm::ProgressivePhotonMapping;
    const bool photon = algorithm == RayTracing::Algorithm::PhotonMapping || progressive;
    const bool vcm = algorithm == RayTracing::Algorithm::VertexConnectionMerging;
    const bool pointGi = algorithm == RayTracing::Algorithm::PointBasedGlobalIllumination;
    const bool fullVisibility = bidirectional || markov || photon || vcm || pointGi;
    std::vector<const char*> samplerLabels;
    for (uint32_t index = 0; index < static_cast<uint32_t>(RayTracing::SampleSequence::Count); ++index)
        samplerLabels.push_back(RayTracing::FindSampler(static_cast<RayTracing::SampleSequence>(index))->Label);
    int displayedSampler = markov ? static_cast<int>(RayTracing::SampleSequence::Independent) : Sampling_;
    if (imguiCombo("Sampler", &displayedSampler, samplerLabels.data(), static_cast<int>(samplerLabels.size()),
        !markov && RayTracing::SupportsIntegrator(RayTracing::BackendType::CPU, algorithm))) Sampling_ = displayedSampler;
    constexpr float MaximumSamplingSeed = 65535;
    float samplingSeed = float(SamplingSeed_);
    if (imguiSlider("Sampling seed", &samplingSeed, 0, MaximumSamplingSeed, 1))
        SamplingSeed_ = uint32_t(samplingSeed);
    if (markov)
    {
        imguiValue("MLT: primary-space Markov mutations");
        imguiValue("Samples = average mutations per pixel");
        float bootstrap = float(Metropolis_.BootstrapSamples), chains = float(Metropolis_.ChainCount);
        if (imguiSlider("Bootstrap paths", &bootstrap, 64, 65536, 64)) Metropolis_.BootstrapSamples = uint32_t(bootstrap);
        if (imguiSlider("Markov chains", &chains, 1, 128, 1)) Metropolis_.ChainCount = uint32_t(chains);
        float largePercent = Metropolis_.LargeStepProbability * 100;
        if (imguiSlider("Large steps %", &largePercent, 1, 100, 1)) Metropolis_.LargeStepProbability = largePercent / 100;
        imguiSlider("Mutation scale", &Metropolis_.MutationScale, 0.0001f, 0.5f, 0.0001f);
        float seed = float(Metropolis_.Seed);
        if (imguiSlider("Chain seed", &seed, 0, 65535, 1)) Metropolis_.Seed = uint32_t(seed);
        imguiValue("Correlated passes: adaptive stopping off");
    }
    if (photon && !progressive)
    {
        imguiValue("Photon mapping: fixed global photon map");
        float paths = float(Photons_.PhotonPaths), seed = float(Photons_.Seed);
        constexpr float MinimumPhotonPaths = 64, MaximumPhotonPaths = 1048576, PhotonBudgetStep = 64;
        constexpr float MinimumRadiusScale = 0.0001f, MaximumRadiusScale = 0.5f;
        constexpr float MaximumGuiSeed = 65535;
        if (imguiSlider("Photon paths", &paths, MinimumPhotonPaths, MaximumPhotonPaths, PhotonBudgetStep))
            Photons_.PhotonPaths = uint32_t(paths);
        imguiSlider("Photon radius / scene", &Photons_.RadiusScale, MinimumRadiusScale, MaximumRadiusScale, MinimumRadiusScale);
        if (imguiSlider("Photon seed", &seed, 0, MaximumGuiSeed, 1)) Photons_.Seed = uint32_t(seed);
        if (imguiCheck("Caustic photons", Photons_.Caustics)) Photons_.Caustics = !Photons_.Caustics;
        if (imguiCheck("Indirect photons", Photons_.Indirect)) Photons_.Indirect = !Photons_.Indirect;
        DrawMessage("Samples refine camera rays; photons refine lighting");
        DrawMessage("Finite radius is biased; adaptive stopping off");
    }
    if (progressive)
    {
        DrawMessage("PPM: samples = progressive camera/photon passes");
        float paths = float(ProgressivePhotons_.PathsPerPass), seed = float(ProgressivePhotons_.Seed);
        constexpr float MinimumPaths = 64, MaximumPaths = 1048576, BudgetStep = 64, MaximumSeed = 65535;
        constexpr float MinimumRadius = 0.0001f, MaximumRadius = 0.5f, MinimumAlpha = 0.01f, MaximumAlpha = 0.99f;
        if (imguiSlider("PPM photons / pass", &paths, MinimumPaths, MaximumPaths, BudgetStep))
            ProgressivePhotons_.PathsPerPass = uint32_t(paths);
        imguiSlider("PPM initial radius / scene", &ProgressivePhotons_.InitialRadiusScale, MinimumRadius, MaximumRadius, MinimumRadius);
        imguiSlider("PPM alpha", &ProgressivePhotons_.Alpha, MinimumAlpha, MaximumAlpha, MinimumAlpha);
        if (imguiSlider("PPM seed", &seed, 0, MaximumSeed, 1)) ProgressivePhotons_.Seed = uint32_t(seed);
        if (imguiCheck("PPM caustics", ProgressivePhotons_.Caustics)) ProgressivePhotons_.Caustics = !ProgressivePhotons_.Caustics;
        if (imguiCheck("PPM indirect", ProgressivePhotons_.Indirect)) ProgressivePhotons_.Indirect = !ProgressivePhotons_.Indirect;
        DrawMessage("Radius shrinks; flux persists across passes");
        DrawMessage("Shared density state: adaptive stopping off");
    }
    if (vcm)
    {
        imguiValue("VCM: samples = connection/merge passes");
        float paths = float(Vcm_.LightPathsPerPass), seed = float(Vcm_.Seed);
        constexpr float MinimumPaths = 64, MaximumPaths = 1048576, BudgetStep = 64, MaximumSeed = 65535;
        constexpr float MinimumRadius = 0.0001f, MaximumRadius = 0.5f, MinimumAlpha = 0.01f,
                        MaximumAlpha = 0.99f;
        if (imguiSlider("VCM light paths / pass", &paths, MinimumPaths, MaximumPaths, BudgetStep))
            Vcm_.LightPathsPerPass = uint32_t(paths);
        imguiSlider("VCM initial radius / scene", &Vcm_.InitialRadiusScale, MinimumRadius, MaximumRadius,
                    MinimumRadius);
        imguiSlider("VCM alpha", &Vcm_.Alpha, MinimumAlpha, MaximumAlpha, MinimumAlpha);
        if (imguiSlider("VCM seed", &seed, 0, MaximumSeed, 1))
            Vcm_.Seed = uint32_t(seed);
        DrawMessage("Shrinking radius; average complete estimates");
        imguiValue("Fixed depth and full visibility required");
        DrawMessage("Shared light paths: adaptive stopping off");
    }
    if (pointGi)
    {
        imguiValue("Point GI: surface cache and subsurface walks");
        if (imguiButton("Use diffuse scene materials", !editor.IsDemoLoading()))
            editor.SetSceneMaterialModel(RayTracing::MaterialModel::Diffuse);
        constexpr float MinimumArea = 0.0001f, MaximumArea = 100, AreaStep = 0.01f;
        constexpr float MinimumBudget = 64, MaximumBudget = 1048576, BudgetStep = 64;
        constexpr float MaximumDirectSamples = 256, MaximumGatherSamples = 512, MaximumSeed = 65535;
        imguiSlider("Point GI patch area", &PointGi_.MaximumArea, MinimumArea, MaximumArea, AreaStep);
        float budget = float(PointGi_.MaximumSurfels), direct = float(PointGi_.DirectSamples),
              gather = float(PointGi_.GatherSamples), seed = float(PointGi_.Seed);
        if (imguiSlider("Point GI surface budget", &budget, MinimumBudget, MaximumBudget, BudgetStep))
            PointGi_.MaximumSurfels = uint32_t(budget);
        if (imguiSlider("Point GI direct bake samples", &direct, 1, MaximumDirectSamples, 1))
            PointGi_.DirectSamples = uint32_t(direct);
        if (imguiSlider("Point GI gather bake samples", &gather, 1, MaximumGatherSamples, 1))
            PointGi_.GatherSamples = uint32_t(gather);
        if (imguiSlider("Point GI seed", &seed, 0, MaximumSeed, 1)) PointGi_.Seed = uint32_t(seed);
        imguiValue("Patch area uses squared world units");
        DrawMessage("Bake samples refine lighting; pixel samples refine the camera");
        DrawMessage("Diffuse materials, source sampling and full visibility required");
        DrawMessage("Shared biased cache: adaptive stopping off");
    }
    const auto previewBackend = backends.empty() ? std::optional<RayTracing::BackendType>{}
        : backends[selected].TransportBackend;
    const bool implemented = previewBackend && RayTracing::SupportsIntegrator(*previewBackend, algorithm);
    if (!previewBackend && spectralSupported && Radiometry_ != 0)
        DrawMessage("Spectral offline rendering selected; raster preview remains RGB");
    const char* geometryLabels[] = {"Flat geometry", "Shared instances"};
    const bool objectMotion = HasObjectMotion(editor.Scene());
    if (objectMotion) Geometry_ = int(RayTracing::GeometryMode::Instanced);
    imguiCombo("Geometry", &Geometry_, geometryLabels, static_cast<int>(std::size(geometryLabels)), !objectMotion);
    if (objectMotion) DrawMessage("Object motion uses shared instance geometry");
    const auto geometry = static_cast<RayTracing::GeometryMode>(Geometry_);
    if (geometry == RayTracing::GeometryMode::Instanced)
        imguiValue("Instances share object geometry");
    const char* accelerationLabels[] = {"BVH", "k-d tree", "BIH"};
    imguiCombo("Acceleration", &Acceleration_, accelerationLabels, 3);
    const auto acceleration = static_cast<RayTracing::AccelerationType>(Acceleration_);
    const bool isBvh = acceleration == RayTracing::AccelerationType::Bvh;
    const char* bvhBuildLabels[] = {"Default", "Spatial splits"};
    imguiCombo("BVH build", &BvhPolicy_, bvhBuildLabels, 2, isBvh);
    const auto bvhPolicy = isBvh ? static_cast<RayTracing::BvhBuildPolicy>(BvhPolicy_)
        : RayTracing::BvhBuildPolicy::Default;
    if (!isBvh) imguiValue("Spatial splits apply to BVH only");
    const auto* accelerationInfo = RayTracing::FindAcceleration(acceleration);
    const bool accelerationReady = accelerationInfo && accelerationInfo->Implemented;
    if (!accelerationReady) imguiValue("Preview unchanged: accelerator pending");
    else if (implemented && previewBackend == RayTracing::BackendType::DXR)
        imguiValue("Preview always uses hardware BVH");
    if (previewBackend && !implemented) DrawMessage("Preview unchanged: implementation pending");
    const bool adaptiveSupported = !fullVisibility && RayTracing::SupportsIntegrator(RayTracing::BackendType::CPU, algorithm);
    if (imguiCheck("Adaptive sampling", Adaptive_.Enabled && adaptiveSupported, adaptiveSupported))
    {
        Adaptive_.Enabled = !Adaptive_.Enabled;
        if (Adaptive_.Enabled)
        {
            const float suggestedLimit = static_cast<float>(2 * (Adaptive_.MinimumSamples + Adaptive_.PilotSamples));
            PreviewSamples_ = std::max(PreviewSamples_, std::min(64.0f, suggestedLimit));
            Samples_ = std::max(Samples_, std::min(256.0f, suggestedLimit));
        }
    }
    if (Adaptive_.Enabled && adaptiveSupported)
    {
        float minimum = static_cast<float>(Adaptive_.MinimumSamples);
        if (imguiSlider("Minimum samples", &minimum, 2, 64, 1))
        {
            Adaptive_.MinimumSamples = static_cast<uint32_t>(minimum);
            PreviewSamples_ = std::max(PreviewSamples_, minimum);
            Samples_ = std::max(Samples_, minimum);
        }
        float pilot = static_cast<float>(Adaptive_.PilotSamples);
        if (imguiSlider("Pilot samples", &pilot, 2, 64, 1)) Adaptive_.PilotSamples = static_cast<uint32_t>(pilot);
        float relativePercent = Adaptive_.RelativeError * 100.0f;
        if (imguiSlider("Relative error %", &relativePercent, 0, 100, 0.5f))
            Adaptive_.RelativeError = relativePercent / 100.0f;
        constexpr float MinimumAbsoluteTarget = 0.0001f, MaximumAbsoluteTarget = 0.02f;
        imguiSlider("Absolute error", &Adaptive_.AbsoluteError, MinimumAbsoluteTarget,
            MaximumAbsoluteTarget, MinimumAbsoluteTarget);
        imguiValue("Pilot is extra; final samples <= limit");
        DrawMessage("RMS target is an estimate, not a guarantee");
    }
    const char* reconstructionLabels[] = {"Off", "Spatial a-trous"};
    int reconstruction = static_cast<int>(Reconstruction_.Filter);
    if (imguiCombo("Denoising", &reconstruction, reconstructionLabels, int(std::size(reconstructionLabels))))
        Reconstruction_.Filter = static_cast<RayTracing::ReconstructionFilter>(reconstruction);
    if (Reconstruction_.Filter == RayTracing::ReconstructionFilter::SpatialAtrous)
    {
        constexpr float MinimumPasses = 1, MaximumPasses = 5, PassStep = 1;
        constexpr float MaximumNoiseScale = 16, NoiseStep = 0.25f;
        constexpr float MaximumRelativeScale = 1, RelativeStep = 0.01f;
        constexpr float MinimumAbsoluteScale = 0.0001f, MaximumAbsoluteScale = 1, AbsoluteStep = 0.001f;
        float passes = float(Reconstruction_.Passes);
        if (imguiSlider("Filter passes", &passes, MinimumPasses, MaximumPasses, PassStep))
            Reconstruction_.Passes = uint32_t(passes);
        imguiSlider("Noise tolerance", &Reconstruction_.NoiseScale, 0, MaximumNoiseScale, NoiseStep,
                    RayTracing::ReconstructionUsesMoments(algorithm));
        imguiSlider("Relative color tolerance", &Reconstruction_.RelativeColorScale, 0, MaximumRelativeScale,
                    RelativeStep);
        imguiSlider("Absolute color tolerance", &Reconstruction_.AbsoluteColorScale, MinimumAbsoluteScale,
                    MaximumAbsoluteScale, AbsoluteStep);
        const auto &view = editor.Scene().SceneCamera;
        const bool geometryAvailable = view.Lens.Radius == 0 && !view.Motion.Enabled &&
                                       !HasObjectMotion(editor.Scene()) && !Medium_.Enabled;
        if (!geometryAvailable)
            Reconstruction_.GeometryGuidance = false;
        if (imguiCheck("Geometry guidance", Reconstruction_.GeometryGuidance, geometryAvailable))
            Reconstruction_.GeometryGuidance = !Reconstruction_.GeometryGuidance;
        if (Reconstruction_.GeometryGuidance)
        {
            constexpr float MinimumNormalScale = 0.01f, MaximumNormalScale = 2, NormalStep = 0.01f;
            constexpr float Percent = 100, MinimumPlanePercent = 0.01f, PlaneStep = 0.1f;
            imguiSlider("Normal tolerance", &Reconstruction_.NormalScale, MinimumNormalScale,
                        MaximumNormalScale, NormalStep);
            float plane = Reconstruction_.RelativePlaneScale * Percent;
            if (imguiSlider("Plane tolerance %", &plane, MinimumPlanePercent, Percent, PlaneStep))
                Reconstruction_.RelativePlaneScale = plane / Percent;
            DrawMessage("Normal, surface plane and material edges guide smoothing");
        }
        else if (!geometryAvailable)
            DrawMessage("Geometry guidance requires depth of field, motion and participating media off");
        DrawMessage("Reconstruction retains raw HDR and sample moments");
        if (!RayTracing::ReconstructionUsesMoments(algorithm))
            DrawMessage("Variance guidance is unavailable for this integrator");
        if (!previewBackend)
            DrawMessage("Offline denoising; DXR preview averages stationary frames");
    }
    const bool denoisedAllocation = Adaptive_.Enabled && adaptiveSupported &&
        Reconstruction_.Filter == RayTracing::ReconstructionFilter::SpatialAtrous;
    if (!denoisedAllocation) Adaptive_.DenoiserAware = false;
    if (imguiCheck("Denoiser-aware sampling", Adaptive_.DenoiserAware, denoisedAllocation))
        Adaptive_.DenoiserAware = !Adaptive_.DenoiserAware;
    if (Adaptive_.DenoiserAware)
    {
        constexpr float Percent = 100;
        float retained = Adaptive_.MinimumVarianceFraction * Percent;
        if (imguiSlider("Minimum variance retained %", &retained, 0, Percent, 1))
            Adaptive_.MinimumVarianceFraction = retained / Percent;
        DrawMessage("Two independent pilots; pilot samples apply to each");
        DrawMessage("Filtered error guides the fresh production budget");
    }
    if (imguiCheck("Micropolygon dicing", Micropolygons_.Enabled))
        Micropolygons_.Enabled = !Micropolygons_.Enabled;
    if (Micropolygons_.Enabled)
    {
        constexpr float MaximumPixelTarget = 128, PixelTargetStep = 1;
        constexpr uint32_t BudgetUnit = 1024;
        imguiSlider("Target edge pixels", &Micropolygons_.TargetEdgePixels, 1, MaximumPixelTarget,
                    PixelTargetStep);
        float level = float(Micropolygons_.MaximumLevel);
        if (imguiSlider("Maximum dicing level", &level, 0, float(RayTracing::MicropolygonData::MaximumLevel),
                        1))
            Micropolygons_.MaximumLevel = uint32_t(level);
        float budget = float(Micropolygons_.TriangleBudget) / BudgetUnit;
        if (imguiSlider("Triangle budget (Ki)", &budget, 1,
                        float(RayTracing::MicropolygonData::MaximumOutputTriangles / BudgetUnit), 1))
            Micropolygons_.TriangleBudget = uint32_t(budget) * BudgetUnit;
        std::vector<const char*> materials{"No displacement"};
        for (const auto& material : editor.Scene().Materials)
            materials.push_back(material.Name.c_str());
        const bool hasDisplacement = Micropolygons_.DisplacementMaterial < editor.Scene().Materials.size();
        int displacement = hasDisplacement ? int(Micropolygons_.DisplacementMaterial) + 1 : 0;
        imguiCombo("Displacement component", &displacement, materials.data(), int(materials.size()));
        Micropolygons_.DisplacementMaterial =
            displacement ? uint32_t(displacement - 1) : std::numeric_limits<uint32_t>::max();
        if (displacement)
        {
            const auto view = BuildOfflineCamera(editor.Scene());
            const auto direction = view.Target - view.Position;
            constexpr float MinimumWaveLength = .001f, RelativeAmplitude = .05f, RelativeWaveLength = .5f,
                            SliderSteps = 1000;
            const float scale = std::max(MinimumWaveLength, std::sqrt(RayTracing::Dot(direction, direction)));
            const float amplitudeLimit =
                std::max(std::abs(Micropolygons_.Amplitude), scale * RelativeAmplitude);
            imguiSlider("Wave amplitude", &Micropolygons_.Amplitude, -amplitudeLimit, amplitudeLimit,
                        scale / SliderSteps);
            imguiSlider("Wave wavelength", &Micropolygons_.Wavelength, MinimumWaveLength,
                        std::max(Micropolygons_.Wavelength, scale * RelativeWaveLength), MinimumWaveLength);
            imguiSlider("Wave phase (cycles)", &Micropolygons_.PhaseCycles, 0, 1, 1 / SliderSteps);
            DrawMessage("World-space Y wave; includes connected material seams. Analytic light proxies "
                        "cannot be displaced.");
        }
        DrawMessage(
            "Dicing level and triangle budget limit geometric detail and memory use.");
    }
    try
    {
        editor.SetMicropolygons(Micropolygons_);
    }
    catch (const std::exception& error)
    {
        Error_ = error.what();
    }
    const bool mediumSupported = algorithm == RayTracing::Algorithm::PathTracing;
    if (!mediumSupported) Medium_.Enabled = false;
    if (imguiCheck("Participating medium", Medium_.Enabled, mediumSupported)) Medium_.Enabled = !Medium_.Enabled;
    if (Medium_.Enabled)
    {
        constexpr float MaximumCoefficient = 2, CoefficientStep = 0.01f, MaximumAnisotropy = 0.99f;
        imguiSlider("Absorption R", &Medium_.Absorption.X, 0, MaximumCoefficient, CoefficientStep);
        imguiSlider("Absorption G", &Medium_.Absorption.Y, 0, MaximumCoefficient, CoefficientStep);
        imguiSlider("Absorption B", &Medium_.Absorption.Z, 0, MaximumCoefficient, CoefficientStep);
        imguiSlider("Scattering R", &Medium_.Scattering.X, 0, MaximumCoefficient, CoefficientStep);
        imguiSlider("Scattering G", &Medium_.Scattering.Y, 0, MaximumCoefficient, CoefficientStep);
        imguiSlider("Scattering B", &Medium_.Scattering.Z, 0, MaximumCoefficient, CoefficientStep);
        imguiSlider("Phase anisotropy", &Medium_.Anisotropy, -MaximumAnisotropy, MaximumAnisotropy, CoefficientStep);
        DrawMessage("Coefficients: inverse world units; positive g scatters forward");
        if (imguiCheck("Medium bounds", MediumBoundsOpen_)) MediumBoundsOpen_ = !MediumBoundsOpen_;
        if (MediumBoundsOpen_)
        {
            constexpr float WorldLimit = 1000, MinimumExtent = 0.01f, PositionStep = 0.01f;
            imguiSlider("Medium minimum X", &Medium_.Minimum.X, -WorldLimit, Medium_.Maximum.X - MinimumExtent, PositionStep);
            imguiSlider("Medium minimum Y", &Medium_.Minimum.Y, -WorldLimit, Medium_.Maximum.Y - MinimumExtent, PositionStep);
            imguiSlider("Medium minimum Z", &Medium_.Minimum.Z, -WorldLimit, Medium_.Maximum.Z - MinimumExtent, PositionStep);
            imguiSlider("Medium maximum X", &Medium_.Maximum.X, Medium_.Minimum.X + MinimumExtent, WorldLimit, PositionStep);
            imguiSlider("Medium maximum Y", &Medium_.Maximum.Y, Medium_.Minimum.Y + MinimumExtent, WorldLimit, PositionStep);
            imguiSlider("Medium maximum Z", &Medium_.Maximum.Z, Medium_.Minimum.Z + MinimumExtent, WorldLimit, PositionStep);
        }
        if (!previewBackend) DrawMessage("DX11: single scattering; DXR/offline: multiple scattering");
    }
    else if (!mediumSupported) DrawMessage("Participating media currently require path tracing");
    const bool temporalAvailable = previewBackend && previewBackend != RayTracing::BackendType::DXR && !Medium_.Enabled &&
                                   editor.Scene().SceneCamera.Lens.Radius == 0 &&
                                   !editor.Scene().SceneCamera.Motion.Enabled && !HasObjectMotion(editor.Scene()) &&
                                   Reconstruction_.Filter == RayTracing::ReconstructionFilter::None;
    bool temporal = editor.Settings().TemporalPreview && temporalAvailable;
    auto temporalOptions = editor.Settings().Temporal;
    if (temporalAvailable && imguiCheck("Temporal preview", temporal)) temporal = !temporal;
    if (temporal)
    {
        constexpr float MinimumHistory = 1, MaximumHistory = 256, PercentScale = 100;
        constexpr float MaximumRelativeDepth = 25, MaximumDepthTolerance = 1, ToleranceStep = 0.001f;
        constexpr float UnitStep = 0.01f, RelativeDepthStep = 0.1f, MinimumHistorySupport = 0.01f;
        float history = float(temporalOptions.MaximumHistory);
        if (imguiSlider("History frames", &history, MinimumHistory, MaximumHistory, 1))
            temporalOptions.MaximumHistory = uint32_t(history);
        float fresh = temporalOptions.MinimumCurrentWeight * PercentScale;
        if (imguiSlider("Minimum fresh %", &fresh, 0, PercentScale, 1))
            temporalOptions.MinimumCurrentWeight = fresh / PercentScale;
        imguiSlider("Depth tolerance", &temporalOptions.AbsoluteDepthTolerance, 0,
                    MaximumDepthTolerance, ToleranceStep);
        float relative = temporalOptions.RelativeDepthTolerance * PercentScale;
        if (imguiSlider("Relative depth %", &relative, 0, MaximumRelativeDepth, RelativeDepthStep))
            temporalOptions.RelativeDepthTolerance = relative / PercentScale;
        imguiSlider("Normal agreement", &temporalOptions.MinimumNormalCosine, 0, 1, UnitStep);
        imguiSlider("History support", &temporalOptions.MinimumHistoryWeight, MinimumHistorySupport, 1, UnitStep);
        imguiSlider("Color clip expansion", &temporalOptions.NeighborhoodExpansion, 0, 1, UnitStep);
        DrawMessage("Continuous fresh frames; history follows accepted images");
    }
    editor.SetTemporalPreview(temporal, temporalOptions);
    const bool offlineTemporalAvailable = !Medium_.Enabled && editor.Scene().SceneCamera.Lens.Radius == 0 &&
                                          !editor.Scene().SceneCamera.Motion.Enabled && !HasObjectMotion(editor.Scene()) &&
                                          Reconstruction_.Filter == RayTracing::ReconstructionFilter::None;
    if (!offlineTemporalAvailable) Temporal_.Enabled = false;
    const bool adaptiveActive = Adaptive_.Enabled && adaptiveSupported;
    const float minimumSamples = adaptiveActive ? static_cast<float>(Adaptive_.MinimumSamples) : 1.0f;
    imguiSlider("Max bounces", &Bounces_, 1, 12, 1);
    const bool path = algorithm == RayTracing::Algorithm::PathTracing || markov || photon || vcm;
    if (bidirectional) imguiValue("BDPT: fixed passes, full visibility");
    if (imguiCheck("Importance sampling", Transport_.ImportanceSampling, path || bidirectional || pointGi))
        Transport_.ImportanceSampling = !Transport_.ImportanceSampling;
    if (imguiCheck("Area-light NEE", bidirectional || pointGi || Transport_.NextEventEstimation, path))
    {
        Transport_.NextEventEstimation = !Transport_.NextEventEstimation;
        if (!Transport_.NextEventEstimation && !vcm)
            Transport_.Mis = RayTracing::MisHeuristic::None;
    }
    const char* misLabels[] = {"None", "Balance", "Power"};
    const bool misSupported = bidirectional || vcm || (path && !photon && Transport_.NextEventEstimation);
    int mis = misSupported ? static_cast<int>(Transport_.Mis) : static_cast<int>(RayTracing::MisHeuristic::None);
    if (imguiCombo(bidirectional || vcm ? "Path-strategy MIS" : "Direct-light MIS", &mis, misLabels, 3,
                   misSupported))
        Transport_.Mis = static_cast<RayTracing::MisHeuristic>(mis);
    if ((bidirectional || vcm) && Transport_.Mis == RayTracing::MisHeuristic::None)
        imguiValue("None: equal valid-strategy weights");
    if (imguiCheck("Russian roulette", !bidirectional && !vcm && !pointGi && Transport_.RussianRoulette, path && !vcm))
        Transport_.RussianRoulette = !Transport_.RussianRoulette;
    float rouletteDepth = static_cast<float>(Transport_.RouletteStartDepth);
    if (imguiSlider("Roulette start depth", &rouletteDepth, 1, 12, 1,
                    path && !vcm && Transport_.RussianRoulette))
        Transport_.RouletteStartDepth = static_cast<uint32_t>(rouletteDepth);
    if (photon) DrawMessage("NEE replaces direct photons; no connection MIS");
    else if (vcm && !Transport_.NextEventEstimation)
        DrawMessage("Source NEE off; connection/merge MIS remains");
    else if (path && !Transport_.NextEventEstimation) DrawMessage("BSDF area sampling; delta lights stay explicit");
    if (implemented && accelerationReady)
    {
        // Medium controls occur after reconstruction controls in this panel.
        // Keep a newly enabled medium from submitting a stale guide selection.
        if (Medium_.Enabled)
            Reconstruction_.GeometryGuidance = false;
        if (fullVisibility && !editor.Settings().ShadowsEnabled) editor.SetShadowsEnabled(true);
        auto adaptive = Adaptive_;
        if (fullVisibility) adaptive.Enabled = false;
        editor.SetPreviewIntegrator(algorithm, static_cast<std::uint32_t>(PreviewSamples_),
                                    static_cast<std::uint32_t>(Bounces_), TransportFor(algorithm, Transport_),
                                    markov ? RayTracing::SampleSequence::Independent
                                           : static_cast<RayTracing::SampleSequence>(Sampling_),
                                    adaptive, acceleration, bvhPolicy, geometry, Metropolis_, Photons_,
                                    ProgressivePhotons_, Vcm_, PointGi_, Reconstruction_, SamplingSeed_, Medium_,
                                    static_cast<RayTracing::RadiometryMode>(displayedRadiometry));
    }
    if (imguiCheck("Shadows", fullVisibility || editor.Settings().ShadowsEnabled, !fullVisibility))
        editor.SetShadowsEnabled(!editor.Settings().ShadowsEnabled);
    if (editor.StatusText().rfind("Frame completed with ", 0) != 0)
        DrawMessage(editor.StatusText());
    imguiSeparatorLine();
    std::string materialError;
    std::vector<std::string> materialIssues;
    const bool workerMaterials = WorkerMaterialsSupported(editor.Scene(), Backend_, algorithm,
        materialError, &materialIssues);
    if (running)
    {
        if (imguiButton("Cancel render")) Cancel();
    }
    else if (imguiButton("Render PNG", WorkerSupports(Backend_, algorithm) && workerMaterials
        && accelerationReady && !editor.IsDemoLoading())) Start(editor);
    for (const auto& issue : materialIssues) DrawMessage(issue);
    if (Preparation_.valid()) imguiValue("Preparing scene...");
    const auto progress = Progress();
    if (progress.TotalFrames > 1)
    {
        const auto frames = "Frames: " + std::to_string(progress.CompletedFrames) + "/" +
                            std::to_string(progress.TotalFrames);
        imguiValue(frames.c_str());
    }
    if (!progress.Stage.empty()) DrawMessage(progress.Stage);
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
    if (!progress.Error.empty()) DrawMessage(progress.Error);
    if (!Error_.empty()) DrawMessage(Error_);
    imguiSeparatorLine();
    const char* backendLabels[] = {"CPU", "CUDA", "Compute Shader", "Metal", "DXR", "Per server (auto)"};
    static_assert(std::size(backendLabels) == std::size(BackendIds));
    const char* resolutions[] = {"1280 x 720", "1920 x 1080", "3840 x 2160 (4K)"};
    imguiCombo("Worker backend", &Backend_, backendLabels, static_cast<int>(std::size(backendLabels)), !running);
    imguiCombo("Output resolution", &Resolution_, resolutions, 3, !running);
    imguiSlider(adaptiveActive ? "Offline sample limit"
                : progressive  ? "Offline PPM passes"
                : vcm          ? "Offline VCM passes"
                               : "Offline samples",
                &Samples_, minimumSamples, 256, 1, !running);
    imguiSlider("Tile size", &TileSize_, 64, 512, 64, !running);
    if (bidirectional) imguiValue("BDPT: larger tiles reduce repeated work");
    imguiInputText("Output directory", OutputDirectory_, !running);
    imguiInputText("Access token (optional)", Token_, !running, true);
    if (LocalWorkers_) DrawMessage(LocalWorkers_->Status());
    for (std::size_t i = 0; i < Servers_.size(); ++i)
    {
        const auto label = "HTTP worker " + std::to_string(i + 1);
        imguiInputText(label.c_str(), Servers_[i], !running);
    }
    if (imguiButton("Add server", !running && Servers_.size() < 32)) Servers_.push_back("http://127.0.0.1:" + std::to_string(7831 + Servers_.size()));
    if (imguiButton("Remove last server", !running && Servers_.size() > 1)) Servers_.pop_back();
    imguiEndScrollArea();
}
}
