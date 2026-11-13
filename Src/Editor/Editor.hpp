#pragma once

#include "Editor/Demos.hpp"
#include "Editor/RenderAdapter.hpp"

#include <array>
#include <filesystem>
#include <future>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace Arvo::Editor
{
bool HasObjectMotion(const SceneSnapshot&);
bool CanAnimateNode(const SceneSnapshot&, SceneNodeId, std::string& error);

class EditorCore
{
public:
    explicit EditorCore(std::filesystem::path contentRoot);
    ~EditorCore();

    EditorCore(const EditorCore&) = delete;
    EditorCore& operator=(const EditorCore&) = delete;

    void RegisterRenderAdapter(std::shared_ptr<IRenderAdapter> adapter);
    void SetPickHandler(PickHandler handler);

    void SelectDemo(DemoId demo);
    void SelectBackend(const std::string& backendId);
    void SetFillMode(FillMode mode);
    void SetShadowsEnabled(bool enabled);
    void SetOutlineEnabled(bool enabled);
    void SetMaterialScattering(std::size_t index, const RayTracing::MaterialScattering& scattering);
    void SetSceneMaterialModel(RayTracing::MaterialModel model);
    void SetCameraLens(const RayTracing::CameraLens& lens);
    void SetCameraMotion(const RayTracing::MotionData::Exposure& motion);
    void SetNodeMotion(SceneNodeId node, const RayTracing::MotionData::RigidMotion& motion);
    void DisableObjectMotion();
    void SetTemporalPreview(bool enabled, const RayTracing::TemporalOptions& options = {});
    void SetMicropolygons(const RayTracing::MicropolygonOptions&);
    void SetPreviewIntegrator(RayTracing::Algorithm integrator, std::uint32_t samples, std::uint32_t bounces,
                              RayTracing::TransportOptions transport = {},
                              RayTracing::SampleSequence sampling = RayTracing::SampleSequence::Independent,
                              RayTracing::AdaptiveOptions adaptive = {},
                              RayTracing::AccelerationType acceleration = RayTracing::AccelerationType::Bvh,
                              RayTracing::BvhBuildPolicy bvhPolicy = RayTracing::BvhBuildPolicy::Default,
                              RayTracing::GeometryMode geometry = RayTracing::GeometryMode::Flat,
                              RayTracing::MetropolisOptions metropolis = {},
                              RayTracing::PhotonMappingOptions photons = {},
                              RayTracing::ProgressivePhotonOptions progressivePhotons = {},
                              RayTracing::VertexConnectionMergingOptions vcm = {},
                              RayTracing::PointGiOptions pointGi = {},
                              RayTracing::ReconstructionOptions reconstruction = {},
                              uint32_t samplingSeed = 0, RayTracing::MediumOptions medium = {},
                              RayTracing::RadiometryMode radiometry = RayTracing::RadiometryMode::RGB);
    void ResizeViewport(std::uint32_t width, std::uint32_t height);
    void SelectNode(SceneNodeId node);
    void HandleViewportPointer(float x, float y, const Rect& viewportBounds);
    void SetCameraMovementKey(char key, bool down) noexcept;
    void HandleCameraPointer(float x, float y, bool rotateButtonDown);
    void ClearCameraInput() noexcept;

    // Tick never waits for background work. Completed jobs are collected here.
    void Tick(float deltaSeconds);
    void RequestRender();

    [[nodiscard]] const SceneSnapshot& Scene() const noexcept { return Scene_; }
    [[nodiscard]] const ViewportSettings& Settings() const noexcept { return Settings_; }
    [[nodiscard]] const Image& ViewportImage() const noexcept { return ViewportImage_; }
    [[nodiscard]] std::uint64_t ViewportRevision() const noexcept { return ViewportRevision_; }
    [[nodiscard]] const std::string& ViewportBackend() const noexcept { return ViewportBackend_; }
    [[nodiscard]] DemoId SelectedDemo() const noexcept { return SelectedDemo_; }
    [[nodiscard]] bool IsDemoLoading() const noexcept { return ActiveDemoLoad_ != nullptr; }
    [[nodiscard]] bool IsPickInProgress() const noexcept { return ActivePickJob_ != nullptr; }
    [[nodiscard]] const std::string& SelectedBackend() const noexcept { return SelectedBackend_; }
    [[nodiscard]] const std::string& StatusText() const noexcept { return StatusText_; }
    [[nodiscard]] std::uint64_t Generation() const noexcept { return Generation_; }
    [[nodiscard]] std::vector<BackendDescriptor> Backends() const;

private:
    struct RenderJob
    {
        std::uint64_t Generation = 0;
        std::string BackendId;
        RenderDomain Domain = RenderDomain::RealtimeRenderer;
        CancellationSource Cancellation;
        std::future<RenderResult> Future;
    };

    struct DemoLoadResult
    {
        SceneSnapshot Scene;
        std::string Error;
    };

    struct DemoLoadJob
    {
        DemoId Demo = DemoId::CornellBox;
        std::future<DemoLoadResult> Future;
    };

    struct PickJob
    {
        std::uint64_t Request = 0;
        std::uint64_t DemoRequest = 0;
        std::future<SceneNodeId> Future;
    };

    void InvalidateAndRender();
    void InvalidateCameraAndRender();
    [[nodiscard]] bool UpdateCameraMovement(float deltaSeconds);
    void StartRender();
    void RetireActiveJob();
    void CollectFinishedJobs();
    void CollectRetiredJobs();
    [[nodiscard]] bool HasBlockingRetiredRenderJob(
        const std::string& backendId,
        RenderDomain domain) const noexcept;
    void StartDemoLoad(DemoId demo);
    void CollectFinishedDemoLoad();
    void RetirePickJob();
    void CollectFinishedPick();
    void CollectRetiredPicks();
    void ApplyResult(RenderResult result, std::uint64_t generation, const std::string& backendId);
    [[nodiscard]] bool HasLoadedSelectedDemo() const noexcept;
    [[nodiscard]] std::shared_ptr<IRenderAdapter> FindAdapter(const std::string& id) const;
    [[nodiscard]] BackendDescriptor FindDescriptor(const std::string& id) const;
    DemoCatalog Demos_;
    SceneSnapshot Scene_;
    ViewportSettings Settings_;
    std::unordered_map<std::string, std::shared_ptr<IRenderAdapter>> Adapters_;
    PickHandler PickHandler_;
    DemoId SelectedDemo_ = DemoId::CornellBox;
    std::string SelectedBackend_ = "renderer.dxr";
    std::string StatusText_ = "No renderer adapter registered";
    Image ViewportImage_;
    std::string ViewportBackend_;
    std::uint64_t ViewportRevision_ = 0;
    std::uint64_t Generation_ = 1;
    std::uint64_t SceneRevision_ = 0;
    std::uint64_t DemoRequest_ = 1;
    std::uint64_t AppliedDemoRequest_ = 1;
    std::uint64_t PickRequest_ = 0;
    bool RenderDirty_ = false;
    bool PendingRenderAfterLoad_ = false;
    std::array<bool, 4> CameraMovementKeys_{};
    bool CameraPointerActive_ = false;
    float PreviousCameraPointerX_ = 0.0f;
    float PreviousCameraPointerY_ = 0.0f;
    std::unique_ptr<RenderJob> ActiveJob_;
    std::vector<std::unique_ptr<RenderJob>> RetiredJobs_;
    std::unique_ptr<DemoLoadJob> ActiveDemoLoad_;
    std::unique_ptr<PickJob> ActivePickJob_;
    std::vector<std::unique_ptr<PickJob>> RetiredPickJobs_;
};

} // namespace Arvo::Editor
