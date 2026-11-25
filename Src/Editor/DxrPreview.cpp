#include "Editor/ModuleAdapters.h"
#include "Maths/ColorSpace.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <future>
#include <mutex>
#include <thread>

namespace Arvo::Editor
{
namespace
{
namespace Ray = RayTracing;
constexpr uint32_t PreviewSamples = 1, ReflectionBounces = 1, TransmissionBounces = 2, SponzaBounceLimit = 3,
                   SettledFrames = 32;
// A 12-bit linear table differs from exact 8-bit sRGB by at most one code.
// Presentation alone uses this approximation; the offline film stays untouched.
constexpr uint32_t DisplayLinearIntervals = 1u << 12;
const std::array<uint8_t, DisplayLinearIntervals + 1>& DisplayTable()
{
    static const auto table = [] {
        std::array<uint8_t, DisplayLinearIntervals + 1> values{};
        for (size_t index = 0; index < values.size(); ++index)
            values[index] = Maths::LinearToSRGB8(float(index) / DisplayLinearIntervals);
        return values;
    }();
    return table;
}
uint8_t DisplayByte(float linear, const std::array<uint8_t, DisplayLinearIntervals + 1>& table)
{
    if (!std::isfinite(linear)) return linear > 0 ? 255 : 0;
    const auto index = uint32_t(std::clamp(linear, 0.0f, 1.0f) * DisplayLinearIntervals + 0.5f);
    return table[index];
}

// Preview traces every viewport pixel, with bounded transport independent of
// the offline estimator. Never trade spatial resolution for a cheaper frame.
// Whitted supplies short ideal reflection/refraction paths. Sponza follows the
// selected path depth up to three bounces to keep viewport work bounded.
// Area emitters become center point sources. Rough surfaces use ideal interfaces
// and SSS uses diffuse albedo. A pinhole camera observes the shutter midpoint.
// Stationary frames accumulate linear radiance until the camera, scene or
// viewport settings change.
class DxrPreviewAdapter final : public IRenderAdapter
{
public:
    DxrPreviewAdapter(BackendDescriptor descriptor, std::shared_ptr<Ray::IRayTracingRHI> rhi)
        : Descriptor_(std::move(descriptor)), RHI_(std::move(rhi)) {}
    BackendDescriptor Descriptor() const override { return Descriptor_; }
    bool IsAvailable() const override { return RHI_ && RHI_->GetBackendInfo().IsAvailable; }
    void ResetTemporalHistory() override
    {
        std::lock_guard<std::mutex> lock(Mutex_);
        History_.reset();
        ++Revision_;
    }
    RenderResult Render(const RenderRequest& request, const CancellationToken& cancellation) override
    {
        const auto failed = [](const std::string& error) {
            RenderResult result;
            result.Status = RenderStatus::Failed;
            result.Error = error;
            return result;
        };
        if (!request.Scene || !RHI_) return failed("DXR preview has no scene or native backend");
        if (cancellation.IsCancellationRequested()) return {RenderStatus::Cancelled};
        std::shared_ptr<const Ray::Scene> scene;
        std::shared_ptr<const History> previous;
        uint64_t revision;
        {
            std::lock_guard<std::mutex> lock(Mutex_);
            if (Scene_ && SceneRevision_ == request.Scene->Revision && Demo_ == request.Scene->Demo &&
                Geometry_ == request.Settings.Geometry) scene = Scene_;
            previous = History_;
            revision = Revision_;
        }
        if (!scene)
        {
            auto candidate = std::make_shared<Ray::Scene>(*BuildOfflineScene(*request.Scene, request.Settings.Geometry));
            for (auto& material : candidate->Materials)
            {
                material.MicrofacetSlope = 0;
                if (material.SubsurfaceMeanFreePath > 0)
                {
                    material.SubsurfaceMeanFreePath = 0;
                    material.Model = Ray::MaterialModel::Diffuse;
                    material.Albedo = material.Albedo * material.SubsurfaceAlbedoScale;
                }
            }
            for (const auto& area : candidate->RectAreaLights)
                candidate->PointLights.push_back({area.Position, area.Color, area.Intensity, area.CastShadows});
            candidate->RectAreaLights.clear();
            if (cancellation.IsCancellationRequested()) return {RenderStatus::Cancelled};
            scene = candidate;
            std::lock_guard<std::mutex> lock(Mutex_);
            Scene_ = scene;
            SceneRevision_ = request.Scene->Revision;
            Demo_ = request.Scene->Demo;
            Geometry_ = request.Settings.Geometry;
        }
        auto camera = BuildOfflineCamera(*request.Scene);
        camera.Lens.Radius = 0;
        const float midpoint = (camera.Motion.Interval.Open + camera.Motion.Interval.Close) / 2;
        camera.Motion.Interval = {midpoint, midpoint};
        Ray::RenderSettings settings;
        settings.Width = std::max(1u, request.Settings.Width);
        settings.Height = std::max(1u, request.Settings.Height);
        const bool sponza = request.Scene->Demo == DemoId::SponzaAtrium;
        settings.Integrator = sponza
            ? Ray::Algorithm::PathTracing : Ray::Algorithm::RayTracing;
        settings.SamplesPerPixel = PreviewSamples;
        const bool hasGlass = std::any_of(scene->Materials.begin(), scene->Materials.end(),
            [](const auto& material) { return material.Model == Ray::MaterialModel::Dielectric; });
        settings.MaxReflectionBounces = sponza ? std::min(request.Settings.MaxBounces, SponzaBounceLimit)
                                               : (hasGlass ? TransmissionBounces : ReflectionBounces);
        settings.Transport.NextEventEstimation = true;
        settings.Transport.ImportanceSampling = true;
        settings.Transport.Mis = Ray::MisHeuristic::None;
        settings.Transport.RussianRoulette = false;
        settings.ShadowsEnabled = request.Settings.ShadowsEnabled;
        // Interactive camera motion coalesces frames without retiring their
        // generation. Its independent revision must also invalidate accumulation.
        const bool reuse = previous && previous->Generation == request.Generation &&
                           previous->CameraRevision == request.Scene->CameraRevision &&
                           previous->Scene == scene && previous->Width == settings.Width &&
                           previous->Height == settings.Height;
        const uint32_t samples = reuse ? previous->Samples : 0;
        settings.SamplingSeed = request.Settings.SamplingSeed + samples + 1;
        auto job = RHI_->RenderAsync(scene, camera, settings);
        while (!job.IsReady())
        {
            if (cancellation.IsCancellationRequested()) job.Cancel();
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        auto rendered = job.GetResult();
        if (cancellation.IsCancellationRequested() || rendered.Status == Ray::RenderStatus::Cancelled)
            return {RenderStatus::Cancelled};
        if (rendered.Status != Ray::RenderStatus::Completed) return failed(rendered.Error);
        const auto pixels = size_t(settings.Width) * settings.Height;
        if (rendered.Film.Pixels.size() != pixels) return failed("Incomplete DXR preview film");
        auto candidate = std::make_shared<History>();
        candidate->Generation = request.Generation;
        candidate->CameraRevision = request.Scene->CameraRevision;
        candidate->Scene = scene;
        candidate->Width = settings.Width;
        candidate->Height = settings.Height;
        candidate->Samples = samples + 1;
        candidate->Mean.resize(pixels);
        RenderResult result;
        result.Status = RenderStatus::Completed;
        result.Output.Width = settings.Width;
        result.Output.Height = settings.Height;
        result.Output.PixelsRGBA8.resize(pixels * 4);
        const auto& display = DisplayTable();
        const auto convert = [&](size_t first, size_t last) {
            for (size_t pixel = first; pixel < last; ++pixel)
            {
                auto value = rendered.Film.Pixels[pixel].MeanRadiance;
                if (reuse) value = previous->Mean[pixel] +
                    (value - previous->Mean[pixel]) / float(candidate->Samples);
                candidate->Mean[pixel] = value;
                auto* rgba = result.Output.PixelsRGBA8.data() + pixel * 4;
                rgba[0] = DisplayByte(value.X, display);
                rgba[1] = DisplayByte(value.Y, display);
                rgba[2] = DisplayByte(value.Z, display);
                rgba[3] = 255;
            }
        };
        constexpr size_t MaximumConversionTasks = 4, MinimumPixelsPerTask = 65536;
        const size_t tasks = std::max(size_t{1}, std::min({MaximumConversionTasks,
            size_t(std::max(1u, std::thread::hardware_concurrency())), pixels / MinimumPixelsPerTask}));
        std::array<std::future<void>, MaximumConversionTasks - 1> conversions;
        for (size_t task = 1; task < tasks; ++task)
            conversions[task - 1] = std::async(std::launch::async, convert, pixels * task / tasks, pixels * (task + 1) / tasks);
        convert(0, pixels / tasks);
        for (size_t task = 1; task < tasks; ++task) conversions[task - 1].get();
        result.ContinueRendering = candidate->Samples < SettledFrames;
        result.Accept = [this, candidate, revision] {
            std::lock_guard<std::mutex> lock(Mutex_);
            if (revision != Revision_) return false;
            History_ = candidate;
            ++Revision_;
            return true;
        };
        return result;
    }
private:
    struct History
    {
        uint64_t Generation = 0, CameraRevision = 0;
        std::shared_ptr<const Ray::Scene> Scene;
        uint32_t Width = 0, Height = 0, Samples = 0;
        std::vector<Ray::Float3> Mean;
    };
    BackendDescriptor Descriptor_;
    std::shared_ptr<Ray::IRayTracingRHI> RHI_;
    std::mutex Mutex_;
    uint64_t Revision_ = 0, SceneRevision_ = 0;
    DemoId Demo_ = DemoId::CornellBox;
    Ray::GeometryMode Geometry_ = Ray::GeometryMode::Flat;
    std::shared_ptr<const Ray::Scene> Scene_;
    std::shared_ptr<const History> History_;
};
} // namespace

std::shared_ptr<IRenderAdapter> CreateDxrPreviewAdapter(
    BackendDescriptor descriptor, std::shared_ptr<RayTracing::IRayTracingRHI> rhi)
{
    return std::make_shared<DxrPreviewAdapter>(std::move(descriptor), std::move(rhi));
}
} // namespace Arvo::Editor
