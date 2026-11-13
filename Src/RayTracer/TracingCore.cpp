#include "Renderer/SubsurfaceValidation.h"
#include "TracingCore.h"
#include "Renderer/AdaptiveReconstruction.h"
#include "RayTracer/Backends/CPU/CpuTransport.h"
#include "RayTracer/Backends/CPU/Motion.h"
#include "RayTracer/Backends/CPU/SpectralTransport.h"
#include "RayTracer/Backends/CPU/BidirectionalPathTracing.h"
#include "RayTracer/Backends/CPU/MetropolisLightTransport.h"
#include "RayTracer/Backends/CPU/PhotonMapping.h"
#include "RayTracer/Backends/CPU/ProgressivePhotonMapping.h"
#include "RayTracer/Backends/CPU/VertexConnectionMerging.h"
#include "RayTracer/Backends/CPU/PointBasedGlobalIllumination.h"

#include "Maths/ColorSpace.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <exception>
#include <limits>
#include <mutex>
#include <thread>
#include <utility>

namespace Arvo
{
namespace RayTracing
{
namespace Detail
{
    struct PersistentExecutor::Impl
    {
        struct Batch
        {
            uint32_t UnitCount = 0;
            uint32_t WorkerLimit = 1;
            UnitFunction Function;
            CancellationProbe Cancellation;
            std::atomic<uint32_t> NextUnit{0};
            std::atomic<uint32_t> RemainingWorkers{0};
            std::atomic<bool> Failed{false};
            std::mutex CompletionMutex;
            std::condition_variable CompletionCondition;
            bool Completed = false;
            std::exception_ptr Exception;
        };

        explicit Impl(uint32_t workerCount)
        {
            const uint32_t count = std::max(1u, workerCount);
            Workers.reserve(count);
            for (uint32_t index = 0; index < count; ++index)
            {
                Workers.emplace_back([this, index]() { WorkerMain(index); });
            }
        }

        ~Impl()
        {
            {
                std::lock_guard<std::mutex> lock(Mutex);
                StopRequested = true;
            }
            WorkCondition.notify_all();
            for (std::thread& worker : Workers)
            {
                if (worker.joinable()) worker.join();
            }
        }

        void WorkerMain(uint32_t workerIndex)
        {
            uint64_t observedGeneration = 0;
            while (true)
            {
                std::shared_ptr<Batch> batch;
                {
                    std::unique_lock<std::mutex> lock(Mutex);
                    WorkCondition.wait(lock, [&]() {
                        return StopRequested || Generation != observedGeneration;
                    });
                    if (StopRequested) return;
                    observedGeneration = Generation;
                    batch = CurrentBatch;
                }
                if (!batch || workerIndex >= batch->WorkerLimit) continue;

                try
                {
                    while (!batch->Failed.load(std::memory_order_relaxed) &&
                           !(batch->Cancellation && batch->Cancellation()))
                    {
                        const uint32_t unit = batch->NextUnit.fetch_add(1, std::memory_order_relaxed);
                        if (unit >= batch->UnitCount) break;
                        batch->Function(unit);
                    }
                }
                catch (...)
                {
                    bool expected = false;
                    if (batch->Failed.compare_exchange_strong(expected, true))
                    {
                        std::lock_guard<std::mutex> lock(batch->CompletionMutex);
                        batch->Exception = std::current_exception();
                    }
                }

                if (batch->RemainingWorkers.fetch_sub(1, std::memory_order_acq_rel) == 1)
                {
                    {
                        std::lock_guard<std::mutex> lock(batch->CompletionMutex);
                        batch->Completed = true;
                    }
                    batch->CompletionCondition.notify_one();
                }
            }
        }

        std::vector<std::thread> Workers;
        std::mutex Mutex;
        std::condition_variable WorkCondition;
        std::shared_ptr<Batch> CurrentBatch;
        uint64_t Generation = 0;
        bool StopRequested = false;
    };

    namespace
    {
        using namespace Cpu;

        Float3 Clamp(const Float3& value, float minimum, float maximum)
        {
            return {std::clamp(value.X, minimum, maximum), std::clamp(value.Y, minimum, maximum),
                std::clamp(value.Z, minimum, maximum)};
        }
        bool IsFinite(const Float3& value)
        { return std::isfinite(value.X) && std::isfinite(value.Y) && std::isfinite(value.Z); }
        bool IsFinite(const Float2& value)
        { return std::isfinite(value.X) && std::isfinite(value.Y); }

        uint8_t EncodeChannel(float value, bool gammaCorrection)
        {
            return gammaCorrection
                ? Maths::LinearToSRGB8(value)
                : Maths::LinearToUNorm8(value);
        }

        bool ValidateInput(
            const Scene& scene,
            const Camera& camera,
            const RenderSettings& settings,
            std::string& error, bool validateMotionSchedule = true)
        {
            if (!ValidateCameraLens(camera.Lens, error) ||
                (validateMotionSchedule && !ValidateSubsurfaceScene(scene, settings, error)) ||
                !ValidateCameraMotionSupport(settings.Integrator, camera.Motion, error) ||
                (validateMotionSchedule && !ValidateSceneMotionSupport(settings.Integrator, scene, error)) ||
                !ValidateCameraSamplePlan(camera, settings, error))
                return false;
            if (!ValidateTileSettings(settings, error)) return false;
            if (!ValidateIntegrator(BackendType::CPU, settings.Integrator, error)) return false;
            if (settings.Width == 0 || settings.Height == 0)
            {
                error = "Image dimensions must be non-zero.";
                return false;
            }
            if (settings.Width > 16384 || settings.Height > 16384)
            {
                error = "Image dimensions exceed the 16384 pixel safety limit.";
                return false;
            }
            if (settings.SamplesPerPixel == 0 || settings.SamplesPerPixel > 4096)
            {
                error = "SamplesPerPixel must be in the range [1, 4096].";
                return false;
            }
            if (settings.MaxReflectionBounces > 32)
            {
                error = "MaxReflectionBounces must not exceed 32.";
                return false;
            }
            if (!std::isfinite(settings.ShadowStrength)
                || settings.ShadowStrength < 0.0f
                || settings.ShadowStrength > 1.0f)
            {
                error = "ShadowStrength must be finite and in the range [0, 1].";
                return false;
            }
            if (!IsFinite(camera.Position) || !IsFinite(camera.Target) || !IsFinite(camera.Up)
                || !std::isfinite(camera.VerticalFieldOfViewDegrees))
            {
                error = "Camera values must be finite.";
                return false;
            }
            if (camera.VerticalFieldOfViewDegrees <= 1.0f || camera.VerticalFieldOfViewDegrees >= 179.0f)
            {
                error = "The camera vertical field of view must be between 1 and 179 degrees.";
                return false;
            }
            const float cameraDirectionLengthSquared = LengthSquared(camera.Target - camera.Position);
            const float cameraUpLengthSquared = LengthSquared(camera.Up);
            if (!std::isfinite(cameraDirectionLengthSquared)
                || !std::isfinite(cameraUpLengthSquared)
                || cameraDirectionLengthSquared < 1.0e-10f
                || cameraUpLengthSquared < 1.0e-10f)
            {
                error = "The camera position, target, and up vector do not form a valid frame.";
                return false;
            }
            const Float3 forward = Normalize(camera.Target - camera.Position);
            const float cameraFrameArea = LengthSquared(Cross(forward, Normalize(camera.Up)));
            if (!std::isfinite(cameraFrameArea) || cameraFrameArea < 1.0e-10f)
            {
                error = "The camera up vector is parallel to its view direction.";
                return false;
            }
            if ((!scene.Triangles.empty() || !scene.Spheres.empty()) && scene.Materials.empty())
            {
                error = "A scene containing geometry must contain at least one material.";
                return false;
            }
            if (!IsFinite(scene.BackgroundTop) || !IsFinite(scene.BackgroundBottom)
                || !std::isfinite(scene.AmbientIntensity) || scene.AmbientIntensity < 0.0f)
            {
                error = "Scene background and ambient values must be finite and non-negative.";
                return false;
            }
            for (const Material& material : scene.Materials)
            {
                if (!IsFinite(material.Albedo)
                    || !IsFinite(material.AmbientColor)
                    || !IsFinite(material.SpecularColor)
                    || !IsFinite(material.Emission)
                    || !std::isfinite(material.Shininess)
                    || !std::isfinite(material.Reflectivity))
                {
                    error = "Material values must be finite.";
                    return false;
                }
                if (material.AmbientColor.X < 0.0f
                    || material.AmbientColor.Y < 0.0f
                    || material.AmbientColor.Z < 0.0f
                    || material.SpecularColor.X < 0.0f
                    || material.SpecularColor.Y < 0.0f
                    || material.SpecularColor.Z < 0.0f
                    || material.Shininess < 0.0f)
                {
                    error =
                        "Material ambient, specular, and shininess values must be "
                        "non-negative.";
                    return false;
                }
            }
            if (!ValidateInstances(scene, error)) return false;
            std::vector<const SceneGeometry*> geometryInputs{&scene};
            for (const auto& geometry : scene.Geometries) geometryInputs.push_back(geometry.get());
            for (const SceneGeometry* geometry : geometryInputs)
            {
                for (const Triangle& triangle : geometry->Triangles)
                {
                    if (triangle.MaterialIndex >= scene.Materials.size())
                    {
                        error = "A triangle has an invalid material index.";
                        return false;
                    }
                    if (!IsFinite(triangle.Position0) || !IsFinite(triangle.Position1)
                        || !IsFinite(triangle.Position2) || !IsFinite(triangle.Normal0)
                        || !IsFinite(triangle.Normal1) || !IsFinite(triangle.Normal2)
                        || !IsFinite(triangle.Color0) || !IsFinite(triangle.Color1)
                        || !IsFinite(triangle.Color2) || !IsFinite(triangle.TexCoord0)
                        || !IsFinite(triangle.TexCoord1) || !IsFinite(triangle.TexCoord2))
                    {
                        error =
                            "Triangle positions, normals, colors, and texture "
                            "coordinates must be finite.";
                        return false;
                    }
                    const Float3 edge1 = triangle.Position1 - triangle.Position0;
                    const Float3 edge2 = triangle.Position2 - triangle.Position0;
                    const float edge1LengthSquared = LengthSquared(edge1);
                    const float edge2LengthSquared = LengthSquared(edge2);
                    const float triangleAreaSquared = LengthSquared(Cross(edge1, edge2));
                    if (!std::isfinite(edge1LengthSquared) || !std::isfinite(edge2LengthSquared)
                        || !std::isfinite(triangleAreaSquared)
                        || triangleAreaSquared < 1.0e-16f)
                    {
                        error = "Degenerate or numerically unsafe triangles are not accepted.";
                        return false;
                    }
                }
                for (const Sphere& sphere : geometry->Spheres)
                {
                    if (sphere.MaterialIndex >= scene.Materials.size())
                    {
                        error = "A sphere has an invalid material index.";
                        return false;
                    }
                    if (!IsFinite(sphere.Center) || !std::isfinite(sphere.Radius)
                        || sphere.Radius <= 0.0f)
                    {
                        error = "Sphere radii must be positive.";
                        return false;
                    }
                    const Float3 radius(sphere.Radius, sphere.Radius, sphere.Radius);
                    if (!IsFinite(sphere.Center - radius) || !IsFinite(sphere.Center + radius))
                    {
                        error = "Sphere bounds must remain finite.";
                        return false;
                    }
                }
            }
            for (const DirectionalLight& light : scene.DirectionalLights)
            {
                const float lightDirectionLengthSquared = LengthSquared(light.DirectionToLight);
                if (!IsFinite(light.DirectionToLight) || !IsFinite(light.Color)
                    || !std::isfinite(light.Intensity) || light.Intensity < 0.0f
                    || !std::isfinite(lightDirectionLengthSquared)
                    || lightDirectionLengthSquared < 1.0e-12f)
                {
                    error = "Directional light values must be finite with a non-zero direction.";
                    return false;
                }
            }
            for (const PointLight& light : scene.PointLights)
            {
                if (!IsFinite(light.Position) || !IsFinite(light.Color)
                    || !std::isfinite(light.Intensity) || light.Intensity < 0.0f)
                {
                    error =
                        "Point light values must be finite with non-negative intensity.";
                    return false;
                }
            }
            for (const RectAreaLight& light : scene.RectAreaLights)
            {
                const float directionLengthSquared = LengthSquared(light.Direction);
                if (!IsFinite(light.Position) || !IsFinite(light.Direction)
                    || !IsFinite(light.Color) || !IsFinite(light.Size)
                    || !std::isfinite(light.Intensity) || light.Intensity < 0.0f
                    || !std::isfinite(directionLengthSquared)
                    || directionLengthSquared < 1.0e-12f
                    || light.Size.X <= 0.0f || light.Size.Y <= 0.0f)
                {
                    error =
                        "Rect area light values must be finite with a non-zero "
                        "direction, positive size, and non-negative intensity.";
                    return false;
                }
                const Float3 direction = Normalize(light.Direction);
                const Float3 helper = std::fabs(direction.Y) < 0.999f
                    ? Float3(0.0f, 1.0f, 0.0f)
                    : Float3(1.0f, 0.0f, 0.0f);
                const Float3 tangent = Normalize(Cross(helper, direction));
                const Float3 bitangent = Normalize(Cross(direction, tangent));
                const Float3 halfTangent = tangent * (light.Size.X * 0.5f);
                const Float3 halfBitangent = bitangent * (light.Size.Y * 0.5f);
                if (!IsFinite(light.Position + halfTangent + halfBitangent)
                    || !IsFinite(light.Position + halfTangent - halfBitangent)
                    || !IsFinite(light.Position - halfTangent + halfBitangent)
                    || !IsFinite(light.Position - halfTangent - halfBitangent))
                {
                    error = "Rect area light bounds must remain finite.";
                    return false;
                }
            }
            return true;
        }

        CameraFrame BuildCameraFrame(const Camera& camera, const RenderSettings& settings)
        {
            CameraFrame frame;
            frame.Position = camera.Position;
            frame.Lens = camera.Lens;
            frame.Motion = camera.Motion;
            frame.Forward = Normalize(camera.Target - camera.Position);
            frame.Right = Normalize(Cross(frame.Forward, camera.Up));
            frame.Up = Normalize(Cross(frame.Right, frame.Forward));
            frame.HalfHeight = std::tan(camera.VerticalFieldOfViewDegrees * Pi / 360.0f);
            frame.HalfWidth = frame.HalfHeight
                * (static_cast<float>(FullImageWidth(settings)) / static_cast<float>(FullImageHeight(settings)));
            return frame;
        }

        using PixelFunction = std::function<void(uint32_t, uint32_t)>;
        using PixelDispatcher = std::function<void(const PixelFunction&)>;

        void StoreFilmPixel(Image& image, const FilmPixel& film, uint32_t x, uint32_t y, bool gamma)
        {
            Float3 color = film.MeanRadiance;
            color.X = std::max(color.X, 0.0f);
            color.Y = std::max(color.Y, 0.0f);
            color.Z = std::max(color.Z, 0.0f);
            color = Clamp(color * Maths::DisplayLinearScale(color.X, color.Y, color.Z), 0.0f, 1.0f);
            auto& pixels = image.GetPixelsRGBA8();
            const size_t offset = (size_t(y) * image.GetWidth() + x) * 4;
            pixels[offset] = EncodeChannel(color.X, gamma);
            pixels[offset + 1] = EncodeChannel(color.Y, gamma);
            pixels[offset + 2] = EncodeChannel(color.Z, gamma);
            pixels[offset + 3] = 255;
        }

        RenderResult RenderCommon(
            const Scene& scene,
            const Camera& camera,
            const RenderSettings& settings,
            const CancellationProbe& cancellation,
            const PixelDispatcher& dispatch,
            const Cpu::BdptDispatcher& dispatchPaths,
            const Cpu::PreparedSpectralFrame* spectral)
        {
            RenderResult result;
            if (cancellation && cancellation())
            {
                result.Status = RenderStatus::Cancelled;
                return result;
            }
            if (!ValidateInput(scene, camera, settings, result.Error)) return result;
            if ((settings.Radiometry == RadiometryMode::Spectral) != (spectral != nullptr))
                return MakeFailedResult("CPU radiometry does not match the prepared frame.");

            if (!ValidateMaterialTransport(scene, BackendType::CPU, settings.Integrator, result.Error))
                return result;

            result.Output = Image(settings.Width, settings.Height);
            result.Film = LinearFilm(settings.Width, settings.Height);

            const auto buildBegin = std::chrono::steady_clock::now();
            const auto accelerator = BuildAccelerator(scene, {settings.Acceleration, settings.BvhPolicy}, cancellation);
            const SceneAccelerator& traversal = *accelerator;
            const auto buildEnd = std::chrono::steady_clock::now();
            if (traversal.WasCancelled() || (cancellation && cancellation()))
            {
                result.Status = RenderStatus::Cancelled;
                return result;
            }

            const CameraFrame cameraFrame = BuildCameraFrame(camera, settings);
            AtomicStatistics statistics;
            const auto renderBegin = std::chrono::steady_clock::now();
            if (settings.Integrator == Algorithm::BidirectionalPathTracing ||
                settings.Integrator == Algorithm::MetropolisLightTransport ||
                settings.Integrator == Algorithm::PhotonMapping ||
                settings.Integrator == Algorithm::ProgressivePhotonMapping ||
                settings.Integrator == Algorithm::PointBasedGlobalIllumination ||
                settings.Integrator == Algorithm::VertexConnectionMerging)
            {
                const bool completed =
                    settings.Integrator == Algorithm::PointBasedGlobalIllumination
                        ? Cpu::RenderPointGiFrame(scene, traversal, cameraFrame, settings, result.Film,
                                                 statistics, cancellation, dispatchPaths, result.Error)
                    : settings.Integrator == Algorithm::VertexConnectionMerging
                        ? Cpu::RenderVertexConnectionMergingFrame(scene, traversal, cameraFrame, settings,
                                                                  result.Film, statistics, cancellation,
                                                                  dispatchPaths, result.Error)
                    : settings.Integrator == Algorithm::ProgressivePhotonMapping
                        ? Cpu::RenderProgressivePhotonFrame(scene, traversal, cameraFrame, settings,
                                                            result.Film, statistics, cancellation,
                                                            dispatchPaths, result.Error)
                    : settings.Integrator == Algorithm::PhotonMapping
                        ? Cpu::RenderPhotonFrame(scene, traversal, cameraFrame, settings, result.Film,
                                                 statistics, cancellation, dispatchPaths, result.Error)
                    : settings.Integrator == Algorithm::MetropolisLightTransport
                        ? Cpu::RenderMetropolisFrame(scene, traversal, cameraFrame, settings, result.Film,
                                                     statistics, cancellation, dispatchPaths, result.Error)
                        : Cpu::RenderBidirectionalFrame(scene, traversal, cameraFrame, settings, result.Film,
                                                        statistics, cancellation, dispatchPaths,
                                                        result.Error);
                if (completed)
                    dispatch([&](uint32_t x, uint32_t y) {
                        StoreFilmPixel(result.Output, result.Film.Pixels[size_t(y) * settings.Width + x],
                            x, y, settings.GammaCorrection);
                    });
                const auto renderEnd = std::chrono::steady_clock::now();
                result.Statistics = statistics.Resolve(
                    std::chrono::duration<double, std::milli>(buildEnd - buildBegin).count(),
                    std::chrono::duration<double, std::milli>(renderEnd - renderBegin).count());
                result.Status = cancellation && cancellation() ? RenderStatus::Cancelled :
                    completed ? RenderStatus::Completed : RenderStatus::Failed;
                return result;
            }
            Cpu::Integrator integrate = nullptr;
            switch (settings.Integrator)
            {
            case Algorithm::RayTracing: integrate = Cpu::Whitted; break;
            case Algorithm::PathTracing: integrate = Cpu::PathTracing; break;
            case Algorithm::DistributedRayTracing: integrate = Cpu::DistributedRayTracing; break;
            default: return MakeFailedResult("The CPU integrator has no implementation");
            }
            const bool sceneMotion = HasSceneMotion(scene);
            const PixelFunction renderPixel = [&](uint32_t x, uint32_t y)
            {
                uint32_t productionSamples = settings.SamplesPerPixel;
                if (settings.ProductionBudgets)
                    productionSamples = settings.ProductionBudgets->Counts[size_t(y) * settings.Width + x];
                else if (settings.Adaptive.Enabled)
                {
                    FilmPixel pilot;
                    for (uint32_t sample = 0; sample < settings.Adaptive.PilotSamples; ++sample)
                    {
                        if (cancellation && cancellation()) return;
                        SampleContext samples = MakePilotSampleContext(settings, x, y, sample);
                        samples.SpectralFrame = spectral;
                        Ray ray = MakeCameraRay(cameraFrame, settings, samples);
                        if (cameraFrame.Lens.Radius > 0)
                            ApplyThinLens(ray, cameraFrame, Sample2D(samples, LensSampleDimension));
                        if (cameraFrame.Motion.Enabled || sceneMotion)
                            ray = SampleCameraMotion(ray, cameraFrame.Position, cameraFrame.Motion,
                                                     cameraFrame.Motion.Interval.Open ==
                                                             cameraFrame.Motion.Interval.Close
                                                         ? 0
                                                         : Sample2D(samples, TimeSampleDimension).X,
                                                     &samples.SceneTime, sceneMotion);
                        statistics.PrimaryRays.fetch_add(1, std::memory_order_relaxed);
                        statistics.PilotRays.fetch_add(1, std::memory_order_relaxed);
                        AccumulateFilmSample(pilot, integrate(scene, traversal, ray, settings, statistics, cancellation, samples));
                    }
                    productionSamples = AdaptiveSampleCount(pilot, settings);
                }
                FilmPixel film;
                for (uint32_t sample = 0; sample < productionSamples; ++sample)
                {
                    if (cancellation && cancellation()) return;
                    SampleContext samples = MakeSampleContext(settings, x, y, sample);
                    samples.SpectralFrame = spectral;
                    samples.SampleCount = productionSamples;
                    Ray ray = MakeCameraRay(cameraFrame, settings, samples);
                    if (cameraFrame.Lens.Radius > 0)
                        ApplyThinLens(ray, cameraFrame, Sample2D(samples, LensSampleDimension));
                    if (cameraFrame.Motion.Enabled || sceneMotion)
                        ray = SampleCameraMotion(ray, cameraFrame.Position, cameraFrame.Motion,
                                                 cameraFrame.Motion.Interval.Open ==
                                                         cameraFrame.Motion.Interval.Close
                                                     ? 0
                                                     : Sample2D(samples, TimeSampleDimension).X,
                                                     &samples.SceneTime, sceneMotion);
                    statistics.PrimaryRays.fetch_add(1, std::memory_order_relaxed);
                    AccumulateFilmSample(film, integrate(scene, traversal, ray, settings, statistics, cancellation, samples));
                }
                result.Film.Pixels[static_cast<size_t>(y) * settings.Width + x] = film;
                StoreFilmPixel(result.Output, film, x, y, settings.GammaCorrection);
                statistics.CompletedPixels.fetch_add(1, std::memory_order_relaxed);
            };

            dispatch(renderPixel);
            const auto renderEnd = std::chrono::steady_clock::now();
            result.Statistics = statistics.Resolve(
                std::chrono::duration<double, std::milli>(buildEnd - buildBegin).count(),
                std::chrono::duration<double, std::milli>(renderEnd - renderBegin).count());
            result.Status = cancellation && cancellation()
                ? RenderStatus::Cancelled
                : RenderStatus::Completed;
            return result;
        }
    }

    PersistentExecutor::PersistentExecutor(uint32_t workerCount)
        : m_Impl(std::make_unique<Impl>(workerCount))
    {
    }

    PersistentExecutor::~PersistentExecutor() = default;

    uint32_t PersistentExecutor::GetWorkerCount() const
    {
        return static_cast<uint32_t>(m_Impl->Workers.size());
    }

    void PersistentExecutor::Execute(
        uint32_t unitCount,
        uint32_t requestedWorkers,
        const UnitFunction& function,
        const CancellationProbe& cancellation)
    {
        if (unitCount == 0) return;
        auto batch = std::make_shared<Impl::Batch>();
        batch->UnitCount = unitCount;
        batch->WorkerLimit = requestedWorkers == 0
            ? GetWorkerCount()
            : std::min(requestedWorkers, GetWorkerCount());
        batch->WorkerLimit = std::max(1u, std::min(batch->WorkerLimit, unitCount));
        batch->RemainingWorkers.store(batch->WorkerLimit, std::memory_order_relaxed);
        batch->Function = function;
        batch->Cancellation = cancellation;
        {
            std::lock_guard<std::mutex> lock(m_Impl->Mutex);
            m_Impl->CurrentBatch = batch;
            ++m_Impl->Generation;
        }
        m_Impl->WorkCondition.notify_all();
        {
            std::unique_lock<std::mutex> lock(batch->CompletionMutex);
            batch->CompletionCondition.wait(lock, [&]() { return batch->Completed; });
        }
        {
            std::lock_guard<std::mutex> lock(m_Impl->Mutex);
            m_Impl->CurrentBatch.reset();
        }
        if (batch->Exception) std::rethrow_exception(batch->Exception);
    }

    RenderResult RenderCpuTiles(
        const Scene& scene,
        const Camera& camera,
        const RenderSettings& settings,
        PersistentExecutor& executor,
        const CancellationProbe& cancellation,
        const Cpu::PreparedSpectralFrame* spectral)
    {
        return RenderCommon(scene, camera, settings, cancellation, [&](const PixelFunction& pixel)
        {
            constexpr uint32_t TileSize = 16;
            const uint32_t tilesX = (settings.Width + TileSize - 1) / TileSize;
            const uint32_t tilesY = (settings.Height + TileSize - 1) / TileSize;
            executor.Execute(tilesX * tilesY, settings.WorkerCount, [&](uint32_t tile)
            {
                const uint32_t firstX = (tile % tilesX) * TileSize;
                const uint32_t firstY = (tile / tilesX) * TileSize;
                for (uint32_t y = firstY; y < std::min(firstY + TileSize, settings.Height); ++y)
                {
                    for (uint32_t x = firstX; x < std::min(firstX + TileSize, settings.Width); ++x)
                    {
                        pixel(x, y);
                    }
                }
            }, cancellation);
        }, [&](uint32_t count, const std::function<void(uint32_t)>& path) {
            executor.Execute(count, settings.WorkerCount, path, cancellation);
        }, spectral);
    }

    bool ValidateRenderInput(
        const Scene& scene,
        const Camera& camera,
        const RenderSettings& settings,
        std::string& error, bool validateMotionSchedule)
    {
        return ValidateInput(scene, camera, settings, error, validateMotionSchedule);
    }
}
}
}
