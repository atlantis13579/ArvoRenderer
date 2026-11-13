#include "Renderer/TemporalSession.h"
#include "Renderer/MicropolygonPreparation.h"
#include <chrono>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace Arvo::RayTracing
{
namespace
{
// Compare semantic fields, never object padding or native resource addresses.
// Resolution/projection/backend changes are checked by temporal reconstruction.
bool SameTransport(const RenderSettings& a, const RenderSettings& b)
{
    const bool sameDicing =
        a.Micropolygons == b.Micropolygons &&
        ((!a.DicingPlan && !b.DicingPlan) ||
         (a.DicingPlan && b.DicingPlan && a.DicingPlan->Fingerprint == b.DicingPlan->Fingerprint &&
          a.DicingPlan->Components == b.DicingPlan->Components));
    return a.Integrator == b.Integrator && a.Radiometry == b.Radiometry && sameDicing &&
           a.Transport == b.Transport && a.Sampling == b.Sampling && a.SamplingSeed == b.SamplingSeed &&
           a.Adaptive == b.Adaptive && a.Metropolis == b.Metropolis && a.Photons == b.Photons &&
           a.ProgressivePhotons == b.ProgressivePhotons && a.Vcm == b.Vcm && a.PointGi == b.PointGi &&
           a.Acceleration == b.Acceleration && a.BvhPolicy == b.BvhPolicy &&
           a.SamplesPerPixel == b.SamplesPerPixel && a.MaxReflectionBounces == b.MaxReflectionBounces &&
           a.ShadowsEnabled == b.ShadowsEnabled && a.ShadowStrength == b.ShadowStrength;
}
bool SameIdentity(const TemporalIdentity& a, const TemporalIdentity& b)
{
    return a.Sequence == b.Sequence && a.Content == b.Content && a.Frame == b.Frame && a.Samples == b.Samples;
}
bool SameView(const Camera& a, const Camera& b)
{
    const auto same = [](Float3 x, Float3 y) { return x.X == y.X && x.Y == y.Y && x.Z == y.Z; };
    return same(a.Position, b.Position) && same(a.Target, b.Target) && same(a.Up, b.Up) &&
           a.VerticalFieldOfViewDegrees == b.VerticalFieldOfViewDegrees && a.Lens == b.Lens &&
           a.Motion == b.Motion;
}
uint64_t Advance(uint64_t value)
{
    if (value == std::numeric_limits<uint64_t>::max())
        throw std::overflow_error("Temporal sequence identity exhausted");
    return value + 1;
}
} // namespace
struct TemporalSession::State
{
    std::mutex Mutex;
    uint64_t Version = 0, Sequence = 1, ContentRevision = 0;
    RenderSettings Requested;
    std::shared_ptr<const TemporalHistory> History;
};
struct TemporalSession::Prepared
{
    std::weak_ptr<State> Owner;
    uint64_t Version = 0, ContentRevision = 0;
    RenderSettings Requested, Sampling;
    TemporalOptions Options;
    TemporalIdentity Identity;
    std::shared_ptr<const TemporalHistory> Previous;
};
TemporalSession::TemporalSession() : State_(std::make_shared<State>()) {}
std::unique_ptr<TemporalSession> TemporalSession::Fork() const
{
    auto branch = std::make_unique<TemporalSession>();
    std::lock_guard<std::mutex> lock(State_->Mutex);
    branch->State_->Version = State_->Version;
    branch->State_->Sequence = State_->Sequence;
    branch->State_->ContentRevision = State_->ContentRevision;
    branch->State_->Requested = State_->Requested;
    branch->State_->History = State_->History;
    return branch;
}
const RenderSettings& TemporalSession::Frame::Settings() const
{
    if (!Data_)
        throw std::logic_error("Unprepared temporal frame");
    return Data_->Sampling;
}
const TemporalIdentity& TemporalSession::Frame::Identity() const
{
    if (!Data_)
        throw std::logic_error("Unprepared temporal frame");
    return Data_->Identity;
}
void TemporalSession::Reset()
{
    std::lock_guard<std::mutex> lock(State_->Mutex);
    const auto version = Advance(State_->Version), sequence = Advance(State_->Sequence);
    State_->Version = version;
    State_->Sequence = sequence;
    State_->History.reset();
}
std::shared_ptr<const TemporalHistory> TemporalSession::History() const
{
    std::lock_guard<std::mutex> lock(State_->Mutex);
    return State_->History;
}
TemporalSession::Frame TemporalSession::Begin(const RenderSettings& settings, uint64_t contentRevision,
                                              const TemporalOptions& options)
{
    std::string error;
    if (!ValidateTileSettings(settings, error) || !ValidateTemporalOptions(options, error))
        throw std::invalid_argument(error);
    if (settings.Medium.Enabled)
        throw std::invalid_argument("Temporal denoising is unavailable for participating media");
    if (settings.Micropolygons.Enabled && !settings.DicingPlan)
        throw std::invalid_argument(
            "Temporal micropolygons require a shared geometry plan before starting the frame");
    if (settings.Width != FullImageWidth(settings) || settings.Height != FullImageHeight(settings) ||
        settings.TileX || settings.TileY || settings.Reconstruction.Filter != ReconstructionFilter::None ||
        settings.SampleStream != CameraSampleStream::Production || settings.ProductionBudgets)
        throw std::invalid_argument("A temporal sequence requires a complete raw film before reconstruction");
    auto prepared = std::make_shared<Prepared>();
    prepared->Owner = State_;
    prepared->Requested = prepared->Sampling = settings;
    prepared->ContentRevision = contentRevision;
    prepared->Options = options;
    std::lock_guard<std::mutex> lock(State_->Mutex);
    // Nonzero seeds jitter even a one-SPP camera. Cycle the complete nonzero
    // seed domain; the legacy zero key remains available to stateless rendering.
    constexpr uint64_t SeedPopulation = std::numeric_limits<uint32_t>::max();
    uint64_t frame = State_->History ? Advance(State_->History->Identity.Frame) : 0;
    uint64_t sequence = State_->Sequence;
    if (frame == SeedPopulation)
    {
        sequence = Advance(sequence);
        frame = 0;
    }
    State_->Version = prepared->Version = Advance(State_->Version);
    prepared->Previous = State_->History;
    const bool compatible = prepared->Previous && State_->ContentRevision == contentRevision &&
                            SameTransport(State_->Requested, settings);
    const uint32_t seed = uint32_t((uint64_t(settings.SamplingSeed) + frame) % SeedPopulation + 1);
    prepared->Sampling.SamplingSeed = seed;
    prepared->Identity = {sequence,
                          compatible ? prepared->Previous->Identity.Content : prepared->Version, frame, seed};
    Frame result;
    result.Data_ = std::move(prepared);
    return result;
}
TemporalSession::Result TemporalSession::Reconstruct(IRayTracingRHI& backend, const Frame& frame,
                                                     const LinearFilm& film,
                                                     const GeometryGuideImage& geometry, const Camera& camera,
                                                     const CancellationToken& cancellation)
{
    return Reconstruct([&](const auto& input, const auto* previous, const auto& options, const auto& token) {
        return backend.ReconstructTemporal(input, previous, options, token);
    }, frame, film, geometry, camera, cancellation);
}
TemporalSession::Result TemporalSession::Reconstruct(const Reconstruction& reconstruct, const Frame& frame,
                                                     const LinearFilm& film,
                                                     const GeometryGuideImage& geometry, const Camera& camera,
                                                     const CancellationToken& cancellation)
{
    Result result;
    const auto prepared = frame.Data_;
    if (!prepared || prepared->Owner.lock() != State_)
    {
        result.Error = "Temporal frame belongs to another or expired sequence";
        return result;
    }
    result.Identity = prepared->Identity;
    {
        std::lock_guard<std::mutex> lock(State_->Mutex);
        if (State_->Version != prepared->Version || cancellation.IsCancellationRequested())
        {
            result.Status = RenderStatus::Cancelled;
            return result;
        }
    }
    if (film.Width != prepared->Sampling.Width || film.Height != prepared->Sampling.Height)
    {
        result.Error = "Temporal film extent differs from its prepared frame";
        return result;
    }
    TemporalFrame input{film, geometry, camera, prepared->Identity};
    if (!reconstruct) { result.Error = "Temporal reconstruction callback is empty"; return result; }
    auto filtered = reconstruct(input, prepared->Previous.get(), prepared->Options, cancellation);
    result.Status = filtered.Status;
    result.Error = std::move(filtered.Error);
    if (filtered.Status != RenderStatus::Completed)
        return result;
    if (!SameIdentity(filtered.History.Identity, prepared->Identity))
    {
        result.Status = RenderStatus::Failed;
        result.Error = "Backend returned history for another temporal frame";
        return result;
    }
    auto history = std::make_shared<const TemporalHistory>(std::move(filtered.History));
    result.Display = std::move(filtered.Display);
    result.Reset = filtered.Reset;
    result.AcceptedHistoryPixels = filtered.AcceptedHistoryPixels;
    // A cancelled/superseded candidate cannot advance frame, seed or history.
    // Capturing a weak owner also makes publication safe after session teardown.
    result.Accept = [prepared, history = std::move(history), cancellation]() {
        auto state = prepared->Owner.lock();
        if (!state || cancellation.IsCancellationRequested())
            return false;
        std::lock_guard<std::mutex> lock(state->Mutex);
        if (state->Version != prepared->Version || cancellation.IsCancellationRequested())
            return false;
        state->Version = Advance(state->Version);
        state->Sequence = prepared->Identity.Sequence;
        state->Requested = prepared->Requested;
        state->ContentRevision = prepared->ContentRevision;
        state->History = history;
        return true;
    };
    return result;
}
TemporalSession::Result TemporalSession::Render(IRayTracingRHI& backend, std::shared_ptr<const Scene> scene,
                                                const Camera& camera, const RenderSettings& settings,
                                                uint64_t contentRevision, const TemporalOptions& options,
                                                const CancellationToken& cancellation)
{
    Result result;
    if (cancellation.IsCancellationRequested())
    {
        result.Status = RenderStatus::Cancelled;
        return result;
    }
    if (!scene || camera.Lens.Radius != 0 || camera.Motion.Enabled || HasSceneMotion(*scene))
    {
        result.Error = "Temporal rendering requires a static scene and instantaneous pinhole camera";
        return result;
    }
    auto prepared = settings;
    if (settings.Micropolygons.Enabled)
    {
        auto diced = backend.PrepareMicropolygons(*scene, camera, settings, cancellation);
        if (diced.Status != RenderStatus::Completed)
        {
            result.Status = diced.Status;
            result.Error = std::move(diced.Error);
            return result;
        }
        scene = std::move(diced.Geometry);
        prepared.Micropolygons.Enabled = false;
        prepared.DicingPlan = std::move(diced.Plan);
    }
    // A camera-dependent topology change invalidates primitive correspondence.
    // Keep the plan identity even after the geometry has been materialized.
    auto frame = Begin(prepared, contentRevision, options);
    auto job = backend.RenderAsync(scene, camera, frame.Settings());
    constexpr auto FramePollInterval = std::chrono::milliseconds(1);
    while (!job.IsReady())
    {
        if (cancellation.IsCancellationRequested())
            job.Cancel();
        std::this_thread::sleep_for(FramePollInterval);
    }
    auto raw = job.GetResult();
    if (raw.Status != RenderStatus::Completed || cancellation.IsCancellationRequested())
    {
        result.Status = cancellation.IsCancellationRequested() ? RenderStatus::Cancelled : raw.Status;
        result.Error = std::move(raw.Error);
        return result;
    }
    // Center-ray geometry is deterministic. A stationary accepted frame already
    // owns the correct guides; a changed pose/content/backend needs a fresh query.
    const auto previous = frame.Data_->Previous;
    const bool reuseGeometry =
        previous && previous->Identity.Content == frame.Identity().Content &&
        previous->Backend == backend.GetBackendInfo().Type && previous->Geometry.Width == settings.Width &&
        previous->Geometry.Height == settings.Height && SameView(previous->View, camera);
    GeometryGuideResult geometry;
    if (!reuseGeometry)
        geometry = backend.CaptureGeometryGuides(*scene, camera, frame.Settings(), cancellation);
    if (!reuseGeometry && geometry.Status != RenderStatus::Completed)
    {
        result.Status = geometry.Status;
        result.Error = std::move(geometry.Error);
        return result;
    }
    result = Reconstruct(backend, frame, raw.Film, reuseGeometry ? previous->Geometry : geometry.Image,
                         camera, cancellation);
    if (result.Status == RenderStatus::Completed)
    {
        result.Film = std::move(raw.Film);
        result.Statistics = raw.Statistics;
    }
    return result;
}
} // namespace Arvo::RayTracing
