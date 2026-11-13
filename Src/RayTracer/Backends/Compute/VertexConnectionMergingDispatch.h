#pragma once
#include "VertexConnectionMergingLayout.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace Arvo::RayTracing::ComputeVcm
{
// Host scheduling only. Scene traversal, light transport, spatial sorting and
// film arithmetic remain in this backend's independent HLSL implementation.
constexpr uint32_t SceneDescriptors = 10, FrameDescriptors = 12;
constexpr uint32_t DescriptorCount = SceneDescriptors + FrameDescriptors;
constexpr uint32_t TransportBatch = 64, LinearBatch = 65536;
constexpr uint64_t MaximumPathCacheBytes = 512ull * 1024 * 1024;
enum class Stage
{
    Initialize,
    LightWalk,
    CameraWalk,
    BuildGrid,
    SortIndex,
    CameraEstimate,
    CameraMerge,
    SplatEstimate,
    Commit,
    Count
};
struct ImageDomain
{
    uint32_t Width, Height, FullWidth, FullHeight, TileX, TileY;
};
struct DispatchLimits
{
    uint32_t Transport = TransportBatch, Linear = LinearBatch;
};
struct DispatchPlan
{
    WalkControls Walk{};
    CouplingControls Coupling{};
    FrameControls Frame{};
    ImageDomain Image{};
    DispatchLimits Limits{};
    uint32_t Passes = 0, Pixels = 0, MetadataCount = 0, VertexCount = 0;
    uint64_t VertexBytes = 0, MetadataBytes = 0, IndexBytes = 0, SplatBytes = 0;
    uint64_t FilmBytes = 0, StatisticsBytes = 0, DeviceBytes = 0, LightCacheBytes = 0;
};
inline bool MakeDispatchPlan(ImageDomain image, uint32_t passes, uint32_t maximumVertices,
                             uint32_t lightPaths, uint32_t seed, Vector center, float worldRadius,
                             uint32_t emitters, double initialRadius, double alpha, uint32_t heuristic,
                             bool nee, DispatchLimits limits, DispatchPlan& output, std::string& error)
{
    output = {};
    error.clear();
    constexpr uint32_t MaximumIndex = std::numeric_limits<uint32_t>::max();
    constexpr uint32_t MaximumGroups = 65535;
    const uint64_t fullPixels = uint64_t(image.FullWidth) * image.FullHeight;
    if (!image.Width || !image.Height || !image.FullWidth || !image.FullHeight ||
        image.TileX >= image.FullWidth || image.TileY >= image.FullHeight ||
        image.Width > image.FullWidth - image.TileX || image.Height > image.FullHeight - image.TileY ||
        fullPixels >= MaximumIndex || !passes || maximumVertices < 3 || maximumVertices > MaximumVertices ||
        !lightPaths || !limits.Transport || !limits.Linear ||
        limits.Transport > MaximumGroups * WalkThreads || limits.Linear > MaximumGroups * WalkThreads ||
        !(worldRadius >= 0) || !std::isfinite(worldRadius) || !std::isfinite(center.X) ||
        !std::isfinite(center.Y) || !std::isfinite(center.Z) || !(initialRadius >= 0) ||
        !std::isfinite(initialRadius) || !(alpha > 0 && alpha < 1) || heuristic > PowerMis)
    {
        error = "Invalid Compute VCM image, population, depth, radius or dispatch limits.";
        return false;
    }
    constexpr uint64_t PathBytes = uint64_t(MaximumVertices) * sizeof(PathVertex) + sizeof(Path);
    const uint64_t metadata = uint64_t(lightPaths) + limits.Transport + 1;
    const uint64_t vertices = metadata * MaximumVertices;
    if (uint64_t(lightPaths) * PathBytes > MaximumPathCacheBytes || vertices > MaximumIndex)
    {
        error = "Compute VCM light paths exceed the 512 MiB path-cache budget; reduce light paths per pass.";
        return false;
    }
    const uint64_t slots = uint64_t(lightPaths) * (maximumVertices - 2);
    uint64_t capacity = 1;
    while (capacity < slots)
        capacity *= 2;
    if (capacity > MaximumIndex)
    {
        error = "Compute VCM spatial index exceeds the structured-buffer element range.";
        return false;
    }
    DispatchPlan plan;
    plan.Image = image;
    plan.Limits = limits;
    plan.Passes = passes;
    plan.Pixels = image.Width * image.Height;
    plan.MetadataCount = uint32_t(metadata);
    plan.VertexCount = uint32_t(vertices);
    plan.Walk.EmissionCenter = center;
    plan.Walk.EmissionRadius = worldRadius;
    plan.Walk.EmitterCount = emitters;
    plan.Walk.Seed = seed;
    plan.Walk.VertexLimit = maximumVertices;
    plan.Coupling.MergeRadius = initialRadius;
    plan.Coupling.MergingSamples = lightPaths;
    plan.Coupling.LightTracingSamples = double(lightPaths) / double(fullPixels);
    plan.Coupling.MaximumVertices = maximumVertices;
    plan.Coupling.Heuristic = heuristic;
    plan.Coupling.NextEventEstimation = nee;
    plan.Frame.InitialRadius = initialRadius;
    plan.Frame.Alpha = alpha;
    plan.Frame.LightPathCount = lightPaths;
    plan.Frame.CameraPathBase = lightPaths;
    plan.Frame.LensPathIndex = plan.MetadataCount - 1;
    plan.Frame.IndexCapacity = uint32_t(capacity);
    plan.VertexBytes = vertices * sizeof(PathVertex);
    plan.MetadataBytes = metadata * sizeof(Path);
    plan.IndexBytes = capacity * sizeof(IndexEntry);
    plan.SplatBytes = capacity * sizeof(Splat);
    plan.FilmBytes = uint64_t(plan.Pixels) * sizeof(FilmPixel);
    // Walk counters are addressed by metadata output slot. Frame estimators use
    // the first Transport rows, so all launches fit one bounded readback window.
    plan.StatisticsBytes = metadata * sizeof(Statistics);
    plan.LightCacheBytes = uint64_t(lightPaths) * PathBytes;
    plan.DeviceBytes = plan.VertexBytes + plan.MetadataBytes + plan.IndexBytes + plan.SplatBytes +
                       plan.FilmBytes + plan.StatisticsBytes + sizeof(uint32_t);
    output = plan;
    return true;
}
struct Dispatch
{
    WalkControls Walk;
    FrameControls Frame;
    Stage Program;
    uint32_t Threads, StatisticsBegin, StatisticsCount;
};
template <class Cancel, class Execute>
bool RunDispatchPlan(const DispatchPlan& plan, Cancel cancelled, Execute execute)
{
    WalkControls walk = plan.Walk;
    FrameControls frame = plan.Frame;
    const auto launch = [&](Stage stage, uint32_t count, uint32_t statisticsBegin, uint32_t statisticsCount) {
        if (cancelled())
            return false;
        const Dispatch command{walk, frame, stage, count, statisticsBegin, statisticsCount};
        // Execute retires its submitted commands before returning, even if the
        // cancellation request arrives while the GPU is working.
        return execute(command) && !cancelled();
    };
    const auto linear = [&](Stage stage, uint32_t total, uint32_t batch, bool counters) {
        for (uint32_t begin = 0; begin < total;)
        {
            frame.WorkBegin = begin;
            frame.WorkCount = std::min(batch, total - begin);
            if (!launch(stage, frame.WorkCount, 0, counters ? frame.WorkCount : 0))
                return false;
            begin += frame.WorkCount;
        }
        return true;
    };
    const auto sort = [&]() {
        for (uint64_t span = 2; span <= frame.IndexCapacity; span *= 2)
            for (uint32_t stride = uint32_t(span / 2); stride; stride /= 2)
            {
                frame.SortSpan = uint32_t(span);
                frame.SortStride = stride;
                if (!linear(Stage::SortIndex, frame.IndexCapacity, plan.Limits.Linear, false))
                    return false;
            }
        return true;
    };
    if (!linear(Stage::Initialize, plan.Pixels, plan.Limits.Linear, false))
        return false;
    for (uint32_t pass = 0; pass < plan.Passes; ++pass)
    {
        walk.Pass = pass;
        for (uint32_t begin = 0; begin < frame.LightPathCount;)
        {
            walk.PathBegin = begin;
            walk.PathCount = std::min(plan.Limits.Transport, frame.LightPathCount - begin);
            walk.OutputOffset = frame.LightPathBase + begin;
            if (!launch(Stage::LightWalk, walk.PathCount, walk.OutputOffset, walk.PathCount))
                return false;
            begin += walk.PathCount;
        }
        if (!linear(Stage::BuildGrid, frame.IndexCapacity, plan.Limits.Linear, false) || !sort())
            return false;
        for (uint32_t begin = 0; begin < plan.Pixels;)
        {
            walk.PathBegin = begin;
            walk.PathCount = std::min(plan.Limits.Transport, plan.Pixels - begin);
            walk.OutputOffset = frame.CameraPathBase;
            if (!launch(Stage::CameraWalk, walk.PathCount, walk.OutputOffset, walk.PathCount))
                return false;
            frame.CameraPixelBegin = begin;
            frame.WorkBegin = begin;
            frame.WorkCount = walk.PathCount;
            // Separate connection and merge kernels avoid a fused traversal/MIS
            // program. Interleave them by prefix to retain the original double
            // accumulation order; every prefix and every global photon remains.
            for (frame.CameraPrefix = 2; frame.CameraPrefix <= plan.Coupling.MaximumVertices;
                 ++frame.CameraPrefix)
                if (!launch(Stage::CameraEstimate, frame.WorkCount, 0, frame.WorkCount) ||
                    !launch(Stage::CameraMerge, frame.WorkCount, 0, frame.WorkCount))
                    return false;
            begin += walk.PathCount;
        }
        if (!linear(Stage::SplatEstimate, frame.IndexCapacity, plan.Limits.Transport, true) || !sort() ||
            !linear(Stage::Commit, plan.Pixels, plan.Limits.Linear, false))
            return false;
    }
    return true;
}
} // namespace Arvo::RayTracing::ComputeVcm
