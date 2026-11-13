#pragma once
#include "VertexConnectionMergingLayout.h"
#include <algorithm>
#include <limits>
#include <string>

namespace Arvo::RayTracing::DxrVcm
{
// DXR dispatch ownership, independently authored for its resumable shader.
// This header contains no transport evaluation. A fenced callback binds the
// offline native library at each stage.
constexpr uint32_t TransportBatch = 64, LinearBatch = 65536;
constexpr uint32_t NeighborhoodSide = 3;
constexpr uint32_t NeighborhoodCells = NeighborhoodSide * NeighborhoodSide * NeighborhoodSide;
struct DispatchPlan
{
    uint32_t Pixels = 0, FullPixels = 0, LightPaths = 0, VertexLimit = 0;
    uint32_t CameraCapacity = 0, CameraBase = 0, LensPath = 0, PathCapacity = 0, VertexCapacity = 0;
    uint32_t PhotonSlots = 0, IndexCapacity = 0;
    uint64_t MergeIterations = 0, CommitIterations = 0;
};
inline bool PlanDispatch(uint32_t width, uint32_t height, uint32_t fullWidth, uint32_t fullHeight,
                         uint32_t lightPaths, uint32_t reflections, DispatchPlan& plan, std::string& error)
{
    constexpr uint64_t InvalidIndex = std::numeric_limits<uint32_t>::max();
    constexpr uint64_t LargestSortingCapacity = uint64_t(1) << 31;
    const uint64_t pixels = uint64_t(width) * height, fullPixels = uint64_t(fullWidth) * fullHeight;
    if (pixels == 0 || width > fullWidth || height > fullHeight || fullPixels >= InvalidIndex ||
        reflections > MaximumReflections)
    {
        error = "DXR VCM dimensions or reflection count exceed shader limits.";
        return false;
    }
    DispatchPlan result;
    result.Pixels = uint32_t(pixels);
    result.FullPixels = uint32_t(fullPixels);
    result.LightPaths = lightPaths;
    result.VertexLimit = reflections + (MaximumVertices - MaximumReflections);
    result.CameraCapacity = std::min(TransportBatch, result.Pixels);
    const uint64_t paths = uint64_t(lightPaths) + result.CameraCapacity + 1;
    const uint64_t slots = uint64_t(lightPaths) * (result.VertexLimit - 2);
    if (paths > InvalidIndex / MaximumVertices || slots > LargestSortingCapacity)
    {
        error = "DXR VCM path/index allocation exceeds 32-bit shader addressing.";
        return false;
    }
    result.CameraBase = lightPaths;
    result.LensPath = lightPaths + result.CameraCapacity;
    result.PathCapacity = uint32_t(paths);
    result.VertexCapacity = result.PathCapacity * MaximumVertices;
    result.PhotonSlots = uint32_t(slots);
    result.IndexCapacity = 1;
    while (result.IndexCapacity < slots)
        result.IndexCapacity *= 2;
    // A worst-case hash collision still terminates: each candidate, empty
    // neighbor transition, prefix transition and final completion costs a step.
    const uint64_t mergeSteps = uint64_t(result.VertexLimit - 2) * (NeighborhoodCells * (slots + 1) + 1) + 1;
    result.MergeIterations = (mergeSteps + MergeWorkBudget - 1) / MergeWorkBudget;
    result.CommitIterations = (slots + 1 + SplatReductionBudget - 1) / SplatReductionBudget;
    plan = result;
    return true;
}
struct DispatchStep
{
    WalkControls Walk;
    FrameControls Frame;
    uint32_t Count;
    bool ReadsStatistics;
};
inline bool UsesStatistics(uint32_t stage)
{
    return stage == CameraStage || stage == LightStage || stage == EstimateStage || stage == MergeStage ||
           stage == SplatStage || stage == CommitStage;
}
template <class Submit>
bool RunDispatches(const DispatchPlan& plan, uint32_t passes, WalkControls walk, FrameControls frame,
                   Submit&& submit, std::string& error)
{
    if (passes == 0)
    {
        error = "DXR VCM needs at least one complete pass.";
        return false;
    }
    walk.PathCapacity = plan.PathCapacity;
    walk.VertexCapacity = plan.VertexCapacity;
    walk.StatisticsCapacity = TransportBatch;
    walk.VertexLimit = plan.VertexLimit;
    frame.LightPathBase = 0;
    frame.LightPathCount = plan.LightPaths;
    frame.CameraPathBase = plan.CameraBase;
    frame.LensPathIndex = plan.LensPath;
    frame.IndexCapacity = frame.BucketCount = plan.IndexCapacity;
    auto issue = [&](uint32_t stage, uint32_t begin, uint32_t count, uint32_t output, uint32_t& unfinished) {
        walk.Stage = stage;
        walk.PathBegin = begin;
        walk.PathCount = count;
        walk.OutputBegin = output;
        frame.WorkBegin = begin;
        frame.WorkCount = count;
        const DispatchStep step{walk, frame, count, UsesStatistics(stage)};
        unfinished = 0;
        if (!submit(step, unfinished))
            return false;
        if (unfinished > count)
        {
            error = "DXR VCM returned an invalid unfinished-lane count.";
            return false;
        }
        return true;
    };
    auto linear = [&](uint32_t stage, uint32_t total, uint32_t batch) {
        for (uint32_t begin = 0; begin < total;)
        {
            const uint32_t count = std::min(batch, total - begin);
            uint32_t unfinished;
            if (!issue(stage, begin, count, stage == LightStage ? begin : 0, unfinished))
                return false;
            begin += count;
        }
        return true;
    };
    auto sort = [&] {
        for (uint64_t span = 2; span <= plan.IndexCapacity; span *= 2)
        {
            frame.SortSpan = uint32_t(span);
            for (uint32_t distance = uint32_t(span / 2); distance > 0; distance /= 2)
            {
                frame.SortDistance = distance;
                if (!linear(SortStage, plan.IndexCapacity, LinearBatch))
                    return false;
            }
        }
        return true;
    };
    auto finish = [&](uint32_t stage, uint32_t begin, uint32_t count, uint64_t limit) {
        for (uint64_t attempt = 0; attempt < limit; ++attempt)
        {
            uint32_t unfinished;
            if (!issue(stage, begin, count, 0, unfinished))
                return false;
            if (unfinished == 0)
                return true;
        }
        error = "DXR VCM failed to finish a bounded merge or film reduction.";
        return false;
    };
    walk.PassIndex = 0;
    if (!linear(InitializeStage, plan.Pixels, LinearBatch))
        return false;
    for (uint32_t pass = 0; pass < passes; ++pass)
    {
        walk.PassIndex = pass;
        if (!linear(LightStage, plan.LightPaths, TransportBatch) ||
            !linear(GridStage, plan.IndexCapacity, LinearBatch) || !sort())
            return false;
        for (uint32_t pixel = 0; pixel < plan.Pixels;)
        {
            const uint32_t count = std::min(plan.CameraCapacity, plan.Pixels - pixel);
            frame.CameraPixelBegin = pixel;
            uint32_t unfinished;
            if (!issue(CameraStage, pixel, count, plan.CameraBase, unfinished) ||
                !issue(EstimateStage, 0, count, 0, unfinished) ||
                !finish(MergeStage, 0, count, plan.MergeIterations))
                return false;
            pixel += count;
        }
        // No camera cache remains live when splats overwrite the spatial index.
        if (!linear(SplatStage, plan.IndexCapacity, TransportBatch) || !sort())
            return false;
        for (uint32_t pixel = 0; pixel < plan.Pixels;)
        {
            const uint32_t count = std::min(TransportBatch, plan.Pixels - pixel);
            if (!finish(CommitStage, pixel, count, plan.CommitIterations))
                return false;
            pixel += count;
        }
    }
    return true;
}
} // namespace Arvo::RayTracing::DxrVcm
