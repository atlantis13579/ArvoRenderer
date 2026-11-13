#include "VertexConnectionMergingFrame.h"
#include <algorithm>
#include <cmath>

namespace Arvo::RayTracing::MetalVcm
{
namespace
{
constexpr uint32_t SpatialSort = 0, SplatSort = 1, PendingCamera = 1, PendingFilm = 2;
constexpr uint32_t ConnectionAndMergingFamilies = 3, RouletteFlag = 4;
constexpr uint32_t GridNeighbors = 27, FrameWorkBudget = 8;
constexpr double EmptyDomainRadius = 1;
Number EncodeRadius(double value)
{
    int exponent = 0;
    double significand = std::frexp(value, &exponent);
    float high = float(significand), low = float(significand - double(high));
    // A rounded significand can reach one; restore the shader's canonical range.
    if (high >= 1)
    {
        high *= 0.5f;
        low *= 0.5f;
        ++exponent;
    }
    return {high, low, exponent, 0};
}
} // namespace
bool PrepareFrame(const FrameRequest& r, FramePlan& output, std::string& error)
{
    output = {};
    if (!ValidateVertexConnectionMergingOptions(r.Options, error))
        return false;
    const auto& c = r.Scene;
    constexpr uint64_t IndexLimit = std::numeric_limits<uint32_t>::max();
    if (!c.FullWidth || !c.FullHeight || uint64_t(c.FullWidth) * c.FullHeight > IndexLimit || !r.Width ||
        !r.Height || r.TileX > c.FullWidth || r.Width > c.FullWidth - r.TileX || r.TileY > c.FullHeight ||
        r.Height > c.FullHeight - r.TileY || !c.TotalSamples || c.MaxReflections > MaximumReflections ||
        (c.TransportFlags & RouletteFlag) || !std::isfinite(c.EmissionRadius) || c.EmissionRadius < 0 ||
        !r.TransportLimit || !r.LinearLimit || r.TransportLimit > TransportBatch ||
        r.LinearLimit > LinearBatch || uint32_t(r.Mis) > uint32_t(MisHeuristic::Power))
    {
        error = "Metal VCM frame dimensions, depth, transport or dispatch limits are invalid.";
        return false;
    }
    FramePlan plan;
    plan.Scene = c;
    plan.Scene.Seed = r.Options.Seed ^ c.SamplingSeed;
    plan.VertexStride = c.MaxReflections + 2;
    plan.TransportLimit = r.TransportLimit;
    plan.LinearLimit = r.LinearLimit;
    plan.Alpha = r.Options.Alpha;
    // The radius derives from the entire emission domain, never a camera crop.
    plan.InitialRadius =
        (c.EmissionRadius > 0 ? double(c.EmissionRadius) : EmptyDomainRadius) * r.Options.InitialRadiusScale;
    auto& f = plan.Frame;
    f.TileX = r.TileX;
    f.TileY = r.TileY;
    f.Width = r.Width;
    f.Height = r.Height;
    f.LightCount = r.Options.LightPathsPerPass;
    f.SlotsPerLight = c.MaxReflections + 1;
    f.CameraBegin = f.LightCount;
    f.PixelCapacity = r.Width * r.Height;
    f.StatisticsCapacity = r.TransportLimit;
    uint64_t splats = uint64_t(f.LightCount) * f.SlotsPerLight, index = 1;
    while (index < splats)
        index *= 2;
    uint64_t paths = uint64_t(f.LightCount) + r.TransportLimit, vertices = paths * plan.VertexStride;
    if (index > IndexLimit || vertices > IndexLimit)
    {
        error = "Metal VCM path or index storage exceeds shader addressing.";
        return false;
    }
    f.SplatCapacity = uint32_t(splats);
    f.IndexCapacity = uint32_t(index);
    auto& coupling = plan.Coupling;
    coupling.PathCapacity = uint32_t(paths);
    coupling.VertexCapacity = uint32_t(vertices);
    coupling.SourceCapacity = c.SourceCount;
    coupling.Heuristic = uint32_t(r.Mis);
    coupling.Families = ConnectionAndMergingFamilies;
    coupling.NextEventEstimation = r.NextEventEstimation ? 1 : 0;
    const std::array<uint64_t, StorageCount> bytes{vertices * sizeof(WalkVertex),
                                                   paths * sizeof(Walk),
                                                   uint64_t(r.TransportLimit) * sizeof(WalkStatistics),
                                                   index * sizeof(IndexEntry),
                                                   uint64_t(f.PixelCapacity) * sizeof(PixelState),
                                                   splats * sizeof(Splat),
                                                   uint64_t(f.PixelCapacity) * sizeof(FilmPixel),
                                                   sizeof(uint32_t),
                                                   sizeof(uint32_t)};
    for (uint32_t i = 0; i < StorageCount; ++i)
    {
        if (bytes[i] > r.MaximumBufferBytes || bytes[i] > std::numeric_limits<size_t>::max())
        {
            error = "Metal VCM storage exceeds the device's maximum buffer length.";
            return false;
        }
        plan.Bytes[i] = size_t(bytes[i]);
    }
    output = plan;
    return true;
}
bool ExecuteFrame(const FramePlan& plan, const FrameCallbacks& callbacks, std::string& error)
{
    if (!callbacks.Execute || !callbacks.Publish || !plan.TransportLimit || !plan.LinearLimit ||
        !plan.Frame.IndexCapacity || !plan.Scene.TotalSamples || !plan.Frame.Width)
    {
        error = "Metal VCM frame execution requires a prepared plan and callbacks.";
        return false;
    }
    Dispatch command;
    command.Frame = plan.Frame;
    command.Coupling = plan.Coupling;
    auto cancelled = [&] {
        if (callbacks.Cancelled && callbacks.Cancelled())
        {
            error = "Metal VCM frame was cancelled.";
            return true;
        }
        return false;
    };
    auto invoke = [&](Stage stage, uint32_t lanes, uint32_t statistics, uint32_t& pending) {
        if (cancelled())
            return false;
        command.Kernel = stage;
        command.Lanes = lanes;
        command.StatisticsRows = statistics;
        pending = 0;
        if (!callbacks.Execute(command, pending, error) || cancelled())
            return false;
        uint32_t allowed = stage == Stage::CameraFrame  ? PendingCamera
                           : stage == Stage::CommitFilm ? PendingFilm
                                                        : 0;
        if (pending & ~allowed)
        {
            error = "Metal VCM returned an unexpected continuation flag.";
            return false;
        }
        return true;
    };
    auto linear = [&](Stage stage, uint32_t count, uint32_t limit) {
        for (uint32_t first = 0; first < count;)
        {
            auto& f = command.Frame;
            f.WorkBegin = first;
            f.WorkCount = std::min(limit, count - first);
            uint32_t pending = 0;
            // One pixel can receive every lens splat. A finite worst-case bound
            // detects a stuck continuation without truncating a dense bucket.
            uint64_t maximum = (uint64_t(f.SplatCapacity) + 1 + FrameWorkBudget - 1) / FrameWorkBudget + 1;
            do
            {
                if (!invoke(stage, f.WorkCount, stage == Stage::ProjectLightPaths ? f.WorkCount : 0, pending))
                    return false;
                if (pending && !--maximum)
                {
                    error = "Metal VCM film continuation did not finish.";
                    return false;
                }
            } while (pending);
            first += f.WorkCount;
        }
        return true;
    };
    auto sort = [&](uint32_t kind) {
        auto& f = command.Frame;
        f.SortKind = kind;
        for (uint64_t width = 2; width <= f.IndexCapacity; width *= 2)
        {
            f.SortWidth = uint32_t(width);
            for (f.SortStride = f.SortWidth / 2; f.SortStride; f.SortStride /= 2)
                if (!linear(Stage::SortIndex, f.IndexCapacity, plan.LinearLimit))
                    return false;
        }
        return true;
    };
    auto walks = [&](Stage stage, uint32_t identity, uint32_t count, uint32_t begin) {
        command.Walks = {identity,
                         count,
                         command.Frame.Pass,
                         begin,
                         begin * plan.VertexStride,
                         plan.VertexStride,
                         plan.Coupling.PathCapacity,
                         plan.Coupling.VertexCapacity,
                         plan.TransportLimit,
                         plan.Scene.SourceCount,
                         plan.Frame.LightCount,
                         0};
        uint32_t pending = 0;
        return invoke(stage, count, count, pending);
    };
    if (!linear(Stage::ClearFilm, plan.Frame.PixelCapacity, plan.LinearLimit))
        return false;
    for (uint32_t pass = 0; pass < plan.Scene.TotalSamples; ++pass)
    {
        auto& f = command.Frame;
        f.Pass = pass;
        f.CameraCount = 0;
        command.Coupling.Radius =
            EncodeRadius(plan.InitialRadius / std::pow(double(pass) + 1, (1 - plan.Alpha) / 2));
        for (uint32_t first = 0; first < f.LightCount;)
        {
            uint32_t count = std::min(plan.TransportLimit, f.LightCount - first);
            if (!walks(Stage::LightWalks, first, count, f.LightBegin + first))
                return false;
            first += count;
        }
        if (!linear(Stage::BuildGrid, f.IndexCapacity, plan.LinearLimit) || !sort(SpatialSort))
            return false;
        for (uint32_t y = 0; y < f.Height; ++y)
            for (uint32_t x = 0; x < f.Width;)
            {
                f.CameraCount = std::min(plan.TransportLimit, f.Width - x);
                uint32_t identity = (f.TileY + y) * plan.Scene.FullWidth + f.TileX + x;
                if (!walks(Stage::CameraWalks, identity, f.CameraCount, f.CameraBegin))
                    return false;
                // At most every camera surface visits every photon plus 27 cell
                // transitions. Include the full s/t connection rectangle.
                uint64_t vertices = plan.VertexStride;
                uint64_t steps =
                    (vertices + 1) * vertices + vertices * (uint64_t(f.SplatCapacity) + GridNeighbors + 2);
                uint64_t maximum = (steps + FrameWorkBudget - 1) / FrameWorkBudget + 1;
                uint32_t pending = 0;
                do
                {
                    if (!invoke(Stage::CameraFrame, f.CameraCount, f.CameraCount, pending))
                        return false;
                    if (pending && !--maximum)
                    {
                        error = "Metal VCM camera continuation did not finish.";
                        return false;
                    }
                } while (pending);
                x += f.CameraCount;
            }
        f.CameraCount = 0;
        if (!linear(Stage::ProjectLightPaths, f.IndexCapacity, plan.TransportLimit) || !sort(SplatSort) ||
            !linear(Stage::CommitFilm, f.PixelCapacity, plan.LinearLimit) || cancelled() ||
            !callbacks.Publish(pass + 1, error) || cancelled())
            return false;
    }
    return true;
}
} // namespace Arvo::RayTracing::MetalVcm
