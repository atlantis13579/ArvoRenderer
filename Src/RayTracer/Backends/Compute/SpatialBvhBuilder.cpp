#include "SpatialBvhBuilder.h"
#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>

namespace Arvo::RayTracing::ComputeAcceleration
{
namespace
{
// Native Compute SBVH builder: binned object SAH competes with clipped spatial
// references and a complete leaf. The existing Compute BVH shader consumes the
// resulting escape links; no other backend or tree builder participates.
// Box-only clipping is conservative for triangles and analytic spheres.
// Stich et al., "Spatial Splits in Bounding Volume Hierarchies" (2009), sections 4.1-4.2.
// https://www.nvidia.com/docs/IO/77714/sbvh.pdf
constexpr uint32_t BinCount = 16, LeafLimit = 8, DepthLimit = 48, ReferenceFactor = 8;
constexpr uint32_t CancellationPeriod = 1024;
constexpr uint64_t NodeLimit = uint64_t(UINT32_MAX) / sizeof(SpatialNode);
constexpr double RelativeTraversalCost = 1.0;
float Get(Float3 v, uint32_t axis)
{
    return axis == 0 ? v.X : axis == 1 ? v.Y : v.Z;
}
void Set(Float3 &v, uint32_t axis, float x)
{
    if (axis == 0)
        v.X = x;
    else if (axis == 1)
        v.Y = x;
    else
        v.Z = x;
}
struct Bounds
{
    Float3 Low = {FLT_MAX, FLT_MAX, FLT_MAX}, High = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
    void Add(Float3 low, Float3 high)
    {
        Low = {std::min(Low.X, low.X), std::min(Low.Y, low.Y), std::min(Low.Z, low.Z)};
        High = {std::max(High.X, high.X), std::max(High.Y, high.Y), std::max(High.Z, high.Z)};
    }
    void Add(const Bounds &b) { Add(b.Low, b.High); }
    double Area() const
    {
        const double x = double(High.X) - Low.X, y = double(High.Y) - Low.Y, z = double(High.Z) - Low.Z;
        return 2 * (x * y + y * z + z * x);
    }
};
struct Piece
{
    Bounds Box;
    uint32_t Id;
};
double Center(const Piece &p, uint32_t axis)
{
    return (double(Get(p.Box.Low, axis)) + Get(p.Box.High, axis)) * 0.5;
}
struct Branch
{
    Bounds Box;
    uint32_t Children = 0, First = 0, Count = 0;
};
struct Work
{
    std::vector<Piece> Pieces;
    uint32_t Node, Depth;
};
struct Split
{
    bool Spatial = false, Valid = false;
    uint32_t Axis = 0, Bin = 0;
    float Plane = 0;
    double Cost = 0;
};
struct Bin
{
    Bounds Box;
    uint32_t Count = 0;
};

class Builder
{
    const std::function<bool()> &Cancelled;
    SpatialTree &Result;
    std::string &Error;
    std::vector<Branch> Branches;
    uint64_t References = 0, ReferenceLimit = 0;
    size_t WorkUnits = 0;
    bool Stop()
    {
        if (!Cancelled || !Cancelled())
            return false;
        Error = "Compute spatial BVH construction was cancelled";
        return true;
    }
    bool Poll() { return (++WorkUnits % CancellationPeriod) == 0 && Stop(); }
    bool Flatten(uint32_t source)
    {
        if (Poll())
            return false;
        const Branch &branch = Branches[source];
        const uint32_t index = static_cast<uint32_t>(Result.Nodes.size());
        Result.Nodes.push_back({branch.Box.Low, branch.First, branch.Box.High, branch.Count});
        if (branch.Count == 0)
        {
            if (!Flatten(branch.Children) || !Flatten(branch.Children + 1))
                return false;
            Result.Nodes[index].FirstOrEscape = static_cast<uint32_t>(Result.Nodes.size());
        }
        return true;
    }

  public:
    Builder(const std::function<bool()> &cancelled, SpatialTree &tree, std::string &error)
        : Cancelled(cancelled), Result(tree), Error(error)
    {
    }
    bool Run(const std::vector<SpatialPrimitive> &input)
    {
        if (Stop())
            return false;
        if (input.size() >= UINT32_MAX)
        {
            Error = "Compute spatial BVH exceeds primitive indexing limits";
            return false;
        }
        if (input.empty())
            return true;
        std::vector<Piece> pieces;
        pieces.reserve(input.size());
        for (const auto &primitive : input)
        {
            if (Poll())
                return false;
            for (uint32_t axis = 0; axis < 3; ++axis)
                if (!std::isfinite(Get(primitive.Minimum, axis)) ||
                    !std::isfinite(Get(primitive.Maximum, axis)) ||
                    Get(primitive.Minimum, axis) > Get(primitive.Maximum, axis))
                {
                    Error = "Invalid Compute spatial BVH primitive bounds";
                    return false;
                }
            pieces.push_back({{primitive.Minimum, primitive.Maximum}, primitive.Reference});
        }
        References = input.size();
        ReferenceLimit = std::min<uint64_t>(UINT32_MAX, input.size() * uint64_t(ReferenceFactor));
        std::vector<Work> pending;
        pending.push_back({std::move(pieces), 0, 0});
        Branches.emplace_back();
        while (!pending.empty())
        {
            if (Stop())
                return false;
            Work work = std::move(pending.back());
            pending.pop_back();
            Bounds box;
            std::array<double, 3> low, high;
            low.fill(DBL_MAX);
            high.fill(-DBL_MAX);
            for (const Piece &piece : work.Pieces)
            {
                if (Poll())
                    return false;
                box.Add(piece.Box);
                for (uint32_t axis = 0; axis < 3; ++axis)
                {
                    const double center = Center(piece, axis);
                    low[axis] = std::min(low[axis], center);
                    high[axis] = std::max(high[axis], center);
                }
            }
            Branches[work.Node].Box = box;
            const double area = box.Area();
            Split split;
            split.Cost = area * work.Pieces.size();
            const bool refine = work.Pieces.size() > LeafLimit && work.Depth < DepthLimit && area > 0 &&
                                Branches.size() + 2 <= NodeLimit;
            if (refine)
                for (uint32_t axis = 0; axis < 3; ++axis)
                {
                    const double extent = high[axis] - low[axis];
                    if (extent > 0)
                    {
                        std::array<Bin, BinCount> bins{};
                        for (const Piece &piece : work.Pieces)
                        {
                            if (Poll())
                                return false;
                            const uint32_t bin =
                                std::min(uint32_t((Center(piece, axis) - low[axis]) / extent * BinCount),
                                         BinCount - 1);
                            ++bins[bin].Count;
                            bins[bin].Box.Add(piece.Box);
                        }
                        std::array<Bounds, BinCount> suffix;
                        std::array<uint32_t, BinCount> suffixCount{};
                        Bounds right;
                        uint32_t rightCount = 0;
                        for (uint32_t bin = BinCount; bin-- > 0;)
                        {
                            if (bins[bin].Count)
                                right.Add(bins[bin].Box);
                            rightCount += bins[bin].Count;
                            suffix[bin] = right;
                            suffixCount[bin] = rightCount;
                        }
                        Bounds left;
                        uint32_t leftCount = 0;
                        for (uint32_t bin = 0; bin + 1 < BinCount; ++bin)
                        {
                            if (bins[bin].Count)
                                left.Add(bins[bin].Box);
                            leftCount += bins[bin].Count;
                            if (!leftCount || !suffixCount[bin + 1])
                                continue;
                            const double cost = RelativeTraversalCost * area + left.Area() * leftCount +
                                                suffix[bin + 1].Area() * suffixCount[bin + 1];
                            if (cost < split.Cost)
                                split = {false, true, axis, bin, 0, cost};
                        }
                    }
                    const double minimum = Get(box.Low, axis), span = double(Get(box.High, axis)) - minimum;
                    for (uint32_t bin = 1; bin < BinCount && span > 0; ++bin)
                    {
                        const float plane = float(minimum + span * bin / BinCount);
                        if (!(plane > minimum && plane < Get(box.High, axis)))
                            continue;
                        Bounds left, right;
                        uint64_t leftCount = 0, rightCount = 0;
                        for (const Piece &piece : work.Pieces)
                        {
                            if (Poll())
                                return false;
                            if (Get(piece.Box.High, axis) <= plane)
                            {
                                left.Add(piece.Box);
                                ++leftCount;
                            }
                            else if (Get(piece.Box.Low, axis) >= plane)
                            {
                                right.Add(piece.Box);
                                ++rightCount;
                            }
                            else
                            {
                                Bounds lower = piece.Box, upper = piece.Box;
                                Set(lower.High, axis, plane);
                                Set(upper.Low, axis, plane);
                                left.Add(lower);
                                right.Add(upper);
                                ++leftCount;
                                ++rightCount;
                            }
                        }
                        if (!leftCount || !rightCount ||
                            leftCount + rightCount - work.Pieces.size() > ReferenceLimit - References)
                            continue;
                        const double cost = RelativeTraversalCost * area + left.Area() * leftCount +
                                            right.Area() * rightCount;
                        if (cost < split.Cost)
                            split = {true, true, axis, bin, plane, cost};
                    }
                }
            if (!split.Valid)
            {
                Branches[work.Node].First = static_cast<uint32_t>(Result.References.size());
                Branches[work.Node].Count = static_cast<uint32_t>(work.Pieces.size());
                for (const Piece &piece : work.Pieces)
                {
                    if (Poll())
                        return false;
                    Result.References.push_back(piece.Id);
                }
                continue;
            }
            std::vector<Piece> left, right;
            for (const Piece &piece : work.Pieces)
            {
                if (Poll())
                    return false;
                if (!split.Spatial)
                {
                    const uint32_t bin = std::min(uint32_t((Center(piece, split.Axis) - low[split.Axis]) /
                                                           (high[split.Axis] - low[split.Axis]) * BinCount),
                                                  BinCount - 1);
                    (bin <= split.Bin ? left : right).push_back(piece);
                }
                else if (Get(piece.Box.High, split.Axis) <= split.Plane)
                    left.push_back(piece);
                else if (Get(piece.Box.Low, split.Axis) >= split.Plane)
                    right.push_back(piece);
                else
                {
                    Piece lower = piece, upper = piece;
                    Set(lower.Box.High, split.Axis, split.Plane);
                    Set(upper.Box.Low, split.Axis, split.Plane);
                    left.push_back(lower);
                    right.push_back(upper);
                }
            }
            References += left.size() + right.size() - work.Pieces.size();
            if (split.Spatial)
                ++Result.SpatialSplits;
            const uint32_t children = static_cast<uint32_t>(Branches.size());
            Branches[work.Node].Children = children;
            Branches.resize(Branches.size() + 2);
            pending.push_back({std::move(right), children + 1, work.Depth + 1});
            pending.push_back({std::move(left), children, work.Depth + 1});
        }
        if (Stop())
            return false;
        return Flatten(0) && !Stop();
    }
};
} // namespace

bool BuildSpatialBvh(const std::vector<SpatialPrimitive> &primitives, const std::function<bool()> &cancelled,
                     SpatialTree &tree, std::string &error)
{
    tree = {};
    if (Builder(cancelled, tree, error).Run(primitives))
        return true;
    tree = {};
    return false;
}
} // namespace Arvo::RayTracing::ComputeAcceleration
