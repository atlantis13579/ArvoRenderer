#include "BvhBuilder.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace Arvo::RayTracing::ComputeBvh
{
namespace
{
// Compute owns its default BVH construction: binned surface-area heuristic partitions with a median fallback.
// Each primitive belongs to one leaf; depth-first escape links let the native
// Bvh shader skip rejected subtrees without a traversal stack. Spatial splitting
// is a separate builder with a different reference-ownership rule.
// Pharr, Jakob and Humphreys, Physically Based Rendering (4th ed.), Section 7.3.
// https://pbr-book.org/4ed/Primitives_and_Intersection_Acceleration/Bounding_Volume_Hierarchies
constexpr std::uint32_t LeafPrimitiveCount = 8;
constexpr std::uint32_t MaximumBuildDepth = 48;
constexpr std::uint32_t SahBinCount = 16;
constexpr float MinimumCentroidExtent = 1.0e-6f;
constexpr size_t CancellationPeriod = 1024;
struct Vec3
{
    float X = 0.0f;
    float Y = 0.0f;
    float Z = 0.0f;
};

Vec3 Subtract(const Vec3 &left, const Vec3 &right)
{
    return {left.X - right.X, left.Y - right.Y, left.Z - right.Z};
}

struct Bounds
{
    Vec3 Minimum = {std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
                    std::numeric_limits<float>::max()};
    Vec3 Maximum = {-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(),
                    -std::numeric_limits<float>::max()};

    void Expand(const Vec3 &point)
    {
        Minimum.X = std::min(Minimum.X, point.X);
        Minimum.Y = std::min(Minimum.Y, point.Y);
        Minimum.Z = std::min(Minimum.Z, point.Z);
        Maximum.X = std::max(Maximum.X, point.X);
        Maximum.Y = std::max(Maximum.Y, point.Y);
        Maximum.Z = std::max(Maximum.Z, point.Z);
    }

    void Expand(const Bounds &bounds)
    {
        Expand(bounds.Minimum);
        Expand(bounds.Maximum);
    }

    float SurfaceArea() const
    {
        const Vec3 extent = Subtract(Maximum, Minimum);
        return 2.0f * (extent.X * extent.Y + extent.Y * extent.Z + extent.Z * extent.X);
    }
};

float Component(const Vec3 &value, std::uint32_t axis)
{
    return axis == 0 ? value.X : (axis == 1 ? value.Y : value.Z);
}

struct PackedPrimitiveRef
{
    std::uint32_t TypeAndIndex = 0;
};

struct BuildPrimitive
{
    Bounds BoundingBox;
    Vec3 Centroid;
    PackedPrimitiveRef Reference;
};

struct BuildNode
{
    Bounds BoundingBox;
    std::uint32_t Left = 0;
    std::uint32_t Right = 0;
    std::uint32_t FirstPrimitive = 0;
    std::uint32_t PrimitiveCount = 0;

    bool IsLeaf() const { return PrimitiveCount != 0; }
};
using PackedBvhNode = Node;
class Builder
{
    struct Bin
    {
        Bounds BoundingBox;
        std::uint32_t Count = 0;
    };
    const std::function<bool()> &Cancellation;
    std::vector<BuildPrimitive> Primitives;
    std::vector<BuildNode> BuildNodes;
    bool IsCancelled() const { return Cancellation && Cancellation(); }
    std::uint32_t BuildRecursive(std::uint32_t begin, std::uint32_t end, std::uint32_t depth)
    {
        const std::uint32_t nodeIndex = static_cast<std::uint32_t>(BuildNodes.size());
        BuildNodes.emplace_back();
        Bounds primitiveBounds;
        Bounds centroidBounds;
        for (std::uint32_t index = begin; index < end; ++index)
        {
            primitiveBounds.Expand(Primitives[index].BoundingBox);
            centroidBounds.Expand(Primitives[index].Centroid);
        }
        BuildNodes[nodeIndex].BoundingBox = primitiveBounds;
        const std::uint32_t count = end - begin;
        if (count <= LeafPrimitiveCount || depth >= MaximumBuildDepth || IsCancelled())
        {
            BuildNodes[nodeIndex].FirstPrimitive = begin;
            BuildNodes[nodeIndex].PrimitiveCount = count;
            return nodeIndex;
        }

        float bestCost = std::numeric_limits<float>::max();
        std::uint32_t bestAxis = 0;
        std::uint32_t bestSplit = 0;
        bool foundSplit = false;
        for (std::uint32_t axis = 0; axis < 3; ++axis)
        {
            const float minimum = Component(centroidBounds.Minimum, axis);
            const float maximum = Component(centroidBounds.Maximum, axis);
            const float extent = maximum - minimum;
            if (extent <= MinimumCentroidExtent)
                continue;
            std::array<Bin, SahBinCount> bins{};
            for (std::uint32_t index = begin; index < end; ++index)
            {
                const float normalized = (Component(Primitives[index].Centroid, axis) - minimum) / extent;
                const std::uint32_t binIndex =
                    std::min(static_cast<std::uint32_t>(normalized * SahBinCount), SahBinCount - 1);
                ++bins[binIndex].Count;
                bins[binIndex].BoundingBox.Expand(Primitives[index].BoundingBox);
            }
            std::array<Bounds, SahBinCount> leftBounds{};
            std::array<Bounds, SahBinCount> rightBounds{};
            std::array<std::uint32_t, SahBinCount> leftCounts{};
            std::array<std::uint32_t, SahBinCount> rightCounts{};
            Bounds runningLeft;
            std::uint32_t runningLeftCount = 0;
            for (std::uint32_t bin = 0; bin < SahBinCount; ++bin)
            {
                if (bins[bin].Count != 0)
                    runningLeft.Expand(bins[bin].BoundingBox);
                runningLeftCount += bins[bin].Count;
                leftBounds[bin] = runningLeft;
                leftCounts[bin] = runningLeftCount;
            }
            Bounds runningRight;
            std::uint32_t runningRightCount = 0;
            for (std::uint32_t bin = SahBinCount; bin-- > 0;)
            {
                if (bins[bin].Count != 0)
                    runningRight.Expand(bins[bin].BoundingBox);
                runningRightCount += bins[bin].Count;
                rightBounds[bin] = runningRight;
                rightCounts[bin] = runningRightCount;
            }
            for (std::uint32_t split = 0; split + 1 < SahBinCount; ++split)
            {
                if (leftCounts[split] == 0 || rightCounts[split + 1] == 0)
                    continue;
                const float cost = leftBounds[split].SurfaceArea() * leftCounts[split] +
                                   rightBounds[split + 1].SurfaceArea() * rightCounts[split + 1];
                if (cost < bestCost)
                {
                    bestCost = cost;
                    bestAxis = axis;
                    bestSplit = split;
                    foundSplit = true;
                }
            }
        }

        std::uint32_t middle = begin;
        if (foundSplit)
        {
            const float minimum = Component(centroidBounds.Minimum, bestAxis);
            const float extent = Component(centroidBounds.Maximum, bestAxis) - minimum;
            const auto middleIterator = std::partition(
                Primitives.begin() + begin, Primitives.begin() + end,
                [minimum, extent, bestAxis, bestSplit](const BuildPrimitive &primitive) {
                    const float normalized = (Component(primitive.Centroid, bestAxis) - minimum) / extent;
                    const std::uint32_t binIndex =
                        std::min(static_cast<std::uint32_t>(normalized * SahBinCount), SahBinCount - 1);
                    return binIndex <= bestSplit;
                });
            middle = static_cast<std::uint32_t>(middleIterator - Primitives.begin());
        }
        if (middle == begin || middle == end)
        {
            const Vec3 extent = Subtract(centroidBounds.Maximum, centroidBounds.Minimum);
            bestAxis = extent.Y > extent.X ? 1u : 0u;
            if (Component(extent, 2) > Component(extent, bestAxis))
                bestAxis = 2;
            middle = begin + count / 2;
            std::nth_element(
                Primitives.begin() + begin, Primitives.begin() + middle, Primitives.begin() + end,
                [bestAxis](const BuildPrimitive &left, const BuildPrimitive &right) {
                    return Component(left.Centroid, bestAxis) < Component(right.Centroid, bestAxis);
                });
        }
        const std::uint32_t left = BuildRecursive(begin, middle, depth + 1);
        const std::uint32_t right = BuildRecursive(middle, end, depth + 1);
        BuildNodes[nodeIndex].Left = left;
        BuildNodes[nodeIndex].Right = right;
        return nodeIndex;
    }
    void Flatten(std::uint32_t buildIndex, std::vector<PackedBvhNode> &output)
    {
        const std::uint32_t outputIndex = static_cast<std::uint32_t>(output.size());
        output.emplace_back();
        const BuildNode &source = BuildNodes[buildIndex];
        output[outputIndex].MinX = source.BoundingBox.Minimum.X;
        output[outputIndex].MinY = source.BoundingBox.Minimum.Y;
        output[outputIndex].MinZ = source.BoundingBox.Minimum.Z;
        output[outputIndex].MaxX = source.BoundingBox.Maximum.X;
        output[outputIndex].MaxY = source.BoundingBox.Maximum.Y;
        output[outputIndex].MaxZ = source.BoundingBox.Maximum.Z;
        if (source.IsLeaf())
        {
            output[outputIndex].FirstOrEscape = source.FirstPrimitive;
            output[outputIndex].PrimitiveCount = source.PrimitiveCount;
            return;
        }
        Flatten(source.Left, output);
        Flatten(source.Right, output);
        output[outputIndex].FirstOrEscape = static_cast<std::uint32_t>(output.size());
        output[outputIndex].PrimitiveCount = 0;
    }

  public:
    explicit Builder(const std::function<bool()> &cancelled) : Cancellation(cancelled) {}
    bool Run(const std::vector<Primitive> &input, Tree &tree, std::string &error)
    {
        const auto stop = [&] {
            if (!IsCancelled())
                return false;
            error = "Compute BVH construction was cancelled";
            return true;
        };
        if (stop())
            return false;
        constexpr size_t MaximumPrimitiveCount = std::numeric_limits<uint32_t>::max() / 2;
        if (input.size() > MaximumPrimitiveCount)
        {
            error = "Compute BVH exceeds 32-bit node indexing";
            return false;
        }
        if (input.empty())
            return true;
        Primitives.reserve(input.size());
        for (size_t index = 0; index < input.size(); ++index)
        {
            if (index % CancellationPeriod == 0 && stop())
                return false;
            const Primitive &primitive = input[index];
            for (uint32_t axis = 0; axis < 3; ++axis)
                if (!std::isfinite(primitive.Minimum[axis]) || !std::isfinite(primitive.Maximum[axis]) ||
                    !std::isfinite(primitive.Centroid[axis]) ||
                    primitive.Minimum[axis] > primitive.Maximum[axis])
                {
                    error = "Invalid Compute BVH input bounds or centroid";
                    return false;
                }
            Primitives.push_back({{{primitive.Minimum[0], primitive.Minimum[1], primitive.Minimum[2]},
                                   {primitive.Maximum[0], primitive.Maximum[1], primitive.Maximum[2]}},
                                  {primitive.Centroid[0], primitive.Centroid[1], primitive.Centroid[2]},
                                  {primitive.EncodedIndex}});
        }
        BuildNodes.reserve(input.size() * 2);
        const uint32_t root = BuildRecursive(0, uint32_t(Primitives.size()), 0);
        if (stop())
            return false;
        tree.Nodes.reserve(BuildNodes.size());
        Flatten(root, tree.Nodes);
        tree.References.reserve(Primitives.size());
        for (size_t index = 0; index < Primitives.size(); ++index)
        {
            if (index % CancellationPeriod == 0 && stop())
                return false;
            tree.References.push_back(Primitives[index].Reference.TypeAndIndex);
        }
        return !stop();
    }
};
} // namespace
bool Build(const std::vector<Primitive> &input, const std::function<bool()> &cancelled, Tree &tree,
           std::string &error)
{
    tree = {};
    if (Builder(cancelled).Run(input, tree, error))
        return true;
    tree = {};
    return false;
}
} // namespace Arvo::RayTracing::ComputeBvh
