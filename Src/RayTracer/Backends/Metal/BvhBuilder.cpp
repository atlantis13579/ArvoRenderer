#include "BvhBuilder.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace Arvo::RayTracing::MetalBvh
{
namespace
{
// Metal owns its default BVH construction: centroid-median splitting on the longest axis.
// Each primitive belongs to one leaf; depth-first escape links let the native
// Bvh shader skip rejected subtrees without a traversal stack. Spatial splitting
// is a separate builder with a different reference-ownership rule.
// Pharr, Jakob and Humphreys, Physically Based Rendering (4th ed.), Section 7.3.
// https://pbr-book.org/4ed/Primitives_and_Intersection_Acceleration/Bounding_Volume_Hierarchies
constexpr std::uint32_t LeafPrimitiveCount = 8u;
constexpr std::uint32_t MaximumBuildDepth = 48u;
constexpr float MinimumCentroidExtent = 1.0e-6f;
constexpr size_t CancellationPeriod = 1024;
struct Vec3
{
    float X = 0.0f;
    float Y = 0.0f;
    float Z = 0.0f;
};

[[nodiscard]] Vec3 Subtract(const Vec3 &left, const Vec3 &right)
{
    return {left.X - right.X, left.Y - right.Y, left.Z - right.Z};
}

struct Bounds
{
    Vec3 Minimum{std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
                 std::numeric_limits<float>::max()};
    Vec3 Maximum{-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(),
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
};

[[nodiscard]] float Component(const Vec3 &value, std::uint32_t axis)
{
    return axis == 0u ? value.X : (axis == 1u ? value.Y : value.Z);
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

    [[nodiscard]] bool IsLeaf() const noexcept { return PrimitiveCount != 0u; }
};
using PackedAccelerationNode = Node;
class Builder
{
    const std::function<bool()> &Cancellation_;
    std::vector<BuildPrimitive> Primitives_;
    std::vector<BuildNode> BuildNodes_;
    [[nodiscard]] bool IsCancelled() const { return Cancellation_ && Cancellation_(); }
    std::uint32_t BuildRecursive(std::uint32_t begin, std::uint32_t end, std::uint32_t depth)
    {
        const std::uint32_t nodeIndex = static_cast<std::uint32_t>(BuildNodes_.size());
        BuildNodes_.emplace_back();
        Bounds primitiveBounds;
        Bounds centroidBounds;
        for (std::uint32_t index = begin; index < end; ++index)
        {
            primitiveBounds.Expand(Primitives_[index].BoundingBox);
            centroidBounds.Expand(Primitives_[index].Centroid);
        }
        BuildNodes_[nodeIndex].BoundingBox = primitiveBounds;
        const std::uint32_t count = end - begin;
        const Vec3 centroidExtent = Subtract(centroidBounds.Maximum, centroidBounds.Minimum);
        if (count <= LeafPrimitiveCount || depth >= MaximumBuildDepth || IsCancelled() ||
            std::max({centroidExtent.X, centroidExtent.Y, centroidExtent.Z}) <= MinimumCentroidExtent)
        {
            BuildNodes_[nodeIndex].FirstPrimitive = begin;
            BuildNodes_[nodeIndex].PrimitiveCount = count;
            return nodeIndex;
        }

        std::uint32_t axis = centroidExtent.Y > centroidExtent.X ? 1u : 0u;
        if (Component(centroidExtent, 2u) > Component(centroidExtent, axis))
        {
            axis = 2u;
        }
        const std::uint32_t middle = begin + count / 2u;
        std::nth_element(Primitives_.begin() + begin, Primitives_.begin() + middle, Primitives_.begin() + end,
                         [axis](const BuildPrimitive &left, const BuildPrimitive &right) {
                             return Component(left.Centroid, axis) < Component(right.Centroid, axis);
                         });
        const std::uint32_t left = BuildRecursive(begin, middle, depth + 1u);
        const std::uint32_t right = BuildRecursive(middle, end, depth + 1u);
        BuildNodes_[nodeIndex].Left = left;
        BuildNodes_[nodeIndex].Right = right;
        return nodeIndex;
    }
    void Flatten(std::uint32_t buildIndex, std::vector<PackedAccelerationNode> &output)
    {
        const std::uint32_t outputIndex = static_cast<std::uint32_t>(output.size());
        output.emplace_back();
        const BuildNode &source = BuildNodes_[buildIndex];
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
    }

  public:
    explicit Builder(const std::function<bool()> &cancelled) : Cancellation_(cancelled) {}
    bool Run(const std::vector<Primitive> &input, Tree &tree, std::string &error)
    {
        const auto stop = [&] {
            if (!IsCancelled())
                return false;
            error = "Metal BVH construction was cancelled";
            return true;
        };
        if (stop())
            return false;
        constexpr size_t MaximumPrimitiveCount = std::numeric_limits<uint32_t>::max() / 2;
        if (input.size() > MaximumPrimitiveCount)
        {
            error = "Metal BVH exceeds 32-bit node indexing";
            return false;
        }
        if (input.empty())
            return true;
        Primitives_.reserve(input.size());
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
                    error = "Invalid Metal BVH input bounds or centroid";
                    return false;
                }
            Primitives_.push_back({{{primitive.Minimum[0], primitive.Minimum[1], primitive.Minimum[2]},
                                    {primitive.Maximum[0], primitive.Maximum[1], primitive.Maximum[2]}},
                                   {primitive.Centroid[0], primitive.Centroid[1], primitive.Centroid[2]},
                                   {primitive.EncodedIndex}});
        }
        BuildNodes_.reserve(input.size() * 2);
        const uint32_t root = BuildRecursive(0, uint32_t(Primitives_.size()), 0);
        if (stop())
            return false;
        tree.Nodes.reserve(BuildNodes_.size());
        Flatten(root, tree.Nodes);
        tree.References.reserve(Primitives_.size());
        for (size_t index = 0; index < Primitives_.size(); ++index)
        {
            if (index % CancellationPeriod == 0 && stop())
                return false;
            tree.References.push_back(Primitives_[index].Reference.TypeAndIndex);
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
} // namespace Arvo::RayTracing::MetalBvh
