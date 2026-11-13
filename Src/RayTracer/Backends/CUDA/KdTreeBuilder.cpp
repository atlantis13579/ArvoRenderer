#include "KdTreeBuilder.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace Arvo::RayTracing::CudaKd
{
    namespace
    {
        // Spatial SAH: unlike a BVH, a crossing primitive belongs to both cells.
        // PBRT, 3rd ed., "Kd-Tree Accelerator":
        // https://pbr-book.org/3ed-2018/Primitives_and_Intersection_Acceleration/Kd-Tree_Accelerator
        constexpr std::uint32_t PreferredLeafSize = 8;
        constexpr std::uint32_t MaximumDepth = 32; // Matches the CUDA traversal stack.
        constexpr std::uint32_t PlaneIntervalCount = 16;
        constexpr std::size_t MaximumReferenceMultiplier = 8;
        constexpr std::size_t CancellationStride = 1024;
        constexpr double TraversalCost = 1.0;
        constexpr double IntersectionCost = 8.0;
        constexpr std::uint32_t LeafAxis = 3;

        double Area(const Node& cell)
        {
            const double x = static_cast<double>(cell.Maximum[0]) - cell.Minimum[0];
            const double y = static_cast<double>(cell.Maximum[1]) - cell.Minimum[1];
            const double z = static_cast<double>(cell.Maximum[2]) - cell.Minimum[2];
            return 2.0 * (x * y + y * z + z * x);
        }

        class Builder
        {
        public:
            Builder(const std::vector<Primitive>& primitives, std::vector<Node>& nodes,
                std::vector<std::uint32_t>& references, const std::function<bool()>& cancellation)
                : Primitives(primitives), Nodes(nodes), References(references), Cancellation(cancellation),
                  ReferenceSlots(primitives.size()), ReferenceLimit(std::min(
                      primitives.size() * MaximumReferenceMultiplier,
                      static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()))),
                  NodeLimit(std::min(ReferenceLimit * 2,
                      static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())))
            {
            }

            bool Run()
            {
                Node root;
                root.Minimum.fill(std::numeric_limits<float>::max());
                root.Maximum.fill(std::numeric_limits<float>::lowest());
                for (std::size_t index = 0; index < Primitives.size(); ++index)
                {
                    if (index % CancellationStride == 0 && Cancelled()) return false;
                    for (std::uint32_t axis = 0; axis < LeafAxis; ++axis)
                    {
                        const auto& primitive = Primitives[index];
                        if (!std::isfinite(primitive.Minimum[axis]) || !std::isfinite(primitive.Maximum[axis]))
                        {
                            InvalidBounds = true;
                            return false;
                        }
                        root.Minimum[axis] = std::min(root.Minimum[axis], primitive.Minimum[axis]);
                        root.Maximum[axis] = std::max(root.Maximum[axis], primitive.Maximum[axis]);
                    }
                }
                std::vector<std::uint32_t> indices(Primitives.size());
                std::iota(indices.begin(), indices.end(), 0u);
                return Subdivide(root, indices, 0);
            }

            bool InvalidBounds = false;

        private:
            bool Cancelled() const { return Cancellation && Cancellation(); }

            bool Subdivide(Node cell, const std::vector<std::uint32_t>& indices, std::uint32_t depth)
            {
                if (Cancelled()) return false;
                const auto nodeIndex = static_cast<std::uint32_t>(Nodes.size());
                Nodes.push_back(cell);
                const double parentArea = Area(cell);
                double bestCost = IntersectionCost * indices.size();
                std::uint32_t bestAxis = LeafAxis;
                float bestPlane = 0.0f;
                std::size_t bestBelow = 0, bestAbove = 0;
                if (indices.size() > PreferredLeafSize && depth < MaximumDepth && parentArea > 0.0
                    && PlannedNodes + 2 <= NodeLimit)
                {
                    for (std::uint32_t axis = 0; axis < LeafAxis; ++axis)
                    {
                        for (std::uint32_t candidate = 1; candidate < PlaneIntervalCount; ++candidate)
                        {
                            const double fraction = static_cast<double>(candidate) / PlaneIntervalCount;
                            const float plane = static_cast<float>(
                                (1.0 - fraction) * cell.Minimum[axis] + fraction * cell.Maximum[axis]);
                            if (!(plane > cell.Minimum[axis] && plane < cell.Maximum[axis])) continue;
                            std::size_t belowCount = 0, aboveCount = 0;
                            for (std::size_t offset = 0; offset < indices.size(); ++offset)
                            {
                                if (offset % CancellationStride == 0 && Cancelled()) return false;
                                const Primitive& primitive = Primitives[indices[offset]];
                                belowCount += primitive.Minimum[axis] <= plane;
                                aboveCount += primitive.Maximum[axis] >= plane;
                            }
                            if (belowCount == indices.size() && aboveCount == indices.size()) continue;
                            const std::size_t duplication = belowCount + aboveCount - indices.size();
                            if (duplication > ReferenceLimit - ReferenceSlots) continue;
                            Node below = cell, above = cell;
                            below.Maximum[axis] = plane;
                            above.Minimum[axis] = plane;
                            const double cost = TraversalCost + IntersectionCost
                                * (Area(below) * belowCount + Area(above) * aboveCount) / parentArea;
                            if (cost < bestCost)
                            {
                                bestCost = cost;
                                bestAxis = axis;
                                bestPlane = plane;
                                bestBelow = belowCount;
                                bestAbove = aboveCount;
                            }
                        }
                    }
                }
                if (bestAxis == LeafAxis)
                {
                    Nodes[nodeIndex].FirstReference = static_cast<std::uint32_t>(References.size());
                    Nodes[nodeIndex].ReferenceCount = static_cast<std::uint32_t>(indices.size());
                    for (std::size_t offset = 0; offset < indices.size(); ++offset)
                    {
                        if (offset % CancellationStride == 0 && Cancelled()) return false;
                        References.push_back(Primitives[indices[offset]].Reference);
                    }
                    return true;
                }
                PlannedNodes += 2;
                ReferenceSlots += bestBelow + bestAbove - indices.size();
                std::vector<std::uint32_t> below, above;
                below.reserve(bestBelow);
                above.reserve(bestAbove);
                for (std::size_t offset = 0; offset < indices.size(); ++offset)
                {
                    if (offset % CancellationStride == 0 && Cancelled()) return false;
                    const auto index = indices[offset];
                    if (Primitives[index].Minimum[bestAxis] <= bestPlane) below.push_back(index);
                    if (Primitives[index].Maximum[bestAxis] >= bestPlane) above.push_back(index);
                }
                Nodes[nodeIndex].SplitAxis = bestAxis;
                Nodes[nodeIndex].SplitPosition = bestPlane;
                Node belowCell = cell, aboveCell = cell;
                belowCell.Maximum[bestAxis] = bestPlane;
                aboveCell.Minimum[bestAxis] = bestPlane;
                if (!Subdivide(belowCell, below, depth + 1)) return false;
                Nodes[nodeIndex].AboveChild = static_cast<std::uint32_t>(Nodes.size());
                return Subdivide(aboveCell, above, depth + 1);
            }

            const std::vector<Primitive>& Primitives;
            std::vector<Node>& Nodes;
            std::vector<std::uint32_t>& References;
            const std::function<bool()>& Cancellation;
            std::size_t ReferenceSlots;
            const std::size_t ReferenceLimit;
            const std::size_t NodeLimit;
            std::size_t PlannedNodes = 1;
        };
    }

    bool Build(const std::vector<Primitive>& primitives, std::vector<Node>& nodes,
        std::vector<std::uint32_t>& references, const std::function<bool()>& cancellation,
        std::string& error)
    {
        nodes.clear();
        references.clear();
        if (primitives.empty()) return true;
        Builder builder(primitives, nodes, references, cancellation);
        if (builder.Run()) return true;
        nodes.clear();
        references.clear();
        error = builder.InvalidBounds ? "CUDA kd-tree requires finite primitive bounds."
                                      : "CUDA kd-tree construction was cancelled.";
        return false;
    }
}
