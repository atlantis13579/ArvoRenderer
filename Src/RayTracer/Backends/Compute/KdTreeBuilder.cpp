#include "KdTreeBuilder.h"

#include <algorithm>
#include <array>
#include <limits>
#include <numeric>

namespace Arvo::RayTracing::ComputeAcceleration
{
    namespace
    {
        // Spatial SAH subdivision follows the kd-tree model in PBRT, 3rd ed.,
        // section 4.4: https://pbr-book.org/3ed-2018/Primitives_and_Intersection_Acceleration/Kd-Tree_Accelerator
        // This backend samples cell-plane candidates rather than copying PBRT's
        // edge-event builder. Straddlers are duplicated, never centroid assigned.
        constexpr uint32_t LeafAxis = 3;
        constexpr uint32_t PreferredLeafSize = 8;
        constexpr uint32_t MaximumDepth = 48; // Must match the native traversal stack.
        constexpr uint32_t CandidateIntervals = 16;
        constexpr size_t ReferenceBudgetMultiplier = 4;
        constexpr uint32_t CancellationCheckStride = 4096;
        constexpr double TraversalCost = 1.0;
        constexpr double IntersectionCost = 1.0;

        float Component(const Float3& point, uint32_t axis)
        {
            return axis == 0 ? point.X : axis == 1 ? point.Y : point.Z;
        }

        void SetComponent(Float3& point, uint32_t axis, float value)
        {
            if (axis == 0) point.X = value;
            else if (axis == 1) point.Y = value;
            else point.Z = value;
        }

        double SurfaceArea(const KdBounds& bounds)
        {
            const double x = static_cast<double>(bounds.Maximum.X) - bounds.Minimum.X;
            const double y = static_cast<double>(bounds.Maximum.Y) - bounds.Minimum.Y;
            const double z = static_cast<double>(bounds.Maximum.Z) - bounds.Minimum.Z;
            return 2.0 * (x * y + y * z + z * x);
        }

        class Builder
        {
        public:
            Builder(const std::vector<KdPrimitive>& primitives,
                const Detail::CancellationProbe& cancellation, KdTree& output)
                : Primitives(primitives), Cancellation(cancellation), Output(output),
                  ReferenceBudget(std::min(primitives.size() * ReferenceBudgetMultiplier,
                      static_cast<size_t>(std::numeric_limits<uint32_t>::max()) / 2)),
                  LiveReferenceCount(primitives.size()) {}

            bool Build(std::string& error)
            {
                if (Cancelled()) return Cancel(error);
                if (Primitives.empty()) return true;
                const float largest = std::numeric_limits<float>::max();
                Output.Bounds.Minimum = Float3(largest, largest, largest);
                Output.Bounds.Maximum = Float3(-largest, -largest, -largest);
                for (size_t index = 0; index < Primitives.size(); ++index)
                {
                    if (index % CancellationCheckStride == 0 && Cancelled()) return Cancel(error);
                    const auto& primitive = Primitives[index];
                    for (uint32_t axis = 0; axis < LeafAxis; ++axis)
                    {
                        SetComponent(Output.Bounds.Minimum, axis, std::min(
                            Component(Output.Bounds.Minimum, axis), Component(primitive.Minimum, axis)));
                        SetComponent(Output.Bounds.Maximum, axis, std::max(
                            Component(Output.Bounds.Maximum, axis), Component(primitive.Maximum, axis)));
                    }
                }
                std::vector<uint32_t> references(Primitives.size());
                std::iota(references.begin(), references.end(), 0u);
                Output.Nodes.reserve(Primitives.size());
                Output.References.reserve(Primitives.size());
                if (!BuildNode(Output.Bounds, references, 0)) return Cancel(error);
                return true;
            }

        private:
            bool Cancelled() const { return Cancellation && Cancellation(); }
            bool Cancel(std::string& error)
            {
                Output = {};
                error = "Compute kd-tree construction was cancelled.";
                return false;
            }

            bool MakeLeaf(uint32_t nodeIndex, const std::vector<uint32_t>& references)
            {
                auto& node = Output.Nodes[nodeIndex];
                node.SplitAxis = LeafAxis;
                node.AboveChildOrFirstReference = static_cast<uint32_t>(Output.References.size());
                node.PrimitiveCount = static_cast<uint32_t>(references.size());
                for (size_t index = 0; index < references.size(); ++index)
                {
                    if (index % CancellationCheckStride == 0 && Cancelled()) return false;
                    Output.References.push_back(Primitives[references[index]].EncodedReference);
                }
                return true;
            }

            bool BuildNode(const KdBounds& bounds, const std::vector<uint32_t>& references, uint32_t depth)
            {
                if (Cancelled()) return false;
                const auto nodeIndex = static_cast<uint32_t>(Output.Nodes.size());
                Output.Nodes.emplace_back();
                const double area = SurfaceArea(bounds);
                if (references.size() <= PreferredLeafSize || depth == MaximumDepth || area <= 0.0)
                    return MakeLeaf(nodeIndex, references);

                uint32_t bestAxis = LeafAxis;
                float bestPlane = 0.0f;
                size_t bestBelow = 0, bestAbove = 0;
                double bestCost = IntersectionCost * references.size();
                std::vector<float> minima(references.size()), maxima(references.size());
                for (uint32_t axis = 0; axis < LeafAxis; ++axis)
                {
                    const float minimum = Component(bounds.Minimum, axis);
                    const float maximum = Component(bounds.Maximum, axis);
                    if (minimum >= maximum) continue;
                    for (size_t index = 0; index < references.size(); ++index)
                    {
                        if (index % CancellationCheckStride == 0 && Cancelled()) return false;
                        minima[index] = Component(Primitives[references[index]].Minimum, axis);
                        maxima[index] = Component(Primitives[references[index]].Maximum, axis);
                    }
                    std::sort(minima.begin(), minima.end());
                    std::sort(maxima.begin(), maxima.end());
                    for (uint32_t candidate = 1; candidate < CandidateIntervals; ++candidate)
                    {
                        const float plane = static_cast<float>(minimum
                            + (static_cast<double>(maximum) - minimum) * candidate / CandidateIntervals);
                        if (plane <= minimum || plane >= maximum) continue;
                        const size_t below = std::upper_bound(minima.begin(), minima.end(), plane) - minima.begin();
                        const size_t above = maxima.end() - std::lower_bound(maxima.begin(), maxima.end(), plane);
                        if (below == 0 || above == 0) continue;
                        const size_t duplicates = below + above - references.size();
                        if (duplicates > ReferenceBudget - LiveReferenceCount) continue;
                        KdBounds lower = bounds, upper = bounds;
                        SetComponent(lower.Maximum, axis, plane);
                        SetComponent(upper.Minimum, axis, plane);
                        const double cost = TraversalCost + IntersectionCost
                            * (SurfaceArea(lower) * below + SurfaceArea(upper) * above) / area;
                        if (cost >= bestCost) continue;
                        bestCost = cost;
                        bestAxis = axis;
                        bestPlane = plane;
                        bestBelow = below;
                        bestAbove = above;
                    }
                }
                if (bestAxis == LeafAxis) return MakeLeaf(nodeIndex, references);

                std::vector<uint32_t> below, above;
                below.reserve(bestBelow);
                above.reserve(bestAbove);
                for (size_t index = 0; index < references.size(); ++index)
                {
                    if (index % CancellationCheckStride == 0 && Cancelled()) return false;
                    const uint32_t reference = references[index];
                    const auto& primitive = Primitives[reference];
                    // Equality belongs to both cells: rays parallel to or exactly
                    // crossing a split plane must not lose coplanar geometry.
                    if (Component(primitive.Minimum, bestAxis) <= bestPlane) below.push_back(reference);
                    if (Component(primitive.Maximum, bestAxis) >= bestPlane) above.push_back(reference);
                }
                LiveReferenceCount += below.size() + above.size() - references.size();
                KdBounds lower = bounds, upper = bounds;
                SetComponent(lower.Maximum, bestAxis, bestPlane);
                SetComponent(upper.Minimum, bestAxis, bestPlane);
                Output.Nodes[nodeIndex].SplitAxis = bestAxis;
                Output.Nodes[nodeIndex].SplitPosition = bestPlane;
                if (!BuildNode(lower, below, depth + 1)) return false;
                Output.Nodes[nodeIndex].AboveChildOrFirstReference = static_cast<uint32_t>(Output.Nodes.size());
                return BuildNode(upper, above, depth + 1);
            }

            const std::vector<KdPrimitive>& Primitives;
            const Detail::CancellationProbe& Cancellation;
            KdTree& Output;
            const size_t ReferenceBudget;
            size_t LiveReferenceCount;
        };
    }

    bool BuildKdTree(const std::vector<KdPrimitive>& primitives,
        const Detail::CancellationProbe& cancellation, KdTree& output, std::string& error)
    {
        output = {};
        // Both depth and duplicate references are bounded. Exhausting either
        // budget makes a complete leaf; no primitive or subtree is discarded.
        return Builder(primitives, cancellation, output).Build(error);
    }
}
