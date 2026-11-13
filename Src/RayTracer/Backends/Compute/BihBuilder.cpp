#include "BihBuilder.h"

#include <algorithm>
#include <array>
#include <limits>
#include <numeric>

namespace Arvo::RayTracing::ComputeAcceleration
{
    namespace
    {
        // Wachter and Keller (2006), "Instant Ray Tracing: The Bounding Interval
        // Hierarchy", https://doi.org/10.2312/EGWR/EGSR06/139-149.
        // Object partitioning stores each reference once. Two child clipping
        // planes may overlap or leave a gap; neither case requires duplication.
        constexpr uint32_t LeafAxis = 3;
        constexpr uint32_t PreferredLeafSize = 8;
        constexpr uint32_t MaximumDepth = 48; // Paired with Bih.hlsl's traversal stack.
        constexpr uint32_t CancellationCheckStride = 4096;
        constexpr size_t MaximumPrimitiveCount = std::numeric_limits<uint32_t>::max() / 2;

        float Coordinate(const Float3& point, uint32_t axis)
        {
            return axis == 0 ? point.X : axis == 1 ? point.Y : point.Z;
        }

        double Center(const BihPrimitive& primitive, uint32_t axis)
        {
            return (static_cast<double>(Coordinate(primitive.Minimum, axis))
                + Coordinate(primitive.Maximum, axis)) * 0.5;
        }

        BihBounds EmptyBounds()
        {
            const float largest = std::numeric_limits<float>::max();
            BihBounds result;
            result.Minimum = {largest, largest, largest};
            result.Maximum = {-largest, -largest, -largest};
            return result;
        }

        void Include(BihBounds& bounds, const BihPrimitive& primitive)
        {
            bounds.Minimum.X = std::min(bounds.Minimum.X, primitive.Minimum.X);
            bounds.Minimum.Y = std::min(bounds.Minimum.Y, primitive.Minimum.Y);
            bounds.Minimum.Z = std::min(bounds.Minimum.Z, primitive.Minimum.Z);
            bounds.Maximum.X = std::max(bounds.Maximum.X, primitive.Maximum.X);
            bounds.Maximum.Y = std::max(bounds.Maximum.Y, primitive.Maximum.Y);
            bounds.Maximum.Z = std::max(bounds.Maximum.Z, primitive.Maximum.Z);
        }

        class BihConstruction
        {
        public:
            BihConstruction(const std::vector<BihPrimitive>& primitives,
                const Detail::CancellationProbe& cancellation, BihTree& output)
                : Primitives(primitives), Cancellation(cancellation), Output(output) {}

            bool Run(std::string& error)
            {
                if (IsCancelled()) return Cancel(error);
                // A nonempty binary tree has at most 2N-1 nodes. This bound keeps
                // all indices representable before any recursive construction.
                if (Primitives.size() > MaximumPrimitiveCount)
                {
                    error = "Compute BIH primitive count exceeds its 32-bit node-index limit.";
                    return false;
                }
                if (Primitives.empty()) return true;
                Output.Bounds = EmptyBounds();
                for (size_t index = 0; index < Primitives.size(); ++index)
                {
                    if (index % CancellationCheckStride == 0 && IsCancelled()) return Cancel(error);
                    Include(Output.Bounds, Primitives[index]);
                }
                Order.resize(Primitives.size());
                std::iota(Order.begin(), Order.end(), 0u);
                Output.Nodes.reserve(Primitives.size());
                Output.References.reserve(Primitives.size());
                if (!Subdivide(0, static_cast<uint32_t>(Order.size()), 0)) return Cancel(error);
                if (IsCancelled()) return Cancel(error);
                return true;
            }

        private:
            bool IsCancelled() const { return Cancellation && Cancellation(); }

            bool Cancel(std::string& error)
            {
                Output = {};
                error = "Compute BIH construction was cancelled.";
                return false;
            }

            bool Leaf(uint32_t nodeIndex, uint32_t begin, uint32_t end)
            {
                auto& node = Output.Nodes[nodeIndex];
                node.Axis = LeafAxis;
                node.RightChildOrFirstReference = static_cast<uint32_t>(Output.References.size());
                node.PrimitiveCount = end - begin;
                for (uint32_t index = begin; index < end; ++index)
                {
                    if (index % CancellationCheckStride == 0 && IsCancelled()) return false;
                    Output.References.push_back(Primitives[Order[index]].EncodedReference);
                }
                return true;
            }

            bool Subdivide(uint32_t begin, uint32_t end, uint32_t depth)
            {
                if (IsCancelled()) return false;
                const auto nodeIndex = static_cast<uint32_t>(Output.Nodes.size());
                Output.Nodes.emplace_back();
                if (end - begin <= PreferredLeafSize || depth == MaximumDepth)
                    return Leaf(nodeIndex, begin, end);

                const double largest = std::numeric_limits<double>::max();
                std::array<double, 3> minimum = {largest, largest, largest};
                std::array<double, 3> maximum = {-largest, -largest, -largest};
                for (uint32_t index = begin; index < end; ++index)
                {
                    if (index % CancellationCheckStride == 0 && IsCancelled()) return false;
                    const auto& primitive = Primitives[Order[index]];
                    for (uint32_t axis = 0; axis < LeafAxis; ++axis)
                    {
                        const double center = Center(primitive, axis);
                        minimum[axis] = std::min(minimum[axis], center);
                        maximum[axis] = std::max(maximum[axis], center);
                    }
                }
                uint32_t axis = 0;
                for (uint32_t candidate = 1; candidate < LeafAxis; ++candidate)
                    if (maximum[candidate] - minimum[candidate] > maximum[axis] - minimum[axis])
                        axis = candidate;
                const double midpoint = minimum[axis] + (maximum[axis] - minimum[axis]) * 0.5;

                // An in-place midpoint partition keeps ownership exclusive. If
                // centers coincide, a deterministic median guarantees progress.
                uint32_t middle = begin;
                for (uint32_t index = begin; index < end; ++index)
                {
                    if (index % CancellationCheckStride == 0 && IsCancelled()) return false;
                    if (Center(Primitives[Order[index]], axis) < midpoint)
                        std::swap(Order[middle++], Order[index]);
                }
                if (middle == begin || middle == end)
                {
                    middle = begin + (end - begin) / 2;
                    std::nth_element(Order.begin() + begin, Order.begin() + middle, Order.begin() + end,
                        [&](uint32_t a, uint32_t b)
                        {
                            const double centerA = Center(Primitives[a], axis), centerB = Center(Primitives[b], axis);
                            return centerA == centerB ? a < b : centerA < centerB;
                        });
                    if (IsCancelled()) return false;
                }

                float leftMaximum = -std::numeric_limits<float>::max();
                float rightMinimum = std::numeric_limits<float>::max();
                for (uint32_t index = begin; index < end; ++index)
                {
                    if (index % CancellationCheckStride == 0 && IsCancelled()) return false;
                    const auto& primitive = Primitives[Order[index]];
                    if (index < middle) leftMaximum = std::max(leftMaximum, Coordinate(primitive.Maximum, axis));
                    else rightMinimum = std::min(rightMinimum, Coordinate(primitive.Minimum, axis));
                }
                Output.Nodes[nodeIndex].Axis = axis;
                Output.Nodes[nodeIndex].LeftMaximum = leftMaximum;
                Output.Nodes[nodeIndex].RightMinimum = rightMinimum;
                if (!Subdivide(begin, middle, depth + 1)) return false;
                Output.Nodes[nodeIndex].RightChildOrFirstReference = static_cast<uint32_t>(Output.Nodes.size());
                return Subdivide(middle, end, depth + 1);
            }

            const std::vector<BihPrimitive>& Primitives;
            const Detail::CancellationProbe& Cancellation;
            BihTree& Output;
            std::vector<uint32_t> Order;
        };
    }

    bool BuildBih(const std::vector<BihPrimitive>& primitives,
        const Detail::CancellationProbe& cancellation, BihTree& output, std::string& error)
    {
        output = {};
        return BihConstruction(primitives, cancellation, output).Run(error);
    }
}
