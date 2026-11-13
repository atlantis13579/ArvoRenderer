#include "BihBuilder.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace Arvo::RayTracing::CudaBih
{
    namespace
    {
        // Wachter and Keller (2006), "Instant Ray Tracing: The Bounding Interval
        // Hierarchy", DOI 10.2312/EGWR/EGSR06/139-149. Object partitioning assigns
        // every primitive once; two tight clipping planes may overlap or leave a gap.
        constexpr std::uint32_t LeafAxis = 3;
        constexpr std::uint32_t LeafTarget = 8;
        constexpr std::uint32_t DepthLimit = 32;
        constexpr std::uint32_t BinCount = 16;
        constexpr std::size_t CancellationStride = 1024;
        constexpr double RelativeTraversalCost = 0.125;

        struct Bounds
        {
            std::array<float, 3> Minimum;
            std::array<float, 3> Maximum;

            Bounds()
            {
                Minimum.fill(std::numeric_limits<float>::max());
                Maximum.fill(std::numeric_limits<float>::lowest());
            }

            void Include(const Primitive& primitive)
            {
                for (std::uint32_t axis = 0; axis < LeafAxis; ++axis)
                {
                    Minimum[axis] = std::min(Minimum[axis], primitive.Minimum[axis]);
                    Maximum[axis] = std::max(Maximum[axis], primitive.Maximum[axis]);
                }
            }

            void Include(const Bounds& other)
            {
                for (std::uint32_t axis = 0; axis < LeafAxis; ++axis)
                {
                    Minimum[axis] = std::min(Minimum[axis], other.Minimum[axis]);
                    Maximum[axis] = std::max(Maximum[axis], other.Maximum[axis]);
                }
            }

            double Area() const
            {
                const double x = static_cast<double>(Maximum[0]) - Minimum[0];
                const double y = static_cast<double>(Maximum[1]) - Minimum[1];
                const double z = static_cast<double>(Maximum[2]) - Minimum[2];
                return 2.0 * (x * y + x * z + y * z);
            }
        };

        struct Bin
        {
            Bounds Box;
            std::uint32_t Count = 0;
        };

        double Center(const Primitive& primitive, std::uint32_t axis)
        {
            return (static_cast<double>(primitive.Minimum[axis]) + primitive.Maximum[axis]) * 0.5;
        }

        std::uint32_t BinIndex(double center, double minimum, double maximum)
        {
            return std::min(BinCount - 1,
                static_cast<std::uint32_t>((center - minimum) / (maximum - minimum) * BinCount));
        }

        class Construction
        {
        public:
            Construction(const std::vector<Primitive>& primitives, std::vector<Node>& nodes,
                std::vector<std::uint32_t>& references, const std::function<bool()>& cancelled)
                : Primitives(primitives), Nodes(nodes), References(references), Cancelled(cancelled),
                  Indices(primitives.size())
            {
                std::iota(Indices.begin(), Indices.end(), 0u);
            }

            bool Run()
            {
                return Emit(0, static_cast<std::uint32_t>(Indices.size()), 0);
            }

        private:
            bool Stop(std::size_t iteration = 0) const
            {
                return iteration % CancellationStride == 0 && Cancelled && Cancelled();
            }

            bool Emit(std::uint32_t begin, std::uint32_t end, std::uint32_t depth)
            {
                if (Stop()) return false;
                Bounds box;
                std::array<double, 3> minimumCenter, maximumCenter;
                minimumCenter.fill(std::numeric_limits<double>::max());
                maximumCenter.fill(std::numeric_limits<double>::lowest());
                for (std::uint32_t offset = begin; offset < end; ++offset)
                {
                    if (Stop(offset)) return false;
                    const Primitive& primitive = Primitives[Indices[offset]];
                    box.Include(primitive);
                    for (std::uint32_t axis = 0; axis < LeafAxis; ++axis)
                    {
                        const double center = Center(primitive, axis);
                        minimumCenter[axis] = std::min(minimumCenter[axis], center);
                        maximumCenter[axis] = std::max(maximumCenter[axis], center);
                    }
                }
                const auto destination = static_cast<std::uint32_t>(Nodes.size());
                Node node;
                node.Minimum = box.Minimum;
                node.Maximum = box.Maximum;
                Nodes.push_back(node);
                const std::uint32_t count = end - begin;
                const double area = box.Area();
                double bestCost = static_cast<double>(count);
                std::uint32_t chosenAxis = LeafAxis, chosenBin = 0;
                if (count > LeafTarget && depth < DepthLimit && area > 0.0)
                {
                    for (std::uint32_t axis = 0; axis < LeafAxis; ++axis)
                    {
                        if (!(maximumCenter[axis] > minimumCenter[axis])) continue;
                        std::array<Bin, BinCount> bins;
                        for (std::uint32_t offset = begin; offset < end; ++offset)
                        {
                            if (Stop(offset)) return false;
                            const auto& primitive = Primitives[Indices[offset]];
                            Bin& bin = bins[BinIndex(Center(primitive, axis), minimumCenter[axis], maximumCenter[axis])];
                            ++bin.Count;
                            bin.Box.Include(primitive);
                        }
                        std::array<double, BinCount> rightArea{};
                        std::array<std::uint32_t, BinCount> rightCount{};
                        Bounds right;
                        std::uint32_t accumulated = 0;
                        for (std::uint32_t index = BinCount; index-- > 0;)
                        {
                            if (bins[index].Count != 0) right.Include(bins[index].Box);
                            accumulated += bins[index].Count;
                            rightCount[index] = accumulated;
                            rightArea[index] = accumulated == 0 ? 0.0 : right.Area();
                        }
                        Bounds left;
                        accumulated = 0;
                        for (std::uint32_t index = 0; index + 1 < BinCount; ++index)
                        {
                            if (bins[index].Count != 0) left.Include(bins[index].Box);
                            accumulated += bins[index].Count;
                            if (accumulated == 0 || rightCount[index + 1] == 0) continue;
                            const double cost = RelativeTraversalCost
                                + (accumulated * left.Area() + rightCount[index + 1] * rightArea[index + 1]) / area;
                            if (cost < bestCost)
                            {
                                bestCost = cost;
                                chosenAxis = axis;
                                chosenBin = index;
                            }
                        }
                    }
                }
                if (chosenAxis == LeafAxis)
                {
                    Nodes[destination].FirstReference = static_cast<std::uint32_t>(References.size());
                    Nodes[destination].ReferenceCount = count;
                    for (std::uint32_t offset = begin; offset < end; ++offset)
                    {
                        if (Stop(offset)) return false;
                        References.push_back(Primitives[Indices[offset]].Reference);
                    }
                    return true;
                }
                std::uint32_t middle = begin;
                for (std::uint32_t offset = begin; offset < end; ++offset)
                {
                    if (Stop(offset)) return false;
                    const auto& primitive = Primitives[Indices[offset]];
                    if (BinIndex(Center(primitive, chosenAxis), minimumCenter[chosenAxis], maximumCenter[chosenAxis]) <= chosenBin)
                        std::swap(Indices[middle++], Indices[offset]);
                }
                Nodes[destination].Axis = chosenAxis;
                if (!Emit(begin, middle, depth + 1)) return false;
                const auto above = static_cast<std::uint32_t>(Nodes.size());
                Nodes[destination].AboveChild = above;
                if (!Emit(middle, end, depth + 1)) return false;
                Nodes[destination].LeftMaximum = Nodes[destination + 1].Maximum[chosenAxis];
                Nodes[destination].RightMinimum = Nodes[above].Minimum[chosenAxis];
                return true;
            }

            const std::vector<Primitive>& Primitives;
            std::vector<Node>& Nodes;
            std::vector<std::uint32_t>& References;
            const std::function<bool()>& Cancelled;
            std::vector<std::uint32_t> Indices;
        };
    }

    bool Build(const std::vector<Primitive>& primitives, std::vector<Node>& nodes,
        std::vector<std::uint32_t>& references, const std::function<bool()>& cancelled,
        std::string& error)
    {
        nodes.clear();
        references.clear();
        if (primitives.empty()) return true;
        // A binary object hierarchy has at most 2*N-1 nodes.
        constexpr std::size_t MaximumPrimitives = std::numeric_limits<std::uint32_t>::max() / 2;
        if (primitives.size() > MaximumPrimitives)
        {
            error = "CUDA BIH exceeds its 32-bit node layout.";
            return false;
        }
        for (std::size_t index = 0; index < primitives.size(); ++index)
        {
            if (index % CancellationStride == 0 && cancelled && cancelled())
            {
                error = "CUDA BIH construction was cancelled.";
                return false;
            }
            for (std::uint32_t axis = 0; axis < LeafAxis; ++axis)
                if (!std::isfinite(primitives[index].Minimum[axis])
                    || !std::isfinite(primitives[index].Maximum[axis])
                    || primitives[index].Minimum[axis] > primitives[index].Maximum[axis])
                {
                    error = "CUDA BIH requires finite, ordered primitive bounds.";
                    return false;
                }
        }
        Construction construction(primitives, nodes, references, cancelled);
        if (construction.Run()) return true;
        nodes.clear();
        references.clear();
        error = "CUDA BIH construction was cancelled.";
        return false;
    }
}
