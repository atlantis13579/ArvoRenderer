#include "BihBuilder.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace Arvo::RayTracing::MetalBih
{
    namespace
    {
        // Wachter/Keller, "Instant Ray Tracing: The Bounding Interval Hierarchy"
        // (2006), DOI 10.2312/EGWR/EGSR06/139-149. Partition objects, then bound
        // each side with its own plane. Geometry is never duplicated.
        constexpr std::uint32_t LeafAxis = 3;
        constexpr std::uint32_t MaximumDepth = 32;
        constexpr std::uint32_t PreferredLeafCount = 8;
        constexpr std::size_t CancellationPeriod = 1024;

        class Builder
        {
        public:
            Builder(const std::vector<Object>& objects, const std::function<bool()>& cancelled,
                std::vector<IntervalNode>& nodes, std::vector<std::uint32_t>& references)
                : Objects_(objects), Cancelled_(cancelled), Nodes_(nodes), References_(references),
                  Ordering_(objects.size())
            {
                std::iota(Ordering_.begin(), Ordering_.end(), 0u);
            }

            bool BuildAll()
            {
                return BuildRange(0u, static_cast<std::uint32_t>(Objects_.size()), 0u);
            }

        private:
            bool Cancelled(std::size_t index = 0) const
            {
                return index % CancellationPeriod == 0 && Cancelled_ && Cancelled_();
            }

            static double Midpoint(const Object& object, std::uint32_t axis)
            {
                return 0.5 * static_cast<double>(object.Lower[axis]) + 0.5 * object.Upper[axis];
            }

            bool BuildRange(std::uint32_t start, std::uint32_t finish, std::uint32_t depth)
            {
                if (Cancelled()) return false;
                IntervalNode node;
                node.Lower.fill(std::numeric_limits<float>::max());
                node.Upper.fill(std::numeric_limits<float>::lowest());
                std::array<double, 3> centerLower, centerUpper;
                centerLower.fill(std::numeric_limits<double>::max());
                centerUpper.fill(std::numeric_limits<double>::lowest());
                for (std::uint32_t position = start; position < finish; ++position)
                {
                    if (Cancelled(position)) return false;
                    const auto& object = Objects_[Ordering_[position]];
                    for (std::uint32_t axis = 0; axis < LeafAxis; ++axis)
                    {
                        node.Lower[axis] = std::min(node.Lower[axis], object.Lower[axis]);
                        node.Upper[axis] = std::max(node.Upper[axis], object.Upper[axis]);
                        const double center = Midpoint(object, axis);
                        centerLower[axis] = std::min(centerLower[axis], center);
                        centerUpper[axis] = std::max(centerUpper[axis], center);
                    }
                }
                const std::uint32_t parent = static_cast<std::uint32_t>(Nodes_.size());
                Nodes_.push_back(node);
                std::uint32_t axis = 0u;
                for (std::uint32_t candidate = 1u; candidate < LeafAxis; ++candidate)
                    if (centerUpper[candidate] - centerLower[candidate] > centerUpper[axis] - centerLower[axis])
                        axis = candidate;
                const bool canPartition = finish - start > PreferredLeafCount && depth < MaximumDepth
                    && centerUpper[axis] > centerLower[axis];
                std::uint32_t boundary = start;
                if (canPartition)
                {
                    // A midpoint on the widest centroid range gives a linear
                    // partition pass. Coincident centers terminate as a leaf.
                    const double split = 0.5 * centerLower[axis] + 0.5 * centerUpper[axis];
                    std::uint32_t scanEnd = finish;
                    while (boundary < scanEnd)
                    {
                        if (Cancelled()) return false;
                        if (Midpoint(Objects_[Ordering_[boundary]], axis) < split)
                            ++boundary;
                        else
                            std::swap(Ordering_[boundary], Ordering_[--scanEnd]);
                    }
                }
                if (boundary == start || boundary == finish)
                {
                    Nodes_[parent].ReferenceBegin = static_cast<std::uint32_t>(References_.size());
                    Nodes_[parent].ReferenceCount = finish - start;
                    for (std::uint32_t position = start; position < finish; ++position)
                    {
                        if (Cancelled(position)) return false;
                        References_.push_back(Objects_[Ordering_[position]].EncodedIndex);
                    }
                    return true;
                }
                Nodes_[parent].Axis = axis;
                if (!BuildRange(start, boundary, depth + 1u)) return false;
                const std::uint32_t right = static_cast<std::uint32_t>(Nodes_.size());
                Nodes_[parent].RightChild = right;
                if (!BuildRange(boundary, finish, depth + 1u)) return false;
                Nodes_[parent].LeftMaximum = Nodes_[parent + 1u].Upper[axis];
                Nodes_[parent].RightMinimum = Nodes_[right].Lower[axis];
                return true;
            }

            const std::vector<Object>& Objects_;
            const std::function<bool()>& Cancelled_;
            std::vector<IntervalNode>& Nodes_;
            std::vector<std::uint32_t>& References_;
            std::vector<std::uint32_t> Ordering_;
        };
    }

    bool Build(const std::vector<Object>& objects, const std::function<bool()>& cancelled,
        std::vector<IntervalNode>& nodes, std::vector<std::uint32_t>& references,
        std::string& error)
    {
        nodes.clear();
        references.clear();
        constexpr std::size_t MaximumObjectCount = std::numeric_limits<std::uint32_t>::max() / 2;
        if (objects.size() > MaximumObjectCount)
        {
            error = "Metal BIH object count exceeds its 32-bit node capacity.";
            return false;
        }
        for (std::size_t index = 0; index < objects.size(); ++index)
        {
            if (index % CancellationPeriod == 0 && cancelled && cancelled())
            {
                error = "Metal BIH construction was cancelled.";
                return false;
            }
            for (std::uint32_t axis = 0; axis < LeafAxis; ++axis)
            {
                const float lower = objects[index].Lower[axis];
                const float upper = objects[index].Upper[axis];
                if (!std::isfinite(lower) || !std::isfinite(upper) || lower > upper)
                {
                    error = "Metal BIH requires finite ordered bounds for each object.";
                    return false;
                }
            }
        }
        if (objects.empty()) return true;
        Builder builder(objects, cancelled, nodes, references);
        if (builder.BuildAll()) return true;
        nodes.clear();
        references.clear();
        error = "Metal BIH construction was cancelled.";
        return false;
    }
}
