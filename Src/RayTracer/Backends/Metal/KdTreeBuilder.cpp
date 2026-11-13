#include "KdTreeBuilder.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace Arvo::RayTracing::MetalKd
{
    namespace
    {
        // PBRT 3rd ed., "Kd-Tree Accelerator": spatial subdivision duplicates
        // straddling primitives. This Metal builder bins endpoint events to
        // evaluate spatial SAH planes without sorting the primitive list.
        // https://pbr-book.org/3ed-2018/Primitives_and_Intersection_Acceleration/Kd-Tree_Accelerator
        constexpr std::uint32_t LeafAxis = 3;
        constexpr std::uint32_t LeafTarget = 8;
        constexpr std::uint32_t DepthLimit = 32; // Must fit the Metal traversal stack.
        constexpr std::size_t CandidateCount = 12;
        constexpr std::size_t ReferenceExpansionLimit = 8;
        constexpr std::size_t CancellationCheckInterval = 1024;
        constexpr double RelativeTraversalCost = 0.125;

        struct PlaneChoice
        {
            std::uint32_t Axis = LeafAxis;
            float Position = 0.0f;
            std::size_t LowerCount = 0;
            std::size_t UpperCount = 0;
        };

        double Surface(const SpatialNode& cell)
        {
            std::array<double, 3> size;
            for (std::size_t axis = 0; axis < size.size(); ++axis)
                size[axis] = static_cast<double>(cell.Upper[axis]) - cell.Lower[axis];
            return 2.0 * (size[0] * size[1] + size[0] * size[2] + size[1] * size[2]);
        }

        class Construction
        {
        public:
            Construction(const std::vector<PrimitiveBounds>& source,
                const std::function<bool()>& cancelled, std::vector<SpatialNode>& nodes,
                std::vector<std::uint32_t>& references)
                : Source_(source), Cancelled_(cancelled), Nodes_(nodes), References_(references),
                  PlannedReferences_(source.size()), ReferenceBudget_(std::min(
                      source.size() * ReferenceExpansionLimit,
                      static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()))),
                  NodeBudget_(std::min(ReferenceBudget_ * 2,
                      static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())))
            {
            }

            bool Execute(std::string& error)
            {
                SpatialNode root;
                root.Lower.fill(std::numeric_limits<float>::max());
                root.Upper.fill(std::numeric_limits<float>::lowest());
                for (std::size_t index = 0; index < Source_.size(); ++index)
                {
                    if (Check(index)) return Abort(error);
                    for (std::uint32_t axis = 0; axis < LeafAxis; ++axis)
                    {
                        const auto& bound = Source_[index];
                        if (!std::isfinite(bound.Lower[axis]) || !std::isfinite(bound.Upper[axis]))
                        {
                            error = "Metal kd-tree encountered a non-finite primitive bound.";
                            return false;
                        }
                        root.Lower[axis] = std::min(root.Lower[axis], bound.Lower[axis]);
                        root.Upper[axis] = std::max(root.Upper[axis], bound.Upper[axis]);
                    }
                }
                std::vector<std::uint32_t> members(Source_.size());
                std::iota(members.begin(), members.end(), 0u);
                if (!Emit(root, members, 0)) return Abort(error);
                return true;
            }

        private:
            bool Check(std::size_t iteration = 0) const
            {
                return iteration % CancellationCheckInterval == 0 && Cancelled_ && Cancelled_();
            }

            bool Abort(std::string& error) const
            {
                error = "Metal kd-tree construction was cancelled.";
                return false;
            }

            PlaneChoice FindPlane(const SpatialNode& cell, const std::vector<std::uint32_t>& members)
            {
                PlaneChoice choice;
                const double area = Surface(cell);
                if (!(area > 0.0)) return choice;
                double winningCost = static_cast<double>(members.size());
                for (std::uint32_t axis = 0; axis < LeafAxis; ++axis)
                {
                    std::array<float, CandidateCount> planes;
                    for (std::size_t candidate = 0; candidate < CandidateCount; ++candidate)
                    {
                        const double fraction = static_cast<double>(candidate + 1) / (CandidateCount + 1);
                        planes[candidate] = static_cast<float>(cell.Lower[axis]
                            + fraction * (static_cast<double>(cell.Upper[axis]) - cell.Lower[axis]));
                    }
                    std::array<std::size_t, CandidateCount + 1> lowerStarts{}, upperEnds{};
                    for (std::size_t member = 0; member < members.size(); ++member)
                    {
                        if (Check(member)) return {};
                        const auto& primitive = Source_[members[member]];
                        const auto firstLower = std::lower_bound(planes.begin(), planes.end(), primitive.Lower[axis]);
                        const auto afterUpper = std::upper_bound(planes.begin(), planes.end(), primitive.Upper[axis]);
                        ++lowerStarts[static_cast<std::size_t>(firstLower - planes.begin())];
                        ++upperEnds[static_cast<std::size_t>(afterUpper - planes.begin())];
                    }
                    std::size_t lowerCount = 0, upperCount = members.size();
                    for (std::size_t candidate = 0; candidate < CandidateCount; ++candidate)
                    {
                        lowerCount += lowerStarts[candidate];
                        upperCount -= upperEnds[candidate];
                        const float plane = planes[candidate];
                        if (!(cell.Lower[axis] < plane && plane < cell.Upper[axis])) continue;
                        if (lowerCount == members.size() && upperCount == members.size()) continue;
                        const auto addedReferences = lowerCount + upperCount - members.size();
                        if (addedReferences > ReferenceBudget_ - PlannedReferences_) continue;
                        SpatialNode lowerCell = cell, upperCell = cell;
                        lowerCell.Upper[axis] = plane;
                        upperCell.Lower[axis] = plane;
                        const double cost = RelativeTraversalCost
                            + (lowerCount * Surface(lowerCell) + upperCount * Surface(upperCell)) / area;
                        if (cost < winningCost)
                        {
                            winningCost = cost;
                            choice = {axis, plane, lowerCount, upperCount};
                        }
                    }
                }
                return choice;
            }

            bool Emit(SpatialNode cell, const std::vector<std::uint32_t>& members, std::uint32_t depth)
            {
                if (Check()) return false;
                const std::size_t destination = Nodes_.size();
                Nodes_.push_back(cell);
                PlaneChoice choice;
                if (members.size() > LeafTarget && depth < DepthLimit && PlannedNodes_ + 2 <= NodeBudget_)
                    choice = FindPlane(cell, members);
                if (Check()) return false;
                if (choice.Axis == LeafAxis)
                {
                    Nodes_[destination].ReferenceBegin = static_cast<std::uint32_t>(References_.size());
                    Nodes_[destination].ReferenceCount = static_cast<std::uint32_t>(members.size());
                    for (std::size_t index = 0; index < members.size(); ++index)
                    {
                        if (Check(index)) return false;
                        References_.push_back(Source_[members[index]].EncodedIndex);
                    }
                    return true;
                }
                PlannedNodes_ += 2;
                PlannedReferences_ += choice.LowerCount + choice.UpperCount - members.size();
                std::vector<std::uint32_t> lowerMembers, upperMembers;
                lowerMembers.reserve(choice.LowerCount);
                upperMembers.reserve(choice.UpperCount);
                for (std::size_t index = 0; index < members.size(); ++index)
                {
                    if (Check(index)) return false;
                    const std::uint32_t member = members[index];
                    if (Source_[member].Lower[choice.Axis] <= choice.Position) lowerMembers.push_back(member);
                    if (Source_[member].Upper[choice.Axis] >= choice.Position) upperMembers.push_back(member);
                }
                Nodes_[destination].Axis = choice.Axis;
                Nodes_[destination].Plane = choice.Position;
                SpatialNode lowerCell = cell, upperCell = cell;
                lowerCell.Upper[choice.Axis] = choice.Position;
                upperCell.Lower[choice.Axis] = choice.Position;
                if (!Emit(lowerCell, lowerMembers, depth + 1)) return false;
                Nodes_[destination].UpperChild = static_cast<std::uint32_t>(Nodes_.size());
                return Emit(upperCell, upperMembers, depth + 1);
            }

            const std::vector<PrimitiveBounds>& Source_;
            const std::function<bool()>& Cancelled_;
            std::vector<SpatialNode>& Nodes_;
            std::vector<std::uint32_t>& References_;
            std::size_t PlannedReferences_;
            const std::size_t ReferenceBudget_;
            const std::size_t NodeBudget_;
            std::size_t PlannedNodes_ = 1;
        };
    }

    bool Build(const std::vector<PrimitiveBounds>& primitives,
        const std::function<bool()>& cancelled, std::vector<SpatialNode>& nodes,
        std::vector<std::uint32_t>& references, std::string& error)
    {
        nodes.clear();
        references.clear();
        if (primitives.empty()) return true;
        Construction construction(primitives, cancelled, nodes, references);
        if (construction.Execute(error)) return true;
        nodes.clear();
        references.clear();
        return false;
    }
}
