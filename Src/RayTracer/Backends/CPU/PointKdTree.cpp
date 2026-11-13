#include "RayTracer/Backends/CPU/PointKdTree.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace Arvo::RayTracing::Cpu
{
// Bentley (1975), "Multidimensional Binary Search Trees Used for Associative
// Searching". Median partitioning bounds depth; largest extent chooses the axis.
// Split-plane pruning is conservative in double precision. The radius query
// returns every closed-ball member once, including duplicate positions.
namespace
{
constexpr uint32_t InvalidNode = std::numeric_limits<uint32_t>::max();
constexpr uint32_t CancellationInterval = 1024;
bool Finite(Float3 point)
{
    return std::isfinite(point.X) && std::isfinite(point.Y) && std::isfinite(point.Z);
}
double Axis(Float3 point, unsigned axis)
{
    return axis == 0 ? point.X : axis == 1 ? point.Y : point.Z;
}
double SquaredDistance(Float3 a, Float3 b)
{
    const double x = double(a.X) - b.X, y = double(a.Y) - b.Y, z = double(a.Z) - b.Z;
    return x * x + y * y + z * z;
}
} // namespace
struct PointKdTree::Impl
{
    struct Node
    {
        Float3 Position;
        uint32_t Original, Left = InvalidNode, Right = InvalidNode;
        unsigned SplitAxis = 0;
    };
    std::vector<Node> Nodes;
    uint32_t Root = InvalidNode;
    Cancellation Probe;
    bool Cancelled = false;
    bool Stop()
    {
        Cancelled = Cancelled || (Probe && Probe());
        return Cancelled;
    }
    uint32_t Partition(uint32_t begin, uint32_t end)
    {
        if (begin == end || Stop())
            return InvalidNode;
        Float3 lower = Nodes[begin].Position, upper = lower;
        for (uint32_t index = begin + 1; index < end; ++index)
        {
            if ((index - begin) % CancellationInterval == 0 && Stop())
                return InvalidNode;
            const Float3 point = Nodes[index].Position;
            lower = {std::min(lower.X, point.X), std::min(lower.Y, point.Y), std::min(lower.Z, point.Z)};
            upper = {std::max(upper.X, point.X), std::max(upper.Y, point.Y), std::max(upper.Z, point.Z)};
        }
        unsigned axis = 0;
        for (unsigned candidate = 1; candidate < 3; ++candidate)
            if (Axis(upper, candidate) - Axis(lower, candidate) > Axis(upper, axis) - Axis(lower, axis))
                axis = candidate;
        const uint32_t median = begin + (end - begin) / 2;
        std::nth_element(Nodes.begin() + begin, Nodes.begin() + median, Nodes.begin() + end,
                         [axis](const Node& a, const Node& b) {
                             const double first = Axis(a.Position, axis), second = Axis(b.Position, axis);
                             return first == second ? a.Original < b.Original : first < second;
                         });
        Nodes[median].SplitAxis = axis;
        Nodes[median].Left = Partition(begin, median);
        Nodes[median].Right = Partition(median + 1, end);
        return median;
    }
    void Visit(uint32_t index, Float3 center, double radiusSquared, const Visitor& visitor) const
    {
        if (index == InvalidNode)
            return;
        const Node& node = Nodes[index];
        const double distance = SquaredDistance(center, node.Position);
        if (distance <= radiusSquared)
            visitor(node.Original, distance);
        const double separation = Axis(center, node.SplitAxis) - Axis(node.Position, node.SplitAxis);
        const uint32_t near = separation < 0 ? node.Left : node.Right,
                       far = separation < 0 ? node.Right : node.Left;
        Visit(near, center, radiusSquared, visitor);
        if (separation * separation <= radiusSquared)
            Visit(far, center, radiusSquared, visitor);
    }
};
PointKdTree::PointKdTree(std::vector<Float3> positions, Cancellation cancellation)
    : Data(std::make_unique<Impl>())
{
    if (positions.size() >= InvalidNode)
        throw std::invalid_argument("Point kd-tree exceeds its index range.");
    Data->Probe = std::move(cancellation);
    Data->Nodes.reserve(positions.size());
    for (uint32_t index = 0; index < positions.size(); ++index)
    {
        if (index % CancellationInterval == 0 && Data->Stop())
            break;
        if (!Finite(positions[index]))
            throw std::invalid_argument("Point kd-tree requires finite positions.");
        Data->Nodes.push_back({positions[index], index});
    }
    Data->Root = Data->Partition(0, uint32_t(Data->Nodes.size()));
    Data->Probe = {};
    if (Data->Cancelled)
    {
        Data->Nodes.clear();
        Data->Root = InvalidNode;
    }
}
PointKdTree::~PointKdTree() = default;
PointKdTree::PointKdTree(PointKdTree&&) noexcept = default;
PointKdTree& PointKdTree::operator=(PointKdTree&&) noexcept = default;
size_t PointKdTree::Size() const
{
    return Data ? Data->Nodes.size() : 0;
}
bool PointKdTree::WasCancelled() const
{
    return Data && Data->Cancelled;
}
void PointKdTree::Search(Float3 center, double squaredRadius, const Visitor& visitor) const
{
    if (WasCancelled())
        throw std::invalid_argument("A cancelled point index cannot serve queries.");
    if (!Finite(center) || !std::isfinite(squaredRadius) || squaredRadius < 0 || !visitor)
        throw std::invalid_argument(
            "Point kd-tree requires a finite center, nonnegative squared radius and visitor.");
    if (Data)
        Data->Visit(Data->Root, center, squaredRadius, visitor);
}
} // namespace Arvo::RayTracing::Cpu
