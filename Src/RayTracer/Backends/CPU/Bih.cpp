#include "RayTracer/Backends/CPU/CpuTransport.h"
#include <algorithm>
#include <array>
#include <numeric>
#include <stdexcept>

namespace Arvo::RayTracing::Cpu
{
namespace
{
// A bounding interval hierarchy partitions OBJECTS, retaining each reference
// exactly once. Two parallel planes clip the inherited cell: x <= leftMaximum
// and x >= rightMinimum. Their intervals may overlap or leave an empty gap.
// This implementation uses midpoint centroid partitioning with a median repair
// for an empty side. It is independent of the CPU BVH and spatial kd-tree.
// Wachter and Keller, "Instant Ray Tracing: The Bounding Interval Hierarchy" (2006).
// https://doi.org/10.2312/EGWR/EGSR06/139-149
constexpr uint32_t LeafPrimitiveLimit = 6;
constexpr uint32_t MaximumDepth = 48;
constexpr uint32_t LeafAxis = 3;
constexpr uint32_t CancellationPollInterval = 2048;

float Component(const Float3 &value, uint32_t axis)
{
    return axis == 0 ? value.X : axis == 1 ? value.Y : value.Z;
}
struct Bounds
{
    Float3 Minimum = {std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
                      std::numeric_limits<float>::max()};
    Float3 Maximum = {std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(),
                      std::numeric_limits<float>::lowest()};
    void Include(const Float3 &point)
    {
        Minimum = {std::min(Minimum.X, point.X), std::min(Minimum.Y, point.Y), std::min(Minimum.Z, point.Z)};
        Maximum = {std::max(Maximum.X, point.X), std::max(Maximum.Y, point.Y), std::max(Maximum.Z, point.Z)};
    }
    double Center(uint32_t axis) const
    {
        return (double(Component(Minimum, axis)) + Component(Maximum, axis)) * 0.5;
    }
};
struct Node
{
    float LeftMaximum = 0, RightMinimum = 0;
    uint32_t First = 0, Count = 0, Axis = LeafAxis;
};
struct Visit
{
    uint32_t Index;
    float Near, Far;
};
bool ClipRoot(const Bounds &bounds, const Ray &ray, Visit &interval)
{
    interval = {0, ray.MinDistance, ray.MaxDistance};
    for (uint32_t axis = 0; axis < 3; ++axis)
    {
        const float origin = Component(ray.Origin, axis), direction = Component(ray.Direction, axis);
        if (direction == 0)
        {
            if (origin < Component(bounds.Minimum, axis) || origin > Component(bounds.Maximum, axis))
                return false;
        }
        else
        {
            float near = (Component(bounds.Minimum, axis) - origin) / direction;
            float far = (Component(bounds.Maximum, axis) - origin) / direction;
            if (near > far)
                std::swap(near, far);
            interval.Near = std::max(interval.Near, near);
            interval.Far = std::min(interval.Far, far);
            if (interval.Near > interval.Far)
                return false;
        }
    }
    return true;
}
} // namespace

struct Bih::Impl
{
    const SceneGeometry &Source;
    CancellationProbe Cancellation;
    bool Cancelled = false;
    Bounds Root;
    std::vector<Bounds> PrimitiveBounds;
    std::vector<uint32_t> References;
    std::vector<Node> Nodes;

    Impl(const SceneGeometry &scene, CancellationProbe cancellation)
        : Source(scene), Cancellation(std::move(cancellation))
    {
        const uint64_t count = uint64_t(scene.Triangles.size()) + scene.Spheres.size();
        if (count >= InvalidRayPrimitiveIndex)
            throw std::invalid_argument("CPU BIH exceeds 32-bit primitive indexing");
        PrimitiveBounds.reserve(static_cast<size_t>(count));
        for (const auto &triangle : scene.Triangles)
        {
            if (PrimitiveBounds.size() % CancellationPollInterval == 0 && Stop())
                return;
            Bounds bounds;
            bounds.Include(triangle.Position0);
            bounds.Include(triangle.Position1);
            bounds.Include(triangle.Position2);
            PrimitiveBounds.push_back(bounds);
            Root.Include(bounds.Minimum);
            Root.Include(bounds.Maximum);
        }
        for (const auto &sphere : scene.Spheres)
        {
            if (PrimitiveBounds.size() % CancellationPollInterval == 0 && Stop())
                return;
            Bounds bounds;
            const Float3 radius(sphere.Radius, sphere.Radius, sphere.Radius);
            bounds.Include(sphere.Center - radius);
            bounds.Include(sphere.Center + radius);
            PrimitiveBounds.push_back(bounds);
            Root.Include(bounds.Minimum);
            Root.Include(bounds.Maximum);
        }
        if (Stop() || count == 0)
            return;
        References.resize(static_cast<size_t>(count));
        std::iota(References.begin(), References.end(), 0u);
        Nodes.emplace_back();
        Build(0, 0, static_cast<uint32_t>(count), 0);
    }
    bool Stop()
    {
        Cancelled = Cancelled || (Cancellation && Cancellation());
        return Cancelled;
    }
    void Build(uint32_t nodeIndex, uint32_t begin, uint32_t count, uint32_t depth)
    {
        if (Stop())
            return;
        // Both children are nonempty: at most 2*N-1 nodes, with no duplicated
        // references. A depth or index limit leaves the entire range intact.
        Nodes[nodeIndex].First = begin;
        Nodes[nodeIndex].Count = count;
        if (count <= LeafPrimitiveLimit || depth == MaximumDepth ||
            Nodes.size() + 2 >= InvalidRayPrimitiveIndex)
            return;
        std::array<double, 3> minimum, maximum;
        minimum.fill(std::numeric_limits<double>::max());
        maximum.fill(std::numeric_limits<double>::lowest());
        for (uint32_t offset = 0; offset < count; ++offset)
        {
            if (offset % CancellationPollInterval == 0 && Stop())
                return;
            const auto &bounds = PrimitiveBounds[References[begin + offset]];
            for (uint32_t axis = 0; axis < 3; ++axis)
            {
                const double center = bounds.Center(axis);
                minimum[axis] = std::min(minimum[axis], center);
                maximum[axis] = std::max(maximum[axis], center);
            }
        }
        uint32_t axis = 0;
        for (uint32_t candidate = 1; candidate < 3; ++candidate)
            if (maximum[candidate] - minimum[candidate] > maximum[axis] - minimum[axis])
                axis = candidate;
        if (!(maximum[axis] > minimum[axis]))
            return;
        const double midpoint = (minimum[axis] + maximum[axis]) * 0.5;
        auto first = References.begin() + begin, last = first + count;
        auto middle = std::partition(
            first, last, [&](uint32_t index) { return PrimitiveBounds[index].Center(axis) < midpoint; });
        if (Stop())
            return;
        if (middle == first || middle == last)
        {
            middle = first + count / 2;
            std::nth_element(first, middle, last, [&](uint32_t a, uint32_t b) {
                const double ca = PrimitiveBounds[a].Center(axis), cb = PrimitiveBounds[b].Center(axis);
                return ca == cb ? a < b : ca < cb;
            });
        }
        float leftMaximum = std::numeric_limits<float>::lowest();
        float rightMinimum = std::numeric_limits<float>::max();
        for (auto current = first; current != last; ++current)
        {
            if (static_cast<size_t>(current - first) % CancellationPollInterval == 0 && Stop())
                return;
            const auto &bounds = PrimitiveBounds[*current];
            if (current < middle)
                leftMaximum = std::max(leftMaximum, Component(bounds.Maximum, axis));
            else
                rightMinimum = std::min(rightMinimum, Component(bounds.Minimum, axis));
        }
        const uint32_t leftCount = static_cast<uint32_t>(middle - first);
        const uint32_t children = static_cast<uint32_t>(Nodes.size());
        Nodes.resize(Nodes.size() + 2);
        Nodes[nodeIndex] = {leftMaximum, rightMinimum, children, 0, axis};
        Build(children, begin, leftCount, depth + 1);
        Build(children + 1, begin + leftCount, count - leftCount, depth + 1);
    }
    bool Intersect(const Ray &ray, SurfaceHit &closest, uint64_t &tests, bool anyHit) const
    {
        Visit root;
        if (Nodes.empty() || !ClipRoot(Root, ray, root))
            return false;
        // One deferred sibling per level; the construction depth proves capacity.
        std::array<Visit, MaximumDepth + 1> stack;
        uint32_t size = 0;
        stack[size++] = root;
        bool found = false;
        while (size)
        {
            const Visit parent = stack[--size];
            if (parent.Near > closest.Distance)
                continue;
            const Node &node = Nodes[parent.Index];
            if (node.Axis == LeafAxis)
            {
                for (uint32_t offset = 0; offset < node.Count; ++offset)
                {
                    const uint32_t index = References[node.First + offset];
                    ++tests;
                    // Keep the original open primitive interval. BIH clipping
                    // restricts visits, not which root of a sphere can be hit.
                    const bool accepted =
                        index < Source.Triangles.size()
                            ? IntersectTriangle(Source.Triangles[index], index, ray, closest)
                            : IntersectSphere(Source.Spheres[index - Source.Triangles.size()], index, ray,
                                              closest);
                    if (accepted)
                    {
                        found = true;
                        if (anyHit)
                            return true;
                    }
                }
                continue;
            }
            Visit left = {node.First, parent.Near, parent.Far};
            Visit right = {node.First + 1, parent.Near, parent.Far};
            const float origin = Component(ray.Origin, node.Axis),
                        direction = Component(ray.Direction, node.Axis);
            bool visitLeft = true, visitRight = true;
            if (direction == 0)
            {
                visitLeft = origin <= node.LeftMaximum;
                visitRight = origin >= node.RightMinimum;
            }
            else
            {
                const float leftPlane = (node.LeftMaximum - origin) / direction;
                const float rightPlane = (node.RightMinimum - origin) / direction;
                if (direction > 0)
                {
                    left.Far = std::min(left.Far, leftPlane);
                    right.Near = std::max(right.Near, rightPlane);
                }
                else
                {
                    left.Near = std::max(left.Near, leftPlane);
                    right.Far = std::min(right.Far, rightPlane);
                }
                visitLeft = left.Near <= left.Far;
                visitRight = right.Near <= right.Far;
            }
            // Independent tests are necessary: a ray in an overlap visits both;
            // a ray parallel to the planes in a gap visits neither.
            if (visitLeft && visitRight)
            {
                if (left.Near > right.Near)
                    std::swap(left, right);
                stack[size++] = right;
                stack[size++] = left;
            }
            else if (visitLeft)
                stack[size++] = left;
            else if (visitRight)
                stack[size++] = right;
        }
        return found;
    }
};

Bih::Bih(const SceneGeometry &scene, CancellationProbe cancellation)
    : Implementation(std::make_unique<Impl>(scene, std::move(cancellation)))
{
}
Bih::~Bih() = default;
bool Bih::WasCancelled() const
{
    return Implementation->Cancelled;
}
bool Bih::Intersect(const Ray &ray, SurfaceHit &hit, uint64_t &tests, bool anyHit) const
{
    return Implementation->Intersect(ray, hit, tests, anyHit);
}
bool Bih::Occluded(const Ray &ray, uint64_t &tests) const
{
    SurfaceHit hit;
    hit.Distance = ray.MaxDistance;
    return Intersect(ray, hit, tests, true);
}
} // namespace Arvo::RayTracing::Cpu
