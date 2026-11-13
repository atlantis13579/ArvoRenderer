#include "RayTracer/Backends/CPU/CpuTransport.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>

namespace Arvo::RayTracing::Cpu
{
namespace
{
// A spatial kd-tree partitions the parent CELL with an axis-aligned plane.
// Straddling primitive references belong to both children. A binned spatial
// SAH chooses the plane; neither centroid sorting nor child AABBs define it.
// Reference/depth/node limits terminate refinement in ordinary kd leaves.
// Traversal carries cell intervals, while primitive tests retain the original
// open ray interval (clipping a sphere to a cell could discard its other root).
// Reference: PBRT 3e, "Kd-Tree Accelerator".
// https://pbr-book.org/3ed-2018/Primitives_and_Intersection_Acceleration/Kd-Tree_Accelerator
constexpr uint32_t CandidateBins = 16;
constexpr uint32_t LeafPrimitiveLimit = 4;
constexpr uint32_t MaximumDepth = 48;
constexpr uint32_t ReferenceMultiplier = 4;
constexpr uint32_t CancellationPollInterval = 4096;
constexpr uint32_t LeafAxis = 3;
constexpr double RelativeTraversalCost = 0.5;

float Axis(const Float3 &value, uint32_t axis)
{
    return axis == 0 ? value.X : axis == 1 ? value.Y : value.Z;
}
void SetAxis(Float3 &value, uint32_t axis, float coordinate)
{
    if (axis == 0)
        value.X = coordinate;
    else if (axis == 1)
        value.Y = coordinate;
    else
        value.Z = coordinate;
}
struct Cell
{
    Float3 Minimum = {std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
                      std::numeric_limits<float>::max()};
    Float3 Maximum = {-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(),
                      -std::numeric_limits<float>::max()};
    void Include(const Float3 &point)
    {
        Minimum = {std::min(Minimum.X, point.X), std::min(Minimum.Y, point.Y), std::min(Minimum.Z, point.Z)};
        Maximum = {std::max(Maximum.X, point.X), std::max(Maximum.Y, point.Y), std::max(Maximum.Z, point.Z)};
    }
    double Area() const
    {
        const double x = double(Maximum.X) - Minimum.X, y = double(Maximum.Y) - Minimum.Y,
                     z = double(Maximum.Z) - Minimum.Z;
        return 2 * (x * y + y * z + z * x);
    }
};
struct Node
{
    uint32_t Axis = LeafAxis;
    float Split = 0;
    uint32_t First = 0;
    uint32_t Count = 0;
};
struct Visit
{
    uint32_t NodeIndex;
    float Near, Far;
};
bool RootInterval(const Cell &bounds, const Ray &ray, float &near, float &far)
{
    near = ray.MinDistance;
    far = ray.MaxDistance;
    for (uint32_t axis = 0; axis < 3; ++axis)
    {
        const float origin = Axis(ray.Origin, axis), direction = Axis(ray.Direction, axis);
        if (direction == 0)
        {
            if (origin < Axis(bounds.Minimum, axis) || origin > Axis(bounds.Maximum, axis))
                return false;
            continue;
        }
        float first = (Axis(bounds.Minimum, axis) - origin) / direction;
        float second = (Axis(bounds.Maximum, axis) - origin) / direction;
        if (first > second)
            std::swap(first, second);
        near = std::max(near, first);
        far = std::min(far, second);
        if (near > far)
            return false;
    }
    return true;
}
} // namespace

struct KdTree::Impl
{
    const SceneGeometry &Source;
    CancellationProbe Cancellation;
    bool Cancelled = false;
    Cell Root;
    std::vector<Cell> PrimitiveBounds;
    std::vector<Node> Nodes;
    std::vector<uint32_t> References;
    size_t AllocatedReferences = 0;
    size_t ReferenceBudget = 0;
    size_t NodeBudget = 0;

    Impl(const SceneGeometry &scene, CancellationProbe cancellation)
        : Source(scene), Cancellation(std::move(cancellation))
    {
        PrimitiveBounds.reserve(scene.Triangles.size() + scene.Spheres.size());
        for (const auto &triangle : scene.Triangles)
        {
            if (PrimitiveBounds.size() % CancellationPollInterval == 0 && Stop())
                return;
            Cell bounds;
            bounds.Include(triangle.Position0);
            bounds.Include(triangle.Position1);
            bounds.Include(triangle.Position2);
            PrimitiveBounds.push_back(bounds);
        }
        for (const auto &sphere : scene.Spheres)
        {
            if (PrimitiveBounds.size() % CancellationPollInterval == 0 && Stop())
                return;
            Cell bounds;
            const Float3 radius(sphere.Radius, sphere.Radius, sphere.Radius);
            bounds.Include(sphere.Center - radius);
            bounds.Include(sphere.Center + radius);
            PrimitiveBounds.push_back(bounds);
        }
        if (PrimitiveBounds.empty() || Stop())
            return;
        for (const auto &bounds : PrimitiveBounds)
        {
            Root.Include(bounds.Minimum);
            Root.Include(bounds.Maximum);
        }
        AllocatedReferences = PrimitiveBounds.size();
        ReferenceBudget = PrimitiveBounds.size() * ReferenceMultiplier;
        NodeBudget = ReferenceBudget * 2 + MaximumDepth * 2 + 1;
        std::vector<uint32_t> indices(PrimitiveBounds.size());
        std::iota(indices.begin(), indices.end(), 0u);
        Nodes.emplace_back();
        Build(0, Root, indices, 0);
    }
    bool Stop()
    {
        Cancelled = Cancelled || (Cancellation && Cancellation());
        return Cancelled;
    }
    void Leaf(uint32_t node, const std::vector<uint32_t> &indices)
    {
        Nodes[node].First = static_cast<uint32_t>(References.size());
        Nodes[node].Count = static_cast<uint32_t>(indices.size());
        References.insert(References.end(), indices.begin(), indices.end());
    }
    void Build(uint32_t node, const Cell &cell, const std::vector<uint32_t> &indices, uint32_t depth)
    {
        if (Stop())
            return;
        if (indices.size() <= LeafPrimitiveLimit || depth == MaximumDepth || Nodes.size() + 2 > NodeBudget)
        {
            Leaf(node, indices);
            return;
        }
        const double area = cell.Area();
        if (!(area > 0))
        {
            Leaf(node, indices);
            return;
        }
        double bestCost = static_cast<double>(indices.size());
        uint32_t bestAxis = LeafAxis;
        float bestSplit = 0;
        for (uint32_t axis = 0; axis < 3; ++axis)
        {
            const double minimum = Axis(cell.Minimum, axis),
                         extent = double(Axis(cell.Maximum, axis)) - minimum;
            if (!(extent > 0))
                continue;
            for (uint32_t bin = 1; bin < CandidateBins; ++bin)
            {
                const float split = static_cast<float>(minimum + extent * bin / CandidateBins);
                if (split <= minimum || split >= Axis(cell.Maximum, axis))
                    continue;
                size_t below = 0, above = 0;
                for (size_t index = 0; index < indices.size(); ++index)
                {
                    if (index % CancellationPollInterval == 0 && Stop())
                        return;
                    const auto &bounds = PrimitiveBounds[indices[index]];
                    below += Axis(bounds.Minimum, axis) <= split;
                    above += Axis(bounds.Maximum, axis) >= split;
                }
                const size_t duplication = below + above - indices.size();
                if (duplication > ReferenceBudget - AllocatedReferences)
                    continue;
                Cell lower = cell, upper = cell;
                SetAxis(lower.Maximum, axis, split);
                SetAxis(upper.Minimum, axis, split);
                const double cost =
                    RelativeTraversalCost + (lower.Area() * below + upper.Area() * above) / area;
                if (cost < bestCost)
                {
                    bestCost = cost;
                    bestAxis = axis;
                    bestSplit = split;
                }
            }
        }
        if (bestAxis == LeafAxis)
        {
            Leaf(node, indices);
            return;
        }
        std::vector<uint32_t> below, above;
        below.reserve(indices.size());
        above.reserve(indices.size());
        for (uint32_t index : indices)
        {
            if (Axis(PrimitiveBounds[index].Minimum, bestAxis) <= bestSplit)
                below.push_back(index);
            if (Axis(PrimitiveBounds[index].Maximum, bestAxis) >= bestSplit)
                above.push_back(index);
        }
        AllocatedReferences += below.size() + above.size() - indices.size();
        const uint32_t first = static_cast<uint32_t>(Nodes.size());
        Nodes.resize(Nodes.size() + 2);
        Nodes[node].Axis = bestAxis;
        Nodes[node].Split = bestSplit;
        Nodes[node].First = first;
        Cell lower = cell, upper = cell;
        SetAxis(lower.Maximum, bestAxis, bestSplit);
        SetAxis(upper.Minimum, bestAxis, bestSplit);
        Build(first, lower, below, depth + 1);
        Build(first + 1, upper, above, depth + 1);
    }
    bool Intersect(const Ray &ray, SurfaceHit &hit, uint64_t &tests, bool anyHit) const
    {
        if (Nodes.empty())
            return false;
        float near, far;
        if (!RootInterval(Root, ray, near, far))
            return false;
        // One pending sibling per depth. Builder depth bound proves capacity.
        std::array<Visit, MaximumDepth + 1> pending;
        uint32_t count = 0;
        pending[count++] = {0, near, far};
        bool found = false;
        while (count)
        {
            Visit visit = pending[--count];
            if (visit.Near > hit.Distance || visit.Near > visit.Far)
                continue;
            const Node &node = Nodes[visit.NodeIndex];
            if (node.Axis == LeafAxis)
            {
                for (uint32_t offset = 0; offset < node.Count; ++offset)
                {
                    const uint32_t primitive = References[node.First + offset];
                    ++tests;
                    const bool accepted =
                        primitive < Source.Triangles.size()
                            ? IntersectTriangle(Source.Triangles[primitive], primitive, ray, hit)
                            : IntersectSphere(Source.Spheres[primitive - Source.Triangles.size()], primitive,
                                              ray, hit);
                    if (accepted)
                    {
                        found = true;
                        if (anyHit)
                            return true;
                    }
                }
                continue;
            }
            const float origin = Axis(ray.Origin, node.Axis), direction = Axis(ray.Direction, node.Axis);
            if (direction == 0)
            {
                if (origin <= node.Split)
                    pending[count++] = {node.First, visit.Near, visit.Far};
                if (origin >= node.Split)
                    pending[count++] = {node.First + 1, visit.Near, visit.Far};
                continue;
            }
            const uint32_t first = direction > 0 ? node.First : node.First + 1;
            const uint32_t second = direction > 0 ? node.First + 1 : node.First;
            const float plane = (node.Split - origin) / direction;
            if (plane < visit.Near)
                pending[count++] = {second, visit.Near, visit.Far};
            else if (plane > visit.Far)
                pending[count++] = {first, visit.Near, visit.Far};
            else
            {
                pending[count++] = {second, plane, visit.Far};
                pending[count++] = {first, visit.Near, plane};
            }
        }
        return found;
    }
};

KdTree::KdTree(const SceneGeometry &scene, CancellationProbe cancellation)
    : Implementation(std::make_unique<Impl>(scene, std::move(cancellation)))
{
}
KdTree::~KdTree() = default;
bool KdTree::WasCancelled() const
{
    return Implementation->Cancelled;
}
bool KdTree::Intersect(const Ray &ray, SurfaceHit &hit, uint64_t &tests, bool anyHit) const
{
    return Implementation->Intersect(ray, hit, tests, anyHit);
}
bool KdTree::Occluded(const Ray &ray, uint64_t &tests) const
{
    SurfaceHit hit;
    hit.Distance = ray.MaxDistance;
    return Intersect(ray, hit, tests, true);
}
} // namespace Arvo::RayTracing::Cpu
