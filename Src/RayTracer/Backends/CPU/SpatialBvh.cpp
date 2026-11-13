#include "RayTracer/Backends/CPU/CpuTransport.h"
#include <algorithm>
#include <array>
#include <stdexcept>

namespace Arvo::RayTracing::Cpu
{
namespace
{
// Spatial splits compete with object partitions and leaf cost at every node.
// A crossing reference is duplicated and its two bounds are clipped. Geometry
// remains unchanged; leaves still intersect the original primitive. Conservative
// reference-box clipping saves build work but gives looser bounds than polygon clipping.
// Stich, Friedrich, Dietrich, "Spatial Splits in Bounding Volume Hierarchies" (2009), sections 4.1-4.2.
// https://www.nvidia.com/docs/IO/77714/sbvh.pdf
constexpr uint32_t CandidateBins = 16, MaximumDepth = 48, LeafSize = 4;
constexpr uint32_t ReferenceMultiplier = 8, CancellationPeriod = 2048;
constexpr double TraversalCost = 0.5;

float Coordinate(const Float3 &value, uint32_t axis)
{
    return axis == 0 ? value.X : axis == 1 ? value.Y : value.Z;
}
void Coordinate(Float3 &value, uint32_t axis, float component)
{
    if (axis == 0)
        value.X = component;
    else if (axis == 1)
        value.Y = component;
    else
        value.Z = component;
}
struct Box
{
    Float3 Min = {std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
                  std::numeric_limits<float>::max()};
    Float3 Max = {std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(),
                  std::numeric_limits<float>::lowest()};
    void Include(Float3 point)
    {
        Min = {std::min(Min.X, point.X), std::min(Min.Y, point.Y), std::min(Min.Z, point.Z)};
        Max = {std::max(Max.X, point.X), std::max(Max.Y, point.Y), std::max(Max.Z, point.Z)};
    }
    void Include(const Box &other)
    {
        Include(other.Min);
        Include(other.Max);
    }
    double Center(uint32_t axis) const
    {
        return (double(Coordinate(Min, axis)) + Coordinate(Max, axis)) * 0.5;
    }
    double Area() const
    {
        const double x = double(Max.X) - Min.X, y = double(Max.Y) - Min.Y, z = double(Max.Z) - Min.Z;
        return 2 * (x * y + x * z + y * z);
    }
};
struct Reference
{
    Box Bounds;
    uint32_t Primitive;
};
struct Node
{
    Box Bounds;
    uint32_t FirstOrEscape = 0, Count = 0;
};
struct Candidate
{
    bool Spatial = false;
    uint32_t Axis = 0;
    float Plane = 0;
    double Cost = 0;
};

bool Overlaps(const Box &bounds, const Ray &ray, float closest)
{
    float near = ray.MinDistance, far = std::min(ray.MaxDistance, closest);
    for (uint32_t axis = 0; axis < 3; ++axis)
    {
        const float origin = Coordinate(ray.Origin, axis), direction = Coordinate(ray.Direction, axis);
        if (direction == 0)
        {
            if (origin < Coordinate(bounds.Min, axis) || origin > Coordinate(bounds.Max, axis))
                return false;
        }
        else
        {
            const float a = (Coordinate(bounds.Min, axis) - origin) / direction;
            const float b = (Coordinate(bounds.Max, axis) - origin) / direction;
            near = std::max(near, std::min(a, b));
            far = std::min(far, std::max(a, b));
            if (near > far)
                return false;
        }
    }
    return true;
}
} // namespace

struct SpatialBvh::Impl
{
    const SceneGeometry &SceneData;
    CancellationProbe Cancellation;
    bool Cancelled = false;
    std::vector<Node> Nodes;
    std::vector<uint32_t> PrimitiveIndices;
    uint64_t ReferenceLimit = 0, ReservedReferences = 0, ReservedNodes = 1, Splits = 0;

    Impl(const SceneGeometry &scene, CancellationProbe cancellation)
        : SceneData(scene), Cancellation(std::move(cancellation))
    {
        const uint64_t count = uint64_t(scene.Triangles.size()) + scene.Spheres.size();
        if (count >= InvalidRayPrimitiveIndex)
            throw std::invalid_argument("CPU spatial BVH exceeds primitive indexing limits");
        if (Stop() || count == 0)
            return;
        ReferenceLimit = std::min<uint64_t>(InvalidRayPrimitiveIndex, count * ReferenceMultiplier);
        ReservedReferences = count;
        std::vector<Reference> references;
        references.reserve(static_cast<size_t>(count));
        for (const auto &triangle : scene.Triangles)
        {
            if (references.size() % CancellationPeriod == 0 && Stop())
                return;
            Box bounds;
            bounds.Include(triangle.Position0);
            bounds.Include(triangle.Position1);
            bounds.Include(triangle.Position2);
            references.push_back({bounds, static_cast<uint32_t>(references.size())});
        }
        for (const auto &sphere : scene.Spheres)
        {
            if (references.size() % CancellationPeriod == 0 && Stop())
                return;
            Box bounds;
            const Float3 radius(sphere.Radius, sphere.Radius, sphere.Radius);
            bounds.Include(sphere.Center - radius);
            bounds.Include(sphere.Center + radius);
            for (uint32_t axis = 0; axis < 3; ++axis)
            {
                Coordinate(
                    bounds.Min, axis,
                    std::nextafter(Coordinate(bounds.Min, axis), std::numeric_limits<float>::lowest()));
                Coordinate(bounds.Max, axis,
                           std::nextafter(Coordinate(bounds.Max, axis), std::numeric_limits<float>::max()));
            }
            references.push_back({bounds, static_cast<uint32_t>(references.size())});
        }
        PrimitiveIndices.reserve(static_cast<size_t>(count));
        Build(references, 0);
        if (Stop())
        {
            Nodes.clear();
            PrimitiveIndices.clear();
        }
    }
    bool Stop()
    {
        Cancelled = Cancelled || (Cancellation && Cancellation());
        return Cancelled;
    }
    void Leaf(uint32_t node, const std::vector<Reference> &references)
    {
        Nodes[node].FirstOrEscape = static_cast<uint32_t>(PrimitiveIndices.size());
        Nodes[node].Count = static_cast<uint32_t>(references.size());
        for (size_t index = 0; index < references.size(); ++index)
        {
            if (index % CancellationPeriod == 0 && Stop())
                return;
            PrimitiveIndices.push_back(references[index].Primitive);
        }
    }
    void Build(const std::vector<Reference> &references, uint32_t depth)
    {
        if (Stop())
            return;
        const uint32_t index = static_cast<uint32_t>(Nodes.size());
        Nodes.emplace_back();
        Box bounds;
        std::array<double, 3> centerMin, centerMax;
        centerMin.fill(std::numeric_limits<double>::max());
        centerMax.fill(std::numeric_limits<double>::lowest());
        for (size_t i = 0; i < references.size(); ++i)
        {
            if (i % CancellationPeriod == 0 && Stop())
                return;
            bounds.Include(references[i].Bounds);
            for (uint32_t axis = 0; axis < 3; ++axis)
            {
                const double center = references[i].Bounds.Center(axis);
                centerMin[axis] = std::min(centerMin[axis], center);
                centerMax[axis] = std::max(centerMax[axis], center);
            }
        }
        Nodes[index].Bounds = bounds;
        const double area = bounds.Area();
        if (references.size() <= LeafSize || depth == MaximumDepth || !(area > 0) ||
            ReservedNodes + 2 >= InvalidRayPrimitiveIndex)
        {
            Leaf(index, references);
            return;
        }
        Candidate best;
        best.Cost = static_cast<double>(references.size());
        bool found = false;
        for (bool spatial : {false, true})
            for (uint32_t axis = 0; axis < 3; ++axis)
            {
                const double minimum = spatial ? Coordinate(bounds.Min, axis) : centerMin[axis];
                const double maximum = spatial ? Coordinate(bounds.Max, axis) : centerMax[axis];
                if (!(maximum > minimum))
                    continue;
                for (uint32_t bin = 1; bin < CandidateBins; ++bin)
                {
                    const float plane =
                        static_cast<float>(minimum + (maximum - minimum) * bin / CandidateBins);
                    if (!(plane > minimum && plane < maximum))
                        continue;
                    Box left, right;
                    uint64_t leftCount = 0, rightCount = 0;
                    for (size_t i = 0; i < references.size(); ++i)
                    {
                        if (i % CancellationPeriod == 0 && Stop())
                            return;
                        const auto &box = references[i].Bounds;
                        if (!spatial)
                        {
                            if (box.Center(axis) < plane)
                            {
                                left.Include(box);
                                ++leftCount;
                            }
                            else
                            {
                                right.Include(box);
                                ++rightCount;
                            }
                        }
                        else if (Coordinate(box.Max, axis) <= plane)
                        {
                            left.Include(box);
                            ++leftCount;
                        }
                        else if (Coordinate(box.Min, axis) >= plane)
                        {
                            right.Include(box);
                            ++rightCount;
                        }
                        else
                        {
                            Box lower = box, upper = box;
                            Coordinate(lower.Max, axis, plane);
                            Coordinate(upper.Min, axis, plane);
                            left.Include(lower);
                            right.Include(upper);
                            ++leftCount;
                            ++rightCount;
                        }
                    }
                    if (leftCount == 0 || rightCount == 0)
                        continue;
                    const uint64_t extra = leftCount + rightCount - references.size();
                    if (extra > ReferenceLimit - ReservedReferences)
                        continue;
                    const double cost =
                        TraversalCost + (left.Area() * leftCount + right.Area() * rightCount) / area;
                    if (cost < best.Cost)
                    {
                        best = {spatial, axis, plane, cost};
                        found = true;
                    }
                }
            }
        if (!found)
        {
            Leaf(index, references);
            return;
        }
        std::vector<Reference> left, right;
        for (size_t i = 0; i < references.size(); ++i)
        {
            if (i % CancellationPeriod == 0 && Stop())
                return;
            const auto &ref = references[i];
            if (!best.Spatial)
            {
                (ref.Bounds.Center(best.Axis) < best.Plane ? left : right).push_back(ref);
            }
            else if (Coordinate(ref.Bounds.Max, best.Axis) <= best.Plane)
                left.push_back(ref);
            else if (Coordinate(ref.Bounds.Min, best.Axis) >= best.Plane)
                right.push_back(ref);
            else
            {
                Reference low = ref, high = ref;
                Coordinate(low.Bounds.Max, best.Axis, best.Plane);
                Coordinate(high.Bounds.Min, best.Axis, best.Plane);
                left.push_back(low);
                right.push_back(high);
            }
        }
        ReservedReferences += left.size() + right.size() - references.size();
        ReservedNodes += 2;
        if (best.Spatial)
            ++Splits;
        Build(left, depth + 1);
        Build(right, depth + 1);
        Nodes[index].FirstOrEscape = static_cast<uint32_t>(Nodes.size());
    }
    bool Intersect(const Ray &ray, SurfaceHit &hit, uint64_t &tests, bool anyHit) const
    {
        bool found = false;
        uint32_t index = 0;
        while (index < Nodes.size())
        {
            const Node &node = Nodes[index];
            if (!Overlaps(node.Bounds, ray, hit.Distance))
            {
                index = node.Count == 0 ? node.FirstOrEscape : index + 1;
                continue;
            }
            for (uint32_t offset = 0; offset < node.Count; ++offset)
            {
                ++tests;
                const uint32_t primitive = PrimitiveIndices[node.FirstOrEscape + offset];
                const bool accepted =
                    primitive < SceneData.Triangles.size()
                        ? IntersectTriangle(SceneData.Triangles[primitive], primitive, ray, hit)
                        : IntersectSphere(SceneData.Spheres[primitive - SceneData.Triangles.size()],
                                          primitive, ray, hit);
                if (accepted)
                {
                    found = true;
                    if (anyHit)
                        return true;
                }
            }
            ++index;
        }
        return found;
    }
};

SpatialBvh::SpatialBvh(const SceneGeometry &scene, CancellationProbe cancellation)
    : Implementation(std::make_unique<Impl>(scene, std::move(cancellation)))
{
}
SpatialBvh::~SpatialBvh() = default;
bool SpatialBvh::WasCancelled() const
{
    return Implementation->Cancelled;
}
bool SpatialBvh::Intersect(const Ray &ray, SurfaceHit &hit, uint64_t &tests, bool anyHit) const
{
    return Implementation->Intersect(ray, hit, tests, anyHit);
}
bool SpatialBvh::Occluded(const Ray &ray, uint64_t &tests) const
{
    SurfaceHit hit;
    hit.Distance = ray.MaxDistance;
    return Intersect(ray, hit, tests, true);
}
uint64_t SpatialBvh::SpatialSplitCount() const
{
    return Implementation->Splits;
}
size_t SpatialBvh::ReferenceCount() const
{
    return Implementation->PrimitiveIndices.size();
}
} // namespace Arvo::RayTracing::Cpu
