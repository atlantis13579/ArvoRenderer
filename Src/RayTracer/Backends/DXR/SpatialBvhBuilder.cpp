#include "SpatialBvhBuilder.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace Arvo::RayTracing::Dxr
{
namespace
{
// Explicit SBVH construction for DXR's software traversal. Driver-built BLASes
// do not expose a spatial-split policy. Object partitions and clipped spatial
// references compete by SAH; source geometry and primitive IDs never change.
// Box-only clipping trades build quality for a simple conservative construction.
// Stich, Friedrich, Dietrich (2009), "Spatial Splits in Bounding Volume Hierarchies".
// https://www.nvidia.com/docs/IO/77714/sbvh.pdf
constexpr uint32_t CandidateCount = 16, LeafSize = 8;
constexpr uint64_t ReferenceMultiplier = 8;
constexpr uint64_t NodeLimit = UINT32_MAX / sizeof(SpatialBvhNode);
constexpr size_t CancellationPeriod = 1024;
constexpr double TraversalCost = 0.5;
float Component(const Float3 &v, uint32_t axis)
{
    return axis == 0 ? v.X : axis == 1 ? v.Y : v.Z;
}
void SetComponent(Float3 &v, uint32_t axis, float value)
{
    if (axis == 0)
        v.X = value;
    else if (axis == 1)
        v.Y = value;
    else
        v.Z = value;
}
struct Bounds
{
    Float3 Minimum{std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
                   std::numeric_limits<float>::max()};
    Float3 Maximum{std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(),
                   std::numeric_limits<float>::lowest()};
    void Add(const Bounds &b)
    {
        Minimum = {std::min(Minimum.X, b.Minimum.X), std::min(Minimum.Y, b.Minimum.Y),
                   std::min(Minimum.Z, b.Minimum.Z)};
        Maximum = {std::max(Maximum.X, b.Maximum.X), std::max(Maximum.Y, b.Maximum.Y),
                   std::max(Maximum.Z, b.Maximum.Z)};
    }
    double Area() const
    {
        const double x = double(Maximum.X) - Minimum.X, y = double(Maximum.Y) - Minimum.Y,
                     z = double(Maximum.Z) - Minimum.Z;
        return 2 * (x * y + y * z + z * x);
    }
};
struct Reference
{
    Bounds Box;
    uint32_t Primitive;
};
struct WorkItem
{
    std::vector<Reference> References;
    uint32_t Node, Depth;
};
struct Partition
{
    bool Valid = false, Spatial = false;
    uint32_t Axis = 0;
    double Position = 0, Cost = 0;
};
double Centroid(const Bounds &b, uint32_t axis)
{
    return (double(Component(b.Minimum, axis)) + Component(b.Maximum, axis)) / 2;
}
class Construction
{
    struct Branch
    {
        Float3 Minimum;
        uint32_t First = 0;
        Float3 Maximum;
        uint32_t Count = 0;
    };
    std::vector<Branch> Branches;
    SpatialBvhData &Tree;
    const std::function<bool()> &Cancelled;
    std::string &Error;
    size_t Operations = 0;
    uint64_t ReferenceBudget = 0, ReservedReferences = 0;
    bool Stop()
    {
        if (Cancelled && Cancelled())
        {
            Error = "DXR spatial BVH construction was cancelled";
            return true;
        }
        return false;
    }
    bool Poll() { return ++Operations % CancellationPeriod == 0 && Stop(); }
    bool WriteTraversal(uint32_t source)
    {
        if (Poll()) return false;
        const Branch& branch = Branches[source];
        const uint32_t index = uint32_t(Tree.Nodes.size());
        Tree.Nodes.push_back({branch.Minimum, branch.First, branch.Maximum, branch.Count});
        if (branch.Count == 0)
        {
            if (!WriteTraversal(branch.First) || !WriteTraversal(branch.First + 1)) return false;
            Tree.Nodes[index].FirstOrEscape = uint32_t(Tree.Nodes.size());
        }
        return true;
    }
    bool Choose(const std::vector<Reference> &refs, const Bounds &parent, Partition &best)
    {
        const double parentArea = parent.Area();
        best.Cost = parentArea * refs.size();
        for (uint32_t axis = 0; axis < 3; ++axis)
        {
            double centroidMin = std::numeric_limits<double>::max(), centroidMax = -centroidMin;
            for (const Reference &ref : refs)
            {
                if (Poll())
                    return false;
                const double c = Centroid(ref.Box, axis);
                centroidMin = std::min(centroidMin, c);
                centroidMax = std::max(centroidMax, c);
            }
            for (uint32_t candidate = 1; candidate < CandidateCount; ++candidate)
            {
                const double fraction = double(candidate) / CandidateCount;
                const double objectPlane = centroidMin + (centroidMax - centroidMin) * fraction;
                const float spatialPlane = float(
                    Component(parent.Minimum, axis) +
                    (double(Component(parent.Maximum, axis)) - Component(parent.Minimum, axis)) * fraction);
                Bounds objectLeft, objectRight, spatialLeft, spatialRight;
                uint64_t objectLeftCount = 0, objectRightCount = 0, spatialLeftCount = 0,
                         spatialRightCount = 0;
                for (const Reference &ref : refs)
                {
                    if (Poll())
                        return false;
                    if (Centroid(ref.Box, axis) < objectPlane)
                    {
                        objectLeft.Add(ref.Box);
                        ++objectLeftCount;
                    }
                    else
                    {
                        objectRight.Add(ref.Box);
                        ++objectRightCount;
                    }
                    if (Component(ref.Box.Maximum, axis) <= spatialPlane)
                    {
                        spatialLeft.Add(ref.Box);
                        ++spatialLeftCount;
                    }
                    else if (Component(ref.Box.Minimum, axis) >= spatialPlane)
                    {
                        spatialRight.Add(ref.Box);
                        ++spatialRightCount;
                    }
                    else
                    {
                        Bounds left = ref.Box, right = ref.Box;
                        SetComponent(left.Maximum, axis, spatialPlane);
                        SetComponent(right.Minimum, axis, spatialPlane);
                        spatialLeft.Add(left);
                        spatialRight.Add(right);
                        ++spatialLeftCount;
                        ++spatialRightCount;
                    }
                }
                if (objectLeftCount && objectRightCount)
                {
                    const double cost = TraversalCost * parentArea + objectLeft.Area() * objectLeftCount +
                                        objectRight.Area() * objectRightCount;
                    if (cost < best.Cost)
                        best = {true, false, axis, objectPlane, cost};
                }
                const uint64_t additional = spatialLeftCount + spatialRightCount - refs.size();
                if (spatialLeftCount && spatialRightCount &&
                    additional <= ReferenceBudget - ReservedReferences &&
                    spatialPlane > Component(parent.Minimum, axis) &&
                    spatialPlane < Component(parent.Maximum, axis))
                {
                    const double cost = TraversalCost * parentArea + spatialLeft.Area() * spatialLeftCount +
                                        spatialRight.Area() * spatialRightCount;
                    if (cost < best.Cost)
                        best = {true, true, axis, spatialPlane, cost};
                }
            }
        }
        return true;
    }

  public:
    Construction(SpatialBvhData &tree, const std::function<bool()> &cancelled, std::string &error)
        : Tree(tree), Cancelled(cancelled), Error(error)
    {
    }
    bool Run(const SceneGeometry &scene)
    {
        if (Stop())
            return false;
        const uint64_t total = uint64_t(scene.Triangles.size()) + scene.Spheres.size();
        if (total >= UINT32_MAX)
        {
            Error = "DXR spatial BVH primitive limit exceeded";
            return false;
        }
        if (!total)
            return true;
        std::vector<Reference> references;
        references.reserve(size_t(total));
        for (const Triangle &triangle : scene.Triangles)
        {
            if (Poll())
                return false;
            Bounds box;
            for (const Float3 &vertex : {triangle.Position0, triangle.Position1, triangle.Position2})
                box.Add({vertex, vertex});
            references.push_back({box, uint32_t(references.size())});
        }
        for (const Sphere &sphere : scene.Spheres)
        {
            if (Poll())
                return false;
            Bounds box;
            for (uint32_t axis = 0; axis < 3; ++axis)
            {
                SetComponent(box.Minimum, axis,
                             std::nextafter(Component(sphere.Center, axis) - sphere.Radius,
                                            std::numeric_limits<float>::lowest()));
                SetComponent(box.Maximum, axis,
                             std::nextafter(Component(sphere.Center, axis) + sphere.Radius,
                                            std::numeric_limits<float>::max()));
            }
            references.push_back({box, uint32_t(references.size())});
        }
        for (const Reference &ref : references)
        {
            if (Poll())
                return false;
            for (uint32_t axis = 0; axis < 3; ++axis)
                if (!std::isfinite(Component(ref.Box.Minimum, axis)) ||
                    !std::isfinite(Component(ref.Box.Maximum, axis)) ||
                    Component(ref.Box.Minimum, axis) > Component(ref.Box.Maximum, axis))
                {
                    Error = "Invalid DXR spatial BVH primitive bounds";
                    return false;
                }
        }
        ReservedReferences = total;
        ReferenceBudget = std::min<uint64_t>(UINT32_MAX, total * ReferenceMultiplier);
        std::vector<WorkItem> pending;
        pending.push_back({std::move(references), 0, 0});
        Branches.resize(1);
        while (!pending.empty())
        {
            if (Stop())
                return false;
            WorkItem work = std::move(pending.back());
            pending.pop_back();
            Bounds parent;
            for (const Reference &ref : work.References)
            {
                if (Poll())
                    return false;
                parent.Add(ref.Box);
            }
            Branches[work.Node].Minimum = parent.Minimum;
            Branches[work.Node].Maximum = parent.Maximum;
            Partition split;
            if (work.References.size() > LeafSize && work.Depth < SpatialBvhMaximumDepth &&
                Branches.size() + 2 < NodeLimit && parent.Area() > 0)
                if (!Choose(work.References, parent, split))
                    return false;
            if (!split.Valid)
            {
                Branches[work.Node].First = uint32_t(Tree.PrimitiveIndices.size());
                Branches[work.Node].Count = uint32_t(work.References.size());
                for (const Reference &ref : work.References)
                {
                    if (Poll())
                        return false;
                    Tree.PrimitiveIndices.push_back(ref.Primitive);
                }
                continue;
            }
            WorkItem left, right;
            left.Node = uint32_t(Branches.size());
            right.Node = left.Node + 1;
            left.Depth = right.Depth = work.Depth + 1;
            for (const Reference &ref : work.References)
            {
                if (Poll())
                    return false;
                if (!split.Spatial)
                    (Centroid(ref.Box, split.Axis) < split.Position ? left.References : right.References)
                        .push_back(ref);
                else if (Component(ref.Box.Maximum, split.Axis) <= split.Position)
                    left.References.push_back(ref);
                else if (Component(ref.Box.Minimum, split.Axis) >= split.Position)
                    right.References.push_back(ref);
                else
                {
                    Reference a = ref, b = ref;
                    SetComponent(a.Box.Maximum, split.Axis, float(split.Position));
                    SetComponent(b.Box.Minimum, split.Axis, float(split.Position));
                    left.References.push_back(a);
                    right.References.push_back(b);
                }
            }
            if (split.Spatial)
                ++Tree.SpatialSplits;
            ReservedReferences += left.References.size() + right.References.size() - work.References.size();
            Branches[work.Node].First = left.Node;
            Branches.resize(Branches.size() + 2);
            pending.push_back(std::move(right));
            pending.push_back(std::move(left));
        }
        Tree.Minimum = Branches[0].Minimum;
        Tree.Maximum = Branches[0].Maximum;
        return WriteTraversal(0) && !Stop();
    }
};
} // namespace
bool BuildSpatialBvh(const SceneGeometry &scene, const std::function<bool()> &cancelled, SpatialBvhData &tree,
                     std::string &error)
{
    tree = {};
    if (Construction(tree, cancelled, error).Run(scene))
        return true;
    tree = {};
    return false;
}
} // namespace Arvo::RayTracing::Dxr
