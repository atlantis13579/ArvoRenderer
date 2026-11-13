#include "SpatialBvhBuilder.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace Arvo::RayTracing::CudaSpatial
{
namespace
{
// CUDA owns this SBVH construction algorithm and consumes its escape links in
// Bvh.cu. Full sorted object SAH competes with spatial-plane SAH. Spatial splits
// clip reference boxes and duplicate IDs, preserving the source triangles/spheres.
// Reference-box clipping is conservative, not polygon tessellation.
// Stich, Friedrich and Dietrich (2009), "Spatial Splits in Bounding Volume Hierarchies".
// https://www.nvidia.com/docs/IO/77714/sbvh.pdf
constexpr uint32_t MaximumDepth = 48, LeafThreshold = 6, SpatialBins = 12;
constexpr uint64_t ReferenceMultiplier = 8;
constexpr size_t CancellationPeriod = 1024;
constexpr double RelativeTraversalCost = 0.75;
struct Bounds
{
    std::array<float, 3> Minimum, Maximum;
    Bounds()
    {
        Minimum.fill(std::numeric_limits<float>::max());
        Maximum.fill(std::numeric_limits<float>::lowest());
    }
    void Extend(const Primitive &primitive)
    {
        for (uint32_t axis = 0; axis < 3; ++axis)
        {
            Minimum[axis] = std::min(Minimum[axis], primitive.Minimum[axis]);
            Maximum[axis] = std::max(Maximum[axis], primitive.Maximum[axis]);
        }
    }
    double Area() const
    {
        const double x = double(Maximum[0]) - Minimum[0], y = double(Maximum[1]) - Minimum[1],
                     z = double(Maximum[2]) - Minimum[2];
        return 2 * (x * y + y * z + z * x);
    }
};
double Center(const Primitive &primitive, uint32_t axis)
{
    return (double(primitive.Minimum[axis]) + primitive.Maximum[axis]) * 0.5;
}
struct Choice
{
    bool Found = false, Spatial = false;
    uint32_t Axis = 0;
    size_t Rank = 0;
    float Plane = 0;
    double Cost = 0;
};

class Construction
{
    const std::function<bool()> &Cancelled;
    Tree &Output;
    std::string &Error;
    uint64_t ReferenceCount = 0, ReferenceLimit = 0, NodeReservation = 1;
    size_t Operations = 0;
    bool Stop()
    {
        if (Cancelled && Cancelled())
        {
            Error = "CUDA spatial BVH construction was cancelled";
            return true;
        }
        return false;
    }
    bool Poll() { return (++Operations % CancellationPeriod) == 0 && Stop(); }
    void Sort(std::vector<Primitive> &primitives, uint32_t axis)
    {
        std::sort(primitives.begin(), primitives.end(), [axis](const Primitive &a, const Primitive &b) {
            const double first = Center(a, axis), second = Center(b, axis);
            return first == second ? a.Reference < b.Reference : first < second;
        });
    }
    bool Leaf(uint32_t index, const std::vector<Primitive> &primitives)
    {
        Output.Nodes[index].FirstOrEscape = static_cast<uint32_t>(Output.References.size());
        Output.Nodes[index].PrimitiveCount = static_cast<uint32_t>(primitives.size());
        for (const Primitive &primitive : primitives)
        {
            if (Poll())
                return false;
            Output.References.push_back(primitive.Reference);
        }
        return true;
    }
    bool Subdivide(std::vector<Primitive> primitives, uint32_t depth)
    {
        if (Stop())
            return false;
        Bounds parent;
        for (const Primitive &primitive : primitives)
        {
            if (Poll())
                return false;
            parent.Extend(primitive);
        }
        const uint32_t node = static_cast<uint32_t>(Output.Nodes.size());
        Output.Nodes.push_back({parent.Minimum, parent.Maximum, 0, 0});
        const double area = parent.Area();
        if (primitives.size() <= LeafThreshold || depth == MaximumDepth || !(area > 0) ||
            NodeReservation + 2 >= UINT32_MAX)
            return Leaf(node, primitives);

        Choice best;
        best.Cost = area * primitives.size();
        for (uint32_t axis = 0; axis < 3; ++axis)
        {
            Sort(primitives, axis);
            if (Stop())
                return false;
            std::vector<Bounds> suffix(primitives.size());
            Bounds right;
            for (size_t index = primitives.size(); index-- > 0;)
            {
                if (Poll())
                    return false;
                right.Extend(primitives[index]);
                suffix[index] = right;
            }
            Bounds left;
            for (size_t rank = 1; rank < primitives.size(); ++rank)
            {
                if (Poll())
                    return false;
                left.Extend(primitives[rank - 1]);
                const double cost = RelativeTraversalCost * area + left.Area() * rank +
                                    suffix[rank].Area() * (primitives.size() - rank);
                if (cost < best.Cost)
                    best = {true, false, axis, rank, 0, cost};
            }
            const double start = parent.Minimum[axis], length = double(parent.Maximum[axis]) - start;
            for (uint32_t bin = 1; bin < SpatialBins && length > 0; ++bin)
            {
                const float plane = float(start + length * bin / SpatialBins);
                if (!(plane > start && plane < parent.Maximum[axis]))
                    continue;
                Bounds lower, upper;
                uint64_t lowerCount = 0, upperCount = 0;
                for (const Primitive &primitive : primitives)
                {
                    if (Poll())
                        return false;
                    if (primitive.Maximum[axis] <= plane)
                    {
                        lower.Extend(primitive);
                        ++lowerCount;
                    }
                    else if (primitive.Minimum[axis] >= plane)
                    {
                        upper.Extend(primitive);
                        ++upperCount;
                    }
                    else
                    {
                        Primitive a = primitive, b = primitive;
                        a.Maximum[axis] = plane;
                        b.Minimum[axis] = plane;
                        lower.Extend(a);
                        upper.Extend(b);
                        ++lowerCount;
                        ++upperCount;
                    }
                }
                const uint64_t extra = lowerCount + upperCount - primitives.size();
                if (lowerCount == 0 || upperCount == 0 || extra > ReferenceLimit - ReferenceCount)
                    continue;
                const double cost =
                    RelativeTraversalCost * area + lower.Area() * lowerCount + upper.Area() * upperCount;
                if (cost < best.Cost)
                    best = {true, true, axis, 0, plane, cost};
            }
        }
        if (!best.Found)
            return Leaf(node, primitives);
        std::vector<Primitive> lower, upper;
        if (!best.Spatial)
        {
            Sort(primitives, best.Axis);
            if (Stop())
                return false;
            lower.assign(primitives.begin(), primitives.begin() + best.Rank);
            upper.assign(primitives.begin() + best.Rank, primitives.end());
        }
        else
        {
            for (const Primitive &primitive : primitives)
            {
                if (Poll())
                    return false;
                if (primitive.Maximum[best.Axis] <= best.Plane)
                    lower.push_back(primitive);
                else if (primitive.Minimum[best.Axis] >= best.Plane)
                    upper.push_back(primitive);
                else
                {
                    Primitive a = primitive, b = primitive;
                    a.Maximum[best.Axis] = best.Plane;
                    b.Minimum[best.Axis] = best.Plane;
                    lower.push_back(a);
                    upper.push_back(b);
                }
            }
            ++Output.SpatialSplits;
        }
        ReferenceCount += lower.size() + upper.size() - primitives.size();
        NodeReservation += 2;
        if (!Subdivide(std::move(lower), depth + 1) || !Subdivide(std::move(upper), depth + 1))
            return false;
        Output.Nodes[node].FirstOrEscape = static_cast<uint32_t>(Output.Nodes.size());
        return true;
    }

  public:
    Construction(const std::function<bool()> &cancelled, Tree &tree, std::string &error)
        : Cancelled(cancelled), Output(tree), Error(error)
    {
    }
    bool Build(const std::vector<Primitive> &primitives)
    {
        if (Stop())
            return false;
        if (primitives.size() >= UINT32_MAX)
        {
            Error = "CUDA spatial BVH primitive limit exceeded";
            return false;
        }
        if (primitives.empty())
            return true;
        for (const Primitive &primitive : primitives)
        {
            if (Poll())
                return false;
            for (uint32_t axis = 0; axis < 3; ++axis)
                if (!std::isfinite(primitive.Minimum[axis]) || !std::isfinite(primitive.Maximum[axis]) ||
                    primitive.Minimum[axis] > primitive.Maximum[axis])
                {
                    Error = "Invalid CUDA spatial BVH primitive bounds";
                    return false;
                }
        }
        ReferenceCount = primitives.size();
        ReferenceLimit = std::min<uint64_t>(UINT32_MAX, ReferenceCount * ReferenceMultiplier);
        return Subdivide(primitives, 0) && !Stop();
    }
};
} // namespace
bool Build(const std::vector<Primitive> &primitives, const std::function<bool()> &cancelled, Tree &tree,
           std::string &error)
{
    tree = {};
    if (Construction(cancelled, tree, error).Build(primitives))
        return true;
    tree = {};
    return false;
}
} // namespace Arvo::RayTracing::CudaSpatial
