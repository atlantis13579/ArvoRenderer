#include "MotionPreparation.h"
#include "InstancingBuilder.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <unordered_map>

namespace Arvo::RayTracing::ComputeInstances
{
namespace
{
// Independent Compute TLAS construction: binned surface-area partitioning over
// instance bounds, with a median fallback and bounded depth. Object bounds use
// interval arithmetic under the affine transform; vertices are never expanded.
// The inverse comes from a pivoted LU factorization and three basis solves.
// PBRT 4th ed., "Transformed Primitives" and "Bounding Volume Hierarchies".
// https://pbr-book.org/4ed/Primitives_and_Intersection_Acceleration
constexpr uint32_t Dimensions = 3, AffineStride = 4, BinCount = 8;
constexpr uint32_t LeafCapacity = 2, DepthLimit = 32, CancellationStride = 1024;
constexpr uint64_t LocalPrimitiveLimit = (uint64_t{1} << 31) - 1;
constexpr uint64_t InstanceLimit = std::numeric_limits<uint32_t>::max() / 2;
using Point = std::array<double, Dimensions>;
struct Box
{
    Point Min{INFINITY, INFINITY, INFINITY}, Max{-INFINITY, -INFINITY, -INFINITY};
    void Add(const Point &p)
    {
        for (uint32_t a = 0; a < Dimensions; ++a)
        {
            Min[a] = std::min(Min[a], p[a]);
            Max[a] = std::max(Max[a], p[a]);
        }
    }
    void Add(const Box &b)
    {
        Add(b.Min);
        Add(b.Max);
    }
    double Area() const
    {
        if (Min[0] > Max[0])
            return 0;
        const double x = Max[0] - Min[0], y = Max[1] - Min[1], z = Max[2] - Min[2];
        return 2 * (x * y + x * z + y * z);
    }
};
uint64_t PrimitiveCount(const SceneGeometry &object)
{
    return uint64_t(object.Triangles.size()) + object.Spheres.size();
}
bool Invert(const GeometryInstance &transform, Instance &target)
{
    double lu[Dimensions][Dimensions]{};
    uint32_t permutation[Dimensions]{0, 1, 2};
    for (const float entry : transform.ObjectToWorld)
        if (!std::isfinite(entry))
            return false;
    for (uint32_t row = 0; row < Dimensions; ++row)
        for (uint32_t col = 0; col < Dimensions; ++col)
            lu[row][col] = transform.ObjectToWorld[row * AffineStride + col];
    for (uint32_t step = 0; step < Dimensions; ++step)
    {
        uint32_t pivot = step;
        for (uint32_t row = step + 1; row < Dimensions; ++row)
            if (std::abs(lu[row][step]) > std::abs(lu[pivot][step]))
                pivot = row;
        if (lu[pivot][step] == 0)
            return false;
        std::swap(permutation[step], permutation[pivot]);
        for (uint32_t col = 0; col < Dimensions; ++col)
            std::swap(lu[step][col], lu[pivot][col]);
        for (uint32_t row = step + 1; row < Dimensions; ++row)
        {
            lu[row][step] /= lu[step][step];
            for (uint32_t col = step + 1; col < Dimensions; ++col)
                lu[row][col] -= lu[row][step] * lu[step][col];
        }
    }
    for (uint32_t basis = 0; basis < Dimensions; ++basis)
    {
        Point solution{};
        for (uint32_t row = 0; row < Dimensions; ++row)
        {
            solution[row] = permutation[row] == basis ? 1 : 0;
            for (uint32_t col = 0; col < row; ++col)
                solution[row] -= lu[row][col] * solution[col];
        }
        for (uint32_t remaining = Dimensions; remaining != 0; --remaining)
        {
            const uint32_t row = remaining - 1;
            for (uint32_t col = row + 1; col < Dimensions; ++col)
                solution[row] -= lu[row][col] * solution[col];
            solution[row] /= lu[row][row];
            target.Inverse[row * AffineStride + basis] = solution[row];
        }
    }
    for (uint32_t row = 0; row < Dimensions; ++row)
    {
        double translation = 0;
        for (uint32_t col = 0; col < Dimensions; ++col)
            translation -= target.Inverse[row * AffineStride + col] *
                           transform.ObjectToWorld[col * AffineStride + Dimensions];
        target.Inverse[row * AffineStride + Dimensions] = translation;
    }
    for (double entry : target.Inverse)
        if (!std::isfinite(entry) || std::abs(entry) > std::numeric_limits<float>::max())
            return false;
    return true;
}
struct Item
{
    Instance Record;
    Box Bounds;
    Point Center;
};
struct Bin
{
    Box Bounds;
    uint32_t Count = 0;
};
class Builder
{
    const std::function<bool()> &Cancelled;
    Hierarchy &Result;
    std::string &Error;
    std::unordered_map<const SceneGeometry *, uint32_t> ObjectLookup;
    std::vector<Box> LocalBounds;
    std::vector<Item> Items;
    bool Stop()
    {
        if (!Cancelled || !Cancelled())
            return false;
        Error = "Compute instance construction was cancelled.";
        return true;
    }
    bool Append(const SceneGeometry &object, const GeometryInstance &transform, uint32_t first)
    {
        if (Stop())
            return false;
        if (PrimitiveCount(object) == 0)
            return true;
        auto found = ObjectLookup.find(&object);
        if (found == ObjectLookup.end())
        {
            Box bounds;
            size_t visited = 0;
            for (const auto &triangle : object.Triangles)
            {
                if (visited++ % CancellationStride == 0 && Stop())
                    return false;
                for (Float3 p : {triangle.Position0, triangle.Position1, triangle.Position2})
                    bounds.Add(Point{p.X, p.Y, p.Z});
            }
            for (const auto &sphere : object.Spheres)
            {
                if (visited++ % CancellationStride == 0 && Stop())
                    return false;
                Point low{sphere.Center.X, sphere.Center.Y, sphere.Center.Z}, high = low;
                for (uint32_t axis = 0; axis < Dimensions; ++axis)
                {
                    low[axis] -= sphere.Radius;
                    high[axis] += sphere.Radius;
                }
                bounds.Add(low);
                bounds.Add(high);
            }
            found = ObjectLookup.emplace(&object, uint32_t(Result.Objects.size())).first;
            Result.Objects.push_back(&object);
            LocalBounds.push_back(bounds);
        }
        Item item{};
        item.Record.ObjectIndex = found->second;
        item.Record.Motion = transform.Motion;
        item.Record.Geometry.FirstPrimitive = first;
        if (!Invert(transform, item.Record))
        {
            Error = "Compute instance has an invalid affine inverse.";
            return false;
        }
        const auto &rest = LocalBounds[found->second];
        Box local;
        if (!SweptBounds(transform.Motion, rest.Min, rest.Max, local.Min, local.Max))
        {
            Error = "Compute motion has invalid swept bounds.";
            return false;
        }
        for (uint32_t row = 0; row < Dimensions; ++row)
        {
            double low = transform.ObjectToWorld[row * AffineStride + Dimensions], high = low;
            for (uint32_t col = 0; col < Dimensions; ++col)
            {
                const double coefficient = transform.ObjectToWorld[row * AffineStride + col];
                const double a = coefficient * local.Min[col], b = coefficient * local.Max[col];
                low += std::min(a, b);
                high += std::max(a, b);
            }
            constexpr double RoundingUlps = 8;
            const double margin = RoundingUlps * std::numeric_limits<float>::epsilon() *
                                  std::max({1.0, std::abs(low), std::abs(high)});
            if (!std::isfinite(low) || !std::isfinite(high) ||
                std::max(std::abs(low), std::abs(high)) + margin >= std::numeric_limits<float>::max())
            {
                Error = "Compute instance world bounds exceed finite float coordinates.";
                return false;
            }
            item.Record.Minimum[row] = std::nextafter(float(low - margin), -INFINITY);
            item.Record.Maximum[row] = std::nextafter(float(high + margin), INFINITY);
            item.Bounds.Min[row] = item.Record.Minimum[row];
            item.Bounds.Max[row] = item.Record.Maximum[row];
            item.Center[row] = (low + high) / 2;
        }
        Items.push_back(item);
        return true;
    }
    bool Split(uint32_t begin, uint32_t end, uint32_t depth)
    {
        if (Stop())
            return false;
        const uint32_t index = uint32_t(Result.Nodes.size());
        Result.Nodes.emplace_back();
        Box bounds, centers;
        for (uint32_t i = begin; i < end; ++i)
        {
            if (i % CancellationStride == 0 && Stop())
                return false;
            bounds.Add(Items[i].Bounds);
            centers.Add(Items[i].Center);
        }
        for (uint32_t axis = 0; axis < Dimensions; ++axis)
        {
            Result.Nodes[index].Minimum[axis] = float(bounds.Min[axis]);
            Result.Nodes[index].Maximum[axis] = float(bounds.Max[axis]);
        }
        if (end - begin <= LeafCapacity || depth == DepthLimit)
        {
            Result.Nodes[index].FirstOrEscape = begin;
            Result.Nodes[index].Count = end - begin;
            return true;
        }
        double cost = INFINITY;
        uint32_t splitAxis = 0, splitBin = 0;
        const auto binOf = [&](const Item &item, uint32_t axis) {
            const double fraction =
                (item.Center[axis] - centers.Min[axis]) / (centers.Max[axis] - centers.Min[axis]);
            return std::min(uint32_t(fraction * BinCount), BinCount - 1);
        };
        for (uint32_t axis = 0; axis < Dimensions; ++axis)
        {
            if (centers.Min[axis] == centers.Max[axis])
                continue;
            std::array<Bin, BinCount> bins;
            for (uint32_t i = begin; i < end; ++i)
            {
                if (i % CancellationStride == 0 && Stop())
                    return false;
                auto &bin = bins[binOf(Items[i], axis)];
                ++bin.Count;
                bin.Bounds.Add(Items[i].Bounds);
            }
            for (uint32_t boundary = 1; boundary < BinCount; ++boundary)
            {
                Box left, right;
                uint32_t leftCount = 0, rightCount = 0;
                for (uint32_t bin = 0; bin < BinCount; ++bin)
                {
                    if (!bins[bin].Count)
                        continue;
                    if (bin < boundary)
                    {
                        left.Add(bins[bin].Bounds);
                        leftCount += bins[bin].Count;
                    }
                    else
                    {
                        right.Add(bins[bin].Bounds);
                        rightCount += bins[bin].Count;
                    }
                }
                if (!leftCount || !rightCount)
                    continue;
                const double candidate = left.Area() * leftCount + right.Area() * rightCount;
                if (candidate < cost)
                {
                    cost = candidate;
                    splitAxis = axis;
                    splitBin = boundary;
                }
            }
        }
        uint32_t middle = begin;
        if (std::isfinite(cost))
            middle =
                uint32_t(std::partition(Items.begin() + begin, Items.begin() + end,
                                        [&](const Item &item) { return binOf(item, splitAxis) < splitBin; }) -
                         Items.begin());
        if (middle == begin || middle == end)
        {
            middle = begin + (end - begin) / 2;
            std::nth_element(Items.begin() + begin, Items.begin() + middle, Items.begin() + end,
                             [splitAxis](const Item &a, const Item &b) {
                                 return a.Center[splitAxis] < b.Center[splitAxis];
                             });
        }
        if (!Split(begin, middle, depth + 1) || !Split(middle, end, depth + 1))
            return false;
        Result.Nodes[index].FirstOrEscape = uint32_t(Result.Nodes.size());
        return true;
    }

  public:
    Builder(const std::function<bool()> &cancelled, Hierarchy &result, std::string &error)
        : Cancelled(cancelled), Result(result), Error(error)
    {
    }
    bool Run(const Scene &scene)
    {
        if (Stop() || !Validate(scene, Error) || !Append(scene, {}, 0))
            return false;
        uint64_t first = PrimitiveCount(scene);
        for (const auto &instance : scene.Instances)
        {
            const auto &object = *scene.Geometries[instance.GeometryIndex];
            if (!Append(object, instance, uint32_t(first)))
                return false;
            first += PrimitiveCount(object);
        }
        if (!Items.empty() && !Split(0, uint32_t(Items.size()), 0))
            return false;
        Result.Instances.reserve(Items.size());
        for (size_t i = 0; i < Items.size(); ++i)
        {
            if (i % CancellationStride == 0 && Stop())
                return false;
            Result.Instances.push_back(Items[i].Record);
        }
        return !Stop();
    }
};
} // namespace
bool Validate(const Scene &scene, std::string &error)
{
    if (scene.Instances.size() >= InstanceLimit || scene.Geometries.size() >= InstanceLimit ||
        PrimitiveCount(scene) > LocalPrimitiveLimit)
    {
        error = "Compute instance counts exceed the native index layout.";
        return false;
    }
    for (const auto &object : scene.Geometries)
        if (!object || PrimitiveCount(*object) > LocalPrimitiveLimit)
        {
            error = "Compute instance geometry is null or exceeds local indexing.";
            return false;
        }
    uint64_t count = PrimitiveCount(scene);
    for (const auto &instance : scene.Instances)
    {
        if (!ValidateRigidMotion(instance.Motion, error))
            return false;
        if (instance.GeometryIndex >= scene.Geometries.size())
        {
            error = "Compute instance geometry index is invalid.";
            return false;
        }
        count += PrimitiveCount(*scene.Geometries[instance.GeometryIndex]);
        if (count > InvalidRayPrimitiveIndex)
        {
            error = "Compute instance primitive IDs exceed 32 bits.";
            return false;
        }
        Instance inverse;
        if (!Invert(instance, inverse))
        {
            error = "Compute instance transform is singular, non-finite or has an unrepresentable inverse.";
            return false;
        }
    }
    return true;
}
bool Build(const Scene &scene, const std::function<bool()> &cancelled, Hierarchy &result, std::string &error)
{
    result = {};
    if (Builder(cancelled, result, error).Run(scene))
        return true;
    result = {};
    return false;
}
} // namespace Arvo::RayTracing::ComputeInstances
