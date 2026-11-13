#include "MotionPreparation.h"
#include "InstancingBuilder.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <unordered_map>

namespace Arvo::RayTracing::CudaInstances
{
namespace
{
// CUDA's host TLAS builder retains one immutable object per shared geometry.
// Gaussian elimination computes inverse affine transforms; midpoint ordering
// produces a bounded-depth median hierarchy with stackless device escape links.
// Geometry vertices stay in object space. Record offsets are filled by the RHI
// after independently building and concatenating the selected CUDA BLASes.
// PBRT (4th ed.), Sections 3.10 and 7.1.2, transformed primitives and ray instancing.
// https://pbr-book.org/4ed/Primitives_and_Intersection_Acceleration/Primitive_Interface_and_Geometric_Primitives
constexpr uint32_t AxisCount = 3, RowWidth = 4, AugmentedWidth = AxisCount * 2;
constexpr uint32_t LeafInstances = 4, CancellationInterval = 1024;
constexpr uint64_t MaximumLocalPrimitives = (uint64_t{1} << 31) - 1;
constexpr uint64_t MaximumInstances = std::numeric_limits<uint32_t>::max() / 2;
using Point = std::array<double, AxisCount>;
struct Bounds
{
    Point Lower{INFINITY, INFINITY, INFINITY}, Upper{-INFINITY, -INFINITY, -INFINITY};
    void Add(Point p)
    {
        for (uint32_t axis = 0; axis < AxisCount; ++axis)
        {
            Lower[axis] = std::min(Lower[axis], p[axis]);
            Upper[axis] = std::max(Upper[axis], p[axis]);
        }
    }
};
uint64_t Count(const SceneGeometry &geometry)
{
    return uint64_t(geometry.Triangles.size()) + geometry.Spheres.size();
}
bool Inverse(const GeometryInstance &source, Record &record)
{
    double augmented[AxisCount][AugmentedWidth]{};
    for (float entry : source.ObjectToWorld)
        if (!std::isfinite(entry))
            return false;
    for (uint32_t row = 0; row < AxisCount; ++row)
    {
        for (uint32_t column = 0; column < AxisCount; ++column)
            augmented[row][column] = source.ObjectToWorld[row * RowWidth + column];
        augmented[row][AxisCount + row] = 1;
    }
    for (uint32_t column = 0; column < AxisCount; ++column)
    {
        uint32_t pivot = column;
        for (uint32_t row = column + 1; row < AxisCount; ++row)
            if (std::abs(augmented[row][column]) > std::abs(augmented[pivot][column]))
                pivot = row;
        if (augmented[pivot][column] == 0)
            return false;
        for (uint32_t entry = 0; entry < AugmentedWidth; ++entry)
            std::swap(augmented[pivot][entry], augmented[column][entry]);
        const double divisor = augmented[column][column];
        for (uint32_t entry = 0; entry < AugmentedWidth; ++entry)
            augmented[column][entry] /= divisor;
        for (uint32_t row = 0; row < AxisCount; ++row)
        {
            if (row == column)
                continue;
            const double multiplier = augmented[row][column];
            for (uint32_t entry = 0; entry < AugmentedWidth; ++entry)
                augmented[row][entry] -= multiplier * augmented[column][entry];
        }
    }
    for (uint32_t row = 0; row < AxisCount; ++row)
    {
        double translation = 0;
        for (uint32_t column = 0; column < AxisCount; ++column)
        {
            const double inverse = augmented[row][AxisCount + column];
            record.WorldToObject[row * RowWidth + column] = inverse;
            translation -= inverse * source.ObjectToWorld[column * RowWidth + AxisCount];
        }
        record.WorldToObject[row * RowWidth + AxisCount] = translation;
    }
    for (double entry : record.WorldToObject)
        if (!std::isfinite(entry) || std::abs(entry) > std::numeric_limits<float>::max())
            return false;
    return true;
}
struct Pending
{
    Record Descriptor;
    Node WorldBounds;
    Point Center;
};
class Builder
{
    const std::function<bool()> &Cancelled;
    Hierarchy &Output;
    std::string &Error;
    std::unordered_map<const SceneGeometry *, uint32_t> ObjectIndices;
    std::vector<Bounds> ObjectBounds;
    std::vector<Pending> Instances;
    bool Stop()
    {
        if (!Cancelled || !Cancelled())
            return false;
        Error = "CUDA instance construction was cancelled.";
        return true;
    }
    bool Add(const SceneGeometry &geometry, const GeometryInstance &instance, uint32_t first)
    {
        if (Stop())
            return false;
        if (Count(geometry) == 0)
            return true;
        auto entry = ObjectIndices.find(&geometry);
        if (entry == ObjectIndices.end())
        {
            Bounds bounds;
            size_t visited = 0;
            for (const auto &triangle : geometry.Triangles)
            {
                if (visited++ % CancellationInterval == 0 && Stop())
                    return false;
                for (Float3 p : {triangle.Position0, triangle.Position1, triangle.Position2})
                    bounds.Add({p.X, p.Y, p.Z});
            }
            for (const auto &sphere : geometry.Spheres)
            {
                if (visited++ % CancellationInterval == 0 && Stop())
                    return false;
                const Point center{sphere.Center.X, sphere.Center.Y, sphere.Center.Z};
                Point lower{}, upper{};
                for (uint32_t axis = 0; axis < AxisCount; ++axis)
                {
                    lower[axis] = center[axis] - sphere.Radius;
                    upper[axis] = center[axis] + sphere.Radius;
                }
                bounds.Add(lower);
                bounds.Add(upper);
            }
            const uint32_t object = uint32_t(Output.Objects.size());
            entry = ObjectIndices.emplace(&geometry, object).first;
            Output.Objects.push_back(&geometry);
            ObjectBounds.push_back(bounds);
        }
        Pending pending{};
        pending.Descriptor.ObjectIndex = entry->second;
        pending.Descriptor.Motion = instance.Motion;
        pending.Descriptor.FirstPrimitive = first;
        if (!Inverse(instance, pending.Descriptor))
        {
            Error = "CUDA instance transform is singular, non-finite or has an unrepresentable inverse.";
            return false;
        }
        const Bounds &rest = ObjectBounds[entry->second];
        Bounds local;
        if (!MotionBounds(instance.Motion, rest.Lower, rest.Upper, local.Lower, local.Upper))
        {
            Error = "CUDA motion has invalid swept bounds.";
            return false;
        }
        Bounds world;
        constexpr uint32_t CornerCount = 1u << AxisCount;
        for (uint32_t corner = 0; corner < CornerCount; ++corner)
        {
            Point point{}, transformed{};
            for (uint32_t axis = 0; axis < AxisCount; ++axis)
                point[axis] = (corner & (1u << axis)) ? local.Upper[axis] : local.Lower[axis];
            for (uint32_t row = 0; row < AxisCount; ++row)
            {
                transformed[row] = instance.ObjectToWorld[row * RowWidth + AxisCount];
                for (uint32_t column = 0; column < AxisCount; ++column)
                    transformed[row] +=
                        double(instance.ObjectToWorld[row * RowWidth + column]) * point[column];
            }
            world.Add(transformed);
        }
        for (uint32_t axis = 0; axis < AxisCount; ++axis)
        {
            const double lower = world.Lower[axis], upper = world.Upper[axis];
            constexpr double BoundRoundingUlps = 8;
            const double margin = BoundRoundingUlps * std::numeric_limits<float>::epsilon() *
                                  std::max({1.0, std::abs(lower), std::abs(upper)});
            if (!std::isfinite(lower) || !std::isfinite(upper) ||
                std::max(std::abs(lower), std::abs(upper)) + margin >= std::numeric_limits<float>::max())
            {
                Error = "CUDA instance bounds exceed finite float coordinates.";
                return false;
            }
            pending.WorldBounds.Minimum[axis] = std::nextafter(float(lower - margin), -INFINITY);
            pending.WorldBounds.Maximum[axis] = std::nextafter(float(upper + margin), INFINITY);
            pending.Descriptor.Minimum[axis] = pending.WorldBounds.Minimum[axis];
            pending.Descriptor.Maximum[axis] = pending.WorldBounds.Maximum[axis];
            pending.Center[axis] = (lower + upper) / 2;
        }
        Instances.push_back(pending);
        return true;
    }
    bool Partition(uint32_t begin, uint32_t end)
    {
        if (Stop())
            return false;
        const uint32_t at = uint32_t(Output.Nodes.size());
        Output.Nodes.emplace_back();
        Bounds nodeBounds, centers;
        for (uint32_t offset = begin; offset < end; ++offset)
        {
            if (offset % CancellationInterval == 0 && Stop())
                return false;
            const auto &item = Instances[offset];
            const auto &box = item.WorldBounds;
            nodeBounds.Add({box.Minimum[0], box.Minimum[1], box.Minimum[2]});
            nodeBounds.Add({box.Maximum[0], box.Maximum[1], box.Maximum[2]});
            centers.Add(item.Center);
        }
        for (uint32_t axis = 0; axis < AxisCount; ++axis)
        {
            Output.Nodes[at].Minimum[axis] = float(nodeBounds.Lower[axis]);
            Output.Nodes[at].Maximum[axis] = float(nodeBounds.Upper[axis]);
        }
        if (end - begin <= LeafInstances)
        {
            Output.Nodes[at].FirstOrEscape = begin;
            Output.Nodes[at].Count = end - begin;
            return true;
        }
        uint32_t axis = 0;
        for (uint32_t candidate = 1; candidate < AxisCount; ++candidate)
            if (centers.Upper[candidate] - centers.Lower[candidate] >
                centers.Upper[axis] - centers.Lower[axis])
                axis = candidate;
        const uint32_t middle = begin + (end - begin) / 2;
        std::nth_element(
            Instances.begin() + begin, Instances.begin() + middle, Instances.begin() + end,
            [axis](const Pending &a, const Pending &b) { return a.Center[axis] < b.Center[axis]; });
        if (!Partition(begin, middle) || !Partition(middle, end))
            return false;
        Output.Nodes[at].FirstOrEscape = uint32_t(Output.Nodes.size());
        return true;
    }

  public:
    Builder(const std::function<bool()> &cancelled, Hierarchy &output, std::string &error)
        : Cancelled(cancelled), Output(output), Error(error)
    {
    }
    bool Run(const Scene &scene)
    {
        if (Stop() || !Validate(scene, Error))
            return false;
        if (!Add(scene, GeometryInstance{}, 0))
            return false;
        uint64_t first = Count(scene);
        for (const auto &instance : scene.Instances)
        {
            const auto &geometry = *scene.Geometries[instance.GeometryIndex];
            if (!Add(geometry, instance, uint32_t(first)))
                return false;
            first += Count(geometry);
        }
        if (!Instances.empty() && !Partition(0, uint32_t(Instances.size())))
            return false;
        Output.Records.reserve(Instances.size());
        for (size_t index = 0; index < Instances.size(); ++index)
        {
            if (index % CancellationInterval == 0 && Stop())
                return false;
            Output.Records.push_back(Instances[index].Descriptor);
        }
        return !Stop();
    }
};
} // namespace
bool Validate(const Scene &scene, std::string &error)
{
    if (scene.Instances.size() >= MaximumInstances || scene.Geometries.size() >= MaximumInstances ||
        Count(scene) > MaximumLocalPrimitives)
    {
        error = "CUDA instance or geometry counts exceed the native layout.";
        return false;
    }
    for (const auto &object : scene.Geometries)
        if (!object || Count(*object) > MaximumLocalPrimitives)
        {
            error = "CUDA instance geometry is null or exceeds 31-bit primitive indexing.";
            return false;
        }
    uint64_t count = Count(scene);
    for (const auto &instance : scene.Instances)
    {
        if (!ValidateRigidMotion(instance.Motion, error))
            return false;
        if (instance.GeometryIndex >= scene.Geometries.size())
        {
            error = "CUDA instance references an invalid geometry index.";
            return false;
        }
        count += Count(*scene.Geometries[instance.GeometryIndex]);
        if (count > InvalidRayPrimitiveIndex)
        {
            error = "CUDA instance hit IDs exceed 32-bit indexing.";
            return false;
        }
        Record inverse;
        if (!Inverse(instance, inverse))
        {
            error = "CUDA instance transform is singular, non-finite or has an unrepresentable inverse.";
            return false;
        }
    }
    return true;
}
bool Build(const Scene &scene, const std::function<bool()> &cancelled, Hierarchy &hierarchy,
           std::string &error)
{
    hierarchy = {};
    if (Builder(cancelled, hierarchy, error).Run(scene))
        return true;
    hierarchy = {};
    return false;
}
} // namespace Arvo::RayTracing::CudaInstances
