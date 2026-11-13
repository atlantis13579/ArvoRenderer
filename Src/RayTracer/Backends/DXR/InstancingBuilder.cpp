#include "MotionPreparation.h"
#include "InstancingBuilder.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>

namespace Arvo::RayTracing::DxrInstances
{
namespace
{
// Independent DXR instance preparation. Morton ordering groups nearby instances;
// a radix split produces preorder nodes for software traversal and hardware-ray
// range validation. Actual hardware tracing uses transformed DXR TLAS instances.
// Each unique object owns one triangle BLAS and/or one procedural-sphere BLAS.
// PBRT 4e, "Linear Bounding Volume Hierarchies"; DXR Functional Spec, "Instancing".
// https://pbr-book.org/4ed/Primitives_and_Intersection_Acceleration/Bounding_Volume_Hierarchies
// https://microsoft.github.io/DirectX-Specs/d3d/Raytracing.html
constexpr uint32_t AxisCount = 3, MatrixStride = 4, MortonBitsPerAxis = 10;
constexpr uint32_t MortonCells = 1u << MortonBitsPerAxis, MortonTopBit = AxisCount * MortonBitsPerAxis - 1;
constexpr uint32_t LeafSize = 4, CancellationStride = 1024;
constexpr uint64_t MaximumHardwareInstances = uint64_t{1} << 24;
using Vector = std::array<double, AxisCount>;
struct Bounds
{
    Vector Low{INFINITY, INFINITY, INFINITY}, High{-INFINITY, -INFINITY, -INFINITY};
    void Include(const Vector &point)
    {
        for (uint32_t a = 0; a < AxisCount; ++a)
        {
            Low[a] = std::min(Low[a], point[a]);
            High[a] = std::max(High[a], point[a]);
        }
    }
};
uint64_t Count(const SceneGeometry &object)
{
    return uint64_t(object.Triangles.size()) + object.Spheres.size();
}
Vector Cross(const Vector &a, const Vector &b)
{
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
double Dot(const Vector &a, const Vector &b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
bool Inverse(const std::array<float, 12> &transform, Record &output)
{
    for (float value : transform)
        if (!std::isfinite(value))
            return false;
    const Vector columns[AxisCount] = {{transform[0], transform[4], transform[8]},
                                       {transform[1], transform[5], transform[9]},
                                       {transform[2], transform[6], transform[10]}};
    const Vector reciprocal[AxisCount] = {Cross(columns[1], columns[2]), Cross(columns[2], columns[0]),
                                          Cross(columns[0], columns[1])};
    const double determinant = Dot(columns[0], reciprocal[0]);
    if (determinant == 0 || !std::isfinite(determinant))
        return false;
    const Vector translation{transform[3], transform[7], transform[11]};
    for (uint32_t row = 0; row < AxisCount; ++row)
    {
        for (uint32_t column = 0; column < AxisCount; ++column)
            output.Inverse[row * MatrixStride + column] = reciprocal[row][column] / determinant;
        output.Inverse[row * MatrixStride + AxisCount] = -Dot(reciprocal[row], translation) / determinant;
    }
    for (double entry : output.Inverse)
        if (!std::isfinite(entry) || std::abs(entry) > std::numeric_limits<float>::max())
            return false;
    return true;
}
struct Pending
{
    Record Instance;
    std::array<float, 12> Transform;
    Vector Center;
    uint32_t MortonCode = 0;
};
class Construction
{
    const std::function<bool()> &Cancelled;
    Hierarchy &Output;
    std::string &Error;
    std::unordered_map<const SceneGeometry *, uint32_t> Objects;
    std::vector<Bounds> LocalBounds;
    std::vector<Pending> PendingInstances;
    bool Stop()
    {
        if (!Cancelled || !Cancelled())
            return false;
        Error = "DXR instance construction was cancelled.";
        return true;
    }
    bool Append(const SceneGeometry &source, const GeometryInstance &placement, uint32_t first)
    {
        const auto& transform = placement.ObjectToWorld;
        if (Stop())
            return false;
        if (Count(source) == 0)
            return true;
        auto found = Objects.find(&source);
        if (found == Objects.end())
        {
            Bounds bounds;
            size_t visited = 0;
            for (const auto &triangle : source.Triangles)
            {
                if (visited++ % CancellationStride == 0 && Stop())
                    return false;
                for (Float3 p : {triangle.Position0, triangle.Position1, triangle.Position2})
                    bounds.Include({p.X, p.Y, p.Z});
            }
            for (const auto &sphere : source.Spheres)
            {
                if (visited++ % CancellationStride == 0 && Stop())
                    return false;
                bounds.Include({double(sphere.Center.X) - sphere.Radius,
                                double(sphere.Center.Y) - sphere.Radius,
                                double(sphere.Center.Z) - sphere.Radius});
                bounds.Include({double(sphere.Center.X) + sphere.Radius,
                                double(sphere.Center.Y) + sphere.Radius,
                                double(sphere.Center.Z) + sphere.Radius});
            }
            const uint32_t index = uint32_t(Output.Objects.size());
            found = Objects.emplace(&source, index).first;
            Output.Objects.push_back(&source);
            LocalBounds.push_back(bounds);
            GeometryRange geometry;
            geometry.ObjectIndex = index;
            geometry.TriangleCount = uint32_t(source.Triangles.size());
            geometry.SphereCount = uint32_t(source.Spheres.size());
            for (uint32_t axis = 0; axis < AxisCount; ++axis)
            {
                geometry.Minimum[axis] = float(bounds.Low[axis]);
                geometry.Maximum[axis] = float(bounds.High[axis]);
            }
            Output.Ranges.push_back(geometry);
        }
        Pending pending;
        pending.Transform = transform;
        pending.Instance.Motion = placement.Motion;
        pending.Instance.Geometry = Output.Ranges[found->second];
        pending.Instance.Geometry.FirstPrimitive = first;
        if (!Inverse(transform, pending.Instance))
        {
            Error = "DXR instance inverse is invalid.";
            return false;
        }
        const auto &rest = LocalBounds[found->second];
        Bounds local;
        if (!SweptLocalBounds(placement.Motion, rest.Low, rest.High, local.Low, local.High))
        {
            Error = "DXR motion has invalid swept bounds.";
            return false;
        }
        Vector center{}, extent{};
        for (uint32_t axis = 0; axis < AxisCount; ++axis)
        {
            center[axis] = (local.High[axis] + local.Low[axis]) / 2;
            extent[axis] = (local.High[axis] - local.Low[axis]) / 2;
        }
        for (uint32_t row = 0; row < AxisCount; ++row)
        {
            double midpoint = transform[row * MatrixStride + AxisCount], radius = 0;
            for (uint32_t column = 0; column < AxisCount; ++column)
            {
                const double coefficient = transform[row * MatrixStride + column];
                midpoint += coefficient * center[column];
                radius += std::abs(coefficient) * extent[column];
            }
            const double low = midpoint - radius, high = midpoint + radius;
            constexpr double BoundsUlps = 8;
            const double expansion = BoundsUlps * std::numeric_limits<float>::epsilon() *
                                     std::max({1.0, std::abs(low), std::abs(high)});
            if (!std::isfinite(low) || !std::isfinite(high) ||
                std::max(std::abs(low), std::abs(high)) + expansion >= std::numeric_limits<float>::max())
            {
                Error = "DXR instance bounds exceed finite float coordinates.";
                return false;
            }
            pending.Instance.Minimum[row] = std::nextafter(float(low - expansion), -INFINITY);
            pending.Instance.Maximum[row] = std::nextafter(float(high + expansion), INFINITY);
            pending.Center[row] = midpoint;
        }
        PendingInstances.push_back(pending);
        return true;
    }
    bool Emit(uint32_t begin, uint32_t end)
    {
        if (Stop())
            return false;
        const uint32_t index = uint32_t(Output.Nodes.size());
        Output.Nodes.emplace_back();
        Bounds bounds;
        for (uint32_t i = begin; i < end; ++i)
        {
            if (i % CancellationStride == 0 && Stop())
                return false;
            const auto &item = PendingInstances[i].Instance;
            bounds.Include({item.Minimum[0], item.Minimum[1], item.Minimum[2]});
            bounds.Include({item.Maximum[0], item.Maximum[1], item.Maximum[2]});
        }
        for (uint32_t a = 0; a < AxisCount; ++a)
        {
            Output.Nodes[index].Minimum[a] = float(bounds.Low[a]);
            Output.Nodes[index].Maximum[a] = float(bounds.High[a]);
        }
        if (end - begin <= LeafSize)
        {
            Output.Nodes[index].FirstOrEscape = begin;
            Output.Nodes[index].Count = end - begin;
            return true;
        }
        uint32_t middle = begin + (end - begin) / 2;
        const uint32_t difference = PendingInstances[begin].MortonCode ^ PendingInstances[end - 1].MortonCode;
        if (difference)
        {
            uint32_t bit = 1u << MortonTopBit;
            while (!(difference & bit))
                bit >>= 1;
            uint32_t lower = begin, upper = end;
            while (lower + 1 < upper)
            {
                const uint32_t probe = lower + (upper - lower) / 2;
                if (PendingInstances[probe].MortonCode & bit)
                    upper = probe;
                else
                    lower = probe;
            }
            middle = upper;
        }
        if (!Emit(begin, middle) || !Emit(middle, end))
            return false;
        Output.Nodes[index].FirstOrEscape = uint32_t(Output.Nodes.size());
        return true;
    }

  public:
    Construction(const std::function<bool()> &cancelled, Hierarchy &output, std::string &error)
        : Cancelled(cancelled), Output(output), Error(error)
    {
    }
    bool Run(const Scene &scene)
    {
        if (Stop() || !Validate(scene, Error) || !Append(scene, GeometryInstance{}, 0))
            return false;
        uint64_t first = Count(scene);
        for (const auto &instance : scene.Instances)
        {
            const auto &geometry = *scene.Geometries[instance.GeometryIndex];
            if (!Append(geometry, instance, uint32_t(first)))
                return false;
            first += Count(geometry);
        }
        if (PendingInstances.empty())
            return true;
        Bounds centers;
        for (const auto &instance : PendingInstances)
            centers.Include(instance.Center);
        for (size_t i = 0; i < PendingInstances.size(); ++i)
        {
            if (i % CancellationStride == 0 && Stop())
                return false;
            auto &pending = PendingInstances[i];
            for (uint32_t a = 0; a < AxisCount; ++a)
            {
                const double span = centers.High[a] - centers.Low[a];
                const double unit = span > 0 ? (pending.Center[a] - centers.Low[a]) / span : 0;
                const uint32_t coordinate = std::min(uint32_t(unit * MortonCells), MortonCells - 1);
                for (uint32_t bit = 0; bit < MortonBitsPerAxis; ++bit)
                    pending.MortonCode |= ((coordinate >> bit) & 1u) << (bit * AxisCount + a);
            }
        }
        std::sort(PendingInstances.begin(), PendingInstances.end(), [](const Pending &a, const Pending &b) {
            return a.MortonCode == b.MortonCode
                       ? a.Instance.Geometry.FirstPrimitive < b.Instance.Geometry.FirstPrimitive
                       : a.MortonCode < b.MortonCode;
        });
        if (!Emit(0, uint32_t(PendingInstances.size())))
            return false;
        for (size_t i = 0; i < PendingInstances.size(); ++i)
        {
            if (i % CancellationStride == 0 && Stop())
                return false;
            Output.Records.push_back(PendingInstances[i].Instance);
            Output.Transforms.push_back(PendingInstances[i].Transform);
        }
        return !Stop();
    }
};
} // namespace
bool Validate(const Scene &scene, std::string &error)
{
    std::vector<const SceneGeometry *> objects{&scene};
    for (const auto &geometry : scene.Geometries)
        objects.push_back(geometry.get());
    for (const auto *object : objects)
        if (!object || object->Triangles.size() > std::numeric_limits<uint32_t>::max() / AxisCount ||
            Count(*object) >= InvalidRayPrimitiveIndex)
        {
            error = "DXR instance geometry is null or exceeds native primitive indexing.";
            return false;
        }
    if (scene.Instances.size() >= MaximumHardwareInstances)
    {
        error = "DXR instance records exceed 24-bit hardware instance IDs.";
        return false;
    }
    uint64_t first = Count(scene);
    uint64_t hardwareCount = uint64_t(!scene.Triangles.empty()) + uint64_t(!scene.Spheres.empty());
    for (const auto &instance : scene.Instances)
    {
        if (!ValidateRigidMotion(instance.Motion, error))
            return false;
        if (instance.GeometryIndex >= scene.Geometries.size())
        {
            error = "DXR instance references an invalid geometry.";
            return false;
        }
        const auto &geometry = *scene.Geometries[instance.GeometryIndex];
        first += Count(geometry);
        hardwareCount += uint64_t(!geometry.Triangles.empty()) + uint64_t(!geometry.Spheres.empty());
        Record inverse;
        if (first > InvalidRayPrimitiveIndex || hardwareCount > MaximumHardwareInstances ||
            !Inverse(instance.ObjectToWorld, inverse))
        {
            error = "DXR instance IDs, transform or inverse exceed the native layout.";
            return false;
        }
    }
    return true;
}
bool Build(const Scene &scene, const std::function<bool()> &cancelled, Hierarchy &output, std::string &error)
{
    output = {};
    if (Construction(cancelled, output, error).Run(scene))
        return true;
    output = {};
    return false;
}
} // namespace Arvo::RayTracing::DxrInstances
