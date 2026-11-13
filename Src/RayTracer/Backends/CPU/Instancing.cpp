#include "MotionPreparation.h"
#include "Motion.h"
#include "MotionInstancing.h"
#include "CpuTransport.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

namespace Arvo::RayTracing::Cpu
{
namespace
{
// A median TLAS stores world bounds and transforms, while each immutable object
// owns one BLAS of the selected kind. Ordinary affine rays keep their direction
// length and t; extreme directions carry a power-of-two parameter scale as well.
// Normals use the inverse transpose;
// nonuniform scales therefore produce ellipsoids from analytic object spheres.
// Pharr, Jakob and Humphreys, Physically Based Rendering (4th ed.), Section 7.1.2.
// https://pbr-book.org/4ed/Primitives_and_Intersection_Acceleration/Primitive_Interface_and_Geometric_Primitives
// IDs enumerate legacy geometry first, then each instance's local primitives.
// Geometry reuse never aliases hit identity, and no world-space vertices are copied.
// The existing float BLAS kernels require representable object-space origins;
// unrepresentable transformed queries fail explicitly instead of reporting misses.
constexpr uint32_t Dimensions = 3, AffineColumns = 4;
constexpr uint32_t InstanceLeafSize = 4, CancellationPeriod = 1024;
using Matrix = std::array<double, Dimensions * AffineColumns>;
using Vector = std::array<double, Dimensions>;
Vector Coordinates(Float3 p)
{
    return {p.X, p.Y, p.Z};
}
struct Box
{
    Vector Low{INFINITY, INFINITY, INFINITY}, High{-INFINITY, -INFINITY, -INFINITY};
    void Include(const Vector &point)
    {
        for (uint32_t axis = 0; axis < Dimensions; ++axis)
        {
            Low[axis] = std::min(Low[axis], point[axis]);
            High[axis] = std::max(High[axis], point[axis]);
        }
    }
    void Include(const Box &box)
    {
        Include(box.Low);
        Include(box.High);
    }
};
bool Invert(const GeometryInstance &instance, Matrix &inverse)
{
    const auto &m = instance.ObjectToWorld;
    for (float value : m)
        if (!std::isfinite(value))
            return false;
    // Cross products form the adjugate rows. Double intermediates avoid float
    // overflow while checking invertibility; a singular transform is an error.
    const double a = m[0], b = m[1], c = m[2], d = m[4], e = m[5], f = m[6];
    const double g = m[8], h = m[9], i = m[10];
    const double determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    if (determinant == 0 || !std::isfinite(determinant))
        return false;
    inverse = {e * i - f * h, c * h - b * i, b * f - c * e, 0, f * g - d * i, a * i - c * g, c * d - a * f, 0,
               d * h - e * g, b * g - a * h, a * e - b * d, 0};
    for (uint32_t row = 0; row < Dimensions; ++row)
    {
        const uint32_t at = row * AffineColumns;
        for (uint32_t column = 0; column < Dimensions; ++column)
            inverse[at + column] /= determinant;
        inverse[at + Dimensions] = -(inverse[at] * m[3] + inverse[at + 1] * m[7] + inverse[at + 2] * m[11]);
    }
    return std::all_of(inverse.begin(), inverse.end(), [](double value) {
        return std::isfinite(value) && std::abs(value) <= std::numeric_limits<float>::max();
    });
}
Vector Transform(const Matrix &matrix, Float3 value, bool point)
{
    const Vector v = Coordinates(value);
    Vector result{};
    for (uint32_t row = 0; row < Dimensions; ++row)
    {
        const uint32_t at = row * AffineColumns;
        double sum = point ? matrix[at + Dimensions] : 0;
        for (uint32_t column = 0; column < Dimensions; ++column)
            sum += matrix[at + column] * v[column];
        result[row] = sum;
    }
    return result;
}
Float3 TransformNormal(const Matrix &inverse, Float3 normal)
{
    const Vector n = Coordinates(normal);
    Vector result{};
    for (uint32_t row = 0; row < Dimensions; ++row)
        for (uint32_t column = 0; column < Dimensions; ++column)
            result[row] += inverse[column * AffineColumns + row] * n[column];
    const double length = std::hypot(result[0], result[1], result[2]);
    return {float(result[0] / length), float(result[1] / length), float(result[2] / length)};
}
bool Overlaps(const Box &box, const Ray &ray, float closest)
{
    const Vector origin = Coordinates(ray.Origin), direction = Coordinates(ray.Direction);
    double low = ray.MinDistance, high = std::min(ray.MaxDistance, closest);
    for (uint32_t axis = 0; axis < Dimensions; ++axis)
    {
        if (direction[axis] == 0)
        {
            if (origin[axis] < box.Low[axis] || origin[axis] > box.High[axis])
                return false;
            continue;
        }
        double near = (box.Low[axis] - origin[axis]) / direction[axis];
        double far = (box.High[axis] - origin[axis]) / direction[axis];
        if (near > far)
            std::swap(near, far);
        low = std::max(low, near);
        high = std::min(high, far);
        if (low > high)
            return false;
    }
    return true;
}
uint64_t PrimitiveCount(const SceneGeometry &geometry)
{
    return uint64_t(geometry.Triangles.size()) + geometry.Spheres.size();
}
float ScaledDistance(float distance, double scale)
{
    const double scaled = double(distance) * scale;
    return scaled > std::numeric_limits<float>::max() ? std::numeric_limits<float>::infinity()
                                                      : float(scaled);
}
} // namespace

bool ValidateInstances(const Scene &scene, std::string &error)
{
    uint64_t count = PrimitiveCount(scene);
    constexpr uint64_t MaximumInstances = std::numeric_limits<uint32_t>::max() / 2 - 1;
    if (scene.Instances.size() > MaximumInstances || scene.Geometries.size() > MaximumInstances ||
        count > InvalidRayPrimitiveIndex)
    {
        error = "Scene geometry exceeds 32-bit instance or primitive indexing.";
        return false;
    }
    for (const auto &geometry : scene.Geometries)
        if (!geometry || PrimitiveCount(*geometry) > InvalidRayPrimitiveIndex)
        {
            error = "An instance geometry is null or exceeds 32-bit primitive indexing.";
            return false;
        }
    for (const auto &instance : scene.Instances)
    {
        if (!ValidateRigidMotion(instance.Motion, error))
            return false;
        if (instance.GeometryIndex >= scene.Geometries.size())
        {
            error = "An instance references an invalid geometry index.";
            return false;
        }
        const uint64_t localCount = PrimitiveCount(*scene.Geometries[instance.GeometryIndex]);
        if (localCount > InvalidRayPrimitiveIndex - count)
        {
            error = "Instance primitive IDs exceed 32-bit indexing.";
            return false;
        }
        count += localCount;
        Matrix inverse;
        if (!Invert(instance, inverse))
        {
            error = "Instance transforms must be finite, invertible and have a representable inverse.";
            return false;
        }
    }
    return true;
}

struct InstancedScene::Impl
{
    struct Object
    {
        std::unique_ptr<SceneAccelerator> Acceleration;
        Box Bounds;
        size_t PrimitiveCount = 0;
    };
    struct Instance
    {
        Matrix WorldToObject;
        MotionData::RigidMotion Motion;
        Box Bounds;
        Vector Center;
        uint32_t ObjectIndex, FirstPrimitive;
    };
    struct Node
    {
        Box Bounds;
        uint32_t First = 0, Count = 0, Escape = 0;
    };
    CancellationProbe Cancellation;
    std::vector<Object> Objects;
    std::vector<Instance> Instances;
    std::vector<uint32_t> Order;
    std::vector<Node> Nodes;
    bool Cancelled = false;

    bool Stop()
    {
        Cancelled |= Cancellation && Cancellation();
        return Cancelled;
    }
    Impl(const Scene &scene, AccelerationOptions options, CancellationProbe cancellation)
        : Cancellation(std::move(cancellation))
    {
        std::string error;
        if (!ValidateAcceleration(options, error) || !ValidateInstances(scene, error))
            throw std::invalid_argument(error);
        if (Stop())
            return;
        std::unordered_map<const SceneGeometry *, uint32_t> objectIndices;
        const auto add = [&](const SceneGeometry &geometry, const GeometryInstance &source, uint32_t first) {
            if (Stop() || PrimitiveCount(geometry) == 0)
                return;
            auto found = objectIndices.find(&geometry);
            if (found == objectIndices.end())
            {
                Object object;
                object.PrimitiveCount = size_t(PrimitiveCount(geometry));
                object.Acceleration = BuildGeometryAccelerator(geometry, options, Cancellation);
                if (object.Acceleration->WasCancelled())
                {
                    Cancelled = true;
                    return;
                }
                size_t processed = 0;
                for (const auto &triangle : geometry.Triangles)
                {
                    if (processed++ % CancellationPeriod == 0 && Stop())
                        return;
                    for (Float3 p : {triangle.Position0, triangle.Position1, triangle.Position2})
                        object.Bounds.Include(Coordinates(p));
                }
                for (const auto &sphere : geometry.Spheres)
                {
                    if (processed++ % CancellationPeriod == 0 && Stop())
                        return;
                    const Vector center = Coordinates(sphere.Center);
                    Vector low{}, high{};
                    for (uint32_t axis = 0; axis < Dimensions; ++axis)
                    {
                        low[axis] = center[axis] - sphere.Radius;
                        high[axis] = center[axis] + sphere.Radius;
                    }
                    object.Bounds.Include(low);
                    object.Bounds.Include(high);
                }
                const uint32_t index = uint32_t(Objects.size());
                Objects.push_back(std::move(object));
                found = objectIndices.emplace(&geometry, index).first;
            }
            Instance instance{};
            instance.ObjectIndex = found->second;
            instance.FirstPrimitive = first;
            Invert(source, instance.WorldToObject);
            instance.Motion = source.Motion;
            const auto &rest = Objects[instance.ObjectIndex].Bounds;
            Box local;
            if (!SweptInstanceBounds(source.Motion, rest.Low, rest.High, local.Low, local.High))
                throw std::invalid_argument("Instance motion has invalid swept bounds.");
            for (uint32_t row = 0; row < Dimensions; ++row)
            {
                double low = source.ObjectToWorld[row * AffineColumns + Dimensions], high = low;
                for (uint32_t column = 0; column < Dimensions; ++column)
                {
                    const double coefficient = source.ObjectToWorld[row * AffineColumns + column];
                    const double a = coefficient * local.Low[column], b = coefficient * local.High[column];
                    low += std::min(a, b);
                    high += std::max(a, b);
                }
                if (!std::isfinite(low) || !std::isfinite(high) ||
                    std::max(std::abs(low), std::abs(high)) > std::numeric_limits<float>::max())
                    throw std::invalid_argument("Instance world bounds exceed finite float coordinates.");
                // Include rounding of object-ray and primitive calculations at the
                // broad phase; exact primitive intervals remain open and unchanged.
                constexpr double BoundRoundingUlps = 8;
                const double padding = BoundRoundingUlps * std::numeric_limits<float>::epsilon() *
                                       std::max({1.0, std::abs(low), std::abs(high)});
                instance.Bounds.Low[row] = low - padding;
                instance.Bounds.High[row] = high + padding;
                instance.Center[row] = (low + high) / 2;
            }
            Instances.push_back(instance);
        };
        add(scene, GeometryInstance{}, 0);
        uint64_t first = PrimitiveCount(scene);
        for (const auto &instance : scene.Instances)
        {
            const SceneGeometry &geometry = *scene.Geometries[instance.GeometryIndex];
            add(geometry, instance, uint32_t(first));
            first += PrimitiveCount(geometry);
            if (Stop())
                break;
        }
        if (!Stop() && !Instances.empty())
        {
            Order.resize(Instances.size());
            std::iota(Order.begin(), Order.end(), 0u);
            Nodes.reserve(Instances.size() * 2);
            Build(0, uint32_t(Order.size()));
        }
        if (Stop())
        {
            Objects.clear();
            Instances.clear();
            Order.clear();
            Nodes.clear();
        }
    }
    uint32_t Build(uint32_t begin, uint32_t end)
    {
        const uint32_t index = uint32_t(Nodes.size());
        Nodes.emplace_back();
        Box centers;
        for (uint32_t offset = begin; offset < end; ++offset)
        {
            if (offset % CancellationPeriod == 0 && Stop())
                return index;
            const Instance &instance = Instances[Order[offset]];
            Nodes[index].Bounds.Include(instance.Bounds);
            centers.Include(instance.Center);
        }
        if (end - begin <= InstanceLeafSize)
        {
            Nodes[index].First = begin;
            Nodes[index].Count = end - begin;
            return index;
        }
        uint32_t axis = 0;
        for (uint32_t candidate = 1; candidate < Dimensions; ++candidate)
            if (centers.High[candidate] - centers.Low[candidate] > centers.High[axis] - centers.Low[axis])
                axis = candidate;
        const uint32_t middle = begin + (end - begin) / 2;
        std::nth_element(
            Order.begin() + begin, Order.begin() + middle, Order.begin() + end,
            [&](uint32_t a, uint32_t b) { return Instances[a].Center[axis] < Instances[b].Center[axis]; });
        Build(begin, middle);
        if (!Stop())
            Build(middle, end);
        Nodes[index].Escape = uint32_t(Nodes.size());
        return index;
    }
    bool Intersect(const Ray &ray, SurfaceHit &hit, uint64_t &tests, bool anyHit, float sceneTime = 0) const
    {
        bool found = false;
        uint32_t nodeIndex = 0;
        while (nodeIndex < Nodes.size())
        {
            const Node &node = Nodes[nodeIndex];
            if (!Overlaps(node.Bounds, ray, hit.Distance))
            {
                nodeIndex = node.Count ? nodeIndex + 1 : node.Escape;
                continue;
            }
            if (node.Count == 0)
            {
                ++nodeIndex;
                continue;
            }
            for (uint32_t offset = 0; offset < node.Count; ++offset)
            {
                const Instance &instance = Instances[Order[node.First + offset]];
                if (!Overlaps(instance.Bounds, ray, hit.Distance))
                    continue;
                Ray localRay = ray;
                Vector origin = Transform(instance.WorldToObject, ray.Origin, true);
                Vector direction = Transform(instance.WorldToObject, ray.Direction, false);
                MotionData::Pose pose;
                if (instance.Motion.Enabled)
                {
                    pose = InterpolateMotion(instance.Motion.Transform, sceneTime);
                    InverseInstanceMotion(pose, origin, direction);
                }
                for (double coordinate : origin)
                    if (!std::isfinite(coordinate) ||
                        std::abs(coordinate) > std::numeric_limits<float>::max())
                        throw std::invalid_argument(
                            "Instance ray origin exceeds object-space float coordinates.");
                localRay.Origin = {float(origin[0]), float(origin[1]), float(origin[2])};
                // Only extreme directions need conditioning. A power-of-two
                // change scales the interval and returned t as well, preserving
                // the original ray parameter and exactly representable endpoints.
                const double magnitude =
                    std::max({std::abs(direction[0]), std::abs(direction[1]), std::abs(direction[2])});
                constexpr double DirectionHeadroom = 4;
                double parameterScale = 1;
                if (magnitude > std::numeric_limits<float>::max() / DirectionHeadroom ||
                    magnitude < std::numeric_limits<float>::min() * DirectionHeadroom)
                {
                    if (magnitude == 0)
                        throw std::invalid_argument("Instance ray lost its object-space direction.");
                    int exponent;
                    std::frexp(magnitude, &exponent);
                    parameterScale = std::ldexp(1.0, exponent);
                }
                localRay.Direction = {float(direction[0] / parameterScale),
                                      float(direction[1] / parameterScale),
                                      float(direction[2] / parameterScale)};
                localRay.MinDistance = ScaledDistance(ray.MinDistance, parameterScale);
                localRay.MaxDistance = ScaledDistance(ray.MaxDistance, parameterScale);
                SurfaceHit localHit;
                localHit.Distance = ScaledDistance(hit.Distance, parameterScale);
                if (!Objects[instance.ObjectIndex].Acceleration->IntersectAtTime(localRay, localHit, tests, sceneTime, anyHit))
                    continue;
                // A BLAS any-hit query need not populate surface attributes.
                if (anyHit)
                    return true;
                const double worldDistance = double(localHit.Distance) / parameterScale;
                if (!std::isfinite(worldDistance) || worldDistance > hit.Distance)
                    continue;
                localHit.Distance = float(worldDistance);
                if (!std::isfinite(localHit.Distance) || localHit.Distance <= ray.MinDistance ||
                    localHit.Distance >= ray.MaxDistance || localHit.Distance > hit.Distance)
                    continue;
                localHit.PrimitiveIndex += instance.FirstPrimitive;
                if (localHit.Distance == hit.Distance && localHit.PrimitiveIndex >= hit.PrimitiveIndex)
                    continue;
                if (instance.Motion.Enabled)
                {
                    localHit.Normal = TransformMotionVector(pose, localHit.Normal);
                    localHit.GeometricNormal = TransformMotionVector(pose, localHit.GeometricNormal);
                }
                localHit.Normal = TransformNormal(instance.WorldToObject, localHit.Normal);
                localHit.GeometricNormal = TransformNormal(instance.WorldToObject, localHit.GeometricNormal);
                // (M^-T n).d = n.(M^-1 d): front/back survives every invertible
                // affine transform, including reflections. Do not face-forward ng.
                const double facing = double(localHit.Normal.X) * ray.Direction.X +
                                      double(localHit.Normal.Y) * ray.Direction.Y +
                                      double(localHit.Normal.Z) * ray.Direction.Z;
                if (facing > 0.0)
                    localHit.Normal = -localHit.Normal;
                hit = localHit;
                found = true;
            }
            ++nodeIndex;
        }
        return found;
    }
};
InstancedScene::InstancedScene(const Scene &scene, AccelerationOptions options,
                               CancellationProbe cancellation)
    : Implementation(std::make_unique<Impl>(scene, options, std::move(cancellation)))
{
}
InstancedScene::~InstancedScene() = default;
bool InstancedScene::WasCancelled() const
{
    return Implementation->Cancelled;
}
bool InstancedScene::Intersect(const Ray &ray, SurfaceHit &hit, uint64_t &tests, bool anyHit) const
{
    return Implementation->Intersect(ray, hit, tests, anyHit);
}
bool InstancedScene::IntersectAtTime(const Ray& ray, SurfaceHit& hit, uint64_t& tests,
                                    float sceneTime, bool anyHit) const
{
    return Implementation->Intersect(ray, hit, tests, anyHit, sceneTime);
}
bool InstancedScene::Occluded(const Ray &ray, uint64_t &tests) const
{
    SurfaceHit hit;
    hit.Distance = ray.MaxDistance;
    return Intersect(ray, hit, tests, true);
}
size_t InstancedScene::GeometryBuildCount() const
{
    return Implementation->Objects.size();
}
size_t InstancedScene::StoredPrimitiveCount() const
{
    size_t count = 0;
    for (const auto &object : Implementation->Objects)
        count += object.PrimitiveCount;
    return count;
}
} // namespace Arvo::RayTracing::Cpu
