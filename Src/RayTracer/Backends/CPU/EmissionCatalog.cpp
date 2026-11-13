#include "Renderer/EmissionCatalog.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace Arvo::RayTracing::Cpu
{
// A light catalog identifies every physical source once. Registered rectangle
// meshes are visibility proxies, so SampledEmitter geometry is not a second
// source. Other emissive triangles/spheres keep virtual traversal IDs, including
// every instance; their existing material emission is two-sided.
// Bounds use transformed vertices and exact affine-sphere coordinate extents.
// Double construction and outward float rounding keep distant/environment disks
// outside geometry after the center is rounded for shader upload.
namespace
{
constexpr uint32_t AxisCount = 3, MatrixStride = 4;
constexpr float Pi = 3.14159265358979323846f;
using Point = std::array<double, AxisCount>;
using Transform = std::array<float, AxisCount * MatrixStride>;
const Transform Identity{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
bool Colored(Float3 color)
{
    for (float channel : {color.X, color.Y, color.Z})
        if (!std::isfinite(channel) || channel < 0)
            throw std::invalid_argument("Invalid emission color.");
    return color.X > 0 || color.Y > 0 || color.Z > 0;
}
void Finite(Float3 value)
{
    if (!(std::isfinite(value.X) && std::isfinite(value.Y) && std::isfinite(value.Z)))
        throw std::invalid_argument("Nonfinite emission geometry.");
}
void Axis(Float3 axis)
{
    Finite(axis);
    if (axis.X == 0 && axis.Y == 0 && axis.Z == 0)
        throw std::invalid_argument("Zero emission axis.");
}
bool Lit(Float3 color, float intensity)
{
    if (!std::isfinite(intensity) || intensity < 0)
        throw std::invalid_argument("Invalid light intensity.");
    const bool positive = Colored(color);
    if (!std::isfinite(double(std::max({color.X, color.Y, color.Z})) * intensity) ||
        double(std::max({color.X, color.Y, color.Z})) * intensity > std::numeric_limits<float>::max())
        throw std::overflow_error("Light intensity exceeds the emission representation.");
    return positive && intensity > 0;
}
Point Apply(const Transform& matrix, Float3 value)
{
    Finite(value);
    Point p{};
    for (uint32_t row = 0; row < AxisCount; ++row)
    {
        const uint32_t i = row * MatrixStride;
        p[row] = double(matrix[i]) * value.X + double(matrix[i + 1]) * value.Y +
                 double(matrix[i + 2]) * value.Z + matrix[i + 3];
    }
    return p;
}
void Validate(const Transform& m)
{
    for (float value : m)
        if (!std::isfinite(value))
            throw std::invalid_argument("Nonfinite emitter instance.");
    const double determinant = double(m[0]) * (double(m[5]) * m[10] - double(m[6]) * m[9]) -
                               double(m[1]) * (double(m[4]) * m[10] - double(m[6]) * m[8]) +
                               double(m[2]) * (double(m[4]) * m[9] - double(m[5]) * m[8]);
    if (determinant == 0)
        throw std::invalid_argument("Singular emitter instance.");
}
struct Bounds
{
    Point Low{INFINITY, INFINITY, INFINITY}, High{-INFINITY, -INFINITY, -INFINITY};
    bool Empty = true;
    void Include(Point p)
    {
        Empty = false;
        for (uint32_t i = 0; i < AxisCount; ++i)
        {
            Low[i] = std::min(Low[i], p[i]);
            High[i] = std::max(High[i], p[i]);
        }
    }
};
bool HasArea(const Triangle& triangle)
{
    // An invertible affine transform preserves nonzero area. Check before
    // translation so a distant instance cannot lose its emitter to cancellation.
    Point a{triangle.Position0.X, triangle.Position0.Y, triangle.Position0.Z};
    Point b{triangle.Position1.X, triangle.Position1.Y, triangle.Position1.Z};
    Point c{triangle.Position2.X, triangle.Position2.Y, triangle.Position2.Z};
    for (uint32_t i = 0; i < AxisCount; ++i)
    {
        b[i] -= a[i];
        c[i] -= a[i];
    }
    return b[1] * c[2] != b[2] * c[1] || b[2] * c[0] != b[0] * c[2] || b[0] * c[1] != b[1] * c[0];
}
} // namespace
EmissionCatalog BuildEmissionCatalog(const Scene& scene)
{
    EmissionCatalog catalog;
    Bounds bounds;
    const auto add = [&](EmitterKind kind, size_t index, uint32_t instance = InvalidEmitterIndex,
                         uint64_t primitive = InvalidRayPrimitiveIndex) {
        if (index > std::numeric_limits<uint32_t>::max() || catalog.Sources.size() >= MaximumUniformEmitters)
            throw std::overflow_error("Emission catalog exceeds uniform selection capacity.");
        catalog.Sources.push_back({kind, uint32_t(index), instance, uint32_t(primitive)});
    };
    for (size_t i = 0; i < scene.DirectionalLights.size(); ++i)
    {
        const auto& light = scene.DirectionalLights[i];
        Axis(light.DirectionToLight);
        if (Lit(light.Color, light.Intensity))
            add(EmitterKind::Directional, i);
    }
    for (size_t i = 0; i < scene.PointLights.size(); ++i)
    {
        const auto& light = scene.PointLights[i];
        Finite(light.Position);
        if (Lit(light.Color, light.Intensity))
            add(EmitterKind::Point, i);
    }
    for (size_t i = 0; i < scene.RectAreaLights.size(); ++i)
    {
        const auto& light = scene.RectAreaLights[i];
        Finite(light.Position);
        Axis(light.Direction);
        if (!(std::isfinite(light.Size.X) && std::isfinite(light.Size.Y) && light.Size.X > 0 &&
              light.Size.Y > 0))
            throw std::invalid_argument("Invalid rectangular emitter size.");
        const float area = light.Size.X * light.Size.Y;
        if (!(area > 0) || !std::isfinite(area) || !std::isfinite(1 / area))
            throw std::overflow_error("Rectangular emitter area density is not representable.");
        if (Lit(light.Color, light.Intensity))
            add(EmitterKind::Rectangle, i);
    }
    const bool top = Colored(scene.BackgroundTop), bottom = Colored(scene.BackgroundBottom);
    if (top || bottom)
        add(EmitterKind::Environment, 0);
    uint64_t primitiveBase = 0;
    const auto geometry = [&](const SceneGeometry& object, const Transform& transform, uint32_t instance) {
        const uint64_t count = uint64_t(object.Triangles.size()) + object.Spheres.size();
        if (count > InvalidRayPrimitiveIndex - primitiveBase)
            throw std::overflow_error("Emitter primitive ID overflow.");
        const auto emits = [&](uint32_t material) {
            if (material >= scene.Materials.size())
                throw std::invalid_argument("Invalid emitter material index.");
            const auto& m = scene.Materials[material];
            const bool nonblack = Colored(m.Emission);
            return nonblack && !m.SampledEmitter;
        };
        for (size_t i = 0; i < object.Triangles.size(); ++i)
        {
            const auto& triangle = object.Triangles[i];
            const Point a = Apply(transform, triangle.Position0), b = Apply(transform, triangle.Position1),
                        c = Apply(transform, triangle.Position2);
            bounds.Include(a);
            bounds.Include(b);
            bounds.Include(c);
            if (emits(triangle.MaterialIndex) && HasArea(triangle))
                add(EmitterKind::Triangle, i, instance, primitiveBase + i);
        }
        for (size_t i = 0; i < object.Spheres.size(); ++i)
        {
            const auto& sphere = object.Spheres[i];
            if (!(std::isfinite(sphere.Radius) && sphere.Radius > 0))
                throw std::invalid_argument("Invalid emitter sphere radius.");
            Point low = Apply(transform, sphere.Center), high = low;
            for (uint32_t row = 0; row < AxisCount; ++row)
            {
                const uint32_t offset = row * MatrixStride;
                const double extent = double(sphere.Radius) * std::hypot(double(transform[offset]),
                                                                         double(transform[offset + 1]),
                                                                         double(transform[offset + 2]));
                low[row] -= extent;
                high[row] += extent;
            }
            bounds.Include(low);
            bounds.Include(high);
            if (emits(sphere.MaterialIndex))
                add(EmitterKind::Sphere, i, instance, primitiveBase + object.Triangles.size() + i);
        }
        primitiveBase += count;
    };
    geometry(scene, Identity, InvalidEmitterIndex);
    for (size_t i = 0; i < scene.Instances.size(); ++i)
    {
        const auto& instance = scene.Instances[i];
        if (i >= InvalidEmitterIndex || instance.GeometryIndex >= scene.Geometries.size() ||
            !scene.Geometries[instance.GeometryIndex])
            throw std::invalid_argument("Invalid emitter geometry instance.");
        Validate(instance.ObjectToWorld);
        geometry(*scene.Geometries[instance.GeometryIndex], instance.ObjectToWorld, uint32_t(i));
    }
    if (!bounds.Empty)
    {
        Point center{}, extent{};
        for (uint32_t axis = 0; axis < AxisCount; ++axis)
        {
            center[axis] = float((bounds.Low[axis] + bounds.High[axis]) / 2);
            extent[axis] = std::max(center[axis] - bounds.Low[axis], bounds.High[axis] - center[axis]);
        }
        catalog.Center = {float(center[0]), float(center[1]), float(center[2])};
        catalog.Radius = std::nextafter(float(std::hypot(extent[0], extent[1], extent[2])), INFINITY);
        const float diskArea = Pi * catalog.Radius * catalog.Radius;
        if (!std::isfinite(diskArea) || !(diskArea > 0) || !std::isfinite(1 / diskArea))
            throw std::overflow_error("Scene emission domain cannot be represented in float densities.");
    }
    return catalog;
}
} // namespace Arvo::RayTracing::Cpu
