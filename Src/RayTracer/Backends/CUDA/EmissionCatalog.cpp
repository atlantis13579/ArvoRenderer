#include "Renderer/EmissionCatalog.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace Arvo::RayTracing::CudaEmissionHost
{
// Native CUDA scene preparation has its own implementation. This CPU-built
// table is uploaded as data; emitted-ray selection itself runs in CUDA.
// Analytic-light proxies are omitted; arbitrary emissive geometry retains the
// traversal's virtual primitive identity. A sphere stays affine/analytic.
namespace
{
constexpr unsigned Axes = 3, RowWidth = 4;
constexpr float Pi = 3.14159265358979323846f;
using Triple = std::array<double, Axes>;
using Matrix = std::array<float, Axes * RowWidth>;
void RequireFinite(Float3 value)
{
    if (!std::isfinite(value.X) || !std::isfinite(value.Y) || !std::isfinite(value.Z))
        throw std::invalid_argument("CUDA emission input is nonfinite.");
}
bool Radiates(Float3 rgb)
{
    RequireFinite(rgb);
    if (rgb.X < 0 || rgb.Y < 0 || rgb.Z < 0)
        throw std::invalid_argument("CUDA emission color is negative.");
    return std::max({rgb.X, rgb.Y, rgb.Z}) > 0;
}
bool LightRadiates(Float3 rgb, float intensity)
{
    bool positive = Radiates(rgb);
    if (intensity < 0 || !std::isfinite(intensity))
        throw std::invalid_argument("Invalid CUDA light intensity.");
    if (double(std::max({rgb.X, rgb.Y, rgb.Z})) * intensity > std::numeric_limits<float>::max())
        throw std::overflow_error("CUDA light strength is not representable.");
    return positive && intensity > 0;
}
void RequireAxis(Float3 direction)
{
    RequireFinite(direction);
    if (direction.X == 0 && direction.Y == 0 && direction.Z == 0)
        throw std::invalid_argument("CUDA emission axis is zero.");
}
Triple TransformPoint(const Matrix& m, Float3 point)
{
    RequireFinite(point);
    Triple p{};
    const double coordinates[]{point.X, point.Y, point.Z};
    for (unsigned row = 0; row < Axes; ++row)
    {
        p[row] = m[row * RowWidth + Axes];
        for (unsigned column = 0; column < Axes; ++column)
            p[row] += double(m[row * RowWidth + column]) * coordinates[column];
    }
    return p;
}
void RequireTransform(const Matrix& m)
{
    for (float x : m)
        if (!std::isfinite(x))
            throw std::invalid_argument("CUDA emitter transform is nonfinite.");
    Triple cross{double(m[5]) * m[10] - double(m[6]) * m[9], double(m[6]) * m[8] - double(m[4]) * m[10],
                 double(m[4]) * m[9] - double(m[5]) * m[8]};
    if (double(m[0]) * cross[0] + double(m[1]) * cross[1] + double(m[2]) * cross[2] == 0)
        throw std::invalid_argument("CUDA emitter transform is singular.");
}
struct Extents
{
    Triple Low{INFINITY, INFINITY, INFINITY}, High{-INFINITY, -INFINITY, -INFINITY};
    bool Occupied = false;
    void Enclose(Triple position)
    {
        Occupied = true;
        for (unsigned axis = 0; axis < Axes; ++axis)
        {
            Low[axis] = std::min(Low[axis], position[axis]);
            High[axis] = std::max(High[axis], position[axis]);
        }
    }
};
struct Preparation
{
    const Scene& Input;
    EmissionCatalog Catalog;
    Extents Bounds;
    uint64_t FirstPrimitive = 0;
    void Add(EmitterKind kind, size_t index, uint32_t instance = InvalidEmitterIndex,
             uint64_t primitive = InvalidRayPrimitiveIndex)
    {
        if (index > std::numeric_limits<uint32_t>::max() || Catalog.Sources.size() >= MaximumUniformEmitters)
            throw std::overflow_error("CUDA emitter table exceeds uniform selection range.");
        Catalog.Sources.push_back({kind, uint32_t(index), instance, uint32_t(primitive)});
    }
    bool MaterialEmits(uint32_t id)
    {
        if (id >= Input.Materials.size())
            throw std::invalid_argument("CUDA emitter references missing material.");
        const auto& material = Input.Materials[id];
        bool positive = Radiates(material.Emission);
        return positive && !material.SampledEmitter;
    }
    void AddObject(const SceneGeometry& object, const Matrix& transform, uint32_t instance)
    {
        const uint64_t count = uint64_t(object.Triangles.size()) + object.Spheres.size();
        if (count > InvalidRayPrimitiveIndex - FirstPrimitive)
            throw std::overflow_error("CUDA emission primitive IDs overflow.");
        for (size_t i = 0; i < object.Triangles.size(); ++i)
        {
            const auto& triangle = object.Triangles[i];
            Triple a = TransformPoint(transform, triangle.Position0),
                   b = TransformPoint(transform, triangle.Position1),
                   c = TransformPoint(transform, triangle.Position2);
            Bounds.Enclose(a);
            Bounds.Enclose(b);
            Bounds.Enclose(c);
            bool emissive = MaterialEmits(triangle.MaterialIndex);
            // Test degeneracy before the invertible affine: large translations
            // may cancel world-space edges but cannot remove a physical source.
            a = {triangle.Position0.X, triangle.Position0.Y, triangle.Position0.Z};
            b = {triangle.Position1.X, triangle.Position1.Y, triangle.Position1.Z};
            c = {triangle.Position2.X, triangle.Position2.Y, triangle.Position2.Z};
            for (unsigned axis = 0; axis < Axes; ++axis)
            {
                b[axis] -= a[axis];
                c[axis] -= a[axis];
            }
            Triple cross{b[1] * c[2] - b[2] * c[1], b[2] * c[0] - b[0] * c[2], b[0] * c[1] - b[1] * c[0]};
            if (emissive && (cross[0] != 0 || cross[1] != 0 || cross[2] != 0))
                Add(EmitterKind::Triangle, i, instance, FirstPrimitive + i);
        }
        for (size_t i = 0; i < object.Spheres.size(); ++i)
        {
            const auto& sphere = object.Spheres[i];
            if (!std::isfinite(sphere.Radius) || sphere.Radius <= 0)
                throw std::invalid_argument("Invalid CUDA emitter sphere radius.");
            Triple minimum = TransformPoint(transform, sphere.Center), maximum = minimum;
            for (unsigned axis = 0; axis < Axes; ++axis)
            {
                unsigned row = axis * RowWidth;
                double half = sphere.Radius * std::hypot(double(transform[row]), double(transform[row + 1]),
                                                         double(transform[row + 2]));
                minimum[axis] -= half;
                maximum[axis] += half;
            }
            Bounds.Enclose(minimum);
            Bounds.Enclose(maximum);
            if (MaterialEmits(sphere.MaterialIndex))
                Add(EmitterKind::Sphere, i, instance, FirstPrimitive + object.Triangles.size() + i);
        }
        FirstPrimitive += count;
    }
    void FinishDomain()
    {
        if (!Bounds.Occupied)
            return;
        Triple center{}, radius{};
        for (unsigned i = 0; i < Axes; ++i)
        {
            center[i] = float((Bounds.Low[i] + Bounds.High[i]) / 2);
            radius[i] = std::max(center[i] - Bounds.Low[i], Bounds.High[i] - center[i]);
        }
        Catalog.Center = {float(center[0]), float(center[1]), float(center[2])};
        Catalog.Radius = std::nextafter(float(std::hypot(radius[0], radius[1], radius[2])), INFINITY);
        float diskArea = Pi * Catalog.Radius * Catalog.Radius;
        if (!(diskArea > 0) || !std::isfinite(diskArea) || !std::isfinite(1 / diskArea))
            throw std::overflow_error("CUDA scene emission domain exceeds float density range.");
    }
};
} // namespace
EmissionCatalog BuildCatalog(const Scene& scene)
{
    Preparation builder{scene, {}, {}, 0};
    for (size_t i = 0; i < scene.DirectionalLights.size(); ++i)
    {
        const auto& light = scene.DirectionalLights[i];
        RequireAxis(light.DirectionToLight);
        if (LightRadiates(light.Color, light.Intensity))
            builder.Add(EmitterKind::Directional, i);
    }
    for (size_t i = 0; i < scene.PointLights.size(); ++i)
    {
        const auto& light = scene.PointLights[i];
        RequireFinite(light.Position);
        if (LightRadiates(light.Color, light.Intensity))
            builder.Add(EmitterKind::Point, i);
    }
    for (size_t i = 0; i < scene.RectAreaLights.size(); ++i)
    {
        const auto& light = scene.RectAreaLights[i];
        RequireFinite(light.Position);
        RequireAxis(light.Direction);
        if (!std::isfinite(light.Size.X) || !std::isfinite(light.Size.Y) || light.Size.X <= 0 ||
            light.Size.Y <= 0)
            throw std::invalid_argument("Invalid CUDA rectangular emitter dimensions.");
        const float area = light.Size.X * light.Size.Y;
        if (!(area > 0) || !std::isfinite(area) || !std::isfinite(1 / area))
            throw std::overflow_error("CUDA rectangle has no representable area PDF.");
        if (LightRadiates(light.Color, light.Intensity))
            builder.Add(EmitterKind::Rectangle, i);
    }
    bool top = Radiates(scene.BackgroundTop), bottom = Radiates(scene.BackgroundBottom);
    if (top || bottom)
        builder.Add(EmitterKind::Environment, 0);
    const Matrix identity{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
    builder.AddObject(scene, identity, InvalidEmitterIndex);
    for (size_t i = 0; i < scene.Instances.size(); ++i)
    {
        const auto& instance = scene.Instances[i];
        if (i >= InvalidEmitterIndex || instance.GeometryIndex >= scene.Geometries.size() ||
            !scene.Geometries[instance.GeometryIndex])
            throw std::invalid_argument("Invalid CUDA emitter geometry reference.");
        RequireTransform(instance.ObjectToWorld);
        builder.AddObject(*scene.Geometries[instance.GeometryIndex], instance.ObjectToWorld, uint32_t(i));
    }
    builder.FinishDomain();
    return std::move(builder.Catalog);
}
} // namespace Arvo::RayTracing::CudaEmissionHost
