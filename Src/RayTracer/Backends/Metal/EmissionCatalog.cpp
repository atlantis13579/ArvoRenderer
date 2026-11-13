#include "Renderer/EmissionCatalog.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace Arvo::RayTracing::MetalEmission
{
// Independent Metal host preparation. The native GPU will consume references;
// this step neither tessellates spheres nor duplicates immutable geometry.
// Instance lights receive unique virtual primitive IDs. SampledEmitter material
// geometry is already represented by an analytic light and must not be counted
// twice. Every other nonblack material emitter remains in the catalog.
namespace
{
constexpr unsigned CoordinateCount = 3, AffineRowSize = 4;
constexpr float Pi = 3.14159265358979323846f;
using Position = std::array<double, CoordinateCount>;
using Affine = std::array<float, CoordinateCount * AffineRowSize>;
void Finite(Float3 v)
{
    if (!std::isfinite(v.X) || !std::isfinite(v.Y) || !std::isfinite(v.Z))
        throw std::invalid_argument("Nonfinite Metal emission input.");
}
bool VisibleEmission(Float3 color)
{
    Finite(color);
    if (color.X < 0 || color.Y < 0 || color.Z < 0)
        throw std::invalid_argument("Negative Metal emission color.");
    return color.X != 0 || color.Y != 0 || color.Z != 0;
}
bool ActiveLight(Float3 color, float strength)
{
    bool bright = VisibleEmission(color);
    if (!std::isfinite(strength) || strength < 0)
        throw std::invalid_argument("Invalid Metal light intensity.");
    if (double(std::max({color.X, color.Y, color.Z})) * strength > std::numeric_limits<float>::max())
        throw std::overflow_error("Metal light strength is out of range.");
    return bright && strength > 0;
}
void Nonzero(Float3 axis)
{
    Finite(axis);
    if (axis.X == 0 && axis.Y == 0 && axis.Z == 0)
        throw std::invalid_argument("Zero Metal emitter direction.");
}
Position Transformed(const Affine& m, Float3 p)
{
    Finite(p);
    Position result{};
    for (unsigned axis = 0; axis < CoordinateCount; ++axis)
    {
        const unsigned row = axis * AffineRowSize;
        result[axis] =
            double(m[row]) * p.X + double(m[row + 1]) * p.Y + double(m[row + 2]) * p.Z + m[row + 3];
    }
    return result;
}
void CheckAffine(const Affine& m)
{
    for (float element : m)
        if (!std::isfinite(element))
            throw std::invalid_argument("Nonfinite Metal light transform.");
    const double determinant = double(m[0]) * (double(m[5]) * m[10] - double(m[6]) * m[9]) -
                               double(m[1]) * (double(m[4]) * m[10] - double(m[6]) * m[8]) +
                               double(m[2]) * (double(m[4]) * m[9] - double(m[5]) * m[8]);
    if (determinant == 0)
        throw std::invalid_argument("Singular Metal light transform.");
}
class Collector
{
    const Scene& Input_;
    EmissionCatalog Result_;
    Position Low_{INFINITY, INFINITY, INFINITY}, High_{-INFINITY, -INFINITY, -INFINITY};
    bool Nonempty_ = false;
    uint64_t Base_ = 0;
    void Point(Position value)
    {
        Nonempty_ = true;
        for (unsigned i = 0; i < CoordinateCount; ++i)
        {
            Low_[i] = std::min(Low_[i], value[i]);
            High_[i] = std::max(High_[i], value[i]);
        }
    }
    bool MaterialLight(uint32_t index)
    {
        if (index >= Input_.Materials.size())
            throw std::invalid_argument("Missing Metal emitter material.");
        const auto& material = Input_.Materials[index];
        bool bright = VisibleEmission(material.Emission);
        return bright && !material.SampledEmitter;
    }

  public:
    explicit Collector(const Scene& scene) : Input_(scene) {}
    void Source(EmitterKind kind, size_t index, uint32_t instance = InvalidEmitterIndex,
                uint64_t primitive = InvalidRayPrimitiveIndex)
    {
        if (Result_.Sources.size() >= MaximumUniformEmitters || index > std::numeric_limits<uint32_t>::max())
            throw std::overflow_error("Metal source table exceeds uniform selection capacity.");
        Result_.Sources.push_back({kind, uint32_t(index), instance, uint32_t(primitive)});
    }
    void Geometry(const SceneGeometry& geometry, const Affine& matrix, uint32_t instance)
    {
        const uint64_t count = uint64_t(geometry.Triangles.size()) + geometry.Spheres.size();
        if (count > InvalidRayPrimitiveIndex - Base_)
            throw std::overflow_error("Metal light primitive identity overflow.");
        for (size_t index = 0; index < geometry.Triangles.size(); ++index)
        {
            const auto& triangle = geometry.Triangles[index];
            Position a = Transformed(matrix, triangle.Position0), b = Transformed(matrix, triangle.Position1),
                     c = Transformed(matrix, triangle.Position2);
            Point(a);
            Point(b);
            Point(c);
            const bool light = MaterialLight(triangle.MaterialIndex);
            // A validated affine cannot turn nonzero object area into zero.
            // Preserve tiny emitters translated to distant world coordinates.
            a = {triangle.Position0.X, triangle.Position0.Y, triangle.Position0.Z};
            b = {triangle.Position1.X, triangle.Position1.Y, triangle.Position1.Z};
            c = {triangle.Position2.X, triangle.Position2.Y, triangle.Position2.Z};
            for (unsigned axis = 0; axis < CoordinateCount; ++axis)
            {
                b[axis] -= a[axis];
                c[axis] -= a[axis];
            }
            bool area =
                b[1] * c[2] != b[2] * c[1] || b[2] * c[0] != b[0] * c[2] || b[0] * c[1] != b[1] * c[0];
            if (light && area)
                Source(EmitterKind::Triangle, index, instance, Base_ + index);
        }
        for (size_t index = 0; index < geometry.Spheres.size(); ++index)
        {
            const auto& sphere = geometry.Spheres[index];
            if (!std::isfinite(sphere.Radius) || !(sphere.Radius > 0))
                throw std::invalid_argument("Invalid Metal light sphere radius.");
            Position lower = Transformed(matrix, sphere.Center), upper = lower;
            for (unsigned axis = 0; axis < CoordinateCount; ++axis)
            {
                const unsigned row = axis * AffineRowSize;
                double extent =
                    double(sphere.Radius) *
                    std::hypot(double(matrix[row]), double(matrix[row + 1]), double(matrix[row + 2]));
                lower[axis] -= extent;
                upper[axis] += extent;
            }
            Point(lower);
            Point(upper);
            if (MaterialLight(sphere.MaterialIndex))
                Source(EmitterKind::Sphere, index, instance, Base_ + geometry.Triangles.size() + index);
        }
        Base_ += count;
    }
    EmissionCatalog Finish()
    {
        if (Nonempty_)
        {
            Position center{}, distance{};
            for (unsigned axis = 0; axis < CoordinateCount; ++axis)
            {
                center[axis] = float((Low_[axis] + High_[axis]) / 2);
                distance[axis] = std::max(center[axis] - Low_[axis], High_[axis] - center[axis]);
            }
            Result_.Center = {float(center[0]), float(center[1]), float(center[2])};
            Result_.Radius =
                std::nextafter(float(std::hypot(distance[0], distance[1], distance[2])), INFINITY);
            const float diskArea = Pi * Result_.Radius * Result_.Radius;
            if (!(diskArea > 0) || !std::isfinite(diskArea) || !std::isfinite(1 / diskArea))
                throw std::overflow_error("Metal emission domain has no representable area density.");
        }
        return std::move(Result_);
    }
};
} // namespace
EmissionCatalog BuildCatalog(const Scene& scene)
{
    Collector collector(scene);
    for (size_t index = 0; index < scene.DirectionalLights.size(); ++index)
    {
        const auto& light = scene.DirectionalLights[index];
        Nonzero(light.DirectionToLight);
        if (ActiveLight(light.Color, light.Intensity))
            collector.Source(EmitterKind::Directional, index);
    }
    for (size_t index = 0; index < scene.PointLights.size(); ++index)
    {
        const auto& light = scene.PointLights[index];
        Finite(light.Position);
        if (ActiveLight(light.Color, light.Intensity))
            collector.Source(EmitterKind::Point, index);
    }
    for (size_t index = 0; index < scene.RectAreaLights.size(); ++index)
    {
        const auto& light = scene.RectAreaLights[index];
        Finite(light.Position);
        Nonzero(light.Direction);
        if (!std::isfinite(light.Size.X) || !std::isfinite(light.Size.Y) ||
            !(light.Size.X > 0 && light.Size.Y > 0))
            throw std::invalid_argument("Invalid Metal rectangular light size.");
        const float area = light.Size.X * light.Size.Y;
        if (!(area > 0) || !std::isfinite(area) || !std::isfinite(1 / area))
            throw std::overflow_error("Metal rectangle area density is not representable.");
        if (ActiveLight(light.Color, light.Intensity))
            collector.Source(EmitterKind::Rectangle, index);
    }
    bool top = VisibleEmission(scene.BackgroundTop), bottom = VisibleEmission(scene.BackgroundBottom);
    if (top || bottom)
        collector.Source(EmitterKind::Environment, 0);
    const Affine identity{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
    collector.Geometry(scene, identity, InvalidEmitterIndex);
    for (size_t index = 0; index < scene.Instances.size(); ++index)
    {
        const auto& instance = scene.Instances[index];
        if (index >= InvalidEmitterIndex || instance.GeometryIndex >= scene.Geometries.size() ||
            !scene.Geometries[instance.GeometryIndex])
            throw std::invalid_argument("Missing Metal emitter instance geometry.");
        CheckAffine(instance.ObjectToWorld);
        collector.Geometry(*scene.Geometries[instance.GeometryIndex], instance.ObjectToWorld,
                           uint32_t(index));
    }
    return collector.Finish();
}
} // namespace Arvo::RayTracing::MetalEmission
