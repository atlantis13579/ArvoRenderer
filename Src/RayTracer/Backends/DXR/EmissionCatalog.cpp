#include "Renderer/EmissionCatalog.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace Arvo::RayTracing::DxrEmission
{
// Independent DXR emitter metadata builder. Hardware BLAS/TLAS handles are not
// emitter identities: each instance has distinct virtual primitive IDs. This
// table follows the same scene order, also used by software DXR accelerators.
// Rectangle visibility meshes never duplicate an analytic source. Unregistered
// material emission is two-sided, as in the existing camera-path hit convention.
namespace
{
constexpr unsigned Dimension = 3, AffineStride = 4;
constexpr float Pi = 3.14159265358979323846f;
using Coordinate = std::array<double, Dimension>;
using Affine = std::array<float, Dimension * AffineStride>;
void FinitePoint(Float3 p)
{
    if (!(std::isfinite(p.X) && std::isfinite(p.Y) && std::isfinite(p.Z)))
        throw std::invalid_argument("DXR emission input is nonfinite.");
}
bool Nonblack(Float3 rgb)
{
    FinitePoint(rgb);
    if (rgb.X < 0 || rgb.Y < 0 || rgb.Z < 0)
        throw std::invalid_argument("DXR emission must be nonnegative.");
    return rgb.X > 0 || rgb.Y > 0 || rgb.Z > 0;
}
bool NonblackLight(Float3 rgb, float intensity)
{
    bool color = Nonblack(rgb);
    if (!std::isfinite(intensity) || intensity < 0)
        throw std::invalid_argument("Invalid DXR emission intensity.");
    if (double(std::max({rgb.X, rgb.Y, rgb.Z})) * intensity > std::numeric_limits<float>::max())
        throw std::overflow_error("DXR emission strength overflow.");
    return color && intensity > 0;
}
void Oriented(Float3 v)
{
    FinitePoint(v);
    if (v.X == 0 && v.Y == 0 && v.Z == 0)
        throw std::invalid_argument("Invalid DXR emission orientation.");
}
Coordinate WorldPoint(Float3 p, const Affine& transform)
{
    FinitePoint(p);
    Coordinate result{};
    for (unsigned axis = 0; axis < Dimension; ++axis)
    {
        const unsigned row = axis * AffineStride;
        result[axis] = double(transform[row]) * p.X + double(transform[row + 1]) * p.Y +
                       double(transform[row + 2]) * p.Z + transform[row + 3];
    }
    return result;
}
void ValidAffine(const Affine& m)
{
    for (float item : m)
        if (!std::isfinite(item))
            throw std::invalid_argument("DXR emitter transform is nonfinite.");
    const double minor0 = double(m[5]) * m[10] - double(m[6]) * m[9];
    const double minor1 = double(m[4]) * m[10] - double(m[6]) * m[8];
    const double minor2 = double(m[4]) * m[9] - double(m[5]) * m[8];
    if (double(m[0]) * minor0 - double(m[1]) * minor1 + double(m[2]) * minor2 == 0)
        throw std::invalid_argument("DXR emitter transform is singular.");
}
struct CatalogState
{
    EmissionCatalog Output;
    Coordinate Lower{INFINITY, INFINITY, INFINITY}, Upper{-INFINITY, -INFINITY, -INFINITY};
    bool HasBounds = false;
    uint64_t PrimitiveOffset = 0;
    void Extend(Coordinate p)
    {
        HasBounds = true;
        for (unsigned axis = 0; axis < Dimension; ++axis)
        {
            Lower[axis] = std::min(Lower[axis], p[axis]);
            Upper[axis] = std::max(Upper[axis], p[axis]);
        }
    }
    void Insert(EmitterKind kind, size_t index, uint32_t instance = InvalidEmitterIndex,
                uint64_t primitive = InvalidRayPrimitiveIndex)
    {
        if (index > std::numeric_limits<uint32_t>::max() || Output.Sources.size() >= MaximumUniformEmitters)
            throw std::overflow_error("DXR emission catalog exceeds selection capacity.");
        Output.Sources.push_back({kind, uint32_t(index), instance, uint32_t(primitive)});
    }
    void Object(const Scene& scene, const SceneGeometry& geometry, const Affine& matrix, uint32_t instance)
    {
        uint64_t primitives = uint64_t(geometry.Triangles.size()) + geometry.Spheres.size();
        if (primitives > InvalidRayPrimitiveIndex - PrimitiveOffset)
            throw std::overflow_error("DXR emitter identity overflow.");
        const auto emissive = [&](uint32_t id) {
            if (id >= scene.Materials.size())
                throw std::invalid_argument("Missing DXR emitter material.");
            const Material& m = scene.Materials[id];
            const bool emitting = Nonblack(m.Emission);
            return emitting && !m.SampledEmitter;
        };
        for (size_t i = 0; i < geometry.Triangles.size(); ++i)
        {
            const Triangle& triangle = geometry.Triangles[i];
            Coordinate a = WorldPoint(triangle.Position0, matrix), b = WorldPoint(triangle.Position1, matrix),
                       c = WorldPoint(triangle.Position2, matrix);
            Extend(a);
            Extend(b);
            Extend(c);
            const bool emitting = emissive(triangle.MaterialIndex);
            // Invertibility preserves area support; avoid cancellation from
            // world translation when deciding whether this source exists.
            a = {triangle.Position0.X, triangle.Position0.Y, triangle.Position0.Z};
            b = {triangle.Position1.X, triangle.Position1.Y, triangle.Position1.Z};
            c = {triangle.Position2.X, triangle.Position2.Y, triangle.Position2.Z};
            for (unsigned axis = 0; axis < Dimension; ++axis)
            {
                b[axis] -= a[axis];
                c[axis] -= a[axis];
            }
            const double areaX = b[1] * c[2] - b[2] * c[1], areaY = b[2] * c[0] - b[0] * c[2],
                         areaZ = b[0] * c[1] - b[1] * c[0];
            if (emitting && (areaX != 0 || areaY != 0 || areaZ != 0))
                Insert(EmitterKind::Triangle, i, instance, PrimitiveOffset + i);
        }
        for (size_t i = 0; i < geometry.Spheres.size(); ++i)
        {
            const Sphere& sphere = geometry.Spheres[i];
            if (!(sphere.Radius > 0) || !std::isfinite(sphere.Radius))
                throw std::invalid_argument("Invalid DXR emitter sphere radius.");
            Coordinate minimum = WorldPoint(sphere.Center, matrix), maximum = minimum;
            for (unsigned axis = 0; axis < Dimension; ++axis)
            {
                unsigned row = axis * AffineStride;
                double extent = sphere.Radius * std::hypot(double(matrix[row]), double(matrix[row + 1]),
                                                           double(matrix[row + 2]));
                minimum[axis] -= extent;
                maximum[axis] += extent;
            }
            Extend(minimum);
            Extend(maximum);
            if (emissive(sphere.MaterialIndex))
                Insert(EmitterKind::Sphere, i, instance, PrimitiveOffset + geometry.Triangles.size() + i);
        }
        PrimitiveOffset += primitives;
    }
    void Domain()
    {
        if (!HasBounds)
            return;
        Coordinate rounded{}, extent{};
        for (unsigned axis = 0; axis < Dimension; ++axis)
        {
            rounded[axis] = float((Lower[axis] + Upper[axis]) / 2);
            extent[axis] = std::max(rounded[axis] - Lower[axis], Upper[axis] - rounded[axis]);
        }
        Output.Center = {float(rounded[0]), float(rounded[1]), float(rounded[2])};
        Output.Radius = std::nextafter(float(std::hypot(extent[0], extent[1], extent[2])), INFINITY);
        const float area = Pi * Output.Radius * Output.Radius;
        if (!std::isfinite(area) || !(area > 0) || !std::isfinite(1 / area))
            throw std::overflow_error("DXR emission domain cannot carry a float area PDF.");
    }
};
} // namespace
EmissionCatalog BuildCatalog(const Scene& scene)
{
    CatalogState state;
    for (size_t i = 0; i < scene.DirectionalLights.size(); ++i)
    {
        const auto& l = scene.DirectionalLights[i];
        Oriented(l.DirectionToLight);
        if (NonblackLight(l.Color, l.Intensity))
            state.Insert(EmitterKind::Directional, i);
    }
    for (size_t i = 0; i < scene.PointLights.size(); ++i)
    {
        const auto& l = scene.PointLights[i];
        FinitePoint(l.Position);
        if (NonblackLight(l.Color, l.Intensity))
            state.Insert(EmitterKind::Point, i);
    }
    for (size_t i = 0; i < scene.RectAreaLights.size(); ++i)
    {
        const auto& l = scene.RectAreaLights[i];
        FinitePoint(l.Position);
        Oriented(l.Direction);
        if (!(l.Size.X > 0 && l.Size.Y > 0 && std::isfinite(l.Size.X) && std::isfinite(l.Size.Y)))
            throw std::invalid_argument("Invalid DXR rectangular emitter dimensions.");
        const float area = l.Size.X * l.Size.Y;
        if (!(area > 0) || !std::isfinite(area) || !std::isfinite(1 / area))
            throw std::overflow_error("DXR rectangular emitter area PDF is out of range.");
        if (NonblackLight(l.Color, l.Intensity))
            state.Insert(EmitterKind::Rectangle, i);
    }
    const bool top = Nonblack(scene.BackgroundTop), bottom = Nonblack(scene.BackgroundBottom);
    if (top || bottom)
        state.Insert(EmitterKind::Environment, 0);
    const Affine identity{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
    state.Object(scene, scene, identity, InvalidEmitterIndex);
    for (size_t i = 0; i < scene.Instances.size(); ++i)
    {
        const auto& instance = scene.Instances[i];
        if (i >= InvalidEmitterIndex || instance.GeometryIndex >= scene.Geometries.size() ||
            !scene.Geometries[instance.GeometryIndex])
            throw std::invalid_argument("DXR emitter instance references missing geometry.");
        ValidAffine(instance.ObjectToWorld);
        state.Object(scene, *scene.Geometries[instance.GeometryIndex], instance.ObjectToWorld, uint32_t(i));
    }
    state.Domain();
    return std::move(state.Output);
}
} // namespace Arvo::RayTracing::DxrEmission
