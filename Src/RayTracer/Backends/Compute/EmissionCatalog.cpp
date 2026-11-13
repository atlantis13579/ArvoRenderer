#include "Renderer/EmissionCatalog.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace Arvo::RayTracing::ComputeEmission
{
// Independent host-side scene preparation for compute light paths. Only metadata
// is shared with other backends. Primitive numbering follows this backend's
// legacy-object / instance ordering; unused geometry assets do not affect bounds.
// A transformed sphere has coordinate radius r * length(matrix row), including
// shear and reflection. Expanding just its center or its diagonal scale is wrong.
namespace
{
constexpr uint32_t Dimensions = 3, MatrixColumns = 4;
constexpr float Pi = 3.14159265358979323846f;
using Vector = std::array<double, Dimensions>;
using Matrix = std::array<float, Dimensions * MatrixColumns>;
class Builder
{
    const Scene& Scene_;
    EmissionCatalog Result_;
    Vector Minimum_{INFINITY, INFINITY, INFINITY}, Maximum_{-INFINITY, -INFINITY, -INFINITY};
    uint64_t PrimitiveBase_ = 0;
    bool GeometryPresent_ = false;
    static void CheckVector(Float3 p)
    {
        if (!std::isfinite(p.X) || !std::isfinite(p.Y) || !std::isfinite(p.Z))
            throw std::invalid_argument("Nonfinite compute emission input.");
    }
    static bool Positive(Float3 rgb)
    {
        CheckVector(rgb);
        if (rgb.X < 0 || rgb.Y < 0 || rgb.Z < 0)
            throw std::invalid_argument("Negative compute emission.");
        return rgb.X > 0 || rgb.Y > 0 || rgb.Z > 0;
    }
    static bool Light(Float3 rgb, float intensity)
    {
        if (!std::isfinite(intensity) || intensity < 0)
            throw std::invalid_argument("Invalid compute light intensity.");
        bool color = Positive(rgb);
        double peak = double(std::max({rgb.X, rgb.Y, rgb.Z})) * intensity;
        if (peak > std::numeric_limits<float>::max())
            throw std::overflow_error("Compute light strength overflow.");
        return color && intensity > 0;
    }
    static void Direction(Float3 p)
    {
        CheckVector(p);
        if (p.X == 0 && p.Y == 0 && p.Z == 0)
            throw std::invalid_argument("Zero compute emission direction.");
    }
    void Append(EmitterKind kind, size_t index, uint32_t instance = InvalidEmitterIndex,
                uint64_t primitive = InvalidRayPrimitiveIndex)
    {
        if (index > std::numeric_limits<uint32_t>::max() || Result_.Sources.size() >= MaximumUniformEmitters)
            throw std::overflow_error("Compute emitter table exceeds selection capacity.");
        Result_.Sources.push_back({kind, uint32_t(index), instance, uint32_t(primitive)});
    }
    static Vector Position(const Matrix& m, Float3 p)
    {
        CheckVector(p);
        Vector result{};
        for (uint32_t row = 0; row < Dimensions; ++row)
        {
            uint32_t offset = MatrixColumns * row;
            result[row] = double(m[offset]) * p.X + double(m[offset + 1]) * p.Y +
                          double(m[offset + 2]) * p.Z + m[offset + 3];
        }
        return result;
    }
    void Include(Vector p)
    {
        GeometryPresent_ = true;
        for (uint32_t axis = 0; axis < Dimensions; ++axis)
        {
            Minimum_[axis] = std::min(Minimum_[axis], p[axis]);
            Maximum_[axis] = std::max(Maximum_[axis], p[axis]);
        }
    }
    bool Emissive(uint32_t material) const
    {
        if (material >= Scene_.Materials.size())
            throw std::invalid_argument("Compute emitter material is missing.");
        const Material& value = Scene_.Materials[material];
        const bool luminous = Positive(value.Emission);
        return luminous && !value.SampledEmitter;
    }
    void Object(const SceneGeometry& geometry, const Matrix& matrix, uint32_t instance)
    {
        uint64_t count = uint64_t(geometry.Triangles.size()) + geometry.Spheres.size();
        if (count > InvalidRayPrimitiveIndex - PrimitiveBase_)
            throw std::overflow_error("Compute emitter primitive ID overflow.");
        for (size_t index = 0; index < geometry.Triangles.size(); ++index)
        {
            const Triangle& triangle = geometry.Triangles[index];
            Vector a = Position(matrix, triangle.Position0), b = Position(matrix, triangle.Position1),
                   c = Position(matrix, triangle.Position2);
            Include(a);
            Include(b);
            Include(c);
            bool emits = Emissive(triangle.MaterialIndex);
            // Nonzero area is invariant under the validated invertible affine.
            // Object-space edges avoid subtracting translated large coordinates.
            a = {triangle.Position0.X, triangle.Position0.Y, triangle.Position0.Z};
            b = {triangle.Position1.X, triangle.Position1.Y, triangle.Position1.Z};
            c = {triangle.Position2.X, triangle.Position2.Y, triangle.Position2.Z};
            for (uint32_t axis = 0; axis < Dimensions; ++axis)
            {
                b[axis] -= a[axis];
                c[axis] -= a[axis];
            }
            bool area =
                b[1] * c[2] != b[2] * c[1] || b[2] * c[0] != b[0] * c[2] || b[0] * c[1] != b[1] * c[0];
            if (emits && area)
                Append(EmitterKind::Triangle, index, instance, PrimitiveBase_ + index);
        }
        for (size_t index = 0; index < geometry.Spheres.size(); ++index)
        {
            const Sphere& sphere = geometry.Spheres[index];
            if (!(sphere.Radius > 0) || !std::isfinite(sphere.Radius))
                throw std::invalid_argument("Invalid compute emissive sphere.");
            Vector lower = Position(matrix, sphere.Center), upper = lower;
            for (uint32_t axis = 0; axis < Dimensions; ++axis)
            {
                uint32_t row = axis * MatrixColumns;
                double extent = sphere.Radius * std::hypot(double(matrix[row]), double(matrix[row + 1]),
                                                           double(matrix[row + 2]));
                lower[axis] -= extent;
                upper[axis] += extent;
            }
            Include(lower);
            Include(upper);
            if (Emissive(sphere.MaterialIndex))
                Append(EmitterKind::Sphere, index, instance,
                       PrimitiveBase_ + geometry.Triangles.size() + index);
        }
        PrimitiveBase_ += count;
    }
    static void MatrixValid(const Matrix& m)
    {
        for (float x : m)
            if (!std::isfinite(x))
                throw std::invalid_argument("Nonfinite compute emitter transform.");
        double determinant = double(m[0]) * (double(m[5]) * m[10] - double(m[6]) * m[9]) +
                             double(m[1]) * (double(m[6]) * m[8] - double(m[4]) * m[10]) +
                             double(m[2]) * (double(m[4]) * m[9] - double(m[5]) * m[8]);
        if (determinant == 0)
            throw std::invalid_argument("Singular compute emitter transform.");
    }

  public:
    explicit Builder(const Scene& scene) : Scene_(scene) {}
    EmissionCatalog Build()
    {
        for (size_t i = 0; i < Scene_.DirectionalLights.size(); ++i)
        {
            const auto& l = Scene_.DirectionalLights[i];
            Direction(l.DirectionToLight);
            if (Light(l.Color, l.Intensity))
                Append(EmitterKind::Directional, i);
        }
        for (size_t i = 0; i < Scene_.PointLights.size(); ++i)
        {
            const auto& l = Scene_.PointLights[i];
            CheckVector(l.Position);
            if (Light(l.Color, l.Intensity))
                Append(EmitterKind::Point, i);
        }
        for (size_t i = 0; i < Scene_.RectAreaLights.size(); ++i)
        {
            const auto& l = Scene_.RectAreaLights[i];
            CheckVector(l.Position);
            Direction(l.Direction);
            if (!(l.Size.X > 0 && l.Size.Y > 0) || !std::isfinite(l.Size.X) || !std::isfinite(l.Size.Y))
                throw std::invalid_argument("Invalid compute rectangle extent.");
            const float area = l.Size.X * l.Size.Y;
            if (!(area > 0) || !std::isfinite(area) || !std::isfinite(1 / area))
                throw std::overflow_error("Compute rectangle area density exceeds float range.");
            if (Light(l.Color, l.Intensity))
                Append(EmitterKind::Rectangle, i);
        }
        bool top = Positive(Scene_.BackgroundTop), bottom = Positive(Scene_.BackgroundBottom);
        if (top || bottom)
            Append(EmitterKind::Environment, 0);
        const Matrix identity{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
        Object(Scene_, identity, InvalidEmitterIndex);
        for (size_t i = 0; i < Scene_.Instances.size(); ++i)
        {
            const auto& instance = Scene_.Instances[i];
            if (i >= InvalidEmitterIndex || instance.GeometryIndex >= Scene_.Geometries.size() ||
                !Scene_.Geometries[instance.GeometryIndex])
                throw std::invalid_argument("Compute emitter references invalid instance geometry.");
            MatrixValid(instance.ObjectToWorld);
            Object(*Scene_.Geometries[instance.GeometryIndex], instance.ObjectToWorld, uint32_t(i));
        }
        if (GeometryPresent_)
        {
            Vector rounded{}, extent{};
            for (uint32_t axis = 0; axis < Dimensions; ++axis)
            {
                rounded[axis] = float((Minimum_[axis] + Maximum_[axis]) / 2);
                extent[axis] = std::max(Maximum_[axis] - rounded[axis], rounded[axis] - Minimum_[axis]);
            }
            Result_.Center = {float(rounded[0]), float(rounded[1]), float(rounded[2])};
            Result_.Radius = std::nextafter(float(std::hypot(extent[0], extent[1], extent[2])), INFINITY);
            float area = Pi * Result_.Radius * Result_.Radius;
            if (!(area > 0) || !std::isfinite(area) || !std::isfinite(1 / area))
                throw std::overflow_error("Compute emission domain has unrepresentable area density.");
        }
        return std::move(Result_);
    }
};
} // namespace
EmissionCatalog BuildCatalog(const Scene& scene)
{
    return Builder(scene).Build();
}
} // namespace Arvo::RayTracing::ComputeEmission
