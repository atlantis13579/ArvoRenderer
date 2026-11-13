#include "Renderer/EmitterData.h"
#include <cmath>
#include <stdexcept>

namespace Arvo::RayTracing::MetalEmission
{
// This host conversion belongs to Metal. MSL consumes eight packed 16-byte
// lanes per source. Geometry uses material radiance and retains affine shape;
// analytic rectangles keep the demo's radiance-times-area intensity convention.
namespace
{
constexpr unsigned Dimension = 3, TransformStride = 4;
using Matrix = std::array<float, Dimension * TransformStride>;
const Matrix UnitTransform{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
float Pack(double v)
{
    if (!std::isfinite(v) || std::abs(v) > std::numeric_limits<float>::max())
        throw std::overflow_error("Metal emitter upload exceeds float range.");
    return float(v);
}
Float3 TransformValue(const Matrix& matrix, std::array<double, Dimension> value, bool translate)
{
    Float3 result;
    float* components[Dimension]{&result.X, &result.Y, &result.Z};
    for (unsigned row = 0; row < Dimension; ++row)
    {
        const unsigned base = row * TransformStride;
        double sum = translate ? matrix[base + Dimension] : 0;
        for (unsigned axis = 0; axis < Dimension; ++axis)
            sum += matrix[base + axis] * value[axis];
        *components[row] = Pack(sum);
    }
    return result;
}
Float3 Intensity(Float3 color, float scale)
{
    return {Pack(double(color.X) * scale), Pack(double(color.Y) * scale), Pack(double(color.Z) * scale)};
}
Float3 Unit(Float3 v)
{
    const double length = std::hypot(double(v.X), double(v.Y), double(v.Z));
    return {float(v.X / length), float(v.Y / length), float(v.Z / length)};
}
void PackShape(const Scene& scene, const EmitterReference& ref, PackedEmitter& destination)
{
    const SceneGeometry* geometry = &scene;
    const Matrix* matrix = &UnitTransform;
    if (ref.Instance != InvalidEmitterIndex)
    {
        const auto& instance = scene.Instances[ref.Instance];
        geometry = scene.Geometries[instance.GeometryIndex].get();
        matrix = &instance.ObjectToWorld;
    }
    uint32_t material;
    if (ref.Kind == EmitterKind::Triangle)
    {
        const auto& t = geometry->Triangles[ref.Index];
        material = t.MaterialIndex;
        destination.Origin = TransformValue(*matrix, {t.Position0.X, t.Position0.Y, t.Position0.Z}, true);
        destination.AxisX =
            TransformValue(*matrix,
                           {double(t.Position1.X) - t.Position0.X, double(t.Position1.Y) - t.Position0.Y,
                            double(t.Position1.Z) - t.Position0.Z},
                           false);
        destination.AxisY =
            TransformValue(*matrix,
                           {double(t.Position2.X) - t.Position0.X, double(t.Position2.Y) - t.Position0.Y,
                            double(t.Position2.Z) - t.Position0.Z},
                           false);
    }
    else
    {
        const auto& s = geometry->Spheres[ref.Index];
        material = s.MaterialIndex;
        destination.Origin = TransformValue(*matrix, {s.Center.X, s.Center.Y, s.Center.Z}, true);
        destination.AxisX = TransformValue(*matrix, {s.Radius, 0, 0}, false);
        destination.AxisY = TransformValue(*matrix, {0, s.Radius, 0}, false);
        destination.AxisZ = TransformValue(*matrix, {0, 0, s.Radius}, false);
    }
    destination.Strength = scene.Materials[material].Emission;
    destination.Flags = EmitterCastsShadows | EmitterTwoSided;
}
} // namespace
PreparedEmissionScene PrepareEmitters(const Scene& scene)
{
    const auto catalog = BuildCatalog(scene);
    PreparedEmissionScene upload{catalog.Center, catalog.Radius, {}};
    upload.Sources.reserve(catalog.Sources.size());
    for (const auto& ref : catalog.Sources)
    {
        PackedEmitter entry;
        entry.Kind = ref.Kind;
        entry.SourceIndex = ref.Index;
        entry.Instance = ref.Instance;
        entry.Primitive = ref.Primitive;
        if (ref.Kind == EmitterKind::Directional)
        {
            const auto& light = scene.DirectionalLights[ref.Index];
            entry.Direction = Unit(light.DirectionToLight);
            entry.Strength = Intensity(light.Color, light.Intensity);
            entry.Flags = light.CastShadows ? EmitterCastsShadows : 0;
        }
        else if (ref.Kind == EmitterKind::Point)
        {
            const auto& light = scene.PointLights[ref.Index];
            entry.Origin = light.Position;
            entry.Strength = Intensity(light.Color, light.Intensity);
            entry.Flags = light.CastShadows ? EmitterCastsShadows : 0;
        }
        else if (ref.Kind == EmitterKind::Rectangle)
        {
            const auto& light = scene.RectAreaLights[ref.Index];
            entry.Origin = light.Position;
            entry.Direction = Unit(light.Direction);
            entry.Strength = Intensity(light.Color, light.Intensity);
            entry.Width = light.Size.X;
            entry.Height = light.Size.Y;
            entry.Flags = light.CastShadows ? EmitterCastsShadows : 0;
        }
        else if (ref.Kind == EmitterKind::Environment)
        {
            entry.BackgroundTop = scene.BackgroundTop;
            entry.BackgroundBottom = scene.BackgroundBottom;
            entry.Flags = EmitterCastsShadows;
        }
        else
            PackShape(scene, ref, entry);
        upload.Sources.push_back(entry);
    }
    return upload;
}
} // namespace Arvo::RayTracing::MetalEmission
