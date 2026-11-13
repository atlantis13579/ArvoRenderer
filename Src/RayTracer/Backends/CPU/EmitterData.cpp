#include "Renderer/EmitterData.h"
#include <cmath>
#include <stdexcept>

namespace Arvo::RayTracing::Cpu
{
// Scene preparation preserves analytical shapes. Transform triangle edge
// vectors before adding translation, so distant instances do not lose area by
// subtracting two rounded world vertices. Affine spheres become c+B*n, where
// B contains the transformed radius vectors; no tessellation or new asset.
namespace
{
constexpr uint32_t Dimensions = 3, MatrixStride = 4;
using Matrix = std::array<float, Dimensions * MatrixStride>;
const Matrix Identity{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
float Representable(double value)
{
    if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<float>::max())
        throw std::overflow_error("CPU emitter data exceeds finite float coordinates.");
    return float(value);
}
Float3 Transform(const Matrix& m, const std::array<double, Dimensions>& p, bool point)
{
    std::array<float, Dimensions> result{};
    for (uint32_t row = 0; row < Dimensions; ++row)
    {
        uint32_t index = row * MatrixStride;
        double value = point ? m[index + Dimensions] : 0;
        for (uint32_t column = 0; column < Dimensions; ++column)
            value += double(m[index + column]) * p[column];
        result[row] = Representable(value);
    }
    return {result[0], result[1], result[2]};
}
Float3 Unit(Float3 value)
{
    double length = std::hypot(double(value.X), double(value.Y), double(value.Z));
    return {float(value.X / length), float(value.Y / length), float(value.Z / length)};
}
Float3 Strength(Float3 color, float intensity)
{
    return {Representable(double(color.X) * intensity), Representable(double(color.Y) * intensity),
            Representable(double(color.Z) * intensity)};
}
} // namespace
PreparedEmissionScene PrepareEmitters(const Scene& scene)
{
    const EmissionCatalog catalog = BuildEmissionCatalog(scene);
    PreparedEmissionScene prepared{catalog.Center, catalog.Radius, {}};
    prepared.Sources.reserve(catalog.Sources.size());
    for (const EmitterReference& reference : catalog.Sources)
    {
        PackedEmitter emitter;
        emitter.Kind = reference.Kind;
        emitter.Primitive = reference.Primitive;
        emitter.SourceIndex = reference.Index;
        emitter.Instance = reference.Instance;
        switch (reference.Kind)
        {
        case EmitterKind::Directional: {
            const auto& light = scene.DirectionalLights[reference.Index];
            emitter.Direction = Unit(light.DirectionToLight);
            emitter.Strength = Strength(light.Color, light.Intensity);
            emitter.Flags = light.CastShadows ? EmitterCastsShadows : 0;
            break;
        }
        case EmitterKind::Point: {
            const auto& light = scene.PointLights[reference.Index];
            emitter.Origin = light.Position;
            emitter.Strength = Strength(light.Color, light.Intensity);
            emitter.Flags = light.CastShadows ? EmitterCastsShadows : 0;
            break;
        }
        case EmitterKind::Rectangle: {
            const auto& light = scene.RectAreaLights[reference.Index];
            emitter.Origin = light.Position;
            emitter.Direction = Unit(light.Direction);
            emitter.Strength = Strength(light.Color, light.Intensity);
            emitter.Width = light.Size.X;
            emitter.Height = light.Size.Y;
            emitter.Flags = light.CastShadows ? EmitterCastsShadows : 0;
            break;
        }
        case EmitterKind::Environment:
            emitter.BackgroundTop = scene.BackgroundTop;
            emitter.BackgroundBottom = scene.BackgroundBottom;
            emitter.Flags = EmitterCastsShadows;
            break;
        case EmitterKind::Triangle:
        case EmitterKind::Sphere: {
            const SceneGeometry* geometry = &scene;
            const Matrix* matrix = &Identity;
            if (reference.Instance != InvalidEmitterIndex)
            {
                const auto& instance = scene.Instances[reference.Instance];
                geometry = scene.Geometries[instance.GeometryIndex].get();
                matrix = &instance.ObjectToWorld;
            }
            uint32_t material;
            if (reference.Kind == EmitterKind::Triangle)
            {
                const auto& triangle = geometry->Triangles[reference.Index];
                material = triangle.MaterialIndex;
                emitter.Origin = Transform(
                    *matrix, {triangle.Position0.X, triangle.Position0.Y, triangle.Position0.Z}, true);
                emitter.AxisX = Transform(*matrix,
                                          {double(triangle.Position1.X) - triangle.Position0.X,
                                           double(triangle.Position1.Y) - triangle.Position0.Y,
                                           double(triangle.Position1.Z) - triangle.Position0.Z},
                                          false);
                emitter.AxisY = Transform(*matrix,
                                          {double(triangle.Position2.X) - triangle.Position0.X,
                                           double(triangle.Position2.Y) - triangle.Position0.Y,
                                           double(triangle.Position2.Z) - triangle.Position0.Z},
                                          false);
            }
            else
            {
                const auto& sphere = geometry->Spheres[reference.Index];
                material = sphere.MaterialIndex;
                emitter.Origin =
                    Transform(*matrix, {sphere.Center.X, sphere.Center.Y, sphere.Center.Z}, true);
                emitter.AxisX = Transform(*matrix, {sphere.Radius, 0, 0}, false);
                emitter.AxisY = Transform(*matrix, {0, sphere.Radius, 0}, false);
                emitter.AxisZ = Transform(*matrix, {0, 0, sphere.Radius}, false);
            }
            emitter.Strength = scene.Materials[material].Emission;
            emitter.Flags = EmitterCastsShadows | EmitterTwoSided;
            break;
        }
        }
        prepared.Sources.push_back(emitter);
    }
    return prepared;
}
} // namespace Arvo::RayTracing::Cpu
