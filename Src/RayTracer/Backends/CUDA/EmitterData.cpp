#include "Renderer/EmitterData.h"
#include <cmath>
#include <stdexcept>

namespace Arvo::RayTracing::CudaEmissionHost
{
// CUDA emitter upload preparation, independent of other backend packers. The
// catalog's virtual primitive IDs are metadata, not flattened geometry offsets.
// Keep geometric emission as material radiance; rectangle Strength uses the
// demo's separate Color*Intensity convention and is divided by area in CUDA.
namespace
{
constexpr unsigned Dimensions = 3, AffineWidth = 4;
using Transform = std::array<float, Dimensions * AffineWidth>;
const Transform Identity{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
float UploadFloat(double value)
{
    if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<float>::max())
        throw std::overflow_error("CUDA emitter upload has an unrepresentable coordinate.");
    return float(value);
}
Float3 Apply(const Transform& m, std::array<double, Dimensions> p, bool origin)
{
    std::array<float, Dimensions> packed{};
    for (unsigned row = 0; row < Dimensions; ++row)
    {
        unsigned offset = row * AffineWidth;
        double value = origin ? m[offset + Dimensions] : 0;
        for (unsigned column = 0; column < Dimensions; ++column)
            value += m[offset + column] * p[column];
        packed[row] = UploadFloat(value);
    }
    return {packed[0], packed[1], packed[2]};
}
Float3 RadiometricStrength(Float3 color, float value)
{
    return {UploadFloat(double(color.X) * value), UploadFloat(double(color.Y) * value),
            UploadFloat(double(color.Z) * value)};
}
Float3 Direction(Float3 v)
{
    double length = std::hypot(double(v.X), double(v.Y), double(v.Z));
    return {float(v.X / length), float(v.Y / length), float(v.Z / length)};
}
} // namespace
PreparedEmissionScene PrepareEmitters(const Scene& scene)
{
    const EmissionCatalog catalog = BuildCatalog(scene);
    PreparedEmissionScene result{catalog.Center, catalog.Radius, {}};
    result.Sources.reserve(catalog.Sources.size());
    for (const auto& ref : catalog.Sources)
    {
        PackedEmitter record;
        record.Kind = ref.Kind;
        record.SourceIndex = ref.Index;
        record.Instance = ref.Instance;
        record.Primitive = ref.Primitive;
        if (ref.Kind == EmitterKind::Directional)
        {
            const auto& light = scene.DirectionalLights[ref.Index];
            record.Direction = Direction(light.DirectionToLight);
            record.Strength = RadiometricStrength(light.Color, light.Intensity);
            record.Flags = light.CastShadows ? EmitterCastsShadows : 0;
        }
        else if (ref.Kind == EmitterKind::Point)
        {
            const auto& light = scene.PointLights[ref.Index];
            record.Origin = light.Position;
            record.Strength = RadiometricStrength(light.Color, light.Intensity);
            record.Flags = light.CastShadows ? EmitterCastsShadows : 0;
        }
        else if (ref.Kind == EmitterKind::Rectangle)
        {
            const auto& light = scene.RectAreaLights[ref.Index];
            record.Origin = light.Position;
            record.Direction = Direction(light.Direction);
            record.Strength = RadiometricStrength(light.Color, light.Intensity);
            record.Width = light.Size.X;
            record.Height = light.Size.Y;
            record.Flags = light.CastShadows ? EmitterCastsShadows : 0;
        }
        else if (ref.Kind == EmitterKind::Environment)
        {
            record.BackgroundTop = scene.BackgroundTop;
            record.BackgroundBottom = scene.BackgroundBottom;
            record.Flags = EmitterCastsShadows;
        }
        else
        {
            const Transform* transform = &Identity;
            const SceneGeometry* geometry = &scene;
            if (ref.Instance != InvalidEmitterIndex)
            {
                const auto& instance = scene.Instances[ref.Instance];
                transform = &instance.ObjectToWorld;
                geometry = scene.Geometries[instance.GeometryIndex].get();
            }
            uint32_t material;
            if (ref.Kind == EmitterKind::Triangle)
            {
                const auto& t = geometry->Triangles[ref.Index];
                material = t.MaterialIndex;
                record.Origin = Apply(*transform, {t.Position0.X, t.Position0.Y, t.Position0.Z}, true);
                record.AxisX =
                    Apply(*transform,
                          {double(t.Position1.X) - t.Position0.X, double(t.Position1.Y) - t.Position0.Y,
                           double(t.Position1.Z) - t.Position0.Z},
                          false);
                record.AxisY =
                    Apply(*transform,
                          {double(t.Position2.X) - t.Position0.X, double(t.Position2.Y) - t.Position0.Y,
                           double(t.Position2.Z) - t.Position0.Z},
                          false);
            }
            else
            {
                const auto& sphere = geometry->Spheres[ref.Index];
                material = sphere.MaterialIndex;
                record.Origin = Apply(*transform, {sphere.Center.X, sphere.Center.Y, sphere.Center.Z}, true);
                record.AxisX = Apply(*transform, {sphere.Radius, 0, 0}, false);
                record.AxisY = Apply(*transform, {0, sphere.Radius, 0}, false);
                record.AxisZ = Apply(*transform, {0, 0, sphere.Radius}, false);
            }
            record.Strength = scene.Materials[material].Emission;
            record.Flags = EmitterCastsShadows | EmitterTwoSided;
        }
        result.Sources.push_back(record);
    }
    return result;
}
} // namespace Arvo::RayTracing::CudaEmissionHost
