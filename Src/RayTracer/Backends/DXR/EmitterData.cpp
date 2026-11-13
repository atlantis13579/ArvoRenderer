#include "Renderer/EmitterData.h"
#include <cmath>
#include <stdexcept>

namespace Arvo::RayTracing::DxrEmission
{
// DXR source records use the same virtual primitive numbering as traversal.
// Hardware instance transforms do not transform an emitted endpoint for us:
// prepare world-space edge vectors and ellipsoid columns on this host path.
namespace
{
constexpr unsigned SpatialDimensions = 3, AffineRowSize = 4;
using Affine = std::array<float, SpatialDimensions * AffineRowSize>;
const Affine Identity{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
float FiniteFloat(double number)
{
    if (!std::isfinite(number) || std::abs(number) > std::numeric_limits<float>::max())
        throw std::overflow_error("DXR source data cannot be represented in float.");
    return float(number);
}
Float3 World(const Affine& transform, double x, double y, double z, bool point)
{
    auto component = [&](unsigned row) {
        unsigned i = row * AffineRowSize;
        double value = transform[i] * x + transform[i + 1] * y + transform[i + 2] * z;
        return FiniteFloat(value + (point ? transform[i + SpatialDimensions] : 0));
    };
    return {component(0), component(1), component(2)};
}
Float3 NormalizeSource(Float3 direction)
{
    double norm = std::hypot(double(direction.X), double(direction.Y), double(direction.Z));
    return {float(direction.X / norm), float(direction.Y / norm), float(direction.Z / norm)};
}
Float3 Energy(Float3 color, float intensity)
{
    return {FiniteFloat(double(color.X) * intensity), FiniteFloat(double(color.Y) * intensity),
            FiniteFloat(double(color.Z) * intensity)};
}
} // namespace
PreparedEmissionScene PrepareEmitters(const Scene& scene)
{
    auto catalog = BuildCatalog(scene);
    PreparedEmissionScene prepared{catalog.Center, catalog.Radius, {}};
    prepared.Sources.reserve(catalog.Sources.size());
    for (const auto& entry : catalog.Sources)
    {
        PackedEmitter source;
        source.Kind = entry.Kind;
        source.Primitive = entry.Primitive;
        source.SourceIndex = entry.Index;
        source.Instance = entry.Instance;
        switch (entry.Kind)
        {
        case EmitterKind::Directional: {
            const auto& light = scene.DirectionalLights[entry.Index];
            source.Direction = NormalizeSource(light.DirectionToLight);
            source.Strength = Energy(light.Color, light.Intensity);
            source.Flags = light.CastShadows ? EmitterCastsShadows : 0;
            break;
        }
        case EmitterKind::Point: {
            const auto& light = scene.PointLights[entry.Index];
            source.Origin = light.Position;
            source.Strength = Energy(light.Color, light.Intensity);
            source.Flags = light.CastShadows ? EmitterCastsShadows : 0;
            break;
        }
        case EmitterKind::Rectangle: {
            const auto& light = scene.RectAreaLights[entry.Index];
            source.Origin = light.Position;
            source.Direction = NormalizeSource(light.Direction);
            source.Width = light.Size.X;
            source.Height = light.Size.Y;
            source.Strength = Energy(light.Color, light.Intensity);
            source.Flags = light.CastShadows ? EmitterCastsShadows : 0;
            break;
        }
        case EmitterKind::Environment:
            source.BackgroundTop = scene.BackgroundTop;
            source.BackgroundBottom = scene.BackgroundBottom;
            source.Flags = EmitterCastsShadows;
            break;
        case EmitterKind::Triangle:
        case EmitterKind::Sphere: {
            const SceneGeometry* mesh = &scene;
            const Affine* transform = &Identity;
            if (entry.Instance != InvalidEmitterIndex)
            {
                const auto& instance = scene.Instances[entry.Instance];
                mesh = scene.Geometries[instance.GeometryIndex].get();
                transform = &instance.ObjectToWorld;
            }
            uint32_t material;
            if (entry.Kind == EmitterKind::Triangle)
            {
                const auto& triangle = mesh->Triangles[entry.Index];
                material = triangle.MaterialIndex;
                const auto a = triangle.Position0, b = triangle.Position1, c = triangle.Position2;
                source.Origin = World(*transform, a.X, a.Y, a.Z, true);
                source.AxisX =
                    World(*transform, double(b.X) - a.X, double(b.Y) - a.Y, double(b.Z) - a.Z, false);
                source.AxisY =
                    World(*transform, double(c.X) - a.X, double(c.Y) - a.Y, double(c.Z) - a.Z, false);
            }
            else
            {
                const auto& sphere = mesh->Spheres[entry.Index];
                material = sphere.MaterialIndex;
                source.Origin = World(*transform, sphere.Center.X, sphere.Center.Y, sphere.Center.Z, true);
                source.AxisX = World(*transform, sphere.Radius, 0, 0, false);
                source.AxisY = World(*transform, 0, sphere.Radius, 0, false);
                source.AxisZ = World(*transform, 0, 0, sphere.Radius, false);
            }
            source.Strength = scene.Materials[material].Emission;
            source.Flags = EmitterCastsShadows | EmitterTwoSided;
            break;
        }
        }
        prepared.Sources.push_back(source);
    }
    return prepared;
}
} // namespace Arvo::RayTracing::DxrEmission
