#include "Renderer/EmitterData.h"
#include <cmath>
#include <stdexcept>

namespace Arvo::RayTracing::ComputeEmission
{
// Compute owns this scene conversion. Triangle edges are transformed as vectors;
// an instance translation never participates in their subtraction. Sphere axes
// retain shear, reflection and radius for the native affine emission sampler.
namespace
{
constexpr unsigned AxisCount = 3, RowStride = 4;
using Matrix = std::array<float, AxisCount * RowStride>;
const Matrix Identity{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
float Checked(double x)
{
    if (!std::isfinite(x) || std::abs(x) > std::numeric_limits<float>::max())
        throw std::overflow_error("Compute emitter coordinates exceed float upload range.");
    return float(x);
}
Float3 Map(const Matrix& matrix, std::array<double, AxisCount> value, bool translate)
{
    std::array<float, AxisCount> result{};
    for (unsigned axis = 0; axis < AxisCount; ++axis)
    {
        unsigned row = axis * RowStride;
        double sum = double(matrix[row]) * value[0] + double(matrix[row + 1]) * value[1] +
                     double(matrix[row + 2]) * value[2];
        if (translate)
            sum += matrix[row + AxisCount];
        result[axis] = Checked(sum);
    }
    return {result[0], result[1], result[2]};
}
Float3 EmissionValue(Float3 color, float intensity)
{
    return {Checked(double(color.X) * intensity), Checked(double(color.Y) * intensity),
            Checked(double(color.Z) * intensity)};
}
Float3 UnitDirection(Float3 direction)
{
    double length = std::hypot(double(direction.X), double(direction.Y), double(direction.Z));
    return {float(direction.X / length), float(direction.Y / length), float(direction.Z / length)};
}
void Geometry(const Scene& scene, const EmitterReference& reference, PackedEmitter& output)
{
    const SceneGeometry* object = &scene;
    const Matrix* transform = &Identity;
    if (reference.Instance != InvalidEmitterIndex)
    {
        const auto& instance = scene.Instances[reference.Instance];
        transform = &instance.ObjectToWorld;
        object = scene.Geometries[instance.GeometryIndex].get();
    }
    uint32_t material = 0;
    if (reference.Kind == EmitterKind::Triangle)
    {
        const auto& triangle = object->Triangles[reference.Index];
        material = triangle.MaterialIndex;
        output.Origin =
            Map(*transform, {triangle.Position0.X, triangle.Position0.Y, triangle.Position0.Z}, true);
        output.AxisX = Map(*transform,
                           {double(triangle.Position1.X) - triangle.Position0.X,
                            double(triangle.Position1.Y) - triangle.Position0.Y,
                            double(triangle.Position1.Z) - triangle.Position0.Z},
                           false);
        output.AxisY = Map(*transform,
                           {double(triangle.Position2.X) - triangle.Position0.X,
                            double(triangle.Position2.Y) - triangle.Position0.Y,
                            double(triangle.Position2.Z) - triangle.Position0.Z},
                           false);
    }
    else
    {
        const auto& sphere = object->Spheres[reference.Index];
        material = sphere.MaterialIndex;
        output.Origin = Map(*transform, {sphere.Center.X, sphere.Center.Y, sphere.Center.Z}, true);
        output.AxisX = Map(*transform, {sphere.Radius, 0, 0}, false);
        output.AxisY = Map(*transform, {0, sphere.Radius, 0}, false);
        output.AxisZ = Map(*transform, {0, 0, sphere.Radius}, false);
    }
    output.Strength = scene.Materials[material].Emission;
    output.Flags = EmitterCastsShadows | EmitterTwoSided;
}
} // namespace
PreparedEmissionScene PrepareEmitters(const Scene& scene)
{
    const auto catalog = BuildCatalog(scene);
    PreparedEmissionScene output{catalog.Center, catalog.Radius, {}};
    output.Sources.reserve(catalog.Sources.size());
    for (const auto& reference : catalog.Sources)
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
            emitter.Direction = UnitDirection(light.DirectionToLight);
            emitter.Strength = EmissionValue(light.Color, light.Intensity);
            emitter.Flags = light.CastShadows ? EmitterCastsShadows : 0;
            break;
        }
        case EmitterKind::Point: {
            const auto& light = scene.PointLights[reference.Index];
            emitter.Origin = light.Position;
            emitter.Strength = EmissionValue(light.Color, light.Intensity);
            emitter.Flags = light.CastShadows ? EmitterCastsShadows : 0;
            break;
        }
        case EmitterKind::Rectangle: {
            const auto& light = scene.RectAreaLights[reference.Index];
            emitter.Origin = light.Position;
            emitter.Direction = UnitDirection(light.Direction);
            emitter.Strength = EmissionValue(light.Color, light.Intensity);
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
        case EmitterKind::Sphere:
            Geometry(scene, reference, emitter);
            break;
        }
        output.Sources.push_back(emitter);
    }
    return output;
}
} // namespace Arvo::RayTracing::ComputeEmission
