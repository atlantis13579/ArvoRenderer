#include "RayTracer/Backends/CPU/EmitterEndpoint.h"
#include "RayTracer/Backends/CPU/SurfaceEmission.h"

namespace Arvo::RayTracing::Cpu
{
// The endpoint combines discrete source selection with this backend's analytic
// or geometric emitted-ray distribution. A null sample consumes its probability;
// never retry another source, which would change the catalog's discrete PMF.
namespace
{
EmissionSource Analytic(const PackedEmitter& value)
{
    EmissionSource source;
    source.Kind = static_cast<EmissionKind>(value.Kind);
    source.Position = value.Origin;
    source.Direction = value.Direction;
    source.Strength = value.Strength;
    source.BackgroundTop = value.BackgroundTop;
    source.BackgroundBottom = value.BackgroundBottom;
    source.Size = {value.Width, value.Height};
    return source;
}
SurfaceEmitter Surface(const PackedEmitter& value)
{
    return {value.Kind == EmitterKind::Triangle ? SurfaceEmitterKind::Triangle
                                                : SurfaceEmitterKind::Ellipsoid,
            value.Origin,
            value.AxisX,
            value.AxisY,
            value.AxisZ,
            value.Strength};
}
static_assert(uint32_t(EmitterKind::Environment) == uint32_t(EmissionKind::Environment));
} // namespace
EmitterEndpoint SampleEmitterEndpoint(const PreparedEmissionScene& scene, float sourceVariate,
                                      Float2 positionVariates, Float2 directionVariates)
{
    EmitterEndpoint endpoint;
    if (scene.Sources.size() > MaximumUniformEmitters)
        return endpoint;
    const auto selection = SelectEmitter(uint32_t(scene.Sources.size()), sourceVariate);
    if (!(selection.Probability > 0))
        return endpoint;
    const auto& source = scene.Sources[selection.Index];
    endpoint.EmitterIndex = selection.Index;
    endpoint.Primitive = source.Primitive;
    endpoint.SelectionProbability = selection.Probability;
    if (source.Kind == EmitterKind::Triangle || source.Kind == EmitterKind::Sphere)
        endpoint.Emission = SampleSurfaceEmission(Surface(source), positionVariates, directionVariates);
    else
        endpoint.Emission = SampleLightEmission(Analytic(source), {scene.Center, scene.Radius},
                                                positionVariates, directionVariates);
    endpoint.Emission.Weight = endpoint.Emission.Weight / selection.Probability;
    return endpoint;
}
EmitterEndpointDensity EmitterEndpointPdf(const PreparedEmissionScene& scene, uint32_t index,
                                          Float3 geometricNormal, Float3 direction)
{
    EmitterEndpointDensity density;
    if (scene.Sources.size() > MaximumUniformEmitters)
        return density;
    density.SelectionProbability = EmitterProbability(uint32_t(scene.Sources.size()), index);
    if (!(density.SelectionProbability > 0))
        return density;
    const auto& source = scene.Sources[index];
    EmissionDensity conditional;
    if (source.Kind == EmitterKind::Triangle || source.Kind == EmitterKind::Sphere)
        conditional = SurfaceEmissionPdf(Surface(source), geometricNormal, direction);
    else
        conditional = LightEmissionPdf(Analytic(source), {scene.Center, scene.Radius}, direction);
    density.Position = conditional.Position;
    density.Direction = conditional.Direction;
    switch (source.Kind)
    {
    case EmitterKind::Point:
        density.Flags = EmissionValid | EmissionDeltaPosition;
        break;
    case EmitterKind::Directional:
        density.Flags =
            conditional.Position > 0 ? EmissionValid | EmissionDeltaDirection | EmissionInfinite : 0;
        break;
    case EmitterKind::Environment:
        density.Flags = conditional.Position > 0 ? EmissionValid | EmissionInfinite : 0;
        break;
    case EmitterKind::Rectangle:
    case EmitterKind::Triangle:
    case EmitterKind::Sphere:
        density.Flags = conditional.Position > 0 ? EmissionValid : 0;
        break;
    }
    return density;
}
} // namespace Arvo::RayTracing::Cpu
