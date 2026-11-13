#include "EmitterData.hlsl"
// DXR endpoint assembly: choose a catalog entry, then sample its conditional
// emitted ray. The root program supplies the three independent sampler leaves.
// Geometric Primitive is a traversal ID, never the emitter-list slot. A null
// ray remains a consumed draw, which preserves the discrete source PMF.
struct EmitterEndpoint
{
    EmissionSample Emission;
    uint EmitterIndex, Primitive;
    float SelectionProbability;
};
struct EmitterEndpointDensity
{
    float SelectionProbability, Position, Direction;
    uint Flags;
};
EmissionSource EndpointLight(PackedEmitter emitter)
{
    EmissionSource light;
    light.Kind = emitter.Kind;
    light.Position = float3(emitter.Origin);
    light.Direction = float3(emitter.Direction);
    light.Strength = float3(emitter.Strength);
    light.BackgroundTop = float3(emitter.BackgroundTop);
    light.BackgroundBottom = float3(emitter.BackgroundBottom);
    light.Size = float2(emitter.Width, emitter.Height);
    return light;
}
SurfaceEmissionSource EndpointGeometry(PackedEmitter emitter)
{
    SurfaceEmissionSource shape;
    shape.Kind = emitter.Kind == PackedEmitterTriangle ? SurfaceEmitterTriangle : SurfaceEmitterEllipsoid;
    shape.Origin = float3(emitter.Origin);
    shape.AxisX = float3(emitter.AxisX);
    shape.AxisY = float3(emitter.AxisY);
    shape.AxisZ = float3(emitter.AxisZ);
    shape.Radiance = float3(emitter.Strength);
    return shape;
}
EmitterEndpoint SampleEmitterEndpoint(StructuredBuffer<PackedEmitter> emitters, uint count,
                                      EmissionDomain domain, float sourceVariate, float2 pointVariates,
                                      float2 rayVariates)
{
    EmitterEndpoint result;
    result.Emission = EmptyLightEmission();
    result.EmitterIndex = InvalidEmitterSelection;
    result.Primitive = InvalidEmitterSelection;
    result.SelectionProbability = 0;
    EmitterSelection choice = SelectEmitter(count, sourceVariate);
    if (choice.Probability <= 0)
        return result;
    PackedEmitter emitter = emitters[choice.Index];
    result.EmitterIndex = choice.Index;
    result.Primitive = emitter.Primitive;
    result.SelectionProbability = choice.Probability;
    switch (emitter.Kind)
    {
    case PackedEmitterTriangle:
    case PackedEmitterSphere: {
        SurfaceEmissionSample ray =
            SampleSurfaceEmission(EndpointGeometry(emitter), pointVariates, rayVariates);
        result.Emission.Origin = ray.Origin;
        result.Emission.Direction = ray.Direction;
        result.Emission.Normal = ray.Normal;
        result.Emission.Value = ray.Value;
        result.Emission.Weight = ray.Weight;
        result.Emission.PositionPdf = ray.PositionPdf;
        result.Emission.DirectionPdf = ray.DirectionPdf;
        result.Emission.Flags = ray.Flags;
        break;
    }
    default:
        result.Emission = SampleLightEmission(EndpointLight(emitter), domain, pointVariates, rayVariates);
        break;
    }
    result.Emission.Weight = result.Emission.Weight / choice.Probability;
    return result;
}
// Position is on source support and both vectors are unit. Keep the discrete
// source PMF separate from continuous area/solid-angle densities for path MIS.
EmitterEndpointDensity EmitterEndpointPdf(StructuredBuffer<PackedEmitter> emitters, uint count,
                                          EmissionDomain domain, uint index, float3 normal, float3 direction)
{
    EmitterEndpointDensity result;
    result.SelectionProbability = EmitterProbability(count, index);
    result.Position = 0;
    result.Direction = 0;
    result.Flags = 0;
    if (result.SelectionProbability <= 0)
        return result;
    PackedEmitter emitter = emitters[index];
    float2 conditional;
    if (emitter.Kind == PackedEmitterTriangle || emitter.Kind == PackedEmitterSphere)
        conditional = SurfaceEmissionPdf(EndpointGeometry(emitter), normal, direction);
    else
        conditional = LightEmissionPdf(EndpointLight(emitter), domain, direction);
    result.Position = conditional.x;
    result.Direction = conditional.y;
    switch (emitter.Kind)
    {
    case PackedEmitterPoint:
        result.Flags = EmissionValid | EmissionDeltaPosition;
        break;
    case PackedEmitterDirectional:
        if (conditional.x > 0)
            result.Flags = EmissionValid | EmissionDeltaDirection | EmissionInfinite;
        break;
    case PackedEmitterEnvironment:
        if (conditional.x > 0)
            result.Flags = EmissionValid | EmissionInfinite;
        break;
    case PackedEmitterRectangle:
    case PackedEmitterTriangle:
    case PackedEmitterSphere:
        if (conditional.x > 0)
            result.Flags = EmissionValid;
        break;
    }
    return result;
}
