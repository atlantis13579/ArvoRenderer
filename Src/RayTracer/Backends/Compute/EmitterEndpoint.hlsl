#include "EmitterData.hlsl"
// The root program supplies this backend's LightSelection, LightEmission and
// SurfaceEmission leaves. Source PMF is discrete; conditional PDFs keep their
// own measures. Only Weight is divided by source PMF, exactly once. A null
// conditional sample must not trigger another source draw.
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
EmissionSource EndpointAnalyticSource(PackedEmitter record)
{
    EmissionSource source;
    source.Kind = record.Kind;
    source.Position = float3(record.Origin);
    source.Direction = float3(record.Direction);
    source.Strength = float3(record.Strength);
    source.BackgroundTop = float3(record.BackgroundTop);
    source.BackgroundBottom = float3(record.BackgroundBottom);
    source.Size = float2(record.Width, record.Height);
    return source;
}
SurfaceEmissionSource EndpointSurfaceSource(PackedEmitter record)
{
    SurfaceEmissionSource source;
    source.Kind = record.Kind == PackedEmitterTriangle ? SurfaceEmitterTriangle : SurfaceEmitterEllipsoid;
    source.Origin = float3(record.Origin);
    source.AxisX = float3(record.AxisX);
    source.AxisY = float3(record.AxisY);
    source.AxisZ = float3(record.AxisZ);
    source.Radiance = float3(record.Strength);
    return source;
}
EmitterEndpoint SampleEmitterEndpoint(StructuredBuffer<PackedEmitter> sources, uint count,
                                      EmissionDomain domain, float sourceU, float2 positionU,
                                      float2 directionU)
{
    EmitterEndpoint endpoint;
    endpoint.Emission = EmptyLightEmission();
    endpoint.EmitterIndex = InvalidSelectedEmitter;
    endpoint.Primitive = InvalidSelectedEmitter;
    endpoint.SelectionProbability = 0;
    EmitterSelection selection = SelectEmitter(count, sourceU);
    if (!(selection.Probability > 0))
        return endpoint;
    PackedEmitter record = sources[selection.Index];
    endpoint.EmitterIndex = selection.Index;
    endpoint.Primitive = record.Primitive;
    endpoint.SelectionProbability = selection.Probability;
    if (record.Kind == PackedEmitterTriangle || record.Kind == PackedEmitterSphere)
    {
        SurfaceEmissionSample sample =
            SampleSurfaceEmission(EndpointSurfaceSource(record), positionU, directionU);
        endpoint.Emission.Origin = sample.Origin;
        endpoint.Emission.Direction = sample.Direction;
        endpoint.Emission.Normal = sample.Normal;
        endpoint.Emission.Value = sample.Value;
        endpoint.Emission.Weight = sample.Weight;
        endpoint.Emission.PositionPdf = sample.PositionPdf;
        endpoint.Emission.DirectionPdf = sample.DirectionPdf;
        endpoint.Emission.Flags = sample.Flags;
    }
    else
        endpoint.Emission =
            SampleLightEmission(EndpointAnalyticSource(record), domain, positionU, directionU);
    endpoint.Emission.Weight = endpoint.Emission.Weight / selection.Probability;
    return endpoint;
}
// Evaluate on known source support, using a unit geometric normal/direction.
// Delta dimensions have zero continuous density, even though sample mass is 1.
EmitterEndpointDensity EmitterEndpointPdf(StructuredBuffer<PackedEmitter> sources, uint count,
                                          EmissionDomain domain, uint index, float3 geometricNormal,
                                          float3 direction)
{
    EmitterEndpointDensity result;
    result.SelectionProbability = EmitterProbability(count, index);
    result.Position = 0;
    result.Direction = 0;
    result.Flags = 0;
    if (!(result.SelectionProbability > 0))
        return result;
    PackedEmitter record = sources[index];
    float2 density = record.Kind == PackedEmitterTriangle || record.Kind == PackedEmitterSphere
                         ? SurfaceEmissionPdf(EndpointSurfaceSource(record), geometricNormal, direction)
                         : LightEmissionPdf(EndpointAnalyticSource(record), domain, direction);
    result.Position = density.x;
    result.Direction = density.y;
    if (record.Kind == PackedEmitterPoint)
        result.Flags = EmissionValid | EmissionDeltaPosition;
    else if (density.x > 0)
    {
        result.Flags = EmissionValid;
        if (record.Kind == PackedEmitterDirectional)
            result.Flags |= EmissionDeltaDirection | EmissionInfinite;
        if (record.Kind == PackedEmitterEnvironment)
            result.Flags |= EmissionInfinite;
    }
    return result;
}
