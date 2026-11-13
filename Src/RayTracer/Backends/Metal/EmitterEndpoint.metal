#include "EmitterData.metal"
// Metal endpoint assembly, independent of the other backend implementations.
// Root kernels supply the selection and conditional emission leaves. Packed
// upload vectors are expanded explicitly; the chosen slot is not a primitive ID.
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
EmissionSource EndpointAnalytic(PackedEmitter entry)
{
    EmissionSource source;
    source.Kind = entry.Kind;
    source.Position = float3(entry.Origin);
    source.Direction = float3(entry.Direction);
    source.Strength = float3(entry.Strength);
    source.BackgroundTop = float3(entry.BackgroundTop);
    source.BackgroundBottom = float3(entry.BackgroundBottom);
    source.Size = float2(entry.Width, entry.Height);
    return source;
}
SurfaceEmissionSource EndpointSurface(PackedEmitter entry)
{
    SurfaceEmissionSource source;
    source.Kind = entry.Kind == PackedEmitterTriangle ? SurfaceEmitterTriangle : SurfaceEmitterEllipsoid;
    source.Origin = float3(entry.Origin);
    source.AxisX = float3(entry.AxisX);
    source.AxisY = float3(entry.AxisY);
    source.AxisZ = float3(entry.AxisZ);
    source.Radiance = float3(entry.Strength);
    return source;
}
EmitterEndpoint SampleEmitterEndpoint(device const PackedEmitter* entries, uint count, EmissionDomain domain,
                                      float sourceU, float2 positionU, float2 directionU)
{
    EmitterEndpoint endpoint{};
    endpoint.EmitterIndex = InvalidSourceSelection;
    endpoint.Primitive = InvalidSourceSelection;
    EmitterSelection selection = SelectEmitter(count, sourceU);
    if (selection.Probability <= 0)
        return endpoint;
    PackedEmitter source = entries[selection.Index];
    endpoint.EmitterIndex = selection.Index;
    endpoint.Primitive = source.Primitive;
    endpoint.SelectionProbability = selection.Probability;
    if (source.Kind == PackedEmitterTriangle || source.Kind == PackedEmitterSphere)
    {
        SurfaceEmissionSample surface = SampleSurfaceEmission(EndpointSurface(source), positionU, directionU);
        endpoint.Emission.Origin = surface.Origin;
        endpoint.Emission.Direction = surface.Direction;
        endpoint.Emission.Normal = surface.Normal;
        endpoint.Emission.Value = surface.Value;
        endpoint.Emission.Weight = surface.Weight;
        endpoint.Emission.PositionPdf = surface.PositionPdf;
        endpoint.Emission.DirectionPdf = surface.DirectionPdf;
        endpoint.Emission.Flags = surface.Flags;
    }
    else
        endpoint.Emission = SampleLightEmission(EndpointAnalytic(source), domain, positionU, directionU);
    // Apply the discrete choice once; keep conditional densities separate for MIS.
    // Null draws are returned without retrying another catalog entry.
    endpoint.Emission.Weight /= selection.Probability;
    return endpoint;
}
// Endpoint position is on support; normal and direction are unit world vectors.
EmitterEndpointDensity EmitterEndpointPdf(device const PackedEmitter* entries, uint count,
                                          EmissionDomain domain, uint index, float3 normal, float3 direction)
{
    EmitterEndpointDensity result{};
    result.SelectionProbability = EmitterProbability(count, index);
    if (result.SelectionProbability <= 0)
        return result;
    PackedEmitter source = entries[index];
    float2 pdf;
    if (source.Kind == PackedEmitterTriangle || source.Kind == PackedEmitterSphere)
        pdf = SurfaceEmissionPdf(EndpointSurface(source), normal, direction);
    else
        pdf = LightEmissionPdf(EndpointAnalytic(source), domain, direction);
    result.Position = pdf.x;
    result.Direction = pdf.y;
    if (source.Kind == PackedEmitterPoint)
        result.Flags = EmissionValid | EmissionDeltaPosition;
    else if (pdf.x > 0)
    {
        result.Flags = EmissionValid;
        if (source.Kind == PackedEmitterDirectional)
            result.Flags |= EmissionInfinite | EmissionDeltaDirection;
        if (source.Kind == PackedEmitterEnvironment)
            result.Flags |= EmissionInfinite;
    }
    return result;
}
