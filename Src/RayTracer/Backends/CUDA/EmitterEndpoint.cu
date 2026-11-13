#include "EmitterData.cuh"
namespace ArvoCudaEndpoint
{
// CUDA root translation units supply LightSelection, LightEmission and
// SurfaceEmission separately. The inverse source PMF belongs in Weight once;
// sample densities remain conditional and delta masses remain identified.
struct Endpoint
{
    ArvoCudaEmission::Sample Emission;
    unsigned EmitterIndex, Primitive;
    float SelectionProbability;
};
struct Density
{
    float SelectionProbability, Position, Direction;
    unsigned Flags;
};
__device__ ArvoCudaEmission::Source Analytic(const PackedEmitter& entry)
{
    return {entry.Kind,
            entry.Origin,
            entry.Direction,
            entry.Strength,
            entry.BackgroundTop,
            entry.BackgroundBottom,
            make_float2(entry.Width, entry.Height)};
}
__device__ ArvoCudaSurfaceEmission::Source Geometry(const PackedEmitter& entry)
{
    unsigned shape =
        entry.Kind == Triangle ? ArvoCudaSurfaceEmission::Triangle : ArvoCudaSurfaceEmission::Ellipsoid;
    return {shape, entry.Origin, entry.AxisX, entry.AxisY, entry.AxisZ, entry.Strength};
}
__device__ Endpoint Emit(const PackedEmitter* entries, unsigned count, ArvoCudaEmission::Domain domain,
                         float sourceU, float2 positionU, float2 directionU)
{
    Endpoint endpoint{};
    endpoint.EmitterIndex = ArvoCudaSelection::InvalidSource;
    endpoint.Primitive = ArvoCudaSelection::InvalidSource;
    const auto choice = ArvoCudaSelection::Select(count, sourceU);
    if (!(choice.Probability > 0))
        return endpoint;
    const PackedEmitter& entry = entries[choice.Index];
    endpoint.EmitterIndex = choice.Index;
    endpoint.Primitive = entry.Primitive;
    endpoint.SelectionProbability = choice.Probability;
    if (entry.Kind == Triangle || entry.Kind == Sphere)
    {
        const auto sample = ArvoCudaSurfaceEmission::Emit(Geometry(entry), positionU, directionU);
        endpoint.Emission = {sample.Origin, sample.Direction,   sample.Normal,       sample.Value,
                             sample.Weight, sample.PositionPdf, sample.DirectionPdf, sample.Flags};
    }
    else
        endpoint.Emission = ArvoCudaEmission::Emit(Analytic(entry), domain, positionU, directionU);
    float3 weight = endpoint.Emission.Weight;
    endpoint.Emission.Weight = make_float3(weight.x / choice.Probability, weight.y / choice.Probability,
                                           weight.z / choice.Probability);
    // Never resample a null endpoint: its source probability is already consumed.
    return endpoint;
}
// Assumes known positional support and unit geometric normal/ray direction.
// Continuous density is zero for a delta dimension; its sampled mass is one.
__device__ Density Pdf(const PackedEmitter* entries, unsigned count, ArvoCudaEmission::Domain domain,
                       unsigned index, float3 normal, float3 direction)
{
    Density density{};
    density.SelectionProbability = ArvoCudaSelection::Probability(count, index);
    if (!(density.SelectionProbability > 0))
        return density;
    const auto& entry = entries[index];
    float2 conditional;
    if (entry.Kind == Triangle || entry.Kind == Sphere)
        conditional = ArvoCudaSurfaceEmission::Pdf(Geometry(entry), normal, direction);
    else
        conditional = ArvoCudaEmission::Pdf(Analytic(entry), domain, direction);
    density.Position = conditional.x;
    density.Direction = conditional.y;
    using namespace ArvoCudaEmission;
    if (entry.Kind == ArvoCudaEndpoint::Point)
        density.Flags = Valid | DeltaPosition;
    else if (conditional.x > 0)
    {
        density.Flags = Valid;
        if (entry.Kind == ArvoCudaEndpoint::Directional)
            density.Flags |= DeltaDirection | Infinite;
        if (entry.Kind == ArvoCudaEndpoint::Environment)
            density.Flags |= Infinite;
    }
    return density;
}
} // namespace ArvoCudaEndpoint
