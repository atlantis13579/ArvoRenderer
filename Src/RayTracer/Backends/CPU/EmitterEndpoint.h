#pragma once
#include "Renderer/EmitterData.h"
#include "RayTracer/Backends/CPU/LightEmission.h"
namespace Arvo::RayTracing::Cpu
{
struct EmitterEndpoint
{
    EmissionSample Emission;
    uint32_t EmitterIndex = InvalidEmitterIndex, Primitive = InvalidRayPrimitiveIndex;
    float SelectionProbability = 0;
};
struct EmitterEndpointDensity
{
    float SelectionProbability = 0, Position = 0, Direction = 0;
    uint32_t Flags = 0;
};
// Weight includes inverse source PMF exactly once. Endpoint position/direction
// PDFs remain conditional (delta masses in samples); keep SelectionProbability
// separately for path-space MIS. Use an independent source-selection variate.
EmitterEndpoint SampleEmitterEndpoint(const PreparedEmissionScene& scene, float sourceVariate,
                                      Float2 positionVariates, Float2 directionVariates);
// Query on known source support, with a unit geometric normal and direction.
// Delta dimensions return zero continuous density and retain their measure flag.
EmitterEndpointDensity EmitterEndpointPdf(const PreparedEmissionScene& scene, uint32_t index,
                                          Float3 geometricNormal, Float3 direction);
} // namespace Arvo::RayTracing::Cpu
