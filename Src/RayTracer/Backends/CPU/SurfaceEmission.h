#pragma once
#include "RayTracer/Backends/CPU/LightEmission.h"

namespace Arvo::RayTracing::Cpu
{
enum class SurfaceEmitterKind : uint32_t
{
    Triangle,
    Ellipsoid
};
struct SurfaceEmitter
{
    SurfaceEmitterKind Kind = SurfaceEmitterKind::Triangle;
    Float3 Origin, AxisX, AxisY, AxisZ, Radiance;
};
// Triangle: Origin is vertex 0; AxisX/Y are its two world-space edges.
// Ellipsoid: Origin is its center; columns AxisX/Y/Z map the unit sphere to the
// surface, including the original sphere radius and the instance transform.
// Inputs, world-area products and outputs must fit finite floats, with positive
// representable area densities. Degenerate/zero-density queries return null.
EmissionSample SampleSurfaceEmission(const SurfaceEmitter& source, Float2 positionVariates,
                                     Float2 directionVariates);
// The query lies on this source and uses its unit geometric normal, never a
// smooth shading normal. Both PDFs are continuous and exclude source selection.
EmissionDensity SurfaceEmissionPdf(const SurfaceEmitter& source, Float3 geometricNormal, Float3 direction);
} // namespace Arvo::RayTracing::Cpu
