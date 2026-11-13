#pragma once
#include "Renderer/BidirectionalPath.h"

namespace Arvo::RayTracing::Cpu
{
PathDensity MakePathDensity(float value, PathMeasure measure);
// Unit geometric normal, displacement from source to destination. A volume
// destination converts solid angle to spatial density without a surface cosine.
// An infinite destination retains angular measure; a Dirac event stays discrete.
PathDensity ConvertPathDensity(PathDensity directional, Float3 displacement, Float3 geometricNormal,
                               PathDestination destination);
// Infinite light emission starts on a perpendicular disk: its first surface
// density is pDisk*abs(Ng.direction), with no inverse-distance-square factor.
PathDensity ProjectedDiskDensity(PathDensity diskArea, Float3 geometricNormal, Float3 unitDirection);
// Inputs contain n vertices and n+1 strategy records, all for the same complete
// path/measure. Specular connection cuts and non-intersectable endpoint hits are
// excluded. Dirac continuation factors use the cancellation convention of PBRT;
// their branch masses still belong in path throughput, never in area densities.
// None means equal partition over eligible strategies, not weight one per cut.
BidirectionalWeights BidirectionalMisWeights(const BidirectionalVertexPdf* vertices, uint32_t count,
                                             const BidirectionalStrategy* strategies, MisHeuristic heuristic);
} // namespace Arvo::RayTracing::Cpu
