#pragma once
#include "Renderer/VertexConnectionMergingPath.h"

namespace Arvo::RayTracing::Cpu
{
VcmDensity MakeVcmDensity(double value);
// Unit geometric normal. Directional PDFs and delta-component probabilities
// both acquire abs(cosine)/distance^2 in VCM's generalized density convention.
VcmDensity VcmSurfaceDensity(VcmDensity directional, Float3 displacement, Float3 geometricNormal);
// The first surface reached from an infinite emitter uses a projected disk;
// no inverse-square factor appears. A reverse infinite endpoint stays angular.
VcmDensity VcmProjectedDiskDensity(VcmDensity disk, Float3 geometricNormal, Float3 unitDirection);
VcmDensity VcmMergeArea(double radius);
VcmDensity VcmMergeVolume(double radius);

// Both strategy arrays contain count+1 entries. Each method supplies its own
// endpoint factors and ATTEMPTED sample count; zero acceptance is not a reason
// to change that count. In particular, merging can remain eligible when the
// connection with the same slot is blocked at its other endpoint.
// Either method array may be null to disable that family; radius zero also
// disables merging. Each path measure uses one shared radius for all its cuts.
// None partitions equally over eligible strategies, rather than assigning
// weight one to each estimator. Volume vertices use the 3D sphere measure.
VcmWeights VertexConnectionMergingMisWeights(const VcmVertexPdf* vertices, uint32_t count,
                                             const VcmStrategy* connections, const VcmStrategy* mergings,
                                             double radius, MisHeuristic heuristic, double volumeRadius = 0);
} // namespace Arvo::RayTracing::Cpu
