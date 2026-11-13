#pragma once
#include "Renderer/RayTracing.h"

namespace Arvo::RayTracing
{
// Configuration/topology validation only. Native backends own every transport
// estimator. The initial single-interior contract accepts closed, outward,
// homogeneous solids with separating convex bounds and no foreign inclusions.
bool HasSubsurfaceMaterials(const Scene&);
bool ValidateSubsurfaceScene(const Scene&, const RenderSettings&, std::string& error);
} // namespace Arvo::RayTracing
