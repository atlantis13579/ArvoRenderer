#ifndef ARVO_DXR_LIMITS
#define ARVO_DXR_LIMITS
// Data-only DXR limits shared by upload validation and shader-local storage.
#ifdef __cplusplus
namespace Arvo::RayTracing::DxrLimits
{
using uint = unsigned int;
#endif
static const uint MaximumReflectionBounces = 32;
// Depth-first binary traversal retains one sibling per level plus the next ray.
static const uint WhittedBranchCapacity = MaximumReflectionBounces + 1;
#ifdef __cplusplus
}
#endif
#endif
