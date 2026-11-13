#ifndef ARVO_COMPUTE_LIMITS
#define ARVO_COMPUTE_LIMITS
// Data-only host/shader bounds. The depth-first Whitted stack stores at most
// one deferred sibling per level plus the current branch.
#ifdef __cplusplus
namespace Arvo::RayTracing::ComputeLimits
{
using uint = unsigned int;
#endif
static const uint MaximumReflectionBounces = 32;
static const uint WhittedBranchCapacity = MaximumReflectionBounces + 1;
#ifdef __cplusplus
}
#endif
#endif
