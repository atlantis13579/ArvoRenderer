#ifndef ARVO_METAL_LIMITS
#define ARVO_METAL_LIMITS
// Data-only host/shader bounds. Depth-first traversal stores one deferred
// sibling per level plus the current branch, even for a binary specular tree.
#if defined(__METAL_VERSION__)
#define ARVO_METAL_BOUND constant
#else
namespace Arvo::RayTracing::MetalLimits
{
using uint = unsigned;
#define ARVO_METAL_BOUND inline constexpr
#endif
ARVO_METAL_BOUND uint MaximumReflectionBounces = 32;
ARVO_METAL_BOUND uint WhittedBranchCapacity = MaximumReflectionBounces + 1;
#undef ARVO_METAL_BOUND
#if !defined(__METAL_VERSION__)
}
#endif
#endif
