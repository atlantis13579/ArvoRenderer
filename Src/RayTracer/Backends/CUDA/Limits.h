#pragma once

namespace Arvo::RayTracing::CudaLimits
{
// Shared host/device bounds only, not an algorithm implementation.
inline constexpr unsigned MaximumReflectionBounces = 32;
inline constexpr unsigned WhittedBranchCapacity = MaximumReflectionBounces + 1;
} // namespace Arvo::RayTracing::CudaLimits
