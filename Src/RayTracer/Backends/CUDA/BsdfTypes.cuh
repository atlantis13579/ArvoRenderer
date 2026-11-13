#pragma once
#include <cuda_runtime.h>
namespace ArvoCuda
{
constexpr unsigned ScatteringReflection = 1u, ScatteringTransmission = 2u;
constexpr unsigned ScatteringDiffuse = 4u, ScatteringGlossy = 8u, ScatteringDelta = 16u;
constexpr unsigned TransportRadiance = 0u, TransportImportance = 1u;
struct BsdfEvaluation
{
    float Value, Pdf, ReversePdf;
};
struct BsdfSample
{
    float3 Direction;
    float Value, Pdf, ReversePdf, RelativeIndex;
    unsigned Event;
};
// Elementary native-vector operations, not scattering or sampling algorithms.
__device__ float BsdfDot(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ float3 BsdfScale(float3 a, float t)
{
    return make_float3(a.x * t, a.y * t, a.z * t);
}
__device__ float3 BsdfSum(float3 a, float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
} // namespace ArvoCuda
