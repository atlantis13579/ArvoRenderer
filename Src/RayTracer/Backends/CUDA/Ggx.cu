#include <cuda_runtime.h>

namespace ArvoCuda
{
// CUDA-owned anisotropic GGX distribution and height-correlated Smith model.
// Heitz (2018), "Sampling the GGX Distribution of Visible Normals":
// https://jcgt.org/published/0007/04/01/
// Local directions are unit length, view.z > 0, alpha > 0, variates in [0,1).
// Below-surface reflected directions are null samples, never rejection-resampled.
// The single-scattering model loses energy to unresolved microfacet interreflection.
constexpr float MicrofacetPi = 3.14159265358979323846f;
__device__ float3 GgxUnit(float3 direction)
{
    const float inverseLength =
        rsqrtf(direction.x * direction.x + direction.y * direction.y + direction.z * direction.z);
    return make_float3(direction.x * inverseLength, direction.y * inverseLength, direction.z * inverseLength);
}
__device__ float GgxProjectedLength(float3 direction, float2 alpha)
{
    const float x = alpha.x * direction.x, y = alpha.y * direction.y;
    return sqrtf(x * x + y * y + direction.z * direction.z);
}
__device__ float GgxDistribution(float3 normal, float2 alpha)
{
    if (normal.z <= 0.0f)
        return 0.0f;
    const float x = normal.x / alpha.x, y = normal.y / alpha.y;
    const float quadratic = x * x + y * y + normal.z * normal.z;
    return 1.0f / (MicrofacetPi * alpha.x * alpha.y * quadratic * quadratic);
}
__device__ float GgxMasking(float3 direction, float2 alpha)
{
    const float cosine = fabsf(direction.z);
    return cosine > 0.0f ? 2.0f * cosine / (cosine + GgxProjectedLength(direction, alpha)) : 0.0f;
}
__device__ float GgxMaskingShadowing(float3 view, float3 light, float2 alpha)
{
    const float v = fabsf(view.z), l = fabsf(light.z);
    if (v == 0.0f || l == 0.0f)
        return 0.0f;
    return 2.0f * v * l / (l * GgxProjectedLength(view, alpha) + v * GgxProjectedLength(light, alpha));
}
__device__ float GgxVisibleNormalPdf(float3 view, float3 normal, float2 alpha)
{
    if (view.z <= 0.0f)
        return 0.0f;
    const float visible = fmaxf(0.0f, view.x * normal.x + view.y * normal.y + view.z * normal.z);
    return GgxDistribution(normal, alpha) * visible * 2.0f / (view.z + GgxProjectedLength(view, alpha));
}
__device__ float3 SampleGgxVisibleNormal(float3 view, float2 alpha, float2 variates)
{
    const float3 stretched = GgxUnit(make_float3(alpha.x * view.x, alpha.y * view.y, view.z));
    const float xySquared = stretched.x * stretched.x + stretched.y * stretched.y;
    const float inverseTangentLength = xySquared > 0.0f ? rsqrtf(xySquared) : 0.0f;
    const float3 tangent = xySquared > 0.0f ? make_float3(-stretched.y * inverseTangentLength,
                                                          stretched.x * inverseTangentLength, 0.0f)
                                            : make_float3(1.0f, 0.0f, 0.0f);
    const float3 bitangent = make_float3(-stretched.z * tangent.y, stretched.z * tangent.x,
                                         stretched.x * tangent.y - stretched.y * tangent.x);
    const float radius = sqrtf(variates.x), phi = 2.0f * MicrofacetPi * variates.y;
    const float diskX = radius * cosf(phi);
    const float blend = 0.5f * (1.0f + stretched.z);
    const float diskY =
        (1.0f - blend) * sqrtf(fmaxf(0.0f, 1.0f - diskX * diskX)) + blend * radius * sinf(phi);
    const float diskZ = sqrtf(fmaxf(0.0f, 1.0f - diskX * diskX - diskY * diskY));
    const float3 hemisphere = make_float3(tangent.x * diskX + bitangent.x * diskY + stretched.x * diskZ,
                                          tangent.y * diskX + bitangent.y * diskY + stretched.y * diskZ,
                                          bitangent.z * diskY + stretched.z * diskZ);
    return GgxUnit(make_float3(alpha.x * hemisphere.x, alpha.y * hemisphere.y, fmaxf(0.0f, hemisphere.z)));
}
__device__ float GgxReflectionPdf(float3 view, float3 light, float2 alpha)
{
    if (view.z <= 0.0f || light.z <= 0.0f)
        return 0.0f;
    const float3 normal = GgxUnit(make_float3(view.x + light.x, view.y + light.y, view.z + light.z));
    return GgxDistribution(normal, alpha) / (2.0f * (view.z + GgxProjectedLength(view, alpha)));
}
__device__ float GgxReflectionFactor(float3 view, float3 light, float2 alpha)
{
    if (view.z <= 0.0f || light.z <= 0.0f)
        return 0.0f;
    const float3 normal = GgxUnit(make_float3(view.x + light.x, view.y + light.y, view.z + light.z));
    return GgxDistribution(normal, alpha) /
           (2.0f * (light.z * GgxProjectedLength(view, alpha) + view.z * GgxProjectedLength(light, alpha)));
}
} // namespace ArvoCuda
