// Compute-owned anisotropic GGX, correlated Smith masking, and visible normals.
// Heitz (2018), "Sampling the GGX Distribution of Visible Normals".
// https://jcgt.org/published/0007/04/01/
// Unit directions use the local +Z frame; alpha > 0 is the slope scale. Zero
// roughness is a delta event, not this continuous distribution. Null reflection
// samples must remain null. Single scattering does not return interreflection energy.
static const float MicrofacetPi = 3.14159265358979323846f;
float GgxProjectedLength(float3 direction, float2 alpha)
{
    return length(float3(alpha * direction.xy, direction.z));
}
float GgxDistribution(float3 normal, float2 alpha)
{
    if (normal.z <= 0.0f) return 0.0f;
    float3 scaled = float3(normal.xy / alpha, normal.z);
    float quadratic = dot(scaled, scaled);
    return 1.0f / (MicrofacetPi * alpha.x * alpha.y * quadratic * quadratic);
}
float GgxMasking(float3 direction, float2 alpha)
{
    float c = abs(direction.z);
    return c > 0.0f ? 2.0f * c / (c + GgxProjectedLength(direction, alpha)) : 0.0f;
}
float GgxMaskingShadowing(float3 view, float3 light, float2 alpha)
{
    float v = abs(view.z), l = abs(light.z);
    if (v == 0.0f || l == 0.0f) return 0.0f;
    return 2.0f * v * l / (l * GgxProjectedLength(view, alpha) + v * GgxProjectedLength(light, alpha));
}
float GgxVisibleNormalPdf(float3 view, float3 normal, float2 alpha)
{
    if (view.z <= 0.0f) return 0.0f;
    return GgxDistribution(normal, alpha) * max(0.0f, dot(view, normal)) * 2.0f /
        (view.z + GgxProjectedLength(view, alpha));
}
float3 SampleGgxVisibleNormal(float3 view, float2 alpha, float2 variates)
{
    float3 stretched = normalize(float3(alpha * view.xy, view.z));
    float xySquared = dot(stretched.xy, stretched.xy);
    float3 tangent = xySquared > 0.0f ? float3(-stretched.y, stretched.x, 0.0f) * rsqrt(xySquared)
        : float3(1.0f, 0.0f, 0.0f);
    float3 bitangent = cross(stretched, tangent);
    float radius = sqrt(variates.x), azimuth = 2.0f * MicrofacetPi * variates.y;
    float x = radius * cos(azimuth);
    float blend = 0.5f * (1.0f + stretched.z);
    float y = (1.0f - blend) * sqrt(max(0.0f, 1.0f - x * x)) + blend * radius * sin(azimuth);
    float z = sqrt(max(0.0f, 1.0f - x * x - y * y));
    float3 hemisphere = tangent * x + bitangent * y + stretched * z;
    return normalize(float3(alpha * hemisphere.xy, max(0.0f, hemisphere.z)));
}
float GgxReflectionPdf(float3 view, float3 light, float2 alpha)
{
    if (view.z <= 0.0f || light.z <= 0.0f) return 0.0f;
    float3 normal = normalize(view + light);
    return GgxDistribution(normal, alpha) / (2.0f * (view.z + GgxProjectedLength(view, alpha)));
}
float GgxReflectionFactor(float3 view, float3 light, float2 alpha)
{
    if (view.z <= 0.0f || light.z <= 0.0f) return 0.0f;
    return GgxDistribution(normalize(view + light), alpha) / (2.0f *
        (light.z * GgxProjectedLength(view, alpha) + view.z * GgxProjectedLength(light, alpha)));
}
