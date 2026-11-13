// Metal-owned anisotropic GGX and height-correlated Smith functions. Directions
// use an orthonormal local +Z frame; view.z > 0, alpha > 0, variates in [0,1).
// Heitz (2018), "Sampling the GGX Distribution of Visible Normals".
// https://jcgt.org/published/0007/04/01/
// Sample the visible hemisphere's projected disk before unstretching. Reflected
// directions below the macrosurface contribute zero; retrying would bias the PDF.
// This single-scattering model excludes interreflection; alpha=0 is a delta lobe.
constant float MicrofacetPi = 3.14159265358979323846f;
float GgxProjectedLength(float3 direction, float2 alpha)
{
    float x = alpha.x * direction.x, y = alpha.y * direction.y;
    return sqrt(x * x + y * y + direction.z * direction.z);
}
float GgxDistribution(float3 normal, float2 alpha)
{
    if (normal.z <= 0.0f) return 0.0f;
    float x = normal.x / alpha.x, y = normal.y / alpha.y;
    float quadratic = x * x + y * y + normal.z * normal.z;
    return 1.0f / (MicrofacetPi * alpha.x * alpha.y * quadratic * quadratic);
}
float GgxMasking(float3 direction, float2 alpha)
{
    float cosine = abs(direction.z);
    if (cosine == 0.0f) return 0.0f;
    return 2.0f * cosine / (cosine + GgxProjectedLength(direction, alpha));
}
float GgxMaskingShadowing(float3 view, float3 light, float2 alpha)
{
    float viewCosine = abs(view.z), lightCosine = abs(light.z);
    if (viewCosine == 0.0f || lightCosine == 0.0f) return 0.0f;
    return 2.0f * viewCosine * lightCosine / (lightCosine * GgxProjectedLength(view, alpha)
        + viewCosine * GgxProjectedLength(light, alpha));
}
float GgxVisibleNormalPdf(float3 view, float3 normal, float2 alpha)
{
    if (view.z <= 0.0f) return 0.0f;
    return GgxDistribution(normal, alpha) * max(0.0f, dot(view, normal)) * 2.0f /
        (view.z + GgxProjectedLength(view, alpha));
}
float3 SampleGgxVisibleNormal(float3 view, float2 alpha, float2 variates)
{
    float3 stretched = normalize(float3(alpha.x * view.x, alpha.y * view.y, view.z));
    float tangentSquared = stretched.x * stretched.x + stretched.y * stretched.y;
    float3 tangent = tangentSquared > 0.0f ? float3(-stretched.y, stretched.x, 0.0f) / sqrt(tangentSquared)
        : float3(1.0f, 0.0f, 0.0f);
    float3 bitangent = cross(stretched, tangent);
    float radius = sqrt(variates.x), phi = 2.0f * MicrofacetPi * variates.y;
    float diskX = radius * cos(phi);
    float blend = (1.0f + stretched.z) * 0.5f;
    float diskY = (1.0f - blend) * sqrt(max(0.0f, 1.0f - diskX * diskX)) + blend * radius * sin(phi);
    float diskZ = sqrt(max(0.0f, 1.0f - diskX * diskX - diskY * diskY));
    float3 hemisphere = tangent * diskX + bitangent * diskY + stretched * diskZ;
    return normalize(float3(alpha.x * hemisphere.x, alpha.y * hemisphere.y, max(0.0f, hemisphere.z)));
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
