// DXR GGX microfacet functions, independent of the compute backend. Local +Z is
// the macrosurface normal, alpha.x/y are positive slope scales. Directions are
// unit vectors and the view is in the upper hemisphere.
// Heitz (2018), "Sampling the GGX Distribution of Visible Normals".
// https://jcgt.org/published/0007/04/01/
// The correlated Smith model accounts for masking/shadowing, not multiple
// scattering. Below-surface reflections remain null samples. Delta reflection
// uses a different probability measure and must not call this alpha>0 model.
static const float MicrofacetPi = 3.14159265358979323846f;
float GgxProjectedLength(float3 direction, float2 alpha)
{
    float scaledX = direction.x * alpha.x, scaledY = direction.y * alpha.y;
    return sqrt(scaledX * scaledX + scaledY * scaledY + direction.z * direction.z);
}
float GgxDistribution(float3 normal, float2 alpha)
{
    if (normal.z <= 0.0f) return 0.0f;
    float slopeX = normal.x / alpha.x, slopeY = normal.y / alpha.y;
    float quadratic = slopeX * slopeX + slopeY * slopeY + normal.z * normal.z;
    return rcp(MicrofacetPi * alpha.x * alpha.y * quadratic * quadratic);
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
    float denominator = lightCosine * GgxProjectedLength(view, alpha)
        + viewCosine * GgxProjectedLength(light, alpha);
    return 2.0f * viewCosine * lightCosine / denominator;
}
float GgxVisibleNormalPdf(float3 view, float3 normal, float2 alpha)
{
    if (view.z <= 0.0f) return 0.0f;
    float projectedDensity = max(0.0f, dot(view, normal)) * GgxDistribution(normal, alpha);
    return 2.0f * projectedDensity / (view.z + GgxProjectedLength(view, alpha));
}
float3 SampleGgxVisibleNormal(float3 view, float2 alpha, float2 variates)
{
    float3 stretched = normalize(float3(alpha.x * view.x, alpha.y * view.y, view.z));
    float tangentSquared = stretched.x * stretched.x + stretched.y * stretched.y;
    float3 tangent = float3(1.0f, 0.0f, 0.0f);
    if (tangentSquared > 0.0f)
        tangent = float3(-stretched.y, stretched.x, 0.0f) / sqrt(tangentSquared);
    float3 bitangent = cross(stretched, tangent);
    float phi = 2.0f * MicrofacetPi * variates.y, radius = sqrt(variates.x);
    float diskX = radius * cos(phi);
    float interpolation = (1.0f + stretched.z) * 0.5f;
    float diskY = (1.0f - interpolation) * sqrt(max(0.0f, 1.0f - diskX * diskX))
        + interpolation * radius * sin(phi);
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
    float denominator = 2.0f * (light.z * GgxProjectedLength(view, alpha)
        + view.z * GgxProjectedLength(light, alpha));
    return GgxDistribution(normalize(view + light), alpha) / denominator;
}
