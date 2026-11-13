// Compute native Importance sampling. No other backend implementation is included.
// Lambertian importance sampling uses p=cos(theta)/pi; uniform hemisphere sampling uses p=1/(2*pi).
float DiffusePdf(float cosine)
{
    if (cosine <= 0.0f) return 0.0f;
    return (TransportFlags & ImportanceBit) != 0u ? cosine / Pi : 1.0f / (2.0f * Pi);
}

float3 SampleDiffuseDirection(float3 normal, float2 variates)
{
    const float u = variates.x;
    const bool importance = (TransportFlags & ImportanceBit) != 0u;
    const float uniformCosine = 1.0f - u;
    const float radius = importance ? sqrt(u) : sqrt(max(0.0f, 1.0f - uniformCosine * uniformCosine));
    const float cosine = importance ? sqrt(max(0.0f, 1.0f - radius * radius)) : uniformCosine;
    const float phi = 2.0f * Pi * variates.y;
    const float3 helper = abs(normal.y) < BasisParallelThreshold ? float3(0, 1, 0) : float3(1, 0, 0);
    const float3 tangent = normalize(cross(helper, normal));
    const float3 bitangent = cross(normal, tangent);
    return normalize(tangent * (radius * cos(phi))
        + bitangent * (radius * sin(phi)) + normal * cosine);
}
