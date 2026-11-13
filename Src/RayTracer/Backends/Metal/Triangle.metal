// Native Moller-Trumbore on (min,max). Normalize the direction's scale locally,
// then recover its original t; affine instance rays must not be renormalized.
bool IntersectTriangle(
    PackedTriangle primitive,
    RayData ray,
    thread HitData& hit)
{
    const float3 position0 = primitive.Position0.xyz;
    const float3 edge1 = primitive.Edge1.xyz;
    const float3 edge2 = primitive.Edge2.xyz;
    const float directionScale = max(abs(ray.Direction.x), max(abs(ray.Direction.y), abs(ray.Direction.z)));
    const float3 scaledDirection = ray.Direction / directionScale;
    const float3 determinantVector = cross(scaledDirection, edge2);
    const float determinant = dot(edge1, determinantVector);
    if (!isfinite(determinant) || fabs(determinant) < MinimumTriangleDeterminant * length(scaledDirection))
    {
        return false;
    }
    const float inverseDeterminant = 1.0f / determinant;
    const float3 originOffset = ray.Origin - position0;
    const float barycentric1 = dot(originOffset, determinantVector) * inverseDeterminant;
    if (barycentric1 < 0.0f || barycentric1 > 1.0f)
    {
        return false;
    }
    const float3 crossValue = cross(originOffset, edge1);
    const float barycentric2 = dot(scaledDirection, crossValue) * inverseDeterminant;
    if (barycentric2 < 0.0f || barycentric1 + barycentric2 > 1.0f)
    {
        return false;
    }
    const float distance = (dot(edge2, crossValue) * inverseDeterminant) / directionScale;
    if (!isfinite(distance) || distance <= ray.MinimumDistance || distance >= ray.MaximumDistance
        || distance > hit.Distance)
    {
        return false;
    }

    const float barycentric0 = 1.0f - barycentric1 - barycentric2;
    const float3 geometricNormal = normalize(cross(edge1, edge2));
    float3 normal = geometricNormal;
    if ((primitive.Flags & 1u) != 0u)
    {
        const float3 smoothNormal = primitive.Normal0.xyz * barycentric0
            + primitive.Normal1.xyz * barycentric1
            + primitive.Normal2.xyz * barycentric2;
        if (dot(smoothNormal, smoothNormal) > MinimumDirectionLengthSquared)
        {
            normal = normalize(smoothNormal);
        }
    }
    if (dot(normal, ray.Direction) > 0.0f)
    {
        normal = -normal;
    }
    hit.Distance = distance;
    hit.Normal = normal;
    hit.GeometricNormal = geometricNormal;
    hit.FrontFace = dot(geometricNormal, ray.Direction / directionScale) < 0.0f;
    hit.Color = primitive.Color0.xyz * barycentric0
        + primitive.Color1.xyz * barycentric1
        + primitive.Color2.xyz * barycentric2;
    hit.MaterialIndex = primitive.MaterialIndex;
    hit.TexCoord = primitive.TexCoord01.xy * barycentric0
        + primitive.TexCoord01.zw * barycentric1
        + primitive.TexCoord2.xy * barycentric2;
    hit.HasTexCoord = 1u;
#if defined(ARVO_SPECTRAL_TRANSPORT)
    hit.VertexSpectrumBase = primitive.Padding0;
    hit.Barycentric = float2(barycentric1, barycentric2);
#endif
    return true;
}
