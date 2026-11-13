// Moller and Trumbore (1997), "Fast, Minimum Storage Ray-Triangle Intersection".
// Rescale only the local direction for numerical range; divide the solved
// parameter by that scale, so affine instance rays can remain unnormalized.
bool IntersectTriangle(PackedTriangle primitive, uint primitiveIndex, RayData ray, inout HitData hit)
{
    static const float RelativeParallelTolerance = 1.0e-8f;
    static const float MinimumNormalLengthSquared = 1.0e-12f;
    const float directionScale = max(abs(ray.Direction.x), max(abs(ray.Direction.y), abs(ray.Direction.z)));
    if (directionScale == 0.0f) return false;
    const float3 direction = ray.Direction / directionScale;
    const float3 edge1 = primitive.Edge1.xyz;
    const float3 edge2 = primitive.Edge2.xyz;
    const float3 determinantVector = cross(direction, edge2);
    const float determinant = dot(edge1, determinantVector);
    if (!isfinite(determinant) || abs(determinant) < RelativeParallelTolerance * length(direction)) return false;
    const float inverseDeterminant = 1.0f / determinant;
    const float3 originOffset = ray.Origin - primitive.Position0.xyz;
    const float barycentric1 = dot(originOffset, determinantVector) * inverseDeterminant;
    if (barycentric1 < 0.0f || barycentric1 > 1.0f) return false;
    const float3 crossValue = cross(originOffset, edge1);
    const float barycentric2 = dot(direction, crossValue) * inverseDeterminant;
    if (barycentric2 < 0.0f || barycentric1 + barycentric2 > 1.0f) return false;
    const float distance = (dot(edge2, crossValue) * inverseDeterminant) / directionScale;
    if (!isfinite(distance) || distance <= ray.MinimumDistance || distance >= ray.MaximumDistance || distance > hit.Distance
        || (distance == hit.Distance && primitiveIndex >= hit.PrimitiveIndex)) return false;

    const float barycentric0 = 1.0f - barycentric1 - barycentric2;
    const float3 geometricNormal = normalize(cross(edge1, edge2));
    float3 normal = geometricNormal;
    if ((primitive.Flags & 1u) != 0u)
    {
        const float3 smoothNormal = primitive.Normal0.xyz * barycentric0
            + primitive.Normal1.xyz * barycentric1 + primitive.Normal2.xyz * barycentric2;
        if (dot(smoothNormal, smoothNormal) > MinimumNormalLengthSquared) normal = normalize(smoothNormal);
    }
    if (dot(normal, ray.Direction) > 0.0f) normal = -normal;
    hit.Distance = distance;
    hit.Normal = normal;
    hit.GeometricNormal = geometricNormal;
    hit.FrontFace = dot(geometricNormal, ray.Direction / directionScale) < 0.0f;
    hit.Color = primitive.Color0.xyz * barycentric0
        + primitive.Color1.xyz * barycentric1 + primitive.Color2.xyz * barycentric2;
    hit.MaterialIndex = primitive.MaterialIndex;
    hit.TexCoord = primitive.TexCoord01.xy * barycentric0
        + primitive.TexCoord01.zw * barycentric1 + primitive.TexCoord2.xy * barycentric2;
    hit.HasTexCoord = 1u;
    hit.PrimitiveIndex = primitiveIndex;
#if defined(ARVO_SPECTRAL_TRANSPORT)
    hit.VertexSpectrumBase = primitive.Padding.x;
    hit.Barycentric = float2(barycentric1, barycentric2);
#endif
    return true;
}
