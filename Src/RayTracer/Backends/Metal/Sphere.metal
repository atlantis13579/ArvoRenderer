// General sphere quadratic for non-unit instance rays. Scaling the direction
// bounds a=d.d; the q formulation avoids subtracting nearly equal roots.
// Both bounds are open, matching native triangle traversal on every backend.
bool IntersectSphere(
    PackedSphere sphere,
    RayData ray,
    thread HitData& hit)
{
    const float3 offset = ray.Origin - sphere.CenterRadius.xyz;
    const float directionScale = max(abs(ray.Direction.x), max(abs(ray.Direction.y), abs(ray.Direction.z)));
    const float3 scaledDirection = ray.Direction / directionScale;
    const float a = dot(scaledDirection, scaledDirection);
    const float halfB = dot(offset, scaledDirection);
    const float c = dot(offset, offset)
        - sphere.CenterRadius.w * sphere.CenterRadius.w;
    const float discriminant = halfB * halfB - a * c;
    if (discriminant < 0.0f)
    {
        return false;
    }
    const float root = sqrt(discriminant);
    const float q = -halfB - copysign(root, halfB);
    float first = q == 0.0f ? -halfB / a : q / a;
    float second = q == 0.0f ? first : c / q;
    if (first > second) { const float temporary = first; first = second; second = temporary; }
    float distance = first / directionScale;
    if (distance <= ray.MinimumDistance)
    {
        distance = second / directionScale;
    }
    if (!isfinite(distance) || distance <= ray.MinimumDistance || distance >= ray.MaximumDistance
        || distance > hit.Distance)
    {
        return false;
    }
    const float3 position = ray.Origin + ray.Direction * distance;
    const float3 geometricNormal = normalize(position - sphere.CenterRadius.xyz);
    float3 normal = geometricNormal;
    if (dot(normal, ray.Direction) > 0.0f)
    {
        normal = -normal;
    }
    hit.Distance = distance;
    hit.Normal = normal;
    hit.GeometricNormal = geometricNormal;
    hit.FrontFace = dot(geometricNormal, ray.Direction / directionScale) < 0.0f;
    hit.Color = float3(1.0f);
    hit.MaterialIndex = sphere.MaterialIndex;
    hit.TexCoord = float2(0.0f);
    hit.HasTexCoord = 0u;
#if defined(ARVO_SPECTRAL_TRANSPORT)
    hit.VertexSpectrumBase = InvalidPrimitiveIndex;
    hit.Barycentric = float2(0);
#endif
    return true;
}
