// Solve a*t^2+2*b*t+c=0, retaining a=dot(d,d) for non-unit instance/query rays.
// Component scaling protects very small/large directions; the q-form avoids
// cancellation between b and the square root for the farther quadratic root.
bool IntersectSphere(PackedSphere sphere, uint primitiveIndex, RayData ray, inout HitData hit)
{
    const float directionScale = max(abs(ray.Direction.x), max(abs(ray.Direction.y), abs(ray.Direction.z)));
    if (directionScale == 0.0f) return false;
    const float3 direction = ray.Direction / directionScale;
    const float3 offset = ray.Origin - sphere.CenterRadius.xyz;
    const float a = dot(direction, direction);
    const float halfB = dot(offset, direction);
    const float c = dot(offset, offset) - sphere.CenterRadius.w * sphere.CenterRadius.w;
    const float discriminant = halfB * halfB - a * c;
    if (!isfinite(discriminant) || discriminant < 0.0f) return false;
    const float root = sqrt(discriminant);
    const float q = -halfB - (halfB >= 0.0f ? root : -root);
    float distance0 = q == 0.0f ? -halfB / a : q / a;
    float distance1 = q == 0.0f ? distance0 : c / q;
    if (distance0 > distance1)
    {
        const float temporary = distance0;
        distance0 = distance1;
        distance1 = temporary;
    }
    distance0 /= directionScale;
    distance1 /= directionScale;
    const float distance = distance0 > ray.MinimumDistance ? distance0 : distance1;
    if (!isfinite(distance) || distance <= ray.MinimumDistance || distance >= ray.MaximumDistance || distance > hit.Distance
        || (distance == hit.Distance && primitiveIndex >= hit.PrimitiveIndex)) return false;
    const float3 position = ray.Origin + ray.Direction * distance;
    const float3 geometricNormal = normalize(position - sphere.CenterRadius.xyz);
    float3 normal = geometricNormal;
    if (dot(normal, ray.Direction) > 0.0f) normal = -normal;
    hit.Distance = distance;
    hit.Normal = normal;
    hit.GeometricNormal = geometricNormal;
    hit.FrontFace = dot(geometricNormal, direction) < 0.0f;
    hit.Color = 1.0f.xxx;
    hit.MaterialIndex = sphere.MaterialIndex;
    hit.TexCoord = 0.0f.xx;
    hit.HasTexCoord = 0u;
    hit.PrimitiveIndex = primitiveIndex;
#if defined(ARVO_SPECTRAL_TRANSPORT)
    hit.VertexSpectrumBase = InvalidPrimitiveIndex;
    hit.Barycentric = 0.0f.xx;
#endif
    return true;
}
