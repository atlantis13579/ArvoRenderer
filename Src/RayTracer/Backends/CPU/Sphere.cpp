#include "RayTracer/Backends/CPU/CpuTransport.h"
#include <algorithm>
#include <cmath>

namespace Arvo::RayTracing::Cpu
{
// The full quadratic a*t*t + 2*halfB*t + c = 0 is required for
// affine ray instancing: normalizing a transformed direction changes t.
// https://pbr-book.org/4ed/Shapes/Spheres
bool IntersectSphere(const Sphere &sphere, uint32_t primitiveIndex, const Ray &ray, SurfaceHit &hit)
{
    const Float3 offset = ray.Origin - sphere.Center;
    const float directionScale =
        std::max({std::abs(ray.Direction.X), std::abs(ray.Direction.Y), std::abs(ray.Direction.Z)});
    if (directionScale == 0.0f)
        return false;
    const Float3 direction = ray.Direction / directionScale;
    const float directionLengthSquared = Dot(direction, direction);
    const float halfB = Dot(offset, direction);
    const float c = Dot(offset, offset) - sphere.Radius * sphere.Radius;
    const float discriminant = halfB * halfB - directionLengthSquared * c;
    if (discriminant < 0.0f)
        return false;
    const float root = std::sqrt(discriminant);
    // Choose the non-cancelling root, then recover the other from their product.
    const float stableRoot = -halfB - std::copysign(root, halfB);
    float nearRoot = stableRoot / directionLengthSquared;
    float farRoot = stableRoot == 0.0f ? nearRoot : c / stableRoot;
    if (nearRoot > farRoot)
        std::swap(nearRoot, farRoot);
    float distance = nearRoot / directionScale;
    if (distance <= ray.MinDistance)
        distance = farRoot / directionScale;
    if (!std::isfinite(distance) || distance <= ray.MinDistance || distance >= ray.MaxDistance ||
        distance > hit.Distance || (distance == hit.Distance && primitiveIndex >= hit.PrimitiveIndex))
        return false;
    const Float3 position = ray.Origin + ray.Direction * distance;
    const Float3 geometricNormal = Normalize(position - sphere.Center);
    Float3 normal = geometricNormal;
    if (Dot(normal, ray.Direction) > 0.0f)
        normal = -normal;
    hit.Distance = distance;
    hit.PrimitiveIndex = primitiveIndex;
    hit.Normal = normal;
    hit.GeometricNormal = geometricNormal;
    hit.FrontFace = Dot(geometricNormal, direction) < 0.0f;
    hit.Color = Float3(1.0f, 1.0f, 1.0f);
    hit.TexCoord = {};
    hit.HasTexCoord = false;
    hit.SourceTriangle = nullptr;
    hit.Barycentric = {};
    hit.MaterialIndex = sphere.MaterialIndex;
    return true;
}

} // namespace Arvo::RayTracing::Cpu
