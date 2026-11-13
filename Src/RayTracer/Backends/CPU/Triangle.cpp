#include "RayTracer/Backends/CPU/CpuTransport.h"
#include <algorithm>
#include <cmath>

namespace Arvo::RayTracing::Cpu
{
// Moller-Trumbore barycentrics preserve the original ray parameter.
// Scale the parallel threshold with direction length: affine instance rays
// must remain unnormalized so their t agrees with world-space traversal.
// https://pbr-book.org/4ed/Shapes/Triangle_Meshes
constexpr float MinimumTriangleDeterminant = 1.0e-8f;
bool IntersectTriangle(const Triangle &triangle, uint32_t primitiveIndex, const Ray &ray, SurfaceHit &hit)
{
    const Float3 edge1 = triangle.Position1 - triangle.Position0;
    const Float3 edge2 = triangle.Position2 - triangle.Position0;
    const float directionScale =
        std::max({std::abs(ray.Direction.X), std::abs(ray.Direction.Y), std::abs(ray.Direction.Z)});
    if (directionScale == 0.0f)
        return false;
    const Float3 direction = ray.Direction / directionScale;
    const Float3 p = Cross(direction, edge2);
    const float determinant = Dot(edge1, p);
    if (std::abs(determinant) < MinimumTriangleDeterminant * std::sqrt(LengthSquared(direction)))
        return false;
    const float inverseDeterminant = 1.0f / determinant;
    const Float3 t = ray.Origin - triangle.Position0;
    const float barycentric1 = Dot(t, p) * inverseDeterminant;
    if (barycentric1 < 0.0f || barycentric1 > 1.0f)
        return false;
    const Float3 q = Cross(t, edge1);
    const float barycentric2 = Dot(direction, q) * inverseDeterminant;
    if (barycentric2 < 0.0f || barycentric1 + barycentric2 > 1.0f)
        return false;
    const float distance = (Dot(edge2, q) * inverseDeterminant) / directionScale;
    if (!std::isfinite(distance) || distance <= ray.MinDistance || distance >= ray.MaxDistance ||
        distance > hit.Distance || (distance == hit.Distance && primitiveIndex >= hit.PrimitiveIndex))
        return false;

    const Float3 geometricNormal = Normalize(Cross(edge1, edge2));
    Float3 normal = geometricNormal;
    if (triangle.SmoothNormals)
    {
        const float barycentric0 = 1.0f - barycentric1 - barycentric2;
        const Float3 smoothNormal = triangle.Normal0 * barycentric0 + triangle.Normal1 * barycentric1 +
                                    triangle.Normal2 * barycentric2;
        if (LengthSquared(smoothNormal) > MinimumDirectionLengthSquared)
            normal = Normalize(smoothNormal);
    }
    const float barycentric0 = 1.0f - barycentric1 - barycentric2;
    if (Dot(normal, ray.Direction) > 0.0f)
        normal = -normal;
    hit.Distance = distance;
    hit.PrimitiveIndex = primitiveIndex;
    hit.Normal = normal;
    hit.GeometricNormal = geometricNormal;
    hit.FrontFace = Dot(geometricNormal, direction) < 0.0f;
    hit.Color =
        triangle.Color0 * barycentric0 + triangle.Color1 * barycentric1 + triangle.Color2 * barycentric2;
    hit.TexCoord = {triangle.TexCoord0.X * barycentric0 + triangle.TexCoord1.X * barycentric1 +
                        triangle.TexCoord2.X * barycentric2,
                    triangle.TexCoord0.Y * barycentric0 + triangle.TexCoord1.Y * barycentric1 +
                        triangle.TexCoord2.Y * barycentric2};
    hit.MaterialIndex = triangle.MaterialIndex;
    hit.SourceTriangle = &triangle;
    hit.Barycentric = {barycentric1, barycentric2};
    hit.HasTexCoord = true;
    return true;
}

} // namespace Arvo::RayTracing::Cpu
