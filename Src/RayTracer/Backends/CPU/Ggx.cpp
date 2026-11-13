#include "RayTracer/Backends/CPU/Microfacet.h"
#include <algorithm>
#include <cmath>

namespace Arvo::RayTracing::Cpu
{
// Anisotropic Trowbridge-Reitz (GGX) NDF, height-correlated Smith masking and
// Heitz (2018), "Sampling the GGX Distribution of Visible Normals".
// https://jcgt.org/published/0007/04/01/
// Stretch the view to a hemisphere, sample its projected visible disk, then
// unstretch the normal. Reflection samples below the macrosurface are null
// events: never retry them or renormalize the reflection PDF.
// This single-scattering model omits microfacet interreflection and can lose
// energy. Delta lobes need a separate probability measure.
namespace
{
constexpr float MicrofacetPi = 3.14159265358979323846f;
float ProjectedLength(Float3 direction, Float2 alpha)
{
    const float x = alpha.X * direction.X, y = alpha.Y * direction.Y;
    return std::sqrt(x * x + y * y + direction.Z * direction.Z);
}
} // namespace

float GgxDistribution(Float3 normal, Float2 alpha)
{
    if (normal.Z <= 0.0f)
        return 0.0f;
    const float x = normal.X / alpha.X, y = normal.Y / alpha.Y;
    const float denominator = x * x + y * y + normal.Z * normal.Z;
    return 1.0f / (MicrofacetPi * alpha.X * alpha.Y * denominator * denominator);
}

float GgxMasking(Float3 direction, Float2 alpha)
{
    const float cosine = std::abs(direction.Z);
    if (cosine == 0.0f)
        return 0.0f;
    return 2.0f * cosine / (cosine + ProjectedLength(direction, alpha));
}

float GgxMaskingShadowing(Float3 view, Float3 light, Float2 alpha)
{
    const float viewCosine = std::abs(view.Z), lightCosine = std::abs(light.Z);
    if (viewCosine == 0.0f || lightCosine == 0.0f)
        return 0.0f;
    return 2.0f * viewCosine * lightCosine /
           (lightCosine * ProjectedLength(view, alpha) + viewCosine * ProjectedLength(light, alpha));
}

float GgxVisibleNormalPdf(Float3 view, Float3 normal, Float2 alpha)
{
    if (view.Z <= 0.0f)
        return 0.0f;
    return GgxDistribution(normal, alpha) * std::max(0.0f, Dot(view, normal)) * 2.0f /
           (view.Z + ProjectedLength(view, alpha));
}

Float3 SampleGgxVisibleNormal(Float3 view, Float2 alpha, Float2 variates)
{
    const Float3 stretched = Normalize({alpha.X * view.X, alpha.Y * view.Y, view.Z});
    const float tangentLengthSquared = stretched.X * stretched.X + stretched.Y * stretched.Y;
    const Float3 tangent = tangentLengthSquared > 0.0f
                               ? Float3(-stretched.Y, stretched.X, 0.0f) / std::sqrt(tangentLengthSquared)
                               : Float3(1.0f, 0.0f, 0.0f);
    const Float3 bitangent = Cross(stretched, tangent);
    const float radius = std::sqrt(variates.X), azimuth = 2.0f * MicrofacetPi * variates.Y;
    const float diskX = radius * std::cos(azimuth);
    const float blend = (1.0f + stretched.Z) * 0.5f;
    const float diskY =
        (1.0f - blend) * std::sqrt(std::max(0.0f, 1.0f - diskX * diskX)) + blend * radius * std::sin(azimuth);
    const float diskZ = std::sqrt(std::max(0.0f, 1.0f - diskX * diskX - diskY * diskY));
    const Float3 hemisphere = tangent * diskX + bitangent * diskY + stretched * diskZ;
    return Normalize({alpha.X * hemisphere.X, alpha.Y * hemisphere.Y, std::max(0.0f, hemisphere.Z)});
}

float GgxReflectionPdf(Float3 view, Float3 light, Float2 alpha)
{
    if (view.Z <= 0.0f || light.Z <= 0.0f)
        return 0.0f;
    const Float3 normal = Normalize(view + light);
    // The reflection Jacobian cancels the visible-normal dot product.
    return GgxDistribution(normal, alpha) / (2.0f * (view.Z + ProjectedLength(view, alpha)));
}

float GgxReflectionFactor(Float3 view, Float3 light, Float2 alpha)
{
    if (view.Z <= 0.0f || light.Z <= 0.0f)
        return 0.0f;
    const Float3 normal = Normalize(view + light);
    // D*G/(4*cos(view)*cos(light)); cancellation avoids divergent Smith lambdas.
    return GgxDistribution(normal, alpha) /
           (2.0f * (light.Z * ProjectedLength(view, alpha) + view.Z * ProjectedLength(light, alpha)));
}
} // namespace Arvo::RayTracing::Cpu
