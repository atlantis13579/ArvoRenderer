#include "RayTracer/Backends/CPU/SurfaceEmission.h"
#include <algorithm>
#include <cmath>

namespace Arvo::RayTracing::Cpu
{
// PBRT 4e, "Triangle Meshes" and "Area Lights":
// https://pbr-book.org/4ed/Shapes/Triangle_Meshes
// https://pbr-book.org/4ed/Light_Sources/Area_Lights
// Material emission is already two-sided in camera paths. Hemisphere selection
// therefore contributes 1/2 to the directional density; initial flux is 2*pi*L/pA.
// For x = center + B*n, n on the unit sphere, the differential world area is
// J(n) dA, J = |cofactor(B)*n| = |det B|*|B^-T*n|. Sampling n uniformly gives
// pA = 1/(4*pi*J), NOT 1/(4*pi*r^2) after nonuniform scale or shear.
namespace
{
constexpr float SurfacePi = 3.14159265358979323846f;
constexpr float SideProbability = 0.5f, BasisParallelThreshold = 0.999f;
float Peak(Float3 v)
{
    return std::max({std::abs(v.X), std::abs(v.Y), std::abs(v.Z)});
}
float StableLength(Float3 v)
{
    const float scale = Peak(v);
    if (!(scale > 0))
        return 0;
    v = v / scale;
    return scale * std::sqrt(Dot(v, v));
}
struct SurfaceMetric
{
    Float3 X, Y, Z, CofactorX, CofactorY, CofactorZ;
    float Scale = 0, Orientation = 0;
};
SurfaceMetric Metric(const SurfaceEmitter& source)
{
    SurfaceMetric m;
    m.Scale = std::max(Peak(source.AxisX), Peak(source.AxisY));
    if (source.Kind == SurfaceEmitterKind::Ellipsoid)
        m.Scale = std::max(m.Scale, Peak(source.AxisZ));
    if (!(m.Scale > 0) || !std::isfinite(m.Scale))
        return {};
    m.X = source.AxisX;
    m.Y = source.AxisY;
    m.Z = source.AxisZ;
    m.CofactorZ = Cross(m.X, m.Y);
    if (source.Kind == SurfaceEmitterKind::Ellipsoid)
    {
        m.CofactorX = Cross(m.Y, m.Z);
        m.CofactorY = Cross(m.Z, m.X);
        const float xScale = Peak(m.X), areaScale = Peak(m.CofactorX);
        if (xScale > 0 && areaScale > 0)
            m.Orientation = Dot(m.X / xScale, m.CofactorX / areaScale);
    }
    return m;
}
Float3 Cofactor(const SurfaceMetric& m, Float3 n)
{
    return m.CofactorX * n.X + m.CofactorY * n.Y + m.CofactorZ * n.Z;
}
float AreaPdf(const SurfaceMetric& m, float jacobian, float referenceDensity)
{
    if (!(m.Scale > 0 && jacobian > 0))
        return 0;
    // World-space cofactors preserve thin surfaces whose common-scale matrix
    // determinant would underflow. The scene must have finite area products.
    const float pdf = referenceDensity / jacobian;
    return pdf > 0 && std::isfinite(pdf) ? pdf : 0;
}
} // namespace
EmissionDensity SurfaceEmissionPdf(const SurfaceEmitter& source, Float3 geometricNormal, Float3 direction)
{
    const SurfaceMetric m = Metric(source);
    if (!(m.Scale > 0))
        return {};
    float positionPdf = 0;
    if (source.Kind == SurfaceEmitterKind::Triangle)
    {
        const float twiceArea = StableLength(m.CofactorZ);
        positionPdf = AreaPdf(m, twiceArea, 2);
        if (!(positionPdf > 0))
            return {};
        geometricNormal = m.CofactorZ / twiceArea;
    }
    else if (source.Kind == SurfaceEmitterKind::Ellipsoid && m.Orientation != 0)
    {
        // B^T*nWorld points along the original sphere normal; no subtraction
        // of a potentially distant world-space center is needed in a PDF query.
        Float3 objectNormal{Dot(m.X, geometricNormal), Dot(m.Y, geometricNormal), Dot(m.Z, geometricNormal)};
        const float length = StableLength(objectNormal);
        if (!(length > 0))
            return {};
        objectNormal = objectNormal / length;
        positionPdf = AreaPdf(m, StableLength(Cofactor(m, objectNormal)), 1 / (4 * SurfacePi));
    }
    if (!(positionPdf > 0))
        return {};
    return {positionPdf, SideProbability * std::abs(Dot(geometricNormal, direction)) / SurfacePi};
}
EmissionSample SampleSurfaceEmission(const SurfaceEmitter& source, Float2 positionVariates,
                                     Float2 directionVariates)
{
    EmissionSample result;
    const SurfaceMetric m = Metric(source);
    if (!(m.Scale > 0))
        return result;
    if (source.Kind == SurfaceEmitterKind::Triangle)
    {
        const float twiceArea = StableLength(m.CofactorZ);
        result.PositionPdf = AreaPdf(m, twiceArea, 2);
        if (!(result.PositionPdf > 0))
            return {};
        result.Normal = m.CofactorZ / twiceArea;
        const float root = std::sqrt(positionVariates.X);
        result.Origin = source.Origin + source.AxisX * (root * (1 - positionVariates.Y)) +
                        source.AxisY * (root * positionVariates.Y);
    }
    else if (source.Kind == SurfaceEmitterKind::Ellipsoid && m.Orientation != 0)
    {
        const float y = 1 - 2 * positionVariates.X, radial = std::sqrt(std::max(0.0f, 1 - y * y)),
                    phi = 2 * SurfacePi * positionVariates.Y;
        const Float3 normal{radial * std::cos(phi), y, radial * std::sin(phi)};
        const Float3 transformed = Cofactor(m, normal);
        const float jacobian = StableLength(transformed);
        result.PositionPdf = AreaPdf(m, jacobian, 1 / (4 * SurfacePi));
        if (!(result.PositionPdf > 0))
            return {};
        result.Normal = (transformed / jacobian) * (m.Orientation < 0 ? -1.0f : 1.0f);
        result.Origin =
            source.Origin + source.AxisX * normal.X + source.AxisY * normal.Y + source.AxisZ * normal.Z;
    }
    else
        return result;
    const bool front = directionVariates.X < SideProbability;
    const float radialU = front ? directionVariates.X / SideProbability
                                : (directionVariates.X - SideProbability) / SideProbability;
    const float cosine = std::sqrt(std::max(0.0f, 1 - radialU));
    if (!(cosine > 0))
        return {};
    const float radius = std::sqrt(radialU), phi = 2 * SurfacePi * directionVariates.Y;
    const Float3 helper =
        std::abs(result.Normal.Y) < BasisParallelThreshold ? Float3{0, 1, 0} : Float3{1, 0, 0};
    const Float3 tangent = Normalize(Cross(helper, result.Normal)), bitangent = Cross(result.Normal, tangent);
    result.Direction = tangent * (radius * std::cos(phi)) + bitangent * (radius * std::sin(phi)) +
                       result.Normal * (front ? cosine : -cosine);
    result.DirectionPdf = SideProbability * cosine / SurfacePi;
    result.Value = source.Radiance;
    result.Weight = (source.Radiance / result.PositionPdf) * (SurfacePi / SideProbability);
    result.Flags = EmissionValid;
    return result;
}
} // namespace Arvo::RayTracing::Cpu
