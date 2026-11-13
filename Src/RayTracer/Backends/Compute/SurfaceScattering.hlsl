// Compute-owned world/local BSDF adapter. Optical constants are independent of
// diffuse textures. Geometry selects media and clips impossible shading-normal
// events; rejected directions retain their null probability mass.
// PBRT v4: https://pbr-book.org/4ed/Reflection_Models/BSDF_Representation
struct SurfaceBsdf
{
    PackedMaterial Material;
    float3 Normal, GeometricNormal, Tangent, Bitangent;
    float3 View, Reflectance;
    float IncidentIndex, TransmittedIndex, ViewGeometryCosine;
    uint FrontFace, Mode;
};
struct SurfaceEvaluation
{
    float3 Value;
    float Pdf, ReversePdf;
};
struct SurfaceSample
{
    float3 Direction, Weight;
    float Pdf, ReversePdf, RelativeIndex;
    uint Event;
};
float3 SurfaceLocal(SurfaceBsdf surface, float3 direction)
{
    return float3(dot(direction, surface.Tangent), dot(direction, surface.Bitangent),
                  dot(direction, surface.Normal));
}
float3 SurfaceWorld(SurfaceBsdf surface, float3 direction)
{
    return normalize(direction.x * surface.Tangent + direction.y * surface.Bitangent
        + direction.z * surface.Normal);
}
SurfaceBsdf MakeSurfaceBsdf(PackedMaterial material, HitData hit, float3 view, float3 color,
                           uint mode = TransportRadiance)
{
    SurfaceBsdf surface = (SurfaceBsdf)0;
    surface.Material = material;
    surface.Normal = hit.Normal;
    surface.GeometricNormal = hit.GeometricNormal;
    surface.FrontFace = hit.FrontFace;
    surface.Mode = mode;
    surface.ViewGeometryCosine = abs(dot(normalize(view), hit.GeometricNormal));
    const float3 helper = abs(hit.Normal.y) < BasisParallelThreshold
        ? float3(0, 1, 0) : float3(1, 0, 0);
    surface.Tangent = normalize(cross(helper, hit.Normal));
    surface.Bitangent = cross(hit.Normal, surface.Tangent);
    surface.View = SurfaceLocal(surface, normalize(view));
    surface.Reflectance = saturate(material.AlbedoReflectivity.xyz * color);
    surface.IncidentIndex = hit.FrontFace != 0u ? material.Optical.y : material.Optical.z;
    surface.TransmittedIndex = hit.FrontFace != 0u ? material.Optical.z : material.Optical.y;
    return surface;
}
bool SurfaceDelta(SurfaceBsdf surface)
{
    return surface.Material.Model != MaterialDiffuse && (surface.Material.Optical.x == 0
        || (surface.Material.Model == MaterialDielectric
            && surface.IncidentIndex == surface.TransmittedIndex));
}
bool SurfaceHemisphere(SurfaceBsdf surface, float3 direction, bool transmission)
{
    const float cosine = dot(surface.GeometricNormal, direction);
    return cosine != 0 && (((cosine > 0) == (surface.FrontFace != 0u)) != transmission);
}
float3 SurfaceOffset(SurfaceBsdf surface, float3 position, float3 direction)
{
    return position + surface.GeometricNormal
        * (dot(surface.GeometricNormal, direction) >= 0 ? RayEpsilon : -RayEpsilon);
}
float SurfaceAdjointCorrection(SurfaceBsdf surface, float3 direction)
{
    // Veach (1997), chapter 5: the adjoint corrects shading-normal asymmetry.
    // It multiplies f in importance mode, never either proposal density.
    if (surface.Mode == TransportRadiance) return 1;
    const float denominator = surface.ViewGeometryCosine * abs(dot(surface.Normal, direction));
    return denominator > 0 ? abs(surface.View.z * dot(surface.GeometricNormal, direction)) / denominator : 0;
}
SurfaceEvaluation EvaluateSurface(SurfaceBsdf surface, float3 direction)
{
    SurfaceEvaluation result = (SurfaceEvaluation)0;
    const float3 light = SurfaceLocal(surface, direction);
    if (surface.View.z <= 0 || !SurfaceHemisphere(surface, direction, light.z < 0)) return result;
    const bool importance = (TransportFlags & ImportanceBit) != 0u;
    const float2 alpha = surface.Material.Optical.xx;
    if (surface.Material.Model == MaterialDiffuse)
    {
        const BsdfEvaluation diffuse = EvaluateDiffuse(surface.View, light, 1, importance);
        result.Value = surface.Reflectance * diffuse.Value;
        result.Pdf = diffuse.Pdf;
        result.ReversePdf = diffuse.ReversePdf;
    }
    else if (surface.Material.Model == MaterialConductor)
    {
        const BsdfEvaluation red = EvaluateConductor(surface.View, light, alpha,
            surface.Material.ConductorIndex.x, surface.Material.ConductorExtinction.x);
        result.Value = float3(red.Value,
            EvaluateConductor(surface.View, light, alpha, surface.Material.ConductorIndex.y,
                              surface.Material.ConductorExtinction.y).Value,
            EvaluateConductor(surface.View, light, alpha, surface.Material.ConductorIndex.z,
                              surface.Material.ConductorExtinction.z).Value);
        result.Pdf = !importance && !SurfaceDelta(surface) && light.z > 0 ? 1 / (2 * Pi) : red.Pdf;
        result.ReversePdf = !importance && !SurfaceDelta(surface) && light.z > 0 ? 1 / (2 * Pi) : red.ReversePdf;
    }
    else if (surface.Material.Model == MaterialDielectric)
    {
        const BsdfEvaluation glass = EvaluateDielectric(surface.View, light, alpha,
            surface.IncidentIndex, surface.TransmittedIndex, surface.Mode);
        result.Value = glass.Value.xxx;
        result.Pdf = !importance && !SurfaceDelta(surface) && light.z != 0 ? 1 / (4 * Pi) : glass.Pdf;
        result.ReversePdf = !importance && !SurfaceDelta(surface) && light.z != 0 ? 1 / (4 * Pi) : glass.ReversePdf;
    }
    result.Value *= SurfaceAdjointCorrection(surface, direction);
    return result;
}
SurfaceSample SampleSurface(SurfaceBsdf surface, float2 variates, float lobeVariate)
{
    SurfaceSample result = (SurfaceSample)0;
    if (surface.View.z <= 0) return result;
    const bool importance = (TransportFlags & ImportanceBit) != 0u;
    const bool delta = SurfaceDelta(surface);
    if (!importance && !delta && surface.Material.Model != MaterialDiffuse)
    {
        // A uniform sphere includes both reflection and transmission. Its PDF
        // remains 1/(4*pi) even where the glass BSDF evaluates to zero.
        const float z = surface.Material.Model == MaterialDielectric ? 1 - 2 * variates.x : variates.x;
        // Keep the exact local horizon null before frame roundoff and the adjoint.
        if (z == 0) return result;
        const float radius = sqrt(max(0.0f, 1 - z * z));
        const float angle = 2 * Pi * variates.y;
        result.Direction = SurfaceWorld(surface, float3(radius * cos(angle), radius * sin(angle), z));
        const SurfaceEvaluation evaluation = EvaluateSurface(surface, result.Direction);
        if (evaluation.Pdf <= 0) return (SurfaceSample)0;
        result.Pdf = evaluation.Pdf;
        result.ReversePdf = evaluation.ReversePdf;
        result.Weight = evaluation.Value * (abs(dot(surface.Normal, result.Direction)) / evaluation.Pdf);
        result.RelativeIndex = z < 0 ? surface.TransmittedIndex / surface.IncidentIndex : 1;
        result.Event = ScatteringGlossy | (z < 0 ? ScatteringTransmission : ScatteringReflection);
        return result;
    }
    const float2 alpha = surface.Material.Optical.xx;
    BsdfSample sample = (BsdfSample)0;
    float3 values = 0.0f.xxx;
    if (surface.Material.Model == MaterialDiffuse)
    {
        sample = SampleDiffuse(surface.View, 1, variates, importance);
        values = sample.Value * surface.Reflectance;
    }
    else if (surface.Material.Model == MaterialConductor)
    {
        // GGX conductor directions and PDFs are achromatic; all three optical
        // channels use this one facet sample, with separate exact Fresnel values.
        sample = SampleConductor(surface.View, alpha, surface.Material.ConductorIndex.x,
                                 surface.Material.ConductorExtinction.x, variates);
        values.x = sample.Value;
        if (delta)
        {
            values.y = ConductorFresnel(surface.View.z, surface.Material.ConductorIndex.y,
                                       surface.Material.ConductorExtinction.y) / surface.View.z;
            values.z = ConductorFresnel(surface.View.z, surface.Material.ConductorIndex.z,
                                       surface.Material.ConductorExtinction.z) / surface.View.z;
        }
        else if (sample.Event != 0u)
        {
            values.y = EvaluateConductor(surface.View, sample.Direction, alpha,
                surface.Material.ConductorIndex.y, surface.Material.ConductorExtinction.y).Value;
            values.z = EvaluateConductor(surface.View, sample.Direction, alpha,
                surface.Material.ConductorIndex.z, surface.Material.ConductorExtinction.z).Value;
        }
    }
    else if (surface.Material.Model == MaterialDielectric)
    {
        sample = SampleDielectric(surface.View, alpha, surface.IncidentIndex, surface.TransmittedIndex,
                                  surface.Mode, variates, lobeVariate);
        values = sample.Value.xxx;
    }
    if (sample.Event == 0u || sample.Pdf <= 0) return result;
    const float3 direction = SurfaceWorld(surface, sample.Direction);
    if (!SurfaceHemisphere(surface, direction, (sample.Event & ScatteringTransmission) != 0u)) return result;
    result.Direction = direction;
    result.Weight = values * (abs(sample.Direction.z) / sample.Pdf * SurfaceAdjointCorrection(surface, direction));
    result.Pdf = sample.Pdf;
    result.ReversePdf = sample.ReversePdf;
    result.Event = sample.Event;
    result.RelativeIndex = sample.RelativeIndex;
    return result;
}
