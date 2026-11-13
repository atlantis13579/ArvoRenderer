// World-space Metal BSDF adapter. Geometric sidedness determines the adjacent
// media and ray offset; the shading frame determines the sampled lobe.
// A clipped shading-normal event keeps its null probability, without resampling.
// Importance transport applies Veach's shading-normal adjoint (1997, ch. 5).
// The input shading normal faces view. A view-dependent fallback normal would
// select a different lobe on the reverse path, invalidating its density.
struct SurfaceBsdf
{
    PackedMaterial Material;
    float3 Normal, GeometricNormal, Tangent, Bitangent, View, Reflectance;
    float IncidentIndex, TransmittedIndex, ViewGeometryCosine;
    uint Mode;
    bool Importance;
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
SurfaceBsdf MakeSurfaceBsdf(PackedMaterial material, HitData hit, float3 view, float3 color, uint flags,
                           uint mode = TransportRadiance)
{
    SurfaceBsdf surface;
    surface.Material = material;
    surface.GeometricNormal = hit.FrontFace != 0u ? hit.GeometricNormal : -hit.GeometricNormal;
    surface.Normal = hit.Normal;
    view = normalize(view);
    surface.Mode = mode;
    surface.ViewGeometryCosine = abs(dot(view, surface.GeometricNormal));
    const float3 helper = abs(surface.Normal.y) < BasisParallelThreshold
        ? float3(0, 1, 0) : float3(1, 0, 0);
    surface.Tangent = normalize(cross(helper, surface.Normal));
    surface.Bitangent = cross(surface.Normal, surface.Tangent);
    surface.View = float3(dot(view, surface.Tangent), dot(view, surface.Bitangent), dot(view, surface.Normal));
    surface.Reflectance = clamp(material.AlbedoReflectivity.xyz * color, 0.0f, 1.0f);
    surface.IncidentIndex = hit.FrontFace != 0u ? material.Optical.y : material.Optical.z;
    surface.TransmittedIndex = hit.FrontFace != 0u ? material.Optical.z : material.Optical.y;
    surface.Importance = (flags & ImportanceBit) != 0u;
    return surface;
}
float3 SurfaceLocal(SurfaceBsdf surface, float3 direction)
{
    return float3(dot(direction, surface.Tangent), dot(direction, surface.Bitangent), dot(direction, surface.Normal));
}
float3 SurfaceWorld(SurfaceBsdf surface, float3 direction)
{
    return normalize(surface.Tangent * direction.x + surface.Bitangent * direction.y + surface.Normal * direction.z);
}
bool SurfaceDelta(SurfaceBsdf surface)
{
    if (surface.Material.Model == MaterialConductor) return surface.Material.Optical.x == 0;
    return surface.Material.Model == MaterialDielectric &&
        (surface.Material.Optical.x == 0 || surface.IncidentIndex == surface.TransmittedIndex);
}
bool SurfaceDirectionValid(SurfaceBsdf surface, float3 world, float cosine)
{
    const float geometricCosine = dot(surface.GeometricNormal, world);
    return cosine > 0 ? geometricCosine > 0 : cosine < 0 && geometricCosine < 0;
}
float3 SurfaceOffset(SurfaceBsdf surface, float3 position, float3 direction)
{
    const float side = dot(surface.GeometricNormal, direction) >= 0 ? RayEpsilon : -RayEpsilon;
    return position + surface.GeometricNormal * side;
}
float3 SurfaceMetalFresnel(SurfaceBsdf surface, float cosine)
{
    // Authored complex indices are already relative to the exterior medium.
    const float3 index = surface.Material.ConductorIndex.xyz;
    const float3 extinction = surface.Material.ConductorExtinction.xyz;
    return float3(ConductorFresnel(cosine, index.x, extinction.x),
                  ConductorFresnel(cosine, index.y, extinction.y),
                  ConductorFresnel(cosine, index.z, extinction.z));
}
float SurfaceAdjointCorrection(SurfaceBsdf surface, float3 incoming)
{
    // Keep energy correction separate from the forward/reverse sampling laws.
    // The local dielectric supplies eta scaling; this factor supplies only the
    // shading/geometric cosine ratio. Delta samples retain probability masses.
    if (surface.Mode == TransportRadiance) return 1;
    const float denominator = surface.ViewGeometryCosine * abs(dot(surface.Normal, incoming));
    return denominator > 0 ? abs(surface.View.z * dot(surface.GeometricNormal, incoming)) / denominator : 0;
}
SurfaceEvaluation EvaluateSurface(SurfaceBsdf surface, float3 incoming)
{
    const float3 local = SurfaceLocal(surface, incoming);
    if (surface.View.z <= 0 || !SurfaceDirectionValid(surface, incoming, local.z)) return {};
    const float2 alpha = float2(surface.Material.Optical.x);
    if (surface.Material.Model == MaterialDiffuse)
    {
        const BsdfEvaluation evaluated = EvaluateDiffuse(surface.View, local, 1, surface.Importance);
        return {surface.Reflectance * (evaluated.Value * SurfaceAdjointCorrection(surface, incoming)),
                evaluated.Pdf, evaluated.ReversePdf};
    }
    if (SurfaceDelta(surface)) return {};
    if (surface.Material.Model == MaterialConductor && local.z > 0)
    {
        const float3 fresnel = SurfaceMetalFresnel(surface, dot(surface.View, normalize(surface.View + local)));
        const float pdf = surface.Importance ? GgxReflectionPdf(surface.View, local, alpha) : 1 / (2 * Pi);
        const float reversePdf = surface.Importance ? GgxReflectionPdf(local, surface.View, alpha) : 1 / (2 * Pi);
        return {fresnel * (GgxReflectionFactor(surface.View, local, alpha) * SurfaceAdjointCorrection(surface, incoming)),
                pdf, reversePdf};
    }
    if (surface.Material.Model == MaterialDielectric)
    {
        const BsdfEvaluation evaluated = EvaluateDielectric(surface.View, local, alpha, surface.IncidentIndex,
                                                            surface.TransmittedIndex, surface.Mode);
        return {float3(evaluated.Value * SurfaceAdjointCorrection(surface, incoming)),
                surface.Importance ? evaluated.Pdf : 1 / (4 * Pi),
                surface.Importance ? evaluated.ReversePdf : 1 / (4 * Pi)};
    }
    return {};
}
SurfaceSample SampleSurface(SurfaceBsdf surface, float2 variates, float lobeVariate)
{
    if (surface.View.z <= 0) return {};
    const float2 alpha = float2(surface.Material.Optical.x);
    const bool uniformProposal = !surface.Importance && !SurfaceDelta(surface) && surface.Material.Model != MaterialDiffuse;
    BsdfSample event{};
    float3 value = float3(0);
    if (surface.Material.Model == MaterialDiffuse)
    {
        event = SampleDiffuse(surface.View, 1, variates, surface.Importance);
        value = surface.Reflectance * event.Value;
    }
    else if (uniformProposal)
    {
        // Uniform proposals retain the physical GGX BSDF. Glass must cover the
        // complete sphere; metals only have reflection on the upper hemisphere.
        const bool glass = surface.Material.Model == MaterialDielectric;
        const float cosine = glass ? 1 - 2 * variates.x : 1 - variates.x;
        // Resolve the exact local horizon before world-frame roundoff; its
        // inverse shading cosine is undefined in adjoint transport.
        if (cosine == 0) return {};
        const float radial = sqrt(max(0.0f, 1 - cosine * cosine));
        const float azimuth = 2 * Pi * variates.y;
        event.Direction = float3(radial * cos(azimuth), radial * sin(azimuth), cosine);
        event.RelativeIndex = cosine < 0 ? surface.TransmittedIndex / surface.IncidentIndex : 1;
        event.Event = (cosine < 0 ? ScatteringTransmission : ScatteringReflection) | ScatteringGlossy;
        const SurfaceEvaluation evaluated = EvaluateSurface(surface, SurfaceWorld(surface, event.Direction));
        value = evaluated.Value;
        event.Pdf = evaluated.Pdf;
        event.ReversePdf = evaluated.ReversePdf;
    }
    else if (surface.Material.Model == MaterialConductor)
    {
        // GGX directions do not depend on Fresnel color, so one direction carries
        // all three complex-index channels without a wavelength/lobe mixture.
        event = SampleConductor(surface.View, alpha,
            surface.Material.ConductorIndex.x, surface.Material.ConductorExtinction.x, variates);
        if (event.Event != 0u)
        {
            if ((event.Event & ScatteringDelta) != 0u)
                value = SurfaceMetalFresnel(surface, surface.View.z) / event.Direction.z;
            else
                value = SurfaceMetalFresnel(surface, dot(surface.View, normalize(surface.View + event.Direction)))
                    * GgxReflectionFactor(surface.View, event.Direction, alpha);
        }
    }
    else if (surface.Material.Model == MaterialDielectric)
    {
        event = SampleDielectric(surface.View, alpha, surface.IncidentIndex, surface.TransmittedIndex,
                                 surface.Mode, variates, lobeVariate);
        // Keep the sampled Snell expression near matched indices; rebuilding its
        // half-vector from world directions can cancel the refractive difference.
        value = float3(event.Value);
    }
    if (event.Event == 0u || event.Pdf <= 0) return {};
    const float3 world = SurfaceWorld(surface, event.Direction);
    if (!SurfaceDirectionValid(surface, world, event.Direction.z)) return {};
    const float correction = uniformProposal ? 1 : SurfaceAdjointCorrection(surface, world);
    return {world, value * (abs(event.Direction.z) / event.Pdf * correction),
            event.Pdf, event.ReversePdf, event.RelativeIndex, event.Event};
}
