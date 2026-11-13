// DXR world-space scattering. The payload's geometric side selects the medium;
// interpolated normals orient the lobe, never the dielectric boundary itself.
// Local BSDFs use unit directions pointing away from the surface and view.z > 0.
// Radiance/importance modes differ in refraction and shading-normal adjoints.
// See Veach (1997), chapter 5. The input shading normal faces the view direction;
// replacing it at grazing incidence would make reverse paths use another lobe.
struct SurfaceBsdf
{
    MaterialData Material;
    float3 Normal, GeometricNormal, Tangent, Bitangent, View, Reflectance;
    float IncidentIndex, TransmittedIndex, ViewGeometryCosine;
    uint Mode;
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

SurfaceBsdf MakeSurfaceBsdf(MaterialData material, Payload hit, float3 view, float3 color,
                           uint mode = TransportRadiance)
{
    SurfaceBsdf surface;
    surface.Material = material;
    const bool front = (hit.Flags & SurfaceFrontFaceFlag) != 0u;
    surface.GeometricNormal = front ? hit.GeometricNormal : -hit.GeometricNormal;
    surface.Normal = hit.Normal;
    view = normalize(view);
    surface.Mode = mode;
    surface.ViewGeometryCosine = abs(dot(view, surface.GeometricNormal));
    const float3 helper = abs(surface.Normal.y) < BasisParallelThreshold
        ? float3(0, 1, 0) : float3(1, 0, 0);
    surface.Tangent = normalize(cross(helper, surface.Normal));
    surface.Bitangent = cross(surface.Normal, surface.Tangent);
    surface.View = float3(dot(view, surface.Tangent), dot(view, surface.Bitangent), dot(view, surface.Normal));
    surface.Reflectance = saturate(color * material.AlbedoReflectivity.xyz);
    surface.IncidentIndex = front ? material.Optical.y : material.Optical.z;
    surface.TransmittedIndex = front ? material.Optical.z : material.Optical.y;
    return surface;
}
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
bool SurfaceDelta(SurfaceBsdf surface)
{
    return (surface.Material.Model == MaterialConductor && surface.Material.Optical.x == 0) ||
        (surface.Material.Model == MaterialDielectric &&
         (surface.Material.Optical.x == 0 || surface.IncidentIndex == surface.TransmittedIndex));
}
bool SurfaceDirectionValid(SurfaceBsdf surface, float3 direction, float localCosine)
{
    // Clipping shading/geometric disagreement retains a null event. Resampling
    // it would change the proposal distribution and bias the estimator.
    const float geometricCosine = dot(direction, surface.GeometricNormal);
    return localCosine > 0 ? geometricCosine > 0 : localCosine < 0 && geometricCosine < 0;
}
float3 SurfaceOffset(SurfaceBsdf surface, float3 position, float3 direction)
{
    return position + surface.GeometricNormal *
        (dot(direction, surface.GeometricNormal) >= 0 ? RayEpsilon : -RayEpsilon);
}
float3 SurfaceConductorFresnel(SurfaceBsdf surface, float cosine)
{
    // MaterialScattering stores these RGB constants relative to the exterior medium.
    const float3 eta = surface.Material.ConductorIndex.xyz;
    const float3 k = surface.Material.ConductorExtinction.xyz;
    return float3(ConductorFresnel(cosine, eta.x, k.x), ConductorFresnel(cosine, eta.y, k.y),
                  ConductorFresnel(cosine, eta.z, k.z));
}
float SurfaceAdjointCorrection(SurfaceBsdf surface, float3 direction)
{
    // The adjoint is an energy correction; PDFs retain the original measure.
    // A zero geometric/view or shading/light cosine contributes no finite path.
    if (surface.Mode == TransportRadiance) return 1;
    const float denominator = surface.ViewGeometryCosine * abs(dot(surface.Normal, direction));
    return denominator > 0 ? abs(surface.View.z * dot(direction, surface.GeometricNormal)) / denominator : 0;
}
SurfaceEvaluation EvaluateSurface(SurfaceBsdf surface, float3 direction)
{
    SurfaceEvaluation result = (SurfaceEvaluation)0;
    const float3 local = SurfaceLocal(surface, direction);
    if (!SurfaceDirectionValid(surface, direction, local.z) || surface.View.z <= 0) return result;
    const bool importance = (TransportFlags & ImportanceBit) != 0u;
    const float2 alpha = surface.Material.Optical.xx;
    if (surface.Material.Model == MaterialDiffuse)
    {
        const BsdfEvaluation diffuse = EvaluateDiffuse(surface.View, local, 1, importance);
        result.Value = surface.Reflectance * diffuse.Value;
        result.Pdf = diffuse.Pdf;
        result.ReversePdf = diffuse.ReversePdf;
    }
    else if (surface.Material.Model == MaterialConductor && !SurfaceDelta(surface) && local.z > 0)
    {
        result.Value = SurfaceConductorFresnel(surface, dot(surface.View, normalize(surface.View + local)))
            * GgxReflectionFactor(surface.View, local, alpha);
        result.Pdf = importance ? GgxReflectionPdf(surface.View, local, alpha) : 1 / (2 * Pi);
        result.ReversePdf = importance ? GgxReflectionPdf(local, surface.View, alpha) : 1 / (2 * Pi);
    }
    else if (surface.Material.Model == MaterialDielectric)
    {
        const BsdfEvaluation glass = EvaluateDielectric(surface.View, local, alpha, surface.IncidentIndex,
                                                        surface.TransmittedIndex, surface.Mode);
        result.Value = glass.Value.xxx;
        result.Pdf = SurfaceDelta(surface) ? 0 : importance ? glass.Pdf : 1 / (4 * Pi);
        result.ReversePdf = SurfaceDelta(surface) ? 0 : importance ? glass.ReversePdf : 1 / (4 * Pi);
    }
    result.Value *= SurfaceAdjointCorrection(surface, direction);
    return result;
}
SurfaceSample SampleSurface(SurfaceBsdf surface, float2 variates, float lobeVariate)
{
    SurfaceSample result = (SurfaceSample)0;
    if (surface.View.z <= 0) return result;
    const bool importance = (TransportFlags & ImportanceBit) != 0u;
    const bool uniformProposal = !importance && !SurfaceDelta(surface) && surface.Material.Model != MaterialDiffuse;
    const float2 alpha = surface.Material.Optical.xx;
    BsdfSample event = (BsdfSample)0;
    float3 value = 0.0.xxx;
    if (surface.Material.Model == MaterialDiffuse)
    {
        event = SampleDiffuse(surface.View, 1, variates, importance);
        value = surface.Reflectance * event.Value;
    }
    else if (uniformProposal)
    {
        // Uniform proposals cover both sides of rough glass, and the reflection
        // hemisphere of rough metals. Evaluation still uses the physical BSDF.
        const bool glass = surface.Material.Model == MaterialDielectric;
        const float cosine = glass ? 1 - 2 * variates.x : 1 - variates.x;
        // A frame roundtrip must not turn this zero-measure horizon into a tiny
        // cosine that the adjoint then amplifies into finite sample energy.
        if (cosine == 0) return result;
        const float radial = sqrt(max(0, 1 - cosine * cosine));
        const float azimuth = 2 * Pi * variates.y;
        event.Direction = float3(radial * cos(azimuth), radial * sin(azimuth), cosine);
        event.Event = (cosine < 0 ? ScatteringTransmission : ScatteringReflection) | ScatteringGlossy;
        event.RelativeIndex = cosine < 0 ? surface.TransmittedIndex / surface.IncidentIndex : 1;
        const SurfaceEvaluation evaluated = EvaluateSurface(surface, SurfaceWorld(surface, event.Direction));
        value = evaluated.Value;
        event.Pdf = evaluated.Pdf;
        event.ReversePdf = evaluated.ReversePdf;
    }
    else if (surface.Material.Model == MaterialConductor)
    {
        // The GGX direction/PDF is independent of the RGB Fresnel amplitudes.
        event = SampleConductor(surface.View, alpha,
            surface.Material.ConductorIndex.x, surface.Material.ConductorExtinction.x, variates);
        if (event.Event != 0u)
        {
            if ((event.Event & ScatteringDelta) != 0u)
                value = SurfaceConductorFresnel(surface, surface.View.z) / event.Direction.z;
            else
                value = SurfaceConductorFresnel(surface, dot(surface.View, normalize(surface.View + event.Direction)))
                    * GgxReflectionFactor(surface.View, event.Direction, alpha);
        }
    }
    else if (surface.Material.Model == MaterialDielectric)
    {
        event = SampleDielectric(surface.View, alpha, surface.IncidentIndex, surface.TransmittedIndex,
                                 surface.Mode, variates, lobeVariate);
        // Preserve the sampled value near matched indices: reconstructing the
        // half-vector can lose its tiny refractive difference through cancellation.
        value = event.Value.xxx;
    }
    if (event.Event == 0u || event.Pdf <= 0) return result;
    const float3 world = SurfaceWorld(surface, event.Direction);
    if (!SurfaceDirectionValid(surface, world, event.Direction.z)) return result;
    result.Direction = world;
    const float correction = uniformProposal ? 1 : SurfaceAdjointCorrection(surface, world);
    result.Weight = value * (abs(event.Direction.z) / event.Pdf * correction);
    result.Pdf = event.Pdf;
    result.ReversePdf = event.ReversePdf;
    result.RelativeIndex = event.RelativeIndex;
    result.Event = event.Event;
    return result;
}
