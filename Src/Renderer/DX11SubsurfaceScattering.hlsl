// DX11 preview: homogeneous semi-infinite single scattering under locally
// uniform incident illumination. Integrating exp[-sigma_t*z*(1/mu_i+1/mu_o)]
// gives albedo*phase/(mu_i+mu_o); refraction adds the solid-angle eta^2 factor.
// Mean free path cancels only in this thick, planar limit. Finite thickness,
// lateral transport, internal reflection and multiple scattering require DXR.
// See PBRT v3, "Subsurface Scattering Using the Diffusion Equation", single
// scattering term. This is neither its diffusion profile nor a BSSRDF solver.
float3 RasterSubsurfaceBsdf(RasterMaterial material, float3 normal, float3 view,
                           float3 light, float albedoScale, float anisotropy)
{
    const float viewCosine = dot(normal, view), lightCosine = dot(normal, light);
    if (viewCosine <= 0 || lightCosine <= 0) return float3(0, 0, 0);
    const float ratio = material.IncidentIndex / material.TransmittedIndex;
    const float3 inwardView = RasterRefract(view, normal, ratio);
    const float3 inwardLight = RasterRefract(light, normal, ratio);
    const float viewDepth = -dot(normal, inwardView), lightDepth = -dot(normal, inwardLight);
    if (viewDepth <= 0 || lightDepth <= 0) return float3(0, 0, 0);
    const float magnitude = abs(anisotropy);
    const float cosine = dot(inwardLight, -inwardView);
    const float aligned = anisotropy < 0 ? -cosine : cosine;
    const float square = (1-magnitude)*(1-magnitude) + 2*magnitude*(1-clamp(aligned, -1.0f, 1.0f));
    const float phase = (1-anisotropy*anisotropy)/(4*RasterPi*square*sqrt(square));
    const float entry = 1-RasterDielectricFresnel(lightCosine, material.IncidentIndex, material.TransmittedIndex);
    const float exit = 1-RasterDielectricFresnel(viewCosine, material.IncidentIndex, material.TransmittedIndex);
    return clamp(material.Reflectance * albedoScale, 0.0f, 1.0f) *
        (phase * entry * exit * ratio * ratio / (viewDepth + lightDepth));
}
float3 RasterSubsurfaceEnvironment(RasterMaterial material, float3 normal, float3 view,
    float3 top, float3 bottom, RasterMedium medium, float3 position, float albedoScale, float anisotropy)
{
    const float3 helper = abs(normal.z) < 1 ? float3(0, 0, 1) : float3(0, 1, 0);
    const float3 tangent = normalize(cross(helper, normal)), bitangent = cross(normal, tangent);
    float3 result = float3(0, 0, 0);
    for (uint sampleIndex = 0; sampleIndex < RasterEnvironmentSamples; ++sampleIndex)
    {
        const float u = (sampleIndex + .5f) / RasterEnvironmentSamples;
        const float rotation = sampleIndex * RasterGoldenRotation;
        const float phi = 2 * RasterPi * (rotation-floor(rotation));
        const float3 light = tangent*(sqrt(u)*cos(phi)) + bitangent*(sqrt(u)*sin(phi)) + normal*sqrt(1-u);
        result += RasterEnvironmentPath(medium, position, light, top, bottom) *
            RasterSubsurfaceBsdf(material, normal, view, light, albedoScale, anisotropy) * RasterPi;
    }
    return result / RasterEnvironmentSamples;
}
