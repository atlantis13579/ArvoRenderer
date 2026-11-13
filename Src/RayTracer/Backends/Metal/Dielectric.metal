// Walter et al. (2007), "Microfacet Models for Refraction through Rough Surfaces".
// Sample a GGX visible facet and a Fresnel branch. PDFs include branch mass;
// reverse densities swap media and directions. Invalid events remain null.
// https://pbr-book.org/4ed/Reflection_Models/Dielectric_BSDF
BsdfEvaluation DielectricTransmission(float3 view, float3 light, float3 normal, float2 alpha,
                                      float incidentIndex, float transmittedIndex, uint mode,
                                      float denominatorSquared, bool sampledFacet)
{
    float viewDot = dot(view, normal), lightDot = dot(light, normal);
    if (viewDot <= 0.0f || lightDot >= 0.0f || denominatorSquared <= 0.0f)
        return {};
    float ratio = incidentIndex / transmittedIndex;
    float transmission = 1.0f - DielectricFresnel(viewDot, incidentIndex, transmittedIndex);
    float reverse = 1.0f - DielectricFresnel(-lightDot, transmittedIndex, incidentIndex);
    // Known pairs use Fresnel reciprocity from the lower index for critical-angle
    // stability. Sampled facets retain their original lobe-selection masses.
    if (!sampledFacet)
        transmission = reverse = incidentIndex > transmittedIndex ? reverse : transmission;
    float jacobian = -lightDot / denominatorSquared;
    float value = transmission * GgxDistribution(normal, alpha) * GgxMaskingShadowing(view, light, alpha) *
                  (viewDot * jacobian / (view.z * -light.z));
    if (mode == TransportRadiance)
        value *= ratio * ratio;
    return {value, transmission * GgxVisibleNormalPdf(view, normal, alpha) * jacobian,
            reverse * GgxVisibleNormalPdf(-light, normal, alpha) * viewDot * ratio * ratio /
                denominatorSquared};
}
BsdfEvaluation EvaluateDielectric(float3 view, float3 light, float2 alpha, float incidentIndex,
                                  float transmittedIndex, uint mode)
{
    if (view.z <= 0.0f || light.z == 0.0f || alpha.x <= 0.0f || alpha.y <= 0.0f ||
        incidentIndex == transmittedIndex)
        return {};
    if (light.z > 0.0f)
    {
        float3 normal = normalize(view + light);
        float fresnel = DielectricFresnel(dot(view, normal), incidentIndex, transmittedIndex);
        return {fresnel * GgxReflectionFactor(view, light, alpha),
                fresnel * GgxReflectionPdf(view, light, alpha),
                fresnel * GgxReflectionPdf(light, view, alpha)};
    }
    float3 halfVector = view * (incidentIndex / transmittedIndex) + light;
    float squared = dot(halfVector, halfVector);
    if (squared == 0.0f)
        return {};
    float3 normal = halfVector / sqrt(squared);
    if (normal.z < 0.0f)
        normal = -normal;
    return DielectricTransmission(view, light, normal, alpha, incidentIndex, transmittedIndex, mode, squared,
                                  false);
}
BsdfSample SampleDielectric(float3 view, float2 alpha, float incidentIndex, float transmittedIndex, uint mode,
                            float2 variates, float lobeVariate)
{
    if (view.z <= 0.0f)
        return {};
    bool delta = incidentIndex == transmittedIndex || (alpha.x == 0.0f && alpha.y == 0.0f);
    if (!delta && (alpha.x <= 0.0f || alpha.y <= 0.0f))
        return {};
    float3 normal = delta ? float3(0.0f, 0.0f, 1.0f) : SampleGgxVisibleNormal(view, alpha, variates);
    float viewDot = dot(view, normal);
    float fresnel = DielectricFresnel(viewDot, incidentIndex, transmittedIndex);
    if (lobeVariate < fresnel)
    {
        float3 light = normal * (2.0f * viewDot) - view;
        if (light.z <= 0.0f)
            return {};
        if (delta)
            return {light, fresnel / light.z, fresnel, fresnel, 1.0f, ScatteringReflection | ScatteringDelta};
        BsdfEvaluation evaluated =
            EvaluateDielectric(view, light, alpha, incidentIndex, transmittedIndex, mode);
        return {light,         evaluated.Value,
                evaluated.Pdf, evaluated.ReversePdf,
                1.0f,          ScatteringReflection | ScatteringGlossy};
    }
    float ratio = incidentIndex / transmittedIndex;
    float3 light;
    if (!RefractDirection(view, normal, ratio, light) || light.z >= 0.0f)
        return {};
    if (delta)
    {
        float mass = 1.0f - fresnel;
        float value = mass / -light.z;
        if (mode == TransportRadiance)
            value *= ratio * ratio;
        float reverse = 1.0f - DielectricFresnel(-light.z, transmittedIndex, incidentIndex);
        return {light,
                value,
                mass,
                reverse,
                transmittedIndex / incidentIndex,
                ScatteringTransmission | ScatteringDelta};
    }
    // Rationalized Snell difference protects nearly index-matched transmission.
    float transmittedCosine = sqrt((1.0f - ratio) * (1.0f + ratio) + ratio * ratio * viewDot * viewDot);
    float denominator = (ratio - 1.0f) * (ratio + 1.0f) / (ratio * viewDot + transmittedCosine);
    BsdfEvaluation evaluated = DielectricTransmission(
        view, light, normal, alpha, incidentIndex, transmittedIndex, mode, denominator * denominator, true);
    return {light,
            evaluated.Value,
            evaluated.Pdf,
            evaluated.ReversePdf,
            transmittedIndex / incidentIndex,
            ScatteringTransmission | ScatteringGlossy};
}
