namespace ArvoCuda
{
// Walter et al. (2007), "Microfacet Models for Refraction through Rough Surfaces".
// The Fresnel proposal accounts for branch probabilities in both PDFs. Radiance
// transmission has the inverse eta-squared factor; importance transport omits it.
// https://pbr-book.org/4ed/Reflection_Models/Dielectric_BSDF
__device__ BsdfEvaluation DielectricTransmission(float3 view, float3 light, float3 normal, float2 alpha,
                                                 float incidentIndex, float transmittedIndex, unsigned mode,
                                                 float denominatorSquared, bool sampledFacet)
{
    float viewDot = BsdfDot(view, normal), lightDot = BsdfDot(light, normal);
    if (viewDot <= 0.0f || lightDot >= 0.0f || denominatorSquared <= 0.0f)
        return {};
    float ratio = incidentIndex / transmittedIndex;
    float transmission = 1.0f - DielectricFresnel(viewDot, incidentIndex, transmittedIndex);
    float reverse = 1.0f - DielectricFresnel(-lightDot, transmittedIndex, incidentIndex);
    // Use the lower-index side for known direction pairs near the critical angle.
    // Sampling must preserve the masses that actually selected its Fresnel branch.
    if (!sampledFacet)
        transmission = reverse = incidentIndex > transmittedIndex ? reverse : transmission;
    float jacobian = -lightDot / denominatorSquared;
    float value = transmission * GgxDistribution(normal, alpha) * GgxMaskingShadowing(view, light, alpha) *
                  (viewDot * jacobian / (view.z * -light.z));
    if (mode == TransportRadiance)
        value *= ratio * ratio;
    float reversePdf = reverse * GgxVisibleNormalPdf(BsdfScale(light, -1.0f), normal, alpha) * viewDot *
                       ratio * ratio / denominatorSquared;
    return {value, transmission * GgxVisibleNormalPdf(view, normal, alpha) * jacobian, reversePdf};
}
__device__ BsdfEvaluation EvaluateDielectric(float3 view, float3 light, float2 alpha, float incidentIndex,
                                             float transmittedIndex, unsigned mode)
{
    if (view.z <= 0.0f || light.z == 0.0f || alpha.x <= 0.0f || alpha.y <= 0.0f ||
        incidentIndex == transmittedIndex)
        return {};
    if (light.z > 0.0f)
    {
        float3 normal = GgxUnit(BsdfSum(view, light));
        float fresnel = DielectricFresnel(BsdfDot(view, normal), incidentIndex, transmittedIndex);
        return {fresnel * GgxReflectionFactor(view, light, alpha),
                fresnel * GgxReflectionPdf(view, light, alpha),
                fresnel * GgxReflectionPdf(light, view, alpha)};
    }
    float3 halfVector = BsdfSum(BsdfScale(view, incidentIndex / transmittedIndex), light);
    float squared = BsdfDot(halfVector, halfVector);
    if (squared == 0.0f)
        return {};
    float3 normal = BsdfScale(halfVector, 1.0f / sqrtf(squared));
    if (normal.z < 0.0f)
        normal = BsdfScale(normal, -1.0f);
    return DielectricTransmission(view, light, normal, alpha, incidentIndex, transmittedIndex, mode, squared,
                                  false);
}
__device__ BsdfSample SampleDielectric(float3 view, float2 alpha, float incidentIndex, float transmittedIndex,
                                       unsigned mode, float2 variates, float lobeVariate)
{
    if (view.z <= 0.0f)
        return {};
    bool delta = incidentIndex == transmittedIndex || (alpha.x == 0.0f && alpha.y == 0.0f);
    if (!delta && (alpha.x <= 0.0f || alpha.y <= 0.0f))
        return {};
    float3 normal = delta ? make_float3(0.0f, 0.0f, 1.0f) : SampleGgxVisibleNormal(view, alpha, variates);
    float cosine = BsdfDot(view, normal);
    float fresnel = DielectricFresnel(cosine, incidentIndex, transmittedIndex);
    if (lobeVariate < fresnel)
    {
        float3 light = BsdfSum(BsdfScale(normal, 2.0f * cosine), BsdfScale(view, -1.0f));
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
    // Rationalize the small Snell cosine difference without an IOR cutoff.
    float transmittedCosine = sqrtf((1.0f - ratio) * (1.0f + ratio) + ratio * ratio * cosine * cosine);
    float denominator = (ratio - 1.0f) * (ratio + 1.0f) / (ratio * cosine + transmittedCosine);
    BsdfEvaluation evaluated = DielectricTransmission(
        view, light, normal, alpha, incidentIndex, transmittedIndex, mode, denominator * denominator, true);
    return {light,
            evaluated.Value,
            evaluated.Pdf,
            evaluated.ReversePdf,
            transmittedIndex / incidentIndex,
            ScatteringTransmission | ScatteringGlossy};
}
} // namespace ArvoCuda
