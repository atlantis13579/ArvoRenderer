// Walter et al. (2007), "Microfacet Models for Refraction through Rough Surfaces".
// Forward/reverse solid-angle PDFs include the Fresnel lobe-selection mass.
// Only radiance transport carries the inverse index-squared transmission factor.
// https://pbr-book.org/4ed/Reflection_Models/Dielectric_BSDF
BsdfEvaluation DielectricTransmission(float3 view, float3 light, float3 facet, float2 alpha,
                                      float incidentIndex, float transmittedIndex, uint mode,
                                      float denominatorSquared, bool sampledFacet)
{
    float outgoingCosine = dot(view, facet), incomingCosine = dot(light, facet);
    if (outgoingCosine <= 0.0 || incomingCosine >= 0.0 || denominatorSquared <= 0.0)
        return (BsdfEvaluation)0;
    float ratio = incidentIndex / transmittedIndex;
    float transmission = 1.0 - DielectricFresnel(outgoingCosine, incidentIndex, transmittedIndex);
    float reverse = 1.0 - DielectricFresnel(-incomingCosine, transmittedIndex, incidentIndex);
    // Reciprocal Fresnel from the lower index is stable near the critical angle.
    // Sampling retains its chosen branch mass instead of reconstructing it.
    if (!sampledFacet)
        transmission = reverse = incidentIndex > transmittedIndex ? reverse : transmission;
    float jacobian = -incomingCosine / denominatorSquared;
    float value = transmission * GgxDistribution(facet, alpha) * GgxMaskingShadowing(view, light, alpha) *
                  (outgoingCosine * jacobian / (view.z * -light.z));
    if (mode == TransportRadiance)
        value *= ratio * ratio;
    float pdf = transmission * GgxVisibleNormalPdf(view, facet, alpha) * jacobian;
    float reversePdf = reverse * GgxVisibleNormalPdf(-light, facet, alpha) * outgoingCosine * ratio * ratio /
                       denominatorSquared;
    return BsdfValue(value, pdf, reversePdf);
}
BsdfEvaluation EvaluateDielectric(float3 view, float3 light, float2 alpha, float incidentIndex,
                                  float transmittedIndex, uint mode)
{
    if (view.z <= 0.0 || light.z == 0.0 || alpha.x <= 0.0 || alpha.y <= 0.0 ||
        incidentIndex == transmittedIndex)
        return (BsdfEvaluation)0;
    if (light.z > 0.0)
    {
        float3 facet = normalize(view + light);
        float fresnel = DielectricFresnel(dot(view, facet), incidentIndex, transmittedIndex);
        return BsdfValue(fresnel * GgxReflectionFactor(view, light, alpha),
                         fresnel * GgxReflectionPdf(view, light, alpha),
                         fresnel * GgxReflectionPdf(light, view, alpha));
    }
    precise float3 halfVector = (incidentIndex / transmittedIndex) * view + light;
    float lengthSquared = dot(halfVector, halfVector);
    if (lengthSquared == 0.0)
        return (BsdfEvaluation)0;
    float3 facet = halfVector / sqrt(lengthSquared);
    if (facet.z < 0.0)
        facet = -facet;
    return DielectricTransmission(view, light, facet, alpha, incidentIndex, transmittedIndex, mode,
                                  lengthSquared, false);
}
BsdfSample SampleDielectric(float3 view, float2 alpha, float incidentIndex, float transmittedIndex, uint mode,
                            float2 variates, float lobeVariate)
{
    if (view.z <= 0.0)
        return (BsdfSample)0;
    bool delta = incidentIndex == transmittedIndex || (alpha.x == 0.0 && alpha.y == 0.0);
    if (!delta && (alpha.x <= 0.0 || alpha.y <= 0.0))
        return (BsdfSample)0;
    float3 facet = delta ? float3(0.0, 0.0, 1.0) : SampleGgxVisibleNormal(view, alpha, variates);
    float cosine = dot(view, facet);
    float reflected = DielectricFresnel(cosine, incidentIndex, transmittedIndex);
    if (lobeVariate < reflected)
    {
        float3 light = facet * (2.0 * cosine) - view;
        if (light.z <= 0.0)
            return (BsdfSample)0;
        if (delta)
            return BsdfEvent(light, BsdfValue(reflected / light.z, reflected, reflected), 1.0,
                             ScatteringReflection | ScatteringDelta);
        return BsdfEvent(light, EvaluateDielectric(view, light, alpha, incidentIndex, transmittedIndex, mode),
                         1.0, ScatteringReflection | ScatteringGlossy);
    }
    float ratio = incidentIndex / transmittedIndex;
    float3 light;
    if (!RefractDirection(view, facet, ratio, light) || light.z >= 0.0)
        return (BsdfSample)0;
    if (delta)
    {
        float mass = 1.0 - reflected;
        float reverse = 1.0 - DielectricFresnel(-light.z, transmittedIndex, incidentIndex);
        float value = mass / -light.z;
        if (mode == TransportRadiance)
            value *= ratio * ratio;
        return BsdfEvent(light, BsdfValue(value, mass, reverse), transmittedIndex / incidentIndex,
                         ScatteringTransmission | ScatteringDelta);
    }
    // Rationalization retains near-matched index differences instead of cancelling cosines.
    precise float cosineSquared = (1.0 - ratio) * (1.0 + ratio) + ratio * ratio * cosine * cosine;
    precise float difference = (ratio - 1.0) * (ratio + 1.0);
    float denominator = difference / (ratio * cosine + sqrt(cosineSquared));
    return BsdfEvent(light,
                     DielectricTransmission(view, light, facet, alpha, incidentIndex, transmittedIndex, mode,
                                            denominator * denominator, true),
                     transmittedIndex / incidentIndex, ScatteringTransmission | ScatteringGlossy);
}
