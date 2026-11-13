// Walter et al. (2007), "Microfacet Models for Refraction through Rough Surfaces".
// Fresnel splits a GGX visible-normal proposal. No rejection resampling is used.
// https://pbr-book.org/4ed/Reflection_Models/Dielectric_BSDF
BsdfEvaluation DielectricTransmission(float3 view, float3 light, float3 normal, float2 alpha,
                                      float incidentIndex, float transmittedIndex, uint mode,
                                      float denominatorSquared, bool sampledFacet)
{
    const float a = dot(view, normal), b = dot(light, normal);
    if (a <= 0.0 || b >= 0.0 || denominatorSquared <= 0.0)
        return (BsdfEvaluation)0;
    const float rho = incidentIndex / transmittedIndex;
    float transmission = 1.0 - DielectricFresnel(a, incidentIndex, transmittedIndex);
    float reverseTransmission = 1.0 - DielectricFresnel(-b, transmittedIndex, incidentIndex);
    // Evaluate known pairs from the lower-index side to avoid critical-angle
    // cancellation. Sampled events keep the actual forward/reverse proposal masses.
    if (!sampledFacet)
        transmission = reverseTransmission =
            incidentIndex > transmittedIndex ? reverseTransmission : transmission;
    const float jacobian = -b / denominatorSquared;
    float value = transmission * GgxDistribution(normal, alpha) * GgxMaskingShadowing(view, light, alpha) *
                  (a * jacobian / (view.z * -light.z));
    if (mode == TransportRadiance)
        value *= rho * rho;
    return BsdfValue(value, transmission * GgxVisibleNormalPdf(view, normal, alpha) * jacobian,
                     reverseTransmission * GgxVisibleNormalPdf(-light, normal, alpha) * a * rho * rho /
                         denominatorSquared);
}
BsdfEvaluation EvaluateDielectric(float3 view, float3 light, float2 alpha, float incidentIndex,
                                  float transmittedIndex, uint mode)
{
    if (view.z <= 0.0 || light.z == 0.0 || any(alpha <= 0.0) || incidentIndex == transmittedIndex)
        return (BsdfEvaluation)0;
    if (light.z > 0.0)
    {
        const float3 normal = normalize(view + light);
        const float fresnel = DielectricFresnel(dot(view, normal), incidentIndex, transmittedIndex);
        return BsdfValue(fresnel * GgxReflectionFactor(view, light, alpha),
                         fresnel * GgxReflectionPdf(view, light, alpha),
                         fresnel * GgxReflectionPdf(light, view, alpha));
    }
    precise float3 halfVector = view * (incidentIndex / transmittedIndex) + light;
    const float squaredLength = dot(halfVector, halfVector);
    if (squaredLength == 0.0)
        return (BsdfEvaluation)0;
    float3 normal = halfVector / sqrt(squaredLength);
    if (normal.z < 0.0)
        normal = -normal;
    return DielectricTransmission(view, light, normal, alpha, incidentIndex, transmittedIndex, mode,
                                  squaredLength, false);
}
BsdfSample SampleDielectric(float3 view, float2 alpha, float incidentIndex, float transmittedIndex, uint mode,
                            float2 variates, float lobeVariate)
{
    if (view.z <= 0.0)
        return (BsdfSample)0;
    const bool delta = incidentIndex == transmittedIndex || all(alpha == 0.0);
    if (!delta && any(alpha <= 0.0))
        return (BsdfSample)0;
    const float3 normal = delta ? float3(0.0, 0.0, 1.0) : SampleGgxVisibleNormal(view, alpha, variates);
    const float cosine = dot(view, normal);
    const float fresnel = DielectricFresnel(cosine, incidentIndex, transmittedIndex);
    if (lobeVariate < fresnel)
    {
        const float3 light = 2.0 * cosine * normal - view;
        if (light.z <= 0.0)
            return (BsdfSample)0;
        if (delta)
            return BsdfEvent(light, BsdfValue(fresnel / light.z, fresnel, fresnel), 1.0,
                             ScatteringReflection | ScatteringDelta);
        return BsdfEvent(light, EvaluateDielectric(view, light, alpha, incidentIndex, transmittedIndex, mode),
                         1.0, ScatteringReflection | ScatteringGlossy);
    }
    const float rho = incidentIndex / transmittedIndex;
    float3 light;
    if (!RefractDirection(view, normal, rho, light) || light.z >= 0.0)
        return (BsdfSample)0;
    if (delta)
    {
        const float mass = 1.0 - fresnel;
        float value = mass / -light.z;
        if (mode == TransportRadiance)
            value *= rho * rho;
        const float reverseMass = 1.0 - DielectricFresnel(-light.z, transmittedIndex, incidentIndex);
        return BsdfEvent(light, BsdfValue(value, mass, reverseMass), transmittedIndex / incidentIndex,
                         ScatteringTransmission | ScatteringDelta);
    }
    // Rationalized Snell denominator: avoid subtracting almost equal cosines.
    precise float transmittedSquared = (1.0 - rho) * (1.0 + rho) + rho * rho * cosine * cosine;
    precise float numerator = (rho - 1.0) * (rho + 1.0);
    const float denominator = numerator / (rho * cosine + sqrt(transmittedSquared));
    return BsdfEvent(light,
                     DielectricTransmission(view, light, normal, alpha, incidentIndex, transmittedIndex, mode,
                                            denominator * denominator, true),
                     transmittedIndex / incidentIndex, ScatteringTransmission | ScatteringGlossy);
}
