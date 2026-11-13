// Native DXR conductor: GGX visible-normal reflection and the exact mirror limit.
// Fresnel uses relative complex index eta+i*k, independent of scene RGB tint.
// PBRT v4, "Conductor BRDF"; Heitz (2018), visible-normal sampling.
BsdfEvaluation EvaluateConductor(float3 view, float3 light, float2 alpha, float eta, float absorption)
{
    if (view.z <= 0.0 || light.z <= 0.0 || alpha.x <= 0.0 || alpha.y <= 0.0)
        return (BsdfEvaluation)0;
    float cosine = dot(view, normalize(view + light));
    return BsdfValue(ConductorFresnel(cosine, eta, absorption) * GgxReflectionFactor(view, light, alpha),
                     GgxReflectionPdf(view, light, alpha), GgxReflectionPdf(light, view, alpha));
}
BsdfSample SampleConductor(float3 view, float2 alpha, float eta, float absorption, float2 variates)
{
    if (view.z <= 0.0)
        return (BsdfSample)0;
    if (alpha.x == 0.0 && alpha.y == 0.0)
    {
        const float3 light = float3(-view.x, -view.y, view.z);
        const float value = ConductorFresnel(view.z, eta, absorption) / view.z;
        return BsdfEvent(light, BsdfValue(value, 1.0, 1.0), 1.0, ScatteringReflection | ScatteringDelta);
    }
    if (alpha.x <= 0.0 || alpha.y <= 0.0)
        return (BsdfSample)0;
    float3 facet = SampleGgxVisibleNormal(view, alpha, variates);
    float3 light = facet * (2.0 * dot(view, facet)) - view;
    if (light.z <= 0.0)
        return (BsdfSample)0;
    return BsdfEvent(light, EvaluateConductor(view, light, alpha, eta, absorption), 1.0,
                     ScatteringReflection | ScatteringGlossy);
}
