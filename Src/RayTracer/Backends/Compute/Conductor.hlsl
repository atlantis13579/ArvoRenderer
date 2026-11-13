// Native compute GGX conductor. The exact (0,0) slope limit is a discrete mirror;
// continuous evaluation excludes that measure. eta and k are relative indices.
// https://pbr-book.org/4ed/Reflection_Models/Conductor_BRDF
BsdfEvaluation EvaluateConductor(float3 view, float3 light, float2 alpha, float eta, float absorption)
{
    if (view.z <= 0.0 || light.z <= 0.0 || any(alpha <= 0.0))
        return (BsdfEvaluation)0;
    const float3 halfVector = normalize(view + light);
    const float reflectance = ConductorFresnel(dot(view, halfVector), eta, absorption);
    return BsdfValue(reflectance * GgxReflectionFactor(view, light, alpha),
                     GgxReflectionPdf(view, light, alpha), GgxReflectionPdf(light, view, alpha));
}
BsdfSample SampleConductor(float3 view, float2 alpha, float eta, float absorption, float2 variates)
{
    if (view.z <= 0.0)
        return (BsdfSample)0;
    if (all(alpha == 0.0))
        return BsdfEvent(float3(-view.xy, view.z),
                         BsdfValue(ConductorFresnel(view.z, eta, absorption) / view.z, 1.0, 1.0), 1.0,
                         ScatteringReflection | ScatteringDelta);
    if (any(alpha <= 0.0))
        return (BsdfSample)0;
    const float3 normal = SampleGgxVisibleNormal(view, alpha, variates);
    const float3 light = 2.0 * dot(view, normal) * normal - view;
    if (light.z <= 0.0)
        return (BsdfSample)0;
    return BsdfEvent(light, EvaluateConductor(view, light, alpha, eta, absorption), 1.0,
                     ScatteringReflection | ScatteringGlossy);
}
