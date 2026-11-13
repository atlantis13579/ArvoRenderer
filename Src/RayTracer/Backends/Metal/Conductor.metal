// Metal-owned conductor BRDF. Complex Fresnel modulates single-scattering GGX;
// exact zero slopes choose a mirror with discrete probability one.
// https://pbr-book.org/4ed/Reflection_Models/Conductor_BRDF
BsdfEvaluation EvaluateConductor(float3 view, float3 light, float2 alpha, float eta, float absorption)
{
    if (view.z <= 0.0f || light.z <= 0.0f || alpha.x <= 0.0f || alpha.y <= 0.0f)
        return {};
    float3 normal = normalize(view + light);
    float fresnel = ConductorFresnel(dot(view, normal), eta, absorption);
    return {fresnel * GgxReflectionFactor(view, light, alpha), GgxReflectionPdf(view, light, alpha),
            GgxReflectionPdf(light, view, alpha)};
}
BsdfSample SampleConductor(float3 view, float2 alpha, float eta, float absorption, float2 variates)
{
    if (view.z <= 0.0f)
        return {};
    if (alpha.x == 0.0f && alpha.y == 0.0f)
        return {float3(-view.x, -view.y, view.z),
                ConductorFresnel(view.z, eta, absorption) / view.z,
                1.0f,
                1.0f,
                1.0f,
                ScatteringReflection | ScatteringDelta};
    if (alpha.x <= 0.0f || alpha.y <= 0.0f)
        return {};
    float3 normal = SampleGgxVisibleNormal(view, alpha, variates);
    float3 light = normal * (2.0f * dot(view, normal)) - view;
    if (light.z <= 0.0f)
        return {}; // Preserve the visible-normal proposal's null mass.
    BsdfEvaluation evaluated = EvaluateConductor(view, light, alpha, eta, absorption);
    return {light,         evaluated.Value,
            evaluated.Pdf, evaluated.ReversePdf,
            1.0f,          ScatteringReflection | ScatteringGlossy};
}
