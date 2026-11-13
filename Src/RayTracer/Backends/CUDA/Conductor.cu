namespace ArvoCuda
{
// Single-scattering GGX with exact complex-index Fresnel, or a discrete mirror
// at alpha=(0,0). A below-surface VNDF reflection is a null event; do not retry.
// https://pbr-book.org/4ed/Reflection_Models/Conductor_BRDF
__device__ BsdfEvaluation EvaluateConductor(float3 view, float3 light, float2 alpha, float eta,
                                            float absorption)
{
    if (view.z <= 0.0f || light.z <= 0.0f || alpha.x <= 0.0f || alpha.y <= 0.0f)
        return {};
    float3 normal = GgxUnit(BsdfSum(view, light));
    float fresnel = ConductorFresnel(BsdfDot(view, normal), eta, absorption);
    return {fresnel * GgxReflectionFactor(view, light, alpha), GgxReflectionPdf(view, light, alpha),
            GgxReflectionPdf(light, view, alpha)};
}
__device__ BsdfSample SampleConductor(float3 view, float2 alpha, float eta, float absorption, float2 variates)
{
    if (view.z <= 0.0f)
        return {};
    if (alpha.x == 0.0f && alpha.y == 0.0f)
        return {make_float3(-view.x, -view.y, view.z),
                ConductorFresnel(view.z, eta, absorption) / view.z,
                1.0f,
                1.0f,
                1.0f,
                ScatteringReflection | ScatteringDelta};
    if (alpha.x <= 0.0f || alpha.y <= 0.0f)
        return {};
    float3 normal = SampleGgxVisibleNormal(view, alpha, variates);
    float3 light = BsdfSum(BsdfScale(normal, 2.0f * BsdfDot(view, normal)), BsdfScale(view, -1.0f));
    if (light.z <= 0.0f)
        return {};
    BsdfEvaluation evaluated = EvaluateConductor(view, light, alpha, eta, absorption);
    return {light,         evaluated.Value,
            evaluated.Pdf, evaluated.ReversePdf,
            1.0f,          ScatteringReflection | ScatteringGlossy};
}
} // namespace ArvoCuda
