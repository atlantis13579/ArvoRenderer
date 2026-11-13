// Native Metal Lambertian scattering with an independently chosen proposal.
// The reverse density uses view.z, because both directions point away from the hit.
constant float DiffusePi = 3.14159265358979323846f;
BsdfEvaluation EvaluateDiffuse(float3 view, float3 light, float reflectance, bool importance)
{
    if (view.z <= 0.0f || light.z <= 0.0f)
        return {};
    return {reflectance / DiffusePi, importance ? light.z / DiffusePi : 1.0f / (2.0f * DiffusePi),
            importance ? view.z / DiffusePi : 1.0f / (2.0f * DiffusePi)};
}
BsdfSample SampleDiffuse(float3 view, float reflectance, float2 variates, bool importance)
{
    if (view.z <= 0.0f)
        return {};
    float cosine = importance ? sqrt(1.0f - variates.x) : 1.0f - variates.x;
    float radial = sqrt(max(0.0f, 1.0f - cosine * cosine));
    float phi = 2.0f * DiffusePi * variates.y;
    float3 light = float3(radial * cos(phi), radial * sin(phi), cosine);
    BsdfEvaluation evaluated = EvaluateDiffuse(view, light, reflectance, importance);
    return {light,         evaluated.Value,
            evaluated.Pdf, evaluated.ReversePdf,
            1.0f,          ScatteringReflection | ScatteringDiffuse};
}
