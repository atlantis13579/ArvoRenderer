namespace ArvoCuda
{
// Native CUDA Lambertian BSDF. A uniform proposal still uses the same rho/pi;
// its f*cos/pdf correction is left explicit in the returned sample.
constexpr float DiffusePi = 3.14159265358979323846f;
__device__ BsdfEvaluation EvaluateDiffuse(float3 view, float3 light, float reflectance, bool importance)
{
    if (view.z <= 0.0f || light.z <= 0.0f)
        return {};
    const float uniform = 1.0f / (2.0f * DiffusePi);
    return {reflectance / DiffusePi, importance ? light.z / DiffusePi : uniform,
            importance ? view.z / DiffusePi : uniform};
}
__device__ BsdfSample SampleDiffuse(float3 view, float reflectance, float2 variates, bool importance)
{
    if (view.z <= 0.0f)
        return {};
    float cosine = importance ? sqrtf(1.0f - variates.x) : 1.0f - variates.x;
    float radial = sqrtf(fmaxf(0.0f, 1.0f - cosine * cosine));
    float phi = 2.0f * DiffusePi * variates.y;
    float3 light = make_float3(radial * cosf(phi), radial * sinf(phi), cosine);
    BsdfEvaluation evaluated = EvaluateDiffuse(view, light, reflectance, importance);
    return {light,         evaluated.Value,
            evaluated.Pdf, evaluated.ReversePdf,
            1.0f,          ScatteringReflection | ScatteringDiffuse};
}
} // namespace ArvoCuda
