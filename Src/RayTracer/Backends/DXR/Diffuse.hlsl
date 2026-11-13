// DXR Lambertian scattering. Uniform and cosine proposals estimate the same
// reflectance; neither changes the physical f=rho/pi used by light sampling.
static const float DiffusePi = 3.14159265358979323846;
BsdfEvaluation EvaluateDiffuse(float3 view, float3 light, float reflectance, bool importance)
{
    if (view.z <= 0.0 || light.z <= 0.0)
        return (BsdfEvaluation)0;
    float forward = 1.0 / (2.0 * DiffusePi), reverse = forward;
    if (importance)
    {
        forward = light.z / DiffusePi;
        reverse = view.z / DiffusePi;
    }
    return BsdfValue(reflectance / DiffusePi, forward, reverse);
}
BsdfSample SampleDiffuse(float3 view, float reflectance, float2 variates, bool importance)
{
    if (view.z <= 0.0)
        return (BsdfSample)0;
    const float cosine = importance ? sqrt(1.0 - variates.x) : 1.0 - variates.x;
    const float azimuth = 2.0 * DiffusePi * variates.y;
    const float radial = sqrt(max(0.0, 1.0 - cosine * cosine));
    float3 light = float3(cos(azimuth) * radial, sin(azimuth) * radial, cosine);
    return BsdfEvent(light, EvaluateDiffuse(view, light, reflectance, importance), 1.0,
                     ScatteringReflection | ScatteringDiffuse);
}
