// Lambert reflectance with cosine or uniform hemisphere proposals. The BRDF
// stays rho/pi; reversing directions changes the cosine proposal's density.
static const float DiffusePi = 3.14159265358979323846;
BsdfEvaluation EvaluateDiffuse(float3 view, float3 light, float reflectance, bool importance)
{
    if (view.z <= 0.0 || light.z <= 0.0)
        return (BsdfEvaluation)0;
    return BsdfValue(reflectance / DiffusePi, importance ? light.z / DiffusePi : 1.0 / (2.0 * DiffusePi),
                     importance ? view.z / DiffusePi : 1.0 / (2.0 * DiffusePi));
}
BsdfSample SampleDiffuse(float3 view, float reflectance, float2 variates, bool importance)
{
    if (view.z <= 0.0)
        return (BsdfSample)0;
    const float z = importance ? sqrt(1.0 - variates.x) : 1.0 - variates.x;
    const float radius = sqrt(max(0.0, 1.0 - z * z));
    const float angle = 2.0 * DiffusePi * variates.y;
    const float3 light = float3(radius * cos(angle), radius * sin(angle), z);
    return BsdfEvent(light, EvaluateDiffuse(view, light, reflectance, importance), 1.0,
                     ScatteringReflection | ScatteringDiffuse);
}
