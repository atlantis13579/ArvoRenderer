// Native compute interior random walk. Chiang et al. (2016), "Practical and
// Controllable Subsurface Scattering for Production Path Tracing". Extinction
// is achromatic; authored RGB/spectral albedo controls collision survival.
struct SubsurfaceFlight { float3 Weight; float Distance; uint Scattered; };
struct SubsurfaceDirection { float3 Direction; float Pdf; float Weight; };
static const float SubsurfacePi = 3.14159265358979323846f;
static const float SubsurfaceOpenUniform = 1.0f - 1.0f / float(1u << 24);
static const float SubsurfaceBasisParallel = 0.999f;
SubsurfaceFlight SubsurfaceFreeFlight(float meanFreePath, float3 albedo, float3 direction,
                                      float endpoint, float variate)
{
    SubsurfaceFlight sample;
    sample.Weight = float3(1, 1, 1); sample.Distance = endpoint; sample.Scattered = 0;
    const float u = clamp(variate, 0.0f, SubsurfaceOpenUniform);
    // Four terms keep small uniforms from rounding 1-u to one in binary32.
    const float LogSeriesThreshold = 0.01f;
    const float opticalDepth = u < LogSeriesThreshold ? u * (1 + u * (1.0f/2 + u * (1.0f/3 + u/4))) : -log(1-u);
    const float metricDistance = opticalDepth * meanFreePath;
    const float speed = length(direction);
    if (metricDistance < endpoint * speed)
    {
        sample.Distance = metricDistance / speed;
        sample.Weight = albedo; sample.Scattered = 1;
    }
    return sample;
}
float SubsurfacePhase(float cosine, float g)
{
    const float magnitude = abs(g), aligned = g < 0 ? -cosine : cosine;
    const float denominator = (1-magnitude)*(1-magnitude) + 2*magnitude*(1-clamp(aligned, -1.0f, 1.0f));
    return (1 - g * g) / (4 * SubsurfacePi * denominator * sqrt(denominator));
}
SubsurfaceDirection SubsurfaceSampleDirection(float3 incoming, float g, float2 variates, bool importance)
{
    const float v = 2 * variates.x - 1;
    float cosine = v;
    if (importance)
    {
        // Cancel the HG inverse-CDF singularity at g=0 before evaluating it.
        const float d = 1 + g * v;
        cosine = ((1 + g * g) * v + g / 2 * (v * v + 3 + g * g * (v * v - 1))) / (d * d);
    }
    cosine = clamp(cosine, -1.0f, 1.0f);
    const float sine = sqrt(max(0.0f, 1 - cosine * cosine));
    const float phi = 2 * SubsurfacePi * variates.y;
    const float3 axis = normalize(incoming);
    const float3 helper = abs(axis.y) < SubsurfaceBasisParallel ? float3(0, 1, 0) : float3(1, 0, 0);
    const float3 tangent = normalize(cross(helper, axis)), bitangent = cross(axis, tangent);
    SubsurfaceDirection sample;
    sample.Direction = normalize(axis * cosine + tangent * (sine * cos(phi)) + bitangent * (sine * sin(phi)));
    const float phase = SubsurfacePhase(dot(axis, sample.Direction), g);
    sample.Pdf = importance ? phase : 1 / (4 * SubsurfacePi);
    sample.Weight = phase / sample.Pdf;
    return sample;
}
