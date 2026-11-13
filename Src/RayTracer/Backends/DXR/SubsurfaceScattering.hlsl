// DXR owns this free-flight and phase estimator for offline subsurface paths.
// Preview substitutes diffuse albedo. Geometry bounds the interior; no thickness texture
// or diffusion-profile asset is required. See Chiang et al., SIGGRAPH 2016.
struct SubsurfaceFlight { float3 Weight; float Distance; uint Scattered; };
struct SubsurfaceDirection { float3 Direction; float Pdf; float Weight; };
static const float SubsurfaceFourPi = 4 * 3.14159265358979323846f;
static const float SubsurfaceLargestUniform = 1 - 1.0f / float(1u << 24);
static const float SubsurfaceParallelThreshold = 0.999f;
SubsurfaceFlight SubsurfaceFreeFlight(float meanFreePath, float3 albedo, float3 direction,
                                      float endpoint, float variate)
{
    const float speed = sqrt(dot(direction, direction));
    const float u = min(max(variate, 0.0f), SubsurfaceLargestUniform);
    const float SmallLogArgument = .01f;
    const float opticalDistance = u < SmallLogArgument ? u + u*u/2 + u*u*u/3 + u*u*u*u/4 : -log(1-u);
    const float distance = opticalDistance * meanFreePath / speed;
    SubsurfaceFlight result;
    result.Scattered = distance < endpoint ? 1 : 0;
    result.Distance = result.Scattered != 0 ? distance : endpoint;
    result.Weight = result.Scattered != 0 ? albedo : float3(1, 1, 1);
    return result;
}
float SubsurfacePhase(float cosine, float anisotropy)
{
    const float g = anisotropy;
    const float magnitude = abs(g), aligned = g < 0 ? -cosine : cosine;
    const float squaredLength = (1-magnitude)*(1-magnitude) + 2*magnitude*(1-clamp(aligned, -1.0f, 1.0f));
    return (1 - g * g) / (SubsurfaceFourPi * squaredLength * sqrt(squaredLength));
}
SubsurfaceDirection SubsurfaceSampleDirection(float3 incoming, float anisotropy, float2 variates,
                                               bool importance)
{
    const float g = anisotropy, centered = variates.x * 2 - 1;
    const float denominator = 1 + g * centered;
    const float numerator = centered * (1 + g * g) +
        (g / 2) * (centered * centered * (1 + g * g) + 3 - g * g);
    const float cosine = clamp(importance ? numerator / (denominator * denominator) : centered, -1.0f, 1.0f);
    const float radius = sqrt(max(0.0f, (1 - cosine) * (1 + cosine)));
    const float angle = (SubsurfaceFourPi / 2) * variates.y;
    const float3 forward = normalize(incoming);
    const float3 auxiliary = abs(forward.z) < SubsurfaceParallelThreshold ? float3(0, 0, 1) : float3(0, 1, 0);
    const float3 right = normalize(cross(auxiliary, forward)), up = cross(forward, right);
    SubsurfaceDirection result;
    result.Direction = normalize(forward * cosine + right * (radius * cos(angle)) + up * (radius * sin(angle)));
    const float evaluated = SubsurfacePhase(dot(forward, result.Direction), g);
    result.Pdf = importance ? evaluated : 1 / SubsurfaceFourPi;
    result.Weight = evaluated / result.Pdf;
    return result;
}
