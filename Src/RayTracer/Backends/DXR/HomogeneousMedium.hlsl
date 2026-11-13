// Native DXR homogeneous-region transport, used by both preview and offline.
// PBRT v3, "Sampling Volume Scattering": the endpoint and medium events have
// different measures; both use the average RGB proposal density, never one channel.
struct MediumParameters
{
    float3 Minimum;
    uint Enabled;
    float3 Maximum;
    float Anisotropy;
    float3 Absorption;
    uint Padding0;
    float3 Scattering;
    uint Padding1;
};
struct MediumFlight
{
    float3 Weight;
    float Distance;
    uint Scattered;
};
static const float MediumOpenUnit = 0.9999999403953552f;
static const float MediumInvFourPi = 1 / (4 * 3.14159265358979323846f);
bool MediumClip(MediumParameters region, float3 start, float3 velocity, float minimum, float maximum,
                out float enter, out float leave)
{
    enter = max(0.0f, minimum);
    leave = maximum;
    if (region.Enabled == 0)
        return false;
    for (uint coordinate = 0; coordinate < 3; ++coordinate)
    {
        if (velocity[coordinate] == 0)
        {
            if (start[coordinate] < region.Minimum[coordinate] ||
                start[coordinate] > region.Maximum[coordinate])
                return false;
            continue;
        }
        const float t0 = (region.Minimum[coordinate] - start[coordinate]) / velocity[coordinate];
        const float t1 = (region.Maximum[coordinate] - start[coordinate]) / velocity[coordinate];
        enter = max(enter, min(t0, t1));
        leave = min(leave, max(t0, t1));
    }
    return leave > enter;
}
float3 MediumTransmission(MediumParameters region, float3 start, float3 velocity, float minimum,
                          float maximum)
{
    float enter, leave;
    if (!MediumClip(region, start, velocity, minimum, maximum, enter, leave))
        return float3(1, 1, 1);
    const float travel = (leave - enter) * length(velocity);
    return exp(-(region.Absorption + region.Scattering) * travel);
}
MediumFlight MediumFreeFlight(MediumParameters region, float3 start, float3 velocity, float minimum,
                              float maximum, float2 random)
{
    MediumFlight event;
    event.Weight = float3(1, 1, 1);
    event.Distance = 0;
    event.Scattered = 0;
    float enter, leave;
    if (!MediumClip(region, start, velocity, minimum, maximum, enter, leave))
        return event;
    const float speed = length(velocity);
    if (!(speed > 0))
        return event;
    const float limit = (leave - enter) * speed;
    const float3 sigma = region.Absorption + region.Scattering;
    if (all(region.Scattering == 0))
    {
        event.Weight = exp(-sigma * limit);
        return event;
    }
    const uint Components = 3;
    const uint component = min(uint(clamp(random.x, 0.0f, MediumOpenUnit) * Components), Components - 1);
    const float u = clamp(random.y, 0.0f, MediumOpenUnit);
    const float SeriesThreshold = 0.0001f;
    const float optical = u < SeriesThreshold ? u * (1 + u * (0.5f + u / 3)) : -log(1 - u);
    const float freePath = sigma[component] == 0 ? limit : optical / sigma[component];
    event.Scattered = freePath < limit ? 1 : 0;
    const float travel = min(freePath, limit);
    event.Distance = enter + travel / speed;
    const float3 beam = exp(-sigma * travel);
    double proposal = (double(beam.x) + beam.y + beam.z) / Components;
    if (event.Scattered != 0)
        proposal =
            (double(beam.x) * sigma.x + double(beam.y) * sigma.y + double(beam.z) * sigma.z) / Components;
    const float3 coefficient = event.Scattered != 0 ? region.Scattering : float3(1, 1, 1);
    event.Weight = proposal > 0 ? float3(double(beam.x) * coefficient.x / proposal,
                                         double(beam.y) * coefficient.y / proposal,
                                         double(beam.z) * coefficient.z / proposal)
                                : float3(0, 0, 0);
    return event;
}
float MediumPhase(float forwardCosine, float g)
{
    const float cosine = clamp(forwardCosine, -1.0f, 1.0f);
    const float distanceToPole = 1 - abs(g);
    const float denominator =
        distanceToPole * distanceToPole + 2 * abs(g) * (1 - (g >= 0 ? cosine : -cosine));
    return MediumInvFourPi * (1 - g) * (1 + g) / (denominator * sqrt(denominator));
}
float3 MediumSampleDirection(float3 propagation, float anisotropy, float2 random, bool importance,
                             out float pdf)
{
    const float g = importance ? anisotropy : 0;
    const float coordinate = 2 * clamp(random.x, 0.0f, MediumOpenUnit) - 1;
    const float denominator = 1 + g * coordinate;
    // Factored inverse CDF is stable at both isotropy and strongly peaked lobes.
    const float distanceToPole = 1 - abs(g);
    const float tail = distanceToPole * distanceToPole / (2 * denominator * denominator);
    const float cosine = clamp(g >= 0 ? 1 - tail * (1 - coordinate) * (2 + g * (1 + coordinate))
                                      : -1 + tail * (1 + coordinate) * (2 + g * (coordinate - 1)),
                               -1.0f, 1.0f);
    const float azimuth = 2 * 3.14159265358979323846f * clamp(random.y, 0.0f, MediumOpenUnit);
    const float sine = sqrt(max(0.0f, 1 - cosine * cosine));
    const float PoleThreshold = 0.999f;
    propagation = normalize(propagation);
    const float3 pole = abs(propagation.y) < PoleThreshold ? float3(0, 1, 0) : float3(1, 0, 0);
    const float3 x = normalize(cross(pole, propagation));
    const float3 direction = normalize(propagation * cosine + x * (sine * cos(azimuth)) +
                                       cross(propagation, x) * (sine * sin(azimuth)));
    pdf = importance ? MediumPhase(dot(propagation, direction), anisotropy) : MediumInvFourPi;
    return direction;
}
