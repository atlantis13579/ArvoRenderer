// Independent Metal homogeneous transport. PBRT v3, "Sampling Volume Scattering".
// Mixture densities are normalized before summation: MSL has no binary64.
// Forward HG scattering is measured along the incident propagation vector.
#include <metal_stdlib>
using namespace metal;
struct MediumParameters
{
    packed_float3 Minimum;
    uint Enabled;
    packed_float3 Maximum;
    float Anisotropy;
    packed_float3 Absorption;
    uint Padding0;
    packed_float3 Scattering;
    uint Padding1;
};
struct MediumFlight
{
    float3 Weight;
    float Distance;
    uint Scattered;
};
static_assert(sizeof(MediumParameters) == 64, "Metal medium ABI");
constant float MediumOpenVariate = 0.9999999403953552f;
constant float MediumFourPi = 4 * 3.14159265358979323846f;
bool MediumClip(MediumParameters medium, float3 origin, float3 direction, float minimum, float maximum,
                thread float& first, thread float& last)
{
    first = max(0.0f, minimum);
    last = maximum;
    if (medium.Enabled == 0)
        return false;
    const float3 low = float3(medium.Minimum), high = float3(medium.Maximum);
    for (uint axis = 0; axis < 3; ++axis)
    {
        if (direction[axis] == 0)
        {
            if (origin[axis] < low[axis] || origin[axis] > high[axis])
                return false;
            continue;
        }
        const float a = (low[axis] - origin[axis]) / direction[axis];
        const float b = (high[axis] - origin[axis]) / direction[axis];
        first = max(first, min(a, b));
        last = min(last, max(a, b));
    }
    return last > first;
}
float3 MediumTransmission(MediumParameters medium, float3 origin, float3 direction, float minimum,
                          float maximum)
{
    float first, last;
    if (!MediumClip(medium, origin, direction, minimum, maximum, first, last))
        return float3(1);
    const float3 extinction = float3(medium.Absorption) + float3(medium.Scattering);
    return exp(-extinction * ((last - first) * length(direction)));
}
MediumFlight MediumFreeFlight(MediumParameters medium, float3 origin, float3 direction, float minimum,
                              float maximum, float2 random)
{
    MediumFlight result{float3(1), 0, 0};
    float first, last;
    if (!MediumClip(medium, origin, direction, minimum, maximum, first, last))
        return result;
    const float speed = length(direction);
    if (speed <= 0)
        return result;
    const float limit = (last - first) * speed;
    const float3 scattering = float3(medium.Scattering);
    const float3 extinction = float3(medium.Absorption) + scattering;
    if (all(scattering == 0))
    {
        result.Weight = exp(-extinction * limit);
        return result;
    }
    const uint Channels = 3;
    const float2 uniform = clamp(random, 0.0f, MediumOpenVariate);
    const uint component = min(uint(uniform.x * Channels), Channels - 1);
    // log1p is not assumed available in MSL. This local series prevents 1-u
    // from rounding to one for very small variates (the omitted term is O(u^4)).
    const float SeriesThreshold = 0.0001f;
    const float u = uniform.y;
    const float optical = u < SeriesThreshold ? u * (1 + u * (0.5f + u / 3)) : -log(1 - u);
    const float distance = extinction[component] > 0 ? optical / extinction[component] : limit;
    result.Scattered = distance < limit;
    const float travelled = min(distance, limit);
    result.Distance = first + travelled / speed;
    const float3 beam = exp(-extinction * travelled);
    if (result.Scattered)
    {
        const float scale = max(extinction.x, max(extinction.y, extinction.z));
        const float3 normalized = extinction / scale;
        const float density =
            (normalized.x * beam.x + normalized.y * beam.y + normalized.z * beam.z) / Channels;
        result.Weight = density > 0 ? (scattering / scale) * beam / density : float3(0);
    }
    else
    {
        const float mass = (beam.x + beam.y + beam.z) / Channels;
        result.Weight = mass > 0 ? beam / mass : float3(0);
    }
    return result;
}
float MediumPhase(float cosine, float g)
{
    const float forward = clamp(cosine, -1.0f, 1.0f), distanceToPole = 1 - abs(g);
    const float denominator =
        distanceToPole * distanceToPole + 2 * abs(g) * (1 - (g >= 0 ? forward : -forward));
    return ((1 - g) * (1 + g) / MediumFourPi) / (denominator * sqrt(denominator));
}
float3 MediumSampleDirection(float3 incident, float anisotropy, float2 random, bool importance,
                             thread float& pdf)
{
    const float g = importance ? anisotropy : 0;
    const float u = 2 * clamp(random.x, 0.0f, MediumOpenVariate) - 1;
    const float d = 1 + g * u, distanceToPole = 1 - abs(g);
    // Factoring the nearer pole keeps float arithmetic stable for extreme g.
    const float tail = distanceToPole * distanceToPole / (2 * d * d);
    const float cosine =
        clamp(g >= 0 ? 1 - tail * (1 - u) * (2 + g * (1 + u)) : -1 + tail * (1 + u) * (2 + g * (u - 1)),
              -1.0f, 1.0f);
    const float phi = MediumFourPi / 2 * clamp(random.y, 0.0f, MediumOpenVariate);
    const float radial = sqrt(max(0.0f, 1 - cosine * cosine));
    incident = normalize(incident);
    const float ParallelThreshold = 0.999f;
    const float3 reference = abs(incident.y) < ParallelThreshold ? float3(0, 1, 0) : float3(1, 0, 0);
    const float3 tangent = normalize(cross(reference, incident));
    const float3 direction =
        normalize(incident * cosine + radial * (tangent * cos(phi) + cross(incident, tangent) * sin(phi)));
    pdf = importance ? MediumPhase(dot(incident, direction), anisotropy) : 1 / MediumFourPi;
    return direction;
}
