// Independent D3D12 compute medium transport (PBRT v3, sections 11.3/15.2).
// Beer attenuation and RGB mixture free paths are evaluated in world length.
// HG directions use the incident propagation vector: positive g is forward.
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
static const float MediumLargestVariate = 0.9999999403953552f;
static const double MediumFourPi = 4 * 3.14159265358979323846;
bool MediumInterval(MediumParameters medium, float3 origin, float3 direction, float minimumDistance,
                    float maximumDistance, out float nearEnd, out float farEnd)
{
    nearEnd = max(0.0f, minimumDistance);
    farEnd = maximumDistance;
    if (medium.Enabled == 0)
        return false;
    for (uint axis = 0; axis < 3; ++axis)
    {
        if (direction[axis] == 0)
        {
            if (origin[axis] < medium.Minimum[axis] || origin[axis] > medium.Maximum[axis])
                return false;
        }
        else
        {
            float first = (medium.Minimum[axis] - origin[axis]) / direction[axis];
            float second = (medium.Maximum[axis] - origin[axis]) / direction[axis];
            nearEnd = max(nearEnd, min(first, second));
            farEnd = min(farEnd, max(first, second));
        }
    }
    return farEnd > nearEnd;
}
float3 MediumBeer(float3 extinction, float distance)
{
    return exp(-extinction * distance);
}
float3 MediumTransmission(MediumParameters medium, float3 origin, float3 direction, float minimumDistance,
                          float maximumDistance)
{
    float first, last;
    if (!MediumInterval(medium, origin, direction, minimumDistance, maximumDistance, first, last))
        return float3(1, 1, 1);
    return MediumBeer(medium.Absorption + medium.Scattering, (last - first) * length(direction));
}
MediumFlight MediumFreeFlight(MediumParameters medium, float3 origin, float3 direction, float minimumDistance,
                              float maximumDistance, float2 variates)
{
    MediumFlight result;
    result.Weight = float3(1, 1, 1);
    result.Distance = 0;
    result.Scattered = 0;
    float first, last;
    if (!MediumInterval(medium, origin, direction, minimumDistance, maximumDistance, first, last))
        return result;
    const float speed = length(direction);
    if (speed <= 0)
        return result;
    const float distanceLimit = (last - first) * speed;
    const float3 extinction = medium.Absorption + medium.Scattering;
    if (all(medium.Scattering == 0))
    {
        result.Weight = MediumBeer(extinction, distanceLimit);
        return result;
    }
    const uint ColorChannels = 3;
    const float2 unitVariates = clamp(variates, 0.0f, MediumLargestVariate);
    const uint channel = min(uint(unitVariates.x * ColorChannels), ColorChannels - 1);
    const float SeriesThreshold = 0.0001f;
    const float u = unitVariates.y;
    const float opticalDepth = u < SeriesThreshold ? u * (1 + u * (0.5f + u / 3)) : -log(1 - u);
    const float sampledDistance =
        extinction[channel] > 0 ? opticalDepth / extinction[channel] : distanceLimit;
    result.Scattered = sampledDistance < distanceLimit ? 1 : 0;
    const float travelled = min(sampledDistance, distanceLimit);
    result.Distance = first + travelled / speed;
    const float3 transmission = MediumBeer(extinction, travelled);
    const double density =
        result.Scattered != 0
            ? (double(extinction.x) * transmission.x + double(extinction.y) * transmission.y +
               double(extinction.z) * transmission.z) /
                  ColorChannels
            : (double(transmission.x) + transmission.y + transmission.z) / ColorChannels;
    const float3 scattering = result.Scattered != 0 ? medium.Scattering : float3(1, 1, 1);
    result.Weight = density > 0 ? float3(double(transmission.x) * scattering.x / density,
                                         double(transmission.y) * scattering.y / density,
                                         double(transmission.z) * scattering.z / density)
                                : float3(0, 0, 0);
    return result;
}
float MediumPhase(float cosine, float anisotropy)
{
    const float forward = clamp(cosine, -1.0f, 1.0f);
    const float poleDistance = 1 - abs(anisotropy);
    const float denominator =
        poleDistance * poleDistance + 2 * abs(anisotropy) * (1 - (anisotropy >= 0 ? forward : -forward));
    return (1 - anisotropy) * (1 + anisotropy) / (float(MediumFourPi) * denominator * sqrt(denominator));
}
float3 MediumSampleDirection(float3 incident, float anisotropy, float2 variates, bool importance,
                             out float pdf)
{
    const float g = importance ? anisotropy : 0;
    const float v = 2 * clamp(variates.x, 0.0f, MediumLargestVariate) - 1;
    const float divisor = 1 + g * v;
    // Factoring around the closer pole avoids cancellation at |g| near one.
    // Both branches retain the exact g=0 limit without division by g.
    const float poleDistance = 1 - abs(g);
    const float cosine = clamp(
        g >= 0 ? 1 - poleDistance * poleDistance * (1 - v) * (2 + g * (1 + v)) / (2 * divisor * divisor)
               : -1 + poleDistance * poleDistance * (1 + v) * (2 + g * (v - 1)) / (2 * divisor * divisor),
        -1.0f, 1.0f);
    const float azimuth = float(MediumFourPi / 2) * clamp(variates.y, 0.0f, MediumLargestVariate);
    const float radial = sqrt(max(0.0f, 1 - cosine * cosine));
    incident = normalize(incident);
    const float ParallelThreshold = 0.999f;
    const float3 helper = abs(incident.y) < ParallelThreshold ? float3(0, 1, 0) : float3(1, 0, 0);
    const float3 tangent = normalize(cross(helper, incident));
    const float3 direction = normalize(
        incident * cosine + radial * (tangent * cos(azimuth) + cross(incident, tangent) * sin(azimuth)));
    pdf = importance ? MediumPhase(dot(incident, direction), anisotropy) : float(1 / MediumFourPi);
    return direction;
}
