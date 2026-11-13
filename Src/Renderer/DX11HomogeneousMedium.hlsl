// Independent DX11 preview of the integral equation of transfer (PBRT v4, 14.1).
// https://pbr-book.org/4ed/Light_Transport_II_Volume_Rendering/The_Equation_of_Transfer
// Analytic Beer attenuation plus single scattering on primary camera segments; no offline
// integrator is included. Optical-distance quadrature resolves dense foreground
// layers without a world-space step-size parameter. Geometry visibility uses
// the preview shadow map; environment occlusion and multiple scattering need DXR.
struct RasterMedium
{
    float3 Minimum;
    uint Enabled;
    float3 Maximum;
    float Anisotropy;
    float3 Absorption;
    float3 Scattering;
};
struct RasterMediumLight
{
    float3 Position;
    uint Type;
    float3 Direction;
    float Intensity;
    float3 Color;
    float2 Size;
};
// The render pipeline supplies scene records and shadow-map access, not transport.
RasterMediumLight RasterGetMediumLight(uint index);
float RasterMediumVisibility(uint index, float3 position);
float3 RasterMediumSky(float3 direction);

static const float RasterMediumPi = 3.14159265358979323846f;
static const float RasterMediumRayLimit = 1.0e30f;
static const uint RasterMediumDistanceSamples = 16;
static const uint RasterMediumEnvironmentSamples = 8;
static const uint RasterMediumAreaSide = 2;
static const float RasterMediumMinimumLightDistance = 1.0e-4f;

float2 RasterMediumInterval(RasterMedium medium, float3 origin, float3 direction, float maximum)
{
    float first = 0, last = maximum;
    if (medium.Enabled == 0)
        return float2(0, 0);
    bool intersects = true;
    for (uint axis = 0; axis < 3; ++axis)
    {
        if (direction[axis] == 0)
        {
            if (origin[axis] < medium.Minimum[axis] || origin[axis] > medium.Maximum[axis])
                intersects = false;
        }
        else
        {
            const float lower = (medium.Minimum[axis] - origin[axis]) / direction[axis];
            const float upper = (medium.Maximum[axis] - origin[axis]) / direction[axis];
            first = max(first, min(lower, upper));
            last = min(last, max(lower, upper));
        }
    }
    return intersects && last > first ? float2(first, last) : float2(0, 0);
}
float3 RasterMediumTransmission(RasterMedium medium, float3 origin, float3 direction, float maximum)
{
    const float2 interval = RasterMediumInterval(medium, origin, direction, maximum);
    const float worldLength = (interval.y - interval.x) * length(direction);
    return exp(-(medium.Absorption + medium.Scattering) * worldLength);
}
float RasterMediumPhase(float cosine, float anisotropy)
{
    const float pole = 1 - abs(anisotropy);
    const float forward = clamp(cosine, -1.0f, 1.0f);
    const float denominator =
        pole * pole + 2 * abs(anisotropy) * (1 - (anisotropy >= 0 ? forward : -forward));
    return (1 - anisotropy) * (1 + anisotropy) / (4 * RasterMediumPi * denominator * sqrt(denominator));
}
float3 RasterMediumEnvironmentDirection(float3 incoming, float anisotropy, uint index)
{
    const float coordinate = 2 * (index + 0.5f) / RasterMediumEnvironmentSamples - 1;
    const float denominator = 1 + anisotropy * coordinate;
    const float pole = 1 - abs(anisotropy);
    const float tail = pole * pole / (2 * denominator * denominator);
    const float cosine =
        clamp(anisotropy >= 0 ? 1 - tail * (1 - coordinate) * (2 + anisotropy * (1 + coordinate))
                              : -1 + tail * (1 + coordinate) * (2 + anisotropy * (coordinate - 1)),
              -1.0f, 1.0f);
    const float GoldenRotation = 0.6180339887498948482f;
    const float rotation = (index + 0.5f) * GoldenRotation;
    const float azimuth = 2 * RasterMediumPi * (rotation - floor(rotation));
    const float radial = sqrt(max(0.0f, 1 - cosine * cosine));
    const float ParallelThreshold = 0.999f;
    const float3 poleAxis = abs(incoming.y) < ParallelThreshold ? float3(0, 1, 0) : float3(1, 0, 0);
    const float3 tangent = normalize(cross(poleAxis, incoming));
    return normalize(incoming * cosine + tangent * (radial * cos(azimuth)) +
                     cross(incoming, tangent) * (radial * sin(azimuth)));
}
float3 RasterMediumLightConnection(RasterMedium medium, float3 position, float3 incoming, float3 direction,
                                   float distance, float3 radiance)
{
    return radiance * RasterMediumTransmission(medium, position, direction, distance) *
           RasterMediumPhase(dot(incoming, direction), medium.Anisotropy);
}
float3 RasterMediumIncident(RasterMedium medium, float3 position, float3 incoming, uint lightCount)
{
    float3 result = float3(0, 0, 0);
    [loop]
    for (uint lightIndex = 0; lightIndex < lightCount; ++lightIndex)
    {
        const RasterMediumLight light = RasterGetMediumLight(lightIndex);
        if (light.Intensity <= 0)
            continue;
        const float3 energy = max(light.Color, float3(0, 0, 0)) * light.Intensity *
                              RasterMediumVisibility(lightIndex, position);
        const uint DirectionalLight = 0, PointLight = 1;
        if (light.Type == DirectionalLight)
        {
            const float directionLength = length(light.Direction);
            if (directionLength > 0)
                result +=
                    RasterMediumLightConnection(medium, position, incoming, light.Direction / directionLength,
                                                RasterMediumRayLimit, energy);
            continue;
        }
        if (light.Type == PointLight)
        {
            const float3 delta = light.Position - position;
            const float distance = length(delta);
            if (distance > RasterMediumMinimumLightDistance)
                result += RasterMediumLightConnection(medium, position, incoming, delta / distance, distance,
                                                      energy / (distance * distance));
            continue;
        }
        if (light.Size.x <= 0 || light.Size.y <= 0 || length(light.Direction) == 0)
            continue;
        const float3 normal = normalize(light.Direction);
        const float ParallelThreshold = 0.999f;
        const float3 helper = abs(normal.y) < ParallelThreshold ? float3(0, 1, 0) : float3(1, 0, 0);
        const float3 tangent = normalize(cross(helper, normal)), bitangent = cross(normal, tangent);
        for (uint y = 0; y < RasterMediumAreaSide; ++y)
            for (uint x = 0; x < RasterMediumAreaSide; ++x)
            {
                const float u = (x + 0.5f) / RasterMediumAreaSide - 0.5f;
                const float v = (y + 0.5f) / RasterMediumAreaSide - 0.5f;
                const float3 offset =
                    light.Position + tangent * (u * light.Size.x) + bitangent * (v * light.Size.y) - position;
                const float distance = length(offset);
                if (distance <= RasterMediumMinimumLightDistance)
                    continue;
                const float3 direction = offset / distance;
                const float cosine = max(0.0f, dot(normal, -direction));
                const float weight =
                    cosine / (distance * distance * RasterMediumAreaSide * RasterMediumAreaSide);
                result += RasterMediumLightConnection(medium, position, incoming, direction, distance,
                                                      energy * weight);
            }
    }
    [loop]
    for (uint index = 0; index < RasterMediumEnvironmentSamples; ++index)
    {
        const float3 direction = RasterMediumEnvironmentDirection(incoming, medium.Anisotropy, index);
        // The HG proposal cancels its phase value. The remaining approximation
        // is deterministic angular quadrature without scene/environment visibility.
        result += RasterMediumSky(direction) *
                  RasterMediumTransmission(medium, position, direction, RasterMediumRayLimit) /
                  RasterMediumEnvironmentSamples;
    }
    return result;
}
float RasterMediumCollisionMass(float opticalDepth)
{
    const float SeriesThreshold = 0.001f;
    return opticalDepth < SeriesThreshold ? opticalDepth * (1 - opticalDepth * (0.5f - opticalDepth / 6))
                                          : 1 - exp(-opticalDepth);
}
float RasterMediumOpticalDistance(float probability)
{
    const float SeriesThreshold = 0.0001f;
    return probability < SeriesThreshold ? probability * (1 + probability * (0.5f + probability / 3))
                                         : -log(1 - probability);
}
float3 RasterIntegrateMedium(RasterMedium medium, float3 origin, float3 direction, float maximum,
                             float3 boundaryRadiance, uint lightCount)
{
    if (medium.Enabled == 0)
        return boundaryRadiance;
    const float speed = length(direction);
    if (speed <= 0)
        return boundaryRadiance;
    const float2 interval = RasterMediumInterval(medium, origin, direction, maximum);
    if (interval.y <= interval.x)
        return boundaryRadiance;
    const float worldLength = (interval.y - interval.x) * speed;
    const float3 start = origin + direction * interval.x;
    direction /= speed;
    const float3 extinction = medium.Absorption + medium.Scattering;
    float3 result = boundaryRadiance * exp(-extinction * worldLength);
    if (all(medium.Scattering == 0))
        return result;
    const bool commonExtinction = extinction.x == extinction.y && extinction.x == extinction.z;
    const uint Channels = 3;
    [loop]
    for (uint channel = 0; channel < (commonExtinction ? 1u : Channels); ++channel)
    {
        const float sigma = extinction[channel];
        if (sigma <= 0 || (!commonExtinction && medium.Scattering[channel] == 0))
            continue;
        const float mass = RasterMediumCollisionMass(sigma * worldLength);
        float3 incident = float3(0, 0, 0);
        [loop]
        for (uint sampleIndex = 0; sampleIndex < RasterMediumDistanceSamples; ++sampleIndex)
        {
            const float u = (sampleIndex + 0.5f) / RasterMediumDistanceSamples;
            const float distance = RasterMediumOpticalDistance(u * mass) / sigma;
            incident += RasterMediumIncident(medium, start + direction * distance, direction, lightCount);
        }
        const float weight = mass / RasterMediumDistanceSamples;
        if (commonExtinction)
            result += incident * (medium.Scattering / sigma) * weight;
        else
        {
            const float contribution = incident[channel] * (medium.Scattering[channel] / sigma) * weight;
            // Shader Model 5 cannot dynamically address a vector component for writing.
            if (channel == 0)
                result.x += contribution;
            else if (channel == 1)
                result.y += contribution;
            else
                result.z += contribution;
        }
    }
    return result;
}
