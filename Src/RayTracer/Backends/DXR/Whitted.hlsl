// Whitted (1980), "An Improved Illumination Model for Shaded Display".
// Independent DXR direct quadrature and ideal reflection/transmission. RayGen
// walks its own branch stack; TraceRay only computes intersections, so optical
// depth does not consume the driver's recursive ray-pipeline depth.
static const float Pi = 3.14159265358979323846f;
static const float MaximumRayDistance = 1.0e30;
static const float BasisParallelThreshold = 0.999;
static const uint AreaLightSamples = 4;
static const uint AreaLightStrataPerAxis = 2;
static const uint SpecularBranchCount = 2;
static const uint ImportanceBit = 1u << 0;
static const uint PixelSeedMultiplier = 9781u;
static const uint SampleSeedMultiplier = 6271u;
static const uint BounceSeedMultiplier = 0x9e3779b9u;
static const uint LightSeedMultiplier = 0x85ebca6bu;
static const uint LightSampleSeedMultiplier = 0xc2b2ae35u;
static const uint SecondDimensionScramble = 0x68bc21ebu;
static const uint LobeSelectionScramble = 0xa511e9b3u;
static const uint AvalancheMultiplierA = 0x7feb352du;
static const uint AvalancheMultiplierB = 0x846ca68bu;
static const uint RandomMantissaMask = 0x00ffffffu;
static const float RandomMantissaRange = 16777216.0;
static const float MinimumDirectionLengthSquared = 1.0e-12;

static const uint DirectionalLightType = 0u;
static const uint PointLightType = 1u;
static const uint RectAreaLightType = 2u;
static const float RayEpsilon = 1.0e-4;

#include "Surface.hlsl"
#include "Limits.h"

#include "MaterialTypes.hlsl"

struct RayTextureData
{
    uint OffsetTexels;
    uint Width;
    uint Height;
    uint Padding;
};

struct LightData
{
    float3 Position;
    uint Type;
    float3 Direction;
    float Intensity;
    float3 Color;
    uint CastShadows;
    float Width;
    float Height;
    uint Padding0;
    uint Padding1;
};

#include "FilmMoments.hlsl"
RWStructuredBuffer<LinearFilmPixel> LinearFilm : register(u2);
RWStructuredBuffer<uint> OutputPixels : register(u0);
RWStructuredBuffer<uint> Statistics : register(u1);
#include "Statistics.hlsl"
StructuredBuffer<MaterialData> Materials : register(t3);
StructuredBuffer<LightData> Lights : register(t4);
StructuredBuffer<RayTextureData> Textures : register(t5);
StructuredBuffer<uint> TextureTexels : register(t6);

#include "Globals.hlsl"
#include "InstancePrograms.hlsli"
#include "Triangle.hlsl"
#include "Sphere.hlsl"
#include "TriangleIntersection.hlsl"
#include "MotionTriangle.hlsl"
#include "Select.hlsl"

uint Hash(uint value)
{
    value ^= value >> 16;
    value *= 0x7feb352d;
    value ^= value >> 15;
    value *= 0x846ca68b;
    value ^= value >> 16;
    return value;
}

float Random01(uint seed)
{
    return float(Hash(seed) & 0x00ffffff) / float(0x01000000);
}

#include "Sequence.hlsl"
#include "Adaptive.hlsl"
#include "ThinLens.hlsl"

float3 Background(float3 direction)
{
    float blend = saturate(direction.y * 0.5 + 0.5);
    return lerp(BackgroundBottom, BackgroundTop, blend);
}

float3 NormalizeSafe(float3 value, float3 fallback)
{
    float lengthSquared = dot(value, value);
    return lengthSquared > MinimumDirectionLengthSquared
        ? value * rsqrt(lengthSquared)
        : fallback;
}

float3 LinearToSRGB(float3 color)
{
    color = saturate(color);
    float3 low = color * 12.92;
    float3 high = 1.055 * pow(color, 1.0 / 2.4) - 0.055;
    return float3(
        color.r <= 0.0031308 ? low.r : high.r,
        color.g <= 0.0031308 ? low.g : high.g,
        color.b <= 0.0031308 ? low.b : high.b);
}

float SRGBToLinear(float value)
{
    return value <= 0.04045
        ? value / 12.92
        : pow((value + 0.055) / 1.055, 2.4);
}

int WrapTextureCoordinate(int value, uint size)
{
    int signedSize = int(size);
    int wrapped = value % signedSize;
    return wrapped < 0 ? wrapped + signedSize : wrapped;
}

float3 LoadDiffuseTexel(RayTextureData descriptor, int x, int y)
{
    uint offset = descriptor.OffsetTexels
        + uint(WrapTextureCoordinate(y, descriptor.Height)) * descriptor.Width
        + uint(WrapTextureCoordinate(x, descriptor.Width));
    uint packed = TextureTexels[offset];
    return float3(
        SRGBToLinear(float(packed & 0xffu) / 255.0),
        SRGBToLinear(float((packed >> 8u) & 0xffu) / 255.0),
        SRGBToLinear(float((packed >> 16u) & 0xffu) / 255.0));
}

float3 SampleDiffuse(uint textureIndex, float2 texCoord, uint hasTexCoord)
{
    if (hasTexCoord == 0u || textureIndex >= TextureCount)
    {
        return 1.0.xxx;
    }
    RayTextureData descriptor = Textures[textureIndex];
    if (descriptor.Width == 0u || descriptor.Height == 0u)
    {
        return 1.0.xxx;
    }
    if (descriptor.Height > TextureTexelCount / descriptor.Width)
    {
        return 1.0.xxx;
    }
    uint pixelCount = descriptor.Width * descriptor.Height;
    if (descriptor.OffsetTexels > TextureTexelCount
        || pixelCount > TextureTexelCount - descriptor.OffsetTexels)
    {
        return 1.0.xxx;
    }
    float x = frac(texCoord.x) * float(descriptor.Width) - 0.5;
    float y = frac(1.0 - texCoord.y) * float(descriptor.Height) - 0.5;
    int x0 = int(floor(x));
    int y0 = int(floor(y));
    float amountX = x - floor(x);
    float amountY = y - floor(y);
    float3 top = lerp(
        LoadDiffuseTexel(descriptor, x0, y0),
        LoadDiffuseTexel(descriptor, x0 + 1, y0),
        amountX);
    float3 bottom = lerp(
        LoadDiffuseTexel(descriptor, x0, y0 + 1),
        LoadDiffuseTexel(descriptor, x0 + 1, y0 + 1),
        amountX);
    return lerp(top, bottom, amountY);
}


#include "BsdfTypes.hlsl"
#include "Fresnel.hlsl"
#include "Ggx.hlsl"
#include "Diffuse.hlsl"
#include "Conductor.hlsl"
#include "Dielectric.hlsl"
#include "SurfaceScattering.hlsl"

struct WhittedBranch
{
    float3 Origin, Direction, Weight;
    uint Depth, IncludeRegisteredEmission;
};
struct WhittedEvents
{
    SurfaceSample Values[SpecularBranchCount];
};
WhittedEvents SplitIdealSurface(SurfaceBsdf surface)
{
    WhittedEvents result = (WhittedEvents)0;
    if (!SurfaceDelta(surface)) return result;
    // Inspect both ends of the lobe CDF instead of drawing a random variate.
    // Multiplying Weight by its probability mass recovers the branch energy.
    // TIR, index-matched glass and metals each have only one distinct event.
    const float finalVariate = asfloat(asuint(1.0f) - 1u);
    result.Values[0] = SampleSurface(surface, 0.0.xx, 0);
    result.Values[1] = SampleSurface(surface, 0.0.xx, finalVariate);
    if (result.Values[0].Event == result.Values[1].Event) result.Values[1] = (SurfaceSample)0;
    return result;
}
float WhittedVisibility(SurfaceBsdf surface, float3 position, float3 direction,
                        float3 endpoint, bool finiteLight, bool shadows, inout uint2 shadowRays)
{
    if (!shadows || ShadowStrength <= 0) return 1;
    const float3 origin = SurfaceOffset(surface, position, direction);
    float maximum = MaximumRayDistance;
    if (finiteLight)
    {
        const float3 segment = endpoint - origin;
        const float distance = length(segment);
        maximum = distance - RayEpsilon;
        if (maximum <= RayEpsilon) return 1;
        direction = segment / distance;
    }
    IncrementLocalCounter(shadowRays);
    const Payload blocker = CastRay(origin, direction, RayEpsilon, maximum);
    return (blocker.Flags & SurfaceHitFlag) != 0u ? 1 - saturate(ShadowStrength) : 1;
}
float3 WhittedDiffuseConnection(SurfaceBsdf surface, float3 position, float3 direction,
    float3 endpoint, float3 radiance, bool finiteLight, bool shadows, inout uint2 shadowRays)
{
    const SurfaceEvaluation scattering = EvaluateSurface(surface, direction);
    if (all(scattering.Value == 0)) return 0.0.xxx;
    const float visible = WhittedVisibility(surface, position, direction, endpoint, finiteLight, shadows, shadowRays);
    return scattering.Value * radiance * (abs(dot(surface.Normal, direction)) * visible);
}
float3 WhittedPhysicalLighting(SurfaceBsdf surface, WhittedEvents events, float3 position,
                               inout uint2 shadowRays)
{
    // Fixed rectangle quadrature for diffuse reflectance; both exact delta
    // connections for ideal surfaces. No stochastic direct estimator is used.
    float3 direct = 0.0.xxx;
    [loop]
    for (uint index = 0; index < LightCount; ++index)
    {
        const LightData light = Lights[index];
        if (light.Intensity <= 0) continue;
        const bool shadows = light.CastShadows != 0u;
        if (light.Type == DirectionalLightType)
        {
            direct += WhittedDiffuseConnection(surface, position, light.Direction, 0.0.xxx,
                light.Color * light.Intensity, false, shadows, shadowRays);
            continue;
        }
        if (light.Type == PointLightType)
        {
            const float3 offset = light.Position - position;
            const float distanceSquared = dot(offset, offset);
            if (distanceSquared <= RayEpsilon * RayEpsilon) continue;
            direct += WhittedDiffuseConnection(surface, position, offset / sqrt(distanceSquared), light.Position,
                light.Color * (light.Intensity / distanceSquared), true, shadows, shadowRays);
            continue;
        }
        if (light.Type != RectAreaLightType || light.Width <= 0 || light.Height <= 0) continue;
        const float area = light.Width * light.Height;
        const float3 emitted = light.Color * (light.Intensity / area);
        const float3 helper = abs(light.Direction.y) < BasisParallelThreshold
            ? float3(0, 1, 0) : float3(1, 0, 0);
        const float3 tangent = normalize(cross(helper, light.Direction));
        const float3 bitangent = cross(light.Direction, tangent);
        if (SurfaceDelta(surface))
        {
            [loop]
            for (uint eventIndex = 0; eventIndex < SpecularBranchCount; ++eventIndex)
            {
                const SurfaceSample event = events.Values[eventIndex];
                if (event.Event == 0u || event.Pdf <= 0 || all(event.Weight == 0)) continue;
                const float facing = -dot(event.Direction, light.Direction);
                if (facing <= 0) continue;
                const float distance = dot(position - light.Position, light.Direction) / facing;
                if (distance <= RayEpsilon) continue;
                const float3 endpoint = position + event.Direction * distance;
                const float3 offset = endpoint - light.Position;
                if (abs(dot(offset, tangent)) > light.Width * 0.5 || abs(dot(offset, bitangent)) > light.Height * 0.5) continue;
                const float visible = WhittedVisibility(surface, position, event.Direction, endpoint, true, shadows, shadowRays);
                direct += event.Weight * emitted * (event.Pdf * visible);
            }
        }
        else
        {
            [loop]
            for (uint sampleIndex = 0; sampleIndex < AreaLightSamples; ++sampleIndex)
            {
                const float2 cell = float2(sampleIndex % AreaLightStrataPerAxis, sampleIndex / AreaLightStrataPerAxis);
                const float2 uv = (cell + 0.5) / AreaLightStrataPerAxis - 0.5;
                const float3 endpoint = light.Position + tangent * (uv.x * light.Width)
                    + bitangent * (uv.y * light.Height);
                const float3 offset = endpoint - position;
                const float distanceSquared = dot(offset, offset);
                if (distanceSquared <= RayEpsilon * RayEpsilon) continue;
                const float3 direction = offset / sqrt(distanceSquared);
                const float facing = -dot(direction, light.Direction);
                if (facing <= 0) continue;
                const float quadrature = area * facing / (AreaLightSamples * distanceSquared);
                direct += WhittedDiffuseConnection(surface, position, direction, endpoint,
                    emitted * quadrature, true, shadows, shadowRays);
            }
        }
    }
    return direct;
}

void AccumulateLightSample(
    float3 lightDirection,
    float3 incidentRadiance,
    float visibility,
    float3 surfaceNormal,
    float3 viewDirection,
    float shininess,
    inout float3 diffuseLight,
    inout float3 specularLight)
{
    float directionLengthSquared = dot(lightDirection, lightDirection);
    if (directionLengthSquared <= MinimumDirectionLengthSquared)
    {
        return;
    }
    float3 direction = lightDirection * rsqrt(directionLengthSquared);
    float normalToLight = max(dot(surfaceNormal, direction), 0.0);
    if (normalToLight <= 0.0)
    {
        return;
    }
    float3 visibleRadiance = incidentRadiance * visibility;
    diffuseLight += visibleRadiance * normalToLight;
    float3 halfVector = NormalizeSafe(
        direction + viewDirection,
        surfaceNormal);
    float specularAmount = pow(
        max(dot(surfaceNormal, halfVector), 0.0),
        max(shininess, 1.0));
    specularLight += visibleRadiance * specularAmount;
}

void AccumulateFiniteLight(
    LightData light,
    float3 lightPosition,
    float emitterWeight,
    bool areaEmitter,
    float3 position,
    float3 surfaceNormal,
    float3 viewDirection,
    float shininess,
    inout float3 diffuseLight,
    inout float3 specularLight,
    inout uint2 shadowRays)
{
    if (light.Intensity <= 0.0 || emitterWeight <= 0.0)
    {
        return;
    }
    float3 toLight = lightPosition - position;
    float distanceSquared = dot(toLight, toLight);
    if (!isfinite(distanceSquared)
        || distanceSquared <= RayEpsilon * RayEpsilon)
    {
        return;
    }
    float distance = sqrt(distanceSquared);
    float3 directionToLight = toLight / distance;
    float surfaceCosine = max(dot(surfaceNormal, directionToLight), 0.0);
    if (surfaceCosine <= 0.0)
    {
        return;
    }
    float emitterCosine = 1.0;
    if (areaEmitter)
    {
        emitterCosine =
            max(dot(light.Direction, -directionToLight), 0.0);
        if (emitterCosine <= 0.0)
        {
            return;
        }
    }

    float visibility = 1.0;
    if (ShadowStrength > 0.0 && light.CastShadows != 0u)
    {
        // Offset both the start and the finite segment definition, avoiding self-shadowing.
        const float3 shadowOrigin = position + surfaceNormal * RayEpsilon;
        const float3 shadowOffset = lightPosition - shadowOrigin;
        const float shadowDistance = length(shadowOffset);
        float maximumDistance = shadowDistance - RayEpsilon;
        if (maximumDistance > RayEpsilon)
        {
            IncrementLocalCounter(shadowRays);
            Payload shadow = CastRay(
                shadowOrigin,
                shadowOffset / shadowDistance,
                RayEpsilon,
                maximumDistance);
            if ((shadow.Flags & SurfaceHitFlag) != 0)
            {
                visibility = 1.0 - ShadowStrength;
            }
        }
    }
    float3 incidentRadiance =
        max(light.Color, 0.0)
        * (max(light.Intensity, 0.0)
            * emitterCosine
            * emitterWeight
            / distanceSquared);
    AccumulateLightSample(
        directionToLight,
        incidentRadiance,
        visibility,
        surfaceNormal,
        viewDirection,
        shininess,
        diffuseLight,
        specularLight);
}

[shader("raygeneration")]
void RayGen()
{
    uint2 pixel = DispatchRaysIndex().xy;
    if (pixel.x >= Width || pixel.y >= Height) return;

    uint2 primaryRays = uint2(0u, 0u);
    uint2 shadowRays = uint2(0u, 0u);
    uint2 reflectionRays = uint2(0u, 0u);
    LinearFilmPixel film = (LinearFilmPixel)0;
    uint productionSamples = AdaptivePhase == PreparedBudgetPhase ? LinearFilm[pixel.y * Width + pixel.x].SampleCount : SamplesPerPixel;
    const uint pilotSamples = AdaptiveEnabled != 0u && AdaptivePhase != PreparedBudgetPhase ? AdaptivePilotSamples : 0u;
    const uint maximumIterations = pilotSamples + SamplesPerPixel;
    for (uint iteration = 0; iteration < maximumIterations; ++iteration)
    {
        const bool pilot = iteration < pilotSamples;
        const uint sample = pilot ? iteration : iteration - pilotSamples;
        const uint targetSamples = pilot ? pilotSamples : productionSamples;
        if (sample >= targetSamples) break;
        IncrementLocalCounter(primaryRays);
        const SampleContext samples = AdaptiveSamples(pixel, sample, targetSamples, pilot);
        float jitterX = 0.5;
        float jitterY = 0.5;
        if (targetSamples > 1u || SamplingSequence != IndependentSequence || SamplingSeed != 0)
        {
            const float2 pixelSample = Sample2D(samples, 0u);
            jitterX = pixelSample.x;
            jitterY = pixelSample.y;
        }
        float u = (float(pixel.x + TileX) + jitterX) / float(FullWidth);
        float v = (float(pixel.y + TileY) + jitterY) / float(FullHeight);
        float screenX = (2.0 * u - 1.0) * HalfWidth;
        float screenY = (1.0 - 2.0 * v) * HalfHeight;
        float3 origin = CameraPosition;
        float3 direction = normalize(
            CameraForward + CameraRight * screenX + CameraUp * screenY);
        if (LensRadius > 0) ApplyThinLens(origin, direction, Sample2D(samples, LensSampleDimension));
        float3 radiance = 0.0;
        // A depth-first walk stores at most depth+1 pending rays. The host and
        // shader share that bound; total work can still grow exponentially.
        // No roulette or low-energy branch truncation changes Whitted's sum.
        WhittedBranch pending[WhittedBranchCapacity];
        WhittedBranch root = (WhittedBranch)0;
        root.Origin = origin;
        root.Direction = direction;
        root.Weight = 1.0.xxx;
        root.IncludeRegisteredEmission = 1u;
        pending[0] = root;
        uint pendingCount = 1;
        [loop]
        while (pendingCount > 0)
        {
            const WhittedBranch branch = pending[--pendingCount];
            origin = branch.Origin;
            direction = branch.Direction;
            const float3 throughput = branch.Weight;
            const uint bounce = branch.Depth;
            if (PrimitiveCount == 0)
            {
                radiance += throughput * Background(direction);
                continue;
            }
            Payload payload = CastRay(
                origin,
                direction,
                RayEpsilon,
                MaximumRayDistance);
            if ((payload.Flags & SurfaceHitFlag) == 0)
            {
                radiance += throughput * Background(direction);
                continue;
            }

            MaterialData material = Materials[min(payload.MaterialIndex, MaterialCount - 1)];
            float3 surface = max(payload.Color, 0.0)
                * max(
                    SampleDiffuse(
                        material.DiffuseTextureIndex,
                        payload.TexCoord,
                        payload.Flags & SurfaceTexCoordFlag),
                    0.0);
            float3 position = origin + direction * asfloat(payload.DistanceBits);
            float3 viewDirection = NormalizeSafe(-direction, payload.Normal);
            if (material.Model != MaterialLegacy)
            {
                const SurfaceBsdf scattering = MakeSurfaceBsdf(material, payload, viewDirection, surface);
                const WhittedEvents events = SplitIdealSurface(scattering);
                const float3 emission = branch.IncludeRegisteredEmission != 0u || material.SampledEmitter == 0u
                    ? material.Emission.xyz : 0.0.xxx;
                radiance += throughput * (emission + WhittedPhysicalLighting(scattering, events, position, shadowRays));
                if (bounce == MaxReflectionBounces) continue;
                [unroll]
                for (uint eventIndex = 0; eventIndex < SpecularBranchCount; ++eventIndex)
                {
                    const SurfaceSample event = events.Values[eventIndex];
                    if (event.Event == 0u || event.Pdf <= 0) continue;
                    const float3 weight = throughput * event.Weight * event.Pdf;
                    if (all(weight == 0)) continue;
                    WhittedBranch continuation;
                    continuation.Origin = SurfaceOffset(scattering, position, event.Direction);
                    continuation.Direction = event.Direction;
                    continuation.Weight = weight;
                    continuation.Depth = bounce + 1;
                    // Direct connections already accounted for registered emitters.
                    continuation.IncludeRegisteredEmission = 0u;
                    pending[pendingCount++] = continuation;
                    IncrementLocalCounter(reflectionRays);
                }
                continue;
            }
            float3 diffuseLight = 0.0;
            float3 specularLight = 0.0;
            for (uint lightIndex = 0; lightIndex < LightCount; ++lightIndex)
            {
                LightData light = Lights[lightIndex];
                if (light.Type == DirectionalLightType)
                {
                    float3 directionToLight = light.Direction;
                    float cosine = max(0.0, dot(payload.Normal, directionToLight));
                    if (cosine <= 0.0 || light.Intensity <= 0.0)
                    {
                        continue;
                    }
                    float visibility = 1.0;
                    if (ShadowStrength > 0.0 && light.CastShadows != 0u)
                    {
                        IncrementLocalCounter(shadowRays);
                        float3 shadowNormal =
                            dot(payload.Normal, directionToLight) >= 0.0
                                ? payload.Normal
                                : -payload.Normal;
                        Payload shadow = CastRay(
                            position + shadowNormal * RayEpsilon,
                            directionToLight,
                            RayEpsilon,
                            MaximumRayDistance);
                        if ((shadow.Flags & SurfaceHitFlag) != 0)
                        {
                            visibility = 1.0 - ShadowStrength;
                        }
                    }
                    AccumulateLightSample(
                        directionToLight,
                        max(light.Color, 0.0) * max(light.Intensity, 0.0),
                        visibility,
                        payload.Normal,
                        viewDirection,
                        material.AmbientShininess.w,
                        diffuseLight,
                        specularLight);
                }
                else if (light.Type == PointLightType)
                {
                    AccumulateFiniteLight(
                        light,
                        light.Position,
                        1.0,
                        false,
                        position,
                        payload.Normal,
                        viewDirection,
                        material.AmbientShininess.w,
                        diffuseLight,
                        specularLight, shadowRays);
                }
                else if (light.Type == RectAreaLightType)
                {
                    float3 emissionDirection = light.Direction;
                    float3 helper =
                        abs(emissionDirection.y) < BasisParallelThreshold
                        ? float3(0.0, 1.0, 0.0)
                        : float3(1.0, 0.0, 0.0);
                    float3 tangent =
                        normalize(cross(helper, emissionDirection));
                    float3 bitangent =
                        normalize(cross(emissionDirection, tangent));
                    // Keep traversal in one loop body; unrolling duplicates the
                    // hardware and software instance paths for every light sample.
                    [loop]
                    for (uint sampleIndex = 0u; sampleIndex < AreaLightSamples; ++sampleIndex)
                    {
                        float offsetX = (sampleIndex & 1u) != 0u ? (1.0f / AreaLightSamples) : -(1.0f / AreaLightSamples);
                        float offsetY = (sampleIndex & 2u) != 0u ? (1.0f / AreaLightSamples) : -(1.0f / AreaLightSamples);

                        float3 samplePosition =
                            light.Position
                            + tangent * (offsetX * light.Width)
                            + bitangent * (offsetY * light.Height);
                        AccumulateFiniteLight(
                            light,
                            samplePosition,
                            (1.0f / AreaLightSamples),
                            true,
                            position,
                            payload.Normal,
                            viewDirection,
                            material.AmbientShininess.w,
                            diffuseLight,
                            specularLight, shadowRays);
                    }
                }
            }

            float3 direct = surface * max(material.AmbientShininess.xyz, 0.0) * max(AmbientIntensity, 0.0)
                + surface * max(material.AlbedoReflectivity.xyz, 0.0) * diffuseLight
                + max(material.Specular.xyz, 0.0) * specularLight + max(material.Emission.xyz, 0.0);
            float reflectivity = saturate(material.AlbedoReflectivity.w);
            if (branch.IncludeRegisteredEmission == 0u && material.SampledEmitter != 0u)
                direct -= material.Emission.xyz;
            radiance += throughput * direct * (1.0 - reflectivity);
            if (reflectivity <= 0.0 || bounce == MaxReflectionBounces) continue;
            float3 reflectedDirection =
                normalize(reflect(direction, payload.Normal));
            float3 reflectionNormal =
                dot(payload.Normal, reflectedDirection) >= 0.0
                    ? payload.Normal
                    : -payload.Normal;
            origin = position + reflectionNormal * RayEpsilon;
            direction = reflectedDirection;
            WhittedBranch continuation;
            continuation.Origin = origin;
            continuation.Direction = direction;
            continuation.Weight = throughput * reflectivity;
            continuation.Depth = bounce + 1;
            continuation.IncludeRegisteredEmission = 1u;
            pending[pendingCount++] = continuation;
            IncrementLocalCounter(reflectionRays);
        }
        AccumulateFilmSample(film, radiance);
        if (pilot && sample + 1u == pilotSamples)
        {
            productionSamples = AdaptiveProductionCount(film);
            film = (LinearFilmPixel)0;
        }
    }

    LinearFilm[pixel.y * Width + pixel.x] = film;
    float3 color = max(film.MeanRadiance, 0.0);
    color /= max(1.0, max(color.r, max(color.g, color.b)));
    if (GammaCorrection != 0) color = LinearToSRGB(color);
    uint3 encoded = uint3(color * 255.0 + 0.5);
    OutputPixels[pixel.y * Width + pixel.x] =
        encoded.x | (encoded.y << 8) | (encoded.z << 16) | 0xff000000;
    AddStatistic(PrimaryCounterWord, primaryRays);
    AddStatistic(ShadowCounterWord, shadowRays);
    AddStatistic(ReflectionCounterWord, reflectionRays);
    AddStatistic(CompletedCounterWord, uint2(1u, 0u));
}
