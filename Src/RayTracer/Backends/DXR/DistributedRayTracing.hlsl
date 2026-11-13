// Cook, Porter and Carpenter (1984), "Distributed Ray Tracing".
// Stochastic area-light quadrature and normalized Phong glossy continuation.
// f=rho*(n+2)/(2*pi)*cos(alpha)^n and p=(n+1)/(2*pi)*cos(alpha)^n;
// the continuation weight is f*cos(theta)/p. Below-surface samples are zero.
// Physical surfaces use this backend's Lambert/GGX/Fresnel implementations.
// Diffuse indirect continuation is absent. This is not network distribution.
// Closed dielectric interiors use sampled free flights and HG phase events.
static const float Pi = 3.14159265358979323846f;
static const float MaximumRayDistance = 1.0e30;
static const float BasisParallelThreshold = 0.999;
static const uint AreaLightSamples = 4;
static const uint ImportanceBit = 1u << 0;
static const uint NeeBit = 1u << 1;
static const uint MisShift = 3u, MisMask = 3u, NoMis = 0u, PowerMis = 2u;
static const uint PixelSeedMultiplier = 9781u;
static const uint SampleSeedMultiplier = 6271u;
static const uint BounceSeedMultiplier = 0x9e3779b9u;
static const uint LightSeedMultiplier = 0x85ebca6bu;
static const uint LightSampleSeedMultiplier = 0xc2b2ae35u;
static const uint SecondDimensionScramble = 0x68bc21ebu;
static const uint LobeSelectionScramble = 0xa511e9b3u;
static const uint DirectBsdfScramble = 0xd1b54a35u;
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

#include "MaterialTypes.hlsl"
#include "HomogeneousMedium.hlsl"
#include "SubsurfaceScattering.hlsl"
#define ARVO_PARTICIPATING_MEDIA
#include "ShutterSampling.hlsl"
#include "MotionInterpolation.hlsl"
#include "CameraMotion.hlsl"
#include "PathTime.hlsl"
cbuffer MediumBlock : register(b2)
{
    MediumParameters ParticipatingMedium;
    MotionExposure CameraExposure;
}

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
#include "MultipleImportance.hlsl"
#include "DirectLighting.hlsl"
#include "VolumeLighting.hlsl"

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

MediumParameters SubsurfaceInterior(MaterialData material)
{
    MediumParameters medium = (MediumParameters)0;
    medium.Enabled = 1;
    medium.Minimum = float3(-MaximumRayDistance, -MaximumRayDistance, -MaximumRayDistance);
    medium.Maximum = float3(MaximumRayDistance, MaximumRayDistance, MaximumRayDistance);
    medium.Anisotropy = material.ConductorIndex.w;
    const float3 albedo = material.AlbedoReflectivity.xyz * material.ConductorExtinction.w;
    medium.Scattering = albedo / material.Optical.w;
    medium.Absorption = (1 - albedo) / material.Optical.w;
    return medium;
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
        float sceneTime = 0;
        if (CameraExposure.Enabled != 0 || CameraExposure.SceneEnabled != 0)
        {
            const float observation = CameraExposure.Open == CameraExposure.Close
                ? 0 : Sample2D(samples, TimeSampleDimension).x;
            const MotionCameraRay exposed = SampleCameraMotion(
                origin, direction, CameraPosition, CameraExposure, observation);
            origin = exposed.Origin;
            direction = exposed.Direction;
            sceneTime = exposed.SceneTime;
        }
        // Publish once before any hardware, callable or procedural traversal.
        // The same clock serves all direct connections and specular bounces.
        BeginPathTime(sceneTime);
        float3 throughput = 1.0;
        bool includeRegisteredEmission = true;
        float3 radiance = 0.0;
        uint interiorMaterial = ~0u;

        for (uint bounce = 0; bounce <= MaxReflectionBounces; ++bounce)
        {
            if (PrimitiveCount == 0)
            {
                radiance += throughput * Background(direction);
                break;
            }
            Payload payload = CastRay(
                origin,
                direction,
                RayEpsilon,
                MaximumRayDistance);
            const bool found = (payload.Flags & SurfaceHitFlag) != 0;
            if (bounce == 0 && found && (payload.Flags & SurfaceFrontFaceFlag) == 0 &&
                Materials[payload.MaterialIndex].Optical.w > 0)
                interiorMaterial = payload.MaterialIndex;
            if (interiorMaterial != ~0u)
            {
                const uint FreeFlightScramble = 0x7a4d2e91u, PhaseScramble = 0x3c6ef372u;
                const uint domain = (bounce + 1u) * BounceSeedMultiplier;
                const MaterialData interior = Materials[interiorMaterial];
                const MediumParameters medium = SubsurfaceInterior(interior);
                const SubsurfaceFlight flight = SubsurfaceFreeFlight(interior.Optical.w,
                    interior.AlbedoReflectivity.xyz * interior.ConductorExtinction.w,
                    direction, found ? asfloat(payload.DistanceBits) : MaximumRayDistance,
                    Sample2D(samples, domain ^ FreeFlightScramble).x);
                throughput *= flight.Weight;
                if (all(throughput == 0)) break;
                if (flight.Scattered != 0)
                {
                    const float3 position = origin + direction * flight.Distance;
                    radiance += throughput * VolumeDirectLighting(medium, position, direction,
                        samples, bounce, shadowRays);
                    if (bounce == MaxReflectionBounces) break;
                    const SubsurfaceDirection phase = SubsurfaceSampleDirection(direction,
                        medium.Anisotropy, Sample2D(samples, domain ^ PhaseScramble),
                        (TransportFlags & ImportanceBit) != 0);
                    throughput *= phase.Weight;
                    origin = position;
                    direction = phase.Direction;
                    includeRegisteredEmission = false;
                    IncrementLocalCounter(reflectionRays);
                    continue;
                }
            }
            if (!found)
            {
                radiance += throughput * Background(direction);
                break;
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
                const float3 emission = includeRegisteredEmission || material.SampledEmitter == 0u
                    ? material.Emission.xyz : 0.0.xxx;
                radiance += throughput * (emission + PhysicalDirectLighting(scattering, position,
                    samples, bounce, shadowRays, 0,
                    interiorMaterial == ~0u ? 0 : Materials[interiorMaterial].Optical.w));
                // Distributed tracing samples a single glossy/ideal continuation.
                // Diffuse surfaces terminate after direct light; no roulette or
                // diffuse indirect bounce turns this into the path integrator.
                if (material.Model == MaterialDiffuse || bounce == MaxReflectionBounces) break;
                const uint dimension = (bounce + 1u) * BounceSeedMultiplier;
                const SurfaceSample continuation = SampleSurface(scattering, Sample2D(samples, dimension),
                    Sample2D(samples, dimension ^ LobeSelectionScramble).x);
                if (continuation.Event == 0u || continuation.Pdf <= 0) break;
                throughput *= continuation.Weight;
                if (all(throughput == 0)) break;
                origin = SurfaceOffset(scattering, position, continuation.Direction);
                direction = continuation.Direction;
                if (material.Optical.w > 0 && (continuation.Event & ScatteringTransmission) != 0)
                    interiorMaterial = (payload.Flags & SurfaceFrontFaceFlag) != 0 ? payload.MaterialIndex : ~0u;
                includeRegisteredEmission = false;
                IncrementLocalCounter(reflectionRays);
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
                        const uint lightDimension = ((bounce + 1u) * BounceSeedMultiplier)
                            ^ ((lightIndex + 1u) * LightSeedMultiplier) ^ ((sampleIndex + 1u) * LightSampleSeedMultiplier);
                        const float2 lightSample = Sample2D(samples, lightDimension);
                        const float offsetX = (float(sampleIndex & 1u) + lightSample.x) * 0.5f - 0.5f;
                        const float offsetY = (float(sampleIndex >> 1u) + lightSample.y) * 0.5f - 0.5f;

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
            if (!includeRegisteredEmission && material.SampledEmitter != 0u)
                direct -= material.Emission.xyz;
            radiance += throughput * direct * (1.0 - reflectivity);
            if (reflectivity <= 0.0 || bounce == MaxReflectionBounces) break;
            const float exponent = max(material.AmbientShininess.w, 1.0f);
            const uint dimension = ((bounce + 1u) * BounceSeedMultiplier);
            const float2 reflectionSample = Sample2D(samples, dimension);
            const float lobeCosine = pow(reflectionSample.x, 1.0f / (exponent + 1.0f));
            const float lobeSine = sqrt(max(0.0f, 1.0f - lobeCosine * lobeCosine));
            const float azimuth = 2.0f * Pi * reflectionSample.y;
            const float3 reflection = normalize(reflect(direction, payload.Normal));
            const float3 helper = abs(reflection.y) < BasisParallelThreshold
                ? float3(0.0f, 1.0f, 0.0f) : float3(1.0f, 0.0f, 0.0f);
            const float3 tangent = normalize(cross(helper, reflection));
            const float3 bitangent = cross(reflection, tangent);
            const float3 outgoing = normalize(tangent * (lobeSine * cos(azimuth))
                + bitangent * (lobeSine * sin(azimuth)) + reflection * lobeCosine);
            const float surfaceCosine = max(0.0f, dot(payload.Normal, outgoing));
            if (surfaceCosine <= 0.0f) break;
            throughput *= reflectivity * (exponent + 2.0f) / (exponent + 1.0f) * surfaceCosine;
            origin = position + payload.Normal * RayEpsilon;
            direction = outgoing;
            includeRegisteredEmission = true;
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
