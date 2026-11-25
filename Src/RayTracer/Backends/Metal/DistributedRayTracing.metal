// Cook, Porter and Carpenter (1984), "Distributed Ray Tracing".
// Physical materials use GGX/Fresnel event sampling and rectangle MIS.
// Legacy materials retain normalized Phong glossy continuation.
// f=rho*(n+2)/(2*pi)*cos(alpha)^n and p=(n+1)/(2*pi)*cos(alpha)^n;
// the continuation weight is f*cos(theta)/p. Below-surface samples are zero.
// Diffuse indirect continuation is absent. This is not network distribution.
// Closed dielectric interiors use sampled free flights and HG phase events.
#include <metal_stdlib>
using namespace metal;
#include "SurfaceTypes.metal"
#include "HomogeneousMedium.metal"
#include "SubsurfaceScattering.metal"
#include "ShutterSampling.metal"
#include "MotionInterpolation.metal"
#include "CameraMotion.metal"
#define ARVO_PARTICIPATING_MEDIA

constant uint LightTypeDirectional = 0u;
constant uint LightTypePoint = 1u;
constant uint LightTypeRectArea = 2u;
constant float RayEpsilon = 1.0e-4f;

constant float Pi = 3.14159265358979323846f;
constant float MaximumRayDistance = 1.0e30f;
constant float BasisParallelThreshold = 0.999f;
constant uint ThreadsPerGroup = 256;
constant uint AreaLightSamples = 4;
constant uint PixelSeedMultiplier = 9781u;
constant uint SampleSeedMultiplier = 6271u;
constant uint BounceSeedMultiplier = 0x9e3779b9u;
constant uint LightSeedMultiplier = 0x85ebca6bu;
constant uint LightSampleSeedMultiplier = 0xc2b2ae35u;
constant uint SecondDimensionScramble = 0x68bc21ebu;
constant uint LobeSelectionScramble = 0xa511e9b3u;
constant uint DirectBsdfScramble = 0xd1b54a35u;
constant uint ImportanceBit = 1u << 0;
constant uint NeeBit = 1u << 1;
constant uint MisShift = 3u, MisMask = 3u, NoMis = 0u, PowerMis = 2u;
constant uint AvalancheMultiplierA = 0x7feb352du;
constant uint AvalancheMultiplierB = 0x846ca68bu;
constant uint RandomMantissaMask = 0x00ffffffu;
constant float RandomMantissaRange = 16777216.0f;

#include "ShadingTypes.metal"

struct FrameConstants
{
    packed_float3 CameraPosition;
    uint Width;
    packed_float3 CameraForward;
    uint Height;
    packed_float3 CameraRight;
    uint TotalSamples;
    packed_float3 CameraUp;
    uint MaxReflectionBounces;
    float HalfWidth;
    float HalfHeight;
    uint MaterialCount;
    uint LightCount;
    packed_float3 BackgroundTop;
    float AmbientIntensity;
    packed_float3 BackgroundBottom;
    uint NodeCount;
    uint PrimitiveRefCount;
    uint TriangleCount;
    uint SphereCount;
    float ShadowStrength;
    uint PixelBegin;
    uint PixelCount;
    uint SampleBegin;
    uint SampleCount;
    uint GammaCorrection;
    uint GroupCount;
    uint TextureCount;
    uint TextureTexelCount;
    uint FullWidth;
    uint FullHeight;
    uint TileX;
    uint TileY;
    uint Integrator;
    uint TransportFlags;
    uint RouletteStartDepth;
    uint SamplingSequence;
    uint AdaptiveEnabled;
    uint AdaptivePilotSamples;
    uint AdaptiveMinimumSamples;
    uint AdaptivePhase;
    float AdaptiveRelativeError;
    float AdaptiveAbsoluteError;
    uint Acceleration;
    uint InstanceNodeCount;
    float LensRadius;
    float FocusDistance;
    uint SamplingSeed;
    uint LensPadding1;
};

struct Counter64Parts
{
    uint Low;
    uint High;
};

struct GroupStatistics
{
    Counter64Parts PrimaryRays;
    Counter64Parts ShadowRays;
    Counter64Parts ReflectionRays;
    Counter64Parts PrimitiveTests;
};

#include "Hash.metal"

#include "Sequence.metal"
#include "FilmMoments.metal"
#include "Adaptive.metal"
#include "ThinLens.metal"

float SRGBToLinear(float value)
{
    return value <= 0.04045f
        ? value / 12.92f
        : pow((value + 0.055f) / 1.055f, 2.4f);
}

int WrapTextureCoordinate(int value, uint size)
{
    const int signedSize = int(size);
    const int wrapped = value % signedSize;
    return wrapped < 0 ? wrapped + signedSize : wrapped;
}

float3 LoadDiffuseTexel(
    device const uint* textureTexels,
    PackedTextureDescriptor descriptor,
    int x,
    int y)
{
    const uint offset = descriptor.OffsetTexels
        + uint(WrapTextureCoordinate(y, descriptor.Height)) * descriptor.Width
        + uint(WrapTextureCoordinate(x, descriptor.Width));
    const uint packed = textureTexels[offset];
    return float3(
        SRGBToLinear(float(packed & 0xffu) / 255.0f),
        SRGBToLinear(float((packed >> 8u) & 0xffu) / 255.0f),
        SRGBToLinear(float((packed >> 16u) & 0xffu) / 255.0f));
}

float3 SampleDiffuse(
    device const PackedTextureDescriptor* textures,
    device const uint* textureTexels,
    constant FrameConstants& constants,
    uint textureIndex,
    float2 texCoord,
    uint hasTexCoord)
{
    if (hasTexCoord == 0u || textureIndex >= constants.TextureCount)
    {
        return float3(1.0f);
    }
    const PackedTextureDescriptor descriptor = textures[textureIndex];
    if (descriptor.Width == 0u || descriptor.Height == 0u
        || descriptor.OffsetTexels >= constants.TextureTexelCount)
    {
        return float3(1.0f);
    }
    const uint availableTexels =
        constants.TextureTexelCount - descriptor.OffsetTexels;
    if (descriptor.Height > availableTexels / descriptor.Width)
    {
        return float3(1.0f);
    }

    const float x = fract(texCoord.x) * float(descriptor.Width) - 0.5f;
    const float y = fract(1.0f - texCoord.y) * float(descriptor.Height) - 0.5f;
    const float floorX = floor(x);
    const float floorY = floor(y);
    const int x0 = int(floorX);
    const int y0 = int(floorY);
    const float amountX = x - floorX;
    const float amountY = y - floorY;
    const float3 top = mix(
        LoadDiffuseTexel(textureTexels, descriptor, x0, y0),
        LoadDiffuseTexel(textureTexels, descriptor, x0 + 1, y0),
        amountX);
    const float3 bottom = mix(
        LoadDiffuseTexel(textureTexels, descriptor, x0, y0 + 1),
        LoadDiffuseTexel(textureTexels, descriptor, x0 + 1, y0 + 1),
        amountX);
    return mix(top, bottom, amountY);
}

void IncrementCounter(thread uint2& counter)
{
    const uint previousLow = counter.x;
    counter.x += 1u;
    if (counter.x < previousLow)
    {
        counter.y += 1u;
    }
}

uint2 AddCounter(uint2 left, uint2 right)
{
    const uint low = left.x + right.x;
    return uint2(low, left.y + right.y + (low < left.x ? 1u : 0u));
}

#include "ExtendedFloat.metal"
#include "TraversalTypes.metal"
#include "Triangle.metal"
#include "Sphere.metal"
#include "Bvh.metal"
#include "KdTree.metal"
#include "Bih.metal"
#include "Dispatch.metal"

bool IsLightVisible(
    SceneGeometryBuffers geometry,
    float3 origin,
    float3 direction,
    float maximumDistance,
    thread uint2& primitiveTests)
{
    RayData shadowRay;
    shadowRay.Origin = origin;
    shadowRay.Direction = direction;
    shadowRay.MinimumDistance = RayEpsilon;
    shadowRay.MaximumDistance = maximumDistance;
    HitData shadowHit;
    shadowHit.Distance = maximumDistance;
    shadowHit.Normal = float3(0.0f);
    shadowHit.Color = float3(1.0f);
    shadowHit.MaterialIndex = 0u;
    shadowHit.TexCoord = float2(0.0f);
    shadowHit.HasTexCoord = 0u;
    return !IntersectScene(
        geometry,
        shadowRay,
        true,
        shadowHit,
        primitiveTests);
}

float3 Background(constant FrameConstants& constants, float3 direction)
{
    const float blend = clamp(direction.y * 0.5f + 0.5f, 0.0f, 1.0f);
    return mix(
        float3(constants.BackgroundBottom),
        float3(constants.BackgroundTop),
        blend);
}

float3 NormalizeSafe(float3 value, float3 fallback)
{
    const float lengthSquared = dot(value, value);
    return lengthSquared > MinimumDirectionLengthSquared
        ? value * rsqrt(lengthSquared)
        : fallback;
}

#include "MultipleImportance.metal"
#include "BsdfTypes.metal"
#include "Fresnel.metal"
#include "Ggx.metal"
#include "Diffuse.metal"
#include "Conductor.metal"
#include "Dielectric.metal"
#include "SurfaceScattering.metal"
#include "LightConnection.metal"
#include "DirectLighting.metal"
#include "VolumeLighting.metal"

void AccumulateLightSample(
    float3 lightDirection,
    float3 incidentRadiance,
    float visibility,
    float3 surfaceNormal,
    float3 viewDirection,
    float shininess,
    thread float3& diffuseLight,
    thread float3& specularLight)
{
    const float directionLengthSquared = dot(lightDirection, lightDirection);
    if (directionLengthSquared <= MinimumDirectionLengthSquared)
    {
        return;
    }
    const float3 direction = lightDirection * rsqrt(directionLengthSquared);
    const float normalToLight = max(dot(surfaceNormal, direction), 0.0f);
    if (normalToLight <= 0.0f)
    {
        return;
    }
    const float3 visibleRadiance = incidentRadiance * visibility;
    diffuseLight += visibleRadiance * normalToLight;
    const float3 halfVector = NormalizeSafe(
        direction + viewDirection,
        surfaceNormal);
    const float specularAmount = pow(
        max(dot(surfaceNormal, halfVector), 0.0f),
        max(shininess, 1.0f));
    specularLight += visibleRadiance * specularAmount;
}

float3 LinearToSRGB(float3 color)
{
    color = clamp(color, float3(0.0f), float3(1.0f));
    const float3 low = color * 12.92f;
    const float3 high = 1.055f * pow(color, float3(1.0f / 2.4f)) - 0.055f;
    return select(high, low, color <= float3(0.0031308f));
}

MediumParameters SubsurfaceInterior(PackedMaterial material)
{
    MediumParameters medium{};
    medium.Enabled = 1;
    medium.Minimum = packed_float3(float3(-MaximumRayDistance));
    medium.Maximum = packed_float3(float3(MaximumRayDistance));
    medium.Anisotropy = material.ConductorIndex.w;
    const float3 albedo = material.AlbedoReflectivity.xyz * material.ConductorExtinction.w;
    medium.Scattering = packed_float3(albedo / material.Optical.w);
    medium.Absorption = packed_float3((float3(1) - albedo) / material.Optical.w);
    return medium;
}
kernel void ArvoTracePixels(
    device const PackedAccelerationNode* nodes [[buffer(0)]],
    device const uint* primitiveRefs [[buffer(1)]],
    device const PackedTriangle* triangles [[buffer(2)]],
    device const PackedSphere* spheres [[buffer(3)]],
    device const PackedMaterial* materials [[buffer(4)]],
    device const PackedLight* lights [[buffer(5)]],
    constant FrameConstants& constants [[buffer(6)]],
    device float4* accumulation [[buffer(7)]],
    device uint* sampleCounts [[buffer(8)]],
    device GroupStatistics* groupStatistics [[buffer(9)]],
    device const PackedTextureDescriptor* textures [[buffer(10)]],
    device const uint* textureTexels [[buffer(11)]],
    device uint* sampleTargets [[buffer(12)]],
    device const InstanceRecord* instances [[buffer(13)]],
    device const InstanceNode* instanceNodes [[buffer(14)]],
    device atomic_uint* instanceError [[buffer(15)]],
    constant MotionExposure& cameraExposure [[buffer(19)]],
    uint gridIndex [[thread_position_in_grid]],
    uint localIndex [[thread_index_in_threadgroup]],
    uint groupPosition [[threadgroup_position_in_grid]])
{
    SceneGeometryBuffers geometry = {{nodes, primitiveRefs, triangles, spheres,
        constants.NodeCount, constants.TriangleCount, constants.Acceleration},
        instances, instanceNodes, instanceError, constants.InstanceNodeCount};
    uint2 primaryRays = uint2(0u);
    uint2 shadowRays = uint2(0u);
    uint2 reflectionRays = uint2(0u);
    uint2 primitiveTests = uint2(0u);
    if (gridIndex < constants.PixelCount)
    {
        const uint pixelIndex = constants.PixelBegin + gridIndex;
        const uint x = pixelIndex % constants.Width;
        const uint y = pixelIndex / constants.Width;
        float4 moments = accumulation[pixelIndex];
        uint completedSamples = sampleCounts[pixelIndex];
        const bool pilot = constants.AdaptiveEnabled != 0u && constants.AdaptivePhase == AdaptivePilotPhase;
        if (constants.AdaptiveEnabled == 0u && constants.SampleBegin == 0u)
            sampleTargets[pixelIndex] = constants.TotalSamples;
        const uint target = pilot ? constants.AdaptivePilotSamples : sampleTargets[pixelIndex];
        for (uint sampleOffset = 0u;
             sampleOffset < constants.SampleCount;
             ++sampleOffset)
        {
            const uint sample = constants.SampleBegin + sampleOffset;
            if (sample >= target) break;
            IncrementCounter(primaryRays);
            const SampleContext samples = AdaptiveSampleContext(x + constants.TileX, y + constants.TileY, sample, target, constants);
            float jitterX = 0.5f;
            float jitterY = 0.5f;
            if (samples.SampleCount > 1u || samples.Sequence != IndependentSequence || samples.SamplingSeed != 0)
            {
                const float2 pixelSample = Sample2D(samples, 0u);
                jitterX = pixelSample.x;
                jitterY = pixelSample.y;
            }
            const float u = (float(x + constants.TileX) + jitterX) / float(constants.FullWidth);
            const float v = (float(y + constants.TileY) + jitterY) / float(constants.FullHeight);
            const float screenX = (2.0f * u - 1.0f) * constants.HalfWidth;
            const float screenY = (1.0f - 2.0f * v) * constants.HalfHeight;
            RayData ray;
            ray.Origin = float3(constants.CameraPosition);
            ray.Direction = normalize(
                float3(constants.CameraForward)
                + float3(constants.CameraRight) * screenX
                + float3(constants.CameraUp) * screenY);
            if (constants.LensRadius > 0)
                ApplyThinLens(ray, constants, Sample2D(samples, LensSampleDimension));
            geometry.Object.SceneTime = 0;
            if (cameraExposure.Enabled || cameraExposure.SceneEnabled)
            {
                const float timeObservation = cameraExposure.Open == cameraExposure.Close
                    ? 0 : Sample2D(samples, TimeSampleDimension).x;
                const MotionCameraRay exposed = SampleCameraMotion(
                    ray.Origin, ray.Direction, float3(constants.CameraPosition), cameraExposure, timeObservation);
                ray.Origin = exposed.Origin;
                ray.Direction = exposed.Direction;
                geometry.Object.SceneTime = exposed.SceneTime;
            }
            // Connections and glossy bounces retain this shutter observation.
            const DirectLightContext directContext = {geometry, lights, constants.LightCount,
                constants.TransportFlags, constants.ShadowStrength, MediumParameters{}};
            ray.MinimumDistance = RayEpsilon;
            ray.MaximumDistance = MaximumRayDistance;
            float3 radiance = float3(0.0f);
            float3 throughput = float3(1.0f);
            bool includeRegisteredEmission = true;
            uint interiorMaterial = ~0u;
            for (uint bounce = 0u;
                 bounce <= constants.MaxReflectionBounces;
                 ++bounce)
            {
                HitData hit;
                hit.Distance = ray.MaximumDistance;
                hit.Normal = float3(0.0f);
                hit.Color = float3(1.0f);
                hit.MaterialIndex = 0u;
                hit.TexCoord = float2(0.0f);
                hit.HasTexCoord = 0u;
                const bool found = IntersectScene(
                        geometry,
                        ray,
                        false,
                        hit,
                        primitiveTests);
                if (bounce == 0 && found && !hit.FrontFace && materials[hit.MaterialIndex].Optical.w > 0)
                    interiorMaterial = hit.MaterialIndex;
                if (interiorMaterial != ~0u)
                {
                    const uint FreeFlightScramble = 0x7a4d2e91u, PhaseScramble = 0x3c6ef372u;
                    const uint domain = (bounce + 1u) * BounceSeedMultiplier;
                    const PackedMaterial interior = materials[interiorMaterial];
                    const MediumParameters medium = SubsurfaceInterior(interior);
                    const SubsurfaceFlight flight = SubsurfaceFreeFlight(interior.Optical.w,
                        interior.AlbedoReflectivity.xyz * interior.ConductorExtinction.w,
                        ray.Direction, found ? hit.Distance : ray.MaximumDistance,
                        Sample2D(samples, domain ^ FreeFlightScramble).x);
                    throughput *= flight.Weight;
                    if (all(throughput == float3(0))) break;
                    if (flight.Scattered)
                    {
                        const float3 position = ray.Origin + ray.Direction * flight.Distance;
                        DirectLightContext volumeContext = directContext;
                        volumeContext.Medium = medium;
                        radiance += throughput * VolumeDirectLighting(volumeContext, position, ray.Direction,
                            samples, bounce, shadowRays, primitiveTests);
                        if (bounce == constants.MaxReflectionBounces) break;
                        const SubsurfaceDirection phase = SubsurfaceSampleDirection(ray.Direction,
                            medium.Anisotropy, Sample2D(samples, domain ^ PhaseScramble),
                            (constants.TransportFlags & ImportanceBit) != 0);
                        throughput *= phase.Weight;
                        ray.Origin = position;
                        ray.Direction = phase.Direction;
                        ray.MinimumDistance = RayEpsilon;
                        ray.MaximumDistance = MaximumRayDistance;
                        includeRegisteredEmission = false;
                        IncrementCounter(reflectionRays);
                        continue;
                    }
                }
                if (!found)
                {
                    radiance += throughput * Background(constants, ray.Direction);
                    break;
                }
                const PackedMaterial material = materials[
                    min(hit.MaterialIndex, constants.MaterialCount - 1u)];
                const float3 surface = max(hit.Color, float3(0.0f))
                    * max(
                        SampleDiffuse(
                            textures,
                            textureTexels,
                            constants,
                            material.DiffuseTextureIndex,
                            hit.TexCoord,
                            hit.HasTexCoord),
                        float3(0.0f));
                const float3 position = ray.Origin + ray.Direction * hit.Distance;
                const float3 viewDirection = NormalizeSafe(-ray.Direction, hit.Normal);
                if (material.Model != MaterialLegacy)
                {
                    const SurfaceBsdf scattering = MakeSurfaceBsdf(material, hit, viewDirection, surface,
                                                                   constants.TransportFlags);
                    const float3 emission = includeRegisteredEmission || material.SampledEmitter == 0u
                        ? material.Emission.xyz : float3(0);
                    radiance += throughput * (emission + PhysicalDirectLighting(directContext, scattering,
                        position, samples, bounce, shadowRays, primitiveTests,
                        interiorMaterial == ~0u ? 0 : materials[interiorMaterial].Optical.w));
                    // Distributed ray tracing follows glossy/specular events only.
                    // A diffuse vertexData receives direct lighting, then ends this walk.
                    if (material.Model == MaterialDiffuse || bounce == constants.MaxReflectionBounces) break;
                    const uint dimension = (bounce + 1u) * BounceSeedMultiplier;
                    const SurfaceSample event = SampleSurface(scattering, Sample2D(samples, dimension),
                        Sample2D(samples, dimension ^ LobeSelectionScramble).x);
                    if (event.Event == 0u || event.Pdf <= 0) break;
                    throughput *= event.Weight;
                    if (all(throughput == float3(0))) break;
                    ray.Origin = SurfaceOffset(scattering, position, event.Direction);
                    ray.Direction = event.Direction;
                    if (material.Optical.w > 0 && (event.Event & ScatteringTransmission) != 0)
                        interiorMaterial = hit.FrontFace ? hit.MaterialIndex : ~0u;
                    includeRegisteredEmission = false;
                    IncrementCounter(reflectionRays);
                    continue;
                }
                float3 diffuseLight = float3(0.0f);
                float3 specularLight = float3(0.0f);
                for (uint lightIndex = 0u;
                     lightIndex < constants.LightCount;
                     ++lightIndex)
                {
                    const PackedLight light = lights[lightIndex];
                    if (light.Intensity <= 0.0f)
                    {
                        continue;
                    }
                    const float3 lightColor = float3(light.Color);
                    if (light.Type == LightTypeDirectional)
                    {
                        const float3 directionToLight =
                            normalize(float3(light.Direction));
                        const float cosine =
                            max(0.0f, dot(hit.Normal, directionToLight));
                        if (cosine <= 0.0f)
                        {
                            continue;
                        }
                        float visibility = 1.0f;
                        if (constants.ShadowStrength > 0.0f
                            && light.CastShadows != 0u)
                        {
                            IncrementCounter(shadowRays);
                            const float3 shadowNormal =
                                dot(hit.Normal, directionToLight) >= 0.0f
                                    ? hit.Normal
                                    : -hit.Normal;
                            const bool visible = IsLightVisible(
                                geometry,
                                position + shadowNormal * RayEpsilon,
                                directionToLight,
                                MaximumRayDistance,
                                primitiveTests);
                            if (!visible)
                            {
                                visibility = 1.0f - constants.ShadowStrength;
                            }
                        }
                        AccumulateLightSample(
                            directionToLight,
                            max(lightColor, float3(0.0f))
                                * max(light.Intensity, 0.0f),
                            visibility,
                            hit.Normal,
                            viewDirection,
                            material.AmbientShininess.w,
                            diffuseLight,
                            specularLight);
                        continue;
                    }
                    if (light.Type != LightTypePoint
                        && light.Type != LightTypeRectArea)
                    {
                        continue;
                    }

                    const bool isArea = light.Type == LightTypeRectArea;
                    float3 emissionDirection = float3(0.0f);
                    float3 tangent = float3(0.0f);
                    float3 bitangent = float3(0.0f);
                    uint finiteSampleCount = 1u;
                    if (isArea)
                    {
                        emissionDirection = normalize(float3(light.Direction));
                        const float3 helper = fabs(emissionDirection.y) < BasisParallelThreshold
                            ? float3(0.0f, 1.0f, 0.0f)
                            : float3(1.0f, 0.0f, 0.0f);
                        tangent = normalize(cross(helper, emissionDirection));
                        bitangent =
                            normalize(cross(emissionDirection, tangent));
                        finiteSampleCount = AreaLightSamples;
                    }
                    for (uint finiteSample = 0u;
                         finiteSample < finiteSampleCount;
                         ++finiteSample)
                    {
                        float3 samplePosition = float3(light.Position);
                        float sampleWeight = 1.0f;
                        if (isArea)
                        {
                            const uint lightDimension = ((bounce + 1u) * BounceSeedMultiplier)
                                ^ ((lightIndex + 1u) * LightSeedMultiplier) ^ ((finiteSample + 1u) * LightSampleSeedMultiplier);
                            const float2 lightSample = Sample2D(samples, lightDimension);
                            const float offsetX = (float(finiteSample & 1u) + lightSample.x) * 0.5f - 0.5f;
                            const float offsetY = (float(finiteSample >> 1u) + lightSample.y) * 0.5f - 0.5f;

                            samplePosition += tangent * (offsetX * light.Width)
                                + bitangent * (offsetY * light.Height);
                            sampleWeight = (1.0f / AreaLightSamples);
                        }
                        const float3 toLight = samplePosition - position;
                        const float distanceSquared = dot(toLight, toLight);
                        if (!isfinite(distanceSquared)
                            || distanceSquared <= RayEpsilon * RayEpsilon)
                        {
                            continue;
                        }
                        const float distance = sqrt(distanceSquared);
                        const float3 directionToLight = toLight / distance;
                        const float surfaceCosine =
                            max(0.0f, dot(hit.Normal, directionToLight));
                        if (surfaceCosine <= 0.0f)
                        {
                            continue;
                        }
                        float emitterCosine = 1.0f;
                        if (isArea)
                        {
                            emitterCosine = max(
                                0.0f,
                                dot(emissionDirection, -directionToLight));
                            if (emitterCosine <= 0.0f)
                            {
                                continue;
                            }
                        }

                        float visibility = 1.0f;
                        // Reconnect the offset origin to the light sample before tracing visibility.
                        const float3 shadowOrigin = position + hit.Normal * RayEpsilon;
                        const float3 shadowOffset = samplePosition - shadowOrigin;
                        const float shadowDistance = length(shadowOffset);
                        const float shadowMaximumDistance = shadowDistance - RayEpsilon;
                        if (constants.ShadowStrength > 0.0f
                            && light.CastShadows != 0u
                            && shadowMaximumDistance > RayEpsilon)
                        {
                            IncrementCounter(shadowRays);
                            const bool visible = IsLightVisible(
                                geometry,
                                shadowOrigin,
                                shadowOffset / shadowDistance,
                                shadowMaximumDistance,
                                primitiveTests);
                            if (!visible)
                            {
                                visibility = 1.0f - constants.ShadowStrength;
                            }
                        }
                        const float3 incidentRadiance =
                            max(lightColor, float3(0.0f))
                            * (max(light.Intensity, 0.0f)
                                * emitterCosine
                                * sampleWeight
                                / distanceSquared);
                        AccumulateLightSample(
                            directionToLight,
                            incidentRadiance,
                            visibility,
                            hit.Normal,
                            viewDirection,
                            material.AmbientShininess.w,
                            diffuseLight,
                            specularLight);
                    }
                }
                float3 direct = surface * max(material.AmbientShininess.xyz, float3(0.0f))
                    * max(constants.AmbientIntensity, 0.0f)
                    + surface * max(material.AlbedoReflectivity.xyz, float3(0.0f)) * diffuseLight
                    + max(material.Specular.xyz, float3(0.0f)) * specularLight
                    + (includeRegisteredEmission || material.SampledEmitter == 0u
                        ? max(material.Emission.xyz, float3(0.0f)) : float3(0));
                const float reflectivity =
                    clamp(material.AlbedoReflectivity.w, 0.0f, 1.0f);

                radiance += throughput * direct * (1.0f - reflectivity);
                if (reflectivity <= 0.0f
                    || bounce == constants.MaxReflectionBounces)
                {
                    break;
                }
                const float exponent = max(material.AmbientShininess.w, 1.0f);
                const uint dimension = ((bounce + 1u) * BounceSeedMultiplier);
                const float2 reflectionSample = Sample2D(samples, dimension);
                const float lobeCosine = pow(reflectionSample.x, 1.0f / (exponent + 1.0f));
                const float lobeSine = sqrt(max(0.0f, 1.0f - lobeCosine * lobeCosine));
                const float azimuth = 2.0f * Pi * reflectionSample.y;
                const float3 reflection = normalize(reflect(ray.Direction, hit.Normal));
                const float3 helper = abs(reflection.y) < BasisParallelThreshold
                    ? float3(0.0f, 1.0f, 0.0f) : float3(1.0f, 0.0f, 0.0f);
                const float3 tangent = normalize(cross(helper, reflection));
                const float3 bitangent = cross(reflection, tangent);
                const float3 outgoing = normalize(tangent * (lobeSine * cos(azimuth))
                    + bitangent * (lobeSine * sin(azimuth)) + reflection * lobeCosine);
                const float surfaceCosine = max(0.0f, dot(hit.Normal, outgoing));
                if (surfaceCosine <= 0.0f) break;
                throughput *= reflectivity * (exponent + 2.0f) / (exponent + 1.0f) * surfaceCosine;
                ray.Origin = position + hit.Normal * RayEpsilon;
                ray.Direction = outgoing;
                includeRegisteredEmission = true;
                IncrementCounter(reflectionRays);
            }
            AccumulateFilmSample(moments, completedSamples, radiance);
        }
        if (pilot && completedSamples == constants.AdaptivePilotSamples)
        {
            sampleTargets[pixelIndex] = AdaptiveSampleBudget(moments, completedSamples, constants);
            moments = float4(0.0f);
            completedSamples = 0u;
        }
        accumulation[pixelIndex] = moments;
        sampleCounts[pixelIndex] = completedSamples;
    }

    threadgroup uint2 primaryShared[ThreadsPerGroup];
    threadgroup uint2 shadowShared[ThreadsPerGroup];
    threadgroup uint2 reflectionShared[ThreadsPerGroup];
    threadgroup uint2 primitiveShared[ThreadsPerGroup];
    primaryShared[localIndex] = primaryRays;
    shadowShared[localIndex] = shadowRays;
    reflectionShared[localIndex] = reflectionRays;
    primitiveShared[localIndex] = primitiveTests;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = ThreadsPerGroup / 2; stride != 0u; stride >>= 1u)
    {
        if (localIndex < stride)
        {
            primaryShared[localIndex] = AddCounter(
                primaryShared[localIndex],
                primaryShared[localIndex + stride]);
            shadowShared[localIndex] = AddCounter(
                shadowShared[localIndex],
                shadowShared[localIndex + stride]);
            reflectionShared[localIndex] = AddCounter(
                reflectionShared[localIndex],
                reflectionShared[localIndex + stride]);
            primitiveShared[localIndex] = AddCounter(
                primitiveShared[localIndex],
                primitiveShared[localIndex + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (localIndex == 0u)
    {
        GroupStatistics value;
        value.PrimaryRays.Low = primaryShared[0].x;
        value.PrimaryRays.High = primaryShared[0].y;
        value.ShadowRays.Low = shadowShared[0].x;
        value.ShadowRays.High = shadowShared[0].y;
        value.ReflectionRays.Low = reflectionShared[0].x;
        value.ReflectionRays.High = reflectionShared[0].y;
        value.PrimitiveTests.Low = primitiveShared[0].x;
        value.PrimitiveTests.High = primitiveShared[0].y;
        groupStatistics[groupPosition] = value;
    }
}

#include "QueryRays.metal"
