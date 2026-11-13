// Whitted (1980), "An Improved Illumination Model for Shaded Display".
// Independent native Metal direct quadrature and deterministic mirror recursion.
// A depth-first stack evaluates both dielectric branches without roulette.
// Legacy Blinn-Phong surfaces retain their original mirror-chain convention.
// https://pbr-book.org/3ed-2018/Introduction/pbrt_System_Overview
#include <metal_stdlib>
using namespace metal;
#include "SurfaceTypes.metal"
#include "Limits.h"

constant uint LightTypeDirectional = 0u;
constant uint LightTypePoint = 1u;
constant uint LightTypeRectArea = 2u;
constant float RayEpsilon = 1.0e-4f;

constant float Pi = 3.14159265358979323846f;
constant float MaximumRayDistance = 1.0e30f;
constant float BasisParallelThreshold = 0.999f;
constant uint ThreadsPerGroup = 256;
constant uint WhittedStrataPerAxis = 2;
constant uint AreaLightSamples = WhittedStrataPerAxis * WhittedStrataPerAxis;
constant uint PixelSeedMultiplier = 9781u;
constant uint SampleSeedMultiplier = 6271u;
constant uint BounceSeedMultiplier = 0x9e3779b9u;
constant uint LightSeedMultiplier = 0x85ebca6bu;
constant uint LightSampleSeedMultiplier = 0xc2b2ae35u;
constant uint SecondDimensionScramble = 0x68bc21ebu;
constant uint LobeSelectionScramble = 0xa511e9b3u;
constant uint ImportanceBit = 1u << 0;
constant uint SpecularEventCount = 2;
// The largest representable float below one selects transmission unless TIR.
constant float UpperUnitVariate = 0x1.fffffep-1f;
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

#include "BsdfTypes.metal"
#include "Fresnel.metal"
#include "Ggx.metal"
#include "Diffuse.metal"
#include "Conductor.metal"
#include "Dielectric.metal"
#include "SurfaceScattering.metal"
#include "LightConnection.metal"

struct SpecularBranch
{
    RayData Ray;
    float3 Weight;
    uint Depth;
    bool IncludeRegisteredEmission;
};

// Whitted uses fixed area quadrature and enumerates discrete lobes. The common
// visibility leaf traces geometric-side offsets; no stochastic estimator is
// used here. Registered rectangle geometry is owned by these connections.
float3 WhittedDirectLighting(DirectLightContext context, SurfaceBsdf surface, float3 position,
                            thread uint2& shadowRays, thread uint2& primitiveTests)
{
    float3 result = float3(0);
    for (uint index = 0; index < context.LightCount; ++index)
    {
        const PackedLight light = context.Lights[index];
        if (light.Intensity <= 0) continue;
        const float3 center = float3(light.Position), normal = float3(light.Direction);
        const float3 color = float3(light.Color);
        const bool shadows = light.CastShadows != 0u;
        if (light.Type == LightTypeDirectional)
        {
            result += PhysicalDeltaLight(context, surface, position, normal, float3(0),
                color * light.Intensity, false, shadows, shadowRays, primitiveTests);
            continue;
        }
        if (light.Type == LightTypePoint)
        {
            const float3 offset = center - position;
            const float distanceSquared = dot(offset, offset);
            if (distanceSquared <= RayEpsilon * RayEpsilon) continue;
            result += PhysicalDeltaLight(context, surface, position, offset / sqrt(distanceSquared), center,
                color * (light.Intensity / distanceSquared), true, shadows, shadowRays, primitiveTests);
            continue;
        }
        if (light.Type != LightTypeRectArea || light.Width <= 0 || light.Height <= 0) continue;
        const float area = light.Width * light.Height;
        const float3 emitted = color * (light.Intensity / area);
        const float3 helper = abs(normal.y) < BasisParallelThreshold ? float3(0, 1, 0) : float3(1, 0, 0);
        const float3 tangent = normalize(cross(helper, normal)), bitangent = cross(normal, tangent);
        if (SurfaceDelta(surface))
        {
            uint previousEvent = 0;
            for (uint branch = 0; branch < SpecularEventCount; ++branch)
            {
                const SurfaceSample event = SampleSurface(surface, float2(0), branch == 0 ? 0 : UpperUnitVariate);
                if (event.Event == 0u || event.Event == previousEvent || event.Pdf <= 0) continue;
                previousEvent = event.Event;
                const float facing = -dot(normal, event.Direction);
                if (facing <= 0) continue;
                const float distance = dot(position - center, normal) / facing;
                if (distance <= RayEpsilon) continue;
                const float3 endpoint = position + event.Direction * distance, offset = endpoint - center;
                if (abs(dot(offset, tangent)) > light.Width * 0.5f ||
                    abs(dot(offset, bitangent)) > light.Height * 0.5f) continue;
                const float visible = PhysicalVisibility(context, surface, position, event.Direction,
                    endpoint, true, shadows, shadowRays, primitiveTests);
                // Remove stochastic branch compensation: deterministic splitting
                // carries Fresnel energy, including the radiance eta^-2 factor.
                result += event.Weight * emitted * (event.Pdf * visible);
            }
        }
        else
        {
            for (uint sample = 0; sample < AreaLightSamples; ++sample)
            {
                const float2 cell = float2(float(sample % WhittedStrataPerAxis), float(sample / WhittedStrataPerAxis));
                const float2 uv = (cell + float2(0.5f)) / float(WhittedStrataPerAxis) - 0.5f;
                const float3 endpoint = center + tangent * (uv.x * light.Width) + bitangent * (uv.y * light.Height);
                const float3 offset = endpoint - position;
                const float distanceSquared = dot(offset, offset);
                if (distanceSquared <= RayEpsilon * RayEpsilon) continue;
                const float3 incoming = offset / sqrt(distanceSquared);
                const float facing = -dot(normal, incoming);
                if (facing <= 0) continue;
                const SurfaceEvaluation evaluated = EvaluateSurface(surface, incoming);
                if (all(evaluated.Value == float3(0))) continue;
                const float visible = PhysicalVisibility(context, surface, position, incoming, endpoint,
                    true, shadows, shadowRays, primitiveTests);
                result += evaluated.Value * emitted * (abs(dot(surface.Normal, incoming)) * visible *
                    area * facing / (AreaLightSamples * distanceSquared));
            }
        }
    }
    return result;
}

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
    uint gridIndex [[thread_position_in_grid]],
    uint localIndex [[thread_index_in_threadgroup]],
    uint3 groupPosition [[threadgroup_position_in_grid]])
{
    const SceneGeometryBuffers geometry = {{nodes, primitiveRefs, triangles, spheres,
        constants.NodeCount, constants.TriangleCount, constants.Acceleration},
        instances, instanceNodes, instanceError, constants.InstanceNodeCount};
    const DirectLightContext directContext = {geometry, lights, constants.LightCount,
        constants.TransportFlags, constants.ShadowStrength};
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
            ray.MinimumDistance = RayEpsilon;
            ray.MaximumDistance = MaximumRayDistance;
            float3 radiance = float3(0.0f);
            SpecularBranch pending[WhittedBranchCapacity];
            uint pendingCount = 1;
            pending[0] = {ray, float3(1), 0, true};
            while (pendingCount != 0u)
            {
                const SpecularBranch current = pending[--pendingCount];
                ray = current.Ray;
                const uint bounce = current.Depth;
                const float3 throughput = current.Weight;
                HitData hit;
                hit.Distance = ray.MaximumDistance;
                hit.Normal = float3(0.0f);
                hit.Color = float3(1.0f);
                hit.MaterialIndex = 0u;
                hit.TexCoord = float2(0.0f);
                hit.HasTexCoord = 0u;
                if (!IntersectScene(
                        geometry,
                        ray,
                        false,
                        hit,
                        primitiveTests))
                {
                    radiance += throughput * Background(constants, ray.Direction);
                    continue;
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
                    const float3 emission = current.IncludeRegisteredEmission || material.SampledEmitter == 0u
                        ? material.Emission.xyz : float3(0);
                    radiance += throughput * (emission + WhittedDirectLighting(directContext, scattering,
                        position, shadowRays, primitiveTests));
                    if (!SurfaceDelta(scattering) || bounce == constants.MaxReflectionBounces) continue;
                    uint previousEvent = 0;
                    for (uint branch = 0; branch < SpecularEventCount; ++branch)
                    {
                        const SurfaceSample event = SampleSurface(scattering, float2(0),
                            branch == 0 ? 0 : UpperUnitVariate);
                        if (event.Event == 0u || event.Event == previousEvent || event.Pdf <= 0) continue;
                        previousEvent = event.Event;
                        const float3 weight = throughput * event.Weight * event.Pdf;
                        if (all(weight == float3(0))) continue;
                        RayData continuation = ray;
                        continuation.Origin = SurfaceOffset(scattering, position, event.Direction);
                        continuation.Direction = event.Direction;
                        // The host enforces MaximumReflectionBounces; depth-first
                        // expansion needs at most depth+1 pending entries.
                        pending[pendingCount++] = {continuation, weight, bounce + 1u, false};
                        IncrementCounter(reflectionRays);
                    }
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
                            float offsetX = (finiteSample & 1u) != 0u ? (1.0f / AreaLightSamples) : -(1.0f / AreaLightSamples);
                            float offsetY = (finiteSample & 2u) != 0u ? (1.0f / AreaLightSamples) : -(1.0f / AreaLightSamples);

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
                    + (current.IncludeRegisteredEmission || material.SampledEmitter == 0u
                        ? max(material.Emission.xyz, float3(0.0f)) : float3(0));
                const float reflectivity =
                    clamp(material.AlbedoReflectivity.w, 0.0f, 1.0f);

                radiance += throughput * direct * (1.0f - reflectivity);
                if (reflectivity <= 0.0f
                    || bounce == constants.MaxReflectionBounces)
                {
                    continue;
                }
                const float3 reflectedDirection =
                    normalize(reflect(ray.Direction, hit.Normal));
                const float3 reflectionNormal =
                    dot(hit.Normal, reflectedDirection) >= 0.0f
                        ? hit.Normal
                        : -hit.Normal;
                ray.Origin = position + reflectionNormal * RayEpsilon;
                ray.Direction = reflectedDirection;
                pending[pendingCount++] = {ray, throughput * reflectivity, bounce + 1u, true};
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
        groupStatistics[groupPosition.x] = value;
    }
}

#include "QueryRays.metal"
