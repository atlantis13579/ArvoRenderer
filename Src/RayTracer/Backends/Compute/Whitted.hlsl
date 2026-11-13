// Whitted (1980), "An Improved Illumination Model for Shaded Display".
// Native compute direct quadrature and deterministic ideal reflection/transmission.
// The legacy demo material is Blinn-Phong; it is not an energy-normalized BRDF.
// This standalone implementation never dispatches the path tracing integrator.
static const uint DirectionalLightType = 0u;
static const uint PointLightType = 1u;
static const uint RectAreaLightType = 2u;
static const float RayEpsilon = 1.0e-4f;


static const float Pi = 3.14159265358979323846f;
static const float MaximumRayDistance = 1.0e30f;
static const float BasisParallelThreshold = 0.999f;
static const uint ThreadsPerGroup = 256;
static const uint AreaLightSamples = 4;
static const uint AreaLightStrataPerAxis = 2;
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
static const float RandomMantissaRange = 16777216.0f;
static const float MinimumDirectionLengthSquared = 1.0e-12f;

#include "Types.hlsl"
#include "Limits.h"

#include "MaterialTypes.hlsl"

struct PackedTextureDescriptor
{
    uint OffsetTexels;
    uint Width;
    uint Height;
    uint Padding;
};

struct PackedLight
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

ByteAddressBuffer Nodes : register(t0);
StructuredBuffer<uint> PrimitiveRefs : register(t1);
StructuredBuffer<PackedTriangle> Triangles : register(t2);
StructuredBuffer<PackedSphere> Spheres : register(t3);
StructuredBuffer<PackedMaterial> Materials : register(t4);
StructuredBuffer<PackedLight> Lights : register(t5);
StructuredBuffer<PackedTextureDescriptor> Textures : register(t6);
StructuredBuffer<uint> TextureTexels : register(t7);
RWStructuredBuffer<float4> Accumulation : register(u0);
RWStructuredBuffer<uint> SampleCounts : register(u1);
RWStructuredBuffer<uint> OutputPixels : register(u2);
RWByteAddressBuffer Statistics : register(u3);
RWStructuredBuffer<uint> SampleTargets : register(u4);

cbuffer Globals : register(b0)
{
    float3 CameraPosition;
    uint Width;
    float3 CameraForward;
    uint Height;
    float3 CameraRight;
    uint TotalSamples;
    float3 CameraUp;
    uint MaxReflectionBounces;
    float HalfWidth;
    float HalfHeight;
    uint MaterialCount;
    uint LightCount;
    float3 BackgroundTop;
    float AmbientIntensity;
    float3 BackgroundBottom;
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
};

uint Hash(uint value)
{
    value ^= value >> 16;
    value *= AvalancheMultiplierA;
    value ^= value >> 15;
    value *= AvalancheMultiplierB;
    value ^= value >> 16;
    return value;
}

float Random01(uint seed)
{
    return float(Hash(seed) & RandomMantissaMask) / RandomMantissaRange;
}

#include "Sequence.hlsl"
#include "FilmMoments.hlsl"
#include "Adaptive.hlsl"
#include "ThinLens.hlsl"

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

float3 LoadDiffuseTexel(PackedTextureDescriptor descriptor, int x, int y)
{
    const uint offset = descriptor.OffsetTexels
        + uint(WrapTextureCoordinate(y, descriptor.Height)) * descriptor.Width
        + uint(WrapTextureCoordinate(x, descriptor.Width));
    const uint packed = TextureTexels[offset];
    return float3(
        SRGBToLinear(float(packed & 0xffu) / 255.0f),
        SRGBToLinear(float((packed >> 8u) & 0xffu) / 255.0f),
        SRGBToLinear(float((packed >> 16u) & 0xffu) / 255.0f));
}

float3 SampleDiffuse(uint textureIndex, float2 texCoord, uint hasTexCoord)
{
    if (hasTexCoord == 0u || textureIndex >= TextureCount)
    {
        return 1.0f.xxx;
    }
    const PackedTextureDescriptor descriptor = Textures[textureIndex];
    if (descriptor.Width == 0u || descriptor.Height == 0u)
    {
        return 1.0f.xxx;
    }
    if (descriptor.Height > TextureTexelCount / descriptor.Width)
    {
        return 1.0f.xxx;
    }
    const uint pixelCount = descriptor.Width * descriptor.Height;
    if (descriptor.OffsetTexels > TextureTexelCount
        || pixelCount > TextureTexelCount - descriptor.OffsetTexels)
    {
        return 1.0f.xxx;
    }
    const float x = frac(texCoord.x) * float(descriptor.Width) - 0.5f;
    const float y = frac(1.0f - texCoord.y) * float(descriptor.Height) - 0.5f;
    const int x0 = int(floor(x));
    const int y0 = int(floor(y));
    const float amountX = x - floor(x);
    const float amountY = y - floor(y);
    const float3 top = lerp(
        LoadDiffuseTexel(descriptor, x0, y0),
        LoadDiffuseTexel(descriptor, x0 + 1, y0),
        amountX);
    const float3 bottom = lerp(
        LoadDiffuseTexel(descriptor, x0, y0 + 1),
        LoadDiffuseTexel(descriptor, x0 + 1, y0 + 1),
        amountX);
    return lerp(top, bottom, amountY);
}

void IncrementCounter(inout uint2 counter)
{
    const uint previousLow = counter.x;
    counter.x += 1u;
    if (counter.x < previousLow) counter.y += 1u;
}

uint2 AddCounter(uint2 left, uint2 right)
{
    const uint low = left.x + right.x;
    return uint2(low, left.y + right.y + (low < left.x ? 1u : 0u));
}

void AtomicAddCounter(uint byteOffset, uint2 value)
{
    uint previousLow;
    Statistics.InterlockedAdd(byteOffset, value.x, previousLow);
    const uint carry = previousLow + value.x < previousLow ? 1u : 0u;
    uint ignored;
    Statistics.InterlockedAdd(byteOffset + 4u, value.y + carry, ignored);
}

#include "Bounds.hlsl"
#include "Triangle.hlsl"
#include "Sphere.hlsl"
#include "Traversal.hlsl"
#include "BsdfTypes.hlsl"
#include "Fresnel.hlsl"
#include "Ggx.hlsl"
#include "Diffuse.hlsl"
#include "Conductor.hlsl"
#include "Dielectric.hlsl"
#include "SurfaceScattering.hlsl"

static const uint SpecularBranchCount = 2;
struct WhittedBranch
{
    RayData Ray;
    float3 Weight;
    uint Depth;
    uint IncludeRegisteredEmission;
};
struct WhittedEvents
{
    SurfaceSample Values[SpecularBranchCount];
};

// Evaluate the endpoints of the discrete lobe CDF, without drawing a variate.
// Pdf is a probability mass here: Weight*Pdf is each branch's f*abs(cos).
// TIR, matched indices and ideal conductors must not duplicate their one event.
WhittedEvents WhittedDeltaEvents(SurfaceBsdf surface)
{
    WhittedEvents events = (WhittedEvents)0;
    if (!SurfaceDelta(surface)) return events;
    const float lastCdfVariate = asfloat(asuint(1.0f) - 1u);
    events.Values[0] = SampleSurface(surface, 0.0f.xx, 0);
    events.Values[1] = SampleSurface(surface, 0.0f.xx, lastCdfVariate);
    if (events.Values[0].Event == events.Values[1].Event) events.Values[1] = (SurfaceSample)0;
    return events;
}
float WhittedVisibility(SurfaceBsdf surface, float3 position, float3 direction,
    float3 endpoint, bool finiteLight, bool shadows, inout uint2 shadowRays, inout uint2 primitiveTests)
{
    if (!shadows || ShadowStrength <= 0) return 1;
    RayData segment;
    segment.Origin = SurfaceOffset(surface, position, direction);
    segment.Direction = direction;
    segment.MinimumDistance = RayEpsilon;
    segment.MaximumDistance = MaximumRayDistance;
    if (finiteLight)
    {
        const float3 toEndpoint = endpoint - segment.Origin;
        const float distance = length(toEndpoint);
        segment.MaximumDistance = distance - RayEpsilon;
        if (segment.MaximumDistance <= segment.MinimumDistance) return 1;
        segment.Direction = toEndpoint / distance;
    }
    HitData blocker = (HitData)0;
    blocker.Distance = segment.MaximumDistance;
    blocker.PrimitiveIndex = InvalidPrimitiveIndex;
    IncrementCounter(shadowRays);
    return IntersectScene(segment, true, blocker, primitiveTests) ? 1 - saturate(ShadowStrength) : 1;
}
float3 WhittedDiffuseLight(SurfaceBsdf surface, float3 position, float3 direction, float3 radiance,
    float3 endpoint, bool finiteLight, bool shadows, float weight,
    inout uint2 shadowRays, inout uint2 primitiveTests)
{
    const SurfaceEvaluation scattering = EvaluateSurface(surface, direction);
    if (all(scattering.Value == 0)) return 0.0f.xxx;
    const float visible = WhittedVisibility(surface, position, direction, endpoint, finiteLight,
                                           shadows, shadowRays, primitiveTests);
    return scattering.Value * radiance * (abs(dot(surface.Normal, direction)) * weight * visible);
}
// Whitted's direct term uses fixed area quadrature for diffuse reflectance and
// exact directional connections for ideal events. This owns registered emission
// after physical continuation, independently of the stochastic direct estimator.
float3 WhittedPhysicalLighting(SurfaceBsdf surface, WhittedEvents events, float3 position,
    inout uint2 shadowRays, inout uint2 primitiveTests)
{
    float3 result = 0.0f.xxx;
    [loop]
    for (uint index = 0; index < LightCount; ++index)
    {
        const PackedLight light = Lights[index];
        if (light.Intensity <= 0) continue;
        if (light.Type == DirectionalLightType)
        {
            result += WhittedDiffuseLight(surface, position, light.Direction, light.Color * light.Intensity,
                0.0f.xxx, false, light.CastShadows != 0u, 1, shadowRays, primitiveTests);
            continue;
        }
        if (light.Type == PointLightType)
        {
            const float3 offset = light.Position - position;
            const float distanceSquared = dot(offset, offset);
            if (distanceSquared <= RayEpsilon * RayEpsilon) continue;
            result += WhittedDiffuseLight(surface, position, offset / sqrt(distanceSquared),
                light.Color * (light.Intensity / distanceSquared), light.Position, true,
                light.CastShadows != 0u, 1, shadowRays, primitiveTests);
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
            [unroll]
            for (uint eventIndex = 0; eventIndex < SpecularBranchCount; ++eventIndex)
            {
                const SurfaceSample event = events.Values[eventIndex];
                if (event.Event == 0u || event.Pdf <= 0) continue;
                const float facing = -dot(light.Direction, event.Direction);
                if (facing <= 0) continue;
                const float distance = dot(position - light.Position, light.Direction) / facing;
                if (distance <= RayEpsilon) continue;
                const float3 endpoint = position + distance * event.Direction;
                const float3 offset = endpoint - light.Position;
                if (abs(dot(offset, tangent)) > light.Width * 0.5f ||
                    abs(dot(offset, bitangent)) > light.Height * 0.5f) continue;
                const float visible = WhittedVisibility(surface, position, event.Direction, endpoint, true,
                    light.CastShadows != 0u, shadowRays, primitiveTests);
                result += event.Weight * emitted * (event.Pdf * visible);
            }
            continue;
        }
        [loop]
        for (uint sampleIndex = 0; sampleIndex < AreaLightSamples; ++sampleIndex)
        {
            const float2 cell = float2(sampleIndex % AreaLightStrataPerAxis,
                                      sampleIndex / AreaLightStrataPerAxis);
            const float2 uv = (cell + 0.5f) / AreaLightStrataPerAxis - 0.5f;
            const float3 endpoint = light.Position + tangent * (uv.x * light.Width)
                + bitangent * (uv.y * light.Height);
            const float3 offset = endpoint - position;
            const float distanceSquared = dot(offset, offset);
            if (distanceSquared <= RayEpsilon * RayEpsilon) continue;
            const float3 direction = offset / sqrt(distanceSquared);
            const float facing = -dot(light.Direction, direction);
            if (facing <= 0) continue;
            const float weight = area * facing / (AreaLightSamples * distanceSquared);
            result += WhittedDiffuseLight(surface, position, direction, emitted, endpoint, true,
                light.CastShadows != 0u, weight, shadowRays, primitiveTests);
        }
    }
    return result;
}

float3 Background(float3 direction)
{
    const float blend = saturate(direction.y * 0.5f + 0.5f);
    return lerp(BackgroundBottom, BackgroundTop, blend);
}

float3 LinearToSRGB(float3 color)
{
    color = saturate(color);
    const float3 low = color * 12.92f;
    const float3 high = 1.055f * pow(color, 1.0f / 2.4f) - 0.055f;
    return float3(
        color.r <= 0.0031308f ? low.r : high.r,
        color.g <= 0.0031308f ? low.g : high.g,
        color.b <= 0.0031308f ? low.b : high.b);
}

void AccumulateLightSample(
    float3 lightDirection,
    float3 radiance,
    float visibility,
    float3 surfaceNormal,
    float3 viewDirection,
    float3 diffuse,
    float3 specular,
    float shininess,
    inout float3 direct)
{
    const float directionLengthSquared = dot(lightDirection, lightDirection);
    if (!isfinite(directionLengthSquared) || directionLengthSquared <= MinimumDirectionLengthSquared)
    {
        return;
    }
    const float3 direction = lightDirection * rsqrt(directionLengthSquared);
    const float normalToLight = max(dot(surfaceNormal, direction), 0.0f);
    if (normalToLight <= 0.0f)
    {
        return;
    }
    const float3 visibleRadiance = radiance * saturate(visibility);
    direct += diffuse * visibleRadiance * normalToLight;
    const float3 halfVector = normalize(direction + viewDirection);
    const float specularAmount = pow(
        max(dot(surfaceNormal, halfVector), 0.0f),
        max(shininess, 1.0f));
    direct += specular * visibleRadiance * specularAmount;
}

void AccumulateFiniteLight(
    PackedLight light,
    float3 lightPosition,
    float emitterWeight,
    bool areaEmitter,
    float3 position,
    float3 surfaceNormal,
    float3 viewDirection,
    float3 diffuse,
    float3 specular,
    float shininess,
    inout float3 direct,
    inout uint2 shadowRays,
    inout uint2 primitiveTests)
{
    if (light.Intensity <= 0.0f || emitterWeight <= 0.0f)
    {
        return;
    }
    const float3 toLight = lightPosition - position;
    const float distanceSquared = dot(toLight, toLight);
    if (!isfinite(distanceSquared)
        || distanceSquared <= RayEpsilon * RayEpsilon)
    {
        return;
    }
    const float distance = sqrt(distanceSquared);
    const float3 directionToLight = toLight / distance;
    const float surfaceCosine =
        max(dot(surfaceNormal, directionToLight), 0.0f);
    if (surfaceCosine <= 0.0f)
    {
        return;
    }
    float emitterCosine = 1.0f;
    if (areaEmitter)
    {
        emitterCosine =
            max(dot(light.Direction, -directionToLight), 0.0f);
        if (emitterCosine <= 0.0f)
        {
            return;
        }
    }

    float visibility = 1.0f;
    if (ShadowStrength > 0.0f && light.CastShadows != 0u)
    {
        RayData shadowRay;
        shadowRay.Origin = position + surfaceNormal * RayEpsilon;
        // Visibility follows the offset-origin segment, excluding the emitter endpoint.
        const float3 shadowOffset = lightPosition - shadowRay.Origin;
        const float shadowDistance = length(shadowOffset);
        shadowRay.MinimumDistance = RayEpsilon;
        shadowRay.MaximumDistance = shadowDistance - RayEpsilon;
        if (shadowRay.MaximumDistance > shadowRay.MinimumDistance)
        {
            shadowRay.Direction = shadowOffset / shadowDistance;
            IncrementCounter(shadowRays);
            HitData shadowHit = (HitData)0;
            shadowHit.Distance = shadowRay.MaximumDistance;
            shadowHit.PrimitiveIndex = InvalidPrimitiveIndex;
            shadowHit.Normal = 0.0f.xxx;
            shadowHit.Color = 1.0f.xxx;
            shadowHit.MaterialIndex = 0u;
            if (IntersectScene(
                    shadowRay,
                    true,
                    shadowHit,
                    primitiveTests))
            {
                visibility = 1.0f - ShadowStrength;
            }
        }
    }
    const float attenuation =
        light.Intensity * emitterCosine * emitterWeight / distanceSquared;
    AccumulateLightSample(
        directionToLight,
        light.Color * attenuation,
        visibility,
        surfaceNormal,
        viewDirection,
        diffuse,
        specular,
        shininess,
        direct);
}

groupshared uint2 PrimaryShared[ThreadsPerGroup];
groupshared uint2 ShadowShared[ThreadsPerGroup];
groupshared uint2 ReflectionShared[ThreadsPerGroup];
groupshared uint2 PrimitiveShared[ThreadsPerGroup];

[numthreads(ThreadsPerGroup, 1, 1)]
void TracePixelsKernel(
    uint3 dispatchThreadId : SV_DispatchThreadID,
    uint groupIndex : SV_GroupIndex)
{
    uint2 primaryRays = uint2(0, 0);
    uint2 shadowRays = uint2(0, 0);
    uint2 reflectionRays = uint2(0, 0);
    uint2 primitiveTests = uint2(0, 0);
    if (dispatchThreadId.x < PixelCount)
    {
        const uint pixelIndex = PixelBegin + dispatchThreadId.x;
        const uint x = pixelIndex % Width;
        const uint y = pixelIndex / Width;
        float4 moments = Accumulation[pixelIndex];
        uint completedSamples = SampleCounts[pixelIndex];
        const bool pilot = AdaptiveEnabled != 0u && AdaptivePhase == AdaptivePilotPhase;
        const uint targetSamples = pilot ? AdaptivePilotSamples : SampleTargets[pixelIndex];
        [loop]
        for (uint sampleOffset = 0; sampleOffset < SampleCount; ++sampleOffset)
        {
            const uint sample = SampleBegin + sampleOffset;
            if (sample >= targetSamples) break;
            IncrementCounter(primaryRays);
            const SampleContext samples = AdaptiveSamples(x + TileX, y + TileY, sample, targetSamples, pilot);
            float jitterX = 0.5f;
            float jitterY = 0.5f;
            if (targetSamples > 1u || SamplingSequence != IndependentSequence || SamplingSeed != 0)
            {
                const float2 pixelSample = Sample2D(samples, 0u);
                jitterX = pixelSample.x;
                jitterY = pixelSample.y;
            }
            const float u = (float(x + TileX) + jitterX) / float(FullWidth);
            const float v = (float(y + TileY) + jitterY) / float(FullHeight);
            const float screenX = (2.0f * u - 1.0f) * HalfWidth;
            const float screenY = (1.0f - 2.0f * v) * HalfHeight;
            RayData ray;
            ray.Origin = CameraPosition;
            ray.Direction = normalize(CameraForward + CameraRight * screenX + CameraUp * screenY);
            if (LensRadius > 0) ApplyThinLens(ray, Sample2D(samples, LensSampleDimension));
            ray.MinimumDistance = RayEpsilon;
            ray.MaximumDistance = MaximumRayDistance;
            float3 radiance = 0.0f.xxx;
            // A depth-first traversal stores one deferred sibling per level.
            // Host validation and this shader share the capacity bound; no branch
            // is discarded for low energy, and total work can grow exponentially.
            WhittedBranch pending[WhittedBranchCapacity];
            WhittedBranch root = (WhittedBranch)0;
            root.Ray = ray;
            root.Weight = 1.0f.xxx;
            root.IncludeRegisteredEmission = 1u;
            pending[0] = root;
            uint pendingCount = 1;
            [loop]
            while (pendingCount != 0)
            {
                const WhittedBranch branch = pending[--pendingCount];
                ray = branch.Ray;
                const float3 throughput = branch.Weight;
                const uint bounce = branch.Depth;
                HitData hit = (HitData)0;
                hit.Distance = ray.MaximumDistance;
                hit.PrimitiveIndex = InvalidPrimitiveIndex;
                hit.Normal = 0.0f.xxx;
                hit.Color = 1.0f.xxx;
                hit.MaterialIndex = 0;
                hit.TexCoord = 0.0f.xx;
                hit.HasTexCoord = 0u;
                if (!IntersectScene(ray, false, hit, primitiveTests))
                {
                    radiance += throughput * Background(ray.Direction);
                    continue;
                }
                const PackedMaterial material = Materials[
                    hit.MaterialIndex < MaterialCount ? hit.MaterialIndex : 0];
                const float3 surface = hit.Color * SampleDiffuse(
                        material.DiffuseTextureIndex,
                        hit.TexCoord,
                        hit.HasTexCoord);
                const float3 diffuse = surface * material.AlbedoReflectivity.xyz;
                const float3 position = ray.Origin + ray.Direction * hit.Distance;
                const float3 viewDirection = normalize(-ray.Direction);
                if (material.Model != MaterialLegacy)
                {
                    const SurfaceBsdf surfaceBsdf = MakeSurfaceBsdf(material, hit, viewDirection, surface);
                    const WhittedEvents events = WhittedDeltaEvents(surfaceBsdf);
                    const float3 emission = branch.IncludeRegisteredEmission != 0u || material.SampledEmitter == 0u
                        ? material.Emission.xyz : 0.0f.xxx;
                    radiance += throughput * (emission + WhittedPhysicalLighting(surfaceBsdf, events,
                        position, shadowRays, primitiveTests));
                    if (bounce == MaxReflectionBounces) continue;
                    [unroll]
                    for (uint eventIndex = 0; eventIndex < SpecularBranchCount; ++eventIndex)
                    {
                        const SurfaceSample event = events.Values[eventIndex];
                        if (event.Event == 0u || event.Pdf <= 0) continue;
                        const float3 weight = throughput * event.Weight * event.Pdf;
                        if (all(weight == 0)) continue;
                        WhittedBranch continuation;
                        continuation.Ray.Origin = SurfaceOffset(surfaceBsdf, position, event.Direction);
                        continuation.Ray.Direction = event.Direction;
                        continuation.Ray.MinimumDistance = RayEpsilon;
                        continuation.Ray.MaximumDistance = MaximumRayDistance;
                        continuation.Weight = weight;
                        continuation.Depth = bounce + 1;
                        continuation.IncludeRegisteredEmission = 0u;
                        pending[pendingCount++] = continuation;
                        IncrementCounter(reflectionRays);
                    }
                    continue;
                }
                float3 direct = material.Emission.xyz
                    + surface * material.AmbientShininess.xyz
                        * max(0.0f, AmbientIntensity);
                [loop]
                for (uint lightIndex = 0; lightIndex < LightCount; ++lightIndex)
                {
                    const PackedLight light = Lights[lightIndex];
                    if (light.Type == DirectionalLightType)
                    {
                        const float3 directionToLight = light.Direction;
                        const float cosine =
                            max(0.0f, dot(hit.Normal, directionToLight));
                        if (cosine <= 0.0f || light.Intensity <= 0.0f)
                        {
                            continue;
                        }
                        float visibility = 1.0f;
                        if (ShadowStrength > 0.0f && light.CastShadows != 0u)
                        {
                            IncrementCounter(shadowRays);
                            RayData shadowRay;
                            const float3 shadowNormal =
                                dot(hit.Normal, directionToLight) >= 0.0f
                                    ? hit.Normal
                                    : -hit.Normal;
                            shadowRay.Origin =
                                position + shadowNormal * RayEpsilon;
                            shadowRay.Direction = directionToLight;
                            shadowRay.MinimumDistance = RayEpsilon;
                            shadowRay.MaximumDistance = MaximumRayDistance;
                            HitData shadowHit = (HitData)0;
                            shadowHit.Distance = shadowRay.MaximumDistance;
                            shadowHit.PrimitiveIndex = InvalidPrimitiveIndex;
                            shadowHit.Normal = 0.0f.xxx;
                            shadowHit.Color = 1.0f.xxx;
                            shadowHit.MaterialIndex = 0u;
                            if (IntersectScene(
                                    shadowRay,
                                    true,
                                    shadowHit,
                                    primitiveTests))
                            {
                                visibility = 1.0f - ShadowStrength;
                            }
                        }
                        AccumulateLightSample(
                            directionToLight,
                            light.Color * light.Intensity,
                            visibility,
                            hit.Normal,
                            viewDirection,
                            diffuse,
                            material.Specular.xyz,
                            material.AmbientShininess.w,
                            direct);
                    }
                    else if (light.Type == PointLightType)
                    {
                        AccumulateFiniteLight(
                            light,
                            light.Position,
                            1.0f,
                            false,
                            position,
                            hit.Normal,
                            viewDirection,
                            diffuse,
                            material.Specular.xyz,
                            material.AmbientShininess.w,
                            direct,
                            shadowRays,
                            primitiveTests);
                    }
                    else if (light.Type == RectAreaLightType)
                    {
                        const float3 emissionDirection = light.Direction;
                        const float3 helper =
                            abs(emissionDirection.y) < BasisParallelThreshold
                            ? float3(0.0f, 1.0f, 0.0f)
                            : float3(1.0f, 0.0f, 0.0f);
                        const float3 tangent =
                            normalize(cross(helper, emissionDirection));
                        const float3 bitangent =
                            normalize(cross(emissionDirection, tangent));
                        [unroll]
                        for (uint sampleIndex = 0u; sampleIndex < AreaLightSamples; ++sampleIndex)
                        {
                            float offsetX = (sampleIndex & 1u) != 0u ? (1.0f / AreaLightSamples) : -(1.0f / AreaLightSamples);
                            float offsetY = (sampleIndex & 2u) != 0u ? (1.0f / AreaLightSamples) : -(1.0f / AreaLightSamples);

                            const float3 samplePosition =
                                light.Position
                                + tangent * (offsetX * light.Width)
                                + bitangent * (offsetY * light.Height);
                            AccumulateFiniteLight(
                                light,
                                samplePosition,
                                (1.0f / AreaLightSamples),
                                true,
                                position,
                                hit.Normal,
                                viewDirection,
                                diffuse,
                                material.Specular.xyz,
                                material.AmbientShininess.w,
                                direct,
                                shadowRays,
                                primitiveTests);
                        }
                    }
                }
                if (branch.IncludeRegisteredEmission == 0u && material.SampledEmitter != 0u)
                    direct -= material.Emission.xyz;
                const float reflectivity = saturate(material.AlbedoReflectivity.w);

                radiance += throughput * direct * (1.0f - reflectivity);
                if (reflectivity <= 0.0f || bounce == MaxReflectionBounces) continue;
                const float3 reflectedDirection =
                    normalize(reflect(ray.Direction, hit.Normal));
                const float3 reflectionNormal =
                    dot(hit.Normal, reflectedDirection) >= 0.0f
                        ? hit.Normal
                        : -hit.Normal;
                ray.Origin = position + reflectionNormal * RayEpsilon;
                ray.Direction = reflectedDirection;
                WhittedBranch continuation;
                continuation.Ray = ray;
                continuation.Weight = throughput * reflectivity;
                continuation.Depth = bounce + 1;
                continuation.IncludeRegisteredEmission = 1u;
                pending[pendingCount++] = continuation;
                IncrementCounter(reflectionRays);
            }
            AccumulateFilmSample(moments, completedSamples, radiance);
            if (pilot && completedSamples == AdaptivePilotSamples)
            {
                SampleTargets[pixelIndex] = AdaptiveSampleBudget(moments, completedSamples);
                moments = 0.0f.xxxx;
                completedSamples = 0u;
            }
        }
        Accumulation[pixelIndex] = moments;
        SampleCounts[pixelIndex] = completedSamples;
    }

    PrimaryShared[groupIndex] = primaryRays;
    ShadowShared[groupIndex] = shadowRays;
    ReflectionShared[groupIndex] = reflectionRays;
    PrimitiveShared[groupIndex] = primitiveTests;
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (uint stride = ThreadsPerGroup / 2; stride != 0; stride >>= 1)
    {
        if (groupIndex < stride)
        {
            PrimaryShared[groupIndex] = AddCounter(
                PrimaryShared[groupIndex], PrimaryShared[groupIndex + stride]);
            ShadowShared[groupIndex] = AddCounter(
                ShadowShared[groupIndex], ShadowShared[groupIndex + stride]);
            ReflectionShared[groupIndex] = AddCounter(
                ReflectionShared[groupIndex], ReflectionShared[groupIndex + stride]);
            PrimitiveShared[groupIndex] = AddCounter(
                PrimitiveShared[groupIndex], PrimitiveShared[groupIndex + stride]);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (groupIndex == 0)
    {
        AtomicAddCounter(0, PrimaryShared[0]);
        AtomicAddCounter(8, ShadowShared[0]);
        AtomicAddCounter(16, ReflectionShared[0]);
        AtomicAddCounter(24, PrimitiveShared[0]);
    }
}


#include "QueryRays.hlsl"
