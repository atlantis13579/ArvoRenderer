// Kajiya (1986), "The Rendering Equation". Native compute path transport.
// Physical materials use f*abs(cos)/pdf continuation and explicit delta flags;
// legacy materials retain their authored diffuse/mirror mixture. The independent
// direct estimator owns registered emission for every physical event measure.
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
static const uint PixelSeedMultiplier = 9781u;
static const uint SampleSeedMultiplier = 6271u;
static const uint BounceSeedMultiplier = 0x9e3779b9u;
static const uint LightSeedMultiplier = 0x85ebca6bu;
static const uint LightSampleSeedMultiplier = 0xc2b2ae35u;
static const uint SecondDimensionScramble = 0x68bc21ebu;
static const uint LobeSelectionScramble = 0xa511e9b3u;
static const uint DirectBsdfScramble = 0xd1b54a35u;
static const uint RouletteScramble = 0x94d049bbu;
static const uint ImportanceBit = 1u << 0;
static const uint NeeBit = 1u << 1;
static const uint RouletteBit = 1u << 2;
static const uint MisShift = 3u;
static const uint MisMask = 3u;
static const uint NoMis = 0u;
static const uint PowerMis = 2u;

static const uint AvalancheMultiplierA = 0x7feb352du;
static const uint AvalancheMultiplierB = 0x846ca68bu;
static const uint RandomMantissaMask = 0x00ffffffu;
static const float RandomMantissaRange = 16777216.0f;
static const float MinimumDirectionLengthSquared = 1.0e-12f;

#define ARVO_SPECTRAL_TRANSPORT
#include "Types.hlsl"

#include "MaterialTypes.hlsl"
#include "HomogeneousMedium.hlsl"
#include "SubsurfaceScattering.hlsl"
#define ARVO_PARTICIPATING_MEDIA
#include "ShutterSampling.hlsl"
#include "MotionInterpolation.hlsl"
#include "CameraMotion.hlsl"
cbuffer MediumBlock : register(b1)
{
    MediumParameters ParticipatingMedium;
    MotionExposure CameraExposure;
}

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

// Veach (1997), ch. 9: the two densities include their sample counts.
#include "MultipleImportance.hlsl"
#include "Importance.hlsl"
#include "RussianRoulette.hlsl"
float AreaSampleWeight(PackedLight light, float3 position, float3 normal, float3 samplePosition)
{
    if (((TransportFlags >> MisShift) & MisMask) == NoMis) return 1.0f / AreaLightSamples;
    const float3 offset = samplePosition - position;
    const float3 direction = normalize(offset);
    const float emitterCosine = dot(light.Direction, -direction);
    if (emitterCosine <= 0.0f) return 0.0f;
    const float lightPdf = dot(offset,offset) / (light.Width*light.Height*emitterCosine);
    return MisWeight(AreaLightSamples*lightPdf, DiffusePdf(dot(normal,direction))) / AreaLightSamples;
}

// Direct BSDF sampling is independent of continuation. Registered emitter hits
// after diffuse continuation stay excluded, preventing a second direct estimate.
bool BsdfEmitterSample(PackedLight light, float3 position, float3 normal, float2 variates,
    float3 tangent, float3 bitangent, out float3 samplePosition, out float weight)
{
    samplePosition = position;
    weight = 0.0f;
    const float3 direction = SampleDiffuseDirection(normal,variates);
    const float emitterCosine = dot(light.Direction,-direction);
    if (emitterCosine <= 0.0f) return false;
    const float distance = dot(position-light.Position,light.Direction) / emitterCosine;
    samplePosition = position + direction*distance;
    const float3 offset = samplePosition-light.Position;
    const float bsdfPdf = DiffusePdf(dot(normal,direction));
    if (distance <= RayEpsilon || bsdfPdf <= 0.0f || abs(dot(offset,tangent)) > light.Width*0.5f
        || abs(dot(offset,bitangent)) > light.Height*0.5f) return false;
    const float lightPdf = distance*distance / (light.Width*light.Height*emitterCosine);
    weight = lightPdf/bsdfPdf * MisWeight(bsdfPdf,AreaLightSamples*lightPdf);
    return true;
}

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

#include "WavelengthSampling.hlsl"
#include "SpectralColor.hlsl"
#include "RgbSpectrum.hlsl"
#include "SpectralFiltering.hlsl"
#include "SpectralTransport.hlsl"

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
#include "DirectLighting.hlsl"
#include "VolumeLighting.hlsl"

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
    direct += diffuse * visibleRadiance * normalToLight
        * (1.0f / Pi);

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
    inout uint2 primitiveTests,
    float wavelength, float sceneTime)
{
    const MediumParameters medium = SpectralMedium(ParticipatingMedium, wavelength);
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
                    primitiveTests, sceneTime))
            {
                visibility = 1.0f - ShadowStrength;
            }
        }
    }
    const float attenuation =
        light.Intensity * emitterCosine * emitterWeight / distanceSquared;
    AccumulateLightSample(
        directionToLight,
        light.Color * attenuation * MediumTransmission(medium, position, directionToLight, 0, distance),
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

MediumParameters SubsurfaceInterior(PackedMaterial material)
{
    MediumParameters result = (MediumParameters)0;
    result.Enabled = 1;
    result.Minimum = float3(-MaximumRayDistance, -MaximumRayDistance, -MaximumRayDistance);
    result.Maximum = float3(MaximumRayDistance, MaximumRayDistance, MaximumRayDistance);
    result.Anisotropy = material.ConductorIndex.w;
    const float3 albedo = material.AlbedoReflectivity.xyz * material.ConductorExtinction.w;
    result.Absorption = (1 - albedo) / material.Optical.w;
    result.Scattering = albedo / material.Optical.w;
    return result;
}
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
            float sceneTime = 0;
            if (CameraExposure.Enabled != 0 || CameraExposure.SceneEnabled != 0)
            {
                float timeVariate = CameraExposure.Open == CameraExposure.Close
                                        ? 0
                                        : Sample2D(samples, TimeSampleDimension).x;
                MotionCameraRay moved = SampleCameraMotion(ray.Origin, ray.Direction, CameraPosition,
                                                           CameraExposure, timeVariate);
                ray.Origin = moved.Origin;
                ray.Direction = moved.Direction;
                sceneTime = moved.SceneTime;
            }
            ray.MinimumDistance = RayEpsilon;
            ray.MaximumDistance = MaximumRayDistance;
            const WavelengthPacket wavelengths = BeginSpectralPath(samples);
            const float wavelength = wavelengths.Lambda[0];
            const MediumParameters exteriorMedium = SpectralMedium(ParticipatingMedium, wavelength);
            uint interiorMaterial = InvalidPrimitiveIndex;
            float3 radiance = 0.0f.xxx;
            float3 throughput = 1.0f.xxx;
            bool includeRegisteredEmission = true;
            float etaScale = 1;
            [loop]
            for (uint bounce = 0; bounce <= MaxReflectionBounces; ++bounce)
            {
                HitData hit = (HitData)0;
                hit.Distance = ray.MaximumDistance;
                hit.PrimitiveIndex = InvalidPrimitiveIndex;
                hit.Normal = 0.0f.xxx;
                hit.Color = 1.0f.xxx;
                hit.MaterialIndex = 0;
                hit.TexCoord = 0.0f.xx;
                hit.HasTexCoord = 0u;
                const bool found = IntersectScene(ray, false, hit, primitiveTests, sceneTime);
                if (bounce == 0 && found && hit.FrontFace == 0 && Materials[hit.MaterialIndex].Optical.w > 0)
                    interiorMaterial = hit.MaterialIndex;
                MediumParameters medium = exteriorMedium;
                PackedMaterial interior = (PackedMaterial)0;
                if (interiorMaterial != InvalidPrimitiveIndex)
                {
                    interior = SpectralMaterial(Materials[interiorMaterial], interiorMaterial, wavelength);
                    medium = SubsurfaceInterior(interior);
                }
                if (medium.Enabled != 0)
                {
                    const uint FreeFlightScramble = 0x7a4d2e91u, PhaseScramble = 0x3c6ef372u;
                    const uint domain = (bounce + 1) * BounceSeedMultiplier;
                    MediumFlight flight;
                    const float2 freeFlight = Sample2D(samples, domain ^ FreeFlightScramble);
                    if (interior.Optical.w > 0)
                    {
                        const SubsurfaceFlight event = SubsurfaceFreeFlight(interior.Optical.w,
                            interior.AlbedoReflectivity.xyz * interior.ConductorExtinction.w, ray.Direction,
                            found ? hit.Distance : ray.MaximumDistance, freeFlight.x);
                        flight.Weight = event.Weight; flight.Distance = event.Distance; flight.Scattered = event.Scattered;
                    }
                    else flight = MediumFreeFlight(medium, ray.Origin, ray.Direction,
                        0, found ? hit.Distance : ray.MaximumDistance, freeFlight);
                    throughput *= flight.Weight;
                    if (all(throughput == 0)) break;
                    if (flight.Scattered != 0)
                    {
                        const float3 position = ray.Origin + ray.Direction * flight.Distance;
                        radiance += throughput * VolumeDirectLighting(medium, position, ray.Direction, samples, bounce, shadowRays, primitiveTests, wavelength, sceneTime);
                        if (bounce == MaxReflectionBounces) break;
                        float3 direction;
                        if (interior.Optical.w > 0)
                        {
                            const SubsurfaceDirection phase = SubsurfaceSampleDirection(ray.Direction, medium.Anisotropy,
                                Sample2D(samples, domain ^ PhaseScramble), (TransportFlags & ImportanceBit) != 0);
                            direction = phase.Direction; throughput *= phase.Weight;
                        }
                        else
                        {
                            float pdf;
                            direction = MediumSampleDirection(ray.Direction, medium.Anisotropy,
                                Sample2D(samples, domain ^ PhaseScramble), (TransportFlags & ImportanceBit) != 0, pdf);
                            throughput *= MediumPhase(dot(ray.Direction, direction), medium.Anisotropy) / pdf;
                        }
                        if ((TransportFlags & RouletteBit) != 0 && bounce + 1 >= RouletteStartDepth)
                        {
                            const float survival = RouletteProbability(throughput * etaScale);
                            if (Sample2D(samples, domain ^ RouletteScramble).x >= survival) break;
                            throughput /= survival;
                        }
                        ray.Origin = position; ray.Direction = direction;
                        ray.MinimumDistance = RayEpsilon; ray.MaximumDistance = MaximumRayDistance;
                        includeRegisteredEmission = false;
                        IncrementCounter(reflectionRays);
                        continue;
                    }
                }
                if (!found)
                {
                    radiance += throughput * SpectralEnvironment(ray.Direction, wavelength);
                    break;
                }
                const uint materialIndex = hit.MaterialIndex < MaterialCount ? hit.MaterialIndex : 0;
                const PackedMaterial material = SpectralMaterial(Materials[materialIndex], materialIndex, wavelength);
                const float3 surface = SpectralSurface(hit, material.DiffuseTextureIndex, wavelength);
                const float3 diffuse = surface * material.AlbedoReflectivity.xyz;
                const float3 position = ray.Origin + ray.Direction * hit.Distance;
                const float3 viewDirection = normalize(-ray.Direction);
                if (material.Model != MaterialLegacy)
                {
                    const SurfaceBsdf bsdf = MakeSurfaceBsdf(material, hit, viewDirection, surface);
                    const float3 emission = includeRegisteredEmission || material.SampledEmitter == 0u
                        ? material.Emission.xyz : 0.0f.xxx;
                    radiance += throughput * (emission + PhysicalDirectLighting(bsdf, position, samples,
                        bounce, shadowRays, primitiveTests, wavelength, sceneTime, interior.Optical.w));
                    if (bounce == MaxReflectionBounces) break;
                    const uint dimension = (bounce + 1u) * BounceSeedMultiplier;
                    const SurfaceSample continuation = SampleSurface(bsdf, Sample2D(samples, dimension),
                        Sample2D(samples, dimension ^ LobeSelectionScramble).x);
                    if (continuation.Event == 0u || continuation.Pdf <= 0) break;
                    throughput *= continuation.Weight;
                    if (all(throughput == 0)) break;
                    // Radiance retains eta scaling. Roulette sees a throughput
                    // with only that reversible boundary factor removed (PBRT v4).
                    etaScale *= continuation.RelativeIndex * continuation.RelativeIndex;
                    if ((TransportFlags & RouletteBit) != 0u && bounce + 1u >= RouletteStartDepth)
                    {
                        const float survival = RouletteProbability(throughput * etaScale);
                        if (Sample2D(samples, dimension ^ RouletteScramble).x >= survival) break;
                        throughput /= survival;
                    }
                    ray.Origin = SurfaceOffset(bsdf, position, continuation.Direction);
                    ray.Direction = continuation.Direction;
                    ray.MinimumDistance = RayEpsilon;
                    ray.MaximumDistance = MaximumRayDistance;
                    if (material.Optical.w > 0 && (continuation.Event & ScatteringTransmission) != 0)
                        interiorMaterial = hit.FrontFace != 0 ? materialIndex : InvalidPrimitiveIndex;
                    includeRegisteredEmission = false;
                    IncrementCounter(reflectionRays);
                    continue;
                }
                float3 direct = material.Emission.xyz;
                [loop]
                for (uint lightIndex = 0; lightIndex < LightCount; ++lightIndex)
                {
                    PackedLight light = Lights[lightIndex];
                    light.Color = SpectralLight(light.Color, lightIndex, wavelength);
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
                                    primitiveTests, sceneTime))
                            {
                                visibility = 1.0f - ShadowStrength;
                            }
                        }
                        AccumulateLightSample(
                            directionToLight,
                            light.Color * light.Intensity * MediumTransmission(medium, position, directionToLight, 0, MaximumRayDistance),
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
                            primitiveTests, wavelength, sceneTime);
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
                        [loop]
                        for (uint sampleIndex = 0u; (TransportFlags & NeeBit) != 0u && sampleIndex < AreaLightSamples; ++sampleIndex)
                        {

                            const uint lightDimension = ((bounce + 1u) * BounceSeedMultiplier)
                                ^ ((lightIndex + 1u) * LightSeedMultiplier) ^ ((sampleIndex + 1u) * LightSampleSeedMultiplier);
                            const float2 lightSample = Sample2D(samples, lightDimension);
                            const float offsetX = (float(sampleIndex & 1u) + lightSample.x) * 0.5f - 0.5f;
                            const float offsetY = (float(sampleIndex >> 1u) + lightSample.y) * 0.5f - 0.5f;

                            const float3 samplePosition =
                                light.Position
                                + tangent * (offsetX * light.Width)
                                + bitangent * (offsetY * light.Height);
                            AccumulateFiniteLight(
                                light,
                                samplePosition,
                                AreaSampleWeight(light, position, hit.Normal, samplePosition),
                                true,
                                position,
                                hit.Normal,
                                viewDirection,
                                diffuse,
                                material.Specular.xyz,
                                material.AmbientShininess.w,
                                direct,
                                shadowRays,
                                primitiveTests, wavelength, sceneTime);
                        }
                        if ((TransportFlags & NeeBit) == 0u || ((TransportFlags >> MisShift) & MisMask) != NoMis)
                        {
                            const uint bsdfDimension = ((bounce+1u)*BounceSeedMultiplier)
                                ^ ((lightIndex+1u)*LightSeedMultiplier) ^ DirectBsdfScramble;
                            float3 samplePosition;
                            float weight;
                            if (BsdfEmitterSample(light, position, hit.Normal, Sample2D(samples, bsdfDimension), tangent, bitangent, samplePosition, weight))
                                AccumulateFiniteLight(light, samplePosition, weight, true, position, hit.Normal,
                                    viewDirection, diffuse, material.Specular.xyz, material.AmbientShininess.w,
                                    direct, shadowRays, primitiveTests, wavelength, sceneTime);
                        }
                    }
                }
                const float reflectivity = saturate(material.AlbedoReflectivity.w);

                radiance += throughput * ((includeRegisteredEmission || material.SampledEmitter == 0u
                    ? material.Emission.xyz : float3(0.0f, 0.0f, 0.0f))
                    + (direct - material.Emission.xyz) * (1.0f - reflectivity));
                if (bounce == MaxReflectionBounces) break;
                const uint dimension = ((bounce + 1u) * BounceSeedMultiplier);
                float3 outgoing;
                const bool mirrorSelected = Sample2D(samples, dimension ^ LobeSelectionScramble).x < reflectivity;
                includeRegisteredEmission = mirrorSelected;
                if (mirrorSelected)
                {
                    outgoing = normalize(reflect(ray.Direction, hit.Normal));
                }
                else
                {
                    outgoing = SampleDiffuseDirection(hit.Normal, Sample2D(samples, dimension));
                    throughput *= clamp(diffuse, 0.0f, 1.0f);
                    if ((TransportFlags & ImportanceBit) == 0u)
                        throughput *= 2.0f * max(0.0f,dot(hit.Normal,outgoing));
                }
                if ((TransportFlags & RouletteBit) != 0u && bounce+1u >= RouletteStartDepth)
                {
                    const float survival = RouletteProbability(throughput * etaScale);
                    if (Sample2D(samples, dimension ^ RouletteScramble).x >= survival) break;
                    throughput /= survival;
                }
                ray.Origin = position + hit.Normal * RayEpsilon;
                ray.Direction = outgoing;
                IncrementCounter(reflectionRays);
                continue;

            }
            const float3 filmSample = SpectralFilm(wavelengths, radiance);
            AccumulateFilmSample(moments, completedSamples, filmSample);
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

#undef ARVO_PARTICIPATING_MEDIA
