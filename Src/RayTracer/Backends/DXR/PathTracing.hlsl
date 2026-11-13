// Kajiya (1986), "The Rendering Equation". Independent DXR path transport.
// RayGen owns the walk; closest-hit only returns surface attributes. Physical
// transport composes DXR-owned BSDFs, a separate direct estimator and eta-aware
// roulette. Legacy Lambert/mirror materials retain their original convention.
static const float Pi = 3.14159265358979323846f;
static const float MaximumRayDistance = 1.0e30;
static const float BasisParallelThreshold = 0.999;
static const uint AreaLightSamples = 4;
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
static const float RandomMantissaRange = 16777216.0;
static const float MinimumDirectionLengthSquared = 1.0e-12;

static const uint DirectionalLightType = 0u;
static const uint PointLightType = 1u;
static const uint RectAreaLightType = 2u;
static const float RayEpsilon = 1.0e-4;

#define ARVO_SPECTRAL_TRANSPORT
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

// Keep pure sampling, spectra and BSDFs in separately compiled callable stages.
// RayGen still owns visibility and the complete offline walk. Interactive preview
// selects the cheaper Whitted library; no offline transport mode is specialized out.
// Callable indices follow the host PathCallableNames after software traversal.
// The only nested call is PathSpectrum -> PathSampling (depth two).
// DXR Functional Spec: https://microsoft.github.io/DirectX-Specs/d3d/Raytracing.html#callable-shaders
#define Sample2D Sample2DInline
#include "Sequence.hlsl"
#undef Sample2D
struct PathSamplingRequest
{
    SampleContext Context;
    uint Dimension;
    float2 Value;
};
static const uint PathSamplingCallableIndex = 2;
[shader("callable")]
void PathSampling(inout PathSamplingRequest request)
{
    request.Value = Sample2DInline(request.Context, request.Dimension);
}
float2 Sample2D(SampleContext context, uint dimension)
{
    PathSamplingRequest request = (PathSamplingRequest)0;
    request.Context = context;
    request.Dimension = dimension;
    CallShader(PathSamplingCallableIndex, request);
    return request.Value;
}
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

// Veach (1997), "Robust Monte Carlo Methods for Light Transport Simulation".
// The separate direct estimator combines four light samples and one BSDF sample.
#include "WavelengthSampling.hlsl"
#include "SpectralColor.hlsl"
#include "RgbSpectrum.hlsl"
#include "SpectralFiltering.hlsl"
#define SamplePathWavelength SamplePathWavelengthInline
#define EvaluateSpectralMaterial EvaluateSpectralMaterialInline
#define EvaluateSpectralSurface EvaluateSpectralSurfaceInline
#define EvaluateSpectralLight EvaluateSpectralLightInline
#define EvaluateSpectralBackground EvaluateSpectralBackgroundInline
#define EvaluateSpectralMedium EvaluateSpectralMediumInline
#define SpectralFilmSample SpectralFilmSampleInline
#include "SpectralTransport.hlsl"
#undef SamplePathWavelength
#undef EvaluateSpectralMaterial
#undef EvaluateSpectralSurface
#undef EvaluateSpectralLight
#undef EvaluateSpectralBackground
#undef EvaluateSpectralMedium
#undef SpectralFilmSample
struct PathSpectrumRequest
{
    MaterialData Material;
    Payload Surface;
    MediumParameters Medium;
    WavelengthPacket Wavelengths;
    SampleContext Context;
    float3 Value, Direction;
    float Wavelength;
    uint Index, Operation;
};
static const uint PathSpectrumCallableIndex = 3;
static const uint PathSpectrumSample = 0, PathSpectrumMaterial = 1, PathSpectrumSurface = 2;
static const uint PathSpectrumLight = 3, PathSpectrumBackground = 4, PathSpectrumMedium = 5, PathSpectrumFilm = 6;
[shader("callable")]
void PathSpectrum(inout PathSpectrumRequest request)
{
    if (request.Operation == PathSpectrumSample)
        request.Wavelengths = SamplePathWavelengthInline(request.Context);
    else if (request.Operation == PathSpectrumMaterial)
        request.Material = EvaluateSpectralMaterialInline(request.Material, request.Index, request.Wavelength);
    else if (request.Operation == PathSpectrumSurface)
        request.Value = EvaluateSpectralSurfaceInline(request.Surface, request.Index, request.Wavelength);
    else if (request.Operation == PathSpectrumLight)
        request.Value = EvaluateSpectralLightInline(request.Value, request.Index, request.Wavelength);
    else if (request.Operation == PathSpectrumBackground)
        request.Value = EvaluateSpectralBackgroundInline(request.Direction, request.Wavelength);
    else if (request.Operation == PathSpectrumMedium)
        request.Medium = EvaluateSpectralMediumInline(request.Medium, request.Wavelength);
    else if (request.Operation == PathSpectrumFilm)
        request.Value = SpectralFilmSampleInline(request.Value, request.Wavelengths);
}
WavelengthPacket SamplePathWavelength(SampleContext context)
{
    PathSpectrumRequest request = (PathSpectrumRequest)0;
    request.Context = context;
    request.Operation = PathSpectrumSample;
    CallShader(PathSpectrumCallableIndex, request);
    return request.Wavelengths;
}
MaterialData EvaluateSpectralMaterial(MaterialData material, uint index, float wavelength)
{
    PathSpectrumRequest request = (PathSpectrumRequest)0;
    request.Material = material;
    request.Index = index;
    request.Wavelength = wavelength;
    request.Operation = PathSpectrumMaterial;
    CallShader(PathSpectrumCallableIndex, request);
    return request.Material;
}
float3 EvaluateSpectralSurface(Payload surface, uint texture, float wavelength)
{
    PathSpectrumRequest request = (PathSpectrumRequest)0;
    request.Surface = surface;
    request.Index = texture;
    request.Wavelength = wavelength;
    request.Operation = PathSpectrumSurface;
    CallShader(PathSpectrumCallableIndex, request);
    return request.Value;
}
float3 EvaluateSpectralLight(float3 color, uint index, float wavelength)
{
    PathSpectrumRequest request = (PathSpectrumRequest)0;
    request.Value = color;
    request.Index = index;
    request.Wavelength = wavelength;
    request.Operation = PathSpectrumLight;
    CallShader(PathSpectrumCallableIndex, request);
    return request.Value;
}
float3 EvaluateSpectralBackground(float3 direction, float wavelength)
{
    PathSpectrumRequest request = (PathSpectrumRequest)0;
    request.Direction = direction;
    request.Wavelength = wavelength;
    request.Operation = PathSpectrumBackground;
    CallShader(PathSpectrumCallableIndex, request);
    return request.Value;
}
MediumParameters EvaluateSpectralMedium(MediumParameters medium, float wavelength)
{
    PathSpectrumRequest request = (PathSpectrumRequest)0;
    request.Medium = medium;
    request.Wavelength = wavelength;
    request.Operation = PathSpectrumMedium;
    CallShader(PathSpectrumCallableIndex, request);
    return request.Medium;
}
float3 SpectralFilmSample(float3 radiance, WavelengthPacket wavelengths)
{
    PathSpectrumRequest request = (PathSpectrumRequest)0;
    request.Value = radiance;
    request.Wavelengths = wavelengths;
    request.Operation = PathSpectrumFilm;
    CallShader(PathSpectrumCallableIndex, request);
    return request.Value;
}

#include "MultipleImportance.hlsl"
#include "Importance.hlsl"
#include "RussianRoulette.hlsl"
#include "BsdfTypes.hlsl"
#include "Fresnel.hlsl"
#include "Ggx.hlsl"
#include "Diffuse.hlsl"
#include "Conductor.hlsl"
#include "Dielectric.hlsl"
#define EvaluateSurface EvaluateSurfaceInline
#define SampleSurface SampleSurfaceInline
#include "SurfaceScattering.hlsl"
#undef EvaluateSurface
#undef SampleSurface

// BSDF evaluation/sampling remains DXR-owned. A callable boundary prevents
// each direct-light and continuation site from cloning the material program.
struct PathSurfaceRequest
{
    SurfaceBsdf Surface;
    float3 Direction;
    uint Operation;
    float2 Variates;
    float Lobe;
    SurfaceEvaluation Evaluation;
    SurfaceSample Sample;
};
static const uint PathSurfaceEvaluate = 0, PathSurfaceSample = 1;
static const uint PathSurfaceCallableIndex = 1;
[shader("callable")]
void PathSurface(inout PathSurfaceRequest request)
{
    if (request.Operation == PathSurfaceEvaluate)
        request.Evaluation = EvaluateSurfaceInline(request.Surface, request.Direction);
    else
        request.Sample = SampleSurfaceInline(request.Surface, request.Variates, request.Lobe);
}
SurfaceEvaluation EvaluateSurface(SurfaceBsdf surface, float3 direction)
{
    PathSurfaceRequest request = (PathSurfaceRequest)0;
    request.Surface = surface;
    request.Direction = direction;
    request.Operation = PathSurfaceEvaluate;
    CallShader(PathSurfaceCallableIndex, request);
    return request.Evaluation;
}
SurfaceSample SampleSurface(SurfaceBsdf surface, float2 variates, float lobe)
{
    PathSurfaceRequest request = (PathSurfaceRequest)0;
    request.Surface = surface;
    request.Variates = variates;
    request.Lobe = lobe;
    request.Operation = PathSurfaceSample;
    CallShader(PathSurfaceCallableIndex, request);
    return request.Sample;
}
#include "DirectLighting.hlsl"
#include "VolumeLighting.hlsl"
float AreaSampleWeight(LightData light, float3 position, float3 normal, float3 samplePosition)
{
    if (((TransportFlags >> MisShift) & MisMask) == NoMis) return 1.0f / AreaLightSamples;
    const float3 offset = samplePosition - position;
    const float3 direction = normalize(offset);
    const float emitterCosine = dot(light.Direction, -direction);
    if (emitterCosine <= 0.0f) return 0.0f;
    const float lightPdf = dot(offset,offset) / (light.Width*light.Height*emitterCosine);
    return MisWeight(AreaLightSamples*lightPdf, DiffusePdf(dot(normal,direction))) / AreaLightSamples;
}

bool BsdfEmitterSample(LightData light, float3 position, float3 normal, float2 variates,
    float3 tangent, float3 bitangent, out float3 samplePosition, out float weight)
{
    samplePosition = position;
    weight = 0.0f;
    const float3 direction = SampleDiffuseDirection(normal, variates);
    const float emitterCosine = dot(light.Direction, -direction);
    if (emitterCosine <= 0.0f) return false;
    const float distance = dot(position-light.Position, light.Direction) / emitterCosine;
    samplePosition = position + direction*distance;
    const float3 offset = samplePosition - light.Position;
    const float bsdfPdf = DiffusePdf(dot(normal,direction));
    if (distance <= RayEpsilon || bsdfPdf <= 0.0f || abs(dot(offset,tangent)) > light.Width*0.5f
        || abs(dot(offset,bitangent)) > light.Height*0.5f) return false;
    const float lightPdf = distance*distance / (light.Width*light.Height*emitterCosine);
    weight = lightPdf/bsdfPdf * MisWeight(bsdfPdf, AreaLightSamples*lightPdf);
    return true;
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
    inout uint2 shadowRays, float wavelength)
{
    const MediumParameters medium = EvaluateSpectralMedium(ParticipatingMedium, wavelength);
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
    incidentRadiance *= MediumTransmission(medium, position, directionToLight, 0, distance);
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
    const float inverseDistance = 1 / material.Optical.w;
    const float3 albedo = material.AlbedoReflectivity.xyz * material.ConductorExtinction.w;
    medium.Scattering = albedo * inverseDistance;
    medium.Absorption = (1 - albedo) * inverseDistance;
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
            float timeVariate =
                CameraExposure.Open == CameraExposure.Close ? 0 : Sample2D(samples, TimeSampleDimension).x;
            MotionCameraRay moved =
                SampleCameraMotion(origin, direction, CameraPosition, CameraExposure, timeVariate);
            origin = moved.Origin;
            direction = moved.Direction;
            sceneTime = moved.SceneTime;
        }
        BeginPathTime(sceneTime);
        const WavelengthPacket wavelengths = SamplePathWavelength(samples);
        const float wavelength = wavelengths.Lambda[0];
        const MediumParameters exteriorMedium = EvaluateSpectralMedium(ParticipatingMedium, wavelength);
        const uint NoSubsurfaceInterior = ~0u;
        uint interiorMaterial = NoSubsurfaceInterior;
        float3 throughput = 1.0;
        float etaScale = 1;
        bool includeRegisteredEmission = true;
        float3 radiance = 0.0;

        for (uint bounce = 0; bounce <= MaxReflectionBounces; ++bounce)
        {
            Payload payload = MissPayload(MaximumRayDistance);
            if (PrimitiveCount != 0)
                payload = CastRay(origin, direction, RayEpsilon, MaximumRayDistance);
            if (bounce == 0 && (payload.Flags & SurfaceHitFlag) != 0 &&
                (payload.Flags & SurfaceFrontFaceFlag) == 0 && Materials[payload.MaterialIndex].Optical.w > 0)
                interiorMaterial = payload.MaterialIndex;
            MediumParameters medium = exteriorMedium;
            MaterialData interior = (MaterialData)0;
            if (interiorMaterial != NoSubsurfaceInterior)
            {
                interior = EvaluateSpectralMaterial(Materials[interiorMaterial], interiorMaterial, wavelength);
                medium = SubsurfaceInterior(interior);
            }
            if (medium.Enabled != 0)
            {
                const uint FreeFlightScramble = 0x7a4d2e91u, PhaseScramble = 0x3c6ef372u;
                const uint domain = (bounce + 1) * BounceSeedMultiplier;
                const float endpoint = (payload.Flags & SurfaceHitFlag) != 0 ? asfloat(payload.DistanceBits) : MaximumRayDistance;
                MediumFlight flight;
                const float2 freeFlight = Sample2D(samples, domain ^ FreeFlightScramble);
                if (interior.Optical.w > 0)
                {
                    const SubsurfaceFlight event = SubsurfaceFreeFlight(interior.Optical.w,
                        interior.AlbedoReflectivity.xyz * interior.ConductorExtinction.w, direction, endpoint, freeFlight.x);
                    flight.Weight = event.Weight; flight.Distance = event.Distance; flight.Scattered = event.Scattered;
                }
                else flight = MediumFreeFlight(medium, origin, direction, 0, endpoint, freeFlight);
                throughput *= flight.Weight;
                if (all(throughput == 0)) break;
                if (flight.Scattered != 0)
                {
                    const float3 position = origin + direction * flight.Distance;
                    radiance += throughput * VolumeDirectLighting(medium, position, direction, samples, bounce, shadowRays, wavelength);
                    if (bounce == MaxReflectionBounces) break;
                    float3 outgoing;
                    if (interior.Optical.w > 0)
                    {
                        const SubsurfaceDirection phase = SubsurfaceSampleDirection(direction, medium.Anisotropy,
                            Sample2D(samples, domain ^ PhaseScramble), (TransportFlags & ImportanceBit) != 0);
                        outgoing = phase.Direction; throughput *= phase.Weight;
                    }
                    else
                    {
                        float pdf;
                        outgoing = MediumSampleDirection(direction, medium.Anisotropy,
                            Sample2D(samples, domain ^ PhaseScramble), (TransportFlags & ImportanceBit) != 0, pdf);
                        throughput *= MediumPhase(dot(direction, outgoing), medium.Anisotropy) / pdf;
                    }
                    if ((TransportFlags & RouletteBit) != 0 && bounce + 1 >= RouletteStartDepth)
                    {
                        const float survival = RouletteProbability(throughput * etaScale);
                        if (Sample2D(samples, domain ^ RouletteScramble).x >= survival) break;
                        throughput /= survival;
                    }
                    origin = position; direction = outgoing;
                    includeRegisteredEmission = false;
                    IncrementLocalCounter(reflectionRays);
                    continue;
                }
            }
            if ((payload.Flags & SurfaceHitFlag) == 0)
            {
                radiance += throughput * EvaluateSpectralBackground(direction, wavelength);
                break;
            }

            const uint materialIndex = min(payload.MaterialIndex, MaterialCount - 1);
            MaterialData material = EvaluateSpectralMaterial(Materials[materialIndex], materialIndex, wavelength);
            float3 surface = EvaluateSpectralSurface(payload, material.DiffuseTextureIndex, wavelength);
            float3 position = origin + direction * asfloat(payload.DistanceBits);
            float3 viewDirection = NormalizeSafe(-direction, payload.Normal);
            if (material.Model != MaterialLegacy)
            {
                const SurfaceBsdf scattering = MakeSurfaceBsdf(material, payload, viewDirection, surface);
                const float3 emission = includeRegisteredEmission || material.SampledEmitter == 0u
                    ? material.Emission.xyz : 0.0.xxx;
                radiance += throughput * (emission + PhysicalDirectLighting(scattering, position,
                    samples, bounce, shadowRays, wavelength, interior.Optical.w));
                if (bounce == MaxReflectionBounces) break;
                const uint dimension = (bounce + 1u) * BounceSeedMultiplier;
                const SurfaceSample continuation = SampleSurface(scattering, Sample2D(samples, dimension),
                    Sample2D(samples, dimension ^ LobeSelectionScramble).x);
                if (continuation.Event == 0u || continuation.Pdf <= 0) break;
                throughput *= continuation.Weight;
                if (all(throughput == 0)) break;
                // Remove only the refractive radiance scale from the termination
                // probability. The transported radiance itself retains eta^-2.
                etaScale *= continuation.RelativeIndex * continuation.RelativeIndex;
                if ((TransportFlags & RouletteBit) != 0u && bounce + 1u >= RouletteStartDepth)
                {
                    const float survival = RouletteProbability(throughput * etaScale);
                    if (Sample2D(samples, dimension ^ RouletteScramble).x >= survival) break;
                    throughput /= survival;
                }
                if (material.Optical.w > 0 && (continuation.Event & ScatteringTransmission) != 0)
                    interiorMaterial = (payload.Flags & SurfaceFrontFaceFlag) != 0 ? materialIndex : NoSubsurfaceInterior;
                origin = SurfaceOffset(scattering, position, continuation.Direction);
                direction = continuation.Direction;
                includeRegisteredEmission = false;
                IncrementLocalCounter(reflectionRays);
                continue;
            }
            float3 diffuseLight = 0.0;
            float3 specularLight = 0.0;
            for (uint lightIndex = 0; lightIndex < LightCount; ++lightIndex)
            {
                LightData light = Lights[lightIndex];
                light.Color = EvaluateSpectralLight(light.Color, lightIndex, wavelength);
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
                        max(light.Color, 0.0) * max(light.Intensity, 0.0) *
                            MediumTransmission(medium, position, directionToLight, 0, MaximumRayDistance),
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
                        specularLight, shadowRays, wavelength);
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
                    [loop]
                    for (uint sampleIndex = 0u; (TransportFlags & NeeBit) != 0u && sampleIndex < AreaLightSamples; ++sampleIndex)
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
                            AreaSampleWeight(light, position, payload.Normal, samplePosition),
                            true,
                            position,
                            payload.Normal,
                            viewDirection,
                            material.AmbientShininess.w,
                            diffuseLight,
                            specularLight, shadowRays, wavelength);
                    }
                    if ((TransportFlags & NeeBit) == 0u || ((TransportFlags >> MisShift) & MisMask) != NoMis)
                    {
                        const uint bsdfDimension = ((bounce+1u)*BounceSeedMultiplier)
                            ^ ((lightIndex+1u)*LightSeedMultiplier) ^ DirectBsdfScramble;
                        float3 samplePosition;
                        float weight;
                        if (BsdfEmitterSample(light, position, payload.Normal, Sample2D(samples, bsdfDimension), tangent, bitangent, samplePosition, weight))
                            AccumulateFiniteLight(light, samplePosition, weight, true, position, payload.Normal,
                                viewDirection, material.AmbientShininess.w, diffuseLight, specularLight, shadowRays, wavelength);
                    }
                }
            }

            float3 direct = max(material.Emission.xyz, 0.0)
                + surface * max(material.AlbedoReflectivity.xyz, 0.0) * diffuseLight / Pi;
            float reflectivity = saturate(material.AlbedoReflectivity.w);

            radiance += throughput * ((includeRegisteredEmission || material.SampledEmitter == 0u
                ? material.Emission.xyz : float3(0.0f, 0.0f, 0.0f))
                + (direct - material.Emission.xyz) * (1.0f - reflectivity));
            if (bounce == MaxReflectionBounces) break;
            const uint dimension = ((bounce + 1u) * BounceSeedMultiplier);
            float3 outgoing;
            includeRegisteredEmission = Sample2D(samples, dimension ^ LobeSelectionScramble).x < reflectivity;
            if (includeRegisteredEmission)
            {
                outgoing = normalize(reflect(direction, payload.Normal));
            }
            else
            {
                outgoing = SampleDiffuseDirection(payload.Normal, Sample2D(samples, dimension));
                throughput *= clamp(surface * material.AlbedoReflectivity.xyz, 0.0f, 1.0f);
                if ((TransportFlags & ImportanceBit) == 0u)
                    throughput *= 2.0f * max(0.0f,dot(payload.Normal,outgoing));
            }
            if ((TransportFlags & RouletteBit) != 0u && bounce+1u >= RouletteStartDepth)
            {
                const float survival = RouletteProbability(throughput * etaScale);
                if (Sample2D(samples, dimension ^ RouletteScramble).x >= survival) break;
                throughput /= survival;
            }
            origin = position + payload.Normal * RayEpsilon;
            direction = outgoing;
            IncrementLocalCounter(reflectionRays);
            continue;

        }
        const float3 filmSample = SpectralFilmSample(radiance, wavelengths);
        AccumulateFilmSample(film, filmSample);
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

#undef ARVO_PARTICIPATING_MEDIA
