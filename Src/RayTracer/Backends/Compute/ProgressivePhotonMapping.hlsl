// Independent Compute stochastic progressive photon mapping.
// Hachisuka and Jensen (2009), "Stochastic Progressive Photon Mapping".
// https://pbr-book.org/3ed-2018/Light_Transport_III_Bidirectional_Methods/Stochastic_Progressive_Photon_Mapping
// Resample one camera visible point per pixel/pass; persist N, R^2 and tau.
// Stream bounded light batches through a hash index and accumulate Phi/M at
// each visible point. Update the uniform-disk estimate only after the WHOLE
// pass. A batch boundary is not a progressive iteration. Memory is O(pixels +
// batch paths * depth), independent of the complete emitted population.
// Pixel-owned binary64 reductions need no double atomics. UAV barriers publish
// emitted records and index links before readers. Hash collision order may
// change rounding; exact cell checks prevent duplicate contributions.
// NEE replaces depth-zero light arrivals. Caustic/indirect classes are disjoint.
// Samples denote complete passes; final density is not a mean of pass estimates.
// Interior collision photons use a separate three-dimensional bandwidth.

static const uint PpmMaximumReflections = 32, PpmThreads = 64, PpmBatchPaths = 1024;
static const uint PpmInvalid = 0xffffffffu, PpmDirect = 1, PpmCaustic = 2, PpmIndirect = 3;
static const uint PpmVolumeFlag = 0x80000000u, PpmClassMask = ~PpmVolumeFlag;
static const uint PpmVisibleSurface = 1, PpmVisibleVolume = 2;
static const uint PpmCausticsBit = 1, PpmIndirectBit = 2;
static const double PpmPi = 3.1415926535897932384626433832795L;
static const double PpmVolumeMeasure = 4.0L / 3.0L;
static const double PpmDoubleLimit = 1.7976931348623157e308L, PpmCoordinateLimit = 1073741824.0L;
static const double PpmNormalAgreement = 0.9L, PpmPlaneFraction = 0.1L;
static const double PpmRootScale = 1.0e20L, PpmRootScaleSqrt = 1.0e10L;
bool PpmFinite(double value)
{
    return value >= -PpmDoubleLimit && value <= PpmDoubleLimit;
}
double PpmSquareRoot(double value)
{
    if (!(value > 0)) return 0;
    double scaled = value, multiplier = 1;
    while (scaled > PpmRootScale)
    {
        scaled /= PpmRootScale;
        multiplier *= PpmRootScaleSqrt;
    }
    while (scaled < 1 / PpmRootScale)
    {
        scaled *= PpmRootScale;
        multiplier /= PpmRootScaleSqrt;
    }
    double root = double(sqrt(float(scaled)));
    root = (root + scaled / root) / 2;
    root = (root + scaled / root) / 2;
    return root * multiplier;
}
double PpmCubeRoot(double value)
{
    if (!(value > 0)) return 0;
    double root = double(pow(float(value), 1.0f / 3.0f));
    if (!(root > 0))
    {
        root = 1;
        while (root * root * root > value) root /= 2;
    }
    root = (2 * root + value / (root * root)) / 3;
    root = (2 * root + value / (root * root)) / 3;
    return root;
}
bool PpmCellValid(double coordinate)
{
    return coordinate > -PpmCoordinateLimit && coordinate < PpmCoordinateLimit;
}
int PpmCellCoordinate(double value)
{
    int integer = int(value);
    return integer - (value < double(integer) ? 1 : 0);
}
uint PpmHash(uint value)
{
    static const uint FirstMultiplier = 0x7feb352du, SecondMultiplier = 0x846ca68bu;
    static const uint OuterShift = 16, MiddleShift = 15;
    value = (value ^ (value >> OuterShift)) * FirstMultiplier;
    value = (value ^ (value >> MiddleShift)) * SecondMultiplier;
    return value ^ (value >> OuterShift);
}
uint PpmBucket(int x, int y, int z, uint buckets)
{
    static const uint MixX = 0x9e3779b9u, MixY = 0x85ebca6bu, MixZ = 0xc2b2ae35u;
    return PpmHash(uint(x) * MixX ^ uint(y) * MixY ^ uint(z) * MixZ) & (buckets - 1);
}
uint PpmClass(uint depth, bool deltaPrefix)
{
    return depth == 0 ? PpmDirect : deltaPrefix ? PpmCaustic : PpmIndirect;
}
bool PpmIncluded(uint kind, bool nee, uint classes)
{
    uint category = kind & PpmClassMask;
    return (category == PpmDirect && ((kind & PpmVolumeFlag) || !nee)) ||
           (category == PpmCaustic && (classes & PpmCausticsBit)) ||
           (category == PpmIndirect && (classes & PpmIndirectBit));
}
bool PpmMergeable(double distanceSquared, double radiusSquared, double normalAgreement, double planeDistance,
                  uint lightDepth, uint cameraDepth, uint maximumDepth)
{
    return cameraDepth <= maximumDepth && lightDepth <= maximumDepth - cameraDepth && radiusSquared > 0 &&
           distanceSquared >= 0 && distanceSquared <= radiusSquared &&
           normalAgreement >= PpmNormalAgreement &&
           planeDistance * planeDistance <= PpmPlaneFraction * PpmPlaneFraction * radiusSquared;
}
struct PpmProgress
{
    double RadiusSquared, EffectiveCount;
    double Red, Green, Blue;
};
PpmProgress PpmAdvance(PpmProgress state, uint arrivals, double red, double green, double blue, double alpha)
{
    if (arrivals == 0)
        return state;
    double count = state.EffectiveCount + alpha * double(arrivals);
    double areaRatio = count / (state.EffectiveCount + double(arrivals));
    state.RadiusSquared *= areaRatio;
    state.EffectiveCount = count;
    state.Red = (state.Red + red) * areaRatio;
    state.Green = (state.Green + green) * areaRatio;
    state.Blue = (state.Blue + blue) * areaRatio;
    return state;
}
PpmProgress PpmAdvanceVolume(PpmProgress state, uint arrivals, double red, double green,
                             double blue, double alpha)
{
    if (arrivals == 0) return state;
    double count = state.EffectiveCount + alpha * double(arrivals);
    double volumeRatio = count / (state.EffectiveCount + double(arrivals));
    double radiusRatio = PpmCubeRoot(volumeRatio);
    state.RadiusSquared *= radiusRatio * radiusRatio;
    state.EffectiveCount = count;
    state.Red = (state.Red + red) * volumeRatio;
    state.Green = (state.Green + green) * volumeRatio;
    state.Blue = (state.Blue + blue) * volumeRatio;
    return state;
}
double PpmResolve(double direct, double flux, double radiusSquared, uint completedPasses, uint pathsPerPass)
{
    if (completedPasses == 0)
        return 0;
    double value = direct / double(completedPasses);
    if (radiusSquared > 0 && pathsPerPass > 0)
        value += flux / (PpmPi * radiusSquared * double(completedPasses) * double(pathsPerPass));
    return value;
}
double PpmResolveVolume(double flux, double radiusSquared, uint completedPasses, uint pathsPerPass)
{
    if (!(radiusSquared > 0) || completedPasses == 0 || pathsPerPass == 0) return 0;
    return flux / (PpmVolumeMeasure * PpmPi * radiusSquared * PpmSquareRoot(radiusSquared) *
                   double(completedPasses) * double(pathsPerPass));
}
float PpmRasterCoordinate(uint pixel, uint extent, float jitter)
{
    // The last pixel can round up to 1 even though jitter < 1. Keep generated
    // camera endpoints inside their half-open raster support after conversion.
    static const float FloatMantissaRange = 16777216.0f;
    static const float LargestBelowOne = 1 - 1 / FloatMantissaRange;
    float coordinate = float((double(pixel) + double(jitter)) / double(extent));
    return coordinate > LargestBelowOne ? LargestBelowOne : coordinate;
}

static const float Pi = 3.14159265358979323846f, RayEpsilon = 1e-4f;
static const float MaximumRayDistance = 1e30f, BasisParallelThreshold = 0.999f;
static const uint ImportanceBit = 1, NeeBit = 2, RouletteBit = 4, MisShift = 3, MisMask = 3;
static const uint PixelSeedMultiplier = 9781, SampleSeedMultiplier = 6271,
                  SecondDimensionScramble = 0x68bc21ebu;
static const uint PpmCameraDomain = 0x43415050u, PpmLightDomain = 0x4c495050u, PpmPassDomain = 0x50415050u;
static const uint PpmSourceDomain = 0x53524345u, PpmPositionDomain = 0x504f5349u,
                  PpmDirectionDomain = 0x44495245u;
static const uint PpmScatterDomain = 0x53434154u, PpmEventDomain = 0x45564e54u,
                  PpmRouletteDomain = 0x52524f55u;
static const uint PpmLensDomain = 0x4c454e53u, PpmDepthDomain = 0x9e3779b9u, PpmRasterDimension = 0;
static const uint PpmVisibleDomain = 0x56495349u, PpmInteriorDomain = 0x494e5445u;
static const uint PpmNumericalFailure = 1, PpmConfigurationFailure = 2, PpmIndexFailure = 4;
#include "Types.hlsl"
#include "MaterialTypes.hlsl"
#include "EmitterData.hlsl"
struct PpmVisiblePoint
{
    float3 Position;
    uint Active;
    float3 Normal;
    uint FrontFace;
    float3 Geometry;
    uint Depth;
    float3 View;
    uint Material;
    float3 Color;
    uint Reserved;
    float3 Throughput;
    uint Padding;
};
struct PpmPixel
{
    double RadiusSquared, EffectiveCount;
    double3 Flux, Direct, Incoming;
    uint Arrivals, Reserved;
    double VolumeRadiusSquared, VolumeEffectiveCount;
    double3 VolumeFlux, VolumeIncoming;
    uint VolumeArrivals;
    uint3 VolumeReserved;
};
struct PpmPhoton
{
    float3 Position;
    uint Next;
    float3 Incoming;
    uint Depth;
    float3 Flux;
    uint Kind;
    float3 Normal;
    uint Reserved;
};
struct PpmTexture
{
    uint OffsetTexels, Width, Height, Reserved;
};
struct PpmStatistics
{
    uint2 Primary, Shadow, Reflection, Tests;
};
ByteAddressBuffer Nodes : register(t0);
StructuredBuffer<uint> PrimitiveRefs : register(t1);
StructuredBuffer<PackedTriangle> Triangles : register(t2);
StructuredBuffer<PackedSphere> Spheres : register(t3);
StructuredBuffer<PackedMaterial> Materials : register(t4);
StructuredBuffer<PackedEmitter> Emitters : register(t5);
StructuredBuffer<PpmTexture> Textures : register(t6);
StructuredBuffer<uint> TextureTexels : register(t7);
RWStructuredBuffer<float4> Accumulation : register(u0);
RWStructuredBuffer<uint> SampleCounts : register(u1);
RWStructuredBuffer<uint> Heads : register(u2);
RWStructuredBuffer<PpmStatistics> Statistics : register(u3);
RWStructuredBuffer<uint> SampleTargets : register(u4);
RWStructuredBuffer<PpmPhoton> Photons : register(u5);
// Instancing.hlsl declares t9/t10 and u6.
RWByteAddressBuffer PpmError : register(u7);
RWStructuredBuffer<PpmVisiblePoint> VisiblePoints : register(u8);
RWStructuredBuffer<PpmPixel> Pixels : register(u9);
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
    float HalfWidth, HalfHeight;
    uint MaterialCount, LightCount;
    float3 BackgroundTop;
    float AmbientIntensity;
    float3 BackgroundBottom;
    uint NodeCount;
    uint PrimitiveRefCount, TriangleCount, SphereCount;
    float ShadowStrength;
    uint PixelBegin, PixelCount, SampleBegin, SampleCount;
    uint GammaCorrection, GroupCount, TextureCount, TextureTexelCount;
    uint FullWidth, FullHeight, TileX, TileY;
    uint Integrator, TransportFlags, RouletteStartDepth, SamplingSequence;
    uint AdaptiveEnabled, AdaptivePilotSamples, AdaptiveMinimumSamples, AdaptivePhase;
    float AdaptiveRelativeError, AdaptiveAbsoluteError;
    uint Acceleration, InstanceNodeCount;
    float LensRadius, FocusDistance;
    uint SamplingSeed;
};
cbuffer PpmControls : register(b1)
{
    float3 EmissionCenter;
    float EmissionRadius;
    double InitialRadius, RadiusAlpha;
    uint EmitterCount, PathsPerPass, LightSeed, PathClasses;
    uint SlotsPerPath, BucketCount, WorkBegin, WorkCount;
    uint PassIndex, LightBegin, LightCountInBatch, ReservedControl;
};
void IncrementCounter(inout uint2 value)
{
    if (++value.x == 0)
        ++value.y;
}
void PpmFail(uint flag)
{
    uint ignored;
    PpmError.InterlockedOr(0, flag, ignored);
}
uint Hash(uint value)
{
    return PpmHash(value);
}
float Random01(uint seed)
{
    static const uint MantissaMask = 0xffffffu;
    static const float MantissaRange = 16777216.0f;
    return float(Hash(seed) & MantissaMask) / MantissaRange;
}
float2 PpmLightSample(uint path, uint domain)
{
    uint key = Hash(LightSeed ^ PpmLightDomain) ^ Hash(PassIndex ^ PpmPassDomain) ^ Hash(path) ^ domain;
    return float2(Random01(key), Random01(key ^ SecondDimensionScramble));
}
#include "Sequence.hlsl"
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
#include "LightSelection.hlsl"
#include "LightEmission.hlsl"
#include "SurfaceEmission.hlsl"
#include "EmitterEndpoint.hlsl"
#include "CameraEndpoint.hlsl"
#include "SubsurfaceScattering.hlsl"
#include "RussianRoulette.hlsl"
bool PpmColor(float3 value)
{
    return all(isfinite(value)) && all(value >= 0);
}
bool PpmWideColor(double3 value)
{
    return PpmFinite(value.x) && PpmFinite(value.y) && PpmFinite(value.z) && all(value >= 0);
}
bool PpmConfiguration()
{
    return MaxReflectionBounces <= PpmMaximumReflections && SlotsPerPath == MaxReflectionBounces + 1 &&
           TotalSamples > 0 && PathsPerPass > 0 && !AdaptiveEnabled && ShadowStrength == 1 &&
           ((TransportFlags >> MisShift) & MisMask) == 0 && BucketCount > 0 &&
           !(BucketCount & (BucketCount - 1)) && InitialRadius >= 0 && PpmFinite(InitialRadius) &&
           RadiusAlpha > 0 && RadiusAlpha < 1 && EmissionRadius >= 0 && isfinite(EmissionRadius) &&
           LightCountInBatch <= PpmBatchPaths && LightBegin <= PathsPerPass &&
           LightCountInBatch <= PathsPerPass - LightBegin;
}
RayData PpmRay(float3 origin, float3 direction)
{
    RayData ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.MinimumDistance = RayEpsilon;
    ray.MaximumDistance = MaximumRayDistance;
    return ray;
}
EmissionDomain PpmDomain()
{
    EmissionDomain domain;
    domain.Center = EmissionCenter;
    domain.Radius = EmissionRadius;
    return domain;
}
bool PpmGridCell(float3 position, out int3 cell)
{
    cell = 0;
    if (!(InitialRadius > 0))
        return false;
    double3 coordinate = (double3(position) - double3(EmissionCenter)) / InitialRadius;
    if (!PpmCellValid(coordinate.x) || !PpmCellValid(coordinate.y) || !PpmCellValid(coordinate.z))
    {
        PpmFail(PpmIndexFailure);
        return false;
    }
    cell = int3(PpmCellCoordinate(coordinate.x), PpmCellCoordinate(coordinate.y),
                PpmCellCoordinate(coordinate.z));
    return true;
}
float PpmDecode(uint channel)
{
    static const float ByteMaximum = 255, Threshold = 0.04045f, Divisor = 12.92f;
    static const float Offset = 0.055f, Scale = 1.055f, Exponent = 2.4f;
    float value = channel / ByteMaximum;
    return value <= Threshold ? value / Divisor : pow((value + Offset) / Scale, Exponent);
}
float3 PpmTexel(PpmTexture texture, int2 pixel)
{
    int2 size = int2(texture.Width, texture.Height);
    pixel = (pixel % size + size) % size;
    uint value = TextureTexels[texture.OffsetTexels + pixel.y * texture.Width + pixel.x];
    static const uint ByteMask = 0xff, GreenShift = 8, BlueShift = 16;
    return float3(PpmDecode(value & ByteMask), PpmDecode((value >> GreenShift) & ByteMask),
                  PpmDecode((value >> BlueShift) & ByteMask));
}
float3 PpmSurfaceColor(HitData hit)
{
    uint id = Materials[hit.MaterialIndex].DiffuseTextureIndex;
    if (!hit.HasTexCoord || id >= TextureCount || !all(isfinite(hit.TexCoord)))
        return hit.Color;
    PpmTexture texture = Textures[id];
    if (texture.Width == 0 || texture.Height == 0 || texture.OffsetTexels > TextureTexelCount ||
        texture.Height > (TextureTexelCount - texture.OffsetTexels) / texture.Width)
        return hit.Color;
    float2 coordinate =
        frac(float2(hit.TexCoord.x, 1 - hit.TexCoord.y)) * float2(texture.Width, texture.Height) - 0.5f;
    int2 corner = int2(floor(coordinate));
    float2 blend = frac(coordinate);
    float3 row0 = lerp(PpmTexel(texture, corner), PpmTexel(texture, corner + int2(1, 0)), blend.x);
    float3 row1 =
        lerp(PpmTexel(texture, corner + int2(0, 1)), PpmTexel(texture, corner + int2(1, 1)), blend.x);
    return hit.Color * lerp(row0, row1, blend.y);
}
bool PpmRectangle(PackedEmitter emitter, RayData ray, inout float distance)
{
    if (emitter.Kind != PackedEmitterRectangle)
        return false;
    float denominator = dot(emitter.Direction, ray.Direction);
    if (denominator == 0)
        return false;
    float candidate = dot(emitter.Origin - ray.Origin, emitter.Direction) / denominator;
    if (!(candidate > ray.MinimumDistance && candidate < ray.MaximumDistance && candidate <= distance))
        return false;
    float3 helper = abs(emitter.Direction.y) < BasisParallelThreshold ? float3(0, 1, 0) : float3(1, 0, 0);
    float3 right = normalize(cross(helper, emitter.Direction)), up = cross(emitter.Direction, right);
    float3 relative = ray.Origin + candidate * ray.Direction - emitter.Origin;
    if (abs(dot(relative, right)) > emitter.Width / 2 || abs(dot(relative, up)) > emitter.Height / 2)
        return false;
    distance = candidate;
    return true;
}
struct PpmEndpoint
{
    HitData Hit;
    float3 Position;
    uint Source;
    bool Surface;
};
PpmEndpoint PpmTrace(RayData ray, inout PpmStatistics statistics)
{
    PpmEndpoint endpoint = (PpmEndpoint)0;
    endpoint.Source = PpmInvalid;
    endpoint.Hit.Distance = ray.MaximumDistance;
    endpoint.Surface = IntersectScene(ray, false, endpoint.Hit, statistics.Tests);
    bool proxy = endpoint.Surface && Materials[endpoint.Hit.MaterialIndex].SampledEmitter != 0;
    float closest = endpoint.Surface ? endpoint.Hit.Distance + (proxy ? RayEpsilon : 0) : ray.MaximumDistance;
    for (uint source = 0; source < EmitterCount; ++source)
        if (Emitters[source].Kind == PackedEmitterRectangle)
        {
            IncrementCounter(statistics.Tests);
            if (PpmRectangle(Emitters[source], ray, closest))
                endpoint.Source = source;
        }
    if (endpoint.Source != PpmInvalid)
        endpoint.Surface = false;
    else
        for (uint source = 0; source < EmitterCount; ++source)
        {
            PackedEmitter emitter = Emitters[source];
            bool matched = endpoint.Surface ? emitter.Primitive != PpmInvalid &&
                                                  emitter.Primitive == endpoint.Hit.PrimitiveIndex
                                            : emitter.Kind == PackedEmitterEnvironment;
            if (matched)
            {
                endpoint.Source = source;
                break;
            }
        }
    endpoint.Position = ray.Origin + ray.Direction * (endpoint.Surface ? endpoint.Hit.Distance : closest);
    return endpoint;
}
float3 PpmRadiance(PackedEmitter emitter, float3 outgoing)
{
    if (emitter.Kind == PackedEmitterEnvironment)
        return lerp(emitter.BackgroundBottom, emitter.BackgroundTop, (1 - outgoing.y) / 2);
    if (emitter.Kind == PackedEmitterRectangle)
        return dot(emitter.Direction, outgoing) > 0 ? emitter.Strength / (emitter.Width * emitter.Height) : 0;
    return emitter.Strength;
}
SurfaceSample PpmMirror(SurfaceBsdf surface)
{
    SurfaceSample event = (SurfaceSample)0;
    event.Direction = SurfaceWorld(surface, float3(-surface.View.xy, surface.View.z));
    if (surface.View.z <= 0 || !SurfaceHemisphere(surface, event.Direction, false))
        return (SurfaceSample)0;
    event.Weight = SurfaceAdjointCorrection(surface, event.Direction).xxx;
    event.Pdf = event.ReversePdf = event.RelativeIndex = 1;
    event.Event = ScatteringReflection | ScatteringDelta;
    return event;
}
// Normalize after scaling, preserving representable inverse-square weights when
// squaring a float segment would overflow. No double sqrt is required by SM 6.0.
double PpmDistance(double3 segment)
{
    double scale = max(abs(segment.x), max(abs(segment.y), abs(segment.z)));
    return scale > 0 ? scale * double(length(float3(segment / scale))) : 0;
}
double3 PpmDirectLight(PpmEndpoint vertex, SurfaceBsdf surface, SampleContext samples, uint dimension,
                       bool filterTransmission, inout PpmStatistics statistics)
{
    EmitterEndpoint endpoint = SampleEmitterEndpoint(
        Emitters, EmitterCount, PpmDomain(), Sample2D(samples, dimension ^ PpmSourceDomain).x,
        Sample2D(samples, dimension ^ PpmPositionDomain), Sample2D(samples, dimension ^ PpmDirectionDomain));
    EmissionSample light = endpoint.Emission;
    if (!(light.Flags & EmissionValid))
        return 0;
    bool infinite = (light.Flags & EmissionInfinite) != 0, pointSource = (light.Flags & EmissionDeltaPosition) != 0;
    float3 direction = -light.Direction;
    double geometry = 1;
    if (!infinite)
    {
        double3 segment = double3(light.Origin) - double3(vertex.Position);
        double distance = PpmDistance(segment);
        if (!(distance > 0) || !PpmFinite(distance))
            return 0;
        direction = float3(segment / distance);
        geometry = double(pointSource ? 1 : abs(dot(light.Normal, direction))) / (distance * distance);
    }
    bool transmitted = filterTransmission &&
        (vertex.Hit.FrontFace ? dot(direction, vertex.Hit.GeometricNormal) < 0
                              : dot(direction, vertex.Hit.GeometricNormal) > 0);
    if (transmitted && !pointSource && !(light.Flags & EmissionDeltaDirection)) return 0;
    double density =
        double(endpoint.SelectionProbability) * double(infinite ? light.DirectionPdf : light.PositionPdf);
    if (!(density > 0))
        return 0;
    SurfaceEvaluation bsdf = EvaluateSurface(surface, direction);
    double3 result = double3(bsdf.Value) * double3(PpmRadiance(Emitters[endpoint.EmitterIndex], -direction)) *
                     (double(abs(dot(surface.Normal, direction))) * geometry / density);
    if (transmitted && vertex.Hit.FrontFace && pointSource)
        result *= double(exp(-float(PpmDistance(double3(light.Origin) - double3(vertex.Position)) /
                                    double(surface.Material.Optical.w))));
    if (!PpmWideColor(result))
    {
        PpmFail(PpmNumericalFailure);
        return 0;
    }
    if (!any(result > 0))
        return 0;
    RayData shadow = PpmRay(SurfaceOffset(surface, vertex.Position, direction), direction);
    if (!infinite)
    {
        double3 segment = double3(light.Origin) - double3(shadow.Origin);
        double distance = PpmDistance(segment);
        if (!(distance > 2 * RayEpsilon))
            return 0;
        shadow.Direction = float3(segment / distance);
        shadow.MaximumDistance = float(distance) - RayEpsilon;
    }
    IncrementCounter(statistics.Shadow);
    HitData hit = (HitData)0;
    hit.Distance = shadow.MaximumDistance;
    if (IntersectScene(shadow, true, hit, statistics.Tests))
        return 0;
    float closest = shadow.MaximumDistance;
    for (uint source = 0; source < EmitterCount; ++source)
        if (Emitters[source].Kind == PackedEmitterRectangle)
        {
            IncrementCounter(statistics.Tests);
            if (PpmRectangle(Emitters[source], shadow, closest))
                return 0;
        }
    return result;
}
bool PpmSelectVisible(inout PpmVisiblePoint visible, inout uint candidates, float variate)
{
    ++candidates;
    if (visible.Active) visible.Throughput *= float(candidates) / float(candidates - 1);
    return candidates == 1 || variate < 1.0f / float(candidates);
}
void PpmCameraPath(RayData ray, SampleContext samples, out PpmVisiblePoint visible, out double3 direct,
                   inout PpmStatistics statistics)
{
    visible = (PpmVisiblePoint)0;
    direct = 0;
    float3 throughput = 1;
    float etaScale = 1;
    uint interiorMaterial = PpmInvalid, candidates = 0;
    for (uint depth = 0; depth <= MaxReflectionBounces + 1; ++depth)
    {
        if (!PpmColor(throughput) || !isfinite(etaScale))
        {
            PpmFail(PpmNumericalFailure);
            return;
        }
        if (!any(throughput > 0))
            return;
        if (depth == 0)
            IncrementCounter(statistics.Primary);
        else
            IncrementCounter(statistics.Reflection);
        PpmEndpoint vertex = PpmTrace(ray, statistics);
        uint domain = PpmCameraDomain ^ (depth * PpmDepthDomain);
        if (vertex.Surface && vertex.Hit.MaterialIndex >= MaterialCount)
        {
            PpmFail(PpmConfigurationFailure);
            return;
        }
        if (depth == 0 && vertex.Surface && !vertex.Hit.FrontFace &&
            Materials[vertex.Hit.MaterialIndex].Optical.w > 0)
            interiorMaterial = vertex.Hit.MaterialIndex;
        if (interiorMaterial != PpmInvalid)
        {
            PackedMaterial interior = Materials[interiorMaterial];
            bool finiteSource = vertex.Source != PpmInvalid &&
                Emitters[vertex.Source].Kind == PackedEmitterRectangle;
            float boundary = vertex.Surface || finiteSource
                ? length(vertex.Position - ray.Origin) : ray.MaximumDistance;
            SubsurfaceFlight flight = SubsurfaceFreeFlight(interior.Optical.w,
                interior.AlbedoReflectivity.xyz * interior.ConductorExtinction.w,
                ray.Direction, boundary, Sample2D(samples, domain ^ PpmInteriorDomain).x);
            throughput *= flight.Weight;
            if (!PpmColor(throughput))
            {
                PpmFail(PpmNumericalFailure);
                return;
            }
            if (!any(throughput > 0)) return;
            if (flight.Scattered)
            {
                if (depth > MaxReflectionBounces) return;
                if (PpmSelectVisible(visible, candidates, Sample2D(samples, domain ^ PpmVisibleDomain).x))
                {
                    visible.Position = ray.Origin + flight.Distance * ray.Direction;
                    visible.Active = PpmVisibleVolume;
                    visible.View = ray.Direction;
                    visible.Material = interiorMaterial;
                    visible.Depth = depth;
                    visible.Throughput = throughput * float(candidates);
                }
                return;
            }
        }
        if (vertex.Source != PpmInvalid)
            direct += double3(throughput) * double3(PpmRadiance(Emitters[vertex.Source], -ray.Direction));
        if (!vertex.Surface || depth > MaxReflectionBounces)
            return;
        PackedMaterial material = Materials[vertex.Hit.MaterialIndex];
        bool legacy = material.Model == MaterialLegacy;
        float reflectivity = legacy ? saturate(material.AlbedoReflectivity.w) : 0;
        if (legacy)
            material.Model = MaterialDiffuse;
        float3 color = PpmSurfaceColor(vertex.Hit);
        SurfaceBsdf surface = MakeSurfaceBsdf(material, vertex.Hit, -ray.Direction, color);
        bool subsurfaceTransmission = material.Optical.w > 0 && !SurfaceDelta(surface);
        if (!SurfaceDelta(surface) && reflectivity < 1 && (TransportFlags & NeeBit))
            direct += double3(throughput) * double(1 - reflectivity) *
                      PpmDirectLight(vertex, surface, samples, domain, subsurfaceTransmission, statistics);
        bool mirror = legacy && Sample2D(samples, domain ^ PpmEventDomain).x < reflectivity;
        if (!mirror && !SurfaceDelta(surface))
        {
            // Legacy lobe selection already accounts for the diffuse probability.
            if (PpmSelectVisible(visible, candidates, Sample2D(samples, domain ^ PpmVisibleDomain).x))
            {
                visible.Position = vertex.Position;
                visible.Active = PpmVisibleSurface;
                visible.Normal = vertex.Hit.Normal;
                visible.FrontFace = vertex.Hit.FrontFace;
                visible.Geometry = vertex.Hit.GeometricNormal;
                visible.Depth = depth;
                visible.View = -ray.Direction;
                visible.Material = vertex.Hit.MaterialIndex;
                visible.Color = color;
                visible.Throughput = throughput * float(candidates);
            }
            if (!subsurfaceTransmission) return;
        }
        SurfaceSample event;
        if (mirror)
            event = PpmMirror(surface);
        else
            event = SampleSurface(surface, Sample2D(samples, domain ^ PpmScatterDomain),
                                  Sample2D(samples, domain ^ PpmEventDomain).x);
        if (event.Event == 0 || !(event.Pdf > 0))
            return;
        if (subsurfaceTransmission && !(event.Event & ScatteringTransmission)) return;
        throughput *= event.Weight;
        etaScale *= event.RelativeIndex * event.RelativeIndex;
        if (!PpmColor(throughput) || !isfinite(etaScale))
        {
            PpmFail(PpmNumericalFailure);
            return;
        }
        if ((TransportFlags & RouletteBit) && depth + 1 >= RouletteStartDepth)
        {
            float survival = RouletteProbability(throughput * etaScale);
            if (Sample2D(samples, domain ^ PpmRouletteDomain).x >= survival)
                return;
            throughput /= survival;
        }
        ray = PpmRay(SurfaceOffset(surface, vertex.Position, event.Direction), event.Direction);
        if (material.Optical.w > 0 && (event.Event & ScatteringTransmission))
            interiorMaterial = vertex.Hit.FrontFace ? vertex.Hit.MaterialIndex : PpmInvalid;
    }
}

[numthreads(PpmThreads, 1, 1)]
void PpmClearKernel(uint3 thread : SV_DispatchThreadID)
{
    if (thread.x >= WorkCount)
        return;
    uint index = WorkBegin + thread.x;
    if (index < BucketCount)
        Heads[index] = PpmInvalid;
    // Each preceding dispatch has already reported its error before a reset.
    if (index == 0)
        PpmError.Store(0, 0);
}[numthreads(PpmThreads, 1, 1)]
void PpmInitializeKernel(uint3 thread : SV_DispatchThreadID)
{
    if (thread.x >= WorkCount)
        return;
    if (!PpmConfiguration())
    {
        PpmFail(PpmConfigurationFailure);
        return;
    }
    uint pixel = WorkBegin + thread.x;
    if (pixel >= Width * Height)
        return;
    PpmPixel state = (PpmPixel)0;
    state.RadiusSquared = InitialRadius * InitialRadius;
    state.VolumeRadiusSquared = state.RadiusSquared;
    Pixels[pixel] = state;
    VisiblePoints[pixel] = (PpmVisiblePoint)0;
    Accumulation[pixel] = 0;
    SampleCounts[pixel] = 0;
    SampleTargets[pixel] = TotalSamples;
}
[numthreads(PpmThreads, 1, 1)]
void PpmCameraKernel(uint3 thread : SV_DispatchThreadID)
{
    if (thread.x >= WorkCount)
        return;
    PpmStatistics statistics = (PpmStatistics)0;
    Statistics[thread.x] = statistics;
    if (!PpmConfiguration())
    {
        PpmFail(PpmConfigurationFailure);
        return;
    }
    uint pixel = WorkBegin + thread.x;
    if (pixel >= Width * Height)
        return;
    uint x = pixel % Width + TileX, y = pixel / Width + TileY;
    SampleContext samples = MakeSampleContext(x, y, FullWidth, PassIndex, TotalSamples, SamplingSequence, SamplingSeed);
    float2 jitter = Sample2D(samples, PpmRasterDimension);
    float2 raster = float2(PpmRasterCoordinate(x, FullWidth, jitter.x),
                          PpmRasterCoordinate(y, FullHeight, jitter.y));
    CameraEndpointFrame camera;
    camera.Position = CameraPosition;
    camera.Forward = CameraForward;
    camera.Right = CameraRight;
    camera.Up = CameraUp;
    camera.HalfWidth = HalfWidth;
    camera.HalfHeight = HalfHeight;
    camera.LensRadius = LensRadius;
    camera.FocusDistance = FocusDistance;
    CameraEndpointSample start = SampleCameraEndpoint(camera, raster, Sample2D(samples, PpmLensDomain));
    if (!(start.Flags & CameraEndpointValid))
    {
        PpmFail(PpmNumericalFailure);
        return;
    }
    PpmVisiblePoint visible;
    double3 direct;
    PpmCameraPath(PpmRay(start.Origin, start.Direction), samples, visible, direct, statistics);
    if (!PpmWideColor(direct))
    {
        PpmFail(PpmNumericalFailure);
        return;
    }
    PpmPixel state = Pixels[pixel];
    state.Direct += direct;
    state.Incoming = 0;
    state.Arrivals = 0;
    state.VolumeIncoming = 0;
    state.VolumeArrivals = 0;
    Pixels[pixel] = state;
    VisiblePoints[pixel] = visible;
    Statistics[thread.x] = statistics;
}[numthreads(PpmThreads, 1, 1)]
void PpmEmitKernel(uint3 thread : SV_DispatchThreadID)
{
    uint lane = thread.x;
    if (lane >= WorkCount || lane >= LightCountInBatch)
        return;
    PpmStatistics statistics = (PpmStatistics)0;
    Statistics[lane] = statistics;
    if (!PpmConfiguration())
    {
        PpmFail(PpmConfigurationFailure);
        return;
    }
    uint first = lane * SlotsPerPath;
    for (uint slot = 0; slot < SlotsPerPath; ++slot)
        Photons[first + slot] = (PpmPhoton)0;
    uint path = LightBegin + lane;
    EmitterEndpoint endpoint = SampleEmitterEndpoint(
        Emitters, EmitterCount, PpmDomain(), PpmLightSample(path, PpmSourceDomain).x,
        PpmLightSample(path, PpmPositionDomain), PpmLightSample(path, PpmDirectionDomain));
    EmissionSample start = endpoint.Emission;
    if (!(start.Flags & EmissionValid))
        return;
    float3 offset = start.Normal * (dot(start.Normal, start.Direction) >= 0 ? RayEpsilon : -RayEpsilon);
    RayData ray = PpmRay(start.Origin + offset, start.Direction);
    float3 flux = start.Weight;
    bool deltaPrefix = true;
    uint interiorMaterial = PpmInvalid;
    for (uint depth = 0; depth <= MaxReflectionBounces; ++depth)
    {
        if (!PpmColor(flux))
        {
            PpmFail(PpmNumericalFailure);
            break;
        }
        if (!any(flux > 0))
            break;
        IncrementCounter(statistics.Reflection);
        PpmEndpoint vertex = PpmTrace(ray, statistics);
        uint domain = PpmLightDomain ^ (depth * PpmDepthDomain);
        if (vertex.Surface && vertex.Hit.MaterialIndex >= MaterialCount)
        {
            PpmFail(PpmConfigurationFailure);
            break;
        }
        if (depth == 0 && vertex.Surface && !vertex.Hit.FrontFace &&
            Materials[vertex.Hit.MaterialIndex].Optical.w > 0)
            interiorMaterial = vertex.Hit.MaterialIndex;
        if (interiorMaterial != PpmInvalid)
        {
            PackedMaterial interior = Materials[interiorMaterial];
            bool finiteSource = vertex.Source != PpmInvalid &&
                Emitters[vertex.Source].Kind == PackedEmitterRectangle;
            float boundary = vertex.Surface || finiteSource
                ? length(vertex.Position - ray.Origin) : ray.MaximumDistance;
            SubsurfaceFlight flight = SubsurfaceFreeFlight(interior.Optical.w,
                interior.AlbedoReflectivity.xyz * interior.ConductorExtinction.w,
                ray.Direction, boundary, PpmLightSample(path, domain ^ PpmInteriorDomain).x);
            float3 collisionPosition = ray.Origin + flight.Distance * ray.Direction;
            if (flight.Scattered)
            {
                PpmPhoton record = (PpmPhoton)0;
                record.Position = collisionPosition;
                record.Next = PpmInvalid;
                record.Incoming = -ray.Direction;
                record.Depth = depth;
                record.Flux = flux;
                record.Kind = PpmClass(depth, deltaPrefix) | PpmVolumeFlag;
                record.Reserved = interiorMaterial;
                Photons[first + depth] = record;
            }
            flux *= flight.Weight;
            if (!PpmColor(flux))
            {
                PpmFail(PpmNumericalFailure);
                break;
            }
            if (!any(flux > 0)) break;
            if (flight.Scattered)
            {
                if (depth == MaxReflectionBounces) break;
                SubsurfaceDirection phase = SubsurfaceSampleDirection(ray.Direction,
                    interior.ConductorIndex.w, PpmLightSample(path, domain ^ PpmScatterDomain),
                    (TransportFlags & ImportanceBit) != 0);
                if (!(phase.Pdf > 0) || !isfinite(phase.Weight))
                {
                    PpmFail(PpmNumericalFailure);
                    break;
                }
                flux *= phase.Weight;
                if (!PpmColor(flux))
                {
                    PpmFail(PpmNumericalFailure);
                    break;
                }
                if ((TransportFlags & RouletteBit) && depth + 1 >= RouletteStartDepth)
                {
                    float survival = RouletteProbability(flux);
                    if (PpmLightSample(path, domain ^ PpmRouletteDomain).x >= survival) break;
                    flux /= survival;
                }
                deltaPrefix = false;
                ray = PpmRay(collisionPosition, phase.Direction);
                continue;
            }
        }
        if (!vertex.Surface)
            break;
        PackedMaterial material = Materials[vertex.Hit.MaterialIndex];
        bool legacy = material.Model == MaterialLegacy;
        float reflectivity = legacy ? saturate(material.AlbedoReflectivity.w) : 0;
        if (legacy)
            material.Model = MaterialDiffuse;
        SurfaceBsdf surface = MakeSurfaceBsdf(material, vertex.Hit, -ray.Direction,
                                              PpmSurfaceColor(vertex.Hit), TransportImportance);
        if (!SurfaceDelta(surface) && reflectivity < 1)
        {
            PpmPhoton record = (PpmPhoton)0;
            record.Position = vertex.Position;
            record.Next = PpmInvalid;
            record.Incoming = -ray.Direction;
            record.Depth = depth;
            record.Flux = flux;
            record.Kind = PpmClass(depth, deltaPrefix);
            record.Normal = vertex.Hit.GeometricNormal;
            record.Reserved = vertex.Hit.MaterialIndex;
            Photons[first + depth] = record;
        }
        if (depth == MaxReflectionBounces)
            break;
        float eventU = PpmLightSample(path, domain ^ PpmEventDomain).x;
        SurfaceSample event;
        if (legacy && eventU < reflectivity)
            event = PpmMirror(surface);
        else
            event = SampleSurface(surface, PpmLightSample(path, domain ^ PpmScatterDomain), eventU);
        if (event.Event == 0 || !(event.Pdf > 0))
            break;
        flux *= event.Weight;
        deltaPrefix = deltaPrefix && (event.Event & ScatteringDelta) != 0;
        if (!PpmColor(flux))
        {
            PpmFail(PpmNumericalFailure);
            break;
        }
        if ((TransportFlags & RouletteBit) && depth + 1 >= RouletteStartDepth)
        {
            float survival = RouletteProbability(flux);
            if (PpmLightSample(path, domain ^ PpmRouletteDomain).x >= survival)
                break;
            flux /= survival;
        }
        ray = PpmRay(SurfaceOffset(surface, vertex.Position, event.Direction), event.Direction);
        if (material.Optical.w > 0 && (event.Event & ScatteringTransmission))
            interiorMaterial = vertex.Hit.FrontFace ? vertex.Hit.MaterialIndex : PpmInvalid;
    }
    Statistics[lane] = statistics;
}
[numthreads(PpmThreads, 1, 1)]
void PpmIndexKernel(uint3 thread : SV_DispatchThreadID)
{
    if (thread.x >= WorkCount)
        return;
    uint index = WorkBegin + thread.x;
    if (index >= LightCountInBatch * SlotsPerPath)
        return;
    PpmPhoton record = Photons[index];
    if (!PpmIncluded(record.Kind, (TransportFlags & NeeBit) != 0, PathClasses))
        return;
    int3 cell;
    if (!PpmGridCell(record.Position, cell))
        return;
    uint old;
    InterlockedExchange(Heads[PpmBucket(cell.x, cell.y, cell.z, BucketCount)], index, old);
    Photons[index].Next = old;
}[numthreads(PpmThreads, 1, 1)]
void PpmGatherKernel(uint3 thread : SV_DispatchThreadID)
{
    if (thread.x >= WorkCount)
        return;
    uint pixel = WorkBegin + thread.x;
    if (pixel >= Width * Height)
        return;
    PpmVisiblePoint visible = VisiblePoints[pixel];
    PpmPixel state = Pixels[pixel];
    if (!visible.Active || !(visible.Active == PpmVisibleVolume
        ? state.VolumeRadiusSquared > 0 : state.RadiusSquared > 0))
        return;
    int3 center;
    if (!PpmGridCell(visible.Position, center))
        return;
    SurfaceBsdf surface = (SurfaceBsdf)0;
    if (visible.Active == PpmVisibleSurface)
    {
        HitData hit = (HitData)0;
        hit.Normal = visible.Normal;
        hit.GeometricNormal = visible.Geometry;
        hit.FrontFace = visible.FrontFace;
        PackedMaterial material = Materials[visible.Material];
        if (material.Model == MaterialLegacy)
            material.Model = MaterialDiffuse;
        surface = MakeSurfaceBsdf(material, hit, visible.View, visible.Color);
    }
    PpmStatistics counters = (PpmStatistics)0;
    for (int z = -1; z <= 1; ++z)
        for (int y = -1; y <= 1; ++y)
            for (int x = -1; x <= 1; ++x)
            {
                int3 cell = center + int3(x, y, z);
                uint link = Heads[PpmBucket(cell.x, cell.y, cell.z, BucketCount)], visited = 0;
                while (link != PpmInvalid)
                {
                    if (link >= LightCountInBatch * SlotsPerPath ||
                        ++visited > LightCountInBatch * SlotsPerPath)
                    {
                        PpmFail(PpmIndexFailure);
                        return;
                    }
                    PpmPhoton photon = Photons[link];
                    link = photon.Next;
                    if (!PpmIncluded(photon.Kind, (TransportFlags & NeeBit) != 0, PathClasses))
                        continue;
                    bool volume = (photon.Kind & PpmVolumeFlag) != 0;
                    if (volume != (visible.Active == PpmVisibleVolume)) continue;
                    int3 stored;
                    if (!PpmGridCell(photon.Position, stored) || any(stored != cell))
                        continue;
                    double3 difference = double3(photon.Position) - double3(visible.Position);
                    double distanceSquared = difference.x * difference.x + difference.y * difference.y +
                                             difference.z * difference.z;
                    if (volume)
                    {
                        if (photon.Reserved != visible.Material || visible.Depth > MaxReflectionBounces ||
                            photon.Depth > MaxReflectionBounces - visible.Depth ||
                            distanceSquared > state.VolumeRadiusSquared)
                            continue;
                        float3 separation = photon.Position - visible.Position;
                        float distance = length(separation);
                        if (distance > 2 * RayEpsilon)
                        {
                            IncrementCounter(counters.Shadow);
                            RayData connection = PpmRay(visible.Position, separation / distance);
                            connection.MaximumDistance = distance - RayEpsilon;
                            HitData blocker = (HitData)0;
                            blocker.Distance = connection.MaximumDistance;
                            if (IntersectScene(connection, true, blocker, counters.Tests)) continue;
                        }
                        PackedMaterial interior = Materials[visible.Material];
                        double phase = double(SubsurfacePhase(dot(photon.Incoming, visible.View),
                                                               interior.ConductorIndex.w));
                        double3 flux = double3(photon.Flux) * double3(visible.Throughput) *
                                       (phase * double(interior.Optical.w));
                        if (!PpmWideColor(flux))
                        {
                            PpmFail(PpmNumericalFailure);
                            return;
                        }
                        state.VolumeIncoming += flux;
                        ++state.VolumeArrivals;
                        continue;
                    }
                    if (Materials[visible.Material].Optical.w > 0 &&
                        (visible.FrontFace ? dot(photon.Incoming, visible.Geometry) <= 0
                                           : dot(photon.Incoming, visible.Geometry) >= 0))
                        continue;
                    double planeDistance = difference.x * visible.Geometry.x + difference.y * visible.Geometry.y +
                                           difference.z * visible.Geometry.z;
                    if (!PpmMergeable(distanceSquared, state.RadiusSquared,
                                      double(dot(photon.Normal, visible.Geometry)), planeDistance, photon.Depth,
                                      visible.Depth, MaxReflectionBounces))
                        continue;
                    double geometricCosine = double(abs(dot(visible.Geometry, photon.Incoming)));
                    if (!(geometricCosine > 0))
                        continue;
                    SurfaceEvaluation bsdf = EvaluateSurface(surface, photon.Incoming);
                    double3 flux = double3(photon.Flux) * double3(bsdf.Value) * double3(visible.Throughput) *
                                   (double(abs(dot(visible.Normal, photon.Incoming))) / geometricCosine);
                    if (!PpmWideColor(flux))
                    {
                        PpmFail(PpmNumericalFailure);
                        return;
                    }
                    state.Incoming += flux;
                    ++state.Arrivals;
                }
            }
    if (!PpmWideColor(state.Incoming) || !PpmWideColor(state.VolumeIncoming))
    {
        PpmFail(PpmNumericalFailure);
        return;
    }
    Pixels[pixel] = state;
    Statistics[thread.x] = counters;
}
[numthreads(PpmThreads, 1, 1)]
void PpmUpdateKernel(uint3 thread : SV_DispatchThreadID)
{
    if (thread.x >= WorkCount)
        return;
    uint pixel = WorkBegin + thread.x;
    if (pixel >= Width * Height)
        return;
    PpmPixel state = Pixels[pixel];
    PpmProgress old;
    old.RadiusSquared = state.RadiusSquared;
    old.EffectiveCount = state.EffectiveCount;
    old.Red = state.Flux.x;
    old.Green = state.Flux.y;
    old.Blue = state.Flux.z;
    PpmProgress next =
        PpmAdvance(old, state.Arrivals, state.Incoming.x, state.Incoming.y, state.Incoming.z, RadiusAlpha);
    state.RadiusSquared = next.RadiusSquared;
    state.EffectiveCount = next.EffectiveCount;
    state.Flux = double3(next.Red, next.Green, next.Blue);
    PpmProgress oldVolume;
    oldVolume.RadiusSquared = state.VolumeRadiusSquared;
    oldVolume.EffectiveCount = state.VolumeEffectiveCount;
    oldVolume.Red = state.VolumeFlux.x;
    oldVolume.Green = state.VolumeFlux.y;
    oldVolume.Blue = state.VolumeFlux.z;
    PpmProgress nextVolume = PpmAdvanceVolume(oldVolume, state.VolumeArrivals,
        state.VolumeIncoming.x, state.VolumeIncoming.y, state.VolumeIncoming.z, RadiusAlpha);
    state.VolumeRadiusSquared = nextVolume.RadiusSquared;
    state.VolumeEffectiveCount = nextVolume.EffectiveCount;
    state.VolumeFlux = double3(nextVolume.Red, nextVolume.Green, nextVolume.Blue);
    uint completed = PassIndex + 1;
    double3 radiance =
        double3(PpmResolve(state.Direct.x, state.Flux.x, state.RadiusSquared, completed, PathsPerPass) +
                    PpmResolveVolume(state.VolumeFlux.x, state.VolumeRadiusSquared, completed, PathsPerPass),
                PpmResolve(state.Direct.y, state.Flux.y, state.RadiusSquared, completed, PathsPerPass) +
                    PpmResolveVolume(state.VolumeFlux.y, state.VolumeRadiusSquared, completed, PathsPerPass),
                PpmResolve(state.Direct.z, state.Flux.z, state.RadiusSquared, completed, PathsPerPass) +
                    PpmResolveVolume(state.VolumeFlux.z, state.VolumeRadiusSquared, completed, PathsPerPass));
    if (!PpmWideColor(radiance) || !PpmColor(float3(radiance)) || !PpmWideColor(state.Flux) ||
        !PpmWideColor(state.VolumeFlux) || !PpmFinite(state.RadiusSquared) ||
        !PpmFinite(state.EffectiveCount) || !PpmFinite(state.VolumeRadiusSquared) ||
        !PpmFinite(state.VolumeEffectiveCount))
    {
        PpmFail(PpmNumericalFailure);
        return;
    }
    Pixels[pixel] = state;
    Accumulation[pixel] = float4(float3(radiance), 0);
    SampleCounts[pixel] = completed;
}
