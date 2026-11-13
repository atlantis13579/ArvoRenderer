// Jensen (1996), "Global Illumination using Photon Maps". Independent Compute
// implementation: emit importance paths, build a hashed grid, then gather along
// camera delta chains. Direct / delta-prefix caustic / indirect arrivals form a
// disjoint partition. NEE replaces only direct photons. This fixed-radius
// Epanechnikov estimate is biased; it is neither path tracing nor progressive PM.
// https://pbr-book.org/3ed-2018/Light_Transport_III_Bidirectional_Methods/Stochastic_Progressive_Photon_Mapping
// GPU insertion order may vary. Double summation limits error, not bitwise drift.
// Interior collisions use a separate normalized three-dimensional kernel.

static const uint PhotonEmpty = 0, PhotonDirect = 1, PhotonCaustic = 2, PhotonIndirect = 3;
static const uint PhotonVolumeFlag = 0x80000000u, PhotonClassMask = ~PhotonVolumeFlag;
static const uint PhotonCausticsBit = 1, PhotonIndirectBit = 2;
static const uint PhotonInvalid = 0xffffffffu, PhotonMaximumReflections = 32;
static const uint PhotonThreads = 64;
static const double PhotonPi = 3.1415926535897932384626433832795L;
static const double PhotonCellLimit = 1073741824.0L; // Signed cell + neighbor must fit int32.
static const double PhotonDoubleMaximum = 1.7976931348623157e308L;
static const float PhotonNormalAlignment = 0.9f, PhotonPlaneFraction = 0.1f;
static const double PhotonVolumeKernelIntegral = 8.0 / 15.0;

// Truncation plus a negative remainder implements floor without a float cast.
bool PhotonFinite(double value)
{
    return value >= -PhotonDoubleMaximum && value <= PhotonDoubleMaximum;
}
bool PhotonCellValid(double value)
{
    return value > -PhotonCellLimit && value < PhotonCellLimit;
}
int PhotonCellCoordinate(double value)
{
    int truncated = int(value);
    return truncated - (value < double(truncated) ? 1 : 0);
}
uint PhotonHash(uint value)
{
    static const uint FirstMultiplier = 0x7feb352du, SecondMultiplier = 0x846ca68bu;
    static const uint FirstShift = 16, MiddleShift = 15;
    value = (value ^ (value >> FirstShift)) * FirstMultiplier;
    value = (value ^ (value >> MiddleShift)) * SecondMultiplier;
    return value ^ (value >> FirstShift);
}
uint PhotonBucket(int x, int y, int z, uint bucketCount)
{
    static const uint AxisX = 0x9e3779b9u, AxisY = 0x85ebca6bu, AxisZ = 0xc2b2ae35u;
    return PhotonHash(uint(x) * AxisX ^ uint(y) * AxisY ^ uint(z) * AxisZ) & (bucketCount - 1);
}
uint PhotonClass(uint lightDepth, bool deltaPrefix)
{
    return lightDepth == 0 ? PhotonDirect : deltaPrefix ? PhotonCaustic : PhotonIndirect;
}
bool PhotonClassIncluded(uint kind, bool nee, uint classes)
{
    return (kind == PhotonDirect && !nee) || (kind == PhotonCaustic && (classes & PhotonCausticsBit)) ||
           (kind == PhotonIndirect && (classes & PhotonIndirectBit));
}
double PhotonKernel(double squaredDistance, double radius, uint emittedPaths)
{
    if (!(radius > 0) || emittedPaths == 0)
        return 0;
    double squaredRadius = radius * radius;
    if (!(squaredDistance >= 0 && squaredDistance < squaredRadius))
        return 0;
    return (2 / (PhotonPi * squaredRadius * double(emittedPaths))) * (1 - squaredDistance / squaredRadius);
}
double PhotonGatherWeight(double squaredDistance, double radius, uint emittedPaths, float normalAlignment,
                          double planeDistance, float shadingCosine, float geometricCosine)
{
    if (normalAlignment < PhotonNormalAlignment || planeDistance > double(PhotonPlaneFraction) * radius ||
        !(geometricCosine > 0))
        return 0;
    return PhotonKernel(squaredDistance, radius, emittedPaths) * double(shadingCosine) /
           double(geometricCosine);
}

static const float Pi = 3.14159265358979323846f, RayEpsilon = 1.0e-4f;
static const float MaximumRayDistance = 1.0e30f, BasisParallelThreshold = 0.999f;
static const uint ImportanceBit = 1, NeeBit = 2, RouletteBit = 4, MisShift = 3, MisMask = 3;
static const uint PhotonNumericalFailure = 1, PhotonInvalidConfiguration = 2;
static const uint PixelSeedMultiplier = 9781, SampleSeedMultiplier = 6271;
static const uint SecondDimensionScramble = 0x68bc21ebu, BounceSeedMultiplier = 0x9e3779b9u;
static const uint PhotonLightDomain = 0x50484f54u, PhotonCameraDomain = 0x43414d50u;
static const uint PhotonLensDomain = 0x4c454e53u, PhotonSourceDomain = 0x534f5552u;
static const uint PhotonPointDomain = 0x504f494eu, PhotonDirectionDomain = 0x44495245u;
static const uint PhotonScatterDomain = 0x53434154u, PhotonLobeDomain = 0x4c4f4245u;
static const uint PhotonRouletteDomain = 0x52524f55u;
static const uint PhotonEmissionDimensions = 5, PhotonBounceDimensions = 4;
static const uint PhotonSourceOffset = 0, PhotonPositionOffset = 1, PhotonDirectionOffset = 3;
static const uint PhotonEventOffset = 2, PhotonRouletteOffset = 3, PhotonRasterDimension = 0;

#include "Types.hlsl"
#include "MaterialTypes.hlsl"
#include "EmitterData.hlsl"
struct PhotonRecord
{
    float3 Position;
    uint Next;
    float3 Incoming;
    uint LightDepth;
    float3 Flux;
    uint Kind;
    float3 Normal;
    uint Reserved;
};
struct PhotonTexture
{
    uint OffsetTexels, Width, Height, Reserved;
};
struct PhotonStatistics
{
    uint2 Primary, Shadow, Reflection, Tests;
};
ByteAddressBuffer Nodes : register(t0);
StructuredBuffer<uint> PrimitiveRefs : register(t1);
StructuredBuffer<PackedTriangle> Triangles : register(t2);
StructuredBuffer<PackedSphere> Spheres : register(t3);
StructuredBuffer<PackedMaterial> Materials : register(t4);
StructuredBuffer<PackedEmitter> Emitters : register(t5);
StructuredBuffer<PhotonTexture> Textures : register(t6);
StructuredBuffer<uint> TextureTexels : register(t7);
RWStructuredBuffer<float4> Accumulation : register(u0);
RWStructuredBuffer<uint> SampleCounts : register(u1);
RWStructuredBuffer<uint> Heads : register(u2);
RWStructuredBuffer<PhotonStatistics> Statistics : register(u3);
RWStructuredBuffer<uint> SampleTargets : register(u4);
RWStructuredBuffer<PhotonRecord> Photons : register(u5);
// Instancing.hlsl owns u6, t9 and t10.
RWByteAddressBuffer PhotonError : register(u7);
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
cbuffer PhotonControls : register(b1)
{
    float3 EmissionCenter;
    float EmissionRadius;
    double GatherRadius;
    uint EmitterCount, PhotonPaths;
    uint SlotsPerPath, BucketCount, PhotonSeed, PhotonClasses;
    uint WorkBegin, WorkCount, CameraSample, ReservedControl;
};
void IncrementCounter(inout uint2 value)
{
    if (++value.x == 0)
        ++value.y;
}
void PhotonFail(uint flag)
{
    uint ignored;
    PhotonError.InterlockedOr(0, flag, ignored);
}
uint Hash(uint value)
{
    return PhotonHash(value);
}
float Random01(uint key)
{
    static const uint MantissaMask = 0x00ffffffu;
    static const float MantissaRange = 16777216.0f;
    return float(Hash(key) & MantissaMask) / MantissaRange;
}
float PhotonVariate(uint path, uint dimension)
{
    return Random01(Hash(PhotonSeed ^ PhotonLightDomain) ^ Hash(path) ^ Hash(dimension + PhotonSourceDomain));
}
float2 PhotonPair(uint path, uint dimension)
{
    return float2(PhotonVariate(path, dimension), PhotonVariate(path, dimension + 1));
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
#include "FilmMoments.hlsl"

bool PhotonValidColor(float3 value)
{
    return all(isfinite(value)) && all(value >= 0);
}
bool PhotonConfiguration()
{
    return PhotonPaths > 0 && MaxReflectionBounces <= PhotonMaximumReflections &&
           SlotsPerPath == MaxReflectionBounces + 1 && BucketCount > 0 &&
           (BucketCount & (BucketCount - 1)) == 0 && !AdaptiveEnabled &&
           ((TransportFlags >> MisShift) & MisMask) == 0 && ShadowStrength == 1 && GatherRadius >= 0 &&
           PhotonFinite(GatherRadius);
}
EmissionDomain PhotonDomain()
{
    EmissionDomain domain;
    domain.Center = EmissionCenter;
    domain.Radius = EmissionRadius;
    return domain;
}
RayData PhotonRay(float3 position, float3 direction, float maximum = MaximumRayDistance)
{
    RayData ray;
    ray.Origin = position;
    ray.Direction = direction;
    ray.MinimumDistance = RayEpsilon;
    ray.MaximumDistance = maximum;
    return ray;
}
float3 PhotonOffset(float3 position, float3 normal, float3 direction)
{
    return position + normal * (dot(normal, direction) >= 0 ? RayEpsilon : -RayEpsilon);
}
float PhotonDecode(uint component)
{
    static const float ByteMaximum = 255, LinearThreshold = 0.04045f, LinearDivisor = 12.92f;
    static const float TransferOffset = 0.055f, TransferScale = 1.055f, TransferExponent = 2.4f;
    float v = component / ByteMaximum;
    return v <= LinearThreshold ? v / LinearDivisor
                                : pow((v + TransferOffset) / TransferScale, TransferExponent);
}
float3 PhotonTexel(PhotonTexture texture, int2 coordinate)
{
    int2 extent = int2(texture.Width, texture.Height);
    int2 wrapped = (coordinate % extent + extent) % extent;
    uint rgba = TextureTexels[texture.OffsetTexels + wrapped.y * texture.Width + wrapped.x];
    static const uint ByteMask = 0xffu, GreenShift = 8, BlueShift = 16;
    return float3(PhotonDecode(rgba & ByteMask), PhotonDecode((rgba >> GreenShift) & ByteMask),
                  PhotonDecode((rgba >> BlueShift) & ByteMask));
}
float3 PhotonSurfaceColor(HitData hit)
{
    uint index = Materials[hit.MaterialIndex].DiffuseTextureIndex;
    if (!hit.HasTexCoord || index >= TextureCount || !all(isfinite(hit.TexCoord)))
        return hit.Color;
    PhotonTexture texture = Textures[index];
    if (!texture.Width || !texture.Height || texture.OffsetTexels > TextureTexelCount ||
        texture.Height > (TextureTexelCount - texture.OffsetTexels) / texture.Width)
        return hit.Color;
    float2 texel =
        frac(float2(hit.TexCoord.x, 1 - hit.TexCoord.y)) * float2(texture.Width, texture.Height) - 0.5f;
    int2 corner = int2(floor(texel));
    float2 blend = frac(texel);
    return hit.Color *
           lerp(lerp(PhotonTexel(texture, corner), PhotonTexel(texture, corner + int2(1, 0)), blend.x),
                lerp(PhotonTexel(texture, corner + int2(0, 1)), PhotonTexel(texture, corner + int2(1, 1)),
                     blend.x),
                blend.y);
}
bool PhotonRectangle(PackedEmitter source, RayData ray, inout float distance)
{
    if (source.Kind != PackedEmitterRectangle)
        return false;
    float projection = dot(source.Direction, ray.Direction);
    if (projection == 0)
        return false;
    float t = dot(source.Origin - ray.Origin, source.Direction) / projection;
    if (!(t > ray.MinimumDistance && t < ray.MaximumDistance && t <= distance))
        return false;
    float3 helper = abs(source.Direction.y) < BasisParallelThreshold ? float3(0, 1, 0) : float3(1, 0, 0);
    float3 right = normalize(cross(helper, source.Direction)), up = cross(source.Direction, right);
    float3 offset = ray.Origin + t * ray.Direction - source.Origin;
    if (abs(dot(offset, right)) > source.Width / 2 || abs(dot(offset, up)) > source.Height / 2)
        return false;
    distance = t;
    return true;
}
// Analytic rectangles compete with geometry in the same closest-hit query.
// Geometry emitters remain surfaces; an analytic rectangle ends a path.
bool PhotonTrace(RayData ray, out HitData hit, out uint source, inout PhotonStatistics counters)
{
    hit = (HitData)0;
    hit.Distance = ray.MaximumDistance;
    source = PhotonInvalid;
    bool surface = IntersectScene(ray, false, hit, counters.Tests);
    // A sampled rectangle may have a coincident display mesh. Resolve that
    // proxy within the ray offset; ordinary occluders keep their exact bounds.
    const bool proxy = surface && hit.MaterialIndex < MaterialCount &&
                       Materials[hit.MaterialIndex].SampledEmitter != 0;
    float closest = hit.Distance + (proxy ? RayEpsilon : 0);
    for (uint emitter = 0; emitter < EmitterCount; ++emitter)
    {
        PackedEmitter light = Emitters[emitter];
        if (light.Kind == PackedEmitterRectangle)
        {
            IncrementCounter(counters.Tests);
            if (PhotonRectangle(light, ray, closest))
            {
                surface = false;
                source = emitter;
            }
        }
    }
    if (source != PhotonInvalid) hit.Distance = closest;
    if (source == PhotonInvalid)
        for (uint emitter = 0; emitter < EmitterCount; ++emitter)
            if (surface ? Emitters[emitter].Primitive != InvalidPrimitiveIndex &&
                              Emitters[emitter].Primitive == hit.PrimitiveIndex
                        : Emitters[emitter].Kind == PackedEmitterEnvironment)
            {
                source = emitter;
                break;
            }
    return surface;
}
float3 PhotonEmission(PackedEmitter source, float3 outward)
{
    if (source.Kind == PackedEmitterEnvironment)
        return lerp(source.BackgroundBottom, source.BackgroundTop, (1 - outward.y) / 2);
    if (source.Kind == PackedEmitterRectangle)
        return dot(source.Direction, outward) > 0 ? source.Strength / (source.Width * source.Height) : 0;
    return source.Strength;
}
SurfaceSample PhotonMirror(SurfaceBsdf surface)
{
    SurfaceSample sample = (SurfaceSample)0;
    sample.Direction = SurfaceWorld(surface, float3(-surface.View.xy, surface.View.z));
    if (!SurfaceHemisphere(surface, sample.Direction, false))
        return (SurfaceSample)0;
    sample.Weight = SurfaceAdjointCorrection(surface, sample.Direction).xxx;
    sample.Pdf = sample.ReversePdf = sample.RelativeIndex = 1;
    sample.Event = ScatteringDelta | ScatteringReflection;
    return sample;
}
bool PhotonGridCell(float3 position, out int3 cell)
{
    cell = 0;
    if (!(GatherRadius > 0))
        return false;
    double3 coordinate = (double3(position) - double3(EmissionCenter)) / GatherRadius;
    if (!PhotonCellValid(coordinate.x) || !PhotonCellValid(coordinate.y) || !PhotonCellValid(coordinate.z))
    {
        PhotonFail(PhotonNumericalFailure);
        return false;
    }
    cell = int3(PhotonCellCoordinate(coordinate.x), PhotonCellCoordinate(coordinate.y),
                PhotonCellCoordinate(coordinate.z));
    return true;
}

[numthreads(PhotonThreads, 1, 1)]
void PhotonClearKernel(uint3 thread : SV_DispatchThreadID)
{
    if (thread.x < BucketCount)
        Heads[thread.x] = PhotonInvalid;
    if (thread.x == 0)
        PhotonError.Store(0, 0);
}

[numthreads(PhotonThreads, 1, 1)]
void PhotonEmitKernel(uint3 thread : SV_DispatchThreadID)
{
    uint lane = thread.x;
    if (lane >= WorkCount)
        return;
    PhotonStatistics counters = (PhotonStatistics)0;
    Statistics[lane] = counters;
    if (!PhotonConfiguration())
    {
        PhotonFail(PhotonInvalidConfiguration);
        return;
    }
    uint path = WorkBegin + lane;
    if (path >= PhotonPaths)
        return;
    uint first = path * SlotsPerPath;
    for (uint slot = 0; slot < SlotsPerPath; ++slot)
        Photons[first + slot] = (PhotonRecord)0;
    EmitterEndpoint endpoint =
        SampleEmitterEndpoint(Emitters, EmitterCount, PhotonDomain(), PhotonVariate(path, PhotonSourceOffset),
                              PhotonPair(path, PhotonPositionOffset), PhotonPair(path, PhotonDirectionOffset));
    EmissionSample emission = endpoint.Emission;
    if (!(emission.Flags & EmissionValid))
        return;
    float3 flux = emission.Weight;
    RayData ray =
        PhotonRay(PhotonOffset(emission.Origin, emission.Normal, emission.Direction), emission.Direction);
    bool deltaPrefix = true;
    uint interiorMaterial = PhotonInvalid;
    for (uint depth = 0; depth < SlotsPerPath; ++depth)
    {
        if (!PhotonValidColor(flux))
        {
            PhotonFail(PhotonNumericalFailure);
            break;
        }
        if (!any(flux > 0))
            break;
        IncrementCounter(counters.Reflection);
        HitData hit;
        uint source;
        bool found = PhotonTrace(ray, hit, source, counters);
        uint dimension = PhotonEmissionDimensions + depth * PhotonBounceDimensions;
        if (found && hit.MaterialIndex >= MaterialCount)
        {
            PhotonFail(PhotonInvalidConfiguration);
            break;
        }
        if (depth == 0 && found && !hit.FrontFace && Materials[hit.MaterialIndex].Optical.w > 0)
            interiorMaterial = hit.MaterialIndex;
        if (interiorMaterial != PhotonInvalid)
        {
            PackedMaterial interior = Materials[interiorMaterial];
            float boundary = found || (source != PhotonInvalid &&
                Emitters[source].Kind == PackedEmitterRectangle) ? hit.Distance : ray.MaximumDistance;
            SubsurfaceFlight flight = SubsurfaceFreeFlight(interior.Optical.w,
                interior.AlbedoReflectivity.xyz * interior.ConductorExtinction.w,
                ray.Direction, boundary, PhotonVariate(path, dimension + PhotonEventOffset));
            if (flight.Scattered)
            {
                PhotonRecord record = (PhotonRecord)0;
                record.Position = ray.Origin + flight.Distance * ray.Direction;
                record.Next = PhotonInvalid;
                record.Incoming = -ray.Direction;
                record.LightDepth = depth;
                record.Flux = flux;
                record.Kind = PhotonClass(depth, deltaPrefix) | PhotonVolumeFlag;
                record.Reserved = interiorMaterial;
                Photons[first + depth] = record;
            }
            flux *= flight.Weight;
            if (!PhotonValidColor(flux))
            {
                PhotonFail(PhotonNumericalFailure);
                break;
            }
            if (!any(flux > 0)) break;
            if (flight.Scattered)
            {
                if (depth == MaxReflectionBounces) break;
                SubsurfaceDirection phase = SubsurfaceSampleDirection(ray.Direction,
                    interior.ConductorIndex.w, PhotonPair(path, dimension),
                    (TransportFlags & ImportanceBit) != 0);
                if (!(phase.Pdf > 0) || !isfinite(phase.Weight))
                {
                    PhotonFail(PhotonNumericalFailure);
                    break;
                }
                flux *= phase.Weight;
                if ((TransportFlags & RouletteBit) && depth + 1 >= RouletteStartDepth)
                {
                    float survival = RouletteProbability(flux);
                    if (PhotonVariate(path, dimension + PhotonRouletteOffset) >= survival) break;
                    flux /= survival;
                }
                deltaPrefix = false;
                ray = PhotonRay(ray.Origin + flight.Distance * ray.Direction, phase.Direction);
                continue;
            }
        }
        if (!found) break;
        float3 position = ray.Origin + hit.Distance * ray.Direction;
        PackedMaterial material = Materials[hit.MaterialIndex];
        bool legacy = material.Model == MaterialLegacy;
        float mirror = legacy ? saturate(material.AlbedoReflectivity.w) : 0;
        if (legacy)
            material.Model = MaterialDiffuse;
        SurfaceBsdf surface =
            MakeSurfaceBsdf(material, hit, -ray.Direction, PhotonSurfaceColor(hit), TransportImportance);
        if (!SurfaceDelta(surface) && mirror < 1)
        {
            PhotonRecord record = (PhotonRecord)0;
            record.Position = position;
            record.Next = PhotonInvalid;
            record.Incoming = -ray.Direction;
            record.LightDepth = depth;
            record.Flux = flux;
            record.Kind = PhotonClass(depth, deltaPrefix);
            record.Normal = hit.GeometricNormal;
            record.Reserved = hit.MaterialIndex;
            Photons[first + depth] = record;
        }
        if (depth == MaxReflectionBounces)
            break;
        float2 directionU = PhotonPair(path, dimension);
        float eventU = PhotonVariate(path, dimension + PhotonEventOffset);
        SurfaceSample sample;
        if (legacy && eventU < mirror)
            sample = PhotonMirror(surface);
        else
            sample = SampleSurface(surface, directionU, eventU);
        if (!sample.Event || !(sample.Pdf > 0))
            break;
        flux *= sample.Weight;
        deltaPrefix = deltaPrefix && (sample.Event & ScatteringDelta) != 0;
        if (!PhotonValidColor(flux))
        {
            PhotonFail(PhotonNumericalFailure);
            break;
        }
        if ((TransportFlags & RouletteBit) && depth + 1 >= RouletteStartDepth)
        {
            float survival = RouletteProbability(flux);
            if (PhotonVariate(path, dimension + PhotonRouletteOffset) >= survival)
                break;
            flux /= survival;
        }
        ray = PhotonRay(SurfaceOffset(surface, position, sample.Direction), sample.Direction);
        if (material.Optical.w > 0 && (sample.Event & ScatteringTransmission) != 0)
            interiorMaterial = hit.FrontFace ? hit.MaterialIndex : PhotonInvalid;
    }
    Statistics[lane] = counters;
}
[numthreads(PhotonThreads, 1, 1)]
void PhotonIndexKernel(uint3 thread : SV_DispatchThreadID)
{
    if (thread.x >= WorkCount)
        return;
    uint recordIndex = WorkBegin + thread.x;
    if (recordIndex >= PhotonPaths * SlotsPerPath || Photons[recordIndex].Kind == PhotonEmpty)
        return;
    int3 cell;
    if (!PhotonGridCell(Photons[recordIndex].Position, cell))
        return;
    uint previous;
    InterlockedExchange(Heads[PhotonBucket(cell.x, cell.y, cell.z, BucketCount)], recordIndex, previous);
    Photons[recordIndex].Next = previous;
    // No list is read in this dispatch. A host UAV barrier publishes every Next
    // after all insertions complete, so no wave waits on another wave's store.
}

float3 PhotonGather(float3 position, HitData hit, SurfaceBsdf surface, uint cameraDepth)
{
    int3 center;
    if (!PhotonGridCell(position, center))
        return 0;
    double3 sum = 0;
    for (int z = -1; z <= 1; ++z)
        for (int y = -1; y <= 1; ++y)
            for (int x = -1; x <= 1; ++x)
            {
                int3 cell = center + int3(x, y, z);
                uint recordIndex = Heads[PhotonBucket(cell.x, cell.y, cell.z, BucketCount)], visited = 0;
                while (recordIndex != PhotonInvalid)
                {
                    if (recordIndex >= PhotonPaths * SlotsPerPath || ++visited > PhotonPaths * SlotsPerPath)
                    {
                        PhotonFail(PhotonNumericalFailure);
                        return 0;
                    }
                    PhotonRecord record = Photons[recordIndex];
                    recordIndex = record.Next;
                    if ((record.Kind & PhotonVolumeFlag) != 0 ||
                        !PhotonClassIncluded(record.Kind & PhotonClassMask,
                                             (TransportFlags & NeeBit) != 0, PhotonClasses) ||
                        record.LightDepth > MaxReflectionBounces - cameraDepth)
                        continue;
                    if (surface.Material.Optical.w > 0 &&
                        (hit.FrontFace ? dot(record.Incoming, hit.GeometricNormal) <= 0
                                       : dot(record.Incoming, hit.GeometricNormal) >= 0))
                        continue;
                    int3 storedCell;
                    if (!PhotonGridCell(record.Position, storedCell) || any(storedCell != cell))
                        continue;
                    // Checking the exact cell makes hash collisions harmless, including
                    // when several of the 27 neighbor cells hash to the same bucket.
                    double3 difference = double3(record.Position) - double3(position);
                    double squaredDistance = difference.x * difference.x + difference.y * difference.y +
                                             difference.z * difference.z;
                    double planeDistance = abs(difference.x * double(hit.GeometricNormal.x) +
                                               difference.y * double(hit.GeometricNormal.y) +
                                               difference.z * double(hit.GeometricNormal.z));
                    float geometricCosine = abs(dot(hit.GeometricNormal, record.Incoming));
                    // Flux is per geometric area; the BSDF is measured against the shading
                    // frame. Plane/normal rejection limits leakage but adds finite-R bias.
                    double weight = PhotonGatherWeight(
                        squaredDistance, GatherRadius, PhotonPaths, dot(hit.GeometricNormal, record.Normal),
                        planeDistance, abs(dot(surface.Normal, record.Incoming)), geometricCosine);
                    if (!(weight > 0))
                        continue;
                    SurfaceEvaluation scattering = EvaluateSurface(surface, record.Incoming);
                    sum += double3(record.Flux) * double3(scattering.Value) * weight;
                }
            }
    if (!PhotonFinite(sum.x) || !PhotonFinite(sum.y) || !PhotonFinite(sum.z))
        PhotonFail(PhotonNumericalFailure);
    return float3(sum);
}
float3 PhotonVolumeGather(float3 position, float3 cameraDirection, uint materialIndex,
                          uint cameraDepth, inout PhotonStatistics counters)
{
    if (!(GatherRadius > 0)) return 0;
    int3 center;
    if (!PhotonGridCell(position, center)) return 0;
    const double squaredRadius = GatherRadius * GatherRadius;
    // A collision-photon density contains sigma_t. The camera free-flight
    // sample already carries scattering albedo, so divide by extinction.
    const double scale = double(Materials[materialIndex].Optical.w) /
        (PhotonPi * GatherRadius * squaredRadius * PhotonVolumeKernelIntegral * PhotonPaths);
    double3 sum = 0;
    for (int z = -1; z <= 1; ++z)
        for (int y = -1; y <= 1; ++y)
            for (int x = -1; x <= 1; ++x)
            {
                int3 cell = center + int3(x, y, z);
                uint recordIndex = Heads[PhotonBucket(cell.x, cell.y, cell.z, BucketCount)], visited = 0;
                while (recordIndex != PhotonInvalid)
                {
                    if (recordIndex >= PhotonPaths * SlotsPerPath || ++visited > PhotonPaths * SlotsPerPath)
                    {
                        PhotonFail(PhotonNumericalFailure);
                        return 0;
                    }
                    PhotonRecord record = Photons[recordIndex];
                    recordIndex = record.Next;
                    if ((record.Kind & PhotonVolumeFlag) == 0 || record.Reserved != materialIndex ||
                        record.LightDepth > MaxReflectionBounces - cameraDepth)
                        continue;
                    uint kind = record.Kind & PhotonClassMask;
                    if (kind == PhotonCaustic && !(PhotonClasses & PhotonCausticsBit)) continue;
                    if (kind == PhotonIndirect && !(PhotonClasses & PhotonIndirectBit)) continue;
                    int3 storedCell;
                    if (!PhotonGridCell(record.Position, storedCell) || any(storedCell != cell))
                        continue;
                    double3 difference = double3(record.Position) - double3(position);
                    double squaredDistance = difference.x * difference.x + difference.y * difference.y +
                                             difference.z * difference.z;
                    if (squaredDistance > squaredRadius) continue;
                    float distance = length(record.Position - position);
                    if (distance > 2 * RayEpsilon)
                    {
                        RayData segment = PhotonRay(position, (record.Position - position) / distance,
                            distance - RayEpsilon);
                        HitData blocker = (HitData)0;
                        blocker.Distance = segment.MaximumDistance;
                        if (IntersectScene(segment, true, blocker, counters.Tests)) continue;
                    }
                    double phase = SubsurfacePhase(dot(record.Incoming, cameraDirection),
                        Materials[materialIndex].ConductorIndex.w);
                    sum += double3(record.Flux) * (scale * (1 - squaredDistance / squaredRadius) * phase);
                }
            }
    if (!PhotonFinite(sum.x) || !PhotonFinite(sum.y) || !PhotonFinite(sum.z))
        PhotonFail(PhotonNumericalFailure);
    return float3(sum);
}
float3 PhotonDirectLighting(SampleContext samples, uint dimension, SurfaceBsdf surface, float3 position,
                            HitData subsurfaceBoundary, bool filterTransmission,
                            inout PhotonStatistics counters)
{
    EmitterEndpoint endpoint = SampleEmitterEndpoint(Emitters, EmitterCount, PhotonDomain(),
                                                     Sample2D(samples, dimension ^ PhotonSourceDomain).x,
                                                     Sample2D(samples, dimension ^ PhotonPointDomain),
                                                     Sample2D(samples, dimension ^ PhotonDirectionDomain));
    EmissionSample light = endpoint.Emission;
    if (!(light.Flags & EmissionValid))
        return 0;
    bool infinite = (light.Flags & EmissionInfinite) != 0,
         deltaPosition = (light.Flags & EmissionDeltaPosition) != 0;
    float3 direction = -light.Direction;
    double geometry = 1;
    if (!infinite)
    {
        float3 segment = light.Origin - position;
        float distance = length(segment);
        if (!(distance > 0) || !isfinite(distance))
            return 0;
        direction = segment / distance;
        geometry = double(deltaPosition ? 1 : abs(dot(light.Normal, direction))) / double(distance) /
                   double(distance);
    }
    bool transmitted = filterTransmission &&
        (subsurfaceBoundary.FrontFace ? dot(direction, subsurfaceBoundary.GeometricNormal) < 0
                                      : dot(direction, subsurfaceBoundary.GeometricNormal) > 0);
    if (transmitted && !deltaPosition && !(light.Flags & EmissionDeltaDirection))
        return 0;
    double density =
        double(endpoint.SelectionProbability) * double(infinite ? light.DirectionPdf : light.PositionPdf);
    if (!(density > 0))
        return 0;
    SurfaceEvaluation scattering = EvaluateSurface(surface, direction);
    float3 emitted = PhotonEmission(Emitters[endpoint.EmitterIndex], -direction);
    float3 value = float3(double3(scattering.Value) * double3(emitted) *
                          (double(abs(dot(surface.Normal, direction))) * geometry / density));
    if (transmitted && subsurfaceBoundary.FrontFace && deltaPosition)
        value *= exp(-length(light.Origin - position) / surface.Material.Optical.w);
    if (!PhotonValidColor(value))
    {
        PhotonFail(PhotonNumericalFailure);
        return 0;
    }
    if (!any(value > 0))
        return 0;
    float3 origin = SurfaceOffset(surface, position, direction);
    RayData shadow = PhotonRay(origin, direction);
    if (!infinite)
    {
        float3 connection = light.Origin - origin;
        float distance = length(connection);
        if (!(distance > 2 * RayEpsilon))
            return 0;
        shadow.Direction = connection / distance;
        shadow.MaximumDistance = distance - RayEpsilon;
    }
    IncrementCounter(counters.Shadow);
    HitData obstruction = (HitData)0;
    obstruction.Distance = shadow.MaximumDistance;
    if (IntersectScene(shadow, true, obstruction, counters.Tests))
        return 0;
    float closest = shadow.MaximumDistance;
    for (uint emitter = 0; emitter < EmitterCount; ++emitter)
        if (Emitters[emitter].Kind == PackedEmitterRectangle)
        {
            IncrementCounter(counters.Tests);
            if (PhotonRectangle(Emitters[emitter], shadow, closest))
                return 0;
        }
    return value;
}
float3 PhotonCameraPath(RayData ray, SampleContext samples, inout PhotonStatistics counters)
{
    float3 radiance = 0, throughput = 1;
    float etaScale = 1;
    uint interiorMaterial = PhotonInvalid;
    for (uint depth = 0; depth <= MaxReflectionBounces + 1; ++depth)
    {
        if (!PhotonValidColor(throughput) || !isfinite(etaScale))
        {
            PhotonFail(PhotonNumericalFailure);
            break;
        }
        if (!any(throughput > 0))
            break;
        if (depth == 0)
            IncrementCounter(counters.Primary);
        else
            IncrementCounter(counters.Reflection);
        HitData hit;
        uint source;
        bool found = PhotonTrace(ray, hit, source, counters);
        uint dimension = PhotonCameraDomain ^ (depth * BounceSeedMultiplier);
        if (found && hit.MaterialIndex >= MaterialCount)
        {
            PhotonFail(PhotonInvalidConfiguration);
            break;
        }
        if (depth == 0 && found && !hit.FrontFace && Materials[hit.MaterialIndex].Optical.w > 0)
            interiorMaterial = hit.MaterialIndex;
        if (interiorMaterial != PhotonInvalid)
        {
            PackedMaterial interior = Materials[interiorMaterial];
            float boundary = found || (source != PhotonInvalid &&
                Emitters[source].Kind == PackedEmitterRectangle) ? hit.Distance : ray.MaximumDistance;
            SubsurfaceFlight flight = SubsurfaceFreeFlight(interior.Optical.w,
                interior.AlbedoReflectivity.xyz * interior.ConductorExtinction.w,
                ray.Direction, boundary, Sample2D(samples, dimension ^ PhotonLobeDomain).x);
            throughput *= flight.Weight;
            if (!PhotonValidColor(throughput))
            {
                PhotonFail(PhotonNumericalFailure);
                break;
            }
            if (!any(throughput > 0)) break;
            if (flight.Scattered)
            {
                if (depth > MaxReflectionBounces) break;
                float3 position = ray.Origin + flight.Distance * ray.Direction;
                radiance += throughput * PhotonVolumeGather(position, ray.Direction, interiorMaterial,
                    depth, counters);
                break;
            }
        }
        if (source != PhotonInvalid)
            radiance += throughput * PhotonEmission(Emitters[source], -ray.Direction);
        if (!found || depth > MaxReflectionBounces)
            break;
        float3 position = ray.Origin + hit.Distance * ray.Direction;
        PackedMaterial material = Materials[hit.MaterialIndex];
        bool legacy = material.Model == MaterialLegacy;
        float mirror = legacy ? saturate(material.AlbedoReflectivity.w) : 0;
        if (legacy)
            material.Model = MaterialDiffuse;
        SurfaceBsdf surface = MakeSurfaceBsdf(material, hit, -ray.Direction, PhotonSurfaceColor(hit));
        bool subsurfaceTransmission = material.Optical.w > 0 && !SurfaceDelta(surface);
        if (!SurfaceDelta(surface) && mirror < 1)
        {
            float3 gathered = PhotonGather(position, hit, surface, depth);
            if (TransportFlags & NeeBit)
                gathered += PhotonDirectLighting(samples, dimension, surface, position,
                    hit, subsurfaceTransmission, counters);
            radiance += throughput * gathered * (1 - mirror);
        }
        SurfaceSample sample;
        if (legacy && mirror > 0)
        {
            sample = PhotonMirror(surface);
            sample.Weight *= mirror;
        }
        else if (SurfaceDelta(surface) || subsurfaceTransmission)
            sample = SampleSurface(surface, Sample2D(samples, dimension ^ PhotonScatterDomain),
                                   Sample2D(samples, dimension ^ PhotonLobeDomain).x);
        else
            break;
        if (!sample.Event || !(sample.Pdf > 0))
            break;
        if (subsurfaceTransmission && !(sample.Event & ScatteringTransmission))
            break;
        throughput *= sample.Weight;
        etaScale *= sample.RelativeIndex * sample.RelativeIndex;
        if (!PhotonValidColor(throughput) || !isfinite(etaScale))
        {
            PhotonFail(PhotonNumericalFailure);
            break;
        }
        if ((TransportFlags & RouletteBit) && depth + 1 >= RouletteStartDepth)
        {
            float survival = RouletteProbability(throughput * etaScale);
            if (Sample2D(samples, dimension ^ PhotonRouletteDomain).x >= survival)
                break;
            throughput /= survival;
        }
        ray = PhotonRay(SurfaceOffset(surface, position, sample.Direction), sample.Direction);
        if (material.Optical.w > 0 && (sample.Event & ScatteringTransmission) != 0)
            interiorMaterial = hit.FrontFace ? hit.MaterialIndex : PhotonInvalid;
    }
    return radiance;
}
[numthreads(PhotonThreads, 1, 1)]
void PhotonCameraKernel(uint3 thread : SV_DispatchThreadID)
{
    uint lane = thread.x;
    if (lane >= WorkCount)
        return;
    PhotonStatistics counters = (PhotonStatistics)0;
    Statistics[lane] = counters;
    if (!PhotonConfiguration())
    {
        PhotonFail(PhotonInvalidConfiguration);
        return;
    }
    uint pixel = WorkBegin + lane;
    if (pixel >= Width * Height)
        return;
    uint x = pixel % Width + TileX, y = pixel / Width + TileY;
    SampleContext samples = MakeSampleContext(x, y, FullWidth, CameraSample, TotalSamples, SamplingSequence, SamplingSeed);
    CameraEndpointFrame frame;
    frame.Position = CameraPosition;
    frame.Forward = CameraForward;
    frame.Right = CameraRight;
    frame.Up = CameraUp;
    frame.HalfWidth = HalfWidth;
    frame.HalfHeight = HalfHeight;
    frame.LensRadius = LensRadius;
    frame.FocusDistance = FocusDistance;
    float2 raster = CameraPixelRaster(x, y, FullWidth, FullHeight, Sample2D(samples, PhotonRasterDimension));
    CameraEndpointSample camera = SampleCameraEndpoint(frame, raster, Sample2D(samples, PhotonLensDomain));
    float3 radiance = 0;
    if (camera.Flags & CameraEndpointValid)
        radiance = PhotonCameraPath(PhotonRay(camera.Origin, camera.Direction), samples, counters);
    if (!PhotonValidColor(radiance))
    {
        PhotonFail(PhotonNumericalFailure);
        return;
    }
    float4 moments = Accumulation[pixel];
    uint count = SampleCounts[pixel];
    AccumulateFilmSample(moments, count, radiance);
    if (!all(isfinite(moments)))
    {
        PhotonFail(PhotonNumericalFailure);
        return;
    }
    Accumulation[pixel] = moments;
    SampleCounts[pixel] = count;
    SampleTargets[pixel] = TotalSamples;
    Statistics[lane] = counters;
}
