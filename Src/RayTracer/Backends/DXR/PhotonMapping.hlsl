// Jensen (1996), "Global Illumination using Photon Maps". Independently authored
// DXR photon transport for offline renders. Preview skips photon maps. This library
// emits importance paths, links a spatial hash, then estimates camera radiance.
// The receiver uses a normalized Epanechnikov disk; this fixed-radius estimator
// has bias. Its camera moments cannot measure photon-map uncertainty, so adaptive
// stopping is unavailable. PPM and VCM require their own transport algorithms.
// https://pbr-book.org/3ed-2018/Light_Transport_III_Bidirectional_Methods/Stochastic_Progressive_Photon_Mapping
// Interior collisions use a separate normalized three-dimensional photon kernel.
static const uint MaximumPhotonReflections = 32, InvalidPhoton = 0xffffffffu;
static const uint EmptyArrival = 0, DirectArrival = 1, CausticArrival = 2, IndirectArrival = 3;
static const uint VolumeArrivalFlag = 0x80000000u, ArrivalClassMask = ~VolumeArrivalFlag;
static const uint CausticsEnabled = 1, IndirectEnabled = 2;
static const uint ClearStage = 0, EmissionStage = 1, IndexStage = 2, CameraStage = 3;
static const uint ScalarBytes = 4, PhotonStride = 64, PositionByte = 0, LinkByte = 12;
static const uint IncomingByte = 16, DepthByte = 28, FluxByte = 32, ClassByte = 44;
static const uint NormalByte = 48, ReservedByte = 60;
static const double DensityPi = 3.1415926535897932384626433832795L;
static const double LargestDouble = 1.7976931348623157e308L, SafeCellBound = 1073741824.0L;
static const float NormalAgreement = 0.9f, PlaneThicknessRatio = 0.1f;
static const double VolumeKernelIntegral = 8.0 / 15.0;
bool FiniteDensity(double value)
{
    return value >= -LargestDouble && value <= LargestDouble;
}
bool ValidCellCoordinate(double value)
{
    return value > -SafeCellBound && value < SafeCellBound;
}
int CellFloor(double value)
{
    int integral = int(value);
    return integral - (double(integral) > value ? 1 : 0);
}
uint MixPhotonWord(uint value)
{
    static const uint FirstMix = 0x85ebca6bu, SecondMix = 0xc2b2ae35u;
    static const uint HalfWordShift = 16, MiddleShift = 13;
    value ^= value >> HalfWordShift;
    value *= FirstMix;
    value ^= value >> MiddleShift;
    value *= SecondMix;
    return value ^ (value >> HalfWordShift);
}
uint SpatialBucket(int x, int y, int z, uint count)
{
    static const uint XDomain = 0x243f6a88u, YDomain = 0x85a308d3u, ZDomain = 0x13198a2eu;
    return MixPhotonWord(MixPhotonWord(uint(x) ^ XDomain) ^ MixPhotonWord(uint(y) ^ YDomain) ^
                         MixPhotonWord(uint(z) ^ ZDomain)) &
           (count - 1);
}
uint ArrivalClass(uint depth, bool deltaHistory)
{
    if (depth == 0)
        return DirectArrival;
    return deltaHistory ? CausticArrival : IndirectArrival;
}
bool AcceptArrival(uint kind, uint lightDepth, uint cameraDepth, uint depthLimit, uint enabled, bool nee)
{
    if (cameraDepth > depthLimit || lightDepth > depthLimit - cameraDepth)
        return false;
    if (kind == DirectArrival)
        return !nee;
    if (kind == CausticArrival)
        return (enabled & CausticsEnabled) != 0;
    return kind == IndirectArrival && (enabled & IndirectEnabled) != 0;
}
double DensityWeight(double distanceSquared, double radius, uint emittedCount, float normalAgreement,
                     double normalSeparation, float shadingProjection, float geometryProjection)
{
    if (!(radius > 0) || emittedCount == 0 || !(geometryProjection > 0) ||
        normalAgreement < NormalAgreement || normalSeparation > double(PlaneThicknessRatio) * radius)
        return 0;
    double radial = distanceSquared / radius / radius;
    if (!(radial >= 0 && radial < 1))
        return 0;
    return (2 * (1 - radial) / DensityPi / radius / radius / double(emittedCount)) *
           double(shadingProjection) / double(geometryProjection);
}
uint PhotonAddress(uint index)
{
    return index * PhotonStride;
}

static const float Pi = 3.14159265358979323846f, RayEpsilon = 1.0e-4f;
static const float MaximumRayDistance = 1.0e30f, BasisParallelThreshold = 0.999f;
static const uint ImportanceBit = 1, NeeBit = 2, RouletteBit = 4, MisShift = 3, MisMask = 3;
static const uint ArithmeticFailure = 1, ConfigurationFailure = 2;
static const uint PixelSeedMultiplier = 9781, SampleSeedMultiplier = 6271,
                  SecondDimensionScramble = 0x68bc21ebu;
static const uint LightStream = 0x44585250u, CameraStream = 0x43414d45u, VertexStream = 0x9e3779b9u;
static const uint LensStream = 0x4c454e53u, SourceStream = 0x534f5552u, PositionStream = 0x504f534eu;
static const uint DirectionStream = 0x44495245u, BsdfStream = 0x42534446u, LobeStream = 0x4c4f4245u;
static const uint RouletteStream = 0x52524f55u, RasterStream = 0;
static const uint SourceDimension = 0, PositionDimension = 1, DirectionDimension = 3;
static const uint SourceDimensions = 5, ScatterDimensions = 4, LobeOffset = 2, RouletteOffset = 3;
#include "Surface.hlsl"
#include "MaterialTypes.hlsl"
#include "Globals.hlsl"
#include "InstancePrograms.hlsli"
#include "Triangle.hlsl"
#include "Sphere.hlsl"
#include "TriangleIntersection.hlsl"
#include "MotionTriangle.hlsl"
#include "Select.hlsl"
#include "FilmMoments.hlsl"
#include "EmitterData.hlsl"
struct ImageRecord
{
    uint OffsetTexels, Width, Height, Reserved;
};
struct IncidentRecord
{
    float3 Position, Incoming, Flux, Normal;
    uint Next, Depth, Kind, MaterialIndex;
};
struct TransportCounts
{
    uint2 Primary, Shadow, Reflection;
};
struct Arrival
{
    Payload Hit;
    float3 Position;
    uint Emitter;
    bool Surface;
};
StructuredBuffer<MaterialData> Materials : register(t3);
StructuredBuffer<ImageRecord> Textures : register(t5);
StructuredBuffer<uint> TextureTexels : register(t6);
StructuredBuffer<PackedEmitter> Emitters : register(t13);
RWStructuredBuffer<uint> OutputPixels : register(u0);
RWStructuredBuffer<uint> Statistics : register(u1);
RWStructuredBuffer<LinearFilmPixel> LinearFilm : register(u2);
RWByteAddressBuffer PhotonRecords : register(u5);
RWByteAddressBuffer BucketHeads : register(u6);
RWByteAddressBuffer PhotonErrors : register(u7);
cbuffer PhotonBatch : register(b1)
{
    float3 EmissionCenter;
    float WorldRadius;
    double MergeRadius;
    uint EmitterCount, EmittedPaths;
    uint SlotsPerPath, BucketCount, PopulationSeed, EnabledClasses;
    uint WorkBegin, WorkCount, Stage, CameraSample;
};
#include "Statistics.hlsl"
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
uint Hash(uint value)
{
    return MixPhotonWord(value);
}
float Random01(uint counter)
{
    static const uint FractionBits = 24, WordBits = 32;
    static const float FractionRange = 16777216.0f;
    return float(Hash(counter) >> (WordBits - FractionBits)) / FractionRange;
}
#include "Sequence.hlsl"
float LightRandom(uint path, uint dimension)
{
    return Random01(Hash(PopulationSeed ^ LightStream) ^ Hash(path + VertexStream) ^ Hash(dimension));
}
float2 LightPair(uint path, uint dimension)
{
    return float2(LightRandom(path, dimension), LightRandom(path, dimension + 1));
}
void FailPhoton(uint flags = ArithmeticFailure)
{
    uint ignored;
    PhotonErrors.InterlockedOr(0, flags, ignored);
}
bool ValidSpectrum(float3 value)
{
    return all(isfinite(value)) && all(value >= 0);
}
void CommitCounts(TransportCounts value)
{
    AddStatistic(PrimaryCounterWord, value.Primary);
    AddStatistic(ShadowCounterWord, value.Shadow);
    AddStatistic(ReflectionCounterWord, value.Reflection);
}
bool ValidPhotonControls()
{
    return MaxReflectionBounces <= MaximumPhotonReflections && SlotsPerPath == MaxReflectionBounces + 1 &&
           EmittedPaths > 0 && BucketCount > 0 && (BucketCount & (BucketCount - 1)) == 0 &&
           !AdaptiveEnabled && ((TransportFlags >> MisShift) & MisMask) == 0 && ShadowStrength == 1 &&
           MergeRadius >= 0 && FiniteDensity(MergeRadius);
}
EmissionDomain LightDomain()
{
    EmissionDomain domain;
    domain.Center = EmissionCenter;
    domain.Radius = WorldRadius;
    return domain;
}
void StoreIncident(uint index, IncidentRecord record)
{
    uint address = PhotonAddress(index);
    PhotonRecords.Store3(address + PositionByte, asuint(record.Position));
    PhotonRecords.Store(address + LinkByte, record.Next);
    PhotonRecords.Store3(address + IncomingByte, asuint(record.Incoming));
    PhotonRecords.Store(address + DepthByte, record.Depth);
    PhotonRecords.Store3(address + FluxByte, asuint(record.Flux));
    PhotonRecords.Store(address + ClassByte, record.Kind);
    PhotonRecords.Store3(address + NormalByte, asuint(record.Normal));
    PhotonRecords.Store(address + ReservedByte, record.MaterialIndex);
}
IncidentRecord LoadIncident(uint index)
{
    uint address = PhotonAddress(index);
    IncidentRecord record;
    record.Position = asfloat(PhotonRecords.Load3(address + PositionByte));
    record.Next = PhotonRecords.Load(address + LinkByte);
    record.Incoming = asfloat(PhotonRecords.Load3(address + IncomingByte));
    record.Depth = PhotonRecords.Load(address + DepthByte);
    record.Flux = asfloat(PhotonRecords.Load3(address + FluxByte));
    record.Kind = PhotonRecords.Load(address + ClassByte);
    record.Normal = asfloat(PhotonRecords.Load3(address + NormalByte));
    record.MaterialIndex = PhotonRecords.Load(address + ReservedByte);
    return record;
}
bool IncidentCell(float3 position, out int3 cell)
{
    cell = 0;
    if (!(MergeRadius > 0))
        return false;
    double3 scaled = (double3(position) - double3(EmissionCenter)) / MergeRadius;
    if (!ValidCellCoordinate(scaled.x) || !ValidCellCoordinate(scaled.y) || !ValidCellCoordinate(scaled.z))
    {
        FailPhoton();
        return false;
    }
    cell = int3(CellFloor(scaled.x), CellFloor(scaled.y), CellFloor(scaled.z));
    return true;
}
float DecodeColor(uint byteValue)
{
    static const float MaximumByte = 255, LinearEnd = 0.04045f, LinearSlope = 12.92f;
    static const float Offset = 0.055f, Scale = 1.055f, Exponent = 2.4f;
    float encoded = byteValue / MaximumByte;
    return encoded <= LinearEnd ? encoded / LinearSlope : pow((encoded + Offset) / Scale, Exponent);
}
float3 ImageTexel(ImageRecord image, int2 pixel)
{
    int2 extent = int2(image.Width, image.Height);
    pixel = (pixel % extent + extent) % extent;
    uint value = TextureTexels[image.OffsetTexels + pixel.y * image.Width + pixel.x];
    static const uint ByteMask = 0xff, GreenShift = 8, BlueShift = 16;
    return float3(DecodeColor(value & ByteMask), DecodeColor((value >> GreenShift) & ByteMask),
                  DecodeColor((value >> BlueShift) & ByteMask));
}
float3 MaterialColor(Payload hit)
{
    uint textureIndex = Materials[hit.MaterialIndex].DiffuseTextureIndex;
    if (!(hit.Flags & SurfaceTexCoordFlag) || textureIndex >= TextureCount || !all(isfinite(hit.TexCoord)))
        return hit.Color;
    ImageRecord image = Textures[textureIndex];
    if (!image.Width || !image.Height || image.OffsetTexels > TextureTexelCount ||
        image.Height > (TextureTexelCount - image.OffsetTexels) / image.Width)
        return hit.Color;
    float2 coordinate =
        frac(float2(hit.TexCoord.x, 1 - hit.TexCoord.y)) * float2(image.Width, image.Height) - 0.5f;
    int2 base = int2(floor(coordinate));
    float2 fraction = frac(coordinate);
    return hit.Color *
           lerp(lerp(ImageTexel(image, base), ImageTexel(image, base + int2(1, 0)), fraction.x),
                lerp(ImageTexel(image, base + int2(0, 1)), ImageTexel(image, base + int2(1, 1)), fraction.x),
                fraction.y);
}
float3 EmittedRadiance(PackedEmitter light, float3 outward)
{
    if (light.Kind == PackedEmitterEnvironment)
        return lerp(light.BackgroundBottom, light.BackgroundTop, (1 - outward.y) / 2);
    if (light.Kind == PackedEmitterRectangle)
        return dot(light.Direction, outward) > 0 ? light.Strength / (light.Width * light.Height) : 0;
    return light.Strength;
}
bool RectangleHit(PackedEmitter light, float3 origin, float3 direction, float limit, inout float distance)
{
    if (light.Kind != PackedEmitterRectangle)
        return false;
    float denominator = dot(direction, light.Direction);
    if (denominator == 0)
        return false;
    float t = dot(light.Origin - origin, light.Direction) / denominator;
    if (!(t > RayEpsilon && t < limit && t <= distance))
        return false;
    float3 axis = abs(light.Direction.y) < BasisParallelThreshold ? float3(0, 1, 0) : float3(1, 0, 0);
    float3 right = normalize(cross(axis, light.Direction)), up = cross(light.Direction, right);
    float3 relative = origin + t * direction - light.Origin;
    if (abs(dot(relative, right)) > light.Width / 2 || abs(dot(relative, up)) > light.Height / 2)
        return false;
    distance = t;
    return true;
}
Arrival NextArrival(float3 origin, float3 direction)
{
    Arrival arrival;
    arrival.Hit = CastRay(origin, direction, RayEpsilon, MaximumRayDistance);
    arrival.Surface = (arrival.Hit.Flags & SurfaceHitFlag) != 0;
    arrival.Emitter = InvalidPhoton;
    const float geometryDistance = asfloat(arrival.Hit.DistanceBits);
    // Let an analytic emitter win its own display proxy within the ray offset,
    // without extending the distance of an unrelated geometric occluder.
    const bool proxy = arrival.Surface && arrival.Hit.MaterialIndex < MaterialCount &&
                       Materials[arrival.Hit.MaterialIndex].SampledEmitter != 0;
    float closest = geometryDistance + (proxy ? RayEpsilon : 0);
    for (uint index = 0; index < EmitterCount; ++index)
        if (RectangleHit(Emitters[index], origin, direction, MaximumRayDistance, closest))
        {
            arrival.Surface = false;
            arrival.Emitter = index;
        }
    if (arrival.Emitter == InvalidPhoton)
        for (uint index = 0; index < EmitterCount; ++index)
        {
            PackedEmitter source = Emitters[index];
            if (arrival.Surface ? source.Primitive != InvalidPrimitiveIndex &&
                                      source.Primitive == arrival.Hit.PrimitiveIndex
                                : source.Kind == PackedEmitterEnvironment)
            {
                arrival.Emitter = index;
                break;
            }
        }
    arrival.Position = origin + (arrival.Surface ? geometryDistance : closest) * direction;
    return arrival;
}
SurfaceSample LegacyReflection(SurfaceBsdf surface)
{
    SurfaceSample result = (SurfaceSample)0;
    result.Direction = SurfaceWorld(surface, float3(-surface.View.xy, surface.View.z));
    if (!SurfaceDirectionValid(surface, result.Direction, surface.View.z))
        return (SurfaceSample)0;
    result.Weight = SurfaceAdjointCorrection(surface, result.Direction).xxx;
    result.Pdf = result.ReversePdf = result.RelativeIndex = 1;
    result.Event = ScatteringReflection | ScatteringDelta;
    return result;
}

void EmissionWalk(uint path)
{
    uint base = path * SlotsPerPath;
    for (uint slot = 0; slot < SlotsPerPath; ++slot)
        PhotonRecords.Store(PhotonAddress(base + slot) + ClassByte, EmptyArrival);
    EmitterEndpoint endpoint =
        SampleEmitterEndpoint(Emitters, EmitterCount, LightDomain(), LightRandom(path, SourceDimension),
                              LightPair(path, PositionDimension), LightPair(path, DirectionDimension));
    EmissionSample launch = endpoint.Emission;
    if (!(launch.Flags & EmissionValid))
        return;
    float3 direction = launch.Direction;
    float3 origin =
        launch.Origin + launch.Normal * (dot(launch.Normal, direction) >= 0 ? RayEpsilon : -RayEpsilon);
    float3 flux = launch.Weight;
    bool deltaHistory = true;
    uint interiorMaterial = InvalidPhoton;
    TransportCounts counts = (TransportCounts)0;
    for (uint depth = 0; depth <= MaxReflectionBounces; ++depth)
    {
        if (!ValidSpectrum(flux))
        {
            FailPhoton();
            break;
        }
        if (!any(flux > 0))
            break;
        IncrementLocalCounter(counts.Reflection);
        Arrival arrival = NextArrival(origin, direction);
        uint dimension = SourceDimensions + ScatterDimensions * depth;
        if (arrival.Surface && arrival.Hit.MaterialIndex >= MaterialCount)
        {
            FailPhoton(ConfigurationFailure);
            break;
        }
        if (depth == 0 && arrival.Surface && !(arrival.Hit.Flags & SurfaceFrontFaceFlag) &&
            Materials[arrival.Hit.MaterialIndex].Optical.w > 0)
            interiorMaterial = arrival.Hit.MaterialIndex;
        if (interiorMaterial != InvalidPhoton)
        {
            MaterialData interior = Materials[interiorMaterial];
            bool finiteSource = arrival.Emitter != InvalidPhoton &&
                Emitters[arrival.Emitter].Kind == PackedEmitterRectangle;
            float boundary = arrival.Surface || finiteSource
                ? length(arrival.Position - origin) : MaximumRayDistance;
            SubsurfaceFlight flight = SubsurfaceFreeFlight(interior.Optical.w,
                interior.AlbedoReflectivity.xyz * interior.ConductorExtinction.w,
                direction, boundary, LightRandom(path, dimension + LobeOffset));
            if (flight.Scattered)
            {
                IncidentRecord photon = (IncidentRecord)0;
                photon.Position = origin + flight.Distance * direction;
                photon.Incoming = -direction;
                photon.Flux = flux;
                photon.Next = InvalidPhoton;
                photon.Depth = depth;
                photon.Kind = ArrivalClass(depth, deltaHistory) | VolumeArrivalFlag;
                photon.MaterialIndex = interiorMaterial;
                StoreIncident(base + depth, photon);
            }
            flux *= flight.Weight;
            if (!ValidSpectrum(flux))
            {
                FailPhoton();
                break;
            }
            if (!any(flux > 0)) break;
            if (flight.Scattered)
            {
                if (depth == MaxReflectionBounces) break;
                SubsurfaceDirection phase = SubsurfaceSampleDirection(direction,
                    interior.ConductorIndex.w, LightPair(path, dimension),
                    (TransportFlags & ImportanceBit) != 0);
                if (!(phase.Pdf > 0) || !isfinite(phase.Weight))
                {
                    FailPhoton();
                    break;
                }
                flux *= phase.Weight;
                if ((TransportFlags & RouletteBit) && depth + 1 >= RouletteStartDepth)
                {
                    float survival = RouletteProbability(flux);
                    if (LightRandom(path, dimension + RouletteOffset) >= survival) break;
                    flux /= survival;
                }
                deltaHistory = false;
                origin += flight.Distance * direction;
                direction = phase.Direction;
                continue;
            }
        }
        if (!arrival.Surface) break;
        MaterialData material = Materials[arrival.Hit.MaterialIndex];
        bool legacy = material.Model == MaterialLegacy;
        float mirror = legacy ? saturate(material.AlbedoReflectivity.w) : 0;
        if (legacy)
            material.Model = MaterialDiffuse;
        SurfaceBsdf surface = MakeSurfaceBsdf(material, arrival.Hit, -direction, MaterialColor(arrival.Hit),
                                              TransportImportance);
        // Store incident flux before selecting the receiver's outgoing event.
        // The camera evaluates that event once; source misses remain in N.
        if (!SurfaceDelta(surface) && mirror < 1)
        {
            IncidentRecord photon;
            photon.Position = arrival.Position;
            photon.Incoming = -direction;
            photon.Flux = flux;
            photon.Normal = arrival.Hit.GeometricNormal;
            photon.Next = InvalidPhoton;
            photon.Depth = depth;
            photon.Kind = ArrivalClass(depth, deltaHistory);
            photon.MaterialIndex = arrival.Hit.MaterialIndex;
            StoreIncident(base + depth, photon);
        }
        if (depth == MaxReflectionBounces)
            break;
        float eventU = LightRandom(path, dimension + LobeOffset);
        SurfaceSample event;
        if (legacy && eventU < mirror)
            event = LegacyReflection(surface);
        else
            event = SampleSurface(surface, LightPair(path, dimension), eventU);
        if (!event.Event || !(event.Pdf > 0))
            break;
        flux *= event.Weight;
        deltaHistory = deltaHistory && (event.Event & ScatteringDelta) != 0;
        if (!ValidSpectrum(flux))
        {
            FailPhoton();
            break;
        }
        if ((TransportFlags & RouletteBit) && depth + 1 >= RouletteStartDepth)
        {
            float survival = RouletteProbability(flux);
            if (LightRandom(path, dimension + RouletteOffset) >= survival)
                break;
            flux /= survival;
        }
        direction = event.Direction;
        origin = SurfaceOffset(surface, arrival.Position, direction);
        if (material.Optical.w > 0 && (event.Event & ScatteringTransmission) != 0)
            interiorMaterial = (arrival.Hit.Flags & SurfaceFrontFaceFlag) != 0
                ? arrival.Hit.MaterialIndex : InvalidPhoton;
    }
    CommitCounts(counts);
}
void InsertIncident(uint index)
{
    uint address = PhotonAddress(index);
    if (PhotonRecords.Load(address + ClassByte) == EmptyArrival)
        return;
    int3 cell;
    if (!IncidentCell(asfloat(PhotonRecords.Load3(address + PositionByte)), cell))
        return;
    uint previous;
    BucketHeads.InterlockedExchange(ScalarBytes * SpatialBucket(cell.x, cell.y, cell.z, BucketCount), index,
                                    previous);
    PhotonRecords.Store(address + LinkByte, previous);
    // The camera stage runs only after a dispatch fence/UAV barrier publishes
    // both the atomic bucket heads and the non-atomic links. No spin waits.
}
float3 SurfaceDensity(Arrival arrival, SurfaceBsdf surface, uint cameraDepth)
{
    int3 query;
    if (!IncidentCell(arrival.Position, query))
        return 0;
    double3 value = 0;
    for (int z = -1; z <= 1; ++z)
        for (int y = -1; y <= 1; ++y)
            for (int x = -1; x <= 1; ++x)
            {
                int3 neighbor = query + int3(x, y, z);
                uint index = BucketHeads.Load(ScalarBytes *
                                              SpatialBucket(neighbor.x, neighbor.y, neighbor.z, BucketCount));
                uint visited = 0;
                while (index != InvalidPhoton)
                {
                    if (index >= EmittedPaths * SlotsPerPath || ++visited > EmittedPaths * SlotsPerPath)
                    {
                        FailPhoton();
                        return 0;
                    }
                    IncidentRecord photon = LoadIncident(index);
                    index = photon.Next;
                    if ((photon.Kind & VolumeArrivalFlag) != 0 ||
                        !AcceptArrival(photon.Kind & ArrivalClassMask, photon.Depth, cameraDepth, MaxReflectionBounces,
                                       EnabledClasses, (TransportFlags & NeeBit) != 0))
                        continue;
                    if (surface.Material.Optical.w > 0 &&
                        ((arrival.Hit.Flags & SurfaceFrontFaceFlag) != 0
                            ? dot(photon.Incoming, arrival.Hit.GeometricNormal) <= 0
                            : dot(photon.Incoming, arrival.Hit.GeometricNormal) >= 0))
                        continue;
                    int3 cell;
                    if (!IncidentCell(photon.Position, cell) || any(cell != neighbor))
                        continue;
                    // Exact cell matching prevents duplicate energy when neighbor buckets
                    // collide. Surface proximity filters limit leakage at finite radius.
                    double3 offset = double3(photon.Position) - double3(arrival.Position);
                    double squared = offset.x * offset.x + offset.y * offset.y + offset.z * offset.z;
                    float3 normal = arrival.Hit.GeometricNormal;
                    double plane = abs(offset.x * double(normal.x) + offset.y * double(normal.y) +
                                       offset.z * double(normal.z));
                    double weight = DensityWeight(
                        squared, MergeRadius, EmittedPaths, dot(normal, photon.Normal), plane,
                        abs(dot(surface.Normal, photon.Incoming)), abs(dot(normal, photon.Incoming)));
                    if (!(weight > 0))
                        continue;
                    SurfaceEvaluation receiver = EvaluateSurface(surface, photon.Incoming);
                    value += double3(photon.Flux) * double3(receiver.Value) * weight;
                }
            }
    if (!FiniteDensity(value.x) || !FiniteDensity(value.y) || !FiniteDensity(value.z))
        FailPhoton();
    return float3(value);
}
float3 VolumeDensity(float3 position, float3 cameraDirection, uint materialIndex, uint cameraDepth,
                     inout TransportCounts counts)
{
    if (!(MergeRadius > 0)) return 0;
    int3 query;
    if (!IncidentCell(position, query)) return 0;
    const double squaredRadius = MergeRadius * MergeRadius;
    // Collision-photon density carries extinction. The camera free-flight
    // event already applies scattering albedo, so divide by sigma_t here.
    const double scale = double(Materials[materialIndex].Optical.w) /
        (DensityPi * MergeRadius * squaredRadius * VolumeKernelIntegral * EmittedPaths);
    double3 value = 0;
    for (int z = -1; z <= 1; ++z)
        for (int y = -1; y <= 1; ++y)
            for (int x = -1; x <= 1; ++x)
            {
                int3 neighbor = query + int3(x, y, z);
                uint index = BucketHeads.Load(ScalarBytes *
                    SpatialBucket(neighbor.x, neighbor.y, neighbor.z, BucketCount));
                uint visited = 0;
                while (index != InvalidPhoton)
                {
                    if (index >= EmittedPaths * SlotsPerPath || ++visited > EmittedPaths * SlotsPerPath)
                    {
                        FailPhoton();
                        return 0;
                    }
                    IncidentRecord photon = LoadIncident(index);
                    index = photon.Next;
                    if ((photon.Kind & VolumeArrivalFlag) == 0 || photon.MaterialIndex != materialIndex ||
                        cameraDepth > MaxReflectionBounces ||
                        photon.Depth > MaxReflectionBounces - cameraDepth)
                        continue;
                    uint kind = photon.Kind & ArrivalClassMask;
                    if (kind == CausticArrival && !(EnabledClasses & CausticsEnabled)) continue;
                    if (kind == IndirectArrival && !(EnabledClasses & IndirectEnabled)) continue;
                    int3 cell;
                    if (!IncidentCell(photon.Position, cell) || any(cell != neighbor)) continue;
                    double3 difference = double3(photon.Position) - double3(position);
                    double squared = difference.x * difference.x + difference.y * difference.y +
                                     difference.z * difference.z;
                    if (squared > squaredRadius) continue;
                    float distance = length(photon.Position - position);
                    if (distance > 2 * RayEpsilon)
                    {
                        IncrementLocalCounter(counts.Shadow);
                        Payload blocker = CastRay(position,
                            (photon.Position - position) / distance,
                            RayEpsilon, distance - RayEpsilon);
                        if (blocker.Flags & SurfaceHitFlag) continue;
                    }
                    double phase = SubsurfacePhase(dot(photon.Incoming, cameraDirection),
                        Materials[materialIndex].ConductorIndex.w);
                    value += double3(photon.Flux) * (scale * (1 - squared / squaredRadius) * phase);
                }
            }
    if (!FiniteDensity(value.x) || !FiniteDensity(value.y) || !FiniteDensity(value.z))
        FailPhoton();
    return float3(value);
}
float3 SampledDirect(Arrival arrival, SurfaceBsdf surface, SampleContext random, uint domain,
                     bool filterTransmission, inout TransportCounts counts)
{
    EmitterEndpoint endpoint = SampleEmitterEndpoint(
        Emitters, EmitterCount, LightDomain(), Sample2D(random, domain ^ SourceStream).x,
        Sample2D(random, domain ^ PositionStream), Sample2D(random, domain ^ DirectionStream));
    EmissionSample sample = endpoint.Emission;
    if (!(sample.Flags & EmissionValid))
        return 0;
    bool infinite = (sample.Flags & EmissionInfinite) != 0,
         deltaPosition = (sample.Flags & EmissionDeltaPosition) != 0;
    float3 incoming = -sample.Direction;
    double geometry = 1;
    if (!infinite)
    {
        float3 segment = sample.Origin - arrival.Position;
        float distance = length(segment);
        if (!(distance > 0) || !isfinite(distance))
            return 0;
        incoming = segment / distance;
        geometry = double(deltaPosition ? 1 : abs(dot(sample.Normal, incoming))) / double(distance) /
                   double(distance);
    }
    bool transmitted = filterTransmission &&
        ((arrival.Hit.Flags & SurfaceFrontFaceFlag) != 0
            ? dot(incoming, arrival.Hit.GeometricNormal) < 0
            : dot(incoming, arrival.Hit.GeometricNormal) > 0);
    if (transmitted && !deltaPosition && !(sample.Flags & EmissionDeltaDirection)) return 0;
    double density =
        double(endpoint.SelectionProbability) * double(infinite ? sample.DirectionPdf : sample.PositionPdf);
    if (!(density > 0))
        return 0;
    SurfaceEvaluation scatter = EvaluateSurface(surface, incoming);
    float3 result =
        float3(double3(scatter.Value) * double3(EmittedRadiance(Emitters[endpoint.EmitterIndex], -incoming)) *
               (double(abs(dot(surface.Normal, incoming))) * geometry / density));
    if (transmitted && (arrival.Hit.Flags & SurfaceFrontFaceFlag) != 0 && deltaPosition)
        result *= exp(-length(sample.Origin - arrival.Position) / surface.Material.Optical.w);
    if (!ValidSpectrum(result))
    {
        FailPhoton();
        return 0;
    }
    if (!any(result > 0))
        return 0;
    float3 start = SurfaceOffset(surface, arrival.Position, incoming);
    float maximum = MaximumRayDistance;
    if (!infinite)
    {
        float3 segment = sample.Origin - start;
        float distance = length(segment);
        if (!(distance > 2 * RayEpsilon))
            return 0;
        incoming = segment / distance;
        maximum = distance - RayEpsilon;
    }
    IncrementLocalCounter(counts.Shadow);
    Payload obstruction = CastRay(start, incoming, RayEpsilon, maximum);
    if (obstruction.Flags & SurfaceHitFlag)
        return 0;
    float closest = maximum;
    for (uint emitter = 0; emitter < EmitterCount; ++emitter)
        if (RectangleHit(Emitters[emitter], start, incoming, maximum, closest))
            return 0;
    return result;
}
float3 SensorEstimate(float3 origin, float3 direction, SampleContext random, inout TransportCounts counts)
{
    float3 result = 0, throughput = 1;
    float refractiveScale = 1;
    uint interiorMaterial = InvalidPhoton;
    // At most max+1 scattering vertices, plus one endpoint query for emission.
    for (uint depth = 0; depth <= MaxReflectionBounces + 1; ++depth)
    {
        if (!ValidSpectrum(throughput) || !isfinite(refractiveScale))
        {
            FailPhoton();
            break;
        }
        if (!any(throughput > 0))
            break;
        if (depth == 0)
            IncrementLocalCounter(counts.Primary);
        else
            IncrementLocalCounter(counts.Reflection);
        Arrival arrival = NextArrival(origin, direction);
        uint domain = CameraStream ^ (VertexStream * depth);
        if (arrival.Surface && arrival.Hit.MaterialIndex >= MaterialCount)
        {
            FailPhoton(ConfigurationFailure);
            break;
        }
        if (depth == 0 && arrival.Surface && !(arrival.Hit.Flags & SurfaceFrontFaceFlag) &&
            Materials[arrival.Hit.MaterialIndex].Optical.w > 0)
            interiorMaterial = arrival.Hit.MaterialIndex;
        if (interiorMaterial != InvalidPhoton)
        {
            MaterialData interior = Materials[interiorMaterial];
            bool finiteSource = arrival.Emitter != InvalidPhoton &&
                Emitters[arrival.Emitter].Kind == PackedEmitterRectangle;
            float boundary = arrival.Surface || finiteSource
                ? length(arrival.Position - origin) : MaximumRayDistance;
            SubsurfaceFlight flight = SubsurfaceFreeFlight(interior.Optical.w,
                interior.AlbedoReflectivity.xyz * interior.ConductorExtinction.w,
                direction, boundary, Sample2D(random, domain ^ LobeStream).x);
            throughput *= flight.Weight;
            if (!ValidSpectrum(throughput))
            {
                FailPhoton();
                break;
            }
            if (!any(throughput > 0)) break;
            if (flight.Scattered)
            {
                if (depth > MaxReflectionBounces) break;
                float3 position = origin + flight.Distance * direction;
                result += throughput * VolumeDensity(position, direction, interiorMaterial, depth, counts);
                break;
            }
        }
        if (arrival.Emitter != InvalidPhoton)
            result += throughput * EmittedRadiance(Emitters[arrival.Emitter], -direction);
        if (!arrival.Surface || depth > MaxReflectionBounces)
            break;
        MaterialData material = Materials[arrival.Hit.MaterialIndex];
        bool legacy = material.Model == MaterialLegacy;
        float mirror = legacy ? saturate(material.AlbedoReflectivity.w) : 0;
        if (legacy)
            material.Model = MaterialDiffuse;
        SurfaceBsdf surface = MakeSurfaceBsdf(material, arrival.Hit, -direction, MaterialColor(arrival.Hit));
        bool subsurfaceTransmission = material.Optical.w > 0 && !SurfaceDelta(surface);
        if (!SurfaceDelta(surface) && mirror < 1)
        {
            float3 illumination = SurfaceDensity(arrival, surface, depth);
            if (TransportFlags & NeeBit)
                illumination += SampledDirect(arrival, surface, random, domain,
                    subsurfaceTransmission, counts);
            result += throughput * illumination * (1 - mirror);
        }
        SurfaceSample event;
        if (legacy && mirror > 0)
        {
            event = LegacyReflection(surface);
            event.Weight *= mirror;
        }
        else if (SurfaceDelta(surface) || subsurfaceTransmission)
            event = SampleSurface(surface, Sample2D(random, domain ^ BsdfStream),
                                  Sample2D(random, domain ^ LobeStream).x);
        else
            break;
        if (!event.Event || !(event.Pdf > 0))
            break;
        if (subsurfaceTransmission && !(event.Event & ScatteringTransmission))
            break;
        throughput *= event.Weight;
        refractiveScale *= event.RelativeIndex * event.RelativeIndex;
        if (!ValidSpectrum(throughput) || !isfinite(refractiveScale))
        {
            FailPhoton();
            break;
        }
        if ((TransportFlags & RouletteBit) && depth + 1 >= RouletteStartDepth)
        {
            float survival = RouletteProbability(throughput * refractiveScale);
            if (Sample2D(random, domain ^ RouletteStream).x >= survival)
                break;
            throughput /= survival;
        }
        direction = event.Direction;
        origin = SurfaceOffset(surface, arrival.Position, direction);
        if (material.Optical.w > 0 && (event.Event & ScatteringTransmission) != 0)
            interiorMaterial = (arrival.Hit.Flags & SurfaceFrontFaceFlag) != 0
                ? arrival.Hit.MaterialIndex : InvalidPhoton;
    }
    return result;
}
uint DisplayPixel(float3 radiance)
{
    static const float LinearEnd = 0.0031308f, LinearScale = 12.92f, Offset = 0.055f, Scale = 1.055f,
                       Exponent = 2.4f;
    static const uint MaximumByte = 255, GreenShift = 8, BlueShift = 16, AlphaShift = 24, ColorChannels = 3;
    float3 color = max(radiance, 0);
    color /= max(1, max(color.x, max(color.y, color.z)));
    if (GammaCorrection)
        for (uint channel = 0; channel < ColorChannels; ++channel)
            color[channel] = color[channel] <= LinearEnd ? color[channel] * LinearScale
                                                         : Scale * pow(color[channel], 1 / Exponent) - Offset;
    uint3 bytes = uint3(saturate(color) * MaximumByte + 0.5f);
    return bytes.x | (bytes.y << GreenShift) | (bytes.z << BlueShift) | (MaximumByte << AlphaShift);
}
void CameraPixel(uint pixel)
{
    uint2 coordinate = uint2(pixel % Width + TileX, pixel / Width + TileY);
    SampleContext random = MakeSampleContext(coordinate.x, coordinate.y, FullWidth, CameraSample,
                                             SamplesPerPixel, SamplingSequence, SamplingSeed);
    CameraEndpointFrame camera;
    camera.Position = CameraPosition;
    camera.Forward = CameraForward;
    camera.Right = CameraRight;
    camera.Up = CameraUp;
    camera.HalfWidth = HalfWidth;
    camera.HalfHeight = HalfHeight;
    camera.LensRadius = LensRadius;
    camera.FocusDistance = FocusDistance;
    float2 raster = CameraPixelRaster(coordinate.x, coordinate.y, FullWidth, FullHeight, Sample2D(random, RasterStream));
    CameraEndpointSample start = SampleCameraEndpoint(camera, raster, Sample2D(random, LensStream));
    TransportCounts counts = (TransportCounts)0;
    float3 value = 0;
    if (start.Flags & CameraEndpointValid)
        value = SensorEstimate(start.Origin, start.Direction, random, counts);
    CommitCounts(counts);
    if (!ValidSpectrum(value))
    {
        FailPhoton();
        return;
    }
    LinearFilmPixel film = LinearFilm[pixel];
    AccumulateFilmSample(film, value);
    if (!ValidSpectrum(film.MeanRadiance) || !isfinite(film.LuminanceM2))
    {
        FailPhoton();
        return;
    }
    LinearFilm[pixel] = film;
    if (film.SampleCount == SamplesPerPixel)
    {
        OutputPixels[pixel] = DisplayPixel(film.MeanRadiance);
        AddStatistic(CompletedCounterWord, uint2(1, 0));
    }
}
[shader("raygeneration")]
void RayGen()
{
    uint lane = DispatchRaysIndex().x;
    if (lane >= WorkCount)
        return;
    uint index = WorkBegin + lane;
    if (Stage == ClearStage)
    {
        if (index < Width * Height)
        {
            LinearFilm[index] = (LinearFilmPixel)0;
            OutputPixels[index] = 0;
        }
        if (index < BucketCount)
            BucketHeads.Store(index * ScalarBytes, InvalidPhoton);
        if (index == 0)
            PhotonErrors.Store(0, 0);
        return;
    }
    if (!ValidPhotonControls())
    {
        FailPhoton(ConfigurationFailure);
        return;
    }
    if (Stage == EmissionStage && index < EmittedPaths)
        EmissionWalk(index);
    else if (Stage == IndexStage && index < EmittedPaths * SlotsPerPath)
        InsertIncident(index);
    else if (Stage == CameraStage && index < Width * Height)
        CameraPixel(index);
}
