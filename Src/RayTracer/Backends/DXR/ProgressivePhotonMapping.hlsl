// Hachisuka and Jensen (2009), "Stochastic Progressive Photon Mapping".
// Independent DXR offline transport; preview does not run progressive light passes.
// Each pass resamples one camera visible point per pixel, gathers bounded light
// batches, then contracts a uniform disk. Persistent flux is normalized by ALL
// attempted light paths. Pass estimates are not independent film samples.
// Interior collision photons use a separate shrinking three-dimensional kernel.
// https://pbr-book.org/3ed-2018/Light_Transport_III_Bidirectional_Methods/Stochastic_Progressive_Photon_Mapping
static const uint PpmMaximumDepth = 32, PpmInvalid = 0xffffffffu;
static const uint PpmEmpty = 0, PpmDirect = 1, PpmCaustic = 2, PpmIndirect = 3;
static const uint PpmVolumeFlag = 0x80000000u, PpmClassMask = ~PpmVolumeFlag;
static const uint PpmVisibleSurface = 1, PpmVisibleVolume = 2;
static const uint PpmCausticsEnabled = 1, PpmIndirectEnabled = 2;
static const uint PpmClear = 0, PpmInitialize = 1, PpmCamera = 2, PpmEmit = 3;
static const uint PpmIndex = 4, PpmGather = 5, PpmUpdate = 6;
static const double PpmPi = 3.1415926535897932384626433832795L;
static const double PpmVolumeMeasure = 4.0L / 3.0L;
static const double PpmRootScale = 1.0e20L, PpmRootScaleSqrt = 1.0e10L;
static const double PpmLargestDouble = 1.7976931348623157e308L, PpmCellLimit = 1073741824.0L;
static const float PpmNormalAgreement = 0.9f, PpmPlaneRatio = 0.1f;
bool PpmFinite(double value)
{
    return value >= -PpmLargestDouble && value <= PpmLargestDouble;
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
float PpmEmitterLimit(float hitDistance, bool sampledProxy, float rayEpsilon)
{
    return hitDistance + (sampledProxy ? rayEpsilon : 0);
}
int PpmFloor(double value)
{
    int result = int(value);
    return result - (double(result) > value ? 1 : 0);
}
bool PpmCellValid(double value)
{
    return value > -PpmCellLimit && value < PpmCellLimit;
}
uint PpmMix(uint word)
{
    static const uint FirstMultiplier = 0x7feb352du, SecondMultiplier = 0x846ca68bu;
    static const uint HalfShift = 16, InteriorShift = 15;
    word ^= word >> HalfShift;
    word *= FirstMultiplier;
    word ^= word >> InteriorShift;
    word *= SecondMultiplier;
    return word ^ (word >> HalfShift);
}
uint PpmBucket(int x, int y, int z, uint buckets)
{
    static const uint AxisX = 0x243f6a88u, AxisY = 0x85a308d3u, AxisZ = 0x13198a2eu;
    return PpmMix(PpmMix(uint(x) ^ AxisX) ^ PpmMix(uint(y) ^ AxisY) ^ PpmMix(uint(z) ^ AxisZ)) &
           (buckets - 1);
}
uint PpmClass(uint depth, bool deltaPrefix)
{
    return depth == 0 ? PpmDirect : deltaPrefix ? PpmCaustic : PpmIndirect;
}
bool PpmAccept(uint kind, uint lightDepth, uint cameraDepth, uint limit, uint classes, bool nee)
{
    bool volume = (kind & PpmVolumeFlag) != 0;
    kind &= PpmClassMask;
    if (cameraDepth > limit || lightDepth > limit - cameraDepth)
        return false;
    if (kind == PpmDirect)
        return volume || !nee;
    if (kind == PpmCaustic)
        return (classes & PpmCausticsEnabled) != 0;
    return kind == PpmIndirect && (classes & PpmIndirectEnabled) != 0;
}
bool PpmMergeable(double distanceSquared, double radiusSquared, double planeDistance, float agreement)
{
    return radiusSquared > 0 && distanceSquared >= 0 && distanceSquared <= radiusSquared &&
           agreement >= PpmNormalAgreement && planeDistance >= 0 &&
           planeDistance * planeDistance <= double(PpmPlaneRatio) * double(PpmPlaneRatio) * radiusSquared;
}
struct PpmRecurrence
{
    double RadiusSquared, Count, Ratio;
};
PpmRecurrence PpmAdvance(double radiusSquared, double count, uint matched, double alpha)
{
    PpmRecurrence next;
    next.Count = count + alpha * double(matched);
    next.Ratio = matched ? next.Count / (count + double(matched)) : 1;
    next.RadiusSquared = radiusSquared * next.Ratio;
    return next;
}
PpmRecurrence PpmAdvanceVolume(double radiusSquared, double count, uint matched, double alpha)
{
    PpmRecurrence next;
    next.Count = count + alpha * double(matched);
    next.Ratio = matched > 0 ? next.Count / (count + double(matched)) : 1;
    double radiusRatio = PpmCubeRoot(next.Ratio);
    next.RadiusSquared = radiusSquared * radiusRatio * radiusRatio;
    return next;
}
double PpmFlux(double flux, double incoming, double ratio)
{
    return (flux + incoming) * ratio;
}
double PpmResolve(double direct, double flux, double radiusSquared, uint passes, uint pathsPerPass)
{
    if (!passes)
        return 0;
    double estimate = direct / double(passes);
    if (radiusSquared > 0 && pathsPerPass)
        estimate += flux / PpmPi / radiusSquared / (double(passes) * double(pathsPerPass));
    return estimate;
}
double PpmResolveVolume(double flux, double radiusSquared, uint passes, uint pathsPerPass)
{
    if (!(radiusSquared > 0) || passes == 0 || pathsPerPass == 0) return 0;
    return flux / (PpmVolumeMeasure * PpmPi * radiusSquared * PpmSquareRoot(radiusSquared) *
                   double(passes) * double(pathsPerPass));
}

static const float Pi = 3.14159265358979323846f, RayEpsilon = 1.0e-4f;
static const float MaximumRayDistance = 1.0e30f, BasisParallelThreshold = 0.999f;
static const uint ImportanceBit = 1, NeeBit = 2, RouletteBit = 4, MisShift = 3, MisMask = 3;
static const uint ArithmeticError = 1, ControlsError = 2;
static const uint PixelSeedMultiplier = 9781, SampleSeedMultiplier = 6271,
                  SecondDimensionScramble = 0x68bc21ebu;
static const uint LightDomainWord = 0x50504d4cu, PassDomainWord = 0x50415353u, VertexDomainWord = 0x9e3779b9u;
static const uint LensDimension = 0x4c454e53u, SourceDimension = 0x534f5552u;
static const uint PositionDimension = 0x504f534eu, DirectionDimension = 0x44495245u;
static const uint ScatterDimension = 0x42534446u, LobeDimension = 0x4c4f4245u,
                  RouletteDimension = 0x52524f55u;
static const uint VisibleDimension = 0x56495349u, InteriorDimension = 0x494e5445u;
static const uint RasterDimension = 0, LightSelectDimension = 0, LightPositionDimension = 1;
static const uint LightDirectionDimension = 3, LightEndpointDimensions = 5, LightVertexDimensions = 4;
static const uint LightLobeOffset = 2, LightRouletteOffset = 3;
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
struct PpmTexture
{
    uint OffsetTexels, Width, Height, Reserved;
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
struct PpmVisible
{
    float3 Position;
    uint Active;
    float3 Normal;
    uint Front;
    float3 GeometricNormal;
    uint Depth;
    float3 View;
    uint Material;
    float3 Color;
    uint Reserved;
    float3 Beta;
    uint Padding;
};
struct PpmPixel
{
    double RadiusSquared, Count;
    double FluxRed, FluxGreen, FluxBlue;
    double DirectRed, DirectGreen, DirectBlue;
    double IncomingRed, IncomingGreen, IncomingBlue;
    uint Matched, Reserved;
    double VolumeRadiusSquared, VolumeCount;
    double VolumeFluxRed, VolumeFluxGreen, VolumeFluxBlue;
    double VolumeIncomingRed, VolumeIncomingGreen, VolumeIncomingBlue;
    uint VolumeMatched;
    uint3 VolumeReserved;
};
struct PpmCounters
{
    uint2 Primary, Shadow, Reflection;
};
struct PpmArrival
{
    Payload Hit;
    float3 Position;
    uint Source;
    bool Surface;
};
StructuredBuffer<MaterialData> Materials : register(t3);
StructuredBuffer<PpmTexture> Textures : register(t5);
StructuredBuffer<uint> TextureTexels : register(t6);
StructuredBuffer<PackedEmitter> Emitters : register(t13);
RWStructuredBuffer<uint> OutputPixels : register(u0);
RWStructuredBuffer<uint> Statistics : register(u1);
RWStructuredBuffer<LinearFilmPixel> LinearFilm : register(u2);
RWStructuredBuffer<PpmPhoton> Photons : register(u5);
RWStructuredBuffer<uint> Heads : register(u6);
RWByteAddressBuffer PpmErrors : register(u7);
RWStructuredBuffer<PpmVisible> VisiblePoints : register(u8);
RWStructuredBuffer<PpmPixel> Pixels : register(u9);
cbuffer ProgressiveBatch : register(b1)
{
    float3 EmissionCenter;
    float WorldRadius;
    double InitialRadius, Alpha;
    uint EmitterCount, PathsPerPass, PopulationSeed, EnabledClasses;
    uint SlotsPerPath, BucketCount, WorkBegin, WorkCount;
    uint PassIndex, LightBegin, BatchLightCount, Stage;
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
    return PpmMix(value);
}
float Random01(uint value)
{
    static const uint FloatFractionShift = 8;
    static const float FloatFractionRange = 16777216.0f;
    return float(Hash(value) >> FloatFractionShift) / FloatFractionRange;
}
#include "Sequence.hlsl"
float PpmLightRandom(uint path, uint dimension)
{
    return Random01(Hash(PopulationSeed ^ LightDomainWord) ^ Hash(PassIndex ^ PassDomainWord) ^ Hash(path) ^
                    Hash(dimension ^ VertexDomainWord));
}
float2 PpmLightPair(uint path, uint dimension)
{
    return float2(PpmLightRandom(path, dimension), PpmLightRandom(path, dimension + 1));
}
void PpmFail(uint flag = ArithmeticError)
{
    uint ignored;
    PpmErrors.InterlockedOr(0, flag, ignored);
}
bool PpmSpectrum(float3 value)
{
    return all(isfinite(value)) && all(value >= 0);
}
void PpmCommit(PpmCounters counts)
{
    AddStatistic(PrimaryCounterWord, counts.Primary);
    AddStatistic(ShadowCounterWord, counts.Shadow);
    AddStatistic(ReflectionCounterWord, counts.Reflection);
}
EmissionDomain PpmDomain()
{
    EmissionDomain domain;
    domain.Center = EmissionCenter;
    domain.Radius = WorldRadius;
    return domain;
}
bool PpmCell(float3 position, out int3 cell)
{
    cell = 0;
    if (!(InitialRadius > 0))
        return false;
    double3 scaled = (double3(position) - double3(EmissionCenter)) / InitialRadius;
    if (!PpmCellValid(scaled.x) || !PpmCellValid(scaled.y) || !PpmCellValid(scaled.z))
    {
        PpmFail();
        return false;
    }
    cell = int3(PpmFloor(scaled.x), PpmFloor(scaled.y), PpmFloor(scaled.z));
    return true;
}
float PpmDecode(uint component)
{
    static const float ByteMaximum = 255, Breakpoint = 0.04045f, Slope = 12.92f;
    static const float Offset = 0.055f, Scale = 1.055f, Power = 2.4f;
    float encoded = component / ByteMaximum;
    return encoded <= Breakpoint ? encoded / Slope : pow((encoded + Offset) / Scale, Power);
}
float3 PpmTexel(PpmTexture texture, int2 texel)
{
    int2 extent = int2(texture.Width, texture.Height);
    texel = (texel % extent + extent) % extent;
    uint packed = TextureTexels[texture.OffsetTexels + texel.y * texture.Width + texel.x];
    static const uint ByteMask = 0xff, GreenShift = 8, BlueShift = 16;
    return float3(PpmDecode(packed & ByteMask), PpmDecode((packed >> GreenShift) & ByteMask),
                  PpmDecode((packed >> BlueShift) & ByteMask));
}
float3 PpmColor(Payload hit)
{
    uint index = Materials[hit.MaterialIndex].DiffuseTextureIndex;
    if (index >= TextureCount || !(hit.Flags & SurfaceTexCoordFlag) || !all(isfinite(hit.TexCoord)))
        return hit.Color;
    PpmTexture texture = Textures[index];
    if (!texture.Width || !texture.Height || texture.OffsetTexels > TextureTexelCount ||
        texture.Height > (TextureTexelCount - texture.OffsetTexels) / texture.Width)
        return hit.Color;
    float2 uv =
        frac(float2(hit.TexCoord.x, 1 - hit.TexCoord.y)) * float2(texture.Width, texture.Height) - 0.5f;
    int2 lower = int2(floor(uv));
    float2 weight = frac(uv);
    return hit.Color *
           lerp(lerp(PpmTexel(texture, lower), PpmTexel(texture, lower + int2(1, 0)), weight.x),
                lerp(PpmTexel(texture, lower + int2(0, 1)), PpmTexel(texture, lower + int2(1, 1)), weight.x),
                weight.y);
}
float3 PpmRadiance(PackedEmitter source, float3 outward)
{
    if (source.Kind == PackedEmitterEnvironment)
        return lerp(source.BackgroundBottom, source.BackgroundTop, (1 - outward.y) / 2);
    if (source.Kind == PackedEmitterRectangle)
        return dot(source.Direction, outward) > 0 ? source.Strength / (source.Width * source.Height) : 0;
    return source.Strength;
}
bool PpmRectangle(PackedEmitter source, float3 origin, float3 direction, float maximum, inout float closest)
{
    if (source.Kind != PackedEmitterRectangle)
        return false;
    float projection = dot(direction, source.Direction);
    if (projection == 0)
        return false;
    float distance = dot(source.Origin - origin, source.Direction) / projection;
    if (!(distance > RayEpsilon && distance < maximum && distance <= closest))
        return false;
    float3 helper = abs(source.Direction.y) < BasisParallelThreshold ? float3(0, 1, 0) : float3(1, 0, 0);
    float3 right = normalize(cross(helper, source.Direction)), up = cross(source.Direction, right);
    float3 relative = origin + distance * direction - source.Origin;
    if (abs(dot(relative, right)) > source.Width / 2 || abs(dot(relative, up)) > source.Height / 2)
        return false;
    closest = distance;
    return true;
}
PpmArrival PpmTrace(float3 origin, float3 direction)
{
    PpmArrival arrival;
    arrival.Hit = CastRay(origin, direction, RayEpsilon, MaximumRayDistance);
    arrival.Surface = (arrival.Hit.Flags & SurfaceHitFlag) != 0;
    arrival.Source = PpmInvalid;
    float hitDistance = asfloat(arrival.Hit.DistanceBits);
    // A sampled lamp's display mesh may precede its analytic rectangle by a
    // rounding-sized offset. Ordinary occluders must keep their exact limit.
    bool proxy = arrival.Surface && Materials[arrival.Hit.MaterialIndex].SampledEmitter != 0;
    float closest = PpmEmitterLimit(hitDistance, proxy, RayEpsilon);
    for (uint source = 0; source < EmitterCount; ++source)
        if (PpmRectangle(Emitters[source], origin, direction, MaximumRayDistance, closest))
        {
            arrival.Source = source;
            arrival.Surface = false;
        }
    if (arrival.Source == PpmInvalid)
        for (uint source = 0; source < EmitterCount; ++source)
        {
            PackedEmitter light = Emitters[source];
            if (arrival.Surface ? light.Primitive != InvalidPrimitiveIndex &&
                                      light.Primitive == arrival.Hit.PrimitiveIndex
                                : light.Kind == PackedEmitterEnvironment)
            {
                arrival.Source = source;
                break;
            }
        }
    arrival.Position = origin + (arrival.Surface ? hitDistance : closest) * direction;
    return arrival;
}
SurfaceSample PpmMirror(SurfaceBsdf surface)
{
    SurfaceSample event = (SurfaceSample)0;
    event.Direction = SurfaceWorld(surface, float3(-surface.View.xy, surface.View.z));
    if (!SurfaceDirectionValid(surface, event.Direction, surface.View.z))
        return (SurfaceSample)0;
    event.Weight = SurfaceAdjointCorrection(surface, event.Direction).xxx;
    event.Pdf = event.ReversePdf = event.RelativeIndex = 1;
    event.Event = ScatteringReflection | ScatteringDelta;
    return event;
}
float3 PpmDirectLight(PpmArrival arrival, SurfaceBsdf surface, SampleContext random, uint domain,
                      bool filterTransmission, inout PpmCounters counts)
{
    EmitterEndpoint endpoint = SampleEmitterEndpoint(
        Emitters, EmitterCount, PpmDomain(), Sample2D(random, domain ^ SourceDimension).x,
        Sample2D(random, domain ^ PositionDimension), Sample2D(random, domain ^ DirectionDimension));
    EmissionSample emission = endpoint.Emission;
    if (!(emission.Flags & EmissionValid))
        return 0;
    bool infinite = (emission.Flags & EmissionInfinite) != 0;
    bool pointSource = (emission.Flags & EmissionDeltaPosition) != 0;
    float3 wi = -emission.Direction;
    double geometry = 1;
    if (!infinite)
    {
        float3 delta = emission.Origin - arrival.Position;
        float distance = length(delta);
        if (!(distance > 0) || !isfinite(distance))
            return 0;
        wi = delta / distance;
        geometry =
            double(pointSource ? 1 : abs(dot(emission.Normal, wi))) / double(distance) / double(distance);
    }
    bool transmitted = filterTransmission &&
        ((arrival.Hit.Flags & SurfaceFrontFaceFlag) != 0
            ? dot(wi, arrival.Hit.GeometricNormal) < 0
            : dot(wi, arrival.Hit.GeometricNormal) > 0);
    if (transmitted && !pointSource && !(emission.Flags & EmissionDeltaDirection)) return 0;
    double pdf = double(endpoint.SelectionProbability) *
                 double(infinite ? emission.DirectionPdf : emission.PositionPdf);
    if (!(pdf > 0))
        return 0;
    SurfaceEvaluation receiver = EvaluateSurface(surface, wi);
    float3 estimate =
        float3(double3(receiver.Value) * double3(PpmRadiance(Emitters[endpoint.EmitterIndex], -wi)) *
               (double(abs(dot(surface.Normal, wi))) * geometry / pdf));
    if (transmitted && (arrival.Hit.Flags & SurfaceFrontFaceFlag) != 0 && pointSource)
        estimate *= exp(-length(emission.Origin - arrival.Position) / surface.Material.Optical.w);
    if (!PpmSpectrum(estimate))
    {
        PpmFail();
        return 0;
    }
    if (!any(estimate > 0))
        return 0;
    float3 start = SurfaceOffset(surface, arrival.Position, wi);
    float maximum = MaximumRayDistance;
    if (!infinite)
    {
        float3 segment = emission.Origin - start;
        float distance = length(segment);
        if (!(distance > 2 * RayEpsilon))
            return 0;
        wi = segment / distance;
        maximum = distance - RayEpsilon;
    }
    IncrementLocalCounter(counts.Shadow);
    Payload blocker = CastRay(start, wi, RayEpsilon, maximum);
    if (blocker.Flags & SurfaceHitFlag)
        return 0;
    float closest = maximum;
    for (uint source = 0; source < EmitterCount; ++source)
        if (PpmRectangle(Emitters[source], start, wi, maximum, closest))
            return 0;
    return estimate;
}
bool PpmSelectVisible(inout PpmVisible visible, inout uint candidates, float variate)
{
    ++candidates;
    if (visible.Active) visible.Beta *= float(candidates) / float(candidates - 1);
    return candidates == 1 || variate < 1.0f / float(candidates);
}
void PpmCameraPath(uint pixel)
{
    PpmPixel state = Pixels[pixel];
    state.IncomingRed = state.IncomingGreen = state.IncomingBlue = 0;
    state.Matched = 0;
    state.VolumeIncomingRed = state.VolumeIncomingGreen = state.VolumeIncomingBlue = 0;
    state.VolumeMatched = 0;
    PpmVisible visible = (PpmVisible)0;
    uint2 coordinate = uint2(pixel % Width + TileX, pixel / Width + TileY);
    SampleContext random = MakeSampleContext(coordinate.x, coordinate.y, FullWidth, PassIndex,
                                             SamplesPerPixel, SamplingSequence, SamplingSeed);
    CameraEndpointFrame frame;
    frame.Position = CameraPosition;
    frame.Forward = CameraForward;
    frame.Right = CameraRight;
    frame.Up = CameraUp;
    frame.HalfWidth = HalfWidth;
    frame.HalfHeight = HalfHeight;
    frame.LensRadius = LensRadius;
    frame.FocusDistance = FocusDistance;
    CameraEndpointSample camera =
        SampleCameraEndpoint(frame,
                             CameraPixelRaster(coordinate.x, coordinate.y, FullWidth, FullHeight,
                                               Sample2D(random, RasterDimension)),
                             Sample2D(random, LensDimension));
    if (!(camera.Flags & CameraEndpointValid))
    {
        PpmFail();
        return;
    }
    float3 origin = camera.Origin, direction = camera.Direction, beta = 1;
    double3 direct = 0;
    float etaScale = 1;
    uint interiorMaterial = PpmInvalid, candidates = 0;
    PpmCounters counts = (PpmCounters)0;
    for (uint depth = 0; depth <= MaxReflectionBounces + 1; ++depth)
    {
        if (!PpmSpectrum(beta) || !isfinite(etaScale))
        {
            PpmFail();
            break;
        }
        if (!any(beta > 0))
            break;
        if (depth == 0)
            IncrementLocalCounter(counts.Primary);
        else
            IncrementLocalCounter(counts.Reflection);
        PpmArrival arrival = PpmTrace(origin, direction);
        uint domain = VertexDomainWord * (depth + 1);
        if (arrival.Surface && arrival.Hit.MaterialIndex >= MaterialCount)
        {
            PpmFail(ControlsError);
            break;
        }
        if (depth == 0 && arrival.Surface && !(arrival.Hit.Flags & SurfaceFrontFaceFlag) &&
            Materials[arrival.Hit.MaterialIndex].Optical.w > 0)
            interiorMaterial = arrival.Hit.MaterialIndex;
        if (interiorMaterial != PpmInvalid)
        {
            MaterialData interior = Materials[interiorMaterial];
            bool finiteSource = arrival.Source != PpmInvalid &&
                Emitters[arrival.Source].Kind == PackedEmitterRectangle;
            float boundary = arrival.Surface || finiteSource
                ? length(arrival.Position - origin) : MaximumRayDistance;
            SubsurfaceFlight flight = SubsurfaceFreeFlight(interior.Optical.w,
                interior.AlbedoReflectivity.xyz * interior.ConductorExtinction.w,
                direction, boundary, Sample2D(random, domain ^ InteriorDimension).x);
            beta *= flight.Weight;
            if (!PpmSpectrum(beta))
            {
                PpmFail();
                break;
            }
            if (!any(beta > 0)) break;
            if (flight.Scattered)
            {
                if (depth > MaxReflectionBounces) break;
                if (PpmSelectVisible(visible, candidates, Sample2D(random, domain ^ VisibleDimension).x))
                {
                    visible.Position = origin + flight.Distance * direction;
                    visible.Active = PpmVisibleVolume;
                    visible.View = direction;
                    visible.Material = interiorMaterial;
                    visible.Depth = depth;
                    visible.Beta = beta * float(candidates);
                }
                break;
            }
        }
        if (arrival.Source != PpmInvalid)
            direct += double3(beta) * double3(PpmRadiance(Emitters[arrival.Source], -direction));
        if (!arrival.Surface || depth > MaxReflectionBounces)
            break;
        MaterialData material = Materials[arrival.Hit.MaterialIndex];
        bool legacy = material.Model == MaterialLegacy;
        float mirror = legacy ? saturate(material.AlbedoReflectivity.w) : 0;
        if (legacy)
            material.Model = MaterialDiffuse;
        float3 color = PpmColor(arrival.Hit);
        SurfaceBsdf surface = MakeSurfaceBsdf(material, arrival.Hit, -direction, color);
        bool subsurfaceTransmission = material.Optical.w > 0 && !SurfaceDelta(surface);
        bool continuous = !SurfaceDelta(surface) && mirror < 1;
        if (continuous && (TransportFlags & NeeBit))
            direct += double3(beta) * double3(PpmDirectLight(arrival, surface, random, domain,
                                                              subsurfaceTransmission, counts)) *
                      double(1 - mirror);
        // A legacy receiver is a sampled mixture: its branch probability already
        // contains (1 - mirror). Multiplying visible beta by it again loses energy.
        bool reflectLegacy = legacy && Sample2D(random, domain ^ LobeDimension).x < mirror;
        if (continuous && !reflectLegacy)
        {
            if (PpmSelectVisible(visible, candidates, Sample2D(random, domain ^ VisibleDimension).x))
            {
                visible.Position = arrival.Position;
                visible.Active = PpmVisibleSurface;
                visible.Normal = arrival.Hit.Normal;
                visible.Front = (arrival.Hit.Flags & SurfaceFrontFaceFlag) != 0;
                visible.GeometricNormal = arrival.Hit.GeometricNormal;
                visible.Depth = depth;
                visible.View = -direction;
                visible.Material = arrival.Hit.MaterialIndex;
                visible.Color = color;
                visible.Beta = beta * float(candidates);
            }
            if (!subsurfaceTransmission) break;
        }
        SurfaceSample event;
        if (reflectLegacy)
            event = PpmMirror(surface);
        else
            event = SampleSurface(surface, Sample2D(random, domain ^ ScatterDimension),
                                  Sample2D(random, domain ^ LobeDimension).x);
        if (!event.Event || !(event.Pdf > 0))
            break;
        if (subsurfaceTransmission && !(event.Event & ScatteringTransmission)) break;
        beta *= event.Weight;
        etaScale *= event.RelativeIndex * event.RelativeIndex;
        if (!PpmSpectrum(beta) || !isfinite(etaScale))
        {
            PpmFail();
            break;
        }
        if ((TransportFlags & RouletteBit) && depth + 1 >= RouletteStartDepth)
        {
            float survival = RouletteProbability(beta * etaScale);
            if (Sample2D(random, domain ^ RouletteDimension).x >= survival)
                break;
            beta /= survival;
        }
        direction = event.Direction;
        origin = SurfaceOffset(surface, arrival.Position, direction);
        if (material.Optical.w > 0 && (event.Event & ScatteringTransmission))
            interiorMaterial = (arrival.Hit.Flags & SurfaceFrontFaceFlag) != 0
                ? arrival.Hit.MaterialIndex : PpmInvalid;
    }
    state.DirectRed += direct.x;
    state.DirectGreen += direct.y;
    state.DirectBlue += direct.z;
    if (!PpmFinite(state.DirectRed) || !PpmFinite(state.DirectGreen) || !PpmFinite(state.DirectBlue))
        PpmFail();
    Pixels[pixel] = state;
    VisiblePoints[pixel] = visible;
    PpmCommit(counts);
}
void PpmLightPath(uint localPath)
{
    uint base = localPath * SlotsPerPath, path = LightBegin + localPath;
    for (uint slot = 0; slot < SlotsPerPath; ++slot)
        Photons[base + slot] = (PpmPhoton)0;
    EmitterEndpoint source = SampleEmitterEndpoint(
        Emitters, EmitterCount, PpmDomain(), PpmLightRandom(path, LightSelectDimension),
        PpmLightPair(path, LightPositionDimension), PpmLightPair(path, LightDirectionDimension));
    EmissionSample launch = source.Emission;
    if (!(launch.Flags & EmissionValid))
        return;
    float3 direction = launch.Direction, flux = launch.Weight;
    float3 origin =
        launch.Origin + launch.Normal * (dot(launch.Normal, direction) >= 0 ? RayEpsilon : -RayEpsilon);
    bool deltaPrefix = true;
    uint interiorMaterial = PpmInvalid;
    PpmCounters counts = (PpmCounters)0;
    for (uint depth = 0; depth <= MaxReflectionBounces; ++depth)
    {
        if (!PpmSpectrum(flux))
        {
            PpmFail();
            break;
        }
        if (!any(flux > 0))
            break;
        IncrementLocalCounter(counts.Reflection);
        PpmArrival arrival = PpmTrace(origin, direction);
        uint dimension = LightEndpointDimensions + depth * LightVertexDimensions;
        if (arrival.Surface && arrival.Hit.MaterialIndex >= MaterialCount)
        {
            PpmFail(ControlsError);
            break;
        }
        if (depth == 0 && arrival.Surface && !(arrival.Hit.Flags & SurfaceFrontFaceFlag) &&
            Materials[arrival.Hit.MaterialIndex].Optical.w > 0)
            interiorMaterial = arrival.Hit.MaterialIndex;
        if (interiorMaterial != PpmInvalid)
        {
            MaterialData interior = Materials[interiorMaterial];
            bool finiteSource = arrival.Source != PpmInvalid &&
                Emitters[arrival.Source].Kind == PackedEmitterRectangle;
            float boundary = arrival.Surface || finiteSource
                ? length(arrival.Position - origin) : MaximumRayDistance;
            SubsurfaceFlight flight = SubsurfaceFreeFlight(interior.Optical.w,
                interior.AlbedoReflectivity.xyz * interior.ConductorExtinction.w,
                direction, boundary, PpmLightRandom(path, dimension + LightLobeOffset));
            float3 collisionPosition = origin + flight.Distance * direction;
            if (flight.Scattered)
            {
                PpmPhoton photon = (PpmPhoton)0;
                photon.Position = collisionPosition;
                photon.Next = PpmInvalid;
                photon.Incoming = -direction;
                photon.Depth = depth;
                photon.Flux = flux;
                photon.Kind = PpmClass(depth, deltaPrefix) | PpmVolumeFlag;
                photon.Reserved = interiorMaterial;
                Photons[base + depth] = photon;
            }
            flux *= flight.Weight;
            if (!PpmSpectrum(flux))
            {
                PpmFail();
                break;
            }
            if (!any(flux > 0)) break;
            if (flight.Scattered)
            {
                if (depth == MaxReflectionBounces) break;
                SubsurfaceDirection phase = SubsurfaceSampleDirection(direction,
                    interior.ConductorIndex.w, PpmLightPair(path, dimension),
                    (TransportFlags & ImportanceBit) != 0);
                if (!(phase.Pdf > 0) || !isfinite(phase.Weight))
                {
                    PpmFail();
                    break;
                }
                flux *= phase.Weight;
                if (!PpmSpectrum(flux))
                {
                    PpmFail();
                    break;
                }
                if ((TransportFlags & RouletteBit) && depth + 1 >= RouletteStartDepth)
                {
                    float survival = RouletteProbability(flux);
                    if (PpmLightRandom(path, dimension + LightRouletteOffset) >= survival) break;
                    flux /= survival;
                }
                deltaPrefix = false;
                origin = collisionPosition;
                direction = phase.Direction;
                continue;
            }
        }
        if (!arrival.Surface)
            break;
        MaterialData material = Materials[arrival.Hit.MaterialIndex];
        bool legacy = material.Model == MaterialLegacy;
        float mirror = legacy ? saturate(material.AlbedoReflectivity.w) : 0;
        if (legacy)
            material.Model = MaterialDiffuse;
        SurfaceBsdf surface =
            MakeSurfaceBsdf(material, arrival.Hit, -direction, PpmColor(arrival.Hit), TransportImportance);
        if (!SurfaceDelta(surface) && mirror < 1)
        {
            PpmPhoton photon = (PpmPhoton)0;
            photon.Position = arrival.Position;
            photon.Next = PpmInvalid;
            photon.Incoming = -direction;
            photon.Depth = depth;
            photon.Flux = flux;
            photon.Kind = PpmClass(depth, deltaPrefix);
            photon.Normal = arrival.Hit.GeometricNormal;
            photon.Reserved = arrival.Hit.MaterialIndex;
            Photons[base + depth] = photon;
        }
        if (depth == MaxReflectionBounces)
            break;
        float lobe = PpmLightRandom(path, dimension + LightLobeOffset);
        SurfaceSample event;
        if (legacy && lobe < mirror)
            event = PpmMirror(surface);
        else
            event = SampleSurface(surface, PpmLightPair(path, dimension), lobe);
        if (!event.Event || !(event.Pdf > 0))
            break;
        flux *= event.Weight;
        deltaPrefix = deltaPrefix && (event.Event & ScatteringDelta) != 0;
        if (!PpmSpectrum(flux))
        {
            PpmFail();
            break;
        }
        if ((TransportFlags & RouletteBit) && depth + 1 >= RouletteStartDepth)
        {
            float survival = RouletteProbability(flux);
            if (PpmLightRandom(path, dimension + LightRouletteOffset) >= survival)
                break;
            flux /= survival;
        }
        direction = event.Direction;
        origin = SurfaceOffset(surface, arrival.Position, direction);
        if (material.Optical.w > 0 && (event.Event & ScatteringTransmission))
            interiorMaterial = (arrival.Hit.Flags & SurfaceFrontFaceFlag) != 0
                ? arrival.Hit.MaterialIndex : PpmInvalid;
    }
    PpmCommit(counts);
}
void PpmInsert(uint index)
{
    PpmPhoton photon = Photons[index];
    int3 cell;
    if (photon.Kind == PpmEmpty || !PpmCell(photon.Position, cell))
        return;
    uint previous;
    InterlockedExchange(Heads[PpmBucket(cell.x, cell.y, cell.z, BucketCount)], index, previous);
    Photons[index].Next = previous;
}
void PpmGatherPixel(uint pixel)
{
    PpmVisible visible = VisiblePoints[pixel];
    PpmPixel state = Pixels[pixel];
    int3 query;
    if (!visible.Active || !(visible.Active == PpmVisibleVolume
        ? state.VolumeRadiusSquared > 0 : state.RadiusSquared > 0) ||
        !PpmCell(visible.Position, query))
        return;
    SurfaceBsdf receiver = (SurfaceBsdf)0;
    if (visible.Active == PpmVisibleSurface)
    {
        Payload hit = (Payload)0;
        hit.Normal = visible.Normal;
        hit.GeometricNormal = visible.GeometricNormal;
        hit.Flags = SurfaceHitFlag | (visible.Front ? SurfaceFrontFaceFlag : 0);
        MaterialData material = Materials[visible.Material];
        if (material.Model == MaterialLegacy)
            material.Model = MaterialDiffuse;
        receiver = MakeSurfaceBsdf(material, hit, visible.View, visible.Color);
    }
    PpmCounters counts = (PpmCounters)0;
    for (int z = -1; z <= 1; ++z)
        for (int y = -1; y <= 1; ++y)
            for (int x = -1; x <= 1; ++x)
            {
                int3 neighbor = query + int3(x, y, z);
                uint index = Heads[PpmBucket(neighbor.x, neighbor.y, neighbor.z, BucketCount)], visited = 0;
                while (index != PpmInvalid)
                {
                    if (index >= BatchLightCount * SlotsPerPath || ++visited > BatchLightCount * SlotsPerPath)
                    {
                        PpmFail();
                        return;
                    }
                    PpmPhoton photon = Photons[index];
                    index = photon.Next;
                    if (!PpmAccept(photon.Kind, photon.Depth, visible.Depth, MaxReflectionBounces,
                                   EnabledClasses, (TransportFlags & NeeBit) != 0))
                        continue;
                    bool volume = (photon.Kind & PpmVolumeFlag) != 0;
                    if (volume != (visible.Active == PpmVisibleVolume)) continue;
                    int3 cell;
                    if (!PpmCell(photon.Position, cell) || any(cell != neighbor))
                        continue;
                    double3 separation = double3(photon.Position) - double3(visible.Position);
                    double distanceSquared = separation.x * separation.x + separation.y * separation.y +
                                             separation.z * separation.z;
                    if (volume)
                    {
                        if (photon.Reserved != visible.Material ||
                            distanceSquared > state.VolumeRadiusSquared)
                            continue;
                        float3 segment = photon.Position - visible.Position;
                        float distance = length(segment);
                        if (distance > 2 * RayEpsilon)
                        {
                            IncrementLocalCounter(counts.Shadow);
                            Payload blocker = CastRay(visible.Position, segment / distance,
                                                      RayEpsilon, distance - RayEpsilon);
                            if (blocker.Flags & SurfaceHitFlag) continue;
                        }
                        MaterialData interior = Materials[visible.Material];
                        double phase = double(SubsurfacePhase(dot(photon.Incoming, visible.View),
                                                                interior.ConductorIndex.w));
                        double3 contribution = double3(visible.Beta) * double3(photon.Flux) *
                                               (phase * double(interior.Optical.w));
                        if (!PpmFinite(contribution.x) || !PpmFinite(contribution.y) ||
                            !PpmFinite(contribution.z) || any(contribution < 0))
                        {
                            PpmFail();
                            return;
                        }
                        ++state.VolumeMatched;
                        state.VolumeIncomingRed += contribution.x;
                        state.VolumeIncomingGreen += contribution.y;
                        state.VolumeIncomingBlue += contribution.z;
                        continue;
                    }
                    if (Materials[visible.Material].Optical.w > 0 &&
                        (visible.Front ? dot(photon.Incoming, visible.GeometricNormal) <= 0
                                       : dot(photon.Incoming, visible.GeometricNormal) >= 0))
                        continue;
                    double plane = abs(separation.x * double(visible.GeometricNormal.x) +
                                       separation.y * double(visible.GeometricNormal.y) +
                                       separation.z * double(visible.GeometricNormal.z));
                    if (!PpmMergeable(distanceSquared, state.RadiusSquared, plane,
                                      dot(visible.GeometricNormal, photon.Normal)))
                        continue;
                    float projection = abs(dot(visible.GeometricNormal, photon.Incoming));
                    if (!(projection > 0))
                        continue;
                    SurfaceEvaluation evaluation = EvaluateSurface(receiver, photon.Incoming);
                    double3 contribution =
                        double3(visible.Beta) * double3(photon.Flux) * double3(evaluation.Value) *
                        (double(abs(dot(receiver.Normal, photon.Incoming))) / double(projection));
                    if (!PpmFinite(contribution.x) || !PpmFinite(contribution.y) ||
                        !PpmFinite(contribution.z) || any(contribution < 0))
                    {
                        PpmFail();
                        return;
                    }
                    ++state.Matched;
                    state.IncomingRed += contribution.x;
                    state.IncomingGreen += contribution.y;
                    state.IncomingBlue += contribution.z;
                }
            }
    // One owner per pixel: no double atomics, and no radius update between batches.
    Pixels[pixel] = state;
    PpmCommit(counts);
}
uint PpmDisplay(float3 value)
{
    static const float Breakpoint = 0.0031308f, LinearScale = 12.92f, Offset = 0.055f, Scale = 1.055f,
                       Exponent = 2.4f;
    static const uint Channels = 3, MaximumByte = 255, GreenShift = 8, BlueShift = 16, AlphaShift = 24;
    float3 color = max(value, 0);
    color /= max(1, max(color.x, max(color.y, color.z)));
    if (GammaCorrection)
        for (uint channel = 0; channel < Channels; ++channel)
            color[channel] = color[channel] <= Breakpoint
                                 ? color[channel] * LinearScale
                                 : Scale * pow(color[channel], 1 / Exponent) - Offset;
    uint3 bytes = uint3(saturate(color) * MaximumByte + 0.5f);
    return bytes.x | (bytes.y << GreenShift) | (bytes.z << BlueShift) | (MaximumByte << AlphaShift);
}
void PpmUpdatePixel(uint pixel)
{
    PpmPixel state = Pixels[pixel];
    PpmRecurrence next = PpmAdvance(state.RadiusSquared, state.Count, state.Matched, Alpha);
    state.RadiusSquared = next.RadiusSquared;
    state.Count = next.Count;
    state.FluxRed = PpmFlux(state.FluxRed, state.IncomingRed, next.Ratio);
    state.FluxGreen = PpmFlux(state.FluxGreen, state.IncomingGreen, next.Ratio);
    state.FluxBlue = PpmFlux(state.FluxBlue, state.IncomingBlue, next.Ratio);
    PpmRecurrence volume = PpmAdvanceVolume(state.VolumeRadiusSquared, state.VolumeCount,
                                            state.VolumeMatched, Alpha);
    state.VolumeRadiusSquared = volume.RadiusSquared;
    state.VolumeCount = volume.Count;
    state.VolumeFluxRed = PpmFlux(state.VolumeFluxRed, state.VolumeIncomingRed, volume.Ratio);
    state.VolumeFluxGreen = PpmFlux(state.VolumeFluxGreen, state.VolumeIncomingGreen, volume.Ratio);
    state.VolumeFluxBlue = PpmFlux(state.VolumeFluxBlue, state.VolumeIncomingBlue, volume.Ratio);
    if (!PpmFinite(state.Count) || !PpmFinite(state.RadiusSquared) || !PpmFinite(state.FluxRed) ||
        !PpmFinite(state.FluxGreen) || !PpmFinite(state.FluxBlue) ||
        !PpmFinite(state.VolumeCount) || !PpmFinite(state.VolumeRadiusSquared) ||
        !PpmFinite(state.VolumeFluxRed) || !PpmFinite(state.VolumeFluxGreen) ||
        !PpmFinite(state.VolumeFluxBlue))
    {
        PpmFail();
        return;
    }
    Pixels[pixel] = state;
    LinearFilmPixel film = (LinearFilmPixel)0;
    film.SampleCount = PassIndex + 1;
    film.MeanRadiance = float3(
        PpmResolve(state.DirectRed, state.FluxRed, state.RadiusSquared, film.SampleCount, PathsPerPass) +
            PpmResolveVolume(state.VolumeFluxRed, state.VolumeRadiusSquared, film.SampleCount, PathsPerPass),
        PpmResolve(state.DirectGreen, state.FluxGreen, state.RadiusSquared, film.SampleCount, PathsPerPass) +
            PpmResolveVolume(state.VolumeFluxGreen, state.VolumeRadiusSquared, film.SampleCount, PathsPerPass),
        PpmResolve(state.DirectBlue, state.FluxBlue, state.RadiusSquared, film.SampleCount, PathsPerPass) +
            PpmResolveVolume(state.VolumeFluxBlue, state.VolumeRadiusSquared, film.SampleCount, PathsPerPass));
    if (!PpmSpectrum(film.MeanRadiance))
    {
        PpmFail();
        return;
    }
    LinearFilm[pixel] = film;
    if (film.SampleCount == SamplesPerPixel)
    {
        OutputPixels[pixel] = PpmDisplay(film.MeanRadiance);
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
    if (Stage == PpmClear)
    {
        if (index < BucketCount)
            Heads[index] = PpmInvalid;
        if (index == 0)
            PpmErrors.Store(0, 0);
        return;
    }
    if (MaxReflectionBounces > PpmMaximumDepth || SlotsPerPath != MaxReflectionBounces + 1 || !PathsPerPass ||
        !BucketCount || (BucketCount & (BucketCount - 1)) || AdaptiveEnabled ||
        ((TransportFlags >> MisShift) & MisMask) || ShadowStrength != 1 || !PpmFinite(InitialRadius) ||
        InitialRadius < 0 || !(Alpha > 0 && Alpha < 1))
    {
        PpmFail(ControlsError);
        return;
    }
    if (Stage == PpmInitialize && index < Width * Height)
    {
        PpmPixel state = (PpmPixel)0;
        state.RadiusSquared = InitialRadius * InitialRadius;
        state.VolumeRadiusSquared = state.RadiusSquared;
        Pixels[index] = state;
        VisiblePoints[index] = (PpmVisible)0;
        LinearFilm[index] = (LinearFilmPixel)0;
        OutputPixels[index] = 0;
    }
    else if (Stage == PpmCamera && index < Width * Height)
        PpmCameraPath(index);
    else if (Stage == PpmEmit && index < BatchLightCount)
        PpmLightPath(index);
    else if (Stage == PpmIndex && index < BatchLightCount * SlotsPerPath)
        PpmInsert(index);
    else if (Stage == PpmGather && index < Width * Height)
        PpmGatherPixel(index);
    else if (Stage == PpmUpdate && index < Width * Height)
        PpmUpdatePixel(index);
}
