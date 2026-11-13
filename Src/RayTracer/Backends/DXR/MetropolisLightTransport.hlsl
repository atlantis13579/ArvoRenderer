// Independent DXR primary-sample-space MLT for offline jobs; preview runs short paths.
// Kelemen et al. (2002), "A Simple and Robust Mutation Strategy for the Metropolis
// Light Transport Algorithm". A deterministic camera estimator maps a primary
// cube to radiance; bootstrap normalization and symmetric Metropolis proposals
// redistribute samples. Neither the path-tracing nor BDPT integrator is reused.
// https://pbr-book.org/3ed-2018/Light_Transport_III_Bidirectional_Methods/Metropolis_Light_Transport
// Quantized signed Gaussian increments wrap on a 24-bit integer torus. Their
// reverse displacement is exactly representable, including across its boundary.
static const uint MarkovMaximumReflections = 32, CameraDimensions = 4, VertexDimensions = 12;
static const uint PrimaryDimensions = CameraDimensions + (MarkovMaximumReflections + 2) * VertexDimensions;
static const uint PrimaryBits = 24, WordBits = 32, PrimaryRange = 1u << PrimaryBits;
static const uint PrimaryMask = PrimaryRange - 1;
static const float PrimaryQuantum = 1.0f / PrimaryRange, MutationTwoPi = 6.2831853071795864769f;
static const uint BootstrapSeedDomain = 0x44585242u, ChainSeedDomain = 0x44585243u;
// Byte offsets mirror MetropolisLayout.h. Keeping one arena preserves the
// existing scene root indices and leaves the MLT signature below 64 DWORDs.
static const uint WordBytes = 4, PassPixelBytes = 4 * WordBytes, DoubleBytes = 2 * WordBytes;
static const uint DistributionBytes = 16, ChainStride = 1728, ChainPathOffset = 16;
static const uint ChainCountersOffset = 48, ChainPrimaryOffset = 80, BootstrapStride = 48, CdfOffset = 32;
static const uint PathColorOffset = 8, PathImportanceOffset = 24;
static const uint ClearStage = 0, BootstrapStage = 1, DistributionStage = 2, InitializeStage = 3,
                  MutateStage = 4, CommitStage = 5;
struct PrimaryVector
{
    uint Values[PrimaryDimensions];
};
struct RandomStep
{
    uint4 State;
    uint Word;
};
struct Mutation
{
    PrimaryVector Primary;
    uint4 Random;
};
uint RotateWord(uint value, uint shift)
{
    return (value << shift) | (value >> (WordBits - shift));
}
RandomStep AdvanceRandom(uint4 state)
{
    // Blackman & Vigna, xoshiro128** 1.1 (2018), public-domain reference:
    // https://prng.di.unimi.it/xoshiro128starstar.c
    static const uint OutputMultiplier = 5, OutputRotation = 7, FinalMultiplier = 9;
    static const uint StateShift = 9, StateRotation = 11;
    RandomStep result;
    result.Word = RotateWord(state.y * OutputMultiplier, OutputRotation) * FinalMultiplier;
    uint temporary = state.y << StateShift;
    uint mixedZ = state.z ^ state.x, mixedW = state.w ^ state.y;
    result.State =
        uint4(state.x ^ mixedW, state.y ^ mixedZ, mixedZ ^ temporary, RotateWord(mixedW, StateRotation));
    return result;
}
uint ScrambleSeed(uint value)
{
    static const uint MultiplierA = 0x7feb352du, MultiplierB = 0x846ca68bu;
    static const uint HalfWord = 16, CenterShift = 15;
    value = (value ^ (value >> HalfWord)) * MultiplierA;
    value = (value ^ (value >> CenterShift)) * MultiplierB;
    return value ^ (value >> HalfWord);
}
uint4 StartRandom(uint seed, uint index, uint domain)
{
    static const uint GoldenWord = 0x9e3779b9u;
    uint value = ScrambleSeed(seed ^ domain) + index * GoldenWord;
    return uint4(ScrambleSeed(value), ScrambleSeed(value + GoldenWord), ScrambleSeed(value + 2 * GoldenWord),
                 ScrambleSeed(value + 3 * GoldenWord) | 1u);
}
float UnitVariate(uint word)
{
    return float(word >> (WordBits - PrimaryBits)) * PrimaryQuantum;
}
PrimaryVector IndependentVector(uint4 random)
{
    PrimaryVector primary;
    [loop] for (uint dimension = 0; dimension < PrimaryDimensions; ++dimension)
    {
        RandomStep step = AdvanceRandom(random);
        random = step.State;
        primary.Values[dimension] = step.Word >> (WordBits - PrimaryBits);
    }
    return primary;
}
Mutation ProposeMutation(PrimaryVector current, uint4 random, bool large, float scale)
{
    Mutation proposal;
    [loop] for (uint first = 0; first < PrimaryDimensions; first += 2)
    {
        RandomStep a = AdvanceRandom(random), b = AdvanceRandom(a.State);
        random = b.State;
        uint x = a.Word >> (WordBits - PrimaryBits), y = b.Word >> (WordBits - PrimaryBits);
        if (!large)
        {
            float radius = sqrt(-2 * log(max(UnitVariate(a.Word), PrimaryQuantum)));
            float angle = MutationTwoPi * UnitVariate(b.Word);
            uint dx = uint(frac(abs(radius * cos(angle)) * scale) * PrimaryRange);
            uint dy = uint(frac(abs(radius * sin(angle)) * scale) * PrimaryRange);
            RandomStep signs = AdvanceRandom(random);
            random = signs.State;
            static const uint XSign = 1, YSign = 2;
            x = current.Values[first] + ((signs.Word & XSign) ? dx : 0u - dx);
            if (first + 1 < PrimaryDimensions)
                y = current.Values[first + 1] + ((signs.Word & YSign) ? dy : 0u - dy);
        }
        proposal.Primary.Values[first] = x & PrimaryMask;
        if (first + 1 < PrimaryDimensions)
            proposal.Primary.Values[first + 1] = y & PrimaryMask;
    }
    proposal.Random = random;
    return proposal;
}

// All scene evaluation and persistent state below belong to the DXR pipeline.
static const float Pi = 3.14159265358979323846f, RayEpsilon = 1.0e-4f;
static const float MaximumRayDistance = 1.0e30f, BasisParallelThreshold = 0.999f;
static const uint ImportanceBit = 1, NeeBit = 2, RouletteBit = 4, MisShift = 3, MisMask = 3;
static const uint ArithmeticFailure = 1, ConfigurationFailure = 2;
static const uint SelectLightDimension = 0, LightPointDimension = 1, LightRayDimension = 3;
static const uint BsdfDirectionDimension = 5, BsdfLobeDimension = 7, RouletteDimension = 8;
static const uint InteriorFreeFlight = 9, InteriorPhaseDirection = 10;
static const double DoubleMaximum = 1.7976931348623157e308L;
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
struct TextureInfo
{
    uint OffsetTexels, Width, Height, Reserved;
};
struct PathValue
{
    float2 Raster;
    float3 Radiance;
    double Importance;
};
struct RayCounts
{
    uint2 Primary, Shadow, Reflection;
};
StructuredBuffer<MaterialData> Materials : register(t3);
StructuredBuffer<TextureInfo> Textures : register(t5);
StructuredBuffer<uint> TextureTexels : register(t6);
StructuredBuffer<PackedEmitter> Emitters : register(t13);
RWStructuredBuffer<uint> OutputPixels : register(u0);
RWStructuredBuffer<uint> Statistics : register(u1);
RWStructuredBuffer<LinearFilmPixel> LinearFilm : register(u2);
RWByteAddressBuffer Arena : register(u5);
RWByteAddressBuffer PassRadiance : register(u6);
RWByteAddressBuffer PathErrors : register(u7);
cbuffer MetropolisBatch : register(b1)
{
    float3 EmissionCenter;
    float EmissionRadius;
    uint EmitterCount, BootstrapCount, ChainCount, MarkovSeed;
    uint WorkBegin, WorkCount, Stage, PassIndex;
    float LargeStepProbability, MutationScale;
    uint2 ReservedControls;
};
#include "Statistics.hlsl"
#include "BsdfTypes.hlsl"
#include "Fresnel.hlsl"
#include "Ggx.hlsl"
#include "Diffuse.hlsl"
#include "Conductor.hlsl"
#include "Dielectric.hlsl"
#include "SurfaceScattering.hlsl"
#include "SubsurfaceScattering.hlsl"
#include "LightSelection.hlsl"
#include "LightEmission.hlsl"
#include "SurfaceEmission.hlsl"
#include "EmitterEndpoint.hlsl"
#include "CameraEndpoint.hlsl"
void ReportFailure(uint flag = ArithmeticFailure)
{
    uint previous;
    PathErrors.InterlockedOr(0, flag, previous);
}
bool FiniteDouble(double value)
{
    return value >= -DoubleMaximum && value <= DoubleMaximum;
}
uint ChainAddress(uint index)
{
    return DistributionBytes + index * ChainStride;
}
uint BootstrapAddress(uint index)
{
    return ChainAddress(ChainCount) + index * BootstrapStride;
}
double ReadDouble(uint address)
{
    uint2 words = Arena.Load2(address);
    return asdouble(words.x, words.y);
}
void WriteDouble(uint address, double value)
{
    uint lower, upper;
    asuint(value, lower, upper);
    Arena.Store2(address, uint2(lower, upper));
}
void WritePath(uint address, PathValue path)
{
    Arena.Store2(address, asuint(path.Raster));
    Arena.Store3(address + PathColorOffset, asuint(path.Radiance));
    WriteDouble(address + PathImportanceOffset, path.Importance);
}
PathValue ReadPath(uint address)
{
    PathValue path;
    path.Raster = asfloat(Arena.Load2(address));
    path.Radiance = asfloat(Arena.Load3(address + PathColorOffset));
    path.Importance = ReadDouble(address + PathImportanceOffset);
    return path;
}
PrimaryVector ReadPrimary(uint address)
{
    PrimaryVector primary;
    [loop] for (uint dimension = 0; dimension < PrimaryDimensions; ++dimension) primary.Values[dimension] =
        Arena.Load(address + dimension * WordBytes);
    return primary;
}
void WritePrimary(uint address, PrimaryVector primary)
{
    [loop] for (uint dimension = 0; dimension < PrimaryDimensions; ++dimension)
        Arena.Store(address + dimension * WordBytes, primary.Values[dimension]);
}
// Coordinate access is read-only. Keep the vector in the caller's storage:
// by-value array arguments otherwise multiply local copies across the path map.
float Primary1D(inout PrimaryVector primary, uint dimension)
{
    return float(primary.Values[dimension]) * PrimaryQuantum;
}
float2 Primary2D(inout PrimaryVector primary, uint dimension)
{
    return float2(Primary1D(primary, dimension), Primary1D(primary, dimension + 1));
}
EmissionDomain WorldDomain()
{
    EmissionDomain domain;
    domain.Center = EmissionCenter;
    domain.Radius = EmissionRadius;
    return domain;
}
void PublishRays(RayCounts rays)
{
    AddStatistic(PrimaryCounterWord, rays.Primary);
    AddStatistic(ShadowCounterWord, rays.Shadow);
    AddStatistic(ReflectionCounterWord, rays.Reflection);
}
float3 OffsetSurface(float3 position, float3 geometry, float3 direction)
{
    return position + geometry * (dot(geometry, direction) >= 0 ? RayEpsilon : -RayEpsilon);
}
float DecodeTexelChannel(uint channel)
{
    static const float ByteMaximum = 255, Threshold = 0.04045f, LinearSlope = 12.92f;
    static const float Offset = 0.055f, Scale = 1.055f, Exponent = 2.4f;
    float value = channel / ByteMaximum;
    return value <= Threshold ? value / LinearSlope : pow((value + Offset) / Scale, Exponent);
}
float3 LoadColor(TextureInfo image, int2 texel)
{
    int2 size = int2(image.Width, image.Height);
    texel = (texel % size + size) % size;
    uint packed = TextureTexels[image.OffsetTexels + texel.y * image.Width + texel.x];
    static const uint ChannelMask = 0xff, GreenShift = 8, BlueShift = 16;
    return float3(DecodeTexelChannel(packed & ChannelMask),
                  DecodeTexelChannel((packed >> GreenShift) & ChannelMask),
                  DecodeTexelChannel((packed >> BlueShift) & ChannelMask));
}
float3 MaterialColor(Payload surface)
{
    uint index = Materials[surface.MaterialIndex].DiffuseTextureIndex;
    if (!(surface.Flags & SurfaceTexCoordFlag) || index >= TextureCount || !all(isfinite(surface.TexCoord)))
        return surface.Color;
    TextureInfo image = Textures[index];
    if (!image.Width || !image.Height || image.OffsetTexels > TextureTexelCount ||
        image.Height > (TextureTexelCount - image.OffsetTexels) / image.Width)
        return surface.Color;
    float2 position =
        frac(float2(surface.TexCoord.x, 1 - surface.TexCoord.y)) * float2(image.Width, image.Height) - 0.5f;
    int2 corner = int2(floor(position));
    float2 blend = frac(position);
    float3 lower = lerp(LoadColor(image, corner), LoadColor(image, corner + int2(1, 0)), blend.x);
    float3 upper =
        lerp(LoadColor(image, corner + int2(0, 1)), LoadColor(image, corner + int2(1, 1)), blend.x);
    return surface.Color * lerp(lower, upper, blend.y);
}
bool RectangleIntersection(PackedEmitter source, float3 origin, float3 direction, float maximum,
                           inout float distance)
{
    if (source.Kind != PackedEmitterRectangle)
        return false;
    float denominator = dot(source.Direction, direction);
    if (denominator == 0)
        return false;
    float candidate = dot(source.Direction, source.Origin - origin) / denominator;
    if (!(candidate > RayEpsilon && candidate < maximum && candidate <= distance))
        return false;
    float3 helper = abs(source.Direction.y) < BasisParallelThreshold ? float3(0, 1, 0) : float3(1, 0, 0);
    float3 tangent = normalize(cross(helper, source.Direction)), bitangent = cross(source.Direction, tangent);
    float3 relative = origin + candidate * direction - source.Origin;
    if (abs(dot(tangent, relative)) > source.Width / 2 || abs(dot(bitangent, relative)) > source.Height / 2)
        return false;
    distance = candidate;
    return true;
}
uint FindEndpoint(uint primitive, bool environment)
{
    for (uint source = 0; source < EmitterCount; ++source)
        if (environment ? Emitters[source].Kind == PackedEmitterEnvironment
                        : primitive != InvalidPrimitiveIndex && Emitters[source].Primitive == primitive)
            return source;
    return InvalidPrimitiveIndex;
}
float3 EndpointRadiance(uint source, float3 launch)
{
    PackedEmitter light = Emitters[source];
    if (light.Kind == PackedEmitterEnvironment)
        return lerp(light.BackgroundBottom, light.BackgroundTop, (1 - launch.y) / 2);
    if (light.Kind == PackedEmitterRectangle)
        return dot(light.Direction, launch) > 0 ? light.Strength / (light.Width * light.Height) : 0;
    return light.Strength;
}
bool ConnectionVisible(float3 origin, float3 direction, float distance, inout RayCounts counts)
{
    IncrementLocalCounter(counts.Shadow);
    Payload blocker = CastRay(origin, direction, RayEpsilon, distance);
    if (blocker.Flags & SurfaceHitFlag)
        return false;
    float nearest = distance;
    for (uint source = 0; source < EmitterCount; ++source)
        if (RectangleIntersection(Emitters[source], origin, direction, distance, nearest))
            return false;
    return true;
}
float TechniqueWeight(float sampledPdf, float competingPdf)
{
    static const uint NoMis = 0, BalanceMis = 1;
    uint mode = (TransportFlags >> MisShift) & MisMask;
    if (mode == NoMis)
        return 1;
    if (!(sampledPdf > 0))
        return 0;
    if (!(competingPdf > 0))
        return 1;
    float commonScale = max(sampledPdf, competingPdf);
    float sampled = sampledPdf / commonScale, other = competingPdf / commonScale;
    if (mode != BalanceMis)
    {
        sampled *= sampled;
        other *= other;
    }
    return sampled / (sampled + other);
}
float VisibleEmissionWeight(uint source, float3 normal, float3 outward, float3 separation, bool infinite,
                            bool previousDelta, float previousPdf)
{
    if (previousDelta || !(TransportFlags & NeeBit))
        return 1;
    if (((TransportFlags >> MisShift) & MisMask) == 0)
        return 0;
    EmitterEndpointDensity density =
        EmitterEndpointPdf(Emitters, EmitterCount, WorldDomain(), source, normal, outward);
    float directPdf = density.SelectionProbability * density.Direction;
    if (!infinite)
    {
        float lightCosine = abs(dot(normal, outward));
        directPdf = lightCosine > 0 ? density.SelectionProbability * density.Position *
                                          dot(separation, separation) / lightCosine
                                    : 0;
    }
    return TechniqueWeight(previousPdf, directPdf);
}
float3 SampleDirect(inout PrimaryVector primary, uint dimensions, SurfaceBsdf surface, float3 position,
                    float continuousFraction, float currentInterior, inout RayCounts counts)
{
    EmitterEndpoint endpoint = SampleEmitterEndpoint(Emitters, EmitterCount, WorldDomain(),
                                                     Primary1D(primary, dimensions + SelectLightDimension),
                                                     Primary2D(primary, dimensions + LightPointDimension),
                                                     Primary2D(primary, dimensions + LightRayDimension));
    EmissionSample light = endpoint.Emission;
    if (!(light.Flags & EmissionValid))
        return 0;
    bool infinite = (light.Flags & EmissionInfinite) != 0;
    bool deltaPosition = (light.Flags & EmissionDeltaPosition) != 0;
    bool delta = (light.Flags & (EmissionDeltaPosition | EmissionDeltaDirection)) != 0;
    if (!delta && !(TransportFlags & NeeBit))
        return 0;
    float3 direction = -light.Direction;
    float distance = MaximumRayDistance, geometry = 1;
    if (!infinite)
    {
        float3 separation = light.Origin - position;
        distance = length(separation);
        if (!(distance > 0) || !isfinite(distance))
            return 0;
        direction = separation / distance;
        geometry = (deltaPosition ? 1 : abs(dot(light.Normal, direction))) / distance / distance;
    }
    float endpointPdf = endpoint.SelectionProbability * (infinite ? light.DirectionPdf : light.PositionPdf);
    if (!(endpointPdf > 0))
        return 0;
    SurfaceEvaluation scattering = EvaluateSurface(surface, direction);
    float angularPdf = infinite || delta ? endpointPdf : geometry > 0 ? endpointPdf / geometry : 0;
    float mis = delta ? 1 : TechniqueWeight(angularPdf, scattering.Pdf * continuousFraction);
    float3 estimate =
        scattering.Value * EndpointRadiance(endpoint.EmitterIndex, -direction) *
        (continuousFraction * abs(dot(surface.Normal, direction)) * geometry * mis / endpointPdf);
    float connectionMeanFreePath = surface.Material.Optical.w > 0
        ? (dot(surface.GeometricNormal, direction) < 0 ? surface.Material.Optical.w : 0)
        : currentInterior;
    if (connectionMeanFreePath > 0) estimate *= exp(-distance / connectionMeanFreePath);
    if (!all(isfinite(estimate)) || !any(estimate > 0))
        return estimate;
    float3 origin = SurfaceOffset(surface, position, direction);
    if (!infinite)
    {
        float3 target = deltaPosition ? light.Origin : OffsetSurface(light.Origin, light.Normal, -direction);
        float3 separation = target - origin;
        distance = length(separation);
        if (!(distance > 2 * RayEpsilon))
            return 0;
        direction = separation / distance;
        distance -= RayEpsilon;
    }
    return ConnectionVisible(origin, direction, distance, counts) ? estimate : 0;
}
float3 SampleInteriorDirect(inout PrimaryVector primary, uint dimensions, float3 position,
                            float3 incoming, MaterialData interior, inout RayCounts counts)
{
    EmitterEndpoint endpoint = SampleEmitterEndpoint(Emitters, EmitterCount, WorldDomain(),
        Primary1D(primary, dimensions + SelectLightDimension),
        Primary2D(primary, dimensions + LightPointDimension),
        Primary2D(primary, dimensions + LightRayDimension));
    EmissionSample light = endpoint.Emission;
    if (!(light.Flags & EmissionValid)) return 0;
    bool infinite = (light.Flags & EmissionInfinite) != 0;
    bool deltaPosition = (light.Flags & EmissionDeltaPosition) != 0;
    bool delta = (light.Flags & (EmissionDeltaPosition | EmissionDeltaDirection)) != 0;
    if (!delta && !(TransportFlags & NeeBit)) return 0;
    float3 direction = -light.Direction;
    float distance = MaximumRayDistance, geometry = 1;
    if (!infinite)
    {
        float3 separation = light.Origin - position;
        float squared = dot(separation, separation);
        if (!(squared > 0) || !isfinite(squared)) return 0;
        distance = sqrt(squared);
        direction = separation / distance;
        geometry = (deltaPosition ? 1 : abs(dot(light.Normal, direction))) / squared;
    }
    float endpointPdf = endpoint.SelectionProbability * (infinite ? light.DirectionPdf : light.PositionPdf);
    if (!(endpointPdf > 0) || !(geometry > 0)) return 0;
    float angularPdf = delta ? endpointPdf : infinite ? endpointPdf : endpointPdf / geometry;
    float phase = SubsurfacePhase(dot(incoming, direction), interior.ConductorIndex.w);
    float phasePdf = (TransportFlags & ImportanceBit) != 0 ? phase : 1 / (4 * Pi);
    float mis = delta ? 1 : TechniqueWeight(angularPdf, phasePdf);
    float3 estimate = EndpointRadiance(endpoint.EmitterIndex, -direction)
        * (phase * geometry * mis / endpointPdf * exp(-distance / interior.Optical.w));
    if (!all(isfinite(estimate)) || !any(estimate > 0)) return estimate;
    if (!infinite)
    {
        float3 target = deltaPosition ? light.Origin : OffsetSurface(light.Origin, light.Normal, -direction);
        float3 separation = target - position;
        distance = length(separation);
        if (!(distance > RayEpsilon)) return 0;
        direction = separation / distance;
        distance -= RayEpsilon;
    }
    return ConnectionVisible(position, direction, distance, counts) ? estimate : 0;
}
PathValue CameraContribution(inout PrimaryVector primary, inout RayCounts counts)
{
    CameraEndpointFrame frame;
    frame.Position = CameraPosition;
    frame.Forward = CameraForward;
    frame.Right = CameraRight;
    frame.Up = CameraUp;
    frame.HalfWidth = HalfWidth;
    frame.HalfHeight = HalfHeight;
    frame.LensRadius = LensRadius;
    frame.FocusDistance = FocusDistance;
    CameraEndpointSample sensor = SampleCameraEndpoint(frame, Primary2D(primary, 0), Primary2D(primary, 2));
    PathValue result = (PathValue)0;
    result.Raster = sensor.Raster;
    if (!(sensor.Flags & CameraEndpointValid))
    {
        ReportFailure();
        return result;
    }
    float3 origin = sensor.Origin, direction = sensor.Direction, beta = 1, previous = origin;
    float previousPdf = 0, etaScale = 1;
    bool previousDelta = true;
    uint interiorMaterial = InvalidPrimitiveIndex;
    [loop] for (uint bounce = 0; bounce <= MaxReflectionBounces + 1; ++bounce)
    {
        uint dimensions = CameraDimensions + bounce * VertexDimensions;
        if (bounce == 0)
            IncrementLocalCounter(counts.Primary);
        else
            IncrementLocalCounter(counts.Reflection);
        Payload hit = CastRay(origin, direction, RayEpsilon, MaximumRayDistance);
        bool found = (hit.Flags & SurfaceHitFlag) != 0;
        float closest = found ? asfloat(hit.DistanceBits) + RayEpsilon : MaximumRayDistance;
        uint rectangle = InvalidPrimitiveIndex;
        for (uint source = 0; source < EmitterCount; ++source)
            if (RectangleIntersection(Emitters[source], origin, direction, MaximumRayDistance, closest))
                rectangle = source;
        if (found && hit.MaterialIndex >= MaterialCount)
        {
            ReportFailure(ConfigurationFailure);
            return result;
        }
        if (bounce == 0 && found && (hit.Flags & SurfaceFrontFaceFlag) == 0 &&
            Materials[hit.MaterialIndex].Optical.w > 0)
            interiorMaterial = hit.MaterialIndex;
        if (interiorMaterial != InvalidPrimitiveIndex)
        {
            MaterialData interior = Materials[interiorMaterial];
            float endpoint = rectangle != InvalidPrimitiveIndex ? closest
                : found ? asfloat(hit.DistanceBits) : MaximumRayDistance;
            SubsurfaceFlight flight = SubsurfaceFreeFlight(interior.Optical.w,
                interior.AlbedoReflectivity.xyz * interior.ConductorExtinction.w,
                direction, endpoint, Primary1D(primary, dimensions + InteriorFreeFlight));
            beta *= flight.Weight;
            if (!all(isfinite(beta))) { ReportFailure(); return result; }
            if (!any(beta > 0)) break;
            if (flight.Scattered != 0)
            {
                if (bounce > MaxReflectionBounces) break;
                float3 position = origin + direction * flight.Distance;
                result.Radiance += beta * SampleInteriorDirect(primary, dimensions, position,
                    direction, interior, counts);
                SubsurfaceDirection phase = SubsurfaceSampleDirection(direction,
                    interior.ConductorIndex.w, Primary2D(primary, dimensions + InteriorPhaseDirection),
                    (TransportFlags & ImportanceBit) != 0);
                if (!(phase.Pdf > 0)) break;
                beta *= phase.Weight;
                if (!all(isfinite(beta))) { ReportFailure(); return result; }
                if (!any(beta > 0)) break;
                if ((TransportFlags & RouletteBit) && bounce + 1 >= RouletteStartDepth)
                {
                    static const float SurvivalFloor = 0.05f, SurvivalCeiling = 0.95f;
                    float survival = clamp(max(beta.x, max(beta.y, beta.z)) * etaScale,
                        SurvivalFloor, SurvivalCeiling);
                    if (Primary1D(primary, dimensions + RouletteDimension) >= survival) break;
                    beta /= survival;
                }
                previous = position;
                previousPdf = phase.Pdf;
                previousDelta = false;
                origin = position;
                direction = phase.Direction;
                continue;
            }
        }
        bool infinite = !found && rectangle == InvalidPrimitiveIndex;
        uint light =
            rectangle != InvalidPrimitiveIndex ? rectangle : FindEndpoint(hit.PrimitiveIndex, infinite);
        float3 position =
            origin + direction * (rectangle != InvalidPrimitiveIndex ? closest : asfloat(hit.DistanceBits));
        if (light != InvalidPrimitiveIndex)
        {
            float3 normal =
                rectangle != InvalidPrimitiveIndex ? Emitters[light].Direction : hit.GeometricNormal;
            float weight = VisibleEmissionWeight(light, normal, -direction, position - previous, infinite,
                                                 previousDelta, previousPdf);
            result.Radiance += beta * EndpointRadiance(light, -direction) * weight;
        }
        if (infinite || rectangle != InvalidPrimitiveIndex || bounce > MaxReflectionBounces)
            break;
        MaterialData material = Materials[hit.MaterialIndex];
        float mirror = material.Model == MaterialLegacy ? saturate(material.AlbedoReflectivity.w) : 0;
        if (material.Model == MaterialLegacy)
            material.Model = MaterialDiffuse;
        SurfaceBsdf surface = MakeSurfaceBsdf(material, hit, -direction, MaterialColor(hit));
        result.Radiance += beta * SampleDirect(primary, dimensions, surface, position, 1 - mirror,
            interiorMaterial == InvalidPrimitiveIndex ? 0 : Materials[interiorMaterial].Optical.w, counts);
        SurfaceSample scatter = (SurfaceSample)0;
        float lobe = Primary1D(primary, dimensions + BsdfLobeDimension);
        if (lobe < mirror)
        {
            scatter.Direction = normalize(reflect(direction, surface.Normal));
            if (dot(surface.Normal, scatter.Direction) <= 0 ||
                !SurfaceDirectionValid(surface, scatter.Direction, dot(surface.Normal, scatter.Direction)))
                break;
            scatter.Pdf = mirror;
            scatter.Weight = 1;
            scatter.RelativeIndex = 1;
            scatter.Event = ScatteringDelta | ScatteringReflection;
        }
        else
        {
            scatter = SampleSurface(surface, Primary2D(primary, dimensions + BsdfDirectionDimension),
                                    mirror > 0 ? (lobe - mirror) / (1 - mirror) : lobe);
            scatter.Pdf *= 1 - mirror;
        }
        if (!scatter.Event || !(scatter.Pdf > 0))
            break;
        beta *= scatter.Weight;
        etaScale *= scatter.RelativeIndex * scatter.RelativeIndex;
        if (!all(isfinite(beta)) || !isfinite(etaScale))
        {
            ReportFailure();
            return result;
        }
        if (!any(beta > 0))
            break;
        if ((TransportFlags & RouletteBit) && bounce + 1 >= RouletteStartDepth)
        {
            static const float SurvivalFloor = 0.05f, SurvivalCeiling = 0.95f;
            float survival =
                clamp(max(beta.x, max(beta.y, beta.z)) * etaScale, SurvivalFloor, SurvivalCeiling);
            if (Primary1D(primary, dimensions + RouletteDimension) >= survival)
                break;
            beta /= survival;
        }
        previous = position;
        previousPdf = scatter.Pdf;
        previousDelta = (scatter.Event & ScatteringDelta) != 0;
        origin = SurfaceOffset(surface, position, scatter.Direction);
        direction = scatter.Direction;
        if (material.Optical.w > 0 && (scatter.Event & ScatteringTransmission) != 0)
            interiorMaterial = (hit.Flags & SurfaceFrontFaceFlag) != 0 ? hit.MaterialIndex : InvalidPrimitiveIndex;
    }
    static const double RedWeight = 0.2126L, GreenWeight = 0.7152L, BlueWeight = 0.0722L;
    result.Importance = double(result.Radiance.x) * RedWeight + double(result.Radiance.y) * GreenWeight +
                        double(result.Radiance.z) * BlueWeight;
    if (!all(isfinite(result.Radiance)) || any(result.Radiance < 0) || !FiniteDouble(result.Importance) ||
        result.Importance < 0)
        ReportFailure();
    return result;
}
void AtomicChannel(uint address, float contribution)
{
    uint expected = PassRadiance.Load(address);
    [allow_uav_condition] while (true)
    {
        float sum = asfloat(expected) + contribution;
        if (!isfinite(sum))
        {
            ReportFailure();
            return;
        }
        uint observed;
        PassRadiance.InterlockedCompareExchange(address, expected, asuint(sum), observed);
        if (expected == observed)
            return;
        expected = observed;
    }
}
void Deposit(PathValue path, double probability, double normalization)
{
    if (!(probability > 0 && path.Importance > 0))
        return;
    uint2 raster = uint2(path.Raster * float2(FullWidth, FullHeight));
    if (raster.x < TileX || raster.y < TileY || raster.x - TileX >= Width || raster.y - TileY >= Height)
        return;
    double scale = probability * normalization / path.Importance;
    float3 value = float3(double(path.Radiance.x) * scale, double(path.Radiance.y) * scale,
                          double(path.Radiance.z) * scale);
    if (!all(isfinite(value)) || any(value < 0))
    {
        ReportFailure();
        return;
    }
    uint address = ((raster.y - TileY) * Width + raster.x - TileX) * PassPixelBytes;
    AtomicChannel(address, value.x);
    AtomicChannel(address + WordBytes, value.y);
    AtomicChannel(address + 2 * WordBytes, value.z);
}
void BootstrapPath(uint invocation)
{
    if (invocation >= WorkCount || invocation + WorkBegin >= BootstrapCount)
        return;
    RayCounts counts = (RayCounts)0;
    PrimaryVector primary =
        IndependentVector(StartRandom(MarkovSeed, invocation + WorkBegin, BootstrapSeedDomain));
    WritePath(BootstrapAddress(invocation + WorkBegin), CameraContribution(primary, counts));
    PublishRays(counts);
}
void BuildDistribution()
{
    double total = 0;
    [loop] for (uint source = 0; source < BootstrapCount; ++source)
    {
        double importance = ReadPath(BootstrapAddress(source)).Importance;
        if (!(importance >= 0) || !FiniteDouble(importance))
            ReportFailure();
        total += importance;
        WriteDouble(BootstrapAddress(source) + CdfOffset, total);
    }
    if (!BootstrapCount || !FiniteDouble(total))
        ReportFailure();
    WriteDouble(0, BootstrapCount ? total / BootstrapCount : 0);
    WriteDouble(DoubleBytes, total);
}
void InitializeChain(uint index)
{
    if (index >= ChainCount)
        return;
    RandomStep selected = AdvanceRandom(StartRandom(MarkovSeed, index, ChainSeedDomain));
    uint address = ChainAddress(index);
    Arena.Store4(address, selected.State);
    Arena.Store4(address + ChainCountersOffset, 0);
    Arena.Store4(address + ChainCountersOffset + 4 * WordBytes, 0);
    PathValue current = (PathValue)0;
    double total = ReadDouble(DoubleBytes);
    if (total > 0)
    {
        double target = double(UnitVariate(selected.Word)) * total;
        uint first = 0, end = BootstrapCount;
        while (first < end)
        {
            uint midpoint = first + (end - first) / 2;
            if (ReadDouble(BootstrapAddress(midpoint) + CdfOffset) <= target)
                first = midpoint + 1;
            else
                end = midpoint;
        }
        current = ReadPath(BootstrapAddress(first));
        WritePrimary(address + ChainPrimaryOffset,
                     IndependentVector(StartRandom(MarkovSeed, first, BootstrapSeedDomain)));
    }
    WritePath(address + ChainPathOffset, current);
}
void AdvanceChain(uint index)
{
    if (index >= ChainCount)
        return;
    double normalization = ReadDouble(0);
    if (!(normalization > 0))
        return;
    uint address = ChainAddress(index);
    uint4 random = Arena.Load4(address);
    PathValue current = ReadPath(address + ChainPathOffset);
    PrimaryVector primary = ReadPrimary(address + ChainPrimaryOffset);
    uint2 proposals = Arena.Load2(address + ChainCountersOffset),
          accepted = Arena.Load2(address + ChainCountersOffset + DoubleBytes);
    uint2 largeSteps = Arena.Load2(address + ChainCountersOffset + 2 * DoubleBytes),
          rejectedPositive = Arena.Load2(address + ChainCountersOffset + 3 * DoubleBytes);
    uint pixels = FullWidth * FullHeight, quota = pixels / ChainCount + (index < pixels % ChainCount ? 1 : 0);
    RayCounts counts = (RayCounts)0;
    [loop] for (uint step = WorkBegin; step < quota && step - WorkBegin < WorkCount; ++step)
    {
        RandomStep choose = AdvanceRandom(random);
        bool large = UnitVariate(choose.Word) < LargeStepProbability;
        Mutation proposal = ProposeMutation(primary, choose.State, large, MutationScale);
        PathValue candidate = CameraContribution(proposal.Primary, counts);
        double ratio = current.Importance > 0 ? candidate.Importance / current.Importance : 1;
        double acceptance = ratio < 1 ? ratio : 1;
        Deposit(current, 1 - acceptance, normalization);
        Deposit(candidate, acceptance, normalization);
        IncrementLocalCounter(proposals);
        if (large)
            IncrementLocalCounter(largeSteps);
        RandomStep decision = AdvanceRandom(proposal.Random);
        random = decision.State;
        if (double(UnitVariate(decision.Word)) < acceptance)
        {
            current = candidate;
            primary = proposal.Primary;
            IncrementLocalCounter(accepted);
        }
        else if (candidate.Importance > 0)
            IncrementLocalCounter(rejectedPositive);
    }
    Arena.Store4(address, random);
    WritePath(address + ChainPathOffset, current);
    WritePrimary(address + ChainPrimaryOffset, primary);
    Arena.Store2(address + ChainCountersOffset, proposals);
    Arena.Store2(address + ChainCountersOffset + DoubleBytes, accepted);
    Arena.Store2(address + ChainCountersOffset + 2 * DoubleBytes, largeSteps);
    Arena.Store2(address + ChainCountersOffset + 3 * DoubleBytes, rejectedPositive);
    PublishRays(counts);
}
uint EncodeDisplay(float3 radiance)
{
    static const float LinearThreshold = 0.0031308f, LinearScale = 12.92f, Offset = 0.055f, Scale = 1.055f,
                       Exponent = 2.4f;
    static const uint ByteMaximum = 255, GreenShift = 8, BlueShift = 16, AlphaShift = 24;
    float3 value = max(radiance, 0);
    value /= max(1, max(value.x, max(value.y, value.z)));
    if (GammaCorrection)
        for (uint component = 0; component < 3; ++component)
            value[component] = value[component] <= LinearThreshold
                                   ? value[component] * LinearScale
                                   : Scale * pow(value[component], 1 / Exponent) - Offset;
    uint3 encoded = uint3(saturate(value) * ByteMaximum + 0.5f);
    return encoded.x | (encoded.y << GreenShift) | (encoded.z << BlueShift) | (ByteMaximum << AlphaShift);
}
[shader("raygeneration")] void RayGen() {
    uint2 invocation = DispatchRaysIndex().xy;
    if (Stage == BootstrapStage)
    {
        BootstrapPath(invocation.x);
        return;
    }
    if (Stage == DistributionStage)
    {
        if (all(invocation == 0))
            BuildDistribution();
        return;
    }
    if (Stage == InitializeStage)
    {
        InitializeChain(invocation.x);
        return;
    }
    if (Stage == MutateStage)
    {
        AdvanceChain(invocation.x);
        return;
    }
    if (invocation.x >= Width || invocation.y >= Height)
        return;
    uint pixel = invocation.y * Width + invocation.x;
    if (Stage == ClearStage)
    {
        PassRadiance.Store4(pixel * PassPixelBytes, 0);
        if (PassIndex == 0)
            LinearFilm[pixel] = (LinearFilmPixel)0;
        if (pixel == 0)
            PathErrors.Store(0, 0);
        return;
    }
    if (Stage != CommitStage)
        return;
    float3 value = asfloat(PassRadiance.Load3(pixel * PassPixelBytes));
    if (!all(isfinite(value)) || any(value < 0))
    {
        ReportFailure();
        return;
    }
    LinearFilmPixel film = LinearFilm[pixel];
    AccumulateFilmSample(film, value);
    if (!all(isfinite(film.MeanRadiance)) || !isfinite(film.LuminanceM2))
    {
        ReportFailure();
        return;
    }
    LinearFilm[pixel] = film;
    if (film.SampleCount == SamplesPerPixel)
    {
        OutputPixels[pixel] = EncodeDisplay(film.MeanRadiance);
        AddStatistic(CompletedCounterWord, uint2(1, 0));
    }
}
