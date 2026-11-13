// Independent Compute PSSMLT. Kelemen et al. (2002), "A Simple and Robust Mutation
// Strategy for the Metropolis Light Transport Algorithm". Bootstrap mean luminance
// normalizes a persistent chain; accepted AND rejected states contribute to film.
// https://pbr-book.org/3ed-2018/Light_Transport_III_Bidirectional_Methods/Metropolis_Light_Transport
// The 24-bit primary torus makes paired +/- integer steps exactly reversible.
// Gaussian magnitudes are quantized, wrapped and assigned an independent sign.
// Each pass contains one mutation per full-film pixel on average. Crops replay
// all chains and retain local splats. No other integrator is included or invoked.
static const uint MltMaximumReflections = 32, MltCameraDimensions = 4, MltVertexDimensions = 12;
static const uint MltDimensions = MltCameraDimensions + (MltMaximumReflections + 2) * MltVertexDimensions;
static const uint MltPrimaryBits = 24, MltPrimaryRange = 1u << MltPrimaryBits;
static const uint MltPrimaryMask = MltPrimaryRange - 1, MltWordBits = 32;
static const float MltUnitScale = 1.0f / MltPrimaryRange, MltTwoPi = 6.2831853071795864769f;
static const uint MltBootstrapDomain = 0x42535450u, MltChainDomain = 0x43484e53u;
struct MltPrimary
{
    uint Values[MltDimensions];
};
struct MltProposal
{
    MltPrimary Primary;
    uint4 Random;
};
uint MltAvalanche(uint value)
{
    static const uint MixA = 0x85ebca6bu, MixB = 0xc2b2ae35u;
    static const uint HalfWord = 16, Middle = 13;
    value ^= value >> HalfWord;
    value *= MixA;
    value ^= value >> Middle;
    value *= MixB;
    return value ^ (value >> HalfWord);
}
uint4 MltSeed(uint seed, uint index, uint domain)
{
    static const uint SaltA = 0x9e3779b9u, SaltB = 0x243f6a88u, SaltC = 0xb7e15162u;
    uint key = MltAvalanche(seed ^ domain) ^ MltAvalanche(index);
    return uint4(MltAvalanche(key), MltAvalanche(key ^ SaltA), MltAvalanche(key ^ SaltB),
                 MltAvalanche(key ^ SaltC) | 1u);
}
uint4 MltNext(uint4 state)
{
    // Marsaglia (2003), "Xorshift RNGs": the nonzero 128-bit state is persistent.
    static const uint ShiftA = 11, ShiftB = 19, ShiftC = 8;
    uint mixed = state.x ^ (state.x << ShiftA);
    return uint4(state.y, state.z, state.w, state.w ^ (state.w >> ShiftB) ^ mixed ^ (mixed >> ShiftC));
}
float MltUniform(uint word)
{
    return float(word >> (MltWordBits - MltPrimaryBits)) * MltUnitScale;
}
MltPrimary MltIndependent(uint4 random)
{
    MltPrimary result;
    [loop] for (uint dimension = 0; dimension < MltDimensions; ++dimension)
    {
        random = MltNext(random);
        result.Values[dimension] = random.w >> (MltWordBits - MltPrimaryBits);
    }
    return result;
}
MltProposal MltPropose(MltPrimary current, uint4 random, bool large, float scale)
{
    MltProposal result;
    [loop] for (uint dimension = 0; dimension < MltDimensions; ++dimension)
    {
        random = MltNext(random);
        uint next = random.w >> (MltWordBits - MltPrimaryBits);
        if (!large)
        {
            float radius = sqrt(-2 * log(max(MltUniform(random.w), MltUnitScale)));
            random = MltNext(random);
            float magnitude = abs(radius * cos(MltTwoPi * MltUniform(random.w))) * scale;
            uint displacement = uint(frac(magnitude) * MltPrimaryRange);
            random = MltNext(random);
            next = current.Values[dimension] + ((random.w & 1u) ? displacement : 0u - displacement);
        }
        result.Primary.Values[dimension] = next & MltPrimaryMask;
    }
    result.Random = random;
    return result;
}

static const float Pi = 3.14159265358979323846f, RayEpsilon = 1.0e-4f;
static const float MaximumRayDistance = 1.0e30f, BasisParallelThreshold = 0.999f;
static const uint ImportanceBit = 1, NeeBit = 2, RouletteBit = 4, MisShift = 3, MisMask = 3;
static const uint MltGroupSize = 64, MltFilmGroupWidth = 8, ScalarBytes = 4, FilmStride = 4 * ScalarBytes;
static const uint MltNumericalFailure = 1, MltInvalidConfiguration = 2;
static const uint LightChoice = 0, LightPosition = 1, LightDirection = 3;
static const uint ScatterDirection = 5, ScatterLobe = 7, RouletteChoice = 8;
static const uint InteriorFreeFlight = 9, InteriorPhaseDirection = 10;
static const double MltDoubleMaximum = 1.7976931348623157e308L;
#include "Types.hlsl"
#include "MaterialTypes.hlsl"
#include "SubsurfaceScattering.hlsl"
#include "EmitterData.hlsl"
struct TextureRecord
{
    uint OffsetTexels, Width, Height, Reserved;
};
struct MltPath
{
    float2 Raster;
    float3 Radiance;
    float Reserved;
    double Importance;
};
struct MltChain
{
    MltPath Current;
    uint4 Random;
    uint2 Proposals, Accepted, LargeSteps, RejectedPositive;
};
struct MltStatistics
{
    uint2 Primary, Shadow, Reflection, Tests;
};
ByteAddressBuffer Nodes : register(t0);
StructuredBuffer<uint> PrimitiveRefs : register(t1);
StructuredBuffer<PackedTriangle> Triangles : register(t2);
StructuredBuffer<PackedSphere> Spheres : register(t3);
StructuredBuffer<PackedMaterial> Materials : register(t4);
StructuredBuffer<PackedEmitter> Emitters : register(t5);
StructuredBuffer<TextureRecord> Textures : register(t6);
StructuredBuffer<uint> TextureTexels : register(t7);
RWStructuredBuffer<float4> Accumulation : register(u0);
RWStructuredBuffer<uint> SampleCounts : register(u1);
RWByteAddressBuffer Pass : register(u2);
RWStructuredBuffer<MltStatistics> Statistics : register(u3);
RWStructuredBuffer<uint> SampleTargets : register(u4);
RWStructuredBuffer<MltChain> Chains : register(u5);
RWByteAddressBuffer MltError : register(u7);
RWStructuredBuffer<MltPrimary> PrimaryStates : register(u8);
RWStructuredBuffer<MltPath> Bootstrap : register(u9);
RWStructuredBuffer<double> Cumulative : register(u10);
RWStructuredBuffer<double2> Distribution : register(u11);
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
cbuffer MetropolisControls : register(b1)
{
    float3 EmissionCenter;
    float EmissionRadius;
    uint EmitterCount, BootstrapCount, ChainCount, ChainSeed;
    uint WorkBegin, WorkCount;
    float LargeStepProbability, MutationScale;
};
void IncrementCounter(inout uint2 value)
{
    uint previous = value.x;
    ++value.x;
    value.y += value.x < previous ? 1 : 0;
}
void MltFail(uint reason = MltNumericalFailure)
{
    uint previous;
    MltError.InterlockedOr(0, reason, previous);
}
bool MltFinite(double value)
{
    return value >= -MltDoubleMaximum && value <= MltDoubleMaximum;
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
#include "LightSelection.hlsl"
#include "LightEmission.hlsl"
#include "SurfaceEmission.hlsl"
#include "EmitterEndpoint.hlsl"
#include "CameraEndpoint.hlsl"
#include "FilmMoments.hlsl"
// These helpers only read primary. An inout formal retains its array storage
// through DXC inlining instead of materializing a complete vector per call.
float MltSample(inout MltPrimary primary, uint dimension)
{
    return float(primary.Values[dimension]) * MltUnitScale;
}
float2 MltPair(inout MltPrimary primary, uint dimension)
{
    return float2(MltSample(primary, dimension), MltSample(primary, dimension + 1));
}
EmissionDomain MltDomain()
{
    EmissionDomain domain;
    domain.Center = EmissionCenter;
    domain.Radius = EmissionRadius;
    return domain;
}
float3 MltOffset(float3 position, float3 normal, float3 direction)
{
    return position + normal * (dot(normal, direction) >= 0 ? RayEpsilon : -RayEpsilon);
}
RayData MltRay(float3 origin, float3 direction, float maximum = MaximumRayDistance)
{
    RayData ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.MinimumDistance = RayEpsilon;
    ray.MaximumDistance = maximum;
    return ray;
}
float MltLinear(uint channel)
{
    static const float ByteMaximum = 255, Threshold = 0.04045f, LinearScale = 12.92f;
    static const float Offset = 0.055f, Scale = 1.055f, Exponent = 2.4f;
    float encoded = channel / ByteMaximum;
    return encoded <= Threshold ? encoded / LinearScale : pow((encoded + Offset) / Scale, Exponent);
}
float3 MltTexel(TextureRecord image, int2 coordinate)
{
    int2 size = int2(image.Width, image.Height);
    coordinate = (coordinate % size + size) % size;
    uint packed = TextureTexels[image.OffsetTexels + coordinate.y * image.Width + coordinate.x];
    static const uint ByteMask = 0xff, GreenShift = 8, BlueShift = 16;
    return float3(MltLinear(packed & ByteMask), MltLinear((packed >> GreenShift) & ByteMask),
                  MltLinear((packed >> BlueShift) & ByteMask));
}
float3 MltColor(HitData hit)
{
    uint imageIndex = Materials[hit.MaterialIndex].DiffuseTextureIndex;
    if (!hit.HasTexCoord || imageIndex >= TextureCount || !all(isfinite(hit.TexCoord)))
        return hit.Color;
    TextureRecord image = Textures[imageIndex];
    if (!image.Width || !image.Height || image.OffsetTexels > TextureTexelCount ||
        image.Height > (TextureTexelCount - image.OffsetTexels) / image.Width)
        return hit.Color;
    float2 texel =
        frac(float2(hit.TexCoord.x, 1 - hit.TexCoord.y)) * float2(image.Width, image.Height) - 0.5f;
    int2 corner = int2(floor(texel));
    float2 fraction = frac(texel);
    float3 first = lerp(MltTexel(image, corner), MltTexel(image, corner + int2(1, 0)), fraction.x);
    float3 second =
        lerp(MltTexel(image, corner + int2(0, 1)), MltTexel(image, corner + int2(1, 1)), fraction.x);
    return hit.Color * lerp(first, second, fraction.y);
}
uint MltFindSource(uint primitive, bool sky)
{
    for (uint index = 0; index < EmitterCount; ++index)
        if (sky ? Emitters[index].Kind == PackedEmitterEnvironment
                : primitive != InvalidPrimitiveIndex && Emitters[index].Primitive == primitive)
            return index;
    return InvalidPrimitiveIndex;
}
float3 MltEmission(PackedEmitter source, float3 outward)
{
    if (source.Kind == PackedEmitterEnvironment)
        return lerp(source.BackgroundBottom, source.BackgroundTop, (1 - outward.y) / 2);
    if (source.Kind == PackedEmitterRectangle)
        return dot(source.Direction, outward) > 0 ? source.Strength / (source.Width * source.Height) : 0;
    return source.Strength;
}
bool MltRectangle(PackedEmitter light, RayData ray, inout float closest)
{
    if (light.Kind != PackedEmitterRectangle)
        return false;
    float denominator = dot(light.Direction, ray.Direction);
    if (denominator == 0)
        return false;
    float distance = dot(light.Direction, light.Origin - ray.Origin) / denominator;
    if (!(distance > ray.MinimumDistance && distance < ray.MaximumDistance && distance <= closest))
        return false;
    float3 helper = abs(light.Direction.y) < BasisParallelThreshold ? float3(0, 1, 0) : float3(1, 0, 0);
    float3 right = normalize(cross(helper, light.Direction)), up = cross(light.Direction, right);
    float3 offset = ray.Origin + ray.Direction * distance - light.Origin;
    if (abs(dot(offset, right)) > light.Width / 2 || abs(dot(offset, up)) > light.Height / 2)
        return false;
    closest = distance;
    return true;
}
bool MltVisible(RayData ray, inout MltStatistics statistics)
{
    IncrementCounter(statistics.Shadow);
    HitData hit = (HitData)0;
    hit.Distance = ray.MaximumDistance;
    if (IntersectScene(ray, true, hit, statistics.Tests))
        return false;
    float closest = ray.MaximumDistance;
    for (uint index = 0; index < EmitterCount; ++index)
        if (Emitters[index].Kind == PackedEmitterRectangle)
        {
            IncrementCounter(statistics.Tests);
            if (MltRectangle(Emitters[index], ray, closest))
                return false;
        }
    return true;
}
float MltMis(float sampled, float alternative)
{
    static const uint MisNone = 0, MisBalance = 1;
    uint heuristic = (TransportFlags >> MisShift) & MisMask;
    if (heuristic == MisNone)
        return 1;
    if (!(sampled > 0))
        return 0;
    if (!(alternative > 0))
        return 1;
    float maximum = max(sampled, alternative);
    float2 terms = float2(sampled, alternative) / maximum;
    if (heuristic != MisBalance)
        terms *= terms;
    return terms.x / (terms.x + terms.y);
}
float MltEmissionMis(uint source, float3 normal, float3 outward, float3 previous, float3 position, bool sky,
                     bool previousDelta, float previousPdf)
{
    if (previousDelta || !(TransportFlags & NeeBit))
        return 1;
    if (((TransportFlags >> MisShift) & MisMask) == 0)
        return 0;
    EmitterEndpointDensity density =
        EmitterEndpointPdf(Emitters, EmitterCount, MltDomain(), source, normal, outward);
    float pdf = density.SelectionProbability * density.Direction;
    if (!sky)
    {
        float cosine = abs(dot(normal, outward));
        float3 segment = position - previous;
        pdf =
            cosine > 0 ? density.SelectionProbability * density.Position * dot(segment, segment) / cosine : 0;
    }
    return MltMis(previousPdf, pdf);
}
float3 MltDirect(inout MltPrimary primary, uint dimension, SurfaceBsdf surface, float3 position,
                 float continuousFraction, float currentInterior, inout MltStatistics statistics)
{
    EmitterEndpoint endpoint = SampleEmitterEndpoint(
        Emitters, EmitterCount, MltDomain(), MltSample(primary, dimension + LightChoice),
        MltPair(primary, dimension + LightPosition), MltPair(primary, dimension + LightDirection));
    EmissionSample sample = endpoint.Emission;
    if (!(sample.Flags & EmissionValid))
        return 0;
    bool infinite = (sample.Flags & EmissionInfinite) != 0,
         deltaPosition = (sample.Flags & EmissionDeltaPosition) != 0;
    bool delta = (sample.Flags & (EmissionDeltaPosition | EmissionDeltaDirection)) != 0;
    if (!delta && !(TransportFlags & NeeBit))
        return 0;
    float3 direction = -sample.Direction;
    float distance = MaximumRayDistance, geometry = 1;
    if (!infinite)
    {
        float3 separation = sample.Origin - position;
        distance = length(separation);
        if (!(distance > 0) || !isfinite(distance))
            return 0;
        direction = separation / distance;
        geometry = (deltaPosition ? 1 : abs(dot(sample.Normal, direction))) / distance / distance;
    }
    SurfaceEvaluation evaluation = EvaluateSurface(surface, direction);
    float density = endpoint.SelectionProbability * (infinite ? sample.DirectionPdf : sample.PositionPdf);
    if (!(density > 0))
        return 0;
    float angularPdf = infinite || delta ? density : geometry > 0 ? density / geometry : 0;
    float weight = delta ? 1 : MltMis(angularPdf, evaluation.Pdf * continuousFraction);
    float3 value = evaluation.Value * MltEmission(Emitters[endpoint.EmitterIndex], -direction) *
                   (continuousFraction * abs(dot(surface.Normal, direction)) * geometry * weight / density);
    float connectionMeanFreePath = surface.Material.Optical.w > 0
        ? (dot(surface.GeometricNormal, direction) < 0 ? surface.Material.Optical.w : 0)
        : currentInterior;
    if (connectionMeanFreePath > 0) value *= exp(-distance / connectionMeanFreePath);
    if (!all(isfinite(value)) || !any(value > 0))
        return value;
    float3 origin = SurfaceOffset(surface, position, direction);
    if (!infinite)
    {
        float3 target = deltaPosition ? sample.Origin : MltOffset(sample.Origin, sample.Normal, -direction);
        float3 separation = target - origin;
        distance = length(separation);
        if (!(distance > 2 * RayEpsilon))
            return 0;
        direction = separation / distance;
        distance -= RayEpsilon;
    }
    return MltVisible(MltRay(origin, direction, distance), statistics) ? value : 0;
}
float3 MltInteriorDirect(inout MltPrimary primary, uint dimension, float3 position,
                         float3 incoming, PackedMaterial interior, inout MltStatistics statistics)
{
    EmitterEndpoint endpoint = SampleEmitterEndpoint(Emitters, EmitterCount, MltDomain(),
        MltSample(primary, dimension + LightChoice), MltPair(primary, dimension + LightPosition),
        MltPair(primary, dimension + LightDirection));
    EmissionSample sample = endpoint.Emission;
    if (!(sample.Flags & EmissionValid)) return 0;
    bool infinite = (sample.Flags & EmissionInfinite) != 0;
    bool deltaPosition = (sample.Flags & EmissionDeltaPosition) != 0;
    bool delta = (sample.Flags & (EmissionDeltaPosition | EmissionDeltaDirection)) != 0;
    if (!delta && !(TransportFlags & NeeBit)) return 0;
    float3 direction = -sample.Direction;
    float distance = MaximumRayDistance, geometry = 1;
    if (!infinite)
    {
        float3 segment = sample.Origin - position;
        float squared = dot(segment, segment);
        if (!(squared > 0) || !isfinite(squared)) return 0;
        distance = sqrt(squared);
        direction = segment / distance;
        geometry = (deltaPosition ? 1 : abs(dot(sample.Normal, direction))) / squared;
    }
    float density = endpoint.SelectionProbability * (infinite ? sample.DirectionPdf : sample.PositionPdf);
    if (!(density > 0) || !(geometry > 0)) return 0;
    float sampledPdf = delta ? density : infinite ? density : density / geometry;
    float phase = SubsurfacePhase(dot(incoming, direction), interior.ConductorIndex.w);
    float phasePdf = (TransportFlags & ImportanceBit) != 0 ? phase : 1 / (4 * Pi);
    float mis = delta ? 1 : MltMis(sampledPdf, phasePdf);
    float3 value = MltEmission(Emitters[endpoint.EmitterIndex], -direction)
        * (phase * geometry * mis / density * exp(-distance / interior.Optical.w));
    if (!all(isfinite(value)) || !any(value > 0)) return value;
    if (!infinite)
    {
        float3 target = deltaPosition ? sample.Origin : MltOffset(sample.Origin, sample.Normal, -direction);
        float3 segment = target - position;
        distance = length(segment);
        if (!(distance > RayEpsilon)) return 0;
        direction = segment / distance;
        distance -= RayEpsilon;
    }
    return MltVisible(MltRay(position, direction, distance), statistics) ? value : 0;
}
MltPath MltEvaluate(inout MltPrimary primary, inout MltStatistics statistics)
{
    CameraEndpointFrame camera;
    camera.Position = CameraPosition;
    camera.Forward = CameraForward;
    camera.Right = CameraRight;
    camera.Up = CameraUp;
    camera.HalfWidth = HalfWidth;
    camera.HalfHeight = HalfHeight;
    camera.LensRadius = LensRadius;
    camera.FocusDistance = FocusDistance;
    CameraEndpointSample sensor = SampleCameraEndpoint(camera, MltPair(primary, 0), MltPair(primary, 2));
    MltPath result = (MltPath)0;
    result.Raster = sensor.Raster;
    if (!(sensor.Flags & CameraEndpointValid))
    {
        MltFail();
        return result;
    }
    RayData ray = MltRay(sensor.Origin, sensor.Direction);
    float3 beta = 1, previous = ray.Origin;
    float previousPdf = 0, etaScale = 1;
    bool previousDelta = true;
    uint interiorMaterial = InvalidPrimitiveIndex;
    [loop] for (uint depth = 0; depth <= MaxReflectionBounces + 1; ++depth)
    {
        uint dimension = MltCameraDimensions + depth * MltVertexDimensions;
        if (depth == 0)
            IncrementCounter(statistics.Primary);
        else
            IncrementCounter(statistics.Reflection);
        HitData hit = (HitData)0;
        hit.Distance = ray.MaximumDistance;
        bool found = IntersectScene(ray, false, hit, statistics.Tests);
        float nearest = found ? hit.Distance + RayEpsilon : ray.MaximumDistance;
        uint rectangle = InvalidPrimitiveIndex;
        for (uint index = 0; index < EmitterCount; ++index)
            if (Emitters[index].Kind == PackedEmitterRectangle)
            {
                IncrementCounter(statistics.Tests);
                if (MltRectangle(Emitters[index], ray, nearest))
                    rectangle = index;
            }
        if (found && hit.MaterialIndex >= MaterialCount)
        {
            MltFail(MltInvalidConfiguration);
            return result;
        }
        if (depth == 0 && found && hit.FrontFace == 0 && Materials[hit.MaterialIndex].Optical.w > 0)
            interiorMaterial = hit.MaterialIndex;
        if (interiorMaterial != InvalidPrimitiveIndex)
        {
            PackedMaterial interior = Materials[interiorMaterial];
            float endpoint = rectangle != InvalidPrimitiveIndex ? nearest
                : found ? hit.Distance : ray.MaximumDistance;
            SubsurfaceFlight flight = SubsurfaceFreeFlight(interior.Optical.w,
                interior.AlbedoReflectivity.xyz * interior.ConductorExtinction.w,
                ray.Direction, endpoint, MltSample(primary, dimension + InteriorFreeFlight));
            beta *= flight.Weight;
            if (!all(isfinite(beta))) { MltFail(); return result; }
            if (!any(beta > 0)) break;
            if (flight.Scattered != 0)
            {
                if (depth > MaxReflectionBounces) break;
                float3 position = ray.Origin + ray.Direction * flight.Distance;
                result.Radiance += beta * MltInteriorDirect(primary, dimension, position,
                    ray.Direction, interior, statistics);
                SubsurfaceDirection phase = SubsurfaceSampleDirection(ray.Direction,
                    interior.ConductorIndex.w, MltPair(primary, dimension + InteriorPhaseDirection),
                    (TransportFlags & ImportanceBit) != 0);
                if (!(phase.Pdf > 0)) break;
                beta *= phase.Weight;
                if (!all(isfinite(beta))) { MltFail(); return result; }
                if (!any(beta > 0)) break;
                if ((TransportFlags & RouletteBit) && depth + 1 >= RouletteStartDepth)
                {
                    static const float MinimumSurvival = 0.05f, MaximumSurvival = 0.95f;
                    float survival = clamp(max(beta.x, max(beta.y, beta.z)) * etaScale,
                        MinimumSurvival, MaximumSurvival);
                    if (MltSample(primary, dimension + RouletteChoice) >= survival) break;
                    beta /= survival;
                }
                previous = position;
                previousPdf = phase.Pdf;
                previousDelta = false;
                ray = MltRay(position, phase.Direction);
                continue;
            }
        }
        bool sky = !found && rectangle == InvalidPrimitiveIndex;
        uint source = rectangle != InvalidPrimitiveIndex ? rectangle : MltFindSource(hit.PrimitiveIndex, sky);
        float3 position =
            ray.Origin + ray.Direction * (rectangle != InvalidPrimitiveIndex ? nearest : hit.Distance);
        if (source != InvalidPrimitiveIndex)
        {
            float3 normal =
                rectangle != InvalidPrimitiveIndex ? Emitters[source].Direction : hit.GeometricNormal;
            float weight = MltEmissionMis(source, normal, -ray.Direction, previous, position, sky,
                                          previousDelta, previousPdf);
            result.Radiance += beta * MltEmission(Emitters[source], -ray.Direction) * weight;
        }
        if (sky || rectangle != InvalidPrimitiveIndex || depth > MaxReflectionBounces)
            break;
        PackedMaterial material = Materials[hit.MaterialIndex];
        float mirror = material.Model == MaterialLegacy ? saturate(material.AlbedoReflectivity.w) : 0;
        if (material.Model == MaterialLegacy)
            material.Model = MaterialDiffuse;
        SurfaceBsdf surface = MakeSurfaceBsdf(material, hit, -ray.Direction, MltColor(hit));
        result.Radiance += beta * MltDirect(primary, dimension, surface, position, 1 - mirror,
            interiorMaterial == InvalidPrimitiveIndex ? 0 : Materials[interiorMaterial].Optical.w, statistics);
        SurfaceSample scattered = (SurfaceSample)0;
        float lobe = MltSample(primary, dimension + ScatterLobe);
        if (lobe < mirror)
        {
            scattered.Direction = normalize(reflect(ray.Direction, surface.Normal));
            if (!SurfaceHemisphere(surface, scattered.Direction, false))
                break;
            scattered.Weight = 1;
            scattered.Pdf = mirror;
            scattered.RelativeIndex = 1;
            scattered.Event = ScatteringDelta | ScatteringReflection;
        }
        else
        {
            scattered = SampleSurface(surface, MltPair(primary, dimension + ScatterDirection),
                                      mirror > 0 ? (lobe - mirror) / (1 - mirror) : lobe);
            scattered.Pdf *= 1 - mirror;
        }
        if (!scattered.Event || !(scattered.Pdf > 0))
            break;
        beta *= scattered.Weight;
        etaScale *= scattered.RelativeIndex * scattered.RelativeIndex;
        if (!all(isfinite(beta)) || !isfinite(etaScale))
        {
            MltFail();
            return result;
        }
        if (!any(beta > 0))
            break;
        if ((TransportFlags & RouletteBit) && depth + 1 >= RouletteStartDepth)
        {
            static const float MinimumSurvival = 0.05f, MaximumSurvival = 0.95f;
            float survival =
                clamp(max(beta.x, max(beta.y, beta.z)) * etaScale, MinimumSurvival, MaximumSurvival);
            if (MltSample(primary, dimension + RouletteChoice) >= survival)
                break;
            beta /= survival;
        }
        previous = position;
        previousPdf = scattered.Pdf;
        previousDelta = (scattered.Event & ScatteringDelta) != 0;
        ray = MltRay(SurfaceOffset(surface, position, scattered.Direction), scattered.Direction);
        if (material.Optical.w > 0 && (scattered.Event & ScatteringTransmission) != 0)
            interiorMaterial = hit.FrontFace != 0 ? hit.MaterialIndex : InvalidPrimitiveIndex;
    }
    static const double RedImportance = 0.2126L, GreenImportance = 0.7152L, BlueImportance = 0.0722L;
    result.Importance = double(result.Radiance.x) * RedImportance +
                        double(result.Radiance.y) * GreenImportance +
                        double(result.Radiance.z) * BlueImportance;
    if (!all(isfinite(result.Radiance)) || any(result.Radiance < 0) || !MltFinite(result.Importance) ||
        result.Importance < 0)
        MltFail();
    return result;
}
void MltAtomicChannel(uint address, float contribution)
{
    uint expected = Pass.Load(address);
    [allow_uav_condition] while (true)
    {
        float sum = asfloat(expected) + contribution;
        if (!isfinite(sum))
        {
            MltFail();
            return;
        }
        uint actual;
        Pass.InterlockedCompareExchange(address, expected, asuint(sum), actual);
        if (actual == expected)
            return;
        expected = actual;
    }
}
void MltSplat(MltPath path, double probability, double normalization)
{
    if (!(probability > 0 && path.Importance > 0))
        return;
    uint2 raster = uint2(path.Raster * float2(FullWidth, FullHeight));
    if (raster.x < TileX || raster.y < TileY || raster.x - TileX >= Width || raster.y - TileY >= Height)
        return;
    double scale = probability * normalization / path.Importance;
    float3 value = float3(scale * double(path.Radiance.x), scale * double(path.Radiance.y),
                          scale * double(path.Radiance.z));
    if (!all(isfinite(value)) || any(value < 0))
    {
        MltFail();
        return;
    }
    uint address = ((raster.y - TileY) * Width + raster.x - TileX) * FilmStride;
    MltAtomicChannel(address, value.x);
    MltAtomicChannel(address + ScalarBytes, value.y);
    MltAtomicChannel(address + 2 * ScalarBytes, value.z);
}
[numthreads(MltGroupSize, 1, 1)] void MetropolisBootstrapKernel(uint3 id : SV_DispatchThreadID) {
    if (id.x >= WorkCount || WorkBegin + id.x >= BootstrapCount)
        return;
    MltStatistics statistics = (MltStatistics)0;
    MltPrimary primary = MltIndependent(MltSeed(ChainSeed, WorkBegin + id.x, MltBootstrapDomain));
    Bootstrap[WorkBegin + id.x] = MltEvaluate(primary, statistics);
    Statistics[id.x] = statistics;
}[numthreads(1, 1, 1)] void MetropolisDistributionKernel(uint3 id : SV_DispatchThreadID)
{
    if (any(id))
        return;
    double total = 0;
    for (uint index = 0; index < BootstrapCount; ++index)
    {
        double importance = Bootstrap[index].Importance;
        if (!(importance >= 0) || !MltFinite(importance))
            MltFail();
        total += importance;
        Cumulative[index] = total;
    }
    if (!BootstrapCount || !MltFinite(total))
        MltFail();
    Distribution[0] = double2(BootstrapCount ? total / BootstrapCount : 0, total);
}
[numthreads(MltGroupSize, 1, 1)] void MetropolisInitializeKernel(uint3 id : SV_DispatchThreadID) {
    if (id.x >= ChainCount)
        return;
    MltChain chain = (MltChain)0;
    chain.Random = MltNext(MltSeed(ChainSeed, id.x, MltChainDomain));
    double total = Distribution[0].y;
    if (total > 0)
    {
        double target = double(MltUniform(chain.Random.w)) * total;
        uint left = 0, right = BootstrapCount;
        while (left < right)
        {
            uint middle = left + (right - left) / 2;
            if (Cumulative[middle] <= target)
                left = middle + 1;
            else
                right = middle;
        }
        chain.Current = Bootstrap[left];
        PrimaryStates[id.x] = MltIndependent(MltSeed(ChainSeed, left, MltBootstrapDomain));
    }
    Chains[id.x] = chain;
}[numthreads(MltGroupSize, 1, 1)] void MetropolisMutateKernel(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= ChainCount)
        return;
    MltStatistics statistics = (MltStatistics)0;
    double normalization = Distribution[0].x;
    if (normalization > 0)
    {
        MltChain chain = Chains[id.x];
        MltPrimary primary = PrimaryStates[id.x];
        uint pixels = FullWidth * FullHeight,
             quota = pixels / ChainCount + (id.x < pixels % ChainCount ? 1 : 0);
        [loop] for (uint step = WorkBegin; step < quota && step - WorkBegin < WorkCount; ++step)
        {
            chain.Random = MltNext(chain.Random);
            bool large = MltUniform(chain.Random.w) < LargeStepProbability;
            MltProposal proposal = MltPropose(primary, chain.Random, large, MutationScale);
            MltPath value = MltEvaluate(proposal.Primary, statistics);
            double ratio = chain.Current.Importance > 0 ? value.Importance / chain.Current.Importance : 1;
            double acceptance = ratio < 1 ? ratio : 1;
            MltSplat(chain.Current, 1 - acceptance, normalization);
            MltSplat(value, acceptance, normalization);
            IncrementCounter(chain.Proposals);
            if (large)
                IncrementCounter(chain.LargeSteps);
            chain.Random = MltNext(proposal.Random);
            if (double(MltUniform(chain.Random.w)) < acceptance)
            {
                chain.Current = value;
                primary = proposal.Primary;
                IncrementCounter(chain.Accepted);
            }
            else if (value.Importance > 0)
                IncrementCounter(chain.RejectedPositive);
        }
        PrimaryStates[id.x] = primary;
        Chains[id.x] = chain;
    }
    Statistics[id.x] = statistics;
}
[numthreads(MltFilmGroupWidth, MltFilmGroupWidth, 1)] void MetropolisClearKernel(
    uint3 id : SV_DispatchThreadID) {
    if (id.x >= Width || id.y >= Height)
        return;
    uint pixel = id.y * Width + id.x;
    Pass.Store4(pixel * FilmStride, uint4(0, 0, 0, 0));
    if (pixel == 0)
    {
        MltError.Store(0, 0);
        if (InstanceNodeCount)
            InstanceError.Store(0, 0);
    }
}[numthreads(MltFilmGroupWidth, MltFilmGroupWidth,
             1)] void MetropolisCommitKernel(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= Width || id.y >= Height)
        return;
    uint pixel = id.y * Width + id.x;
    float3 value = asfloat(Pass.Load3(pixel * FilmStride));
    if (!all(isfinite(value)) || any(value < 0))
    {
        MltFail();
        return;
    }
    float4 moments = Accumulation[pixel];
    uint count = SampleCounts[pixel];
    AccumulateFilmSample(moments, count, value);
    if (!all(isfinite(moments)))
    {
        MltFail();
        return;
    }
    Accumulation[pixel] = moments;
    SampleCounts[pixel] = count;
    SampleTargets[pixel] = TotalSamples;
}
