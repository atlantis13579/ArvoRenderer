// Independent DXR offline point-based GI; preview skips the surfel bake.
// Closed dielectric solids use nonlocal interior random walks at bake and camera time.
// Christensen (2008), "Point-Based Approximate Color Bleeding":
// https://graphics.pixar.com/library/PointBasedColorBleeding/paper.pdf
// This variant traces nearest equal-area disks instead of approximating distant
// clusters. Disks cache two-sided reflected radiance; source emission is handled
// by direct sampling. Jacobi iteration keeps reflection orders disjoint. Finite
// disks introduce geometric bias, which more camera samples cannot eliminate.
// Each ray-generation stage processes bounded work; a complete order is fenced
// before swapping its read/write buffers. Film moments are conditional on the
// fixed bake, so this implementation does not expose adaptive sample stopping.
static const float Pi = 3.14159265358979323846f, RayEpsilon = 1e-4f;
static const float MaximumRayDistance = 1e30f, BasisParallelThreshold = 0.999f;
static const double PointPi = 3.1415926535897932384626433832795L;
static const double PointDoubleMaximum = 1.7976931348623157e308L;
static const uint PointInvalid = 0xffffffffu, PointAlgorithm = 8, PointMaximumDepth = 32;
static const uint PointInitialize = 0, PointBake = 1, PointCamera = 2;
static const uint PointWorkLimit = 1024, PointSampleLimit = 32, PointStackSize = 32;
static const uint PointArithmeticError = 1, PointControlError = 2, PointBoundsError = 4;
static const uint ImportanceBit = 1, NeeBit = 2;
static const uint PixelSeedMultiplier = 9781, SampleSeedMultiplier = 6271,
                  SecondDimensionScramble = 0x68bc21ebu;
static const uint PointSelectDimension = 0x53454c45u, PointEndpointDimension = 0x454e4450u,
                  PointGatherDimension = 0x47415448u, PointBakeDomain = 0x42414b45u;
static const uint PointInterfaceDimension = 0x53534946u, PointBranchDimension = 0x53534252u;
static const uint PointFlightDimension = 0x5353464cu, PointPhaseDimension = 0x53535048u;
static const uint PointBounceScramble = 0x9e3779b9u;
#include "Surface.hlsl"
#include "Globals.hlsl"
#include "MaterialTypes.hlsl"
#include "InstancePrograms.hlsli"
#include "Triangle.hlsl"
#include "Sphere.hlsl"
#include "TriangleIntersection.hlsl"
#include "MotionTriangle.hlsl"
#include "Select.hlsl"
#include "FilmMoments.hlsl"
#include "EmitterData.hlsl"
struct PointSurface
{
    float3 Position; uint Material;
    float3 Geometry; uint Primitive;
    float3 Shading; uint HasTexCoord;
    float3 Color; uint Patch;
    float2 TexCoord; uint2 Reserved;
    double RadiusSquared, Area;
};
struct PointNode
{
    double3 Minimum; uint Left, Right;
    double3 Maximum; uint Begin, Count;
};
struct PointRadiance { double3 Front, Back; };
struct PointTexture { uint Offset, Width, Height, Reserved; };
struct PointCounters { uint2 Primary, Shadow, Reflection; };
struct PointReceiver { float3 Position, Normal, Geometry; double3 Reflectance; };
struct PointRay { float3 Origin, Direction; float Minimum, Maximum; };
StructuredBuffer<MaterialData> Materials : register(t3);
StructuredBuffer<PointTexture> Textures : register(t5);
StructuredBuffer<uint> TextureTexels : register(t6);
StructuredBuffer<PackedEmitter> Emitters : register(t13);
StructuredBuffer<PointSurface> PointSurfaces : register(t14);
StructuredBuffer<PointNode> PointHierarchy : register(t15);
StructuredBuffer<uint> PointIndices : register(t16);
RWStructuredBuffer<uint> OutputPixels : register(u0);
RWStructuredBuffer<uint> Statistics : register(u1);
RWStructuredBuffer<LinearFilmPixel> LinearFilm : register(u2);
RWStructuredBuffer<PointRadiance> PointPrevious : register(u5);
RWStructuredBuffer<PointRadiance> PointNext : register(u6);
RWStructuredBuffer<PointRadiance> PointTotal : register(u7);
RWByteAddressBuffer PointErrors : register(u8);
cbuffer PointBatch : register(b1)
{
    uint PointStage, PointBegin, PointCount, PointOrder;
    uint PointSurfaceCount, PointNodeCount, PointSourceCount, PointSeed;
    uint PointDirectSamples, PointGatherSamples, PointSampleBegin, PointSampleCount;
};
#include "Statistics.hlsl"
uint Hash(uint value)
{
    static const uint AvalancheFirst = 0x7feb352du, AvalancheSecond = 0x846ca68bu;
    static const uint OuterShift = 16, InnerShift = 15;
    value ^= value >> OuterShift; value *= AvalancheFirst;
    value ^= value >> InnerShift; value *= AvalancheSecond;
    return value ^ (value >> OuterShift);
}
float Random01(uint seed)
{
    static const uint FractionShift = 8;
    static const float FractionRange = 16777216.0f;
    return float(Hash(seed) >> FractionShift) / FractionRange;
}
#include "Sequence.hlsl"
#include "ThinLens.hlsl"
#include "Importance.hlsl"
#include "LightSelection.hlsl"
#include "LightEmission.hlsl"
#include "SurfaceEmission.hlsl"
#include "BsdfTypes.hlsl"
#include "Fresnel.hlsl"
#include "Ggx.hlsl"
#include "Diffuse.hlsl"
#include "Conductor.hlsl"
#include "Dielectric.hlsl"
#include "SurfaceScattering.hlsl"
#include "SubsurfaceScattering.hlsl"
void PointFail(uint kind) { uint ignored; PointErrors.InterlockedOr(0, kind, ignored); }
double PointDot(double3 a, double3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
bool PointColorValid(double3 value) { return all(value >= 0) && all(value <= PointDoubleMaximum); }
bool PointWorkValid(uint population, uint samples)
{
    bool valid = Integrator == PointAlgorithm && MaxReflectionBounces <= PointMaximumDepth &&
        AdaptiveEnabled == 0 && ShadowStrength == 1 && (TransportFlags & ~ImportanceBit) == NeeBit &&
        PointCount <= PointWorkLimit && PointBegin <= population && PointCount <= population - PointBegin &&
        PointSampleCount > 0 && PointSampleCount <= PointSampleLimit && PointSampleBegin < samples &&
        PointSampleCount <= samples - PointSampleBegin && PointSourceCount <= UniformEmitterLimit;
    if (!valid) PointFail(PointControlError);
    return valid;
}
float PointSrgb(uint channel)
{
    static const float ByteMaximum = 255, Cutoff = 0.04045f, LinearDivisor = 12.92f;
    static const float Offset = 0.055f, Scale = 1.055f, Exponent = 2.4f;
    float x = float(channel) / ByteMaximum;
    return x <= Cutoff ? x / LinearDivisor : pow((x + Offset) / Scale, Exponent);
}
float3 PointTexturePixel(PointTexture texture, int x, int y)
{
    int width = int(texture.Width), height = int(texture.Height);
    x = (x % width + width) % width; y = (y % height + height) % height;
    uint packed = TextureTexels[texture.Offset + uint(y) * texture.Width + uint(x)];
    static const uint ByteMask = 0xff, GreenShift = 8, BlueShift = 16;
    return float3(PointSrgb(packed & ByteMask), PointSrgb((packed >> GreenShift) & ByteMask),
                  PointSrgb((packed >> BlueShift) & ByteMask));
}
double3 PointAlbedo(uint materialIndex, float3 color, float2 uv, bool textured)
{
    if (materialIndex >= MaterialCount) { PointFail(PointBoundsError); return 0; }
    MaterialData material = Materials[materialIndex];
    if (material.SampledEmitter != 0) return 0;
    if (material.Model == MaterialDielectric && material.Optical.w > 0) return 0;
    bool diffuse = material.Model == MaterialDiffuse || (material.Model == MaterialLegacy &&
                    material.AlbedoReflectivity.w == 0 && all(material.Specular.xyz == 0));
    if (!diffuse) { PointFail(PointControlError); return 0; }
    if (textured && material.DiffuseTextureIndex < TextureCount && all(isfinite(uv)))
    {
        PointTexture texture = Textures[material.DiffuseTextureIndex];
        if (texture.Width > 0 && texture.Height > 0 && texture.Offset <= TextureTexelCount &&
            texture.Height <= (TextureTexelCount - texture.Offset) / texture.Width)
        {
            float2 location = frac(float2(uv.x, 1 - uv.y)) * float2(texture.Width, texture.Height) - 0.5f;
            int2 corner = int2(floor(location)); float2 weight = frac(location);
            float3 lower = PointTexturePixel(texture, corner.x, corner.y) * (1 - weight.x) +
                           PointTexturePixel(texture, corner.x + 1, corner.y) * weight.x;
            float3 upper = PointTexturePixel(texture, corner.x, corner.y + 1) * (1 - weight.x) +
                           PointTexturePixel(texture, corner.x + 1, corner.y + 1) * weight.x;
            color *= lower * (1 - weight.y) + upper * weight.y;
        }
    }
    double3 value = double3(color) * double3(material.AlbedoReflectivity.xyz);
    if (!PointColorValid(value)) { PointFail(PointArithmeticError); return 0; }
    return clamp(value, 0.0L, 1.0L);
}
PointRay PointSpawn(PointReceiver receiver, float3 direction)
{
    PointRay ray;
    ray.Origin = receiver.Position + receiver.Geometry * RayEpsilon; ray.Direction = direction;
    ray.Minimum = RayEpsilon; ray.Maximum = MaximumRayDistance;
    return ray;
}
uint PointRectangle(PointRay ray, inout double nearest, out bool front)
{
    uint identity = PointInvalid; front = false;
    [loop] for (uint source = 0; source < PointSourceCount; ++source)
    {
        PackedEmitter light = Emitters[source];
        if (light.Kind != PackedEmitterRectangle) continue;
        double denominator = PointDot(double3(ray.Direction), double3(light.Direction));
        if (denominator == 0) continue;
        double distance = PointDot(double3(light.Origin) - double3(ray.Origin), double3(light.Direction)) / denominator;
        if (!(distance >= ray.Minimum && distance <= nearest && distance <= ray.Maximum)) continue;
        double3 offset = double3(ray.Origin) + double3(ray.Direction) * distance - double3(light.Origin);
        LightEmissionAxes axes = EmissionAxes(normalize(light.Direction));
        if (abs(PointDot(offset, double3(axes.Right))) > double(light.Width) / 2 ||
            abs(PointDot(offset, double3(axes.Up))) > double(light.Height) / 2) continue;
        if (distance == nearest && identity != PointInvalid) continue;
        identity = source; nearest = distance; front = denominator < 0;
    }
    return identity;
}
bool PointIntersectsNode(PointNode node, PointRay ray, double closest)
{
    double lower = ray.Minimum, upper = min(closest, double(ray.Maximum));
    [unroll] for (uint component = 0; component < 3; ++component)
    {
        double direction = ray.Direction[component], origin = ray.Origin[component];
        if (direction == 0)
        { if (origin < node.Minimum[component] || origin > node.Maximum[component]) return false; }
        else
        {
            double a = (node.Minimum[component] - origin) / direction;
            double b = (node.Maximum[component] - origin) / direction;
            lower = max(lower, min(a, b)); upper = min(upper, max(a, b));
            if (lower > upper) return false;
        }
    }
    return true;
}
uint PointNearestDisk(PointRay ray, uint ignored, out bool front, out double distance)
{
    distance = ray.Maximum; front = false;
    if (PointNodeCount == 0) return PointInvalid;
    uint stack[PointStackSize], pending = 1, visited = 0, nearest = PointInvalid;
    stack[0] = 0;
    [loop] while (pending > 0)
    {
        uint index = stack[--pending];
        if (index >= PointNodeCount || ++visited > PointNodeCount)
        { PointFail(PointBoundsError); return PointInvalid; }
        PointNode node = PointHierarchy[index];
        if (!PointIntersectsNode(node, ray, distance)) continue;
        if (node.Count == 0)
        {
            if (pending + 2 > PointStackSize || node.Left <= index || node.Right <= index)
            { PointFail(PointBoundsError); return PointInvalid; }
            stack[pending++] = node.Right; stack[pending++] = node.Left;
            continue;
        }
        if (node.Begin > PointSurfaceCount || node.Count > PointSurfaceCount - node.Begin)
        { PointFail(PointBoundsError); return PointInvalid; }
        [loop] for (uint item = 0; item < node.Count; ++item)
        {
            uint identity = PointIndices[node.Begin + item];
            if (identity >= PointSurfaceCount) { PointFail(PointBoundsError); return PointInvalid; }
            if (identity == ignored) continue;
            PointSurface surface = PointSurfaces[identity];
            double3 normal = double3(surface.Geometry);
            double denominator = PointDot(normal, double3(ray.Direction));
            if (denominator == 0) continue;
            double candidate = PointDot(normal, double3(surface.Position) - double3(ray.Origin)) / denominator;
            if (!(candidate >= ray.Minimum && candidate <= distance)) continue;
            double3 residual = double3(ray.Origin) + double3(ray.Direction) * candidate - double3(surface.Position);
            if (PointDot(residual, residual) > surface.RadiusSquared) continue;
            if (candidate == distance && nearest != PointInvalid && identity > nearest) continue;
            distance = candidate; nearest = identity; front = denominator < 0;
        }
    }
    return nearest;
}
double3 PointDirectLight(PointReceiver receiver, SampleContext random, inout PointCounters counters)
{
    EmitterSelection selected = SelectEmitter(PointSourceCount, Sample2D(random, PointSelectDimension).x);
    if (selected.Index == InvalidEmitterSelection) return 0;
    PackedEmitter emitter = Emitters[selected.Index];
    float2 u = Sample2D(random, PointEndpointDimension);
    double3 radiance = 0;
    double probability = selected.Probability, areaDensity = 0;
    float3 direction = 0, endpoint = 0, emitterNormal = 0;
    bool finite = false, oneSided = false;
    switch (emitter.Kind)
    {
    case PackedEmitterDirectional:
        direction = normalize(emitter.Direction); radiance = double3(emitter.Strength); break;
    case PackedEmitterEnvironment:
        direction = SampleDiffuseDirection(receiver.Normal, u);
        probability *= double(DiffusePdf(dot(receiver.Normal, direction)));
        radiance = double3(lerp(emitter.BackgroundBottom, emitter.BackgroundTop, saturate((direction.y + 1) / 2)));
        break;
    case PackedEmitterPoint:
        finite = true; endpoint = emitter.Origin; radiance = double3(emitter.Strength); break;
    case PackedEmitterRectangle:
    {
        if (!(emitter.Width > 0 && emitter.Height > 0)) { PointFail(PointControlError); return 0; }
        emitterNormal = normalize(emitter.Direction);
        LightEmissionAxes axes = EmissionAxes(emitterNormal);
        endpoint = emitter.Origin + axes.Right * ((u.x - 0.5f) * emitter.Width) +
                   axes.Up * ((u.y - 0.5f) * emitter.Height);
        areaDensity = 1 / (double(emitter.Width) * double(emitter.Height));
        radiance = double3(emitter.Strength) * areaDensity; finite = true; oneSided = true;
        break;
    }
    case PackedEmitterTriangle:
    case PackedEmitterSphere:
    {
        SurfaceEmissionSource source;
        source.Kind = emitter.Kind == PackedEmitterTriangle ? SurfaceEmitterTriangle : SurfaceEmitterEllipsoid;
        source.Origin = emitter.Origin; source.AxisX = emitter.AxisX; source.AxisY = emitter.AxisY;
        source.AxisZ = emitter.AxisZ; source.Radiance = emitter.Strength;
        SurfaceEmissionSample sample = SampleSurfaceEmission(source, u, float2(0.25f, 0.5f));
        if ((sample.Flags & SurfaceEmitterValid) == 0) return 0;
        endpoint = sample.Origin; emitterNormal = sample.Normal; areaDensity = sample.PositionPdf;
        radiance = double3(sample.Value); finite = true;
        break;
    }
    default: PointFail(PointControlError); return 0;
    }
    if (finite)
    {
        float3 segment = endpoint - receiver.Position;
        float segmentLength = length(segment);
        if (!(segmentLength > RayEpsilon)) return 0;
        direction = segment / segmentLength;
        double distanceSquared = PointDot(double3(segment), double3(segment));
        if (emitter.Kind == PackedEmitterPoint) radiance /= distanceSquared;
        else
        {
            double sourceCosine = PointDot(double3(emitterNormal), -double3(direction));
            if (!oneSided) sourceCosine = abs(sourceCosine);
            if (!(sourceCosine > 0 && areaDensity > 0)) return 0;
            probability *= distanceSquared * areaDensity / sourceCosine;
        }
    }
    double receiverCosine = dot(receiver.Normal, direction);
    if (!(probability > 0 && receiverCosine > 0 && dot(receiver.Geometry, direction) > 0)) return 0;
    PointRay shadow = PointSpawn(receiver, direction);
    if (finite)
    {
        float3 segment = endpoint - shadow.Origin; float distance = length(segment);
        if (!(distance > 2 * RayEpsilon)) return 0;
        shadow.Direction = segment / distance; shadow.Maximum = distance - RayEpsilon;
    }
    IncrementLocalCounter(counters.Shadow);
    Payload blocker = CastRay(shadow.Origin, shadow.Direction, shadow.Minimum, shadow.Maximum);
    if ((blocker.Flags & SurfaceHitFlag) != 0) return 0;
    double nearest = shadow.Maximum; bool front;
    if (PointRectangle(shadow, nearest, front) != PointInvalid) return 0;
    return radiance * (receiverCosine / probability);
}
double3 PointIndirectLight(PointReceiver receiver, uint ignored, SampleContext random, bool camera,
                           inout PointCounters counters)
{
    float3 direction = SampleDiffuseDirection(receiver.Normal, Sample2D(random, PointGatherDimension));
    double cosine = dot(receiver.Normal, direction), pdf = DiffusePdf(float(cosine));
    if (!(pdf > 0 && dot(receiver.Geometry, direction) > 0)) return 0;
    PointRay ray = PointSpawn(receiver, direction);
    IncrementLocalCounter(counters.Reflection);
    bool front; double distance;
    uint identity = PointNearestDisk(ray, ignored, front, distance);
    if (identity == PointInvalid) return 0;
    bool rectangleFront;
    if (PointRectangle(ray, distance, rectangleFront) != PointInvalid) return 0;
    PointRadiance light;
    if (camera) light = PointTotal[identity]; else light = PointPrevious[identity];
    return (front ? light.Front : light.Back) * (cosine / pdf);
}
struct PointSubsurfaceExit
{
    PointReceiver Receiver;
    double3 Throughput;
    uint Reached;
};
PointSubsurfaceExit PointSubsurfaceWalk(PointRay ray, Payload entry, float3 entryPosition,
                                        bool cameraInside, SampleContext random,
                                        inout PointCounters counters)
{
    PointSubsurfaceExit result = (PointSubsurfaceExit)0;
    if (entry.MaterialIndex >= MaterialCount) { PointFail(PointBoundsError); return result; }
    const MaterialData material = Materials[entry.MaterialIndex];
    double3 throughput = double3(1, 1, 1);
    if (!cameraInside)
    {
        const SurfaceBsdf boundary = MakeSurfaceBsdf(material, entry, -ray.Direction,
                                                      entry.Color, TransportRadiance);
        const SurfaceSample launch = SampleSurface(boundary,
            Sample2D(random, PointInterfaceDimension), Sample2D(random, PointBranchDimension).x);
        if ((launch.Event & ScatteringTransmission) == 0 || !(launch.Pdf > 0)) return result;
        throughput *= double3(launch.Weight);
        ray.Origin = SurfaceOffset(boundary, entryPosition, launch.Direction);
        ray.Direction = launch.Direction;
        ray.Minimum = RayEpsilon; ray.Maximum = MaximumRayDistance;
    }
    [loop] for (uint bounce = 0; bounce < PointMaximumDepth; ++bounce)
    {
        IncrementLocalCounter(counters.Reflection);
        Payload hit = CastRay(ray.Origin, ray.Direction, ray.Minimum, ray.Maximum);
        if ((hit.Flags & SurfaceHitFlag) == 0 || hit.MaterialIndex != entry.MaterialIndex)
            return result;
        const uint dimension = (bounce + 1) * PointBounceScramble;
        const SubsurfaceFlight flight = SubsurfaceFreeFlight(material.Optical.w,
            material.AlbedoReflectivity.xyz * material.ConductorExtinction.w,
            ray.Direction, asfloat(hit.DistanceBits),
            Sample2D(random, dimension ^ PointFlightDimension).x);
        throughput *= double3(flight.Weight);
        if (!PointColorValid(throughput) || !any(throughput > 0)) return result;
        if (flight.Scattered != 0)
        {
            const SubsurfaceDirection phase = SubsurfaceSampleDirection(ray.Direction,
                material.ConductorIndex.w, Sample2D(random, dimension ^ PointPhaseDimension),
                (TransportFlags & ImportanceBit) != 0);
            if (!(phase.Pdf > 0)) return result;
            throughput *= double(phase.Weight);
            ray.Origin += ray.Direction * flight.Distance;
            ray.Direction = phase.Direction;
            continue;
        }
        const float3 position = ray.Origin + ray.Direction * asfloat(hit.DistanceBits);
        const SurfaceBsdf boundary = MakeSurfaceBsdf(material, hit, -ray.Direction,
                                                      hit.Color, TransportRadiance);
        const SurfaceSample event = SampleSurface(boundary,
            Sample2D(random, dimension ^ PointInterfaceDimension),
            Sample2D(random, dimension ^ PointBranchDimension).x);
        if (!event.Event || !(event.Pdf > 0)) return result;
        throughput *= double3(event.Weight);
        if (!PointColorValid(throughput) || !any(throughput > 0)) return result;
        if ((event.Event & ScatteringTransmission) == 0)
        {
            ray.Origin = SurfaceOffset(boundary, position, event.Direction);
            ray.Direction = event.Direction;
            continue;
        }
        PointReceiver receiver;
        receiver.Position = position;
        receiver.Geometry = normalize(hit.GeometricNormal);
        receiver.Normal = dot(hit.Normal, receiver.Geometry) >= 0 ?
            normalize(hit.Normal) : -normalize(hit.Normal);
        receiver.Reflectance = 0;
        result.Receiver = receiver;
        result.Throughput = throughput;
        result.Reached = 1;
        return result;
    }
    return result;
}
Payload PointSurfelHit(PointSurface surfel)
{
    Payload hit = (Payload)0;
    hit.Normal = surfel.Shading;
    hit.GeometricNormal = surfel.Geometry;
    hit.Color = surfel.Color;
    hit.MaterialIndex = surfel.Material;
    hit.PrimitiveIndex = surfel.Primitive;
    hit.TexCoord = surfel.TexCoord;
    hit.Flags = SurfaceHitFlag | SurfaceFrontFaceFlag |
        (surfel.HasTexCoord != 0 ? SurfaceTexCoordFlag : 0);
    return hit;
}
double3 PointSubsurfaceBake(PointSurface surfel, SampleContext random,
                             inout PointCounters counters)
{
    PointRay ray;
    ray.Origin = surfel.Position; ray.Direction = -surfel.Geometry;
    ray.Minimum = RayEpsilon; ray.Maximum = MaximumRayDistance;
    const PointSubsurfaceExit exit = PointSubsurfaceWalk(ray, PointSurfelHit(surfel), surfel.Position,
                                                         false, random, counters);
    if (exit.Reached == 0) return 0;
    const double3 incident = PointOrder == 0
        ? PointDirectLight(exit.Receiver, random, counters)
        : PointIndirectLight(exit.Receiver, PointInvalid, random, false, counters);
    return exit.Throughput * incident / PointPi;
}
void PointPublishCounters(PointCounters counters)
{
    AddStatistic(PrimaryCounterWord, counters.Primary);
    AddStatistic(ShadowCounterWord, counters.Shadow);
    AddStatistic(ReflectionCounterWord, counters.Reflection);
}
void PointBakeSurface(uint identity)
{
    PointSurface surface = PointSurfaces[identity];
    PointReceiver receiver;
    receiver.Position = surface.Position;
    receiver.Reflectance = PointAlbedo(surface.Material, surface.Color, surface.TexCoord, surface.HasTexCoord != 0);
    const bool subsurface = surface.Material < MaterialCount &&
        Materials[surface.Material].Model == MaterialDielectric &&
        Materials[surface.Material].Optical.w > 0;
    PointCounters counters = (PointCounters)0;
    PointRadiance increment = (PointRadiance)0;
    uint samples = PointOrder == 0 ? PointDirectSamples : PointGatherSamples;
    [unroll] for (uint side = 0; side < 2; ++side)
    {
        receiver.Normal = side == 0 ? surface.Shading : -surface.Shading;
        receiver.Geometry = side == 0 ? surface.Geometry : -surface.Geometry;
        double3 reflected = 0;
        if (any(receiver.Reflectance > 0) || (subsurface && side == 0))
        {
            [loop] for (uint i = PointSampleBegin; i < PointSampleBegin + PointSampleCount; ++i)
            {
                SampleContext random = MakeSampleContext(identity, side, PointSurfaceCount, i, samples, SamplingSequence, SamplingSeed);
                random.PixelSeed = Hash(random.PixelSeed ^ Hash(PointSeed) ^ Hash(PointOrder) ^ PointBakeDomain);
                if (subsurface)
                    reflected += PointSubsurfaceBake(surface, random, counters) / double(samples);
                else
                {
                    const double3 irradiance = PointOrder == 0
                        ? PointDirectLight(receiver, random, counters)
                        : PointIndirectLight(receiver, identity, random, false, counters);
                    reflected += (receiver.Reflectance * irradiance) / (PointPi * double(samples));
                }
            }
        }
        if (side == 0) increment.Front = reflected; else increment.Back = reflected;
    }
    PointRadiance next = increment, total;
    if (PointSampleBegin > 0)
    { PointRadiance old = PointNext[identity]; next.Front += old.Front; next.Back += old.Back; }
    if (PointOrder == 0) total = next;
    else
    { total = PointTotal[identity]; total.Front += increment.Front; total.Back += increment.Back; }
    if (!PointColorValid(next.Front) || !PointColorValid(next.Back) ||
        !PointColorValid(total.Front) || !PointColorValid(total.Back)) PointFail(PointArithmeticError);
    else { PointNext[identity] = next; PointTotal[identity] = total; }
    PointPublishCounters(counters);
}
double3 PointCameraRadiance(uint pixel, uint sampleIndex, inout PointCounters counters)
{
    uint2 coordinate = uint2(TileX + pixel % Width, TileY + pixel / Width);
    SampleContext random = MakeSampleContext(coordinate.x, coordinate.y, FullWidth, sampleIndex, SamplesPerPixel, SamplingSequence, SamplingSeed);
    float2 jitter = Sample2D(random, 0);
    static const float LargestBelowOne = 0.999999940395355224609375f;
    float u = min(float((double(coordinate.x) + double(jitter.x)) / double(FullWidth)), LargestBelowOne);
    float v = min(float((double(coordinate.y) + double(jitter.y)) / double(FullHeight)), LargestBelowOne);
    PointRay ray;
    ray.Origin = CameraPosition;
    ray.Direction = normalize(CameraForward + CameraRight * ((2 * u - 1) * HalfWidth) +
                              CameraUp * ((1 - 2 * v) * HalfHeight));
    ray.Minimum = RayEpsilon; ray.Maximum = MaximumRayDistance;
    ApplyThinLens(ray.Origin, ray.Direction, Sample2D(random, LensSampleDimension));
    random.PixelSeed = Hash(random.PixelSeed ^ Hash(PointSeed) ^ PointGatherDimension);
    IncrementLocalCounter(counters.Primary);
    Payload hit = CastRay(ray.Origin, ray.Direction, ray.Minimum, ray.Maximum);
    bool found = (hit.Flags & SurfaceHitFlag) != 0;
    double nearest = found ? double(asfloat(hit.DistanceBits)) : double(ray.Maximum);
    if (found && hit.MaterialIndex < MaterialCount && Materials[hit.MaterialIndex].SampledEmitter != 0)
        nearest += RayEpsilon;
    bool front; uint rectangle = PointRectangle(ray, nearest, front);
    if (rectangle != PointInvalid)
    {
        PackedEmitter light = Emitters[rectangle];
        return front ? double3(light.Strength) / (double(light.Width) * double(light.Height)) : double3(0, 0, 0);
    }
    if (!found) return double3(lerp(BackgroundBottom, BackgroundTop, saturate((ray.Direction.y + 1) / 2)));
    if (hit.MaterialIndex >= MaterialCount) { PointFail(PointBoundsError); return 0; }
    PointReceiver receiver;
    receiver.Position = ray.Origin + ray.Direction * asfloat(hit.DistanceBits);
    receiver.Geometry = normalize(hit.GeometricNormal);
    if (dot(receiver.Geometry, ray.Direction) > 0) receiver.Geometry = -receiver.Geometry;
    receiver.Normal = normalize(hit.Normal);
    if (dot(receiver.Normal, receiver.Geometry) < 0) receiver.Normal = -receiver.Normal;
    if (dot(receiver.Normal, -ray.Direction) <= 0) receiver.Normal = receiver.Geometry;
    double3 radiance = double3(Materials[hit.MaterialIndex].Emission.xyz);
    const MaterialData material = Materials[hit.MaterialIndex];
    if (material.Model == MaterialDielectric && material.Optical.w > 0)
    {
        const PointSubsurfaceExit exit = PointSubsurfaceWalk(ray, hit, receiver.Position,
            (hit.Flags & SurfaceFrontFaceFlag) == 0, random, counters);
        if (exit.Reached != 0)
        {
            double3 irradiance = PointDirectLight(exit.Receiver, random, counters);
            if (MaxReflectionBounces > 0)
                irradiance += PointIndirectLight(exit.Receiver, PointInvalid, random, true, counters);
            radiance += exit.Throughput * irradiance / PointPi;
        }
    }
    else
    {
        receiver.Reflectance = PointAlbedo(hit.MaterialIndex, hit.Color, hit.TexCoord,
                                            (hit.Flags & SurfaceTexCoordFlag) != 0);
        if (any(receiver.Reflectance > 0))
        {
            double3 irradiance = PointDirectLight(receiver, random, counters);
            if (MaxReflectionBounces > 0)
                irradiance += PointIndirectLight(receiver, PointInvalid, random, true, counters);
            radiance += receiver.Reflectance * irradiance / PointPi;
        }
    }
    return radiance;
}
uint PointDisplay(float3 radiance)
{
    static const float LinearThreshold = 0.0031308f, LinearScale = 12.92f;
    static const float NonlinearScale = 1.055f, NonlinearOffset = 0.055f, NonlinearExponent = 2.4f;
    static const uint ByteMaximum = 255, GreenShift = 8, BlueShift = 16, AlphaShift = 24;
    float3 color = max(radiance, 0);
    color /= max(1, max(color.x, max(color.y, color.z)));
    if (GammaCorrection != 0)
        [unroll] for (uint channel = 0; channel < 3; ++channel)
            color[channel] = color[channel] <= LinearThreshold ? color[channel] * LinearScale :
                             NonlinearScale * pow(color[channel], 1 / NonlinearExponent) - NonlinearOffset;
    uint3 encoded = uint3(saturate(color) * ByteMaximum + 0.5f);
    return encoded.x | (encoded.y << GreenShift) | (encoded.z << BlueShift) | (ByteMaximum << AlphaShift);
}
void PointCameraPixel(uint pixel)
{
    LinearFilmPixel film = LinearFilm[pixel];
    if (film.SampleCount != PointSampleBegin) { PointFail(PointControlError); return; }
    PointCounters counters = (PointCounters)0;
    [loop] for (uint sample = PointSampleBegin; sample < PointSampleBegin + PointSampleCount; ++sample)
    {
        double3 radiance = PointCameraRadiance(pixel, sample, counters);
        float3 value = float3(radiance);
        if (!PointColorValid(radiance) || !all(isfinite(value))) { PointFail(PointArithmeticError); return; }
        AccumulateFilmSample(film, value);
        if (!all(isfinite(film.MeanRadiance)) || !isfinite(film.LuminanceM2))
        { PointFail(PointArithmeticError); return; }
    }
    LinearFilm[pixel] = film;
    if (film.SampleCount == SamplesPerPixel)
    {
        OutputPixels[pixel] = PointDisplay(film.MeanRadiance);
        AddStatistic(CompletedCounterWord, uint2(1, 0));
    }
    PointPublishCounters(counters);
}
[shader("raygeneration")]
void RayGen()
{
    uint lane = DispatchRaysIndex().x;
    if (lane >= PointCount) return;
    uint identity = PointBegin + lane;
    if (PointStage == PointInitialize)
    {
        if (identity >= Width * Height) return;
        if (identity == 0) PointErrors.Store(0, 0);
        LinearFilm[identity] = (LinearFilmPixel)0; OutputPixels[identity] = 0;
        return;
    }
    if (PointStage == PointBake)
    {
        if (!PointWorkValid(PointSurfaceCount, PointOrder == 0 ? PointDirectSamples : PointGatherSamples)) return;
        if (PointOrder >= MaxReflectionBounces) { PointFail(PointControlError); return; }
        PointBakeSurface(identity);
    }
    else if (PointStage == PointCamera)
    {
        if (PointWorkValid(Width * Height, SamplesPerPixel)) PointCameraPixel(identity);
    }
    else PointFail(PointControlError);
}
