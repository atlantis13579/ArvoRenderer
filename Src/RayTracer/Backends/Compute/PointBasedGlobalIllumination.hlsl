// Independent Compute point-based transport. Diffuse surfels retain their
// view-independent cache; closed dielectric solids use interior random walks
// to transfer incident illumination between distinct surface locations.
// Christensen (2008), "Point-Based Approximate Color Bleeding":
// https://graphics.pixar.com/library/PointBasedColorBleeding/paper.pdf
// Equal-area disks store reflected radiance on both sides. Unlike the paper's
// far-field cluster approximation, this version traces the nearest disk using
// a stackless hierarchy. It is a biased geometric approximation; raising pixel
// SPP alone cannot remove the finite-patch bias. Direct illumination traces the
// actual scene. Disjoint Jacobi orders contain reflected light only, so analytic
// lights and intrinsic emission are never counted again through the cache.
// Each dispatch owns a bounded sample window. Division uses the complete bake
// budget, and the host publishes a whole reflection order before readers start.

static const float Pi = 3.14159265358979323846f, RayEpsilon = 1e-4f;
static const float MaximumRayDistance = 1e30f, BasisParallelThreshold = 0.999f;
static const double GiPi = 3.1415926535897932384626433832795L;
static const double GiDoubleLimit = 1.7976931348623157e308L;
static const uint GiInvalid = 0xffffffffu, GiAlgorithm = 8, GiMaximumReflections = 32;
static const uint GiThreads = 64, GiWorkBatch = 1024, GiSampleBatch = 32;
static const uint ImportanceBit = 1, NeeBit = 2;
static const uint PixelSeedMultiplier = 9781, SampleSeedMultiplier = 6271,
                  SecondDimensionScramble = 0x68bc21ebu;
static const uint GiSourceDimension = 0x534f5552u, GiPositionDimension = 0x504f5349u,
                  GiGatherDimension = 0x47415448u, GiBakeDomain = 0x42414b45u;
static const uint GiInterfaceDimension = 0x53534946u, GiBranchDimension = 0x53534252u;
static const uint GiFlightDimension = 0x5353464cu, GiPhaseDimension = 0x53535048u;
static const uint GiBounceScramble = 0x9e3779b9u;
static const uint GiNumericalError = 1, GiIndexError = 2, GiConfigurationError = 4;
#include "Types.hlsl"
#include "MaterialTypes.hlsl"
#include "EmitterData.hlsl"
struct GiSurface
{
    float3 Position; uint Material;
    float3 Geometry; uint Primitive;
    float3 Normal; uint HasTexCoord;
    float3 Color; uint Patch;
    float2 TexCoord;
    double Area, Radius;
    uint2 Padding;
};
struct GiBounds
{
    double3 Minimum; uint Begin, Count;
    double3 Maximum; uint Escape, Padding;
};
struct GiRadiance { double3 Front, Back; };
struct GiTexture { uint Offset, Width, Height, Reserved; };
struct GiStatistics { uint2 Primary, Shadow, Reflection, Tests; };
ByteAddressBuffer Nodes : register(t0);
StructuredBuffer<uint> PrimitiveRefs : register(t1);
StructuredBuffer<PackedTriangle> Triangles : register(t2);
StructuredBuffer<PackedSphere> Spheres : register(t3);
StructuredBuffer<PackedMaterial> Materials : register(t4);
StructuredBuffer<PackedEmitter> Emitters : register(t5);
StructuredBuffer<GiTexture> Textures : register(t6);
StructuredBuffer<uint> TextureTexels : register(t7);
// Instancing owns t9/t10 and u6. Point GI descriptors follow that scene ABI.
StructuredBuffer<GiSurface> Surfaces : register(t11);
StructuredBuffer<GiBounds> Hierarchy : register(t12);
StructuredBuffer<uint> SurfaceIndices : register(t13);
RWStructuredBuffer<float4> Accumulation : register(u0);
RWStructuredBuffer<uint> SampleCounts : register(u1);
RWStructuredBuffer<GiRadiance> PreviousOrder : register(u2);
RWStructuredBuffer<GiStatistics> Statistics : register(u3);
RWStructuredBuffer<uint> SampleTargets : register(u4);
RWStructuredBuffer<GiRadiance> NextOrder : register(u5);
RWByteAddressBuffer GiErrors : register(u7);
RWStructuredBuffer<GiRadiance> TotalRadiance : register(u8);
cbuffer Globals : register(b0)
{
    float3 CameraPosition; uint Width;
    float3 CameraForward; uint Height;
    float3 CameraRight; uint TotalSamples;
    float3 CameraUp; uint MaxReflectionBounces;
    float HalfWidth, HalfHeight; uint MaterialCount, LightCount;
    float3 BackgroundTop; float AmbientIntensity;
    float3 BackgroundBottom; uint NodeCount;
    uint PrimitiveRefCount, TriangleCount, SphereCount; float ShadowStrength;
    uint PixelBegin, PixelCount, SampleBegin, SampleCount;
    uint GammaCorrection, GroupCount, TextureCount, TextureTexelCount;
    uint FullWidth, FullHeight, TileX, TileY;
    uint Integrator, TransportFlags, RouletteStartDepth, SamplingSequence;
    uint AdaptiveEnabled, AdaptivePilotSamples, AdaptiveMinimumSamples, AdaptivePhase;
    float AdaptiveRelativeError, AdaptiveAbsoluteError; uint Acceleration, InstanceNodeCount;
    float LensRadius, FocusDistance;
    uint SamplingSeed;
};
cbuffer PointGiControls : register(b1)
{
    uint GiSurfaceCount, GiBoundsCount, GiSourceCount, GiSeed;
    uint GiDirectSamples, GiGatherSamples, GiBegin, GiCount;
    uint GiOrder, GiFirstSample, GiSamples, GiReserved;
};
uint Hash(uint value)
{
    static const uint AvalancheA = 0x7feb352du, AvalancheB = 0x846ca68bu;
    static const uint OuterShift = 16, MiddleShift = 15;
    value = (value ^ (value >> OuterShift)) * AvalancheA;
    value = (value ^ (value >> MiddleShift)) * AvalancheB;
    return value ^ (value >> OuterShift);
}
float Random01(uint seed)
{
    static const uint MantissaMask = 0xffffffu;
    static const float MantissaRange = 16777216.0f;
    return float(Hash(seed) & MantissaMask) / MantissaRange;
}
void IncrementCounter(inout uint2 value) { if (++value.x == 0) ++value.y; }
#include "Sequence.hlsl"
#include "Bounds.hlsl"
#include "Triangle.hlsl"
#include "Sphere.hlsl"
#include "Traversal.hlsl"
#include "Importance.hlsl"
#include "LightSelection.hlsl"
#include "LightEmission.hlsl"
#include "SurfaceEmission.hlsl"
#include "ThinLens.hlsl"
#include "FilmMoments.hlsl"
#include "BsdfTypes.hlsl"
#include "Fresnel.hlsl"
#include "Ggx.hlsl"
#include "Diffuse.hlsl"
#include "Conductor.hlsl"
#include "Dielectric.hlsl"
#include "SurfaceScattering.hlsl"
#include "SubsurfaceScattering.hlsl"
void GiFail(uint flag) { uint old; GiErrors.InterlockedOr(0, flag, old); }
bool GiFinite(double3 value) { return all(value >= 0) && all(value <= GiDoubleLimit); }
double GiDot(double3 a, double3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
bool GiConfiguration(uint population, uint samples)
{
    bool valid = Integrator == GiAlgorithm && MaxReflectionBounces <= GiMaximumReflections &&
        AdaptiveEnabled == 0 && ShadowStrength == 1 &&
        (TransportFlags & ~ImportanceBit) == NeeBit && GiReserved == 0 &&
        GiCount <= GiWorkBatch && GiBegin <= population && GiCount <= population - GiBegin &&
        GiSamples > 0 && GiSamples <= GiSampleBatch && GiFirstSample < samples &&
        GiSamples <= samples - GiFirstSample && GiSourceCount <= MaximumUniformEmitterCount;
    if (!valid) GiFail(GiConfigurationError);
    return valid;
}
float3 GiSky(float3 direction)
{
    return lerp(BackgroundBottom, BackgroundTop, saturate((direction.y + 1) / 2));
}
float GiDecode(uint byte)
{
    static const float ByteMaximum = 255, SrgbThreshold = 0.04045f, SrgbDivisor = 12.92f;
    static const float SrgbOffset = 0.055f, SrgbScale = 1.055f, SrgbPower = 2.4f;
    float value = float(byte) / ByteMaximum;
    return value <= SrgbThreshold ? value / SrgbDivisor : pow((value + SrgbOffset) / SrgbScale, SrgbPower);
}
float3 GiTexel(GiTexture texture, int2 position)
{
    int2 size = int2(texture.Width, texture.Height);
    uint2 wrapped = uint2((position % size + size) % size);
    uint pixel = TextureTexels[texture.Offset + wrapped.y * texture.Width + wrapped.x];
    static const uint ByteMask = 0xff, GreenShift = 8, BlueShift = 16;
    return float3(GiDecode(pixel & ByteMask), GiDecode((pixel >> GreenShift) & ByteMask),
                  GiDecode((pixel >> BlueShift) & ByteMask));
}
double3 GiReflectance(GiSurface surface)
{
    if (surface.Material >= MaterialCount) { GiFail(GiIndexError); return 0; }
    PackedMaterial material = Materials[surface.Material];
    if (material.SampledEmitter != 0) return 0;
    if (material.Optical.w > 0 && material.Model == MaterialDielectric) return 0;
    if (material.Model != MaterialDiffuse &&
        (material.Model != MaterialLegacy || material.AlbedoReflectivity.w != 0 || any(material.Specular.xyz != 0)))
    { GiFail(GiConfigurationError); return 0; }
    float3 color = surface.Color;
    uint textureIndex = material.DiffuseTextureIndex;
    if (surface.HasTexCoord != 0 && textureIndex < TextureCount && all(isfinite(surface.TexCoord)))
    {
        GiTexture texture = Textures[textureIndex];
        if (texture.Width > 0 && texture.Height > 0 && texture.Offset <= TextureTexelCount &&
            texture.Height <= (TextureTexelCount - texture.Offset) / texture.Width)
        {
            float2 coordinate = frac(float2(surface.TexCoord.x, 1 - surface.TexCoord.y)) *
                                float2(texture.Width, texture.Height) - 0.5f;
            int2 corner = int2(floor(coordinate)); float2 weight = frac(coordinate);
            float3 rowA = lerp(GiTexel(texture, corner), GiTexel(texture, corner + int2(1, 0)), weight.x);
            float3 rowB = lerp(GiTexel(texture, corner + int2(0, 1)), GiTexel(texture, corner + 1), weight.x);
            color *= lerp(rowA, rowB, weight.y);
        }
    }
    double3 reflectance = double3(material.AlbedoReflectivity.xyz) * double3(color);
    if (!GiFinite(reflectance)) { GiFail(GiNumericalError); return 0; }
    return clamp(reflectance, 0.0L, 1.0L);
}
RayData GiRay(GiSurface surface, bool front, float3 direction)
{
    RayData ray;
    ray.Origin = surface.Position + (front ? surface.Geometry : -surface.Geometry) * RayEpsilon;
    ray.Direction = direction; ray.MinimumDistance = RayEpsilon; ray.MaximumDistance = MaximumRayDistance;
    return ray;
}
uint GiRectangle(RayData ray, double maximum, out bool front, out double distance, inout uint2 tests)
{
    uint nearest = GiInvalid; distance = maximum; front = false;
    [loop] for (uint i = 0; i < GiSourceCount; ++i)
    {
        PackedEmitter source = Emitters[i];
        if (source.Kind != PackedEmitterRectangle) continue;
        IncrementCounter(tests);
        double denominator = GiDot(double3(source.Direction), double3(ray.Direction));
        if (denominator == 0) continue;
        double candidate = GiDot(double3(source.Origin) - double3(ray.Origin), double3(source.Direction)) / denominator;
        if (candidate < ray.MinimumDistance || candidate > distance || candidate > ray.MaximumDistance) continue;
        double3 offset = double3(ray.Origin) + double3(ray.Direction) * candidate - double3(source.Origin);
        LightEmissionFrame basis = EmissionFrame(normalize(source.Direction));
        if (abs(GiDot(offset, double3(basis.Tangent))) > double(source.Width) / 2 ||
            abs(GiDot(offset, double3(basis.Bitangent))) > double(source.Height) / 2) continue;
        if (candidate == distance && nearest != GiInvalid) continue;
        nearest = i; distance = candidate; front = denominator < 0;
    }
    return nearest;
}
bool GiBox(GiBounds box, RayData ray, double closest)
{
    double near = ray.MinimumDistance, far = min(double(ray.MaximumDistance), closest);
    [unroll] for (uint axis = 0; axis < 3; ++axis)
    {
        double origin = ray.Origin[axis], direction = ray.Direction[axis];
        if (direction == 0)
        { if (origin < box.Minimum[axis] || origin > box.Maximum[axis]) return false; }
        else
        {
            double a = (box.Minimum[axis] - origin) / direction, b = (box.Maximum[axis] - origin) / direction;
            near = max(near, min(a, b)); far = min(far, max(a, b));
            if (near > far) return false;
        }
    }
    return true;
}
uint GiDisk(RayData ray, uint ignored, out bool front, out double distance, inout uint2 tests)
{
    uint nearest = GiInvalid, nodeIndex = 0;
    distance = ray.MaximumDistance; front = false;
    [loop] while (nodeIndex < GiBoundsCount)
    {
        GiBounds node = Hierarchy[nodeIndex];
        if (node.Escape <= nodeIndex || node.Escape > GiBoundsCount || node.Begin > GiSurfaceCount ||
            node.Count > GiSurfaceCount - node.Begin) { GiFail(GiIndexError); return GiInvalid; }
        if (!GiBox(node, ray, distance)) { nodeIndex = node.Escape; continue; }
        if (node.Count == 0) { ++nodeIndex; continue; }
        [loop] for (uint entry = 0; entry < node.Count; ++entry)
        {
            uint identity = SurfaceIndices[node.Begin + entry];
            if (identity >= GiSurfaceCount) { GiFail(GiIndexError); return GiInvalid; }
            if (identity == ignored) continue;
            IncrementCounter(tests);
            GiSurface disk = Surfaces[identity];
            double3 normal = double3(disk.Geometry);
            double denominator = GiDot(normal, double3(ray.Direction));
            if (denominator == 0) continue;
            double candidate = GiDot(double3(disk.Position) - double3(ray.Origin), normal) / denominator;
            if (candidate < ray.MinimumDistance || candidate > distance) continue;
            double3 delta = double3(ray.Origin) + double3(ray.Direction) * candidate - double3(disk.Position);
            if (GiDot(delta, delta) > disk.Radius * disk.Radius) continue;
            if (candidate == distance && nearest != GiInvalid && identity > nearest) continue;
            nearest = identity; distance = candidate; front = denominator < 0;
        }
        nodeIndex = node.Escape;
    }
    return nearest;
}
double3 GiDirect(GiSurface receiver, bool front, SampleContext random, inout GiStatistics counters)
{
    EmitterSelection selection = SelectEmitter(GiSourceCount, Sample2D(random, GiSourceDimension).x);
    if (selection.Index == InvalidSelectedEmitter) return 0;
    PackedEmitter source = Emitters[selection.Index];
    float3 normal = front ? receiver.Normal : -receiver.Normal;
    float3 geometry = front ? receiver.Geometry : -receiver.Geometry;
    float2 variates = Sample2D(random, GiPositionDimension);
    float3 direction = 0, endpoint = 0, sourceNormal = 0;
    double3 value = 0;
    double pdf = selection.Probability, distance = MaximumRayDistance, areaPdf = 0;
    bool finiteSource = false, oneSided = false;
    if (source.Kind == PackedEmitterDirectional)
    { direction = normalize(source.Direction); value = double3(source.Strength); }
    else if (source.Kind == PackedEmitterEnvironment)
    {
        direction = SampleDiffuseDirection(normal, variates);
        pdf *= double(DiffusePdf(dot(normal, direction)));
        value = double3(lerp(source.BackgroundBottom, source.BackgroundTop, saturate((direction.y + 1) / 2)));
    }
    else if (source.Kind == PackedEmitterPoint)
    { endpoint = source.Origin; finiteSource = true; value = double3(source.Strength); }
    else if (source.Kind == PackedEmitterRectangle)
    {
        if (!(source.Width > 0 && source.Height > 0)) { GiFail(GiConfigurationError); return 0; }
        LightEmissionFrame basis = EmissionFrame(normalize(source.Direction));
        endpoint = source.Origin + basis.Tangent * ((variates.x - 0.5f) * source.Width) +
                basis.Bitangent * ((variates.y - 0.5f) * source.Height);
        sourceNormal = normalize(source.Direction); finiteSource = true; oneSided = true;
        areaPdf = 1 / (double(source.Width) * double(source.Height)); value = double3(source.Strength) * areaPdf;
    }
    else if (source.Kind == PackedEmitterTriangle || source.Kind == PackedEmitterSphere)
    {
        SurfaceEmissionSource surface;
        surface.Kind = source.Kind == PackedEmitterTriangle ? SurfaceEmitterTriangle : SurfaceEmitterEllipsoid;
        surface.Origin = source.Origin; surface.AxisX = source.AxisX; surface.AxisY = source.AxisY;
        surface.AxisZ = source.AxisZ; surface.Radiance = source.Strength;
        SurfaceEmissionSample sample = SampleSurfaceEmission(surface, variates, float2(0.25f, 0.5f));
        if ((sample.Flags & SurfaceEmitterValid) == 0) return 0;
        endpoint = sample.Origin; sourceNormal = sample.Normal; areaPdf = sample.PositionPdf;
        finiteSource = true; value = double3(sample.Value);
    }
    else { GiFail(GiConfigurationError); return 0; }
    if (finiteSource)
    {
        float3 offset = endpoint - receiver.Position;
        float lengthToSource = length(offset);
        if (!(lengthToSource > RayEpsilon)) return 0;
        direction = offset / lengthToSource; distance = lengthToSource;
        double distanceSquared = GiDot(double3(offset), double3(offset));
        if (source.Kind == PackedEmitterPoint) value /= distanceSquared;
        else
        {
            double cosine = GiDot(double3(sourceNormal), -double3(direction));
            if (!oneSided) cosine = abs(cosine);
            if (!(cosine > 0 && areaPdf > 0)) return 0;
            pdf *= areaPdf * distanceSquared / cosine;
        }
    }
    double cosine = dot(normal, direction);
    if (!(pdf > 0 && cosine > 0 && dot(geometry, direction) > 0)) return 0;
    RayData shadow = GiRay(receiver, front, direction);
    if (finiteSource)
    {
        float3 offset = endpoint - shadow.Origin;
        float lengthToSource = length(offset);
        if (!(lengthToSource > 2 * RayEpsilon)) return 0;
        shadow.Direction = offset / lengthToSource;
        shadow.MaximumDistance = lengthToSource - RayEpsilon;
    }
    HitData hit = (HitData)0; hit.Distance = shadow.MaximumDistance;
    IncrementCounter(counters.Shadow);
    bool blocked = IntersectScene(shadow, true, hit, counters.Tests), lightFront;
    double rectangleDistance;
    if (GiRectangle(shadow, shadow.MaximumDistance, lightFront, rectangleDistance, counters.Tests) != GiInvalid)
        blocked = true;
    return blocked ? double3(0, 0, 0) : value * (cosine / pdf);
}
double3 GiGather(GiSurface receiver, uint ignored, bool front, SampleContext random,
                 bool camera, inout GiStatistics counters)
{
    float3 normal = front ? receiver.Normal : -receiver.Normal;
    float3 geometry = front ? receiver.Geometry : -receiver.Geometry;
    float3 direction = SampleDiffuseDirection(normal, Sample2D(random, GiGatherDimension));
    double cosine = dot(normal, direction), pdf = DiffusePdf(float(cosine));
    if (!(pdf > 0 && dot(geometry, direction) > 0)) return 0;
    RayData ray = GiRay(receiver, front, direction);
    bool side; double distance;
    IncrementCounter(counters.Reflection);
    uint identity = GiDisk(ray, ignored, side, distance, counters.Tests);
    if (identity == GiInvalid) return 0;
    bool rectangleSide; double rectangleDistance;
    if (GiRectangle(ray, distance, rectangleSide, rectangleDistance, counters.Tests) != GiInvalid) return 0;
    GiRadiance incoming;
    if (camera) incoming = TotalRadiance[identity]; else incoming = PreviousOrder[identity];
    return (side ? incoming.Front : incoming.Back) * (cosine / pdf);
}
struct GiSubsurfaceExit
{
    GiSurface Surface;
    double3 Throughput;
    uint Reached;
};
GiSubsurfaceExit GiSubsurfaceWalk(RayData ray, HitData entry, float3 entryPosition,
                                  bool cameraInside, SampleContext random, inout GiStatistics counters)
{
    GiSubsurfaceExit result = (GiSubsurfaceExit)0;
    if (entry.MaterialIndex >= MaterialCount) { GiFail(GiIndexError); return result; }
    const PackedMaterial material = Materials[entry.MaterialIndex];
    double3 throughput = 1;
    if (!cameraInside)
    {
        const SurfaceBsdf boundary = MakeSurfaceBsdf(material, entry, -ray.Direction,
                                                       entry.Color, TransportRadiance);
        const SurfaceSample launch = SampleSurface(boundary,
            Sample2D(random, GiInterfaceDimension), Sample2D(random, GiBranchDimension).x);
        if ((launch.Event & ScatteringTransmission) == 0 || !(launch.Pdf > 0)) return result;
        throughput *= double3(launch.Weight);
        ray.Origin = SurfaceOffset(boundary, entryPosition, launch.Direction);
        ray.Direction = launch.Direction;
        ray.MinimumDistance = RayEpsilon;
        ray.MaximumDistance = MaximumRayDistance;
    }
    [loop] for (uint bounce = 0; bounce < GiMaximumReflections; ++bounce)
    {
        HitData hit = (HitData)0;
        hit.Distance = ray.MaximumDistance;
        IncrementCounter(counters.Reflection);
        if (!IntersectScene(ray, false, hit, counters.Tests) ||
            hit.MaterialIndex != entry.MaterialIndex)
            return result;
        const uint dimension = (bounce + 1) * GiBounceScramble;
        const SubsurfaceFlight flight = SubsurfaceFreeFlight(material.Optical.w,
            material.AlbedoReflectivity.xyz * material.ConductorExtinction.w,
            ray.Direction, hit.Distance, Sample2D(random, dimension ^ GiFlightDimension).x);
        throughput *= double3(flight.Weight);
        if (!GiFinite(throughput) || !any(throughput > 0)) return result;
        if (flight.Scattered != 0)
        {
            const SubsurfaceDirection phase = SubsurfaceSampleDirection(ray.Direction,
                material.ConductorIndex.w, Sample2D(random, dimension ^ GiPhaseDimension),
                (TransportFlags & ImportanceBit) != 0);
            if (!(phase.Pdf > 0)) return result;
            throughput *= double(phase.Weight);
            ray.Origin += ray.Direction * flight.Distance;
            ray.Direction = phase.Direction;
            continue;
        }
        const float3 position = ray.Origin + ray.Direction * hit.Distance;
        const SurfaceBsdf boundary = MakeSurfaceBsdf(material, hit, -ray.Direction,
                                                       hit.Color, TransportRadiance);
        const SurfaceSample event = SampleSurface(boundary,
            Sample2D(random, dimension ^ GiInterfaceDimension),
            Sample2D(random, dimension ^ GiBranchDimension).x);
        if (event.Event == 0 || !(event.Pdf > 0)) return result;
        throughput *= double3(event.Weight);
        if (!GiFinite(throughput) || !any(throughput > 0)) return result;
        if ((event.Event & ScatteringTransmission) == 0)
        {
            ray.Origin = SurfaceOffset(boundary, position, event.Direction);
            ray.Direction = event.Direction;
            continue;
        }
        GiSurface exit = (GiSurface)0;
        exit.Position = position;
        exit.Geometry = normalize(hit.GeometricNormal);
        exit.Normal = dot(hit.Normal, exit.Geometry) >= 0 ? normalize(hit.Normal) :
            -normalize(hit.Normal);
        exit.Material = hit.MaterialIndex;
        exit.Primitive = hit.PrimitiveIndex;
        exit.Color = hit.Color;
        exit.TexCoord = hit.TexCoord;
        exit.HasTexCoord = hit.HasTexCoord;
        result.Surface = exit;
        result.Throughput = throughput;
        result.Reached = 1;
        return result;
    }
    return result;
}
HitData GiSurfelHit(GiSurface surface)
{
    HitData hit = (HitData)0;
    hit.Normal = surface.Normal;
    hit.GeometricNormal = surface.Geometry;
    hit.FrontFace = 1;
    hit.Color = surface.Color;
    hit.MaterialIndex = surface.Material;
    hit.PrimitiveIndex = surface.Primitive;
    hit.TexCoord = surface.TexCoord;
    hit.HasTexCoord = surface.HasTexCoord;
    return hit;
}
double3 GiSubsurfaceBake(GiSurface receiver, SampleContext random, bool indirect,
                         inout GiStatistics counters)
{
    const GiSubsurfaceExit exit = GiSubsurfaceWalk(GiRay(receiver, true, -receiver.Geometry),
        GiSurfelHit(receiver), receiver.Position, false, random, counters);
    if (exit.Reached == 0) return 0;
    const double3 incident = indirect ? GiGather(exit.Surface, GiInvalid, true, random, false, counters) :
                                        GiDirect(exit.Surface, true, random, counters);
    return exit.Throughput * incident / GiPi;
}
SampleContext GiBakeSample(uint identity, uint side, uint sample, uint count)
{
    SampleContext result = MakeSampleContext(identity, side, GiSurfaceCount, sample, count, SamplingSequence, SamplingSeed);
    result.PixelSeed = Hash(result.PixelSeed ^ Hash(GiSeed) ^ Hash(GiOrder) ^ GiBakeDomain);
    return result;
}
[numthreads(GiThreads, 1, 1)]
void PointGiInitialize(uint3 thread : SV_DispatchThreadID)
{
    uint pixel = GiBegin + thread.x;
    if (thread.x >= GiCount || pixel >= Width * Height) return;
    Accumulation[pixel] = 0; SampleCounts[pixel] = 0; SampleTargets[pixel] = TotalSamples;
    if (pixel == 0) GiErrors.Store(0, 0);
}
[numthreads(GiThreads, 1, 1)]
void PointGiBake(uint3 thread : SV_DispatchThreadID)
{
    if (thread.x >= GiCount) return;
    GiStatistics counters = (GiStatistics)0;
    Statistics[thread.x] = counters;
    uint samples = GiOrder == 0 ? GiDirectSamples : GiGatherSamples;
    if (!GiConfiguration(GiSurfaceCount, samples)) return;
    if (GiOrder >= MaxReflectionBounces) { GiFail(GiConfigurationError); return; }
    uint identity = GiBegin + thread.x;
    GiSurface receiver = Surfaces[identity]; double3 rho = GiReflectance(receiver);
    const bool subsurface = receiver.Material < MaterialCount &&
        Materials[receiver.Material].Model == MaterialDielectric &&
        Materials[receiver.Material].Optical.w > 0;
    GiRadiance increment = (GiRadiance)0;
    [unroll] for (uint side = 0; side < 2; ++side)
    {
        double3 irradiance = 0;
        if (any(rho > 0) || (subsurface && side == 0))
        {
            [loop] for (uint sample = GiFirstSample; sample < GiFirstSample + GiSamples; ++sample)
            {
                SampleContext random = GiBakeSample(identity, side, sample, samples);
                if (subsurface)
                    irradiance += GiSubsurfaceBake(receiver, random, GiOrder != 0, counters) /
                                  double(samples);
                else
                    irradiance += (GiOrder == 0 ? GiDirect(receiver, side == 0, random, counters) :
                        GiGather(receiver, identity, side == 0, random, false, counters)) / double(samples);
            }
        }
        double3 reflected = subsurface ? irradiance : rho * irradiance / GiPi;
        if (side == 0) increment.Front = reflected; else increment.Back = reflected;
    }
    GiRadiance next = increment;
    if (GiFirstSample != 0)
    {
        GiRadiance partial = NextOrder[identity];
        next.Front += partial.Front; next.Back += partial.Back;
    }
    GiRadiance total = next;
    if (GiOrder != 0)
    {
        total = TotalRadiance[identity];
        total.Front += increment.Front; total.Back += increment.Back;
    }
    if (!GiFinite(next.Front) || !GiFinite(next.Back) || !GiFinite(total.Front) || !GiFinite(total.Back))
        GiFail(GiNumericalError);
    else { NextOrder[identity] = next; TotalRadiance[identity] = total; }
    Statistics[thread.x] = counters;
}
double3 GiCameraSample(uint x, uint y, uint sampleIndex, inout GiStatistics counters)
{
    SampleContext random = MakeSampleContext(x, y, FullWidth, sampleIndex, TotalSamples, SamplingSequence, SamplingSeed);
    float2 jitter = Sample2D(random, 0);
    float2 raster = float2((double(x) + jitter.x) / double(FullWidth),
                          (double(y) + jitter.y) / double(FullHeight));
    static const float LargestRaster = 0.999999940395355224609375f;
    raster = min(raster, LargestRaster);
    RayData ray;
    ray.Origin = CameraPosition;
    ray.Direction = normalize(CameraForward + CameraRight * ((2 * raster.x - 1) * HalfWidth) +
                              CameraUp * ((1 - 2 * raster.y) * HalfHeight));
    ray.MinimumDistance = RayEpsilon; ray.MaximumDistance = MaximumRayDistance;
    ApplyThinLens(ray, Sample2D(random, LensSampleDimension));
    random.PixelSeed = Hash(random.PixelSeed ^ Hash(GiSeed) ^ GiGatherDimension);
    HitData hit = (HitData)0; hit.Distance = ray.MaximumDistance;
    IncrementCounter(counters.Primary);
    bool found = IntersectScene(ray, false, hit, counters.Tests);
    bool front; double distance;
    bool sampledProxy = found && hit.MaterialIndex < MaterialCount && Materials[hit.MaterialIndex].SampledEmitter != 0;
    uint rectangle = GiRectangle(ray, found ? double(hit.Distance) + (sampledProxy ? RayEpsilon : 0) : double(ray.MaximumDistance),
                                  front, distance, counters.Tests);
    if (rectangle != GiInvalid)
    {
        PackedEmitter source = Emitters[rectangle];
        return front ? double3(source.Strength) / (double(source.Width) * source.Height) : double3(0, 0, 0);
    }
    if (!found) return double3(GiSky(ray.Direction));
    if (hit.MaterialIndex >= MaterialCount) { GiFail(GiIndexError); return 0; }
    GiSurface receiver = (GiSurface)0;
    receiver.Position = ray.Origin + ray.Direction * hit.Distance;
    receiver.Geometry = normalize(hit.GeometricNormal);
    if (dot(receiver.Geometry, ray.Direction) > 0) receiver.Geometry = -receiver.Geometry;
    receiver.Normal = normalize(hit.Normal);
    if (dot(receiver.Normal, receiver.Geometry) < 0) receiver.Normal = -receiver.Normal;
    if (!(dot(receiver.Normal, -ray.Direction) > 0)) receiver.Normal = receiver.Geometry;
    receiver.Material = hit.MaterialIndex; receiver.Primitive = hit.PrimitiveIndex;
    receiver.Color = hit.Color; receiver.TexCoord = hit.TexCoord; receiver.HasTexCoord = hit.HasTexCoord;
    const PackedMaterial material = Materials[hit.MaterialIndex];
    double3 result = double3(material.Emission.xyz);
    if (material.Model == MaterialDielectric && material.Optical.w > 0)
    {
        const GiSubsurfaceExit exit = GiSubsurfaceWalk(ray, hit, receiver.Position,
            hit.FrontFace == 0, random, counters);
        if (exit.Reached != 0)
        {
            double3 incident = GiDirect(exit.Surface, true, random, counters);
            if (MaxReflectionBounces > 0)
                incident += GiGather(exit.Surface, GiInvalid, true, random, true, counters);
            result += exit.Throughput * incident / GiPi;
        }
    }
    else
    {
        double3 rho = GiReflectance(receiver);
        if (any(rho > 0))
        {
            double3 irradiance = GiDirect(receiver, true, random, counters);
            if (MaxReflectionBounces > 0)
                irradiance += GiGather(receiver, GiInvalid, true, random, true, counters);
            result += rho * irradiance / GiPi;
        }
    }
    return result;
}
[numthreads(GiThreads, 1, 1)]
void PointGiCamera(uint3 thread : SV_DispatchThreadID)
{
    if (thread.x >= GiCount) return;
    GiStatistics counters = (GiStatistics)0; Statistics[thread.x] = counters;
    if (!GiConfiguration(Width * Height, TotalSamples)) return;
    uint pixel = GiBegin + thread.x;
    float4 moments = Accumulation[pixel]; uint count = SampleCounts[pixel];
    if (count != GiFirstSample) { GiFail(GiConfigurationError); return; }
    uint x = TileX + pixel % Width, y = TileY + pixel / Width;
    [loop] for (uint sample = GiFirstSample; sample < GiFirstSample + GiSamples; ++sample)
    {
        double3 radiance = GiCameraSample(x, y, sample, counters);
        float3 value = float3(radiance);
        if (!GiFinite(radiance) || !all(isfinite(value))) { GiFail(GiNumericalError); return; }
        AccumulateFilmSample(moments, count, value);
        if (!all(isfinite(moments))) { GiFail(GiNumericalError); return; }
    }
    Accumulation[pixel] = moments; SampleCounts[pixel] = count; Statistics[thread.x] = counters;
}
