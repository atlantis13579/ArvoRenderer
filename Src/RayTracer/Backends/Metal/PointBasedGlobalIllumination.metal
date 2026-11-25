// Independent native Metal point-based GI. Closed dielectric solids transfer
// light between surface disks through interior free-flight random walks.
// Christensen (2008), "Point-Based Approximate Color Bleeding":
// https://graphics.pixar.com/library/PointBasedColorBleeding/paper.pdf
// Reflected radiance is baked on two-sided equivalent-area disks. This variant
// resolves nearest-disk visibility instead of using distant cluster lobes.
// Source emission remains outside the cache and enters only through NEE or a
// primary emitter hit. Jacobi orders read a completed previous-order buffer.
// Finite patch bias and bake noise remain when camera SPP increases. Film M2
// is conditional on this fixed cache, so adaptive allocation is unavailable.
// Metal's scaled float expansion is not binary64. Compile with fast math OFF.
#include <metal_stdlib>
using namespace metal;
constant float Pi = 3.14159265358979323846f, PiResidual = -8.742278000372485e-8f;
constant float RayEpsilon = 1e-4f, MaximumRayDistance = 1e30f, BasisParallelThreshold = 0.999f;
constant uint ImportanceBit = 1, NeeBit = 2;
constant uint GiInvalid = 0xffffffffu, GiAlgorithm = 8, GiMaximumReflections = 32;
constant uint GiWorkBatch = 1024, GiSampleBatch = 32;
constant uint GiNumericalError = 1, GiConfigurationError = 2, GiIndexError = 4;
constant int GiExponentLimit = 16384;
constant uint AvalancheMultiplierA = 0x7feb352du, AvalancheMultiplierB = 0x846ca68bu;
constant uint RandomMantissaMask = 0x00ffffffu;
constant float RandomMantissaRange = 16777216.0f;
constant uint PixelSeedMultiplier = 9781, SampleSeedMultiplier = 6271, SecondDimensionScramble = 0x68bc21ebu;
constant uint GiSourceDomain = 0x534f5552u, GiEndpointDomain = 0x454e4450u;
constant uint GiGatherDomain = 0x47415448u, GiBakeDomain = 0x42414b45u;
constant uint GiInterfaceDomain = 0x53534946u, GiBranchDomain = 0x53534252u;
constant uint GiFlightDomain = 0x5353464cu, GiPhaseDomain = 0x53535048u;
constant uint GiBounceScramble = 0x9e3779b9u;
#include "Hash.metal"
#include "Sequence.metal"
#include "SurfaceTypes.metal"
#include "ShadingTypes.metal"
#include "ExtendedFloat.metal"
#include "EmitterData.metal"
struct FrameConstants
{
    packed_float3 CameraPosition;
    uint Width;
    packed_float3 CameraForward;
    uint Height;
    packed_float3 CameraRight;
    uint TotalSamples;
    packed_float3 CameraUp;
    uint MaxReflectionBounces;
    float HalfWidth, HalfHeight;
    uint MaterialCount, LightCount;
    packed_float3 BackgroundTop;
    float AmbientIntensity;
    packed_float3 BackgroundBottom;
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
    uint SamplingSeed, LensPadding1;
};
struct GiControls
{
    uint Surfaces, Nodes, Sources, Seed;
    uint DirectSamples, GatherSamples, Begin, Count;
    uint Order, FirstSample, Samples, Reserved;
};
using GiNumber = WideMath::Number;
struct GiVector
{
    GiNumber X, Y, Z;
};
struct GiSurface
{
    packed_float3 Position;
    uint Material;
    packed_float3 GeometricNormal;
    uint Primitive;
    packed_float3 ShadingNormal;
    uint HasTexCoord;
    packed_float3 Color;
    uint Patch;
    float2 TexCoord;
    uint Reserved[2];
    GiNumber Area, RadiusSquared;
};
struct GiNode
{
    GiNumber Minimum[3], Maximum[3];
    uint Begin, Count, Escape, Reserved;
};
struct GiRadiance
{
    GiVector Front, Back;
};
struct GiStatistics
{
    uint2 Primary, Shadow, Reflection, Tests;
};
static_assert(sizeof(FrameConstants) == 240 && sizeof(GiControls) == 48 && sizeof(GiSurface) == 112 &&
                  sizeof(GiNode) == 112 && sizeof(GiRadiance) == 96 && sizeof(GiStatistics) == 32,
              "Metal Point GI resource ABI mismatch");
void IncrementCounter(thread uint2& counter)
{
    if (++counter.x == 0)
        ++counter.y;
}
#include "TraversalTypes.metal"
#include "Triangle.metal"
#include "Sphere.metal"
#include "Bvh.metal"
#include "KdTree.metal"
#include "Bih.metal"
#include "Dispatch.metal"
#include "Importance.metal"
#include "LightSelection.metal"
#include "LightEmission.metal"
#include "SurfaceEmission.metal"
#include "ThinLens.metal"
#include "FilmMoments.metal"
#include "BsdfTypes.metal"
#include "Fresnel.metal"
#include "Ggx.metal"
#include "Diffuse.metal"
#include "Conductor.metal"
#include "Dielectric.metal"
#include "SurfaceScattering.metal"
#include "SubsurfaceScattering.metal"
struct GiWorld
{
    SceneGeometryBuffers Geometry;
    device const PackedMaterial* Materials;
    device const PackedEmitter* Sources;
    device const PackedTextureDescriptor* Textures;
    device const uint* Texels;
    constant FrameConstants* Frame;
    constant GiControls* Controls;
    device atomic_uint* Error;
};
struct GiCache
{
    device const GiSurface* Surfaces;
    device const GiNode* Nodes;
    device const uint* Indices;
};
struct GiReceiver
{
    float3 Position, Geometry, Normal;
    GiVector Rho;
};
void GiFail(GiWorld world, uint flag)
{
    atomic_fetch_or_explicit(world.Error, flag, memory_order_relaxed);
}
GiNumber GiFloat(float value)
{
    return WideMath::FromFloat(value);
}
GiNumber GiPi()
{
    return WideMath::Normalize(Pi, PiResidual, 0);
}
GiVector GiWide(float3 value)
{
    return {GiFloat(value.x), GiFloat(value.y), GiFloat(value.z)};
}
GiVector GiAdd(GiVector a, GiVector b)
{
    return {WideMath::Add(a.X, b.X), WideMath::Add(a.Y, b.Y), WideMath::Add(a.Z, b.Z)};
}
GiVector GiSubtract(GiVector a, GiVector b)
{
    return {WideMath::Subtract(a.X, b.X), WideMath::Subtract(a.Y, b.Y), WideMath::Subtract(a.Z, b.Z)};
}
GiVector GiScale(GiVector a, GiNumber b)
{
    return {WideMath::Multiply(a.X, b), WideMath::Multiply(a.Y, b), WideMath::Multiply(a.Z, b)};
}
GiVector GiProduct(GiVector a, GiVector b)
{
    return {WideMath::Multiply(a.X, b.X), WideMath::Multiply(a.Y, b.Y), WideMath::Multiply(a.Z, b.Z)};
}
GiNumber GiDot(GiVector a, GiVector b)
{
    return WideMath::Add(WideMath::Add(WideMath::Multiply(a.X, b.X), WideMath::Multiply(a.Y, b.Y)),
                         WideMath::Multiply(a.Z, b.Z));
}
GiNumber GiAxis(GiVector value, uint axis)
{
    return axis == 0 ? value.X : axis == 1 ? value.Y : value.Z;
}
float GiNarrow(GiNumber value)
{
    return as_type<float>(WideMath::Encode(value));
}
float3 GiNarrow(GiVector value)
{
    return float3(GiNarrow(value.X), GiNarrow(value.Y), GiNarrow(value.Z));
}
bool GiFinite(GiNumber value)
{
    return isfinite(value.High) && isfinite(value.Low) && value.Exponent > -GiExponentLimit &&
           value.Exponent < GiExponentLimit;
}
bool GiColor(GiVector value)
{
    return GiFinite(value.X) && GiFinite(value.Y) && GiFinite(value.Z) && value.X.High >= 0 &&
           value.Y.High >= 0 && value.Z.High >= 0;
}
bool GiNonzero(GiVector value)
{
    return value.X.High > 0 || value.Y.High > 0 || value.Z.High > 0;
}
GiVector GiReflected(GiReceiver receiver, GiVector irradiance)
{
    return GiScale(GiProduct(receiver.Rho, irradiance), WideMath::Divide(GiFloat(1), GiPi()));
}
bool GiConfiguration(GiWorld world, uint population, uint samples)
{
    constant auto& f = *world.Frame;
    constant auto& c = *world.Controls;
    bool valid = f.Integrator == GiAlgorithm && f.MaxReflectionBounces <= GiMaximumReflections &&
                 !f.AdaptiveEnabled && f.ShadowStrength == 1 &&
                 (f.TransportFlags & ~ImportanceBit) == NeeBit && c.Reserved == 0 && c.Count <= GiWorkBatch &&
                 c.Begin <= population && c.Count <= population - c.Begin && c.Samples > 0 &&
                 c.Samples <= GiSampleBatch && c.FirstSample < samples &&
                 c.Samples <= samples - c.FirstSample && c.Sources <= MaximumUniformSources;
    if (!valid)
        GiFail(world, GiConfigurationError);
    return valid;
}
float GiDecodeChannel(uint channel)
{
    const float ByteMaximum = 255, SrgbCutoff = 0.04045f, LinearDivisor = 12.92f;
    const float Offset = 0.055f, Scale = 1.055f, Exponent = 2.4f;
    float value = float(channel) / ByteMaximum;
    return value <= SrgbCutoff ? value / LinearDivisor : pow((value + Offset) / Scale, Exponent);
}
float3 GiTexel(GiWorld world, PackedTextureDescriptor texture, int2 p)
{
    int width = int(texture.Width), height = int(texture.Height);
    int x = (p.x % width + width) % width, y = (p.y % height + height) % height;
    uint texel = world.Texels[texture.OffsetTexels + uint(y) * texture.Width + uint(x)];
    const uint ByteMask = 0xff, GreenShift = 8, BlueShift = 16;
    return float3(GiDecodeChannel(texel & ByteMask), GiDecodeChannel((texel >> GreenShift) & ByteMask),
                  GiDecodeChannel((texel >> BlueShift) & ByteMask));
}
GiVector GiReflectance(GiWorld world, uint materialId, float3 color, float2 uv, bool hasUv)
{
    if (materialId >= world.Frame->MaterialCount)
    {
        GiFail(world, GiIndexError);
        return {};
    }
    const PackedMaterial material = world.Materials[materialId];
    if (material.SampledEmitter)
        return {};
    if (material.Model == MaterialDielectric && material.Optical.w > 0)
        return {};
    if (material.Model != MaterialDiffuse &&
        (material.Model != MaterialLegacy || material.AlbedoReflectivity.w != 0 ||
         any(material.Specular.xyz != float3(0))))
    {
        GiFail(world, GiConfigurationError);
        return {};
    }
    if (hasUv && material.DiffuseTextureIndex < world.Frame->TextureCount && all(isfinite(uv)))
    {
        const PackedTextureDescriptor texture = world.Textures[material.DiffuseTextureIndex];
        if (texture.Width && texture.Height && texture.OffsetTexels <= world.Frame->TextureTexelCount &&
            texture.Height <= (world.Frame->TextureTexelCount - texture.OffsetTexels) / texture.Width)
        {
            float2 coordinate =
                fract(float2(uv.x, 1 - uv.y)) * float2(float(texture.Width), float(texture.Height)) - 0.5f;
            int2 origin = int2(floor(coordinate));
            float2 weight = fract(coordinate);
            float3 row0 =
                mix(GiTexel(world, texture, origin), GiTexel(world, texture, origin + int2(1, 0)), weight.x);
            float3 row1 = mix(GiTexel(world, texture, origin + int2(0, 1)),
                              GiTexel(world, texture, origin + int2(1, 1)), weight.x);
            color *= mix(row0, row1, weight.y);
        }
    }
    GiVector rho = GiProduct(GiWide(color), GiWide(material.AlbedoReflectivity.xyz));
    if (!GiColor(rho))
    {
        GiFail(world, GiNumericalError);
        return {};
    }
    const GiNumber one = GiFloat(1);
    return {WideMath::Minimum(rho.X, one), WideMath::Minimum(rho.Y, one), WideMath::Minimum(rho.Z, one)};
}
RayData GiSpawn(GiReceiver receiver, float3 direction)
{
    return {receiver.Position + receiver.Geometry * RayEpsilon, direction, RayEpsilon, MaximumRayDistance};
}
uint GiRectangle(GiWorld world, RayData ray, thread GiNumber& distance, thread bool& front,
                 thread uint2& tests)
{
    uint identity = GiInvalid;
    front = false;
    const GiVector origin = GiWide(ray.Origin), direction = GiWide(ray.Direction);
    for (uint index = 0; index < world.Controls->Sources; ++index)
    {
        const PackedEmitter light = world.Sources[index];
        if (light.Kind != PackedEmitterRectangle)
            continue;
        IncrementCounter(tests);
        GiVector normal = GiWide(float3(light.Direction));
        GiNumber denominator = GiDot(normal, direction);
        if (denominator.High == 0)
            continue;
        GiNumber candidate =
            WideMath::Divide(GiDot(GiSubtract(GiWide(float3(light.Origin)), origin), normal), denominator);
        if (WideMath::Compare(candidate, GiFloat(ray.MinimumDistance)) < 0 ||
            WideMath::Compare(candidate, distance) > 0 ||
            WideMath::Compare(candidate, GiFloat(ray.MaximumDistance)) > 0)
            continue;
        const GiVector delta =
            GiSubtract(GiAdd(origin, GiScale(direction, candidate)), GiWide(float3(light.Origin)));
        float3 tangent, bitangent;
        EmissionFrame(normalize(float3(light.Direction)), tangent, bitangent);
        if (WideMath::Compare(WideMath::Absolute(GiDot(delta, GiWide(tangent))),
                              WideMath::Shift(GiFloat(light.Width), -1)) > 0 ||
            WideMath::Compare(WideMath::Absolute(GiDot(delta, GiWide(bitangent))),
                              WideMath::Shift(GiFloat(light.Height), -1)) > 0)
            continue;
        if (WideMath::Compare(candidate, distance) == 0 && identity != GiInvalid)
            continue;
        distance = candidate;
        identity = index;
        front = denominator.High < 0;
    }
    return identity;
}
bool GiBox(GiNode node, RayData ray, GiNumber closest)
{
    GiNumber low = GiFloat(ray.MinimumDistance),
             high = WideMath::Minimum(GiFloat(ray.MaximumDistance), closest);
    GiVector origin = GiWide(ray.Origin), direction = GiWide(ray.Direction);
    for (uint axis = 0; axis < 3; ++axis)
    {
        GiNumber o = GiAxis(origin, axis), d = GiAxis(direction, axis);
        if (d.High == 0)
        {
            if (WideMath::Compare(o, node.Minimum[axis]) < 0 || WideMath::Compare(o, node.Maximum[axis]) > 0)
                return false;
        }
        else
        {
            GiNumber a = WideMath::Divide(WideMath::Subtract(node.Minimum[axis], o), d);
            GiNumber b = WideMath::Divide(WideMath::Subtract(node.Maximum[axis], o), d);
            low = WideMath::Maximum(low, WideMath::Minimum(a, b));
            high = WideMath::Minimum(high, WideMath::Maximum(a, b));
            if (WideMath::Compare(low, high) > 0)
                return false;
        }
    }
    return true;
}
uint GiDisk(GiWorld world, GiCache cache, RayData ray, uint ignored, thread bool& front,
            thread GiNumber& distance, thread uint2& tests)
{
    uint nearest = GiInvalid, nodeIndex = 0;
    front = false;
    distance = GiFloat(ray.MaximumDistance);
    GiVector origin = GiWide(ray.Origin), direction = GiWide(ray.Direction);
    constant auto& c = *world.Controls;
    while (nodeIndex < c.Nodes)
    {
        GiNode node = cache.Nodes[nodeIndex];
        if (node.Escape <= nodeIndex || node.Escape > c.Nodes || node.Begin > c.Surfaces ||
            node.Count > c.Surfaces - node.Begin)
        {
            GiFail(world, GiIndexError);
            return GiInvalid;
        }
        if (!GiBox(node, ray, distance))
        {
            nodeIndex = node.Escape;
            continue;
        }
        if (!node.Count)
        {
            ++nodeIndex;
            continue;
        }
        for (uint i = 0; i < node.Count; ++i)
        {
            uint identity = cache.Indices[node.Begin + i];
            if (identity >= c.Surfaces)
            {
                GiFail(world, GiIndexError);
                return GiInvalid;
            }
            if (identity == ignored)
                continue;
            IncrementCounter(tests);
            const GiSurface surface = cache.Surfaces[identity];
            const GiVector center = GiWide(float3(surface.Position)),
                           normal = GiWide(float3(surface.GeometricNormal));
            GiNumber denominator = GiDot(direction, normal);
            if (denominator.High == 0)
                continue;
            GiNumber candidate = WideMath::Divide(GiDot(GiSubtract(center, origin), normal), denominator);
            if (WideMath::Compare(candidate, GiFloat(ray.MinimumDistance)) < 0 ||
                WideMath::Compare(candidate, distance) > 0)
                continue;
            GiVector delta = GiSubtract(GiAdd(origin, GiScale(direction, candidate)), center);
            if (WideMath::Compare(GiDot(delta, delta), surface.RadiusSquared) > 0)
                continue;
            if (WideMath::Compare(candidate, distance) == 0 && nearest != GiInvalid && identity > nearest)
                continue;
            nearest = identity;
            distance = candidate;
            front = denominator.High < 0;
        }
        nodeIndex = node.Escape;
    }
    return nearest;
}
GiVector GiDirect(GiWorld world, GiReceiver receiver, SampleContext random, thread GiStatistics& statistics)
{
    EmitterSelection selected = SelectEmitter(world.Controls->Sources, Sample2D(random, GiSourceDomain).x);
    if (selected.Index == InvalidSourceSelection)
        return {};
    PackedEmitter source = world.Sources[selected.Index];
    float2 u = Sample2D(random, GiEndpointDomain);
    float3 direction = float3(0), endpoint = float3(0), normal = float3(0);
    GiVector radiance{};
    GiNumber probability = GiFloat(selected.Probability), areaPdf{};
    bool finite = false, oneSided = false;
    if (source.Kind == PackedEmitterDirectional)
    {
        direction = normalize(float3(source.Direction));
        radiance = GiWide(float3(source.Strength));
    }
    else if (source.Kind == PackedEmitterEnvironment)
    {
        direction = SampleDiffuseDirection(receiver.Normal, u, world.Frame->TransportFlags);
        probability = WideMath::Multiply(
            probability, GiFloat(DiffusePdf(dot(receiver.Normal, direction), world.Frame->TransportFlags)));
        radiance = GiWide(mix(float3(source.BackgroundBottom), float3(source.BackgroundTop),
                              clamp((direction.y + 1) / 2, 0.0f, 1.0f)));
    }
    else if (source.Kind == PackedEmitterPoint)
    {
        finite = true;
        endpoint = float3(source.Origin);
        radiance = GiWide(float3(source.Strength));
    }
    else if (source.Kind == PackedEmitterRectangle)
    {
        if (!(source.Width > 0 && source.Height > 0))
        {
            GiFail(world, GiConfigurationError);
            return {};
        }
        normal = normalize(float3(source.Direction));
        float3 tangent, bitangent;
        EmissionFrame(normal, tangent, bitangent);
        endpoint = float3(source.Origin) + tangent * ((u.x - 0.5f) * source.Width) +
                   bitangent * ((u.y - 0.5f) * source.Height);
        areaPdf =
            WideMath::Divide(GiFloat(1), WideMath::Multiply(GiFloat(source.Width), GiFloat(source.Height)));
        radiance = GiScale(GiWide(float3(source.Strength)), areaPdf);
        finite = true;
        oneSided = true;
    }
    else if (source.Kind == PackedEmitterTriangle || source.Kind == PackedEmitterSphere)
    {
        SurfaceEmissionSource emitter;
        emitter.Kind =
            source.Kind == PackedEmitterTriangle ? SurfaceEmitterTriangle : SurfaceEmitterEllipsoid;
        emitter.Origin = float3(source.Origin);
        emitter.AxisX = float3(source.AxisX);
        emitter.AxisY = float3(source.AxisY);
        emitter.AxisZ = float3(source.AxisZ);
        emitter.Radiance = float3(source.Strength);
        SurfaceEmissionSample sample = SampleSurfaceEmission(emitter, u, float2(0.25f, 0.5f));
        if (!(sample.Flags & SurfaceEmitterValid))
            return {};
        endpoint = sample.Origin;
        normal = sample.Normal;
        areaPdf = GiFloat(sample.PositionPdf);
        radiance = GiWide(sample.Value);
        finite = true;
    }
    else
    {
        GiFail(world, GiConfigurationError);
        return {};
    }
    if (finite)
    {
        float3 segment = endpoint - receiver.Position;
        float distance = length(segment);
        if (!(distance > RayEpsilon))
            return {};
        direction = segment / distance;
        GiNumber squared = GiDot(GiWide(segment), GiWide(segment));
        if (source.Kind == PackedEmitterPoint)
            radiance = GiScale(radiance, WideMath::Divide(GiFloat(1), squared));
        else
        {
            GiNumber cosine = GiDot(GiWide(normal), GiWide(-direction));
            if (!oneSided)
                cosine = WideMath::Absolute(cosine);
            if (!(cosine.High > 0 && areaPdf.High > 0))
                return {};
            probability = WideMath::Multiply(probability,
                                             WideMath::Divide(WideMath::Multiply(squared, areaPdf), cosine));
        }
    }
    float cosine = dot(receiver.Normal, direction);
    if (!(probability.High > 0 && cosine > 0 && dot(receiver.Geometry, direction) > 0))
        return {};
    RayData shadow = GiSpawn(receiver, direction);
    if (finite)
    {
        float3 segment = endpoint - shadow.Origin;
        float distance = length(segment);
        if (!(distance > 2 * RayEpsilon))
            return {};
        shadow.Direction = segment / distance;
        shadow.MaximumDistance = distance - RayEpsilon;
    }
    HitData hit{};
    hit.Distance = shadow.MaximumDistance;
    IncrementCounter(statistics.Shadow);
    if (IntersectScene(world.Geometry, shadow, true, hit, statistics.Tests))
        return {};
    GiNumber maximum = GiFloat(shadow.MaximumDistance);
    bool front;
    if (GiRectangle(world, shadow, maximum, front, statistics.Tests) != GiInvalid)
        return {};
    return GiScale(radiance, WideMath::Divide(GiFloat(cosine), probability));
}
GiVector GiGather(GiWorld world, GiCache cache, device const GiRadiance* radiance, GiReceiver receiver,
                  uint ignored, SampleContext random, thread GiStatistics& statistics)
{
    float3 direction = SampleDiffuseDirection(receiver.Normal, Sample2D(random, GiGatherDomain),
                                              world.Frame->TransportFlags);
    float cosine = dot(receiver.Normal, direction), pdf = DiffusePdf(cosine, world.Frame->TransportFlags);
    if (!(pdf > 0 && dot(receiver.Geometry, direction) > 0))
        return {};
    RayData ray = GiSpawn(receiver, direction);
    bool front;
    GiNumber distance;
    IncrementCounter(statistics.Reflection);
    uint identity = GiDisk(world, cache, ray, ignored, front, distance, statistics.Tests);
    if (identity == GiInvalid)
        return {};
    bool rectangleFront;
    if (GiRectangle(world, ray, distance, rectangleFront, statistics.Tests) != GiInvalid)
        return {};
    const GiRadiance incoming = radiance[identity];
    return GiScale(front ? incoming.Front : incoming.Back, WideMath::Divide(GiFloat(cosine), GiFloat(pdf)));
}
struct GiSubsurfaceExit
{
    GiReceiver Receiver;
    GiVector Throughput;
    bool Reached;
};
GiSubsurfaceExit GiSubsurfaceWalk(GiWorld world, RayData ray, HitData entry, float3 entryPosition,
                                  bool cameraInside, SampleContext random,
                                  thread GiStatistics& statistics)
{
    GiSubsurfaceExit result{};
    if (entry.MaterialIndex >= world.Frame->MaterialCount)
    {
        GiFail(world, GiIndexError);
        return result;
    }
    const PackedMaterial material = world.Materials[entry.MaterialIndex];
    GiVector throughput = GiWide(float3(1));
    const bool importance = (world.Frame->TransportFlags & ImportanceBit) != 0;
    if (!cameraInside)
    {
        const SurfaceBsdf boundary = MakeSurfaceBsdf(material, entry, -ray.Direction,
            entry.Color, world.Frame->TransportFlags, TransportRadiance);
        const SurfaceSample launch = SampleSurface(boundary,
            Sample2D(random, GiInterfaceDomain), Sample2D(random, GiBranchDomain).x);
        if (!(launch.Event & ScatteringTransmission) || !(launch.Pdf > 0)) return result;
        throughput = GiProduct(throughput, GiWide(launch.Weight));
        ray = {SurfaceOffset(boundary, entryPosition, launch.Direction), launch.Direction,
               RayEpsilon, MaximumRayDistance};
    }
    for (uint bounce = 0; bounce < GiMaximumReflections; ++bounce)
    {
        HitData boundary{};
        boundary.Distance = ray.MaximumDistance;
        IncrementCounter(statistics.Reflection);
        if (!IntersectScene(world.Geometry, ray, false, boundary, statistics.Tests) ||
            boundary.MaterialIndex != entry.MaterialIndex)
            return result;
        const uint dimension = (bounce + 1) * GiBounceScramble;
        const SubsurfaceFlight flight = SubsurfaceFreeFlight(material.Optical.w,
            material.AlbedoReflectivity.xyz * material.ConductorExtinction.w,
            ray.Direction, boundary.Distance, Sample2D(random, dimension ^ GiFlightDomain).x);
        throughput = GiProduct(throughput, GiWide(flight.Weight));
        if (!GiColor(throughput) || !GiNonzero(throughput)) return result;
        if (flight.Scattered)
        {
            const SubsurfaceDirection phase = SubsurfaceSampleDirection(ray.Direction,
                material.ConductorIndex.w, Sample2D(random, dimension ^ GiPhaseDomain), importance);
            if (!(phase.Pdf > 0)) return result;
            throughput = GiScale(throughput, GiFloat(phase.Weight));
            ray = {ray.Origin + ray.Direction * flight.Distance, phase.Direction,
                   RayEpsilon, MaximumRayDistance};
            continue;
        }
        const float3 position = ray.Origin + ray.Direction * boundary.Distance;
        const SurfaceBsdf interface = MakeSurfaceBsdf(material, boundary, -ray.Direction,
            boundary.Color, world.Frame->TransportFlags, TransportRadiance);
        const SurfaceSample event = SampleSurface(interface,
            Sample2D(random, dimension ^ GiInterfaceDomain),
            Sample2D(random, dimension ^ GiBranchDomain).x);
        if (!event.Event || !(event.Pdf > 0)) return result;
        throughput = GiProduct(throughput, GiWide(event.Weight));
        if (!GiColor(throughput) || !GiNonzero(throughput)) return result;
        if (!(event.Event & ScatteringTransmission))
        {
            ray = {SurfaceOffset(interface, position, event.Direction), event.Direction,
                   RayEpsilon, MaximumRayDistance};
            continue;
        }
        GiReceiver receiver{};
        receiver.Position = position;
        receiver.Geometry = normalize(boundary.GeometricNormal);
        receiver.Normal = dot(boundary.Normal, receiver.Geometry) >= 0 ?
            normalize(boundary.Normal) : -normalize(boundary.Normal);
        result.Receiver = receiver;
        result.Throughput = throughput;
        result.Reached = true;
        return result;
    }
    return result;
}
HitData GiSurfelHit(GiSurface surfel)
{
    HitData hit{};
    hit.Normal = float3(surfel.ShadingNormal);
    hit.GeometricNormal = float3(surfel.GeometricNormal);
    hit.Color = float3(surfel.Color);
    hit.MaterialIndex = surfel.Material;
    hit.PrimitiveIndex = surfel.Primitive;
    hit.TexCoord = surfel.TexCoord;
    hit.HasTexCoord = surfel.HasTexCoord;
    hit.FrontFace = 1;
    return hit;
}
GiVector GiSubsurfaceBake(GiWorld world, GiCache cache, device const GiRadiance* previous,
                           GiSurface surfel, SampleContext random, thread GiStatistics& statistics)
{
    RayData ray{float3(surfel.Position), -float3(surfel.GeometricNormal),
                RayEpsilon, MaximumRayDistance};
    const GiSubsurfaceExit exit = GiSubsurfaceWalk(world, ray, GiSurfelHit(surfel),
        float3(surfel.Position), false, random, statistics);
    if (!exit.Reached) return {};
    const GiVector incident = world.Controls->Order == 0
        ? GiDirect(world, exit.Receiver, random, statistics)
        : GiGather(world, cache, previous, exit.Receiver, GiInvalid, random, statistics);
    return GiScale(GiProduct(exit.Throughput, incident), WideMath::Divide(GiFloat(1), GiPi()));
}
void GiBakeSurface(GiWorld world, GiCache cache, device const GiRadiance* previous, device GiRadiance* next,
                   device GiRadiance* total, uint identity, thread GiStatistics& statistics)
{
    const GiSurface surface = cache.Surfaces[identity];
    GiReceiver receiver;
    receiver.Position = float3(surface.Position);
    receiver.Rho = GiReflectance(world, surface.Material, float3(surface.Color), surface.TexCoord,
                                 surface.HasTexCoord != 0);
    const bool subsurface = surface.Material < world.Frame->MaterialCount &&
        world.Materials[surface.Material].Model == MaterialDielectric &&
        world.Materials[surface.Material].Optical.w > 0;
    constant auto& c = *world.Controls;
    const uint budget = c.Order == 0 ? c.DirectSamples : c.GatherSamples;
    const GiNumber weight = WideMath::Divide(GiFloat(1), GiFloat(float(budget)));
    GiRadiance increment{};
    for (uint side = 0; side < 2; ++side)
    {
        receiver.Normal = float3(surface.ShadingNormal) * (side == 0 ? 1.0f : -1.0f);
        receiver.Geometry = float3(surface.GeometricNormal) * (side == 0 ? 1.0f : -1.0f);
        GiVector irradiance{};
        if (GiNonzero(receiver.Rho) || (subsurface && side == 0))
            for (uint i = c.FirstSample; i < c.FirstSample + c.Samples; ++i)
            {
                SampleContext random =
                    MakeSampleContext(identity, side, c.Surfaces, i, budget, world.Frame->SamplingSequence, world.Frame->SamplingSeed);
                random.PixelSeed = Hash(random.PixelSeed ^ Hash(c.Seed) ^ Hash(c.Order) ^ GiBakeDomain);
                GiVector sample = subsurface
                    ? GiSubsurfaceBake(world, cache, previous, surface, random, statistics)
                    : c.Order == 0 ? GiDirect(world, receiver, random, statistics)
                                   : GiGather(world, cache, previous, receiver, identity, random, statistics);
                irradiance = GiAdd(irradiance, GiScale(sample, weight));
            }
        GiVector reflected = subsurface ? irradiance : GiReflected(receiver, irradiance);
        if (side == 0)
            increment.Front = reflected;
        else
            increment.Back = reflected;
    }
    GiRadiance outgoing = increment;
    if (c.FirstSample)
    {
        outgoing.Front = GiAdd(outgoing.Front, next[identity].Front);
        outgoing.Back = GiAdd(outgoing.Back, next[identity].Back);
    }
    GiRadiance combined = outgoing;
    if (c.Order)
    {
        combined.Front = GiAdd(total[identity].Front, increment.Front);
        combined.Back = GiAdd(total[identity].Back, increment.Back);
    }
    if (!GiColor(outgoing.Front) || !GiColor(outgoing.Back) || !GiColor(combined.Front) ||
        !GiColor(combined.Back))
        GiFail(world, GiNumericalError);
    else
    {
        next[identity] = outgoing;
        total[identity] = combined;
    }
}
GiVector GiCameraRadiance(GiWorld world, GiCache cache, device const GiRadiance* total, uint pixel,
                          uint sample, thread GiStatistics& statistics)
{
    constant auto& f = *world.Frame;
    const uint x = f.TileX + pixel % f.Width, y = f.TileY + pixel / f.Width;
    SampleContext random = MakeSampleContext(x, y, f.FullWidth, sample, f.TotalSamples, f.SamplingSequence, f.SamplingSeed);
    float2 jitter = Sample2D(random, 0);
    const float LargestBelowOne = 0.999999940395355224609375f;
    const float rasterX =
        min(LargestBelowOne, GiNarrow(WideMath::Divide(WideMath::Add(GiFloat(float(x)), GiFloat(jitter.x)),
                                                       GiFloat(float(f.FullWidth)))));
    const float rasterY =
        min(LargestBelowOne, GiNarrow(WideMath::Divide(WideMath::Add(GiFloat(float(y)), GiFloat(jitter.y)),
                                                       GiFloat(float(f.FullHeight)))));
    RayData ray{float3(f.CameraPosition),
                normalize(float3(f.CameraForward) +
                          float3(f.CameraRight) * ((2 * rasterX - 1) * f.HalfWidth) +
                          float3(f.CameraUp) * ((1 - 2 * rasterY) * f.HalfHeight)),
                RayEpsilon, MaximumRayDistance};
    ApplyThinLens(ray, f, Sample2D(random, LensSampleDimension));
    random.PixelSeed = Hash(random.PixelSeed ^ Hash(world.Controls->Seed) ^ GiGatherDomain);
    HitData hit{};
    hit.Distance = ray.MaximumDistance;
    IncrementCounter(statistics.Primary);
    bool found = IntersectScene(world.Geometry, ray, false, hit, statistics.Tests);
    GiNumber nearest = GiFloat(found ? hit.Distance : ray.MaximumDistance);
    if (found && hit.MaterialIndex < f.MaterialCount && world.Materials[hit.MaterialIndex].SampledEmitter)
        nearest = WideMath::Add(nearest, GiFloat(RayEpsilon));
    bool front;
    uint rectangle = GiRectangle(world, ray, nearest, front, statistics.Tests);
    if (rectangle != GiInvalid)
    {
        PackedEmitter source = world.Sources[rectangle];
        return front ? GiScale(GiWide(float3(source.Strength)),
                               WideMath::Divide(GiFloat(1), WideMath::Multiply(GiFloat(source.Width),
                                                                               GiFloat(source.Height))))
                     : GiVector{};
    }
    if (!found)
        return GiWide(mix(float3(f.BackgroundBottom), float3(f.BackgroundTop),
                          clamp((ray.Direction.y + 1) / 2, 0.0f, 1.0f)));
    if (hit.MaterialIndex >= f.MaterialCount)
    {
        GiFail(world, GiIndexError);
        return {};
    }
    GiReceiver receiver;
    receiver.Position = ray.Origin + ray.Direction * hit.Distance;
    receiver.Geometry = normalize(hit.GeometricNormal);
    if (dot(receiver.Geometry, ray.Direction) > 0)
        receiver.Geometry = -receiver.Geometry;
    receiver.Normal = normalize(hit.Normal);
    if (dot(receiver.Normal, receiver.Geometry) < 0)
        receiver.Normal = -receiver.Normal;
    if (dot(receiver.Normal, ray.Direction) >= 0)
        receiver.Normal = receiver.Geometry;
    GiVector radiance = GiWide(world.Materials[hit.MaterialIndex].Emission.xyz);
    const PackedMaterial material = world.Materials[hit.MaterialIndex];
    if (material.Model == MaterialDielectric && material.Optical.w > 0)
    {
        const GiSubsurfaceExit exit = GiSubsurfaceWalk(world, ray, hit, receiver.Position,
            hit.FrontFace == 0, random, statistics);
        if (exit.Reached)
        {
            GiVector irradiance = GiDirect(world, exit.Receiver, random, statistics);
            if (f.MaxReflectionBounces)
                irradiance = GiAdd(irradiance, GiGather(world, cache, total, exit.Receiver,
                                                         GiInvalid, random, statistics));
            radiance = GiAdd(radiance, GiScale(GiProduct(exit.Throughput, irradiance),
                                                WideMath::Divide(GiFloat(1), GiPi())));
        }
    }
    else
    {
        receiver.Rho = GiReflectance(world, hit.MaterialIndex, hit.Color, hit.TexCoord,
                                      hit.HasTexCoord != 0);
        if (GiNonzero(receiver.Rho))
        {
            GiVector irradiance = GiDirect(world, receiver, random, statistics);
            if (f.MaxReflectionBounces)
                irradiance = GiAdd(irradiance, GiGather(world, cache, total, receiver,
                                                         GiInvalid, random, statistics));
            radiance = GiAdd(radiance, GiReflected(receiver, irradiance));
        }
    }
    return radiance;
}
kernel void ArvoPointGiInitialize(constant FrameConstants& frame [[buffer(6)]],
                                  device float4* film [[buffer(7)]], device uint* counts [[buffer(8)]],
                                  device uint* targets [[buffer(12)]],
                                  constant GiControls& controls [[buffer(22)]],
                                  device atomic_uint* errors [[buffer(23)]],
                                  uint lane [[thread_position_in_grid]])
{
    uint pixel = controls.Begin + lane;
    if (lane >= controls.Count || pixel >= frame.Width * frame.Height)
        return;
    if (!pixel)
        atomic_store_explicit(errors, 0u, memory_order_relaxed);
    film[pixel] = float4(0);
    counts[pixel] = 0;
    targets[pixel] = frame.TotalSamples;
}
kernel void ArvoPointGiBake(
    device const PackedAccelerationNode* nodes [[buffer(0)]], device const uint* references [[buffer(1)]],
    device const PackedTriangle* triangles [[buffer(2)]], device const PackedSphere* spheres [[buffer(3)]],
    device const PackedMaterial* materials [[buffer(4)]], device const PackedEmitter* sources [[buffer(5)]],
    constant FrameConstants& frame [[buffer(6)]], device GiStatistics* statistics [[buffer(9)]],
    device const PackedTextureDescriptor* textures [[buffer(10)]], device const uint* texels [[buffer(11)]],
    device const InstanceRecord* instances [[buffer(13)]],
    device const InstanceNode* instanceNodes [[buffer(14)]], device atomic_uint* instanceError [[buffer(15)]],
    device const GiSurface* surfaces [[buffer(16)]], device const GiNode* hierarchy [[buffer(17)]],
    device const uint* indices [[buffer(18)]], device const GiRadiance* previous [[buffer(19)]],
    device GiRadiance* next [[buffer(20)]], device GiRadiance* total [[buffer(21)]],
    constant GiControls& controls [[buffer(22)]], device atomic_uint* errors [[buffer(23)]],
    uint lane [[thread_position_in_grid]])
{
    if (lane >= controls.Count)
        return;
    GiStatistics counters{};
    statistics[lane] = counters;
    SceneGeometryBuffers geometry{
        {nodes, references, triangles, spheres, frame.NodeCount, frame.TriangleCount, frame.Acceleration},
        instances,
        instanceNodes,
        instanceError,
        frame.InstanceNodeCount};
    GiWorld world{geometry, materials, sources, textures, texels, &frame, &controls, errors};
    if (!GiConfiguration(world, controls.Surfaces,
                         controls.Order == 0 ? controls.DirectSamples : controls.GatherSamples))
        return;
    if (controls.Order >= frame.MaxReflectionBounces)
    {
        GiFail(world, GiConfigurationError);
        return;
    }
    GiBakeSurface(world, {surfaces, hierarchy, indices}, previous, next, total, controls.Begin + lane,
                  counters);
    statistics[lane] = counters;
}
kernel void ArvoPointGiCamera(
    device const PackedAccelerationNode* nodes [[buffer(0)]], device const uint* references [[buffer(1)]],
    device const PackedTriangle* triangles [[buffer(2)]], device const PackedSphere* spheres [[buffer(3)]],
    device const PackedMaterial* materials [[buffer(4)]], device const PackedEmitter* sources [[buffer(5)]],
    constant FrameConstants& frame [[buffer(6)]], device float4* film [[buffer(7)]],
    device uint* counts [[buffer(8)]], device GiStatistics* statistics [[buffer(9)]],
    device const PackedTextureDescriptor* textures [[buffer(10)]], device const uint* texels [[buffer(11)]],
    device const InstanceRecord* instances [[buffer(13)]],
    device const InstanceNode* instanceNodes [[buffer(14)]], device atomic_uint* instanceError [[buffer(15)]],
    device const GiSurface* surfaces [[buffer(16)]], device const GiNode* hierarchy [[buffer(17)]],
    device const uint* indices [[buffer(18)]], device const GiRadiance* total [[buffer(21)]],
    constant GiControls& controls [[buffer(22)]], device atomic_uint* errors [[buffer(23)]],
    uint lane [[thread_position_in_grid]])
{
    if (lane >= controls.Count)
        return;
    GiStatistics counters{};
    statistics[lane] = counters;
    SceneGeometryBuffers geometry{
        {nodes, references, triangles, spheres, frame.NodeCount, frame.TriangleCount, frame.Acceleration},
        instances,
        instanceNodes,
        instanceError,
        frame.InstanceNodeCount};
    GiWorld world{geometry, materials, sources, textures, texels, &frame, &controls, errors};
    if (!GiConfiguration(world, frame.Width * frame.Height, frame.TotalSamples))
        return;
    uint pixel = controls.Begin + lane, completed = counts[pixel];
    float4 moments = film[pixel];
    if (completed != controls.FirstSample)
    {
        GiFail(world, GiConfigurationError);
        return;
    }
    for (uint sample = controls.FirstSample; sample < controls.FirstSample + controls.Samples; ++sample)
    {
        GiVector radiance =
            GiCameraRadiance(world, {surfaces, hierarchy, indices}, total, pixel, sample, counters);
        float3 value = GiNarrow(radiance);
        if (!GiColor(radiance) || !all(isfinite(value)))
        {
            GiFail(world, GiNumericalError);
            return;
        }
        AccumulateFilmSample(moments, completed, value);
        if (!all(isfinite(moments)))
        {
            GiFail(world, GiNumericalError);
            return;
        }
    }
    film[pixel] = moments;
    counts[pixel] = completed;
    statistics[lane] = counters;
}
