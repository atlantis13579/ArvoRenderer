// Independent Compute VCM. Georgiev et al. (2012), "Light Transport Simulation
// with Vertex Connection and Merging"; "Implementing VCM", rev. 2, section 4.
// https://www.iliyan.com/publications/ImplementingVCM/ImplementingVCM_TechRep2012_rev2.pdf
//
// Each lane writes one bounded camera OR light walk. Vertices and path metadata
// have separate UAVs, keeping every structured-buffer stride below 2048 bytes.
// Arrival/return densities include delta branch masses and geometric Jacobians.
// Stored incoming directions are actual traced directions, not reconstructed
// from offset positions. Fixed depth avoids unaccounted roulette survival PMFs.
// The coupling entry handles source hits, source/surface connections, lens
// splats and disk merges with one joint VC/VM denominator. Only changed BSDF
// directions are reevaluated; unchanged delta events retain their branch mass.
// Frame kernels sort a spatial grid, gather camera estimates, then reuse the
// index to reduce lens splats deterministically and average complete passes.
// The Compute RHI keeps light paths global, streams camera batches, and retires
// each command batch before observing cancellation or releasing job resources.
// Connections and merges dispatch separately at each camera prefix. This keeps
// the large visibility/BSDF/MIS program out of the merging grid loop, while
// preserving every strategy, photon and double-precision accumulation step.
// Compile with -HV 2021 -Gis; the joint MIS leaf requires IEEE-strict arithmetic.
#include "Types.hlsl"
#include "BsdfTypes.hlsl"
#include "EmitterData.hlsl"
#include "VertexConnectionMergingMis.hlsl"
#include "CameraEndpoint.hlsl"
#include "LightSelection.hlsl"
#include "LightEmission.hlsl"
#include "SurfaceEmission.hlsl"
#include "EmitterEndpoint.hlsl"
#include "SubsurfaceScattering.hlsl"

static const float Pi = 3.14159265358979323846f, RayEpsilon = 1e-4f;
static const float MaximumRayDistance = 1e30f, BasisParallelThreshold = 0.999f;
static const uint ImportanceBit = 1;
static const uint VcmCameraVertex = 0, VcmLightVertex = 1, VcmSurfaceVertex = 2;
static const uint VcmEnvironmentVertex = 3, VcmInvalidVertex = 4, VcmVolumeVertex = 5;
static const uint VcmInvalidInput = 0, VcmNullEndpoint = 1, VcmVertexLimit = 2;
static const uint VcmEscaped = 3, VcmAbsorbed = 4, VcmNumericalFailure = 5;
static const uint VcmCameraStream = 0x56434341u, VcmLightStream = 0x56434c49u;
static const uint VcmBounceKey = 0x9e3779b9u, VcmLobeKey = 0xa511e9b3u;
static const uint VcmLensDimension = 0x4c454e53u, VcmSourceDimension = 0x56435352u;
static const uint VcmPositionDimension = 0x5643504fu, VcmDirectionDimension = 0x56434449u;
static const uint VcmInteriorFlightDimension = 0x5643464cu, VcmInteriorPhaseDimension = 0x56435048u;
static const uint VcmWalkThreads = 64, VcmInvalidConfiguration = 1, VcmFailedArithmetic = 2;

struct VcmPathVertex
{
    float3 Position;
    uint Kind;
    float3 ToPrevious;
    uint Material;
    float3 Normal;
    uint FrontFace;
    float3 Geometry;
    uint Emitter;
    float3 Color;
    uint Primitive;
    float3 Throughput;
    float Pdf;
    float3 Continuation;
    float ReversePdf;
    uint Event;
    float RelativeIndex;
    uint Reserved0, Reserved1;
    VcmDensity Arrival, ReturnToPrevious;
};
struct VcmPath
{
    uint Count, Limit, Mode, End;
    float3 LaunchDirection;
    float PositionPdf;
    float3 LaunchWeight;
    float DirectionPdf;
    float3 EndpointValue;
    float SelectionProbability;
    float3 EndpointNormal;
    uint Emitter;
    float2 Lens, Raster;
    uint EndpointFlags, VertexBase, Reserved0, Reserved1;
};
struct VcmStatistics
{
    uint2 Primary, Shadow, Reflection, Tests;
};
struct VcmRandomInput
{
    uint PixelX, PixelY, PixelSeed, Index, Count, Sequence;
};
struct VcmScatteringSample
{
    float3 Direction, Weight;
    float Pdf, ReversePdf, RelativeIndex;
    uint Event;
};
struct VcmScatteringEvaluation
{
    float3 Value;
    float Pdf, ReversePdf;
};
struct VcmCouplingQuery
{
    uint LightPath, CameraPath, S, T;
    float2 Lens;
    uint Merge, Reserved;
};
struct VcmCouplingParameters
{
    // Merge population counts ALL attempted light walks. Light-tracing density
    // is that population divided by full-film pixels, including cropped jobs.
    double MergeRadius, MergingSamples, LightTracingSamples;
    uint MaximumVertices, Heuristic, NextEventEstimation, QueryCount, QueryBegin, OutputOffset;
};
struct VcmContribution
{
    float3 Radiance;
    uint Flags;
    double MisWeight;
    float2 Raster;
};
struct VcmIndexEntry
{
    uint X, Y, Z, Identity;
};
struct VcmFilmPixel
{
    double Red, Green, Blue, PassRed, PassGreen, PassBlue;
    uint CompletedPasses, Reserved0, Reserved1, Reserved2;
};
struct VcmSplat
{
    float3 Radiance;
    uint Pixel;
};
struct VcmFrameParameters
{
    double InitialRadius, Alpha;
    // The complete light population persists while camera slots stream in batches.
    uint LightPathBase, LightPathCount, CameraPathBase, LensPathIndex;
    // Capacity is a power of two covering LightPathCount*(MaximumVertices-2).
    uint IndexCapacity, WorkBegin, WorkCount, CameraPixelBegin;
    uint SortSpan, SortStride, CameraPrefix, Reserved;
};
struct VcmImageDomain
{
    uint Width, Height, FullWidth, FullHeight, TileX, TileY;
};
static const uint VcmPairDimension = 0x56435052u, VcmSplatLensDimension = 0x56434c53u;
static const uint VcmSplatContribution = 1, VcmInvalidContribution = 2;
void IncrementCounter(inout uint2 counter)
{
    ++counter.x;
    if (counter.x == 0)
        ++counter.y;
}
bool VcmFiniteVector(float3 value)
{
    return isfinite(value.x) && isfinite(value.y) && isfinite(value.z);
}
bool VcmValidPower(float3 value)
{
    return VcmFiniteVector(value) && value.x >= 0 && value.y >= 0 && value.z >= 0;
}
bool VcmNonzero(float3 value)
{
    return value.x > 0 || value.y > 0 || value.z > 0;
}
bool VcmValidDirection(float3 value)
{
    const float squared = dot(value, value);
    return VcmFiniteVector(value) && squared > 0 && isfinite(squared);
}
RayData VcmRay(float3 origin, float3 direction)
{
    RayData ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.MinimumDistance = RayEpsilon;
    ray.MaximumDistance = MaximumRayDistance;
    return ray;
}
float3 VcmDepart(VcmPathVertex vertex, float3 direction)
{
    if (vertex.Kind == VcmVolumeVertex) return vertex.Position;
    return vertex.Position +
           vertex.Geometry * (dot(vertex.Geometry, direction) < 0 ? -RayEpsilon : RayEpsilon);
}

#include "MaterialTypes.hlsl"
struct VcmTextureRecord
{
    uint OffsetTexels, Width, Height, Reserved;
};
ByteAddressBuffer Nodes : register(t0);
StructuredBuffer<uint> PrimitiveRefs : register(t1);
StructuredBuffer<PackedTriangle> Triangles : register(t2);
StructuredBuffer<PackedSphere> Spheres : register(t3);
StructuredBuffer<PackedMaterial> Materials : register(t4);
StructuredBuffer<PackedEmitter> Emitters : register(t5);
StructuredBuffer<VcmTextureRecord> Textures : register(t6);
StructuredBuffer<uint> TextureTexels : register(t7);
RWStructuredBuffer<VcmPathVertex> VcmVertices : register(u0);
RWStructuredBuffer<VcmPath> VcmPaths : register(u1);
RWStructuredBuffer<VcmContribution> VcmContributions : register(u2);
RWStructuredBuffer<VcmStatistics> VcmCounters : register(u3);
RWStructuredBuffer<VcmIndexEntry> VcmIndex : register(u4);
RWStructuredBuffer<VcmFilmPixel> VcmFilm : register(u5);
RWByteAddressBuffer VcmError : register(u7);
RWStructuredBuffer<VcmSplat> VcmSplats : register(u8);
RWStructuredBuffer<float4> VcmResolved : register(u9);
RWStructuredBuffer<uint> VcmSampleCounts : register(u10);
RWStructuredBuffer<uint> VcmSampleTargets : register(u11);
StructuredBuffer<VcmCouplingQuery> VcmCouplingQueries : register(t11);
float VcmMeanFreePath(uint material) { return Materials[material].Optical.w; }
float3 VcmInteriorAlbedo(uint material)
{
    return Materials[material].AlbedoReflectivity.xyz * Materials[material].ConductorExtinction.w;
}
float VcmAnisotropy(uint material) { return Materials[material].ConductorIndex.w; }
cbuffer VcmCouplingControls : register(b2)
{
    VcmCouplingParameters Coupling;
};
cbuffer VcmFrameControls : register(b3)
{
    VcmFrameParameters Frame;
};
// Matches ComputeShaderRayTracingRHI's existing frame data. A CBV includes
// explicit tail padding; the shared scene fields retain their original offsets.
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
cbuffer VcmWalkControls : register(b1)
{
    float3 EmissionCenter;
    float EmissionRadius;
    uint EmitterCount, PathBegin, PathCount, WalkSeed;
    uint WalkPass, WalkLimit, OutputOffset, WalkReserved;
};
bool VcmImportanceSampling() { return (TransportFlags & ImportanceBit) != 0; }
static const uint PixelSeedMultiplier = 9781u, SampleSeedMultiplier = 6271u;
static const uint SecondDimensionScramble = 0x68bc21ebu;
uint Hash(uint value)
{
    static const uint MixerA = 0x7feb352du, MixerB = 0x846ca68bu;
    static const uint HalfWord = 16, MiddleShift = 15;
    value = (value ^ (value >> HalfWord)) * MixerA;
    value = (value ^ (value >> MiddleShift)) * MixerB;
    return value ^ (value >> HalfWord);
}
float Random01(uint value)
{
    static const uint MantissaMask = 0x00ffffffu;
    static const float MantissaRange = 16777216.0f;
    return float(Hash(value) & MantissaMask) / MantissaRange;
}
#include "Sequence.hlsl"
#include "Bounds.hlsl"
#include "Triangle.hlsl"
#include "Sphere.hlsl"
#include "Traversal.hlsl"
#include "Fresnel.hlsl"
#include "Ggx.hlsl"
#include "Diffuse.hlsl"
#include "Conductor.hlsl"
#include "Dielectric.hlsl"
#include "SurfaceScattering.hlsl"

float2 VcmDraw(VcmRandomInput input, uint dimension)
{
    SampleContext context;
    context.PixelX = input.PixelX;
    context.PixelY = input.PixelY;
    context.PixelSeed = input.PixelSeed;
    context.SampleIndex = input.Index;
    context.SampleCount = input.Count;
    context.Sequence = input.Sequence;
    context.SamplingSeed = SamplingSeed;
    return Sample2D(context, dimension);
}
void VcmStore(uint index, VcmPathVertex vertex)
{
    VcmVertices[index] = vertex;
}
float VcmMirrorProbability(VcmPathVertex vertex)
{
    PackedMaterial material = Materials[vertex.Material];
    return material.Model == MaterialLegacy ? saturate(material.AlbedoReflectivity.w) : 0;
}
SurfaceBsdf VcmSurfaceFrame(VcmPathVertex vertex, float3 view, uint mode)
{
    PackedMaterial material = Materials[vertex.Material];
    if (material.Model == MaterialLegacy)
        material.Model = MaterialDiffuse;
    HitData hit = (HitData)0;
    hit.Normal = dot(vertex.Normal, view) < 0 ? -vertex.Normal : vertex.Normal;
    hit.GeometricNormal = vertex.Geometry;
    hit.FrontFace = dot(vertex.Geometry, view) > 0;
    return MakeSurfaceBsdf(material, hit, view, vertex.Color, mode);
}
VcmScatteringEvaluation VcmContinuousEvaluation(VcmPathVertex vertex, float3 view, float3 outgoing, uint mode)
{
    const SurfaceEvaluation evaluated = EvaluateSurface(VcmSurfaceFrame(vertex, view, mode), outgoing);
    VcmScatteringEvaluation result;
    result.Value = evaluated.Value;
    result.Pdf = evaluated.Pdf;
    result.ReversePdf = evaluated.ReversePdf;
    return result;
}
VcmScatteringSample VcmContinuousSample(VcmPathVertex vertex, uint mode, float2 variates, float lobe)
{
    const SurfaceSample sampled = SampleSurface(VcmSurfaceFrame(vertex, vertex.ToPrevious, mode), variates, lobe);
    VcmScatteringSample result;
    result.Direction = sampled.Direction;
    result.Weight = sampled.Weight;
    result.Pdf = sampled.Pdf;
    result.ReversePdf = sampled.ReversePdf;
    result.RelativeIndex = sampled.RelativeIndex;
    result.Event = sampled.Event;
    return result;
}
float VcmLinearChannel(uint channel)
{
    static const float ByteMaximum = 255, SrgbThreshold = 0.04045f, LinearScale = 12.92f;
    static const float SrgbOffset = 0.055f, SrgbScale = 1.055f, SrgbExponent = 2.4f;
    const float encoded = channel / ByteMaximum;
    return encoded <= SrgbThreshold ? encoded / LinearScale
                                    : pow((encoded + SrgbOffset) / SrgbScale, SrgbExponent);
}
float3 VcmTexel(VcmTextureRecord texture, int x, int y)
{
    const uint wrappedX = uint((x % int(texture.Width) + int(texture.Width)) % int(texture.Width));
    const uint wrappedY = uint((y % int(texture.Height) + int(texture.Height)) % int(texture.Height));
    const uint pixel = TextureTexels[texture.OffsetTexels + wrappedY * texture.Width + wrappedX];
    static const uint ChannelMask = 255, GreenShift = 8, BlueShift = 16;
    return float3(VcmLinearChannel(pixel & ChannelMask),
                  VcmLinearChannel((pixel >> GreenShift) & ChannelMask),
                  VcmLinearChannel((pixel >> BlueShift) & ChannelMask));
}
float3 VcmSurfaceColor(HitData hit)
{
    const uint textureIndex = Materials[hit.MaterialIndex].DiffuseTextureIndex;
    if (hit.HasTexCoord == 0 || textureIndex >= TextureCount || !all(isfinite(hit.TexCoord)))
        return hit.Color;
    const VcmTextureRecord texture = Textures[textureIndex];
    if (texture.Width == 0 || texture.Height == 0 || texture.Height > TextureTexelCount / texture.Width ||
        texture.OffsetTexels > TextureTexelCount ||
        texture.Width * texture.Height > TextureTexelCount - texture.OffsetTexels)
        return hit.Color;
    const float2 location =
        frac(float2(hit.TexCoord.x, 1 - hit.TexCoord.y)) * float2(texture.Width, texture.Height) - 0.5f;
    const int2 lower = int2(floor(location));
    const float2 blend = frac(location);
    const float3 first =
        lerp(VcmTexel(texture, lower.x, lower.y), VcmTexel(texture, lower.x + 1, lower.y), blend.x);
    const float3 second =
        lerp(VcmTexel(texture, lower.x, lower.y + 1), VcmTexel(texture, lower.x + 1, lower.y + 1), blend.x);
    return hit.Color * lerp(first, second, blend.y);
}
uint VcmFindSource(uint primitive, bool environment)
{
    for (uint i = 0; i < EmitterCount; ++i)
    {
        PackedEmitter source = Emitters[i];
        if (environment ? source.Kind == PackedEmitterEnvironment
                        : primitive != InvalidPrimitiveIndex && source.Primitive == primitive)
            return i;
    }
    return InvalidPrimitiveIndex;
}
VcmPathVertex VcmIntersect(RayData ray, inout VcmStatistics statistics)
{
    VcmPathVertex vertex = (VcmPathVertex)0;
    vertex.Material = vertex.Primitive = vertex.Emitter = InvalidPrimitiveIndex;
    vertex.Reserved0 = InvalidPrimitiveIndex;
    vertex.ToPrevious = -ray.Direction;
    HitData hit = (HitData)0;
    hit.Distance = ray.MaximumDistance;
    const bool found = IntersectScene(ray, false, hit, statistics.Tests);
    if (found && hit.MaterialIndex < MaterialCount && hit.FrontFace == 0 &&
        VcmMeanFreePath(hit.MaterialIndex) > 0)
        vertex.Reserved0 = hit.MaterialIndex;
    // Analytic area sources replace their raster geometry proxies at equal depth.
    float nearest = found ? hit.Distance + RayEpsilon : ray.MaximumDistance;
    for (uint i = 0; i < EmitterCount; ++i)
    {
        const PackedEmitter source = Emitters[i];
        if (source.Kind != PackedEmitterRectangle || !(source.Width > 0 && source.Height > 0))
            continue;
        IncrementCounter(statistics.Tests);
        const float3 normal = normalize(source.Direction);
        const float denominator = dot(normal, ray.Direction);
        if (denominator == 0)
            continue;
        const float distance = dot(source.Origin - ray.Origin, normal) / denominator;
        if (!(distance > ray.MinimumDistance && distance < ray.MaximumDistance && distance <= nearest))
            continue;
        const LightEmissionFrame frame = EmissionFrame(normal);
        const float3 position = ray.Origin + distance * ray.Direction, relative = position - source.Origin;
        if (abs(dot(relative, frame.Tangent)) <= source.Width / 2 &&
            abs(dot(relative, frame.Bitangent)) <= source.Height / 2)
        {
            nearest = distance;
            vertex.Position = position;
            vertex.Geometry = vertex.Normal = normal;
            vertex.Emitter = i;
        }
    }
    if (vertex.Emitter != InvalidPrimitiveIndex)
    {
        vertex.Kind = VcmLightVertex;
        return vertex;
    }
    if (!found)
    {
        vertex.Kind = VcmEnvironmentVertex;
        vertex.Emitter = VcmFindSource(InvalidPrimitiveIndex, true);
        return vertex;
    }
    if (hit.MaterialIndex >= MaterialCount)
    {
        vertex.Kind = VcmInvalidVertex;
        return vertex;
    }
    vertex.Kind = VcmSurfaceVertex;
    vertex.Position = ray.Origin + hit.Distance * ray.Direction;
    vertex.Geometry = hit.GeometricNormal;
    vertex.Normal = hit.Normal;
    vertex.FrontFace = hit.FrontFace;
    vertex.Material = hit.MaterialIndex;
    vertex.Primitive = hit.PrimitiveIndex;
    vertex.Color = VcmSurfaceColor(hit);
    vertex.Emitter = VcmFindSource(hit.PrimitiveIndex, false);
    return vertex;
}
VcmPathVertex VcmLoad(uint index) { return VcmVertices[index]; }
VcmPath VcmLoadPath(uint index) { return VcmPaths[index]; }
VcmIndexEntry VcmLoadIndex(uint index) { return VcmIndex[index]; }
void VcmStoreIndex(uint index, VcmIndexEntry entry) { VcmIndex[index] = entry; }
VcmFilmPixel VcmLoadFilm(uint index) { return VcmFilm[index]; }
void VcmStoreFilm(uint index, VcmFilmPixel pixel) { VcmFilm[index] = pixel; }
VcmSplat VcmLoadSplat(uint index) { return VcmSplats[index]; }
void VcmStoreSplat(uint index, VcmSplat value) { VcmSplats[index] = value; }
uint VcmSourceCount() { return EmitterCount; }
PackedEmitter VcmSource(uint index) { return Emitters[index]; }
EmissionDomain VcmDomain()
{
    EmissionDomain domain;
    domain.Center = EmissionCenter;
    domain.Radius = EmissionRadius;
    return domain;
}
EmitterEndpointDensity VcmSourcePdf(uint index, float3 normal, float3 direction)
{
    return EmitterEndpointPdf(Emitters, EmitterCount, VcmDomain(), index, normal, direction);
}
CameraEndpointFrame VcmSensor()
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
    return frame;
}
bool VcmUnoccluded(RayData ray, inout VcmStatistics statistics)
{
    IncrementCounter(statistics.Shadow);
    // Closest-hit also checks analytic rectangle emitters and geometry proxies.
    const VcmPathVertex hit = VcmIntersect(ray, statistics);
    return hit.Kind == VcmEnvironmentVertex;
}

uint VcmSegmentInterior(VcmPathVertex source, VcmPathVertex target)
{
    if (source.Kind == VcmVolumeVertex) return source.Material;
    if (target.Kind == VcmVolumeVertex) return target.Material;
    if (source.Kind == VcmLightVertex && source.Reserved0 != InvalidPrimitiveIndex &&
        VcmMeanFreePath(source.Reserved0) > 0)
        return source.Reserved0;
    if (target.Kind == VcmLightVertex && target.Reserved0 != InvalidPrimitiveIndex &&
        VcmMeanFreePath(target.Reserved0) > 0)
        return target.Reserved0;
    const float3 edge = target.Position - source.Position;
    if (source.Kind == VcmSurfaceVertex && VcmMeanFreePath(source.Material) > 0 &&
        dot(source.Geometry, edge) < 0)
        return source.Material;
    if (target.Kind == VcmSurfaceVertex && VcmMeanFreePath(target.Material) > 0 &&
        dot(target.Geometry, -edge) < 0)
        return target.Material;
    return InvalidPrimitiveIndex;
}
double VcmSegmentTransmittance(VcmPathVertex source, VcmPathVertex target)
{
    const uint material = VcmSegmentInterior(source, target);
    return material == InvalidPrimitiveIndex ? 1 :
        double(exp(-length(target.Position - source.Position) / VcmMeanFreePath(material)));
}
VcmDensity VcmEdgeArrival(VcmDensity angular, VcmPathVertex source, VcmPathVertex target)
{
    if (!VcmPositive(angular)) return VcmLogDensity(0, 0);
    const float3 segment = target.Position - source.Position;
    const double x = VcmGeometryComponent(segment.x), y = VcmGeometryComponent(segment.y),
                 z = VcmGeometryComponent(segment.z);
    const double squared = x * x + y * y + z * z;
    if (!(squared > 0) || !VcmFinite(squared)) return VcmLogDensity(0, 0);
    VcmDensity density = VcmSurfaceArrival(angular, segment, target.Geometry);
    if (target.Kind == VcmVolumeVertex)
        density = VcmLogDensity(angular.Log2Value - VcmLog2Positive(squared), 1);
    if (!VcmPositive(density)) return density;
    const uint material = VcmSegmentInterior(source, target);
    if (material == InvalidPrimitiveIndex)
    {
        if (target.Kind == VcmVolumeVertex) return VcmLogDensity(0, 0);
        return density;
    }
    const double meanFreePath = double(VcmMeanFreePath(material));
    density.Log2Value -= double(length(segment)) / meanFreePath * VcmInverseLnTwo;
    if (target.Kind == VcmVolumeVertex)
        density.Log2Value -= VcmLog2Positive(meanFreePath);
    return VcmLogDensity(density.Log2Value, VcmFinite(density.Log2Value) ? 1 : 0);
}
VcmScatteringSample VcmScatter(VcmPathVertex vertex, uint mode, float2 variates, float lobe)
{
    if (vertex.Kind == VcmVolumeVertex)
    {
        const SubsurfaceDirection phase = SubsurfaceSampleDirection(-vertex.ToPrevious,
            VcmAnisotropy(vertex.Material), variates, VcmImportanceSampling());
        VcmScatteringSample sample = (VcmScatteringSample)0;
        sample.Direction = phase.Direction;
        sample.Weight = VcmInteriorAlbedo(vertex.Material) * phase.Weight;
        sample.Pdf = sample.ReversePdf = phase.Pdf;
        sample.RelativeIndex = 1;
        sample.Event = ScatteringGlossy;
        return sample;
    }
    const float mirror = VcmMirrorProbability(vertex);
    if (lobe < mirror)
    {
        VcmScatteringSample sample = (VcmScatteringSample)0;
        const float3 normal = dot(vertex.Normal, vertex.ToPrevious) < 0 ? -vertex.Normal : vertex.Normal;
        const float3 outgoing = normalize(-vertex.ToPrevious + normal * (2 * dot(normal, vertex.ToPrevious)));
        const float viewGeometry = dot(vertex.Geometry, vertex.ToPrevious),
                    lightGeometry = dot(vertex.Geometry, outgoing);
        const float viewShading = dot(normal, vertex.ToPrevious), lightShading = dot(normal, outgoing);
        if (!(viewGeometry * lightGeometry > 0 && viewShading > 0 && lightShading > 0))
            return sample;
        const float correction =
            mode == TransportRadiance ? 1 : abs(lightGeometry * viewShading / (viewGeometry * lightShading));
        sample.Direction = outgoing;
        sample.Weight = float3(correction, correction, correction);
        sample.Pdf = sample.ReversePdf = mirror;
        sample.RelativeIndex = 1;
        sample.Event = ScatteringDelta | ScatteringReflection;
        return sample;
    }
    VcmScatteringSample sample = VcmContinuousSample(vertex, mode, variates, lobe);
    // Mixture selection cancels from throughput, but remains in both PDFs.
    sample.Pdf *= 1 - mirror;
    sample.ReversePdf *= 1 - mirror;
    return sample;
}
VcmPath VcmEmptyPath(uint mode, uint limit, uint base)
{
    VcmPath path = (VcmPath)0;
    path.Mode = mode;
    path.Limit = limit;
    path.VertexBase = base;
    path.Emitter = InvalidPrimitiveIndex;
    path.End = VcmInvalidInput;
    return path;
}
VcmPath VcmExtend(VcmPath path, VcmPathVertex previous, RayData ray, float3 power, VcmRandomInput random,
                  inout VcmStatistics statistics)
{
    VcmDensity proposal = VcmDensityFromValue(path.DirectionPdf);
    const uint stream = path.Mode == TransportRadiance ? VcmCameraStream : VcmLightStream;
    uint interior = InvalidPrimitiveIndex;
    path.End = VcmVertexLimit;
    while (path.Count < path.Limit)
    {
        if (!VcmValidPower(power) || !VcmPositive(proposal) || !VcmFiniteVector(ray.Origin) ||
            !VcmValidDirection(ray.Direction))
        {
            path.End = VcmNumericalFailure;
            break;
        }
        if (!VcmNonzero(power))
        {
            path.End = VcmAbsorbed;
            break;
        }
        if (path.Count == 1 && path.Mode == TransportRadiance)
            IncrementCounter(statistics.Primary);
        else
            IncrementCounter(statistics.Reflection);
        VcmPathVertex vertex = VcmIntersect(ray, statistics);
        const bool projectedDisk = path.Mode == TransportImportance && path.Count == 1 &&
                                   (path.EndpointFlags & EmissionInfinite) != 0;
        if (path.Count == 1 && vertex.Reserved0 != InvalidPrimitiveIndex)
            interior = vertex.Reserved0;
        const uint flightDimension = stream ^ (path.Count * VcmBounceKey);
        if (interior != InvalidPrimitiveIndex)
        {
            const float endpoint = vertex.Kind == VcmEnvironmentVertex ? ray.MaximumDistance :
                length(vertex.Position - ray.Origin);
            const SubsurfaceFlight flight = SubsurfaceFreeFlight(VcmMeanFreePath(interior),
                VcmInteriorAlbedo(interior), ray.Direction, endpoint,
                VcmDraw(random, flightDimension ^ VcmInteriorFlightDimension).x);
            if (flight.Scattered != 0)
            {
                // The local phase and albedo are evaluated once at a merge.
                vertex.Kind = VcmVolumeVertex;
                vertex.Position = ray.Origin + ray.Direction * flight.Distance;
                vertex.Material = interior;
                vertex.Emitter = vertex.Primitive = InvalidPrimitiveIndex;
                vertex.Geometry = vertex.Normal = float3(0, 0, 0);
                vertex.FrontFace = 0;
            }
        }
        vertex.Throughput = power;
        if (vertex.Kind == VcmEnvironmentVertex)
            vertex.Arrival = proposal;
        else if (projectedDisk)
            vertex.Arrival =
                VcmDiskArrival(VcmDensityFromValue(path.PositionPdf), vertex.Geometry, ray.Direction);
        else
            vertex.Arrival = VcmEdgeArrival(proposal, previous, vertex);
        if (vertex.Kind == VcmInvalidVertex || !VcmPositive(vertex.Arrival) ||
            !VcmFiniteVector(vertex.Position))
        {
            path.End = VcmNumericalFailure;
            break;
        }
        const uint slot = path.VertexBase + path.Count;
        ++path.Count;
        VcmStore(slot, vertex);
        if (vertex.Kind != VcmSurfaceVertex && vertex.Kind != VcmVolumeVertex)
        {
            path.End = vertex.Kind == VcmEnvironmentVertex ? VcmEscaped : VcmAbsorbed;
            break;
        }
        if (path.Count == path.Limit)
            break;
        const uint dimension = stream ^ (path.Count * VcmBounceKey);
        const VcmScatteringSample sample = VcmScatter(vertex, path.Mode,
            VcmDraw(random, dimension ^ (vertex.Kind == VcmVolumeVertex ? VcmInteriorPhaseDimension : 0)),
                                                      VcmDraw(random, dimension ^ VcmLobeKey).x);
        vertex.Continuation = sample.Direction;
        vertex.Pdf = sample.Pdf;
        vertex.ReversePdf = sample.ReversePdf;
        vertex.RelativeIndex = sample.RelativeIndex;
        vertex.Event = sample.Event;
        VcmStore(slot, vertex);
        if (sample.Event == 0)
        {
            path.End = VcmAbsorbed;
            break;
        }
        if (!(sample.Pdf > 0 && sample.ReversePdf >= 0 && sample.RelativeIndex > 0) ||
            !isfinite(sample.Pdf) || !isfinite(sample.ReversePdf) || !isfinite(sample.RelativeIndex) ||
            !VcmValidDirection(sample.Direction) || !VcmValidPower(sample.Weight))
        {
            path.End = VcmNumericalFailure;
            break;
        }
        const VcmDensity reverse = VcmDensityFromValue(sample.ReversePdf);
        // The reverse event returns to an angular infinite endpoint, even when
        // the actual first arrival used a projected disk rather than p(direction).
        if (projectedDisk)
            vertex.ReturnToPrevious = reverse;
        else
            vertex.ReturnToPrevious = VcmEdgeArrival(reverse, vertex, previous);
        VcmStore(slot, vertex);
        previous = vertex;
        power *= sample.Weight;
        proposal = VcmDensityFromValue(sample.Pdf);
        ray = VcmRay(VcmDepart(vertex, sample.Direction), sample.Direction);
        if (vertex.Kind == VcmSurfaceVertex && VcmMeanFreePath(vertex.Material) > 0 &&
            (sample.Event & ScatteringTransmission) != 0)
            interior = vertex.FrontFace != 0 ? vertex.Material : InvalidPrimitiveIndex;
    }
    return path;
}
VcmPath VcmTraceCamera(CameraEndpointFrame frame, float2 raster, VcmRandomInput random, uint limit, uint base,
                       inout VcmStatistics statistics)
{
    VcmPath path = VcmEmptyPath(TransportRadiance, limit, base);
    if (limit == 0 || limit > VcmMaximumVertices)
        return path;
    const CameraEndpointSample camera =
        SampleCameraEndpoint(frame, raster, VcmDraw(random, VcmLensDimension));
    if ((camera.Flags & CameraEndpointValid) == 0)
    {
        path.End = VcmNullEndpoint;
        return path;
    }
    path.LaunchDirection = camera.Direction;
    path.PositionPdf = camera.PositionPdf;
    path.LaunchWeight = float3(camera.Weight, camera.Weight, camera.Weight);
    path.DirectionPdf = camera.DirectionPdf;
    path.EndpointNormal = frame.Forward;
    path.EndpointValue = float3(camera.Importance, camera.Importance, camera.Importance);
    path.EndpointFlags = camera.Flags;
    path.Lens = camera.LensPoint;
    path.Raster = camera.Raster;
    VcmPathVertex root = (VcmPathVertex)0;
    root.Kind = VcmCameraVertex;
    root.Position = camera.Origin;
    root.Geometry = root.Normal = frame.Forward;
    root.Material = root.Primitive = root.Emitter = InvalidPrimitiveIndex;
    root.Reserved0 = InvalidPrimitiveIndex;
    root.Throughput = path.LaunchWeight;
    root.Arrival = VcmDensityFromValue(camera.PositionPdf);
    path.Count = 1;
    VcmStore(base, root);
    if (!VcmPositive(root.Arrival) || !VcmValidPower(root.Throughput) || !VcmFiniteVector(root.Position))
    {
        path.End = VcmNumericalFailure;
        return path;
    }
    return VcmExtend(path, root, VcmRay(root.Position, camera.Direction), root.Throughput, random,
                     statistics);
}
VcmPath VcmTraceLight(EmitterEndpoint endpoint, VcmRandomInput random, uint limit, uint base,
                      inout VcmStatistics statistics)
{
    VcmPath path = VcmEmptyPath(TransportImportance, limit, base);
    if (limit == 0 || limit > VcmMaximumVertices)
        return path;
    const EmissionSample emission = endpoint.Emission;
    if ((emission.Flags & EmissionValid) == 0)
    {
        path.End = VcmNullEndpoint;
        return path;
    }
    path.LaunchDirection = emission.Direction;
    path.PositionPdf = emission.PositionPdf;
    path.LaunchWeight = emission.Weight;
    path.DirectionPdf = emission.DirectionPdf;
    path.EndpointValue = emission.Value;
    path.EndpointNormal = emission.Normal;
    path.SelectionProbability = endpoint.SelectionProbability;
    path.Emitter = endpoint.EmitterIndex;
    path.EndpointFlags = emission.Flags;
    const float endpointPdf =
        (emission.Flags & EmissionInfinite) != 0 ? emission.DirectionPdf : emission.PositionPdf;
    VcmPathVertex root = (VcmPathVertex)0;
    root.Kind = VcmLightVertex;
    root.Position = emission.Origin;
    root.Geometry = root.Normal = emission.Normal;
    root.Material = InvalidPrimitiveIndex;
    root.Reserved0 = InvalidPrimitiveIndex;
    root.Primitive = endpoint.Primitive;
    root.Emitter = endpoint.EmitterIndex;
    root.Arrival = VcmDensityFromValue(endpointPdf);
    // Source selection is discrete and stays separate from the endpoint density.
    root.Throughput = emission.Value / (endpoint.SelectionProbability * endpointPdf);
    path.Count = 1;
    VcmStore(base, root);
    if (!VcmPositive(root.Arrival) || !(path.SelectionProbability > 0 && path.SelectionProbability <= 1) ||
        !VcmValidPower(root.Throughput) || !VcmFiniteVector(root.Position) ||
        !VcmValidPower(emission.Weight) || !VcmValidDirection(emission.Direction))
    {
        path.End = VcmNumericalFailure;
        return path;
    }
    const float3 origin =
        (emission.Flags & EmissionDeltaPosition) != 0 ? root.Position : VcmDepart(root, emission.Direction);
    return VcmExtend(path, root, VcmRay(origin, emission.Direction), emission.Weight, random, statistics);
}

// Connections replace at most two scattering directions. A merge identifies
// two sampled surfaces as one canonical vertex, but retains both edge positions.
// Accessors keep that distinction explicit without copying entire subpaths.
struct VcmJoinedPath
{
    VcmPath Light, Camera;
    VcmPathVertex Lens;
    CameraEndpointEvaluation LensPdf;
    uint S, T, Count, Merge;
};
struct VcmBridge
{
    float3 Direction;
    double SquaredDistance;
    uint Valid;
};
VcmBridge VcmMakeBridge(float3 displacement)
{
    VcmBridge bridge = (VcmBridge)0;
    const double x = VcmGeometryComponent(displacement.x), y = VcmGeometryComponent(displacement.y),
                 z = VcmGeometryComponent(displacement.z);
    bridge.SquaredDistance = x * x + y * y + z * z;
    const float scale = max(abs(displacement.x), max(abs(displacement.y), abs(displacement.z)));
    if (!(scale > 0) || !isfinite(scale) || !(bridge.SquaredDistance > 0) ||
        !VcmFinite(bridge.SquaredDistance))
        return bridge;
    bridge.Direction = normalize(displacement / scale);
    bridge.Valid = VcmValidDirection(bridge.Direction) ? 1 : 0;
    return bridge;
}
VcmScatteringEvaluation VcmEvaluate(VcmPathVertex vertex, float3 view, float3 outgoing, uint mode)
{
    VcmScatteringEvaluation value = (VcmScatteringEvaluation)0;
    if (vertex.Kind == VcmVolumeVertex)
    {
        const float phase = SubsurfacePhase(dot(-view, outgoing), VcmAnisotropy(vertex.Material));
        value.Value = VcmInteriorAlbedo(vertex.Material) * phase;
        value.Pdf = value.ReversePdf = VcmImportanceSampling() ? phase : 1 / (4 * Pi);
        return value;
    }
    if (vertex.Kind != VcmSurfaceVertex)
        return value;
    value = VcmContinuousEvaluation(vertex, view, outgoing, mode);
    const float continuous = 1 - VcmMirrorProbability(vertex);
    value.Value *= float3(continuous, continuous, continuous);
    value.Pdf *= continuous;
    value.ReversePdf *= continuous;
    return value;
}
bool VcmUsable(VcmPath path, uint mode)
{
    return path.Mode == mode && path.Count > 0 && path.Count <= path.Limit &&
           path.Limit <= VcmMaximumVertices && path.End != VcmInvalidInput && path.End != VcmNumericalFailure;
}
bool VcmValidCoupling(VcmCouplingParameters options)
{
    return options.MaximumVertices >= 2 && options.MaximumVertices <= VcmMaximumVertices &&
           options.Heuristic <= VcmPowerMis && options.MergeRadius >= 0 && VcmFinite(options.MergeRadius) &&
           options.MergingSamples >= 0 && VcmFinite(options.MergingSamples) &&
           options.LightTracingSamples >= 0 && VcmFinite(options.LightTracingSamples);
}
VcmPathVertex VcmJoinedVertex(VcmJoinedPath path, uint index, bool rightSide)
{
    if (index + 1 == path.Count)
        return path.Lens;
    if (index < path.S && !(path.Merge != 0 && index + 1 == path.S && rightSide))
        return VcmLoad(path.Light.VertexBase + index);
    return VcmLoad(path.Camera.VertexBase + path.Count - 1 - index);
}
float3 VcmJoinedDisplacement(VcmJoinedPath path, uint from, uint to)
{
    const bool forward = from < to;
    const VcmPathVertex source = VcmJoinedVertex(path, from, forward);
    const VcmPathVertex destination = VcmJoinedVertex(path, to, !forward);
    return destination.Position - source.Position;
}
float3 VcmJoinedDirection(VcmJoinedPath path, uint from, uint to)
{
    if ((path.Light.EndpointFlags & EmissionInfinite) != 0)
    {
        if (from == 0)
            return path.Light.LaunchDirection;
        if (to == 0)
            return -path.Light.LaunchDirection;
    }
    return VcmMakeBridge(VcmJoinedDisplacement(path, from, to)).Direction;
}
VcmDensity VcmJoinedArrival(VcmJoinedPath path, uint from, uint to, double probability)
{
    const VcmDensity density = VcmDensityFromValue(probability);
    if (to == 0 && (path.Light.EndpointFlags & EmissionInfinite) != 0)
        return density;
    const bool forward = from < to;
    const VcmPathVertex source = VcmJoinedVertex(path, from, forward);
    const VcmPathVertex destination = VcmJoinedVertex(path, to, !forward);
    return VcmEdgeArrival(density, source, destination);
}
VcmWeights VcmCouplingWeights(VcmJoinedPath path, VcmCouplingParameters options)
{
    VcmVertexPdf densities[VcmMaximumVertices];
    VcmStrategy connections[VcmMaximumStrategies], mergings[VcmMaximumStrategies];
    for (uint i = 0; i < VcmMaximumVertices; ++i)
        densities[i] = (VcmVertexPdf)0;
    for (uint cut = 0; cut < VcmMaximumStrategies; ++cut)
    {
        connections[cut] = (VcmStrategy)0;
        mergings[cut] = (VcmStrategy)0;
    }
    for (uint i = 1; i + 1 < path.Count; ++i)
    {
        const VcmPathVertex left = VcmJoinedVertex(path, i, false), right = VcmJoinedVertex(path, i, true);
        if (right.Kind == VcmVolumeVertex)
            densities[i].Flags |= VcmVolumeScattering;
        const bool replaced = path.Merge != 0 ? i + 1 == path.S : (i == path.S || i + 1 == path.S);
        float forward, reverse;
        if (replaced)
        {
            const float3 toLight =
                path.Merge != 0 || i + 1 == path.S ? left.ToPrevious : VcmJoinedDirection(path, i, i - 1);
            const float3 toCamera =
                path.Merge != 0 || i == path.S ? right.ToPrevious : VcmJoinedDirection(path, i, i + 1);
            const VcmScatteringEvaluation evaluation =
                VcmEvaluate(right, toLight, toCamera, TransportImportance);
            forward = evaluation.Pdf;
            reverse = evaluation.ReversePdf;
        }
        else
        {
            // Reversing a camera prefix swaps its forward/reverse proposal PDFs.
            forward = i < path.S ? right.Pdf : right.ReversePdf;
            reverse = i < path.S ? right.ReversePdf : right.Pdf;
            if ((right.Event & ScatteringDelta) != 0)
                densities[i].Flags |= VcmDeltaScattering;
        }
        densities[i + 1].Forward = VcmJoinedArrival(path, i, i + 1, forward);
        densities[i - 1].Reverse = VcmJoinedArrival(path, i, i - 1, reverse);
    }
    const bool infinite = (path.Light.EndpointFlags & EmissionInfinite) != 0;
    const VcmPathVertex source = VcmJoinedVertex(path, 0, false), first = VcmJoinedVertex(path, 1, false);
    const float3 emittedDirection =
        path.S == 1 && !infinite ? VcmJoinedDirection(path, 0, 1) : path.Light.LaunchDirection;
    const EmitterEndpointDensity emission =
        VcmSourcePdf(path.Light.Emitter, source.Geometry, emittedDirection);
    const VcmDensity position =
        VcmDensityFromValue((emission.Flags & EmissionDeltaPosition) != 0 ? 1 : emission.Position);
    const double angularPdf = (emission.Flags & EmissionDeltaDirection) != 0 ? 1 : emission.Direction;
    VcmDensity lightEndpoint = position;
    if (infinite)
    {
        lightEndpoint = VcmDensityFromValue(angularPdf);
        densities[1].Forward = VcmDiskArrival(position, first.Geometry, emittedDirection);
    }
    else
        densities[1].Forward = VcmJoinedArrival(path, 0, 1, angularPdf);
    if ((emission.Flags & (EmissionDeltaPosition | EmissionDeltaDirection)) != 0)
        densities[0].Flags = VcmEndpointNotIntersectable;
    densities[path.Count - 2].Reverse =
        VcmJoinedArrival(path, path.Count - 1, path.Count - 2, path.LensPdf.DirectionPdf);
    densities[path.Count - 1].Flags = VcmEndpointNotIntersectable;
    const VcmDensity cameraEndpoint =
        VcmDensityFromValue((path.LensPdf.Flags & CameraDeltaPosition) != 0 ? 1 : path.LensPdf.PositionPdf);
    for (uint cut = 0; cut < path.Count; ++cut)
    {
        VcmStrategy vc = (VcmStrategy)0;
        vc.LightEndpoint = lightEndpoint;
        vc.LightSelection = emission.SelectionProbability;
        if (cut == 0)
        {
            vc.LightEndpoint = VcmDensityFromValue(1);
            vc.LightSelection = 1;
        }
        vc.CameraEndpoint = cameraEndpoint;
        vc.SampleCount = cut + 1 == path.Count ? options.LightTracingSamples : 1;
        vc.Enabled = cut <= path.Light.Limit && path.Count - cut <= path.Camera.Limit &&
                     (cut != 1 || options.NextEventEstimation != 0) && (path.Count > 2 || cut == 0);
        connections[cut] = vc;
        if (cut >= 2)
        {
            VcmStrategy vm = (VcmStrategy)0;
            vm.LightEndpoint = lightEndpoint;
            vm.CameraEndpoint = cameraEndpoint;
            vm.LightSelection = emission.SelectionProbability;
            vm.SampleCount = options.MergingSamples;
            vm.Enabled = cut <= path.Light.Limit && path.Count + 1 - cut <= path.Camera.Limit;
            mergings[cut] = vm;
        }
    }
    return VcmJointWeights(densities, path.Count, connections, mergings, options.MergeRadius,
                           options.Heuristic, VcmConnections | VcmMergings);
}
struct VcmSpectrum
{
    double Red, Green, Blue;
};
VcmSpectrum VcmSpectrumFrom(float3 value)
{
    VcmSpectrum spectrum;
    spectrum.Red = value.x;
    spectrum.Green = value.y;
    spectrum.Blue = value.z;
    return spectrum;
}
VcmSpectrum VcmMultiply(VcmSpectrum value, float3 factor)
{
    value.Red *= double(factor.x);
    value.Green *= double(factor.y);
    value.Blue *= double(factor.z);
    return value;
}
bool VcmValidSpectrum(VcmSpectrum value)
{
    return value.Red >= 0 && value.Green >= 0 && value.Blue >= 0 && VcmFinite(value.Red) &&
           VcmFinite(value.Green) && VcmFinite(value.Blue);
}
bool VcmNonzeroSpectrum(VcmSpectrum value)
{
    return value.Red > 0 || value.Green > 0 || value.Blue > 0;
}
VcmContribution VcmResolve(VcmContribution result, VcmSpectrum value, double scale, double misWeight)
{
    result.MisWeight = misWeight;
    if (!VcmValidSpectrum(value) || !(scale >= 0 && misWeight >= 0) || !VcmFinite(scale) ||
        !VcmFinite(misWeight))
    {
        result.Flags |= VcmInvalidContribution;
        return result;
    }
    const float3 radiance =
        float3(float(value.Red * scale), float(value.Green * scale), float(value.Blue * scale));
    if (VcmValidPower(radiance))
        result.Radiance = radiance;
    else
        result.Flags |= VcmInvalidContribution;
    return result;
}
float3 VcmEmissionValue(PackedEmitter source, float3 outgoing)
{
    if (source.Kind == PackedEmitterEnvironment)
    {
        const float blend = (1 - outgoing.y) / 2;
        return (1 - blend) * source.BackgroundBottom + blend * source.BackgroundTop;
    }
    if (source.Kind == PackedEmitterRectangle)
    {
        if (dot(source.Direction, outgoing) > 0 && source.Width > 0 && source.Height > 0)
            return source.Strength / (source.Width * source.Height);
        return float3(0, 0, 0);
    }
    return source.Strength;
}
bool VcmVisibleSegment(float3 origin, float3 target, inout VcmStatistics statistics)
{
    const float3 delta = target - origin;
    const float distance = CameraVectorLength(delta);
    if (!(distance > 2 * RayEpsilon) || !isfinite(distance))
        return false;
    RayData ray = VcmRay(origin, delta / distance);
    ray.MaximumDistance = distance - RayEpsilon;
    return VcmUnoccluded(ray, statistics);
}
CameraEndpointEvaluation VcmStoredCameraPdf(VcmPath camera)
{
    CameraEndpointEvaluation pdf = EmptyCameraEvaluation();
    pdf.Raster = camera.Raster;
    pdf.Importance = camera.EndpointValue.x;
    pdf.PositionPdf = camera.PositionPdf;
    pdf.DirectionPdf = camera.DirectionPdf;
    pdf.Flags = camera.EndpointFlags;
    return pdf;
}
VcmContribution VcmCouple(VcmPath light, VcmPath camera, VcmCouplingQuery query,
                          VcmCouplingParameters options, inout VcmStatistics statistics)
{
    VcmContribution result = (VcmContribution)0;
    if (!VcmValidCoupling(options) || query.Merge > 1)
    {
        result.Flags = VcmInvalidContribution;
        return result;
    }
    const uint s = query.S, t = query.T;
    if (!VcmUsable(camera, TransportRadiance) || t == 0 || t > camera.Count || s > VcmMaximumVertices)
        return result;
    const uint count = s + t - query.Merge;
    if (count < 2 || count > options.MaximumVertices || (query.Merge != 0 && (s < 2 || t < 2)))
        return result;
    VcmJoinedPath joined;
    joined.Light = light;
    joined.Camera = camera;
    joined.Lens = VcmLoad(camera.VertexBase);
    joined.LensPdf = VcmStoredCameraPdf(camera);
    joined.S = s;
    joined.T = t;
    joined.Count = count;
    joined.Merge = query.Merge;
    result.Raster = camera.Raster;
    const VcmPathVertex eye = VcmLoad(camera.VertexBase + t - 1);
    if (s == 0)
    {
        if (t < 2 || eye.Emitter >= VcmSourceCount())
            return result;
        joined.Light = VcmEmptyPath(TransportImportance, options.MaximumVertices, 0);
        joined.Light.Emitter = eye.Emitter;
        joined.Light.LaunchDirection = eye.ToPrevious;
        joined.Light.EndpointFlags = VcmSourcePdf(eye.Emitter, eye.Geometry, eye.ToPrevious).Flags;
        const VcmSpectrum value = VcmMultiply(VcmSpectrumFrom(eye.Throughput),
                                              VcmEmissionValue(VcmSource(eye.Emitter), eye.ToPrevious));
        const VcmWeights weights = VcmCouplingWeights(joined, options);
        return VcmResolve(result, value, weights.Connection[0], weights.Connection[0]);
    }
    if (!VcmUsable(light, TransportImportance) || s > light.Count || light.Emitter >= VcmSourceCount())
        return result;
    const VcmPathVertex photon = VcmLoad(light.VertexBase + s - 1);
    if (query.Merge != 0)
    {
        if (!(options.MergeRadius > 0 && options.MergingSamples > 0) || photon.Kind != eye.Kind ||
            (eye.Kind != VcmSurfaceVertex && eye.Kind != VcmVolumeVertex))
            return result;
        const bool volume = eye.Kind == VcmVolumeVertex;
        if (volume && eye.Material != photon.Material)
            return result;
        static const double MinimumNormalAgreement = 0.9, RelativePlaneThickness = 0.1;
        // Subtract in double: rounding a float displacement down to the radius
        // could admit a point outside the grid's mathematically bounded stencil.
        const double x = VcmGeometryComponent(photon.Position.x) - VcmGeometryComponent(eye.Position.x),
                     y = VcmGeometryComponent(photon.Position.y) - VcmGeometryComponent(eye.Position.y),
                     z = VcmGeometryComponent(photon.Position.z) - VcmGeometryComponent(eye.Position.z);
        const double planeDistance =
            VcmAbsolute(x * eye.Geometry.x + y * eye.Geometry.y + z * eye.Geometry.z);
        const double squared = x * x + y * y + z * z;
        if (squared > options.MergeRadius * options.MergeRadius || (!volume &&
            (planeDistance > RelativePlaneThickness * options.MergeRadius ||
             dot(photon.Geometry, eye.Geometry) < MinimumNormalAgreement)))
            return result;
        if (volume && squared > double(2 * RayEpsilon) * (2 * RayEpsilon) &&
            !VcmVisibleSegment(eye.Position, photon.Position, statistics))
            return result;
        const double projected = volume ? 1 : abs(dot(eye.Geometry, photon.ToPrevious));
        if (!(projected > 0))
            return result;
        const VcmScatteringEvaluation bsdf =
            VcmEvaluate(eye, eye.ToPrevious, photon.ToPrevious, TransportRadiance);
        VcmSpectrum value =
            VcmMultiply(VcmMultiply(VcmSpectrumFrom(photon.Throughput), eye.Throughput), bsdf.Value);
        const VcmWeights weights = VcmCouplingWeights(joined, options);
        const double mis = weights.Merging[s];
        if (mis == 0)
            return result;
        // Combine the kernel and MIS scale before exponentiation. The second
        // camera arrival stays physical in the joint MIS, never common-normalized.
        VcmDensity measure = VcmDiskArea(options.MergeRadius);
        if (volume) measure = VcmSphereVolume(options.MergeRadius);
        const double logarithm = VcmLog2Positive(mis) - measure.Log2Value -
                                 VcmLog2Positive(options.MergingSamples);
        const double kernel = logarithm <= 0 ? VcmRelativeExp2(logarithm) : 1 / VcmRelativeExp2(-logarithm);
        const double shadingArea = volume ? 1 : abs(dot(eye.Normal, photon.ToPrevious)) / projected;
        return VcmResolve(result, value, kernel * shadingArea, mis);
    }
    if ((s == 1 && (t == 1 || options.NextEventEstimation == 0)) ||
        (s > 1 && photon.Kind != VcmSurfaceVertex && photon.Kind != VcmVolumeVertex) ||
        (t > 1 && eye.Kind != VcmSurfaceVertex && eye.Kind != VcmVolumeVertex))
        return result;
    const bool infiniteSource = s == 1 && (light.EndpointFlags & EmissionInfinite) != 0;
    float3 direction, target;
    double geometry = 1;
    VcmSpectrum value;
    if (t == 1)
    {
        if (!(options.LightTracingSamples > 0))
            return result;
        const CameraEndpointFrame sensor = VcmSensor();
        const CameraConnectionSample lens = SampleCameraConnection(sensor, photon.Position, query.Lens);
        if ((lens.Flags & CameraEndpointValid) == 0)
            return result;
        direction = lens.Direction;
        target = lens.Position;
        joined.Lens.Position = lens.Position;
        joined.LensPdf = EvaluateCameraEndpoint(sensor, lens.LensPoint, -direction);
        result.Raster = lens.Raster;
        result.Flags |= VcmSplatContribution;
        // The connection's solid-angle lens PDF is NOT the camera ray PDF in MIS.
        // The frame reducer divides this splat by LightTracingSamples exactly once.
        value = VcmSpectrumFrom(photon.Throughput);
        value =
            VcmMultiply(value, VcmEvaluate(photon, photon.ToPrevious, direction, TransportImportance).Value);
        geometry = double(lens.Weight) *
            (photon.Kind == VcmVolumeVertex ? 1 : abs(dot(photon.Normal, direction))) *
            VcmSegmentTransmittance(photon, joined.Lens);
    }
    else
    {
        if (infiniteSource)
            direction = light.LaunchDirection;
        else
        {
            const VcmBridge bridge = VcmMakeBridge(eye.Position - photon.Position);
            if (bridge.Valid == 0)
                return result;
            direction = bridge.Direction;
            geometry = 1 / bridge.SquaredDistance;
        }
        target = VcmDepart(eye, -direction);
        value = VcmSpectrumFrom(eye.Throughput);
        value = VcmMultiply(value, VcmEvaluate(eye, eye.ToPrevious, -direction, TransportRadiance).Value);
        geometry *= eye.Kind == VcmVolumeVertex ? 1 : abs(dot(eye.Normal, direction));
        if (s == 1)
        {
            const double endpointPdf = infiniteSource ? light.DirectionPdf : light.PositionPdf;
            if (!(endpointPdf > 0 && light.SelectionProbability > 0))
            {
                result.Flags |= VcmInvalidContribution;
                return result;
            }
            geometry /= endpointPdf * double(light.SelectionProbability);
            if (!infiniteSource)
                geometry *= VcmSegmentTransmittance(photon, eye);
            if (!infiniteSource && (light.EndpointFlags & EmissionDeltaPosition) == 0)
                geometry *= abs(dot(photon.Geometry, direction));
            value = VcmMultiply(value, VcmEmissionValue(VcmSource(light.Emitter), direction));
        }
        else
        {
            value = VcmMultiply(VcmMultiply(value, photon.Throughput),
                                VcmEvaluate(photon, photon.ToPrevious, direction, TransportImportance).Value);
            geometry *= (photon.Kind == VcmVolumeVertex ? 1 : abs(dot(photon.Normal, direction))) *
                        VcmSegmentTransmittance(photon, eye);
        }
    }
    if (!VcmValidSpectrum(value) || !(geometry >= 0) || !VcmFinite(geometry))
    {
        result.Flags |= VcmInvalidContribution;
        return result;
    }
    if (!VcmNonzeroSpectrum(value) || geometry == 0)
        return result;
    bool visible;
    if (infiniteSource)
        visible = VcmUnoccluded(VcmRay(VcmDepart(eye, -direction), -direction), statistics);
    else
    {
        float3 origin = VcmDepart(photon, direction);
        if (s == 1 && (light.EndpointFlags & EmissionDeltaPosition) != 0)
            origin = photon.Position;
        visible = VcmVisibleSegment(origin, target, statistics);
    }
    if (!visible || (joined.LensPdf.Flags & CameraEndpointValid) == 0)
        return result;
    const VcmWeights weights = VcmCouplingWeights(joined, options);
    return VcmResolve(result, value, geometry * weights.Connection[s], weights.Connection[s]);
}

// A lexicographically sorted uniform grid avoids hash collisions and unordered
// floating-point atomics. The same index buffer later sorts splats by full-film
// pixel and stable source identity. Bitonic compare/exchange pairs are disjoint;
// the dispatcher inserts a UAV barrier between stages. See Batcher (1968),
// "Sorting networks and their applications", and GPU Gems 2, chapter 46.
bool VcmKeyBefore(VcmIndexEntry a, VcmIndexEntry b)
{
    if (a.Identity == InvalidPrimitiveIndex)
        return false;
    if (b.Identity == InvalidPrimitiveIndex)
        return true;
    if (a.X != b.X)
        return a.X < b.X;
    if (a.Y != b.Y)
        return a.Y < b.Y;
    if (a.Z != b.Z)
        return a.Z < b.Z;
    return a.Identity < b.Identity;
}
bool VcmSameCell(VcmIndexEntry a, VcmIndexEntry b)
{
    return a.X == b.X && a.Y == b.Y && a.Z == b.Z;
}
VcmIndexEntry VcmEmptyIndex()
{
    VcmIndexEntry entry = (VcmIndexEntry)0;
    entry.Identity = InvalidPrimitiveIndex;
    return entry;
}
uint VcmEncodeCell(double value)
{
    // Truncation plus one correction implements floor without float-only HLSL
    // floor(). Flipping the sign bit preserves signed order in unsigned keys.
    const int truncated = int(value);
    const int cell = double(truncated) > value ? truncated - 1 : truncated;
    static const uint SignedOrderBit = 0x80000000u;
    return uint(cell) ^ SignedOrderBit;
}
VcmIndexEntry VcmCellKey(float3 position, float3 center, double radius)
{
    VcmIndexEntry key = VcmEmptyIndex();
    if (!(radius > 0) || !VcmFinite(radius) || !VcmFiniteVector(position) || !VcmFiniteVector(center))
        return key;
    const double x = (VcmGeometryComponent(position.x) - VcmGeometryComponent(center.x)) / radius;
    const double y = (VcmGeometryComponent(position.y) - VcmGeometryComponent(center.y)) / radius;
    const double z = (VcmGeometryComponent(position.z) - VcmGeometryComponent(center.z)) / radius;
    // Leave room for the complete neighboring-cell stencil without integer wrap.
    static const double CellLimit = 2147483645.0;
    if (!VcmFinite(x) || !VcmFinite(y) || !VcmFinite(z) || VcmAbsolute(x) > CellLimit ||
        VcmAbsolute(y) > CellLimit || VcmAbsolute(z) > CellLimit)
        return key;
    key.X = VcmEncodeCell(x);
    key.Y = VcmEncodeCell(y);
    key.Z = VcmEncodeCell(z);
    key.Identity = 0;
    return key;
}
double VcmProgressiveRadius(double initial, double alpha, uint pass)
{
    // Every pass estimates radiance with one common bandwidth. This is the VCM
    // schedule r_i = r_0/(i+1)^((1-alpha)/2), not PPM's persistent flux recurrence.
    if (!(initial >= 0 && alpha > 0 && alpha < 1) || !VcmFinite(initial))
        return -1;
    const double exponent = -(1 - alpha) / 2 * VcmLog2Positive(double(pass) + 1);
    return initial * VcmRelativeExp2(exponent);
}
bool VcmValidImage(VcmImageDomain image)
{
    return image.Width > 0 && image.Height > 0 && image.FullWidth > 0 && image.FullHeight > 0 &&
           image.TileX < image.FullWidth && image.TileY < image.FullHeight &&
           image.Width <= image.FullWidth - image.TileX && image.Height <= image.FullHeight - image.TileY &&
           image.FullWidth <= (InvalidPrimitiveIndex - 1) / image.FullHeight;
}
bool VcmValidFrame(VcmFrameParameters frame, VcmCouplingParameters options, VcmImageDomain image)
{
    const uint surfaces = options.MaximumVertices > 2 ? options.MaximumVertices - 2 : 0;
    return VcmValidImage(image) && VcmValidCoupling(options) && surfaces > 0 && frame.LightPathCount > 0 &&
           frame.LightPathCount <= InvalidPrimitiveIndex / surfaces &&
           frame.LightPathBase <= InvalidPrimitiveIndex - frame.LightPathCount && frame.IndexCapacity > 0 &&
           (frame.IndexCapacity & (frame.IndexCapacity - 1)) == 0 &&
           frame.IndexCapacity >= frame.LightPathCount * surfaces &&
           VcmProgressiveRadius(frame.InitialRadius, frame.Alpha, 0) >= 0;
}
VcmCouplingParameters VcmFrameCoupling(VcmFrameParameters frame, VcmCouplingParameters options,
                                       VcmImageDomain image, uint pass)
{
    options.MergeRadius = VcmProgressiveRadius(frame.InitialRadius, frame.Alpha, pass);
    options.MergingSamples = double(frame.LightPathCount);
    options.LightTracingSamples = double(frame.LightPathCount) / (double(image.FullWidth) * image.FullHeight);
    return options;
}
bool VcmBuildGridEntry(uint index, VcmFrameParameters frame, VcmCouplingParameters options, float3 center)
{
    if (index >= frame.IndexCapacity)
        return false;
    VcmIndexEntry entry = VcmEmptyIndex();
    const uint surfaces = options.MaximumVertices - 2;
    if (index < frame.LightPathCount * surfaces && options.MergeRadius > 0)
    {
        const uint pathIndex = frame.LightPathBase + index / surfaces, prefix = 2 + index % surfaces;
        const VcmPath light = VcmLoadPath(pathIndex);
        if (light.End == VcmInvalidInput || light.End == VcmNumericalFailure)
            return false;
        if (VcmUsable(light, TransportImportance) && prefix <= light.Count)
        {
            const VcmPathVertex vertex = VcmLoad(light.VertexBase + prefix - 1);
            if (vertex.Kind == VcmSurfaceVertex || vertex.Kind == VcmVolumeVertex)
            {
                entry = VcmCellKey(vertex.Position, center, options.MergeRadius);
                if (entry.Identity == InvalidPrimitiveIndex)
                    return false;
                entry.Identity = index;
            }
        }
    }
    VcmStoreIndex(index, entry);
    return true;
}
bool VcmSortPair(uint index, VcmFrameParameters frame)
{
    if (index >= frame.IndexCapacity || frame.SortSpan < 2 || frame.SortStride == 0 ||
        frame.SortSpan > frame.IndexCapacity || frame.SortStride >= frame.SortSpan ||
        (frame.SortSpan & (frame.SortSpan - 1)) != 0 || (frame.SortStride & (frame.SortStride - 1)) != 0)
        return false;
    const uint partner = index ^ frame.SortStride;
    if (partner <= index)
        return true;
    if (partner >= frame.IndexCapacity)
        return false;
    const VcmIndexEntry a = VcmLoadIndex(index), b = VcmLoadIndex(partner);
    const bool ascending = (index & frame.SortSpan) == 0;
    if (ascending ? VcmKeyBefore(b, a) : VcmKeyBefore(a, b))
    {
        VcmStoreIndex(index, b);
        VcmStoreIndex(partner, a);
    }
    return true;
}
uint VcmLowerBound(VcmIndexEntry key, uint capacity)
{
    uint begin = 0, end = capacity;
    while (begin < end)
    {
        const uint middle = begin + (end - begin) / 2;
        if (VcmKeyBefore(VcmLoadIndex(middle), key))
            begin = middle + 1;
        else
            end = middle;
    }
    return begin;
}
bool VcmCollect(inout VcmSpectrum sum, VcmContribution contribution)
{
    if ((contribution.Flags & VcmInvalidContribution) != 0 || !VcmValidPower(contribution.Radiance))
        return false;
    sum.Red += double(contribution.Radiance.x);
    sum.Green += double(contribution.Radiance.y);
    sum.Blue += double(contribution.Radiance.z);
    return VcmValidSpectrum(sum);
}
bool VcmCameraEstimateVertex(uint pixel, uint cameraIndex, VcmFrameParameters frame,
                             VcmCouplingParameters options, VcmRandomInput random, float3 center,
                             uint t, bool merging, inout VcmStatistics statistics)
{
    const VcmPath camera = VcmLoadPath(cameraIndex);
    if (!VcmUsable(camera, TransportRadiance) || t < 2 || t > options.MaximumVertices)
        return false;
    const float choice = VcmDraw(random, VcmPairDimension).x;
    if (!(choice >= 0 && choice < 1))
        return false;
    const uint paired = min(uint(double(choice) * double(frame.LightPathCount)), frame.LightPathCount - 1);
    const VcmPath light = VcmLoadPath(frame.LightPathBase + paired);
    if (light.End == VcmInvalidInput || light.End == VcmNumericalFailure)
        return false;
    VcmFilmPixel state = VcmLoadFilm(pixel);
    VcmSpectrum sum;
    sum.Red = state.PassRed;
    sum.Green = state.PassGreen;
    sum.Blue = state.PassBlue;
    // A pass starts at the first surface prefix, even for an escaped/null walk.
    // Only Commit publishes the completed sample; partial phases stay private.
    if (!merging && t == 2)
        sum = VcmSpectrumFrom(float3(0, 0, 0));
    VcmCouplingQuery query = (VcmCouplingQuery)0;
    query.T = t;
    if (!merging && t <= camera.Count)
    {
        query.Merge = 0;
        for (uint s = 0; s + t <= options.MaximumVertices && (s == 0 || s <= light.Count); ++s)
        {
            query.S = s;
            if (!VcmCollect(sum, VcmCouple(light, camera, query, options, statistics)))
                return false;
        }
    }
    if (merging && t <= camera.Count)
    {
        const VcmPathVertex eye = VcmLoad(camera.VertexBase + t - 1);
        if ((eye.Kind != VcmSurfaceVertex && eye.Kind != VcmVolumeVertex) ||
            !(options.MergeRadius > 0))
            return true;
        const VcmIndexEntry cell = VcmCellKey(eye.Position, center, options.MergeRadius);
        if (cell.Identity == InvalidPrimitiveIndex)
            return false;
        query.Merge = 1;
        // A radius-wide grid needs precisely the adjacent 3x3x3 cells. Full
        // positions, geometric normals and sheet tests remain in VcmCouple.
        for (int z = -1; z <= 1; ++z)
            for (int y = -1; y <= 1; ++y)
                for (int x = -1; x <= 1; ++x)
                {
                    VcmIndexEntry key = cell;
                    key.X += x;
                    key.Y += y;
                    key.Z += z;
                    uint found = VcmLowerBound(key, frame.IndexCapacity);
                    while (found < frame.IndexCapacity)
                    {
                        const VcmIndexEntry record = VcmLoadIndex(found++);
                        if (record.Identity == InvalidPrimitiveIndex || !VcmSameCell(key, record))
                            break;
                        const uint surfaces = options.MaximumVertices - 2;
                        if (record.Identity >= frame.LightPathCount * surfaces)
                            return false;
                        query.S = 2 + record.Identity % surfaces;
                        if (query.S + t - 1 <= options.MaximumVertices)
                        {
                            const VcmPath photon =
                                VcmLoadPath(frame.LightPathBase + record.Identity / surfaces);
                            if (!VcmCollect(sum, VcmCouple(photon, camera, query, options, statistics)))
                                return false;
                        }
                    }
                }
    }
    state.PassRed = sum.Red;
    state.PassGreen = sum.Green;
    state.PassBlue = sum.Blue;
    VcmStoreFilm(pixel, state);
    return true;
}
uint VcmSplatPixel(float2 raster, VcmImageDomain image)
{
    if (!(raster.x >= 0 && raster.x < 1 && raster.y >= 0 && raster.y < 1))
        return InvalidPrimitiveIndex;
    const uint x = min(uint(double(raster.x) * image.FullWidth), image.FullWidth - 1);
    const uint y = min(uint(double(raster.y) * image.FullHeight), image.FullHeight - 1);
    return y * image.FullWidth + x;
}
bool VcmSplatEstimate(uint index, VcmFrameParameters frame, VcmCouplingParameters options,
                      VcmImageDomain image, VcmRandomInput random, inout VcmStatistics statistics)
{
    if (index >= frame.IndexCapacity)
        return false;
    VcmIndexEntry entry = VcmEmptyIndex();
    VcmSplat splat = (VcmSplat)0;
    splat.Pixel = InvalidPrimitiveIndex;
    const uint surfaces = options.MaximumVertices - 2;
    if (index < frame.LightPathCount * surfaces)
    {
        const VcmPath light = VcmLoadPath(frame.LightPathBase + index / surfaces);
        const VcmPath camera = VcmLoadPath(frame.LensPathIndex);
        if (light.End == VcmInvalidInput || light.End == VcmNumericalFailure ||
            !VcmUsable(camera, TransportRadiance))
            return false;
        VcmCouplingQuery query = (VcmCouplingQuery)0;
        query.S = 2 + index % surfaces;
        query.T = 1;
        query.Lens = VcmDraw(random, VcmSplatLensDimension ^ (query.S * VcmBounceKey));
        const VcmContribution value = VcmCouple(light, camera, query, options, statistics);
        if ((value.Flags & VcmInvalidContribution) != 0)
            return false;
        if (VcmNonzero(value.Radiance))
        {
            if ((value.Flags & VcmSplatContribution) == 0)
                return false;
            splat.Pixel = VcmSplatPixel(value.Raster, image);
            if (splat.Pixel == InvalidPrimitiveIndex)
                return false;
            splat.Radiance = value.Radiance;
            entry.X = splat.Pixel;
            entry.Identity = index;
        }
    }
    VcmStoreSplat(index, splat);
    VcmStoreIndex(index, entry);
    return true;
}
bool VcmCommitPixel(uint pixel, uint pass, VcmFrameParameters frame, VcmCouplingParameters options,
                    VcmImageDomain image)
{
    if (pixel >= image.Width * image.Height)
        return false;
    VcmFilmPixel state = VcmLoadFilm(pixel);
    if (state.CompletedPasses != pass || !(options.LightTracingSamples > 0))
        return false;
    const uint x = image.TileX + pixel % image.Width, y = image.TileY + pixel / image.Width;
    VcmIndexEntry key = (VcmIndexEntry)0;
    key.X = y * image.FullWidth + x;
    uint found = VcmLowerBound(key, frame.IndexCapacity);
    VcmSpectrum sum;
    sum.Red = state.PassRed;
    sum.Green = state.PassGreen;
    sum.Blue = state.PassBlue;
    while (found < frame.IndexCapacity)
    {
        const VcmIndexEntry entry = VcmLoadIndex(found++);
        if (entry.Identity == InvalidPrimitiveIndex || !VcmSameCell(entry, key))
            break;
        if (entry.Identity >= frame.IndexCapacity)
            return false;
        const VcmSplat splat = VcmLoadSplat(entry.Identity);
        if (splat.Pixel != key.X || !VcmValidPower(splat.Radiance))
            return false;
        // MIS already counted light walks per full-film pixel; normalize the
        // separately sampled light-tracing estimator exactly once at reduction.
        sum.Red += double(splat.Radiance.x) / options.LightTracingSamples;
        sum.Green += double(splat.Radiance.y) / options.LightTracingSamples;
        sum.Blue += double(splat.Radiance.z) / options.LightTracingSamples;
    }
    sum.Red += state.Red;
    sum.Green += state.Green;
    sum.Blue += state.Blue;
    if (!VcmValidSpectrum(sum))
        return false;
    const double completed = double(pass) + 1;
    if (!VcmValidPower(
            float3(float(sum.Red / completed), float(sum.Green / completed), float(sum.Blue / completed))))
        return false;
    state.Red = sum.Red;
    state.Green = sum.Green;
    state.Blue = sum.Blue;
    state.CompletedPasses = pass + 1;
    VcmStoreFilm(pixel, state);
    return true;
}

#include "VertexConnectionMergingRootSignature.hlsl"
void VcmSignal(uint value)
{
    uint ignored;
    VcmError.InterlockedOr(0, value, ignored);
}
VcmImageDomain VcmImage()
{
    VcmImageDomain image;
    image.Width = Width;
    image.Height = Height;
    image.FullWidth = FullWidth;
    image.FullHeight = FullHeight;
    image.TileX = TileX;
    image.TileY = TileY;
    return image;
}
VcmRandomInput VcmPixelRandom(uint pixel)
{
    VcmRandomInput random;
    random.PixelX = TileX + pixel % Width;
    random.PixelY = TileY + pixel / Width;
    random.PixelSeed = (random.PixelY * FullWidth + random.PixelX) * PixelSeedMultiplier;
    if (SamplingSeed != 0)
        random.PixelSeed = Hash(random.PixelSeed ^ Hash(SamplingSeed));
    random.Index = WalkPass;
    random.Count = TotalSamples;
    random.Sequence = SamplingSequence;
    return random;
}
VcmRandomInput VcmLightRandom(uint globalPath)
{
    VcmRandomInput random;
    random.PixelX = globalPath;
    random.PixelY = 0;
    random.PixelSeed = Hash(WalkSeed ^ (globalPath * PixelSeedMultiplier));
    random.Index = WalkPass;
    random.Count = TotalSamples;
    random.Sequence = SamplingSequence;
    return random;
}
bool VcmFrameLane(uint lane, out uint index)
{
    index = 0;
    if (lane >= Frame.WorkCount)
        return false;
    if (Frame.WorkBegin > InvalidPrimitiveIndex - lane || WalkPass >= TotalSamples ||
        SamplingSequence > BlueNoiseSequence || !VcmValidFrame(Frame, Coupling, VcmImage()))
    {
        VcmSignal(VcmInvalidConfiguration);
        return false;
    }
    index = Frame.WorkBegin + lane;
    return true;
}
bool VcmValidDispatch(uint lane)
{
    if (lane >= PathCount)
        return false;
    const uint maximumIndex = InvalidPrimitiveIndex;
    if (WalkLimit == 0 || WalkLimit > VcmMaximumVertices || WalkPass >= TotalSamples ||
        SamplingSequence > BlueNoiseSequence || PathBegin > maximumIndex - lane ||
        OutputOffset > maximumIndex - lane ||
        OutputOffset + lane > (maximumIndex - (VcmMaximumVertices - 1)) / VcmMaximumVertices)
    {
        VcmSignal(VcmInvalidConfiguration);
        return false;
    }
    return true;
}
void VcmPublish(uint output, VcmPath path, VcmStatistics statistics)
{
    VcmPaths[output] = path;
    VcmCounters[output] = statistics;
    if (path.End == VcmNumericalFailure)
        VcmSignal(VcmFailedArithmetic);
}
[RootSignature(ArvoComputeVcmRootSignature)]
[numthreads(VcmWalkThreads, 1, 1)]
void VcmTraceCameraPaths(uint3 id : SV_DispatchThreadID)
{
    if (!VcmValidDispatch(id.x))
        return;
    const uint pixel = PathBegin + id.x;
    if (Width == 0 || Height == 0 || pixel / Width >= Height || TileX >= FullWidth || TileY >= FullHeight ||
        Width > FullWidth - TileX || Height > FullHeight - TileY)
    {
        VcmSignal(VcmInvalidConfiguration);
        return;
    }
    const uint x = TileX + pixel % Width, y = TileY + pixel / Width;
    const VcmRandomInput random = VcmPixelRandom(pixel);
    CameraEndpointFrame frame;
    frame.Position = CameraPosition;
    frame.Forward = CameraForward;
    frame.Right = CameraRight;
    frame.Up = CameraUp;
    frame.HalfWidth = HalfWidth;
    frame.HalfHeight = HalfHeight;
    frame.LensRadius = LensRadius;
    frame.FocusDistance = FocusDistance;
    const float2 raster = CameraPixelRaster(x, y, FullWidth, FullHeight, VcmDraw(random, 0));
    const uint output = OutputOffset + id.x;
    VcmStatistics statistics = (VcmStatistics)0;
    const VcmPath path =
        VcmTraceCamera(frame, raster, random, WalkLimit, output * VcmMaximumVertices, statistics);
    VcmPublish(output, path, statistics);
}
[RootSignature(ArvoComputeVcmRootSignature)]
[numthreads(VcmWalkThreads, 1, 1)]
void VcmTraceLightPaths(uint3 id : SV_DispatchThreadID)
{
    if (!VcmValidDispatch(id.x))
        return;
    const uint globalPath = PathBegin + id.x, output = OutputOffset + id.x;
    const VcmRandomInput random = VcmLightRandom(globalPath);
    EmissionDomain domain;
    domain.Center = EmissionCenter;
    domain.Radius = EmissionRadius;
    const EmitterEndpoint endpoint =
        SampleEmitterEndpoint(Emitters, EmitterCount, domain, VcmDraw(random, VcmSourceDimension).x,
                              VcmDraw(random, VcmPositionDimension), VcmDraw(random, VcmDirectionDimension));
    VcmStatistics statistics = (VcmStatistics)0;
    const VcmPath path = VcmTraceLight(endpoint, random, WalkLimit, output * VcmMaximumVertices, statistics);
    VcmPublish(output, path, statistics);
}
[RootSignature(ArvoComputeVcmRootSignature)]
[numthreads(VcmWalkThreads, 1, 1)]
void VcmEvaluateCouplings(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= Coupling.QueryCount)
        return;
    const uint maximumIndex = InvalidPrimitiveIndex;
    uint metadataCount, vertexCount, outputCount, statisticsCount, stride;
    VcmPaths.GetDimensions(metadataCount, stride);
    VcmVertices.GetDimensions(vertexCount, stride);
    VcmContributions.GetDimensions(outputCount, stride);
    VcmCounters.GetDimensions(statisticsCount, stride);
    if (Coupling.QueryBegin > maximumIndex - id.x || Coupling.OutputOffset > maximumIndex - id.x ||
        Coupling.OutputOffset + id.x >= outputCount || Coupling.OutputOffset + id.x >= statisticsCount)
    {
        VcmSignal(VcmInvalidConfiguration);
        return;
    }
    // t11 is a root SRV. The host must validate its upload byte extent, while
    // descriptor-backed path/vertex/output buffers expose their element counts.
    const VcmCouplingQuery query = VcmCouplingQueries[Coupling.QueryBegin + id.x];
    VcmContribution value = (VcmContribution)0;
    VcmStatistics statistics = (VcmStatistics)0;
    if (query.CameraPath >= metadataCount || (query.S != 0 && query.LightPath >= metadataCount))
        value.Flags = VcmInvalidContribution;
    else
    {
        const VcmPath camera = VcmPaths[query.CameraPath];
        VcmPath light = VcmEmptyPath(TransportImportance, Coupling.MaximumVertices, 0);
        if (query.S != 0)
            light = VcmPaths[query.LightPath];
        if (camera.VertexBase > vertexCount || camera.Count > vertexCount - camera.VertexBase ||
            light.VertexBase > vertexCount || light.Count > vertexCount - light.VertexBase)
            value.Flags = VcmInvalidContribution;
        else
            value = VcmCouple(light, camera, query, Coupling, statistics);
    }
    const uint output = Coupling.OutputOffset + id.x;
    VcmContributions[output] = value;
    VcmCounters[output] = statistics;
    if ((value.Flags & VcmInvalidContribution) != 0)
        VcmSignal(VcmFailedArithmetic);
}
[RootSignature(ArvoComputeVcmRootSignature)]
[numthreads(VcmWalkThreads, 1, 1)]
void VcmInitializeFrame(uint3 id : SV_DispatchThreadID)
{
    uint pixel;
    if (!VcmFrameLane(id.x, pixel))
        return;
    uint capacity, stride;
    VcmFilm.GetDimensions(capacity, stride);
    if (pixel >= Width * Height || pixel >= capacity)
    {
        VcmSignal(VcmInvalidConfiguration);
        return;
    }
    VcmStoreFilm(pixel, (VcmFilmPixel)0);
    VcmResolved[pixel] = float4(0, 0, 0, 0);
    VcmSampleCounts[pixel] = 0;
    VcmSampleTargets[pixel] = TotalSamples;
    if (pixel == 0)
    {
        uint pathCount, vertexCount;
        VcmPaths.GetDimensions(pathCount, stride);
        VcmVertices.GetDimensions(vertexCount, stride);
        if (Frame.LensPathIndex >= pathCount || Frame.LensPathIndex >= vertexCount / VcmMaximumVertices)
        {
            VcmSignal(VcmInvalidConfiguration);
            return;
        }
        VcmStatistics statistics = (VcmStatistics)0;
        VcmPath lens = VcmTraceCamera(VcmSensor(), float2(0.5, 0.5), VcmLightRandom(0), 1,
                                      Frame.LensPathIndex * VcmMaximumVertices, statistics);
        lens.Limit = Coupling.MaximumVertices;
        if (!VcmUsable(lens, TransportRadiance))
            VcmSignal(VcmFailedArithmetic);
        VcmPaths[Frame.LensPathIndex] = lens;
    }
}
[RootSignature(ArvoComputeVcmRootSignature)]
[numthreads(VcmWalkThreads, 1, 1)]
void VcmBuildGrid(uint3 id : SV_DispatchThreadID)
{
    uint index;
    if (!VcmFrameLane(id.x, index))
        return;
    const VcmCouplingParameters options = VcmFrameCoupling(Frame, Coupling, VcmImage(), WalkPass);
    if (!VcmBuildGridEntry(index, Frame, options, EmissionCenter))
        VcmSignal(VcmFailedArithmetic);
}
[RootSignature(ArvoComputeVcmRootSignature)]
[numthreads(VcmWalkThreads, 1, 1)]
void VcmSortIndex(uint3 id : SV_DispatchThreadID)
{
    uint index;
    if (!VcmFrameLane(id.x, index))
        return;
    if (!VcmSortPair(index, Frame))
        VcmSignal(VcmInvalidConfiguration);
}
void VcmEvaluateCameraPhase(uint lane, bool merging)
{
    uint pixel;
    if (!VcmFrameLane(lane, pixel))
        return;
    uint pathCount, stride;
    VcmPaths.GetDimensions(pathCount, stride);
    if (pixel >= Width * Height || pixel < Frame.CameraPixelBegin || Frame.CameraPathBase >= pathCount ||
        pixel - Frame.CameraPixelBegin >= pathCount - Frame.CameraPathBase)
    {
        VcmSignal(VcmInvalidConfiguration);
        return;
    }
    const uint cameraIndex = Frame.CameraPathBase + pixel - Frame.CameraPixelBegin;
    const VcmCouplingParameters options = VcmFrameCoupling(Frame, Coupling, VcmImage(), WalkPass);
    VcmStatistics statistics = (VcmStatistics)0;
    if (!VcmCameraEstimateVertex(pixel, cameraIndex, Frame, options, VcmPixelRandom(pixel), EmissionCenter,
                                 Frame.CameraPrefix, merging, statistics))
        VcmSignal(VcmFailedArithmetic);
    VcmCounters[lane] = statistics;
}
[RootSignature(ArvoComputeVcmRootSignature)]
[numthreads(VcmWalkThreads, 1, 1)]
void VcmEvaluateCameraFrame(uint3 id : SV_DispatchThreadID)
{
    VcmEvaluateCameraPhase(id.x, false);
}
[RootSignature(ArvoComputeVcmRootSignature)]
[numthreads(VcmWalkThreads, 1, 1)]
void VcmEvaluateCameraMerges(uint3 id : SV_DispatchThreadID)
{
    VcmEvaluateCameraPhase(id.x, true);
}
[RootSignature(ArvoComputeVcmRootSignature)]
[numthreads(VcmWalkThreads, 1, 1)]
void VcmEvaluateSplats(uint3 id : SV_DispatchThreadID)
{
    uint index;
    if (!VcmFrameLane(id.x, index))
        return;
    const VcmCouplingParameters options = VcmFrameCoupling(Frame, Coupling, VcmImage(), WalkPass);
    const uint path = index / (Coupling.MaximumVertices - 2);
    VcmStatistics statistics = (VcmStatistics)0;
    if (!VcmSplatEstimate(index, Frame, options, VcmImage(), VcmLightRandom(path), statistics))
        VcmSignal(VcmFailedArithmetic);
    VcmCounters[id.x] = statistics;
}
[RootSignature(ArvoComputeVcmRootSignature)]
[numthreads(VcmWalkThreads, 1, 1)]
void VcmCommitFrame(uint3 id : SV_DispatchThreadID)
{
    uint pixel;
    if (!VcmFrameLane(id.x, pixel))
        return;
    const VcmCouplingParameters options = VcmFrameCoupling(Frame, Coupling, VcmImage(), WalkPass);
    if (!VcmCommitPixel(pixel, WalkPass, Frame, options, VcmImage()))
    {
        VcmSignal(VcmFailedArithmetic);
        return;
    }
    const VcmFilmPixel state = VcmLoadFilm(pixel);
    const double samples = double(state.CompletedPasses);
    // Shared photon populations and changing bandwidths do not supply an IID M2.
    VcmResolved[pixel] =
        float4(float(state.Red / samples), float(state.Green / samples), float(state.Blue / samples), 0);
    VcmSampleCounts[pixel] = state.CompletedPasses;
    VcmSampleTargets[pixel] = TotalSamples;
}
