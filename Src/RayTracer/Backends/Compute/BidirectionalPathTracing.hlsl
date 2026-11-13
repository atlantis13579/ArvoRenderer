// Independent Compute BDPT. Veach (1997), chapter 10; PBRT 3e, section 16.3.
// Radiance and adjoint walks meet on surfaces, emitters or the lens. A complete
// pass includes cross-pixel light splats; only then does it become a film sample.
// https://pbr-book.org/3ed-2018/Light_Transport_III_Bidirectional_Methods/Bidirectional_Path_Tracing
// Interior collision vertices use volume measure and reversible phase PDFs.
static const float Pi = 3.14159265358979323846f, RayEpsilon = 1.0e-4f;
static const float MaximumRayDistance = 1.0e30f, BasisParallelThreshold = 0.999f;
static const uint PixelSeedMultiplier = 9781u, SampleSeedMultiplier = 6271u;
static const uint BounceSeedMultiplier = 0x9e3779b9u, SecondDimensionScramble = 0x68bc21ebu;
static const uint LobeSelectionScramble = 0xa511e9b3u, ImportanceBit = 1u;
static const uint MisShift = 3u, MisMask = 3u;
static const uint CameraWalkDimension = 0x43414d50u, LightWalkDimension = 0x4c495450u;
static const uint SourceDimension = 0x53524345u, PositionDimension = 0x53504f53u;
static const uint DirectionDimension = 0x53444952u, LensDimension = 0x4c454e53u;
static const uint ConnectionLensDimension = 0x434c454eu;
static const uint InteriorFlightDimension = 0x49464c54u, InteriorPhaseDimension = 0x49504853u;
static const uint CameraVertex = 0, LightVertex = 1, SurfaceVertex = 2, InfiniteVertex = 3,
                  VolumeVertex = 4;
static const uint NumericalFailure = 1, PairGroupSize = 64, FilmGroupWidth = 8;
#include "RadianceAccumulator.hlsl"
static const uint ScalarBytes = 4, ColorChannels = 3;
static const uint ChannelStride = RadianceSumWords * ScalarBytes, FilmStride = ColorChannels * ChannelStride;
#include "Types.hlsl"
#include "MaterialTypes.hlsl"
#include "EmitterData.hlsl"
#include "BidirectionalMis.hlsl"

struct TextureRecord
{
    uint OffsetTexels, Width, Height, Reserved;
};
struct BdptVertex
{
    float3 Position;
    uint Kind;
    float3 Previous;
    uint Material;
    float3 Normal;
    uint FrontFace;
    float3 Geometry;
    uint Emitter;
    float3 Color;
    float Pdf;
    float3 Throughput;
    float ReversePdf;
    uint Event, Reserved0, Reserved1, Reserved2;
};
struct PathState
{
    uint Base, Count, Limit, Mode;
    float3 Direction;
    float PositionPdf;
    float3 Weight;
    float DirectionPdf;
    float2 Lens, Raster;
    uint Flags;
    float SelectionProbability;
};
struct ThreadStatistics
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
RWStructuredBuffer<ThreadStatistics> Statistics : register(u3);
RWStructuredBuffer<uint> SampleTargets : register(u4);
RWStructuredBuffer<BdptVertex> Vertices : register(u5);
RWByteAddressBuffer BdptError : register(u7);
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
cbuffer BidirectionalControls : register(b1)
{
    float3 EmissionCenter;
    float EmissionRadius;
    uint EmitterCount, PathBegin, PathCount, ReservedControl;
};
uint Hash(uint value)
{
    static const uint AvalancheA = 0x7feb352du, AvalancheB = 0x846ca68bu;
    static const uint HalfWord = 16, MiddleShift = 15;
    value ^= value >> HalfWord;
    value *= AvalancheA;
    value ^= value >> MiddleShift;
    value *= AvalancheB;
    return value ^ (value >> HalfWord);
}
float Random01(uint value)
{
    static const uint MantissaMask = 0x00ffffffu;
    static const float MantissaRange = 16777216.0f;
    return float(Hash(value) & MantissaMask) / MantissaRange;
}
void IncrementCounter(inout uint2 counter)
{
    uint previous = counter.x;
    ++counter.x;
    if (counter.x < previous)
        ++counter.y;
}
void FailArithmetic()
{
    uint ignored;
    BdptError.InterlockedOr(0, NumericalFailure, ignored);
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
#include "SubsurfaceScattering.hlsl"
#include "LightSelection.hlsl"
#include "LightEmission.hlsl"
#include "SurfaceEmission.hlsl"
#include "EmitterEndpoint.hlsl"
#include "CameraEndpoint.hlsl"
#include "FilmMoments.hlsl"

CameraEndpointFrame CameraFrame()
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
EmissionDomain WorldDomain()
{
    EmissionDomain domain;
    domain.Center = EmissionCenter;
    domain.Radius = EmissionRadius;
    return domain;
}
float3 Offset(BdptVertex vertex, float3 direction)
{
    if (vertex.Kind == VolumeVertex) return vertex.Position;
    return vertex.Position +
           vertex.Geometry * (dot(vertex.Geometry, direction) >= 0 ? RayEpsilon : -RayEpsilon);
}
RayData MakeRay(float3 origin, float3 direction, float maximum = MaximumRayDistance)
{
    RayData ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.MinimumDistance = RayEpsilon;
    ray.MaximumDistance = maximum;
    return ray;
}
float LinearChannel(uint channel)
{
    static const float ByteMaximum = 255, Threshold = 0.04045f, LinearScale = 12.92f;
    static const float Offset = 0.055f, Scale = 1.055f, Exponent = 2.4f;
    float x = channel / ByteMaximum;
    return x <= Threshold ? x / LinearScale : pow((x + Offset) / Scale, Exponent);
}
float3 Texel(TextureRecord texture, int2 coordinate)
{
    int2 size = int2(texture.Width, texture.Height);
    coordinate = (coordinate % size + size) % size;
    uint texel = TextureTexels[texture.OffsetTexels + coordinate.y * texture.Width + coordinate.x];
    static const uint ByteMask = 0xffu, GreenShift = 8, BlueShift = 16;
    return float3(LinearChannel(texel & ByteMask), LinearChannel((texel >> GreenShift) & ByteMask),
                  LinearChannel((texel >> BlueShift) & ByteMask));
}
float3 ReflectanceColor(HitData hit)
{
    uint index = Materials[hit.MaterialIndex].DiffuseTextureIndex;
    if (hit.HasTexCoord == 0 || index >= TextureCount || !all(isfinite(hit.TexCoord)))
        return hit.Color;
    TextureRecord texture = Textures[index];
    if (texture.Width == 0 || texture.Height == 0 || texture.Height > TextureTexelCount / texture.Width ||
        texture.OffsetTexels > TextureTexelCount ||
        texture.Width * texture.Height > TextureTexelCount - texture.OffsetTexels)
        return hit.Color;
    float2 location =
        frac(float2(hit.TexCoord.x, 1 - hit.TexCoord.y)) * float2(texture.Width, texture.Height) - 0.5f;
    int2 corner = int2(floor(location));
    float2 fraction = frac(location);
    return hit.Color * lerp(lerp(Texel(texture, corner), Texel(texture, corner + int2(1, 0)), fraction.x),
                            lerp(Texel(texture, corner + int2(0, 1)), Texel(texture, corner + 1), fraction.x),
                            fraction.y);
}
SurfaceBsdf SurfaceFrame(BdptVertex vertex, float3 view, uint mode)
{
    HitData hit = (HitData)0;
    hit.Normal = dot(vertex.Normal, view) < 0 ? -vertex.Normal : vertex.Normal;
    hit.GeometricNormal = vertex.Geometry;
    hit.FrontFace = dot(vertex.Geometry, view) > 0;
    PackedMaterial material = Materials[vertex.Material];
    if (material.Model == MaterialLegacy)
        material.Model = MaterialDiffuse;
    return MakeSurfaceBsdf(material, hit, view, vertex.Color, mode);
}
SurfaceEvaluation Scattering(BdptVertex vertex, float3 view, float3 direction, uint mode)
{
    if (vertex.Kind == VolumeVertex)
    {
        float phase = SubsurfacePhase(dot(-view, direction), Materials[vertex.Material].ConductorIndex.w);
        float pdf = (TransportFlags & ImportanceBit) != 0 ? phase : 1 / (4 * Pi);
        SurfaceEvaluation result = (SurfaceEvaluation)0;
        result.Value = phase.xxx;
        result.Pdf = result.ReversePdf = pdf;
        return result;
    }
    SurfaceEvaluation evaluation = EvaluateSurface(SurfaceFrame(vertex, view, mode), direction);
    PackedMaterial material = Materials[vertex.Material];
    if (material.Model == MaterialLegacy)
    {
        float diffuse = 1 - saturate(material.AlbedoReflectivity.w);
        evaluation.Value *= diffuse;
        evaluation.Pdf *= diffuse;
        evaluation.ReversePdf *= diffuse;
    }
    return evaluation;
}
SurfaceSample Continuation(BdptVertex vertex, uint mode, float2 variates, float lobe)
{
    if (vertex.Kind == VolumeVertex)
    {
        SubsurfaceDirection phase = SubsurfaceSampleDirection(-vertex.Previous,
            Materials[vertex.Material].ConductorIndex.w, variates, (TransportFlags & ImportanceBit) != 0);
        SurfaceSample result = (SurfaceSample)0;
        result.Direction = phase.Direction;
        result.Weight = phase.Weight.xxx;
        result.Pdf = result.ReversePdf = phase.Pdf;
        result.RelativeIndex = 1;
        result.Event = ScatteringGlossy;
        return result;
    }
    PackedMaterial material = Materials[vertex.Material];
    SurfaceBsdf surface = SurfaceFrame(vertex, vertex.Previous, mode);
    float mirror = material.Model == MaterialLegacy ? saturate(material.AlbedoReflectivity.w) : 0;
    if (lobe < mirror)
    {
        SurfaceSample sample = (SurfaceSample)0;
        float3 direction = reflect(-vertex.Previous, surface.Normal);
        if (!SurfaceHemisphere(surface, direction, false))
            return sample;
        sample.Direction = direction;
        sample.Weight = SurfaceAdjointCorrection(surface, direction).xxx;
        sample.Pdf = sample.ReversePdf = mirror;
        sample.RelativeIndex = 1;
        sample.Event = ScatteringReflection | ScatteringDelta;
        return sample;
    }
    SurfaceSample sampled = SampleSurface(surface, variates, lobe);
    if (material.Model == MaterialLegacy)
    {
        sampled.Pdf *= 1 - mirror;
        sampled.ReversePdf *= 1 - mirror;
    }
    return sampled;
}
uint SourceForHit(uint primitive, bool environment)
{
    for (uint i = 0; i < EmitterCount; ++i)
        if (environment ? Emitters[i].Kind == PackedEmitterEnvironment
                        : (primitive != InvalidPrimitiveIndex && Emitters[i].Primitive == primitive))
            return i;
    return InvalidPrimitiveIndex;
}
bool HitRectangle(PackedEmitter source, RayData ray, inout float distance)
{
    if (source.Kind != PackedEmitterRectangle)
        return false;
    float denominator = dot(source.Direction, ray.Direction);
    if (denominator == 0)
        return false;
    float candidate = dot(source.Direction, source.Origin - ray.Origin) / denominator;
    if (!(candidate > ray.MinimumDistance && candidate < ray.MaximumDistance && candidate <= distance))
        return false;
    float3 helper = abs(source.Direction.y) < BasisParallelThreshold ? float3(0, 1, 0) : float3(1, 0, 0);
    float3 tangent = normalize(cross(helper, source.Direction)), bitangent = cross(source.Direction, tangent);
    float3 offset = ray.Origin + candidate * ray.Direction - source.Origin;
    if (abs(dot(offset, tangent)) > source.Width / 2 || abs(dot(offset, bitangent)) > source.Height / 2)
        return false;
    distance = candidate;
    return true;
}
float3 EmittedValue(PackedEmitter source, float3 outward)
{
    if (source.Kind == PackedEmitterEnvironment)
        return lerp(source.BackgroundBottom, source.BackgroundTop, (1 - outward.y) / 2);
    if (source.Kind == PackedEmitterRectangle)
        return dot(source.Direction, outward) > 0 ? source.Strength / (source.Width * source.Height)
                                                  : float3(0, 0, 0);
    return source.Strength;
}
bool Occluded(RayData ray, inout ThreadStatistics statistics)
{
    IncrementCounter(statistics.Shadow);
    HitData hit = (HitData)0;
    hit.Distance = ray.MaximumDistance;
    if (IntersectScene(ray, true, hit, statistics.Tests))
        return true;
    for (uint i = 0; i < EmitterCount; ++i)
        if (Emitters[i].Kind == PackedEmitterRectangle)
        {
            IncrementCounter(statistics.Tests);
            float distance = ray.MaximumDistance;
            if (HitRectangle(Emitters[i], ray, distance))
                return true;
        }
    return false;
}
bool Visible(float3 start, float3 end, inout ThreadStatistics statistics)
{
    float3 segment = end - start;
    float distance = length(segment);
    return distance > 2 * RayEpsilon && isfinite(distance) &&
           !Occluded(MakeRay(start, segment / distance, distance - RayEpsilon), statistics);
}
void Walk(inout PathState path, RayData ray, float3 throughput, SampleContext samples,
          inout ThreadStatistics statistics)
{
    uint stream = path.Mode == TransportRadiance ? CameraWalkDimension : LightWalkDimension;
    uint interiorMaterial = InvalidPrimitiveIndex;
    [loop] while (path.Count < path.Limit)
    {
        if (!all(isfinite(throughput)))
        {
            FailArithmetic();
            return;
        }
        if (!any(throughput > 0))
            return;
        if (path.Mode == TransportRadiance && path.Count == 1)
            IncrementCounter(statistics.Primary);
        else
            IncrementCounter(statistics.Reflection);
        HitData hit = (HitData)0;
        hit.Distance = ray.MaximumDistance;
        bool found = IntersectScene(ray, false, hit, statistics.Tests);
        if (found && hit.MaterialIndex >= MaterialCount)
        {
            FailArithmetic();
            return;
        }
        uint rectangle = InvalidPrimitiveIndex;
        float closest = found ? hit.Distance + RayEpsilon : ray.MaximumDistance;
        for (uint source = 0; source < EmitterCount; ++source)
            if (Emitters[source].Kind == PackedEmitterRectangle)
            {
                IncrementCounter(statistics.Tests);
                if (HitRectangle(Emitters[source], ray, closest))
                    rectangle = source;
            }
        BdptVertex vertex = (BdptVertex)0;
        vertex.Previous = -ray.Direction;
        vertex.Throughput = throughput;
        vertex.Emitter = InvalidPrimitiveIndex;
        uint slot = path.Base + path.Count++;
        if (path.Count == 2 && found && !hit.FrontFace && Materials[hit.MaterialIndex].Optical.w > 0)
            interiorMaterial = hit.MaterialIndex;
        if (interiorMaterial != InvalidPrimitiveIndex)
        {
            PackedMaterial interior = Materials[interiorMaterial];
            float endpoint = rectangle != InvalidPrimitiveIndex ? closest :
                (found ? hit.Distance : ray.MaximumDistance);
            uint dimension = stream ^ ((path.Count - 1) * BounceSeedMultiplier);
            SubsurfaceFlight flight = SubsurfaceFreeFlight(interior.Optical.w,
                interior.AlbedoReflectivity.xyz * interior.ConductorExtinction.w,
                ray.Direction, endpoint, Sample2D(samples, dimension ^ InteriorFlightDimension).x);
            if (flight.Scattered != 0)
            {
                vertex.Kind = VolumeVertex;
                vertex.Position = ray.Origin + ray.Direction * flight.Distance;
                vertex.Material = interiorMaterial;
                vertex.Throughput *= flight.Weight;
                if (!all(isfinite(vertex.Position)) || !all(isfinite(vertex.Throughput)))
                {
                    FailArithmetic();
                    return;
                }
                if (path.Count == path.Limit)
                {
                    Vertices[slot] = vertex;
                    return;
                }
                SurfaceSample phase = Continuation(vertex, path.Mode,
                    Sample2D(samples, dimension ^ InteriorPhaseDimension), 0);
                vertex.Pdf = phase.Pdf;
                vertex.ReversePdf = phase.ReversePdf;
                vertex.Event = phase.Event;
                Vertices[slot] = vertex;
                if (!(phase.Pdf > 0)) return;
                throughput = vertex.Throughput * phase.Weight;
                ray = MakeRay(vertex.Position, phase.Direction);
                continue;
            }
        }
        if (rectangle != InvalidPrimitiveIndex)
        {
            vertex.Kind = LightVertex;
            vertex.Position = ray.Origin + closest * ray.Direction;
            vertex.Geometry = Emitters[rectangle].Direction;
            vertex.Emitter = rectangle;
            Vertices[slot] = vertex;
            return;
        }
        if (!found)
        {
            vertex.Kind = InfiniteVertex;
            vertex.Emitter = SourceForHit(InvalidPrimitiveIndex, true);
            Vertices[slot] = vertex;
            return;
        }
        vertex.Kind = SurfaceVertex;
        vertex.Position = ray.Origin + hit.Distance * ray.Direction;
        vertex.Normal = hit.Normal;
        vertex.Geometry = hit.GeometricNormal;
        vertex.FrontFace = hit.FrontFace;
        vertex.Material = hit.MaterialIndex;
        vertex.Color = ReflectanceColor(hit);
        vertex.Emitter = SourceForHit(hit.PrimitiveIndex, false);
        if (path.Count == path.Limit)
        {
            Vertices[slot] = vertex;
            return;
        }
        uint dimension = stream ^ (path.Count * BounceSeedMultiplier);
        SurfaceSample sampled = Continuation(vertex, path.Mode, Sample2D(samples, dimension),
                                             Sample2D(samples, dimension ^ LobeSelectionScramble).x);
        vertex.Pdf = sampled.Pdf;
        vertex.ReversePdf = sampled.ReversePdf;
        vertex.Event = sampled.Event;
        Vertices[slot] = vertex;
        if (sampled.Event == 0 || !(sampled.Pdf > 0))
            return;
        throughput *= sampled.Weight;
        ray = MakeRay(Offset(vertex, sampled.Direction), sampled.Direction);
        if (Materials[hit.MaterialIndex].Optical.w > 0 && (sampled.Event & ScatteringTransmission) != 0)
            interiorMaterial = hit.FrontFace != 0 ? hit.MaterialIndex : InvalidPrimitiveIndex;
    }
}
PathState GenerateCamera(uint base, SampleContext samples, bool walkCamera, inout ThreadStatistics statistics)
{
    PathState path = (PathState)0;
    path.Base = base;
    path.Mode = TransportRadiance;
    path.Limit = MaxReflectionBounces + 1 + 2; // First scattering vertex and both endpoints.
    float2 raster = CameraPixelRaster(samples.PixelX, samples.PixelY, FullWidth, FullHeight,
                                      Sample2D(samples, 0));
    CameraEndpointSample camera =
        SampleCameraEndpoint(CameraFrame(), raster, Sample2D(samples, LensDimension));
    if ((camera.Flags & CameraEndpointValid) == 0)
        return path;
    path.Flags = camera.Flags;
    path.PositionPdf = camera.PositionPdf;
    path.DirectionPdf = camera.DirectionPdf;
    path.Direction = camera.Direction;
    path.Lens = camera.LensPoint;
    path.Raster = raster;
    path.Count = 1;
    BdptVertex root = (BdptVertex)0;
    root.Kind = CameraVertex;
    root.Position = camera.Origin;
    root.Geometry = CameraForward;
    root.Throughput = 1;
    root.Emitter = InvalidPrimitiveIndex;
    Vertices[base] = root;
    if (walkCamera)
        Walk(path, MakeRay(camera.Origin, camera.Direction), float3(1, 1, 1), samples, statistics);
    return path;
}
PathState GenerateLight(uint base, SampleContext samples, inout ThreadStatistics statistics)
{
    PathState path = (PathState)0;
    path.Base = base;
    path.Mode = TransportImportance;
    path.Limit = MaxReflectionBounces + 1 + 2;
    EmitterEndpoint endpoint =
        SampleEmitterEndpoint(Emitters, EmitterCount, WorldDomain(), Sample2D(samples, SourceDimension).x,
                              Sample2D(samples, PositionDimension), Sample2D(samples, DirectionDimension));
    EmissionSample emitted = endpoint.Emission;
    if ((emitted.Flags & EmissionValid) == 0)
        return path;
    path.Flags = emitted.Flags;
    path.PositionPdf = emitted.PositionPdf;
    path.DirectionPdf = emitted.DirectionPdf;
    path.Direction = emitted.Direction;
    path.Weight = emitted.Weight;
    path.SelectionProbability = endpoint.SelectionProbability;
    path.Count = 1;
    BdptVertex root = (BdptVertex)0;
    root.Kind = LightVertex;
    root.Position = emitted.Origin;
    root.Geometry = emitted.Normal;
    root.Emitter = endpoint.EmitterIndex;
    float density = (emitted.Flags & EmissionInfinite) != 0 ? emitted.DirectionPdf : emitted.PositionPdf;
    root.Throughput = emitted.Value / (path.SelectionProbability * density);
    Vertices[base] = root;
    float3 origin =
        (emitted.Flags & EmissionDeltaPosition) != 0 ? root.Position : Offset(root, emitted.Direction);
    Walk(path, MakeRay(origin, emitted.Direction), emitted.Weight, samples, statistics);
    return path;
}

BdptVertex CompletedVertex(PathState light, PathState camera, uint s, uint t, uint index,
                           BdptVertex cameraRoot)
{
    if (index + 1 == s + t)
        return cameraRoot;
    return Vertices[index < s ? light.Base + index : camera.Base + s + t - 1 - index];
}
CameraEndpointEvaluation CameraDensity(PathState camera)
{
    CameraEndpointEvaluation evaluation = EmptyCameraEvaluation();
    evaluation.Raster = camera.Raster;
    evaluation.Flags = camera.Flags;
    evaluation.PositionPdf = camera.PositionPdf;
    evaluation.DirectionPdf = camera.DirectionPdf;
    return evaluation;
}
uint SegmentInterior(BdptVertex from, BdptVertex to)
{
    // Validated SSS solids have separated, outward-facing closed boundaries.
    if (from.Kind == VolumeVertex) return from.Material;
    if (to.Kind == VolumeVertex) return to.Material;
    float3 segment = to.Position - from.Position;
    if (from.Kind == SurfaceVertex && Materials[from.Material].Optical.w > 0 &&
        dot(from.Geometry, segment) < 0)
        return from.Material;
    if (to.Kind == SurfaceVertex && Materials[to.Material].Optical.w > 0 &&
        dot(to.Geometry, -segment) < 0)
        return to.Material;
    return InvalidPrimitiveIndex;
}
float SegmentTransmittance(BdptVertex from, BdptVertex to)
{
    uint interior = SegmentInterior(from, to);
    return interior == InvalidPrimitiveIndex ? 1 :
        exp(-length(to.Position - from.Position) / Materials[interior].Optical.w);
}
PathDensity DensityAtVertex(float pdf, uint event, BdptVertex from, BdptVertex to, bool infinite)
{
    PathDensity angular = MakePathDensity(pdf, (event & ScatteringDelta) != 0 ? PathDirac : PathSolidAngle);
    uint destination = infinite ? PathInfiniteDestination :
        (to.Kind == VolumeVertex ? PathVolumeDestination : PathSurfaceDestination);
    PathDensity density = ConvertPathDensity(angular, to.Position - from.Position, to.Geometry,
                                            destination);
    if (!ValidPathDensity(density) || infinite) return density;
    uint interior = SegmentInterior(from, to);
    if (interior == InvalidPrimitiveIndex)
    {
        if (destination == PathVolumeDestination) return ZeroPathDensity();
        return density;
    }
    float meanFreePath = Materials[interior].Optical.w;
    density.Log2Value -= length(to.Position - from.Position) / meanFreePath * log2(exp(1.0f));
    if (destination == PathVolumeDestination) density.Log2Value -= log2(meanFreePath);
    if (!isfinite(density.Log2Value)) return ZeroPathDensity();
    return density;
}
// Reconstruct a canonical light-to-camera path. Interior cached densities are
// exchanged for a reversed camera prefix; changed connection directions must
// instead evaluate a new BSDF pair and discard the old sampled-event flag.
float StrategyWeight(PathState light, PathState camera, uint s, uint t, BdptVertex cameraRoot,
                     CameraEndpointEvaluation cameraPdf, uint emitter, uint flags, float3 launchDirection)
{
    uint count = s + t;
    if (count == 2)
        return s == 0 ? 1.0f : 0.0f;
    bool infinite = (flags & EmissionInfinite) != 0;
    BidirectionalVertexPdf densities[BdptMaximumVertices];
    BidirectionalStrategy strategies[BdptMaximumStrategies];
    for (uint clear = 0; clear < BdptMaximumVertices; ++clear)
        densities[clear] = (BidirectionalVertexPdf)0;
    for (uint clearStrategy = 0; clearStrategy < BdptMaximumStrategies; ++clearStrategy)
        strategies[clearStrategy] = (BidirectionalStrategy)0;
    for (uint i = 1; i + 1 < count; ++i)
    {
        BdptVertex previous = CompletedVertex(light, camera, s, t, i - 1, cameraRoot);
        BdptVertex current = CompletedVertex(light, camera, s, t, i, cameraRoot);
        BdptVertex next = CompletedVertex(light, camera, s, t, i + 1, cameraRoot);
        float forward, reverse;
        uint event = 0;
        if ((s > 0 && i == s - 1) || i == s)
        {
            float3 toLight =
                i == 1 && infinite ? -launchDirection : normalize(previous.Position - current.Position);
            SurfaceEvaluation evaluated = Scattering(
                current, toLight, normalize(next.Position - current.Position), TransportImportance);
            forward = evaluated.Pdf;
            reverse = evaluated.ReversePdf;
        }
        else
        {
            forward = i < s ? current.Pdf : current.ReversePdf;
            reverse = i < s ? current.ReversePdf : current.Pdf;
            event = current.Event;
        }
        densities[i].Flags = (event & ScatteringDelta) != 0 ? PathDeltaScattering : 0;
        densities[i + 1].Forward = DensityAtVertex(forward, event, current, next, false);
        densities[i - 1].Reverse = DensityAtVertex(reverse, event, current, previous, i == 1 && infinite);
    }
    BdptVertex root = CompletedVertex(light, camera, s, t, 0, cameraRoot);
    BdptVertex first = CompletedVertex(light, camera, s, t, 1, cameraRoot);
    float3 emissionDirection = infinite ? launchDirection : normalize(first.Position - root.Position);
    EmitterEndpointDensity endpoint =
        EmitterEndpointPdf(Emitters, EmitterCount, WorldDomain(), emitter, root.Geometry, emissionDirection);
    bool deltaPosition = (endpoint.Flags & EmissionDeltaPosition) != 0;
    bool distant = (endpoint.Flags & EmissionDeltaDirection) != 0;
    PathDensity position =
        MakePathDensity(deltaPosition ? 1 : endpoint.Position, deltaPosition ? PathDirac : PathArea);
    PathDensity direction =
        MakePathDensity(distant ? 1 : endpoint.Direction, distant ? PathDirac : PathSolidAngle);
    if (infinite)
        densities[1].Forward = ProjectedDiskDensity(position, first.Geometry, emissionDirection);
    else
        densities[1].Forward = DensityAtVertex(distant ? 1 : endpoint.Direction,
                                               distant ? ScatteringDelta : 0,
                                               root, first, false);
    densities[0].Flags = deltaPosition || distant ? PathEndpointNotIntersectable : 0;
    BdptVertex last = CompletedVertex(light, camera, s, t, count - 2, cameraRoot);
    densities[count - 2].Reverse = DensityAtVertex(cameraPdf.DirectionPdf, 0, cameraRoot, last, false);
    bool pinhole = (cameraPdf.Flags & CameraDeltaPosition) != 0;
    densities[count - 1].Flags = pinhole ? PathEndpointNotIntersectable : 0;
    for (uint cut = 0; cut < count; ++cut)
    {
        strategies[cut].Enabled = cut <= light.Limit && count - cut <= camera.Limit;
        if (cut == 0)
            strategies[cut].LightEndpoint = MakePathDensity(1, PathDirac);
        else if (infinite)
            strategies[cut].LightEndpoint = direction;
        else
            strategies[cut].LightEndpoint = position;
        strategies[cut].LightSelection = cut == 0 ? 1 : endpoint.SelectionProbability;
        strategies[cut].CameraEndpoint =
            MakePathDensity(pinhole ? 1 : cameraPdf.PositionPdf, pinhole ? PathDirac : PathArea);
        strategies[cut].SampleCount = 1;
    }
    BidirectionalWeights weights =
        BidirectionalMisWeightsLocal(densities, count, strategies, (TransportFlags >> MisShift) & MisMask);
    return weights.Values[s];
}
float3 HitEmission(PathState light, PathState camera, uint t)
{
    BdptVertex vertex = Vertices[camera.Base + t - 1];
    if (vertex.Emitter >= EmitterCount)
        return 0;
    float3 value = vertex.Throughput * EmittedValue(Emitters[vertex.Emitter], vertex.Previous);
    if (!all(isfinite(value)))
    {
        FailArithmetic();
        return 0;
    }
    if (!any(value > 0))
        return 0;
    EmitterEndpointDensity endpoint = EmitterEndpointPdf(Emitters, EmitterCount, WorldDomain(),
                                                         vertex.Emitter, vertex.Geometry, vertex.Previous);
    float weight = StrategyWeight(light, camera, 0, t, Vertices[camera.Base], CameraDensity(camera),
                                  vertex.Emitter, endpoint.Flags, vertex.Previous);
    return value * weight;
}
float3 ConnectEmitter(PathState light, PathState camera, uint t, inout ThreadStatistics statistics)
{
    BdptVertex vertex = Vertices[camera.Base + t - 1], root = Vertices[light.Base];
    if (vertex.Kind != SurfaceVertex && vertex.Kind != VolumeVertex)
        return 0;
    bool infinite = (light.Flags & EmissionInfinite) != 0,
         deltaPosition = (light.Flags & EmissionDeltaPosition) != 0;
    float3 direction;
    float geometry = 1;
    if (infinite)
        direction = -light.Direction;
    else
    {
        float3 segment = root.Position - vertex.Position;
        float squared = dot(segment, segment);
        if (!(squared > 0) || !isfinite(squared))
            return 0;
        direction = segment * rsqrt(squared);
        geometry = (deltaPosition ? 1 : abs(dot(root.Geometry, direction))) / squared;
    }
    SurfaceEvaluation bsdf = Scattering(vertex, vertex.Previous, direction, TransportRadiance);
    float density = light.SelectionProbability * (infinite ? light.DirectionPdf : light.PositionPdf);
    float cosine = vertex.Kind == VolumeVertex ? 1 : abs(dot(vertex.Normal, direction));
    float3 value = vertex.Throughput * bsdf.Value * EmittedValue(Emitters[root.Emitter], -direction) *
                   (cosine * geometry / density *
                    (infinite ? 1 : SegmentTransmittance(vertex, root)));
    if (!all(isfinite(value)))
    {
        FailArithmetic();
        return 0;
    }
    if (!any(value > 0))
        return 0;
    if (infinite)
    {
        if (Occluded(MakeRay(Offset(vertex, direction), direction), statistics))
            return 0;
    }
    else if (!Visible(Offset(vertex, direction), deltaPosition ? root.Position : Offset(root, -direction),
                      statistics))
        return 0;
    return value * StrategyWeight(light, camera, 1, t, Vertices[camera.Base], CameraDensity(camera),
                                  root.Emitter, light.Flags, light.Direction);
}
struct ConnectionContribution
{
    float3 Value;
    float2 Raster;
};
ConnectionContribution ConnectSurfaces(PathState light, PathState camera, uint s, uint t, float2 lens,
                                       inout ThreadStatistics statistics)
{
    ConnectionContribution contribution = (ConnectionContribution)0;
    BdptVertex a = Vertices[light.Base + s - 1], b = Vertices[camera.Base + t - 1];
    if ((a.Kind != SurfaceVertex && a.Kind != VolumeVertex) ||
        (t > 1 && b.Kind != SurfaceVertex && b.Kind != VolumeVertex))
        return contribution;
    BdptVertex cameraRoot = Vertices[camera.Base];
    CameraEndpointEvaluation cameraPdf = CameraDensity(camera);
    float3 towardCamera, target, value;
    contribution.Raster = camera.Raster;
    if (t == 1)
    {
        CameraConnectionSample connection = SampleCameraConnection(CameraFrame(), a.Position, lens);
        if ((connection.Flags & CameraEndpointValid) == 0)
            return contribution;
        towardCamera = connection.Direction;
        target = connection.Position;
        SurfaceEvaluation bsdf = Scattering(a, a.Previous, towardCamera, TransportImportance);
        cameraRoot.Position = target;
        float cosine = a.Kind == VolumeVertex ? 1 : abs(dot(a.Normal, towardCamera));
        value = a.Throughput * bsdf.Value *
                (cosine * connection.Weight * SegmentTransmittance(a, cameraRoot));
        cameraPdf = EvaluateCameraEndpoint(CameraFrame(), connection.LensPoint, -towardCamera);
        contribution.Raster = connection.Raster;
    }
    else
    {
        float3 segment = b.Position - a.Position;
        float squared = dot(segment, segment);
        if (!(squared > 0) || !isfinite(squared))
            return contribution;
        towardCamera = segment * rsqrt(squared);
        SurfaceEvaluation lightBsdf = Scattering(a, a.Previous, towardCamera, TransportImportance);
        SurfaceEvaluation cameraBsdf = Scattering(b, b.Previous, -towardCamera, TransportRadiance);
        float cosineA = a.Kind == VolumeVertex ? 1 : abs(dot(a.Normal, towardCamera));
        float cosineB = b.Kind == VolumeVertex ? 1 : abs(dot(b.Normal, towardCamera));
        value = a.Throughput * lightBsdf.Value * b.Throughput * cameraBsdf.Value *
                (cosineA * cosineB / squared * SegmentTransmittance(a, b));
        target = Offset(b, -towardCamera);
    }
    if (!all(isfinite(value)))
    {
        FailArithmetic();
        return contribution;
    }
    if (!any(value > 0) || !Visible(Offset(a, towardCamera), target, statistics))
        return contribution;
    BdptVertex root = Vertices[light.Base];
    contribution.Value = value * StrategyWeight(light, camera, s, t, cameraRoot, cameraPdf, root.Emitter,
                                                light.Flags, light.Direction);
    return contribution;
}
bool LocalPixel(uint2 coordinate, out uint pixel)
{
    pixel = 0;
    if (any(coordinate < uint2(TileX, TileY)) ||
        any(coordinate - uint2(TileX, TileY) >= uint2(Width, Height)))
        return false;
    pixel = (coordinate.y - TileY) * Width + coordinate.x - TileX;
    return true;
}
void Splat(uint pixel, float3 value)
{
    for (uint channel = 0; channel < ColorChannels; ++channel)
        if (!AccumulateRadiance(Pass, pixel * FilmStride + channel * ChannelStride, value[channel]))
            FailArithmetic();
}
[numthreads(PairGroupSize, 1, 1)]
void BidirectionalPairsKernel(uint3 dispatch : SV_DispatchThreadID)
{
    uint local = dispatch.x;
    if (local >= PathCount)
        return;
    uint identity = PathBegin + local;
    uint2 coordinate = uint2(identity % FullWidth, identity / FullWidth);
    uint pixel;
    bool owned = LocalPixel(coordinate, pixel);
    SampleContext samples =
        MakeSampleContext(coordinate.x, coordinate.y, FullWidth, SampleBegin, TotalSamples, SamplingSequence, SamplingSeed);
    ThreadStatistics statistics = (ThreadStatistics)0;
    uint base = local * 2 * BdptMaximumVertices;
    PathState camera = GenerateCamera(base + BdptMaximumVertices, samples, owned, statistics);
    PathState light = GenerateLight(base, samples, statistics);
    float3 result = 0;
    for (uint t = 2; t <= camera.Count; ++t)
        result += HitEmission(light, camera, t);
    if (camera.Count > 0 && light.Count > 0)
    {
        for (uint direct = 2; direct <= camera.Count && direct < camera.Limit; ++direct)
            result += ConnectEmitter(light, camera, direct, statistics);
        [loop]
        for (uint s = 2; s <= light.Count; ++s)
        {
            [loop]
            for (uint t = 1; t <= camera.Count && s + t <= camera.Limit; ++t)
            {
                float2 lens = Sample2D(samples, ConnectionLensDimension ^ (s * BounceSeedMultiplier));
                ConnectionContribution connection = ConnectSurfaces(light, camera, s, t, lens, statistics);
                if (t > 1)
                    result += connection.Value;
                else if (any(connection.Value > 0) && CameraUnitSquare(connection.Raster))
                {
                    uint target;
                    if (LocalPixel(uint2(connection.Raster * float2(FullWidth, FullHeight)), target))
                        Splat(target, connection.Value);
                }
            }
        }
    }
    if (owned)
        Splat(pixel, result);
    Statistics[local] = statistics;
}

[numthreads(FilmGroupWidth, FilmGroupWidth, 1)]
void BidirectionalClearKernel(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= Width || id.y >= Height)
        return;
    uint pixel = id.y * Width + id.x;
    for (uint word = 0; word < ColorChannels * RadianceSumWords; ++word)
        Pass.Store(pixel * FilmStride + word * ScalarBytes, 0);
    if (pixel == 0)
    {
        BdptError.Store(0, 0);
        if (InstanceNodeCount > 0)
            InstanceError.Store(0, 0);
    }
}
[numthreads(FilmGroupWidth, FilmGroupWidth, 1)]
void BidirectionalCommitKernel(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= Width || id.y >= Height)
        return;
    uint pixel = id.y * Width + id.x;
    float3 value;
    for (uint channel = 0; channel < ColorChannels; ++channel)
        value[channel] = ResolveRadiance(Pass, pixel * FilmStride + channel * ChannelStride);
    if (!all(isfinite(value)))
    {
        FailArithmetic();
        return;
    }
    float4 moments = Accumulation[pixel];
    uint count = SampleCounts[pixel];
    AccumulateFilmSample(moments, count, value);
    if (!all(isfinite(moments)))
    {
        FailArithmetic();
        return;
    }
    Accumulation[pixel] = moments;
    SampleCounts[pixel] = count;
    SampleTargets[pixel] = TotalSamples;
}
