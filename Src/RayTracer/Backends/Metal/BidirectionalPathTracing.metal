// Independent Metal BDPT. Veach (1997), chapter 10; PBRT 3e, section 16.3.
// Radiance/adjoint subpaths connect on surfaces, source endpoints and the lens.
// Only a complete pass, including incoming light splats, is a film observation.
// https://pbr-book.org/3ed-2018/Light_Transport_III_Bidirectional_Methods/Bidirectional_Path_Tracing
// Interior collisions extend complete paths to volume measure.
#include <metal_stdlib>
using namespace metal;
constant float Pi = 3.14159265358979323846f;
constant float RayEpsilon = 1.0e-4f, MaximumRayDistance = 1.0e30f, BasisParallelThreshold = 0.999f;
constant uint PixelSeedMultiplier = 9781u, SampleSeedMultiplier = 6271u;
constant uint BounceSeedMultiplier = 0x9e3779b9u, SecondDimensionScramble = 0x68bc21ebu;
constant uint LobeSelectionScramble = 0xa511e9b3u, ImportanceBit = 1u, MisShift = 3u, MisMask = 3u;
constant uint AvalancheMultiplierA = 0x7feb352du, AvalancheMultiplierB = 0x846ca68bu;
constant uint RandomMantissaMask = 0x00ffffffu;
constant float RandomMantissaRange = 16777216.0f;
constant uint CameraDomain = 0x43414d50u, LightDomain = 0x4c495450u;
constant uint EmitterDimension = 0x53524345u, OriginDimension = 0x53504f53u, RayDimension = 0x53444952u;
constant uint LensDimension = 0x4c454e53u, ConnectionLensDimension = 0x434c454eu;
constant uint InteriorFlightDimension = 0x49464c54u, InteriorPhaseDimension = 0x49504853u;
constant uint CameraKind = 0, EmitterKind = 1, SurfaceKind = 2, InfiniteKind = 3,
              VolumeKind = 4;
#include "RadianceAccumulator.metal"
constant uint NumericalFailure = 1, ColorChannels = 3, PassComponents = ColorChannels * RadianceWords;
#include "SurfaceTypes.metal"
#include "ShadingTypes.metal"
#include "Hash.metal"
#include "Sequence.metal"
#include "FilmMoments.metal"

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
struct BidirectionalControls
{
    packed_float3 Center;
    float Radius;
    uint Sources, FirstPath, PathCount, Reserved;
};
struct PathVertex
{
    packed_float3 Position;
    uint Kind;
    packed_float3 View;
    uint Material;
    packed_float3 Normal;
    uint Source;
    packed_float3 Geometry;
    uint Event;
    packed_float3 Color;
    float Forward;
    packed_float3 Beta;
    float Reverse;
};
struct Path
{
    uint Begin, Count, Limit, Mode;
    float3 LaunchDirection;
    float SourcePmf, PositionPdf, DirectionPdf;
    float2 Raster;
    uint EndpointFlags;
};
struct ThreadStatistics
{
    uint2 Primary, Shadow, Reflection, Tests;
};
static_assert(sizeof(PathVertex) == 96 && sizeof(BidirectionalControls) == 32 &&
                  sizeof(FrameConstants) == 240,
              "Metal BDPT upload layout mismatch");
void IncrementCounter(thread uint2& count)
{
    ++count.x;
    if (count.x == 0)
        ++count.y;
}
#include "ExtendedFloat.metal"
#include "TraversalTypes.metal"
#include "Triangle.metal"
#include "Sphere.metal"
#include "Bvh.metal"
#include "KdTree.metal"
#include "Bih.metal"
#include "Dispatch.metal"
#include "BsdfTypes.metal"
#include "Fresnel.metal"
#include "Ggx.metal"
#include "Diffuse.metal"
#include "Conductor.metal"
#include "Dielectric.metal"
#include "SurfaceScattering.metal"
#include "SubsurfaceScattering.metal"
#include "LightSelection.metal"
#include "LightEmission.metal"
#include "SurfaceEmission.metal"
#include "EmitterEndpoint.metal"
#include "CameraEndpoint.metal"
#include "BidirectionalMis.metal"

struct TransportContext
{
    SceneGeometryBuffers Geometry;
    device const PackedMaterial* Materials;
    device const PackedTextureDescriptor* Textures;
    device const uint* Texels;
    device const PackedEmitter* Emitters;
    constant FrameConstants* Frame;
    constant BidirectionalControls* Batch;
    device PathVertex* Vertices;
    device atomic_uint* Pass;
    device atomic_uint* Error;
};
bool Positive(float3 value)
{
    return value.x > 0 || value.y > 0 || value.z > 0;
}
void ReportFailure(TransportContext context)
{
    atomic_store_explicit(context.Error, NumericalFailure, memory_order_relaxed);
}
EmissionDomain EmissionBounds(TransportContext context)
{
    return {float3(context.Batch->Center), context.Batch->Radius};
}
CameraEndpointFrame CameraModel(constant FrameConstants& frame)
{
    return {float3(frame.CameraPosition),
            float3(frame.CameraForward),
            float3(frame.CameraRight),
            float3(frame.CameraUp),
            frame.HalfWidth,
            frame.HalfHeight,
            frame.LensRadius,
            frame.FocusDistance};
}
RayData Ray(float3 origin, float3 direction, float maximum = MaximumRayDistance)
{
    return {origin, direction, RayEpsilon, maximum};
}
float3 Offset(PathVertex vertex, float3 direction)
{
    if (vertex.Kind == VolumeKind) return float3(vertex.Position);
    float3 geometry = float3(vertex.Geometry);
    return float3(vertex.Position) + geometry * (dot(geometry, direction) >= 0 ? RayEpsilon : -RayEpsilon);
}
float LinearChannel(uint channel)
{
    const float ByteRange = 255, Threshold = 0.04045f, LinearSlope = 12.92f;
    const float Offset = 0.055f, Scale = 1.055f, Exponent = 2.4f;
    float value = float(channel) / ByteRange;
    return value <= Threshold ? value / LinearSlope : pow((value + Offset) / Scale, Exponent);
}
float3 LoadTexel(TransportContext context, PackedTextureDescriptor texture, int x, int y)
{
    x = (x % int(texture.Width) + int(texture.Width)) % int(texture.Width);
    y = (y % int(texture.Height) + int(texture.Height)) % int(texture.Height);
    uint bits = context.Texels[texture.OffsetTexels + uint(y) * texture.Width + uint(x)];
    const uint ByteMask = 0xffu, GreenShift = 8, BlueShift = 16;
    return float3(LinearChannel(bits & ByteMask), LinearChannel((bits >> GreenShift) & ByteMask),
                  LinearChannel((bits >> BlueShift) & ByteMask));
}
float3 HitColor(TransportContext context, HitData hit)
{
    uint index = context.Materials[hit.MaterialIndex].DiffuseTextureIndex;
    if (!hit.HasTexCoord || index >= context.Frame->TextureCount || !isfinite(hit.TexCoord.x) ||
        !isfinite(hit.TexCoord.y))
        return hit.Color;
    PackedTextureDescriptor texture = context.Textures[index];
    uint available = context.Frame->TextureTexelCount;
    if (!texture.Width || !texture.Height || texture.OffsetTexels > available ||
        texture.Height > (available - texture.OffsetTexels) / texture.Width)
        return hit.Color;
    float x = fract(hit.TexCoord.x) * float(texture.Width) - 0.5f;
    float y = fract(1 - hit.TexCoord.y) * float(texture.Height) - 0.5f;
    int column = int(floor(x)), row = int(floor(y));
    return hit.Color * mix(mix(LoadTexel(context, texture, column, row),
                               LoadTexel(context, texture, column + 1, row), fract(x)),
                           mix(LoadTexel(context, texture, column, row + 1),
                               LoadTexel(context, texture, column + 1, row + 1), fract(x)),
                           fract(y));
}
SurfaceBsdf ScatteringFrame(TransportContext context, PathVertex vertex, float3 view, uint mode)
{
    HitData hit{};
    float3 normal = float3(vertex.Normal), geometry = float3(vertex.Geometry);
    hit.Normal = dot(normal, view) < 0 ? -normal : normal;
    hit.GeometricNormal = geometry;
    hit.FrontFace = dot(geometry, view) > 0;
    PackedMaterial material = context.Materials[vertex.Material];
    if (material.Model == MaterialLegacy)
        material.Model = MaterialDiffuse;
    return MakeSurfaceBsdf(material, hit, view, float3(vertex.Color), context.Frame->TransportFlags, mode);
}
SurfaceEvaluation Evaluate(TransportContext context, PathVertex vertex, float3 view, float3 direction,
                           uint mode)
{
    if (vertex.Kind == VolumeKind)
    {
        float phase = SubsurfacePhase(dot(-view, direction),
                                      context.Materials[vertex.Material].ConductorIndex.w);
        float pdf = (context.Frame->TransportFlags & ImportanceBit) ? phase : 1 / (4 * Pi);
        return {float3(phase), pdf, pdf};
    }
    SurfaceEvaluation value = EvaluateSurface(ScatteringFrame(context, vertex, view, mode), direction);
    PackedMaterial material = context.Materials[vertex.Material];
    if (material.Model == MaterialLegacy)
    {
        float probability = 1 - clamp(material.AlbedoReflectivity.w, 0.0f, 1.0f);
        value.Value *= probability;
        value.Pdf *= probability;
        value.ReversePdf *= probability;
    }
    return value;
}
SurfaceSample Continue(TransportContext context, PathVertex vertex, uint mode, float2 directionU, float lobeU)
{
    if (vertex.Kind == VolumeKind)
    {
        auto phase = SubsurfaceSampleDirection(-float3(vertex.View),
            context.Materials[vertex.Material].ConductorIndex.w, directionU,
            (context.Frame->TransportFlags & ImportanceBit) != 0);
        return {phase.Direction, float3(phase.Weight), phase.Pdf, phase.Pdf, 1, ScatteringGlossy};
    }
    SurfaceBsdf frame = ScatteringFrame(context, vertex, float3(vertex.View), mode);
    PackedMaterial material = context.Materials[vertex.Material];
    float mirror = material.Model == MaterialLegacy ? clamp(material.AlbedoReflectivity.w, 0.0f, 1.0f) : 0;
    if (lobeU < mirror)
    {
        float3 direction = reflect(-float3(vertex.View), frame.Normal);
        if (!SurfaceDirectionValid(frame, direction, dot(direction, frame.Normal)))
            return {};
        return {direction, float3(SurfaceAdjointCorrection(frame, direction)),
                mirror,    mirror,
                1,         ScatteringReflection | ScatteringDelta};
    }
    SurfaceSample sample = SampleSurface(frame, directionU, lobeU);
    if (material.Model == MaterialLegacy)
    {
        sample.Pdf *= 1 - mirror;
        sample.ReversePdf *= 1 - mirror;
    }
    return sample;
}
uint SourceIndex(TransportContext context, uint primitive, bool sky)
{
    for (uint source = 0; source < context.Batch->Sources; ++source)
        if (sky ? context.Emitters[source].Kind == PackedEmitterEnvironment
                : (primitive != InvalidPrimitiveIndex && context.Emitters[source].Primitive == primitive))
            return source;
    return InvalidPrimitiveIndex;
}
bool IntersectEmitter(PackedEmitter emitter, RayData ray, thread float& distance)
{
    if (emitter.Kind != PackedEmitterRectangle)
        return false;
    float3 normal = float3(emitter.Direction);
    float denominator = dot(normal, ray.Direction);
    if (denominator == 0)
        return false;
    float candidate = dot(normal, float3(emitter.Origin) - ray.Origin) / denominator;
    if (!(candidate > ray.MinimumDistance && candidate < ray.MaximumDistance && candidate <= distance))
        return false;
    float3 helper = fabs(normal.y) < BasisParallelThreshold ? float3(0, 1, 0) : float3(1, 0, 0);
    float3 tangent = normalize(cross(helper, normal)), bitangent = cross(normal, tangent);
    float3 relative = ray.Origin + candidate * ray.Direction - float3(emitter.Origin);
    if (fabs(dot(relative, tangent)) > emitter.Width / 2 ||
        fabs(dot(relative, bitangent)) > emitter.Height / 2)
        return false;
    distance = candidate;
    return true;
}
float3 Emission(PackedEmitter source, float3 direction)
{
    if (source.Kind == PackedEmitterEnvironment)
        return mix(float3(source.BackgroundBottom), float3(source.BackgroundTop), (1 - direction.y) / 2);
    if (source.Kind == PackedEmitterRectangle)
        return dot(float3(source.Direction), direction) > 0
                   ? float3(source.Strength) / (source.Width * source.Height)
                   : float3(0);
    return float3(source.Strength);
}
bool Occluded(TransportContext context, RayData ray, thread ThreadStatistics& statistics)
{
    IncrementCounter(statistics.Shadow);
    HitData hit{};
    hit.Distance = ray.MaximumDistance;
    if (IntersectScene(context.Geometry, ray, true, hit, statistics.Tests))
        return true;
    float nearest = ray.MaximumDistance;
    for (uint source = 0; source < context.Batch->Sources; ++source)
        if (context.Emitters[source].Kind == PackedEmitterRectangle)
        {
            IncrementCounter(statistics.Tests);
            if (IntersectEmitter(context.Emitters[source], ray, nearest))
                return true;
        }
    return false;
}
bool ConnectionVisible(TransportContext context, float3 origin, float3 target,
                       thread ThreadStatistics& statistics)
{
    float3 segment = target - origin;
    float distance = length(segment);
    return distance > 2 * RayEpsilon && isfinite(distance) &&
           !Occluded(context, Ray(origin, segment / distance, distance - RayEpsilon), statistics);
}
// Fixed-depth walks retain terminal vertices. PDFs describe the selected
// continuation, while Beta is the throughput before the current scattering.
void Extend(TransportContext context, thread Path& path, RayData ray, float3 beta, SampleContext samples,
            thread ThreadStatistics& statistics)
{
    uint domain = path.Mode == TransportRadiance ? CameraDomain : LightDomain;
    uint interiorMaterial = InvalidPrimitiveIndex;
    while (path.Count < path.Limit)
    {
        if (!all(isfinite(beta)))
        {
            ReportFailure(context);
            return;
        }
        if (!Positive(beta))
            return;
        if (path.Mode == TransportRadiance && path.Count == 1)
            IncrementCounter(statistics.Primary);
        else
            IncrementCounter(statistics.Reflection);
        HitData hit{};
        hit.Distance = ray.MaximumDistance;
        bool found = IntersectScene(context.Geometry, ray, false, hit, statistics.Tests);
        if (found && hit.MaterialIndex >= context.Frame->MaterialCount)
        {
            ReportFailure(context);
            return;
        }
        float nearest = found ? hit.Distance + RayEpsilon : ray.MaximumDistance;
        uint rectangle = InvalidPrimitiveIndex;
        for (uint source = 0; source < context.Batch->Sources; ++source)
            if (context.Emitters[source].Kind == PackedEmitterRectangle)
            {
                IncrementCounter(statistics.Tests);
                if (IntersectEmitter(context.Emitters[source], ray, nearest))
                    rectangle = source;
            }
        PathVertex vertex{};
        vertex.View = packed_float3(-ray.Direction);
        vertex.Beta = packed_float3(beta);
        vertex.Source = InvalidPrimitiveIndex;
        uint slot = path.Begin + path.Count;
        if (path.Count == 1 && found && !hit.FrontFace &&
            context.Materials[hit.MaterialIndex].Optical.w > 0)
            interiorMaterial = hit.MaterialIndex;
        if (interiorMaterial != InvalidPrimitiveIndex)
        {
            PackedMaterial interior = context.Materials[interiorMaterial];
            float endpoint = rectangle != InvalidPrimitiveIndex ? nearest :
                (found ? hit.Distance : ray.MaximumDistance);
            uint dimension = domain ^ (path.Count * BounceSeedMultiplier);
            auto flight = SubsurfaceFreeFlight(interior.Optical.w,
                float3(interior.AlbedoReflectivity.xyz) * interior.ConductorExtinction.w,
                ray.Direction, endpoint, Sample2D(samples, dimension ^ InteriorFlightDimension).x);
            if (flight.Scattered)
            {
                vertex.Kind = VolumeKind;
                vertex.Position = packed_float3(ray.Origin + ray.Direction * flight.Distance);
                vertex.Material = interiorMaterial;
                vertex.Beta = packed_float3(beta * flight.Weight);
                if (!all(isfinite(float3(vertex.Position))) || !all(isfinite(float3(vertex.Beta))))
                {
                    ReportFailure(context);
                    return;
                }
                ++path.Count;
                if (path.Count == path.Limit)
                {
                    context.Vertices[slot] = vertex;
                    return;
                }
                auto phase = Continue(context, vertex, path.Mode,
                                      Sample2D(samples, dimension ^ InteriorPhaseDimension), 0);
                vertex.Forward = phase.Pdf;
                vertex.Reverse = phase.ReversePdf;
                vertex.Event = phase.Event;
                context.Vertices[slot] = vertex;
                if (!(phase.Pdf > 0)) return;
                beta = float3(vertex.Beta) * phase.Weight;
                ray = Ray(float3(vertex.Position), phase.Direction);
                continue;
            }
        }
        if (rectangle != InvalidPrimitiveIndex)
        {
            vertex.Kind = EmitterKind;
            vertex.Position = packed_float3(ray.Origin + nearest * ray.Direction);
            vertex.Geometry = context.Emitters[rectangle].Direction;
            vertex.Source = rectangle;
        }
        else if (!found)
        {
            vertex.Kind = InfiniteKind;
            vertex.Source = SourceIndex(context, InvalidPrimitiveIndex, true);
        }
        else
        {
            vertex.Kind = SurfaceKind;
            vertex.Position = packed_float3(ray.Origin + hit.Distance * ray.Direction);
            vertex.Normal = packed_float3(hit.Normal);
            vertex.Geometry = packed_float3(hit.GeometricNormal);
            vertex.Material = hit.MaterialIndex;
            vertex.Color = packed_float3(HitColor(context, hit));
            vertex.Source = SourceIndex(context, hit.PrimitiveIndex, false);
        }
        ++path.Count;
        if (vertex.Kind != SurfaceKind || path.Count == path.Limit)
        {
            context.Vertices[slot] = vertex;
            return;
        }
        uint dimension = domain ^ (path.Count * BounceSeedMultiplier);
        SurfaceSample next = Continue(context, vertex, path.Mode, Sample2D(samples, dimension),
                                      Sample2D(samples, dimension ^ LobeSelectionScramble).x);
        vertex.Forward = next.Pdf;
        vertex.Reverse = next.ReversePdf;
        vertex.Event = next.Event;
        context.Vertices[slot] = vertex;
        if (!next.Event || !(next.Pdf > 0))
            return;
        beta *= next.Weight;
        ray = Ray(Offset(vertex, next.Direction), next.Direction);
        if (context.Materials[vertex.Material].Optical.w > 0 &&
            (next.Event & ScatteringTransmission))
            interiorMaterial = hit.FrontFace ? vertex.Material : InvalidPrimitiveIndex;
    }
}
Path CameraPath(TransportContext context, uint begin, SampleContext samples, bool owned,
                thread ThreadStatistics& statistics)
{
    constant FrameConstants& frame = *context.Frame;
    Path path{};
    path.Begin = begin;
    path.Limit = frame.MaxReflectionBounces + 1 + 2; // First scattering vertex and both endpoints.
    path.Mode = TransportRadiance;
    float2 jitter = Sample2D(samples, 0);
    path.Raster = CameraPixelRaster(samples.PixelX, samples.PixelY, frame.FullWidth, frame.FullHeight, jitter);
    CameraEndpointSample camera =
        SampleCameraEndpoint(CameraModel(frame), path.Raster, Sample2D(samples, LensDimension));
    if (!(camera.Flags & CameraEndpointValid))
        return path;
    path.Count = 1;
    path.EndpointFlags = camera.Flags;
    path.PositionPdf = camera.PositionPdf;
    path.DirectionPdf = camera.DirectionPdf;
    PathVertex root{};
    root.Kind = CameraKind;
    root.Position = packed_float3(camera.Origin);
    root.Geometry = frame.CameraForward;
    root.Beta = packed_float3(float3(1));
    root.Source = InvalidPrimitiveIndex;
    context.Vertices[begin] = root;
    if (owned)
        Extend(context, path, Ray(camera.Origin, camera.Direction), float3(1), samples, statistics);
    return path;
}
Path LightPath(TransportContext context, uint begin, SampleContext samples,
               thread ThreadStatistics& statistics)
{
    Path path{};
    path.Begin = begin;
    path.Limit = context.Frame->MaxReflectionBounces + 1 + 2;
    path.Mode = TransportImportance;
    EmitterEndpoint endpoint =
        SampleEmitterEndpoint(context.Emitters, context.Batch->Sources, EmissionBounds(context),
                              Sample2D(samples, EmitterDimension).x, Sample2D(samples, OriginDimension),
                              Sample2D(samples, RayDimension));
    EmissionSample emission = endpoint.Emission;
    if (!(emission.Flags & EmissionValid))
        return path;
    path.Count = 1;
    path.EndpointFlags = emission.Flags;
    path.SourcePmf = endpoint.SelectionProbability;
    path.PositionPdf = emission.PositionPdf;
    path.DirectionPdf = emission.DirectionPdf;
    path.LaunchDirection = emission.Direction;
    PathVertex root{};
    root.Kind = EmitterKind;
    root.Position = packed_float3(emission.Origin);
    root.Geometry = packed_float3(emission.Normal);
    root.Source = endpoint.EmitterIndex;
    float density = (emission.Flags & EmissionInfinite) != 0 ? emission.DirectionPdf : emission.PositionPdf;
    root.Beta = packed_float3(emission.Value / (path.SourcePmf * density));
    context.Vertices[begin] = root;
    float3 origin =
        (emission.Flags & EmissionDeltaPosition) != 0 ? emission.Origin : Offset(root, emission.Direction);
    Extend(context, path, Ray(origin, emission.Direction), emission.Weight, samples, statistics);
    return path;
}
PathVertex CompleteVertex(TransportContext context, Path light, Path camera, uint s, uint t, uint i,
                          PathVertex lens)
{
    if (i == s + t - 1)
        return lens;
    return context.Vertices[i < s ? light.Begin + i : camera.Begin + s + t - 1 - i];
}
CameraEndpointEvaluation CameraPdf(Path path)
{
    CameraEndpointEvaluation result{};
    result.Raster = path.Raster;
    result.Flags = path.EndpointFlags;
    result.PositionPdf = path.PositionPdf;
    result.DirectionPdf = path.DirectionPdf;
    return result;
}
uint SegmentInterior(TransportContext context, PathVertex from, PathVertex to)
{
    // Scene validation guarantees separated, outward-facing closed SSS solids.
    if (from.Kind == VolumeKind) return from.Material;
    if (to.Kind == VolumeKind) return to.Material;
    float3 segment = float3(to.Position) - float3(from.Position);
    if (from.Kind == SurfaceKind && context.Materials[from.Material].Optical.w > 0 &&
        dot(float3(from.Geometry), segment) < 0)
        return from.Material;
    if (to.Kind == SurfaceKind && context.Materials[to.Material].Optical.w > 0 &&
        dot(float3(to.Geometry), -segment) < 0)
        return to.Material;
    return InvalidPrimitiveIndex;
}
float SegmentTransmittance(TransportContext context, PathVertex from, PathVertex to)
{
    uint interior = SegmentInterior(context, from, to);
    return interior == InvalidPrimitiveIndex ? 1 :
        exp(-length(float3(to.Position) - float3(from.Position)) /
            context.Materials[interior].Optical.w);
}
PathDensity VertexDensity(TransportContext context, float pdf, uint event, PathVertex from,
                          PathVertex to, bool infinite)
{
    PathDensity density = MakePathDensity(pdf, (event & ScatteringDelta) != 0 ? PathDirac : PathSolidAngle);
    uint target = infinite ? PathInfiniteDestination :
        (to.Kind == VolumeKind ? PathVolumeDestination : PathSurfaceDestination);
    density = ConvertPathDensity(density, float3(to.Position) - float3(from.Position),
                                 float3(to.Geometry), target);
    if (!PathDensityValid(density) || infinite) return density;
    uint interior = SegmentInterior(context, from, to);
    if (interior == InvalidPrimitiveIndex)
        return target == PathVolumeDestination ? PathDensity{} : density;
    float meanFreePath = context.Materials[interior].Optical.w;
    density.Log2Value -= length(float3(to.Position) - float3(from.Position)) /
                         meanFreePath * log2(exp(1.0f));
    if (target == PathVolumeDestination) density.Log2Value -= log2(meanFreePath);
    return isfinite(density.Log2Value) ? density : PathDensity{};
}
// Neighbor directions change at the connection. Re-evaluate their BSDF PDFs
// and clear discarded delta flags; reverse cached PDFs in the camera prefix.
// Source PMFs remain ordinary factors, while true Dirac constraints retain their measure.
float Weight(TransportContext context, Path light, Path camera, uint s, uint t, PathVertex lens,
             CameraEndpointEvaluation sensor, uint source, uint flags, float3 emissionDirection)
{
    uint count = s + t;
    if (count == 2)
        return s == 0 ? 1.0f : 0.0f;
    bool infinite = (flags & EmissionInfinite) != 0;
    BidirectionalVertexPdf density[BdptMaximumVertices]{};
    BidirectionalStrategy strategy[BdptMaximumStrategies]{};
    for (uint i = 1; i + 1 < count; ++i)
    {
        PathVertex previous = CompleteVertex(context, light, camera, s, t, i - 1, lens);
        PathVertex current = CompleteVertex(context, light, camera, s, t, i, lens);
        PathVertex next = CompleteVertex(context, light, camera, s, t, i + 1, lens);
        float forward, reverse;
        uint event = 0;
        if ((s != 0 && i == s - 1) || i == s)
        {
            float3 view = i == 1 && infinite
                              ? -emissionDirection
                              : normalize(float3(previous.Position) - float3(current.Position));
            SurfaceEvaluation scattering =
                Evaluate(context, current, view, normalize(float3(next.Position) - float3(current.Position)),
                         TransportImportance);
            forward = scattering.Pdf;
            reverse = scattering.ReversePdf;
        }
        else
        {
            event = current.Event;
            forward = i < s ? current.Forward : current.Reverse;
            reverse = i < s ? current.Reverse : current.Forward;
        }
        density[i].Flags = (event & ScatteringDelta) != 0 ? PathDeltaScattering : 0;
        density[i + 1].Forward = VertexDensity(context, forward, event, current, next, false);
        density[i - 1].Reverse = VertexDensity(context, reverse, event, current, previous,
                                              infinite && i == 1);
    }
    PathVertex root = CompleteVertex(context, light, camera, s, t, 0, lens);
    PathVertex first = CompleteVertex(context, light, camera, s, t, 1, lens);
    float3 launch = infinite ? emissionDirection : normalize(float3(first.Position) - float3(root.Position));
    EmitterEndpointDensity emitter =
        EmitterEndpointPdf(context.Emitters, context.Batch->Sources, EmissionBounds(context), source,
                           float3(root.Geometry), launch);
    bool deltaPosition = (emitter.Flags & EmissionDeltaPosition) != 0;
    bool deltaDirection = (emitter.Flags & EmissionDeltaDirection) != 0;
    PathDensity position =
        MakePathDensity(deltaPosition ? 1 : emitter.Position, deltaPosition ? PathDirac : PathArea);
    PathDensity direction =
        MakePathDensity(deltaDirection ? 1 : emitter.Direction, deltaDirection ? PathDirac : PathSolidAngle);
    if (infinite)
        density[1].Forward = ProjectedDiskDensity(position, float3(first.Geometry), launch);
    else
        density[1].Forward = VertexDensity(context, deltaDirection ? 1 : emitter.Direction,
                                           deltaDirection ? ScatteringDelta : 0, root, first, false);
    density[0].Flags = deltaPosition || deltaDirection ? PathEndpointNotIntersectable : 0;
    PathVertex last = CompleteVertex(context, light, camera, s, t, count - 2, lens);
    density[count - 2].Reverse = VertexDensity(context, sensor.DirectionPdf, 0, lens, last, false);
    bool pinhole = (sensor.Flags & CameraDeltaPosition) != 0;
    density[count - 1].Flags = pinhole ? PathEndpointNotIntersectable : 0;
    for (uint cut = 0; cut < count; ++cut)
    {
        strategy[cut].Enabled = cut <= light.Limit && count - cut <= camera.Limit;
        strategy[cut].LightEndpoint = cut == 0   ? MakePathDensity(1, PathDirac)
                                      : infinite ? direction
                                                 : position;
        strategy[cut].CameraEndpoint =
            MakePathDensity(pinhole ? 1 : sensor.PositionPdf, pinhole ? PathDirac : PathArea);
        strategy[cut].LightSelection = cut == 0 ? 1 : emitter.SelectionProbability;
        strategy[cut].SampleCount = 1;
    }
    return BidirectionalMisWeightsLocal(density, count, strategy,
                                        (context.Frame->TransportFlags >> MisShift) & MisMask)
        .Values[s];
}
float3 HitEmission(TransportContext context, Path light, Path camera, uint t)
{
    PathVertex hit = context.Vertices[camera.Begin + t - 1];
    if (hit.Source >= context.Batch->Sources)
        return float3(0);
    float3 value = float3(hit.Beta) * Emission(context.Emitters[hit.Source], float3(hit.View));
    if (!all(isfinite(value)))
    {
        ReportFailure(context);
        return float3(0);
    }
    if (!Positive(value))
        return float3(0);
    EmitterEndpointDensity emitter =
        EmitterEndpointPdf(context.Emitters, context.Batch->Sources, EmissionBounds(context), hit.Source,
                           float3(hit.Geometry), float3(hit.View));
    return value * Weight(context, light, camera, 0, t, context.Vertices[camera.Begin], CameraPdf(camera),
                          hit.Source, emitter.Flags, float3(hit.View));
}
float3 ConnectSource(TransportContext context, Path light, Path camera, uint t,
                     thread ThreadStatistics& statistics)
{
    PathVertex surface = context.Vertices[camera.Begin + t - 1], source = context.Vertices[light.Begin];
    if (surface.Kind != SurfaceKind && surface.Kind != VolumeKind)
        return float3(0);
    bool infinite = (light.EndpointFlags & EmissionInfinite) != 0;
    bool deltaPosition = (light.EndpointFlags & EmissionDeltaPosition) != 0;
    float3 direction = -light.LaunchDirection;
    float geometry = 1;
    if (!infinite)
    {
        float3 segment = float3(source.Position) - float3(surface.Position);
        float squared = dot(segment, segment);
        if (!(squared > 0) || !isfinite(squared))
            return float3(0);
        direction = segment * rsqrt(squared);
        geometry = (deltaPosition ? 1 : fabs(dot(float3(source.Geometry), direction))) / squared;
    }
    SurfaceEvaluation bsdf = Evaluate(context, surface, float3(surface.View), direction, TransportRadiance);
    float density = light.SourcePmf * (infinite ? light.DirectionPdf : light.PositionPdf);
    float cosine = surface.Kind == VolumeKind ? 1 : fabs(dot(float3(surface.Normal), direction));
    float3 value = float3(surface.Beta) * bsdf.Value * Emission(context.Emitters[source.Source], -direction) *
                   (cosine * geometry / density *
                    (infinite ? 1 : SegmentTransmittance(context, surface, source)));
    if (!all(isfinite(value)))
    {
        ReportFailure(context);
        return float3(0);
    }
    if (!Positive(value))
        return float3(0);
    if (infinite)
    {
        if (Occluded(context, Ray(Offset(surface, direction), direction), statistics))
            return float3(0);
    }
    else if (!ConnectionVisible(context, Offset(surface, direction),
                                deltaPosition ? float3(source.Position) : Offset(source, -direction),
                                statistics))
        return float3(0);
    return value * Weight(context, light, camera, 1, t, context.Vertices[camera.Begin], CameraPdf(camera),
                          source.Source, light.EndpointFlags, light.LaunchDirection);
}
struct Contribution
{
    float3 Value;
    float2 Raster;
};
Contribution ConnectSurface(TransportContext context, Path light, Path camera, uint s, uint t, float2 lensU,
                            thread ThreadStatistics& statistics)
{
    Contribution result{};
    PathVertex a = context.Vertices[light.Begin + s - 1], b = context.Vertices[camera.Begin + t - 1];
    if ((a.Kind != SurfaceKind && a.Kind != VolumeKind) ||
        (t > 1 && b.Kind != SurfaceKind && b.Kind != VolumeKind))
        return result;
    PathVertex lens = context.Vertices[camera.Begin];
    CameraEndpointEvaluation sensor = CameraPdf(camera);
    result.Raster = camera.Raster;
    float3 direction, target, value;
    if (t == 1)
    {
        CameraConnectionSample connection =
            SampleCameraConnection(CameraModel(*context.Frame), float3(a.Position), lensU);
        if (!(connection.Flags & CameraEndpointValid))
            return result;
        direction = connection.Direction;
        target = connection.Position;
        SurfaceEvaluation bsdf = Evaluate(context, a, float3(a.View), direction, TransportImportance);
        lens.Position = packed_float3(target);
        float cosine = a.Kind == VolumeKind ? 1 : fabs(dot(float3(a.Normal), direction));
        value = float3(a.Beta) * bsdf.Value *
                (cosine * connection.Weight * SegmentTransmittance(context, a, lens));
        sensor = EvaluateCameraEndpoint(CameraModel(*context.Frame), connection.LensPoint, -direction);
        result.Raster = connection.Raster;
    }
    else
    {
        float3 segment = float3(b.Position) - float3(a.Position);
        float squared = dot(segment, segment);
        if (!(squared > 0) || !isfinite(squared))
            return result;
        direction = segment * rsqrt(squared);
        SurfaceEvaluation lightBsdf = Evaluate(context, a, float3(a.View), direction, TransportImportance);
        SurfaceEvaluation cameraBsdf = Evaluate(context, b, float3(b.View), -direction, TransportRadiance);
        float cosineA = a.Kind == VolumeKind ? 1 : fabs(dot(float3(a.Normal), direction));
        float cosineB = b.Kind == VolumeKind ? 1 : fabs(dot(float3(b.Normal), direction));
        float geometry = cosineA * cosineB / squared;
        value = float3(a.Beta) * lightBsdf.Value * float3(b.Beta) * cameraBsdf.Value *
                (geometry * SegmentTransmittance(context, a, b));
        target = Offset(b, -direction);
    }
    if (!all(isfinite(value)))
    {
        ReportFailure(context);
        return result;
    }
    if (!Positive(value) || !ConnectionVisible(context, Offset(a, direction), target, statistics))
        return result;
    uint source = context.Vertices[light.Begin].Source;
    result.Value = value * Weight(context, light, camera, s, t, lens, sensor, source, light.EndpointFlags,
                                  light.LaunchDirection);
    return result;
}
bool LocalPixel(constant FrameConstants& frame, uint x, uint y, thread uint& index)
{
    index = 0;
    if (x < frame.TileX || y < frame.TileY || x - frame.TileX >= frame.Width ||
        y - frame.TileY >= frame.Height)
        return false;
    index = (y - frame.TileY) * frame.Width + x - frame.TileX;
    return true;
}
void AddContribution(TransportContext context, uint pixel, float3 value)
{
    for (uint channel = 0; channel < ColorChannels; ++channel)
        if (!RadianceAdd(context.Pass + pixel * PassComponents + channel * RadianceWords, value[channel]))
            ReportFailure(context);
}
void EvaluatePair(TransportContext context, uint local, thread ThreadStatistics& statistics)
{
    constant FrameConstants& frame = *context.Frame;
    uint identity = context.Batch->FirstPath + local;
    uint x = identity % frame.FullWidth, y = identity / frame.FullWidth, pixel;
    bool owned = LocalPixel(frame, x, y, pixel);
    SampleContext samples = MakeSampleContext(x, y, frame.FullWidth, frame.SampleBegin, frame.TotalSamples,
                                              frame.SamplingSequence, frame.SamplingSeed);
    uint base = local * 2 * BdptMaximumVertices;
    Path camera = CameraPath(context, base + BdptMaximumVertices, samples, owned, statistics);
    Path light = LightPath(context, base, samples, statistics);
    float3 color = float3(0);
    for (uint t = 2; t <= camera.Count; ++t)
        color += HitEmission(context, light, camera, t);
    if (camera.Count && light.Count)
    {
        for (uint t = 2; t <= camera.Count && t < camera.Limit; ++t)
            color += ConnectSource(context, light, camera, t, statistics);
        for (uint s = 2; s <= light.Count; ++s)
        {
            float2 lens = Sample2D(samples, ConnectionLensDimension ^ (s * BounceSeedMultiplier));
            for (uint t = 1; t <= camera.Count && s + t <= camera.Limit; ++t)
            {
                Contribution connection = ConnectSurface(context, light, camera, s, t, lens, statistics);
                if (t > 1)
                    color += connection.Value;
                else if (Positive(connection.Value) && CameraValidRaster(connection.Raster))
                {
                    uint target;
                    if (LocalPixel(frame, uint(connection.Raster.x * float(frame.FullWidth)),
                                   uint(connection.Raster.y * float(frame.FullHeight)), target))
                        AddContribution(context, target, connection.Value);
                }
            }
        }
    }
    if (owned)
        AddContribution(context, pixel, color);
}

kernel void ArvoBidirectionalPairs(
    device const PackedAccelerationNode* nodes [[buffer(0)]], device const uint* references [[buffer(1)]],
    device const PackedTriangle* triangles [[buffer(2)]], device const PackedSphere* spheres [[buffer(3)]],
    device const PackedMaterial* materials [[buffer(4)]], device const PackedEmitter* emitters [[buffer(5)]],
    constant FrameConstants& frame [[buffer(6)]], device ThreadStatistics* statistics [[buffer(9)]],
    device const PackedTextureDescriptor* textures [[buffer(10)]], device const uint* texels [[buffer(11)]],
    device const InstanceRecord* instances [[buffer(13)]],
    device const InstanceNode* instanceNodes [[buffer(14)]],
    device atomic_uint* traversalError [[buffer(15)]], device PathVertex* vertices [[buffer(16)]],
    device atomic_uint* pass [[buffer(17)]], constant BidirectionalControls& batch [[buffer(18)]],
    device atomic_uint* error [[buffer(19)]], uint local [[thread_position_in_grid]])
{
    if (local >= batch.PathCount)
        return;
    SceneGeometryBuffers geometry{
        {nodes, references, triangles, spheres, frame.NodeCount, frame.TriangleCount, frame.Acceleration},
        instances,
        instanceNodes,
        traversalError,
        frame.InstanceNodeCount};
    TransportContext context{geometry, materials, textures, texels, emitters,
                             &frame,   &batch,    vertices, pass,   error};
    ThreadStatistics result{};
    EvaluatePair(context, local, result);
    statistics[local] = result;
}
kernel void ArvoBidirectionalClear(constant FrameConstants& frame [[buffer(6)]],
                                   device atomic_uint* pass [[buffer(17)]],
                                   device atomic_uint* error [[buffer(19)]],
                                   uint pixel [[thread_position_in_grid]])
{
    if (pixel >= frame.Width * frame.Height)
        return;
    for (uint channel = 0; channel < PassComponents; ++channel)
        atomic_store_explicit(pass + pixel * PassComponents + channel, 0u, memory_order_relaxed);
    if (pixel == 0)
        atomic_store_explicit(error, 0u, memory_order_relaxed);
}
kernel void ArvoBidirectionalCommit(constant FrameConstants& frame [[buffer(6)]],
                                    device float4* accumulation [[buffer(7)]],
                                    device uint* counts [[buffer(8)]], device uint* targets [[buffer(12)]],
                                    device atomic_uint* pass [[buffer(17)]],
                                    device atomic_uint* error [[buffer(19)]],
                                    uint pixel [[thread_position_in_grid]])
{
    if (pixel >= frame.Width * frame.Height)
        return;
    float3 color;
    for (uint channel = 0; channel < ColorChannels; ++channel)
        color[channel] = RadianceRound(pass + pixel * PassComponents + channel * RadianceWords);
    if (!all(isfinite(color)))
    {
        atomic_store_explicit(error, NumericalFailure, memory_order_relaxed);
        return;
    }
    float4 moments = accumulation[pixel];
    uint count = counts[pixel];
    AccumulateFilmSample(moments, count, color);
    if (!all(isfinite(moments)))
    {
        atomic_store_explicit(error, NumericalFailure, memory_order_relaxed);
        return;
    }
    accumulation[pixel] = moments;
    counts[pixel] = count;
    targets[pixel] = frame.TotalSamples;
}
