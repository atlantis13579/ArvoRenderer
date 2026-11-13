// Independent DXR bidirectional path tracing. Veach (1997), chapter 10;
// PBRT 3e, "Bidirectional Path Tracing". Ray generation owns both random walks
// and all connections. Hit/callable shaders only return surface attributes.
// https://pbr-book.org/3ed-2018/Light_Transport_III_Bidirectional_Methods/Bidirectional_Path_Tracing
// A film observation is one full-image pass, including light-to-lens splats.
// Interior collisions add volume vertices with free-flight path densities.
static const float Pi = 3.14159265358979323846f;
static const float RayEpsilon = 1.0e-4f, MaximumRayDistance = 1.0e30f;
static const float BasisParallelThreshold = 0.999f;
static const uint PixelSeedMultiplier = 9781u, SampleSeedMultiplier = 6271u;
static const uint BounceSeedMultiplier = 0x9e3779b9u, SecondDimensionScramble = 0x68bc21ebu;
static const uint LobeSelectionScramble = 0xa511e9b3u, ImportanceBit = 1u;
static const uint MisShift = 3u, MisMask = 3u;
static const uint CameraStream = 0x43414d50u, LightStream = 0x4c495450u;
static const uint ChooseSourceDimension = 0x53524345u, SourcePointDimension = 0x53504f53u;
static const uint SourceRayDimension = 0x53444952u, CameraLensDimension = 0x4c454e53u;
static const uint ConnectLensDimension = 0x434c454eu;
static const uint InteriorFlightDimension = 0x49464c54u, InteriorPhaseDimension = 0x49504853u;
static const uint SensorNode = 0, EmitterNode = 1, SurfaceNode = 2, EscapeNode = 3,
                  VolumeNode = 4;
static const uint ClearStage = 0, ConnectStage = 1, CommitStage = 2;
#include "RadianceAccumulator.hlsl"
static const uint ArithmeticError = 1, WordBytes = 4, ColorChannels = 3;
static const uint PassChannelBytes = SplatSumWordCount * WordBytes;
static const uint PassPixelBytes = ColorChannels * PassChannelBytes;

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
#include "BidirectionalMis.hlsl"

struct TextureInfo { uint OffsetTexels, Width, Height, Reserved; };
struct PathVertex
{
    float3 Position;
    uint Kind;
    float3 Incoming;
    uint Material;
    float3 ShadingNormal;
    uint Emitter;
    float3 GeometricNormal;
    uint Event;
    float3 Color;
    float ForwardPdf;
    float3 Throughput;
    float ReversePdf;
};
struct Subpath
{
    uint First, Count, Limit, Mode;
    float3 EmissionDirection;
    float SourceProbability;
    float PositionPdf, DirectionPdf;
    float2 Raster;
    uint EndpointFlags;
};
struct PathCounters { uint2 Primary, Shadow, Reflection; };
StructuredBuffer<MaterialData> Materials : register(t3);
StructuredBuffer<TextureInfo> Textures : register(t5);
StructuredBuffer<uint> TextureTexels : register(t6);
StructuredBuffer<PackedEmitter> Emitters : register(t13);
RWStructuredBuffer<uint> OutputPixels : register(u0);
RWStructuredBuffer<uint> Statistics : register(u1);
RWStructuredBuffer<LinearFilmPixel> LinearFilm : register(u2);
RWStructuredBuffer<PathVertex> PathVertices : register(u5);
RWByteAddressBuffer PassRadiance : register(u6);
RWByteAddressBuffer PathErrors : register(u7);
cbuffer BidirectionalBatch : register(b1)
{
    float3 EmissionCenter;
    float EmissionRadius;
    uint EmitterCount, FirstPath, PassIndex, Stage;
};
#include "Statistics.hlsl"

uint Hash(uint bits)
{
    static const uint FirstMix = 0x7feb352du, SecondMix = 0x846ca68bu;
    static const uint HalfWord = 16u, MiddleShift = 15u;
    bits ^= bits >> HalfWord;
    bits *= FirstMix;
    bits ^= bits >> MiddleShift;
    bits *= SecondMix;
    return bits ^ (bits >> HalfWord);
}
float Random01(uint bits)
{
    static const uint MantissaMask = 0x00ffffffu;
    static const float MantissaRange = 16777216.0f;
    return float(Hash(bits) & MantissaMask) / MantissaRange;
}
#include "Sequence.hlsl"
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

void ReportArithmeticError()
{
    uint ignored;
    PathErrors.InterlockedOr(0, ArithmeticError, ignored);
}
EmissionDomain SourceDomain()
{
    EmissionDomain domain;
    domain.Center = EmissionCenter;
    domain.Radius = EmissionRadius;
    return domain;
}
CameraEndpointFrame SensorFrame()
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
float3 OffsetVertex(PathVertex vertex, float3 direction)
{
    if (vertex.Kind == VolumeNode) return vertex.Position;
    float sign = dot(vertex.GeometricNormal, direction) >= 0 ? 1 : -1;
    return vertex.Position + vertex.GeometricNormal * (sign * RayEpsilon);
}
float DecodeChannel(uint byte)
{
    static const float ByteRange = 255, Threshold = 0.04045f;
    static const float LinearSlope = 12.92f, NonlinearScale = 1.055f, NonlinearOffset = 0.055f;
    static const float Exponent = 2.4f;
    float value = float(byte) / ByteRange;
    return value <= Threshold ? value / LinearSlope
                              : pow((value + NonlinearOffset) / NonlinearScale, Exponent);
}
float3 TextureSample(TextureInfo texture, int2 texel)
{
    int2 size = int2(texture.Width, texture.Height);
    texel = (texel % size + size) % size;
    uint encoded = TextureTexels[texture.OffsetTexels + texel.y * texture.Width + texel.x];
    static const uint ByteMask = 0xffu, GreenShift = 8u, BlueShift = 16u;
    return float3(DecodeChannel(encoded & ByteMask), DecodeChannel((encoded >> GreenShift) & ByteMask),
                  DecodeChannel((encoded >> BlueShift) & ByteMask));
}
float3 SurfaceColor(Payload hit)
{
    uint index = Materials[hit.MaterialIndex].DiffuseTextureIndex;
    if ((hit.Flags & SurfaceTexCoordFlag) == 0 || index >= TextureCount || !all(isfinite(hit.TexCoord)))
        return hit.Color;
    TextureInfo texture = Textures[index];
    if (texture.Width == 0 || texture.Height == 0 || texture.Height > TextureTexelCount / texture.Width ||
        texture.OffsetTexels > TextureTexelCount ||
        texture.Width * texture.Height > TextureTexelCount - texture.OffsetTexels)
        return hit.Color;
    float2 location = frac(float2(hit.TexCoord.x, 1 - hit.TexCoord.y)) *
                      float2(texture.Width, texture.Height) - 0.5f;
    int2 base = int2(floor(location));
    float2 blend = frac(location);
    return hit.Color * lerp(lerp(TextureSample(texture, base), TextureSample(texture, base + int2(1, 0)), blend.x),
                            lerp(TextureSample(texture, base + int2(0, 1)), TextureSample(texture, base + 1), blend.x), blend.y);
}
SurfaceBsdf VertexScattering(PathVertex vertex, float3 incoming, uint mode)
{
    Payload attributes = (Payload)0;
    attributes.Normal = dot(vertex.ShadingNormal, incoming) < 0 ? -vertex.ShadingNormal : vertex.ShadingNormal;
    attributes.GeometricNormal = vertex.GeometricNormal;
    attributes.Flags = dot(vertex.GeometricNormal, incoming) > 0 ? SurfaceFrontFaceFlag : 0;
    MaterialData material = Materials[vertex.Material];
    if (material.Model == MaterialLegacy) material.Model = MaterialDiffuse;
    return MakeSurfaceBsdf(material, attributes, incoming, vertex.Color, mode);
}
SurfaceEvaluation EvaluateVertex(PathVertex vertex, float3 incoming, float3 outgoing, uint mode)
{
    if (vertex.Kind == VolumeNode)
    {
        float phase = SubsurfacePhase(dot(-incoming, outgoing), Materials[vertex.Material].ConductorIndex.w);
        float pdf = (TransportFlags & ImportanceBit) != 0 ? phase : 1 / (4 * Pi);
        SurfaceEvaluation result = (SurfaceEvaluation)0;
        result.Value = phase.xxx;
        result.Pdf = result.ReversePdf = pdf;
        return result;
    }
    SurfaceEvaluation value = EvaluateSurface(VertexScattering(vertex, incoming, mode), outgoing);
    MaterialData material = Materials[vertex.Material];
    if (material.Model == MaterialLegacy)
    {
        float diffuseProbability = 1 - saturate(material.AlbedoReflectivity.w);
        value.Value *= diffuseProbability;
        value.Pdf *= diffuseProbability;
        value.ReversePdf *= diffuseProbability;
    }
    return value;
}
SurfaceSample ScatterVertex(PathVertex vertex, uint mode, float2 directionVariates, float lobeVariate)
{
    if (vertex.Kind == VolumeNode)
    {
        SubsurfaceDirection phase = SubsurfaceSampleDirection(-vertex.Incoming,
            Materials[vertex.Material].ConductorIndex.w, directionVariates,
            (TransportFlags & ImportanceBit) != 0);
        SurfaceSample result = (SurfaceSample)0;
        result.Direction = phase.Direction;
        result.Weight = phase.Weight.xxx;
        result.Pdf = result.ReversePdf = phase.Pdf;
        result.RelativeIndex = 1;
        result.Event = ScatteringGlossy;
        return result;
    }
    SurfaceBsdf bsdf = VertexScattering(vertex, vertex.Incoming, mode);
    MaterialData material = Materials[vertex.Material];
    float mirrorProbability = material.Model == MaterialLegacy ? saturate(material.AlbedoReflectivity.w) : 0;
    if (lobeVariate < mirrorProbability)
    {
        SurfaceSample mirror = (SurfaceSample)0;
        float3 outgoing = reflect(-vertex.Incoming, bsdf.Normal);
        if (!SurfaceDirectionValid(bsdf, outgoing, dot(outgoing, bsdf.Normal))) return mirror;
        mirror.Direction = outgoing;
        mirror.Weight = SurfaceAdjointCorrection(bsdf, outgoing).xxx;
        mirror.Pdf = mirror.ReversePdf = mirrorProbability;
        mirror.RelativeIndex = 1;
        mirror.Event = ScatteringReflection | ScatteringDelta;
        return mirror;
    }
    SurfaceSample sample = SampleSurface(bsdf, directionVariates, lobeVariate);
    if (material.Model == MaterialLegacy)
    {
        sample.Pdf *= 1 - mirrorProbability;
        sample.ReversePdf *= 1 - mirrorProbability;
    }
    return sample;
}
uint EmitterAt(uint primitive, bool escaped)
{
    for (uint source = 0; source < EmitterCount; ++source)
        if (escaped ? Emitters[source].Kind == PackedEmitterEnvironment
                    : (primitive != InvalidPrimitiveIndex && Emitters[source].Primitive == primitive))
            return source;
    return InvalidPrimitiveIndex;
}
bool IntersectRectangle(PackedEmitter source, float3 origin, float3 direction, float maximum,
                        inout float nearest)
{
    if (source.Kind != PackedEmitterRectangle) return false;
    float denominator = dot(source.Direction, direction);
    if (denominator == 0) return false;
    float distance = dot(source.Direction, source.Origin - origin) / denominator;
    if (!(distance > RayEpsilon && distance < maximum && distance <= nearest)) return false;
    float3 helper = abs(source.Direction.y) < BasisParallelThreshold ? float3(0, 1, 0) : float3(1, 0, 0);
    float3 tangent = normalize(cross(helper, source.Direction)), bitangent = cross(source.Direction, tangent);
    float3 relative = origin + distance * direction - source.Origin;
    if (abs(dot(relative, tangent)) > source.Width / 2 || abs(dot(relative, bitangent)) > source.Height / 2)
        return false;
    nearest = distance;
    return true;
}
float3 SourceRadiance(uint index, float3 direction)
{
    PackedEmitter source = Emitters[index];
    if (source.Kind == PackedEmitterEnvironment)
        return lerp(source.BackgroundBottom, source.BackgroundTop, (1 - direction.y) / 2);
    if (source.Kind == PackedEmitterRectangle)
        return dot(source.Direction, direction) > 0 ? source.Strength / (source.Width * source.Height)
                                                   : float3(0, 0, 0);
    return source.Strength;
}
bool Shadowed(float3 origin, float3 direction, float maximum, inout PathCounters rays)
{
    IncrementLocalCounter(rays.Shadow);
    Payload hit = CastRay(origin, direction, RayEpsilon, maximum);
    if ((hit.Flags & SurfaceHitFlag) != 0) return true;
    float nearest = maximum;
    for (uint index = 0; index < EmitterCount; ++index)
        if (IntersectRectangle(Emitters[index], origin, direction, maximum, nearest)) return true;
    return false;
}
bool VisibleSegment(float3 begin, float3 end, inout PathCounters rays)
{
    float3 difference = end - begin;
    float distance = length(difference);
    return distance > 2 * RayEpsilon && isfinite(distance) &&
           !Shadowed(begin, difference / distance, distance - RayEpsilon, rays);
}

// Every stored throughput precedes that vertex's scattering. The final surface
// is retained even at the depth limit, because a connection may end there.
void ExtendSubpath(inout Subpath path, float3 origin, float3 direction, float3 beta,
                   SampleContext samples, inout PathCounters rays)
{
    uint stream = path.Mode == TransportRadiance ? CameraStream : LightStream;
    uint interiorMaterial = InvalidPrimitiveIndex;
    [loop]
    while (path.Count < path.Limit)
    {
        if (!all(isfinite(beta))) { ReportArithmeticError(); return; }
        if (!any(beta > 0)) return;
        if (path.Mode == TransportRadiance && path.Count == 1) IncrementLocalCounter(rays.Primary);
        else IncrementLocalCounter(rays.Reflection);
        Payload hit = CastRay(origin, direction, RayEpsilon, MaximumRayDistance);
        bool surface = (hit.Flags & SurfaceHitFlag) != 0;
        if (surface && hit.MaterialIndex >= MaterialCount) { ReportArithmeticError(); return; }
        float nearest = surface ? asfloat(hit.DistanceBits) + RayEpsilon : MaximumRayDistance;
        uint rectangle = InvalidPrimitiveIndex;
        for (uint emitter = 0; emitter < EmitterCount; ++emitter)
            if (IntersectRectangle(Emitters[emitter], origin, direction, MaximumRayDistance, nearest))
                rectangle = emitter;
        PathVertex vertex = (PathVertex)0;
        vertex.Incoming = -direction;
        vertex.Throughput = beta;
        vertex.Emitter = InvalidPrimitiveIndex;
        uint destination = path.First + path.Count;
        if (path.Count == 1 && surface && (hit.Flags & SurfaceFrontFaceFlag) == 0 &&
            Materials[hit.MaterialIndex].Optical.w > 0)
            interiorMaterial = hit.MaterialIndex;
        if (interiorMaterial != InvalidPrimitiveIndex)
        {
            MaterialData interior = Materials[interiorMaterial];
            float endpoint = rectangle != InvalidPrimitiveIndex ? nearest :
                (surface ? asfloat(hit.DistanceBits) : MaximumRayDistance);
            uint dimension = stream ^ (path.Count * BounceSeedMultiplier);
            SubsurfaceFlight flight = SubsurfaceFreeFlight(interior.Optical.w,
                interior.AlbedoReflectivity.xyz * interior.ConductorExtinction.w,
                direction, endpoint, Sample2D(samples, dimension ^ InteriorFlightDimension).x);
            if (flight.Scattered != 0)
            {
                vertex.Kind = VolumeNode;
                vertex.Position = origin + direction * flight.Distance;
                vertex.Material = interiorMaterial;
                vertex.Throughput *= flight.Weight;
                if (!all(isfinite(vertex.Position)) || !all(isfinite(vertex.Throughput)))
                { ReportArithmeticError(); return; }
                ++path.Count;
                if (path.Count == path.Limit)
                {
                    PathVertices[destination] = vertex;
                    return;
                }
                SurfaceSample phase = ScatterVertex(vertex, path.Mode,
                    Sample2D(samples, dimension ^ InteriorPhaseDimension), 0);
                vertex.ForwardPdf = phase.Pdf;
                vertex.ReversePdf = phase.ReversePdf;
                vertex.Event = phase.Event;
                PathVertices[destination] = vertex;
                if (!(phase.Pdf > 0)) return;
                beta = vertex.Throughput * phase.Weight;
                direction = phase.Direction;
                origin = vertex.Position;
                continue;
            }
        }
        if (rectangle != InvalidPrimitiveIndex)
        {
            vertex.Kind = EmitterNode;
            vertex.Position = origin + nearest * direction;
            vertex.GeometricNormal = Emitters[rectangle].Direction;
            vertex.Emitter = rectangle;
        }
        else if (!surface)
        {
            vertex.Kind = EscapeNode;
            vertex.Emitter = EmitterAt(InvalidPrimitiveIndex, true);
        }
        else
        {
            vertex.Kind = SurfaceNode;
            vertex.Position = origin + asfloat(hit.DistanceBits) * direction;
            vertex.ShadingNormal = hit.Normal;
            vertex.GeometricNormal = hit.GeometricNormal;
            vertex.Material = hit.MaterialIndex;
            vertex.Color = SurfaceColor(hit);
            vertex.Emitter = EmitterAt(hit.PrimitiveIndex, false);
        }
        ++path.Count;
        if (vertex.Kind != SurfaceNode || path.Count == path.Limit)
        {
            PathVertices[destination] = vertex;
            return;
        }
        uint dimension = stream ^ (path.Count * BounceSeedMultiplier);
        SurfaceSample continuation = ScatterVertex(vertex, path.Mode, Sample2D(samples, dimension),
                                                    Sample2D(samples, dimension ^ LobeSelectionScramble).x);
        vertex.ForwardPdf = continuation.Pdf;
        vertex.ReversePdf = continuation.ReversePdf;
        vertex.Event = continuation.Event;
        PathVertices[destination] = vertex;
        if (continuation.Event == 0 || !(continuation.Pdf > 0)) return;
        beta *= continuation.Weight;
        direction = continuation.Direction;
        origin = OffsetVertex(vertex, direction);
        if (Materials[vertex.Material].Optical.w > 0 &&
            (continuation.Event & ScatteringTransmission) != 0)
            interiorMaterial = (hit.Flags & SurfaceFrontFaceFlag) != 0 ? vertex.Material : InvalidPrimitiveIndex;
    }
}
Subpath CameraSubpath(uint first, SampleContext samples, bool insideTile, inout PathCounters rays)
{
    Subpath path = (Subpath)0;
    path.First = first;
    path.Limit = MaxReflectionBounces + 1 + 2;
    path.Mode = TransportRadiance;
    path.Raster = CameraPixelRaster(samples.PixelX, samples.PixelY, FullWidth, FullHeight, Sample2D(samples, 0));
    CameraEndpointSample sensor = SampleCameraEndpoint(SensorFrame(), path.Raster, Sample2D(samples, CameraLensDimension));
    if ((sensor.Flags & CameraEndpointValid) == 0) return path;
    path.EndpointFlags = sensor.Flags;
    path.PositionPdf = sensor.PositionPdf;
    path.DirectionPdf = sensor.DirectionPdf;
    path.Count = 1;
    PathVertex root = (PathVertex)0;
    root.Kind = SensorNode;
    root.Position = sensor.Origin;
    root.GeometricNormal = CameraForward;
    root.Throughput = 1;
    root.Emitter = InvalidPrimitiveIndex;
    PathVertices[first] = root;
    if (insideTile) ExtendSubpath(path, sensor.Origin, sensor.Direction, float3(1, 1, 1), samples, rays);
    return path;
}
Subpath LightSubpath(uint first, SampleContext samples, inout PathCounters rays)
{
    Subpath path = (Subpath)0;
    path.First = first;
    path.Limit = MaxReflectionBounces + 1 + 2;
    path.Mode = TransportImportance;
    EmitterEndpoint endpoint = SampleEmitterEndpoint(Emitters, EmitterCount, SourceDomain(),
        Sample2D(samples, ChooseSourceDimension).x, Sample2D(samples, SourcePointDimension),
        Sample2D(samples, SourceRayDimension));
    EmissionSample ray = endpoint.Emission;
    if ((ray.Flags & EmissionValid) == 0) return path;
    path.Count = 1;
    path.EndpointFlags = ray.Flags;
    path.PositionPdf = ray.PositionPdf;
    path.DirectionPdf = ray.DirectionPdf;
    path.EmissionDirection = ray.Direction;
    path.SourceProbability = endpoint.SelectionProbability;
    PathVertex root = (PathVertex)0;
    root.Kind = EmitterNode;
    root.Position = ray.Origin;
    root.GeometricNormal = ray.Normal;
    root.Emitter = endpoint.EmitterIndex;
    float density = (ray.Flags & EmissionInfinite) != 0 ? ray.DirectionPdf : ray.PositionPdf;
    root.Throughput = ray.Value / (density * endpoint.SelectionProbability);
    PathVertices[first] = root;
    float3 origin = (ray.Flags & EmissionDeltaPosition) != 0 ? ray.Origin : OffsetVertex(root, ray.Direction);
    ExtendSubpath(path, origin, ray.Direction, ray.Weight, samples, rays);
    return path;
}

PathVertex CompleteVertex(Subpath light, Subpath camera, uint s, uint t, uint index, PathVertex sensor)
{
    if (index == s + t - 1) return sensor;
    uint slot = index < s ? light.First + index : camera.First + s + t - 1 - index;
    return PathVertices[slot];
}
CameraEndpointEvaluation CameraMeasure(Subpath path)
{
    CameraEndpointEvaluation result = EmptyCameraEvaluation();
    result.Raster = path.Raster;
    result.Flags = path.EndpointFlags;
    result.PositionPdf = path.PositionPdf;
    result.DirectionPdf = path.DirectionPdf;
    return result;
}
uint SegmentInterior(PathVertex origin, PathVertex destination)
{
    // Scene validation guarantees separated, outward-facing closed SSS solids.
    if (origin.Kind == VolumeNode) return origin.Material;
    if (destination.Kind == VolumeNode) return destination.Material;
    float3 segment = destination.Position - origin.Position;
    if (origin.Kind == SurfaceNode && Materials[origin.Material].Optical.w > 0 &&
        dot(origin.GeometricNormal, segment) < 0)
        return origin.Material;
    if (destination.Kind == SurfaceNode && Materials[destination.Material].Optical.w > 0 &&
        dot(destination.GeometricNormal, -segment) < 0)
        return destination.Material;
    return InvalidPrimitiveIndex;
}
float SegmentTransmittance(PathVertex origin, PathVertex destination)
{
    uint interior = SegmentInterior(origin, destination);
    return interior == InvalidPrimitiveIndex ? 1 :
        exp(-length(destination.Position - origin.Position) / Materials[interior].Optical.w);
}
PathDensity SurfaceMeasure(float probability, uint event, PathVertex origin, PathVertex destination, bool infinite)
{
    PathDensity density = MakePathDensity(probability, (event & ScatteringDelta) != 0 ? PathDirac : PathSolidAngle);
    uint target = infinite ? PathInfiniteDestination :
        (destination.Kind == VolumeNode ? PathVolumeDestination : PathSurfaceDestination);
    density = ConvertPathDensity(density, destination.Position - origin.Position,
                                 destination.GeometricNormal, target);
    if (!PathPdfExists(density) || infinite) return density;
    uint interior = SegmentInterior(origin, destination);
    if (interior == InvalidPrimitiveIndex)
    {
        if (target == PathVolumeDestination) return EmptyPathDensity();
        return density;
    }
    float meanFreePath = Materials[interior].Optical.w;
    density.Log2Value -= length(destination.Position - origin.Position) /
                         meanFreePath * log2(exp(1.0f));
    if (target == PathVolumeDestination) density.Log2Value -= log2(meanFreePath);
    if (!isfinite(density.Log2Value)) return EmptyPathDensity();
    return density;
}
// Reconstruct every density on this completed light-to-camera path. The two
// connection neighbors have new directions; their discarded continuation event
// must not exclude a valid strategy. Reversing a camera prefix swaps its PDFs.
float ConnectionWeight(Subpath light, Subpath camera, uint s, uint t, PathVertex sensor,
                       CameraEndpointEvaluation cameraMeasure, uint emitter, uint flags, float3 emissionDirection)
{
    uint count = s + t;
    if (count == 2) return s == 0 ? 1.0f : 0.0f;
    bool infinite = (flags & EmissionInfinite) != 0;
    BidirectionalVertexPdf densities[BdptMaximumVertices];
    BidirectionalStrategy cuts[BdptMaximumStrategies];
    for (uint clear = 0; clear < BdptMaximumVertices; ++clear) densities[clear] = (BidirectionalVertexPdf)0;
    for (uint clearCut = 0; clearCut < BdptMaximumStrategies; ++clearCut) cuts[clearCut] = (BidirectionalStrategy)0;
    for (uint i = 1; i + 1 < count; ++i)
    {
        PathVertex previous = CompleteVertex(light, camera, s, t, i - 1, sensor);
        PathVertex current = CompleteVertex(light, camera, s, t, i, sensor);
        PathVertex next = CompleteVertex(light, camera, s, t, i + 1, sensor);
        float forward, reverse;
        uint event = 0;
        if ((s > 0 && i == s - 1) || i == s)
        {
            float3 incoming = i == 1 && infinite ? -emissionDirection : normalize(previous.Position - current.Position);
            SurfaceEvaluation scattering = EvaluateVertex(current, incoming, normalize(next.Position - current.Position), TransportImportance);
            forward = scattering.Pdf;
            reverse = scattering.ReversePdf;
        }
        else
        {
            event = current.Event;
            forward = i < s ? current.ForwardPdf : current.ReversePdf;
            reverse = i < s ? current.ReversePdf : current.ForwardPdf;
        }
        densities[i].Flags = (event & ScatteringDelta) != 0 ? PathDeltaScattering : 0;
        densities[i + 1].Forward = SurfaceMeasure(forward, event, current, next, false);
        densities[i - 1].Reverse = SurfaceMeasure(reverse, event, current, previous, infinite && i == 1);
    }
    PathVertex root = CompleteVertex(light, camera, s, t, 0, sensor);
    PathVertex first = CompleteVertex(light, camera, s, t, 1, sensor);
    float3 launch = infinite ? emissionDirection : normalize(first.Position - root.Position);
    EmitterEndpointDensity source = EmitterEndpointPdf(Emitters, EmitterCount, SourceDomain(), emitter, root.GeometricNormal, launch);
    bool deltaPosition = (source.Flags & EmissionDeltaPosition) != 0;
    bool deltaDirection = (source.Flags & EmissionDeltaDirection) != 0;
    PathDensity position = MakePathDensity(deltaPosition ? 1 : source.Position, deltaPosition ? PathDirac : PathArea);
    PathDensity direction = MakePathDensity(deltaDirection ? 1 : source.Direction, deltaDirection ? PathDirac : PathSolidAngle);
    if (infinite) densities[1].Forward = ProjectedDiskDensity(position, first.GeometricNormal, launch);
    else densities[1].Forward = SurfaceMeasure(deltaDirection ? 1 : source.Direction,
        deltaDirection ? ScatteringDelta : 0, root, first, false);
    densities[0].Flags = deltaPosition || deltaDirection ? PathEndpointNotIntersectable : 0;
    PathVertex last = CompleteVertex(light, camera, s, t, count - 2, sensor);
    densities[count - 2].Reverse = SurfaceMeasure(cameraMeasure.DirectionPdf, 0, sensor, last, false);
    bool pinhole = (cameraMeasure.Flags & CameraDeltaPosition) != 0;
    densities[count - 1].Flags = pinhole ? PathEndpointNotIntersectable : 0;
    for (uint cut = 0; cut < count; ++cut)
    {
        cuts[cut].Enabled = cut <= light.Limit && count - cut <= camera.Limit;
        if (cut == 0) cuts[cut].LightEndpoint = MakePathDensity(1, PathDirac);
        else if (infinite) cuts[cut].LightEndpoint = direction;
        else cuts[cut].LightEndpoint = position;
        cuts[cut].CameraEndpoint = MakePathDensity(pinhole ? 1 : cameraMeasure.PositionPdf, pinhole ? PathDirac : PathArea);
        cuts[cut].LightSelection = cut == 0 ? 1 : source.SelectionProbability;
        cuts[cut].SampleCount = 1;
    }
    BidirectionalWeights weights = BidirectionalMisWeightsLocal(densities, count, cuts, (TransportFlags >> MisShift) & MisMask);
    return weights.Values[s];
}
float3 CameraEmission(Subpath light, Subpath camera, uint t)
{
    PathVertex terminal = PathVertices[camera.First + t - 1];
    if (terminal.Emitter >= EmitterCount) return 0;
    float3 value = terminal.Throughput * SourceRadiance(terminal.Emitter, terminal.Incoming);
    if (!all(isfinite(value))) { ReportArithmeticError(); return 0; }
    if (!any(value > 0)) return 0;
    EmitterEndpointDensity endpoint = EmitterEndpointPdf(Emitters, EmitterCount, SourceDomain(),
        terminal.Emitter, terminal.GeometricNormal, terminal.Incoming);
    return value * ConnectionWeight(light, camera, 0, t, PathVertices[camera.First], CameraMeasure(camera),
                                     terminal.Emitter, endpoint.Flags, terminal.Incoming);
}
float3 DirectConnection(Subpath light, Subpath camera, uint t, inout PathCounters rays)
{
    PathVertex from = PathVertices[camera.First + t - 1], source = PathVertices[light.First];
    if (from.Kind != SurfaceNode && from.Kind != VolumeNode) return 0;
    bool infinite = (light.EndpointFlags & EmissionInfinite) != 0;
    bool deltaPosition = (light.EndpointFlags & EmissionDeltaPosition) != 0;
    float3 outgoing = -light.EmissionDirection;
    float geometry = 1;
    if (!infinite)
    {
        float3 difference = source.Position - from.Position;
        float squaredDistance = dot(difference, difference);
        if (!(squaredDistance > 0) || !isfinite(squaredDistance)) return 0;
        outgoing = difference * rsqrt(squaredDistance);
        geometry = (deltaPosition ? 1 : abs(dot(source.GeometricNormal, outgoing))) / squaredDistance;
    }
    SurfaceEvaluation bsdf = EvaluateVertex(from, from.Incoming, outgoing, TransportRadiance);
    float probability = light.SourceProbability * (infinite ? light.DirectionPdf : light.PositionPdf);
    float cosine = from.Kind == VolumeNode ? 1 : abs(dot(from.ShadingNormal, outgoing));
    float3 value = from.Throughput * bsdf.Value * SourceRadiance(source.Emitter, -outgoing) *
                   (geometry * cosine / probability *
                    (infinite ? 1 : SegmentTransmittance(from, source)));
    if (!all(isfinite(value))) { ReportArithmeticError(); return 0; }
    if (!any(value > 0)) return 0;
    if (infinite)
    {
        if (Shadowed(OffsetVertex(from, outgoing), outgoing, MaximumRayDistance, rays)) return 0;
    }
    else if (!VisibleSegment(OffsetVertex(from, outgoing),
                             deltaPosition ? source.Position : OffsetVertex(source, -outgoing), rays)) return 0;
    return value * ConnectionWeight(light, camera, 1, t, PathVertices[camera.First], CameraMeasure(camera),
                                     source.Emitter, light.EndpointFlags, light.EmissionDirection);
}
float3 SurfaceConnection(Subpath light, Subpath camera, uint s, uint t, float2 lens,
                         out float2 raster, inout PathCounters rays)
{
    PathVertex from = PathVertices[light.First + s - 1], to = PathVertices[camera.First + t - 1];
    raster = camera.Raster;
    if ((from.Kind != SurfaceNode && from.Kind != VolumeNode) ||
        (t > 1 && to.Kind != SurfaceNode && to.Kind != VolumeNode)) return 0;
    PathVertex sensor = PathVertices[camera.First];
    CameraEndpointEvaluation sensorMeasure = CameraMeasure(camera);
    float3 direction, target, value;
    if (t == 1)
    {
        CameraConnectionSample connection = SampleCameraConnection(SensorFrame(), from.Position, lens);
        if ((connection.Flags & CameraEndpointValid) == 0) return 0;
        direction = connection.Direction;
        target = connection.Position;
        SurfaceEvaluation bsdf = EvaluateVertex(from, from.Incoming, direction, TransportImportance);
        sensor.Position = target;
        float cosine = from.Kind == VolumeNode ? 1 : abs(dot(from.ShadingNormal, direction));
        value = from.Throughput * bsdf.Value *
                (cosine * connection.Weight * SegmentTransmittance(from, sensor));
        sensorMeasure = EvaluateCameraEndpoint(SensorFrame(), connection.LensPoint, -direction);
        raster = connection.Raster;
    }
    else
    {
        float3 difference = to.Position - from.Position;
        float squaredDistance = dot(difference, difference);
        if (!(squaredDistance > 0) || !isfinite(squaredDistance)) return 0;
        direction = difference * rsqrt(squaredDistance);
        SurfaceEvaluation a = EvaluateVertex(from, from.Incoming, direction, TransportImportance);
        SurfaceEvaluation b = EvaluateVertex(to, to.Incoming, -direction, TransportRadiance);
        float cosineA = from.Kind == VolumeNode ? 1 : abs(dot(from.ShadingNormal, direction));
        float cosineB = to.Kind == VolumeNode ? 1 : abs(dot(to.ShadingNormal, direction));
        float geometry = cosineA * cosineB / squaredDistance;
        value = from.Throughput * a.Value * to.Throughput * b.Value *
                (geometry * SegmentTransmittance(from, to));
        target = OffsetVertex(to, -direction);
    }
    if (!all(isfinite(value))) { ReportArithmeticError(); return 0; }
    if (!any(value > 0) || !VisibleSegment(OffsetVertex(from, direction), target, rays)) return 0;
    uint emitter = PathVertices[light.First].Emitter;
    return value * ConnectionWeight(light, camera, s, t, sensor, sensorMeasure, emitter,
                                     light.EndpointFlags, light.EmissionDirection);
}
bool TilePixel(uint2 global, out uint index)
{
    index = 0;
    uint2 offset = uint2(TileX, TileY);
    if (any(global < offset) || any(global - offset >= uint2(Width, Height))) return false;
    index = (global.y - TileY) * Width + global.x - TileX;
    return true;
}
void AddRadiance(uint pixel, float3 value)
{
    for (uint channel = 0; channel < ColorChannels; ++channel)
        if (!SplatAccumulate(PassRadiance, pixel * PassPixelBytes + channel * PassChannelBytes, value[channel]))
            ReportArithmeticError();
}
void TracePair(uint local)
{
    uint identity = FirstPath + local;
    uint2 global = uint2(identity % FullWidth, identity / FullWidth);
    uint pixel;
    bool owned = TilePixel(global, pixel);
    SampleContext samples = MakeSampleContext(global.x, global.y, FullWidth, PassIndex, SamplesPerPixel, SamplingSequence, SamplingSeed);
    PathCounters rays = (PathCounters)0;
    uint first = local * 2 * BdptMaximumVertices;
    Subpath camera = CameraSubpath(first + BdptMaximumVertices, samples, owned, rays);
    Subpath light = LightSubpath(first, samples, rays);
    float3 cameraValue = 0;
    for (uint t = 2; t <= camera.Count; ++t) cameraValue += CameraEmission(light, camera, t);
    if (light.Count != 0 && camera.Count != 0)
    {
        for (uint direct = 2; direct <= camera.Count && direct < camera.Limit; ++direct)
            cameraValue += DirectConnection(light, camera, direct, rays);
        [loop]
        for (uint s = 2; s <= light.Count; ++s)
        {
            float2 lens = Sample2D(samples, ConnectLensDimension ^ (s * BounceSeedMultiplier));
            [loop]
            for (uint t = 1; t <= camera.Count && s + t <= camera.Limit; ++t)
            {
                float2 raster;
                float3 value = SurfaceConnection(light, camera, s, t, lens, raster, rays);
                if (t > 1) cameraValue += value;
                else if (any(value > 0) && CameraInRaster(raster))
                {
                    uint target;
                    if (TilePixel(uint2(raster * float2(FullWidth, FullHeight)), target)) AddRadiance(target, value);
                }
            }
        }
    }
    if (owned) AddRadiance(pixel, cameraValue);
    AddStatistic(PrimaryCounterWord, rays.Primary);
    AddStatistic(ShadowCounterWord, rays.Shadow);
    AddStatistic(ReflectionCounterWord, rays.Reflection);
}
uint DisplayPixel(float3 value)
{
    static const float LinearThreshold = 0.0031308f, LinearSlope = 12.92f;
    static const float NonlinearScale = 1.055f, NonlinearOffset = 0.055f, Exponent = 2.4f;
    static const uint ByteMaximum = 255, GreenShift = 8, BlueShift = 16, AlphaShift = 24;
    float3 color = max(value, 0);
    color /= max(1, max(color.x, max(color.y, color.z)));
    if (GammaCorrection != 0)
        for (uint channel = 0; channel < 3; ++channel)
            color[channel] = color[channel] <= LinearThreshold ? LinearSlope * color[channel]
                : NonlinearScale * pow(color[channel], 1 / Exponent) - NonlinearOffset;
    uint3 bytes = uint3(saturate(color) * ByteMaximum + 0.5f);
    return bytes.x | (bytes.y << GreenShift) | (bytes.z << BlueShift) | (ByteMaximum << AlphaShift);
}
[shader("raygeneration")]
void RayGen()
{
    uint2 dispatch = DispatchRaysIndex().xy;
    if (Stage == ConnectStage) { TracePair(dispatch.x); return; }
    uint pixel = dispatch.y * Width + dispatch.x;
    if (Stage == ClearStage)
    {
        for (uint word = 0; word < ColorChannels * SplatSumWordCount; ++word)
            PassRadiance.Store(pixel * PassPixelBytes + word * WordBytes, 0);
        if (PassIndex == 0) LinearFilm[pixel] = (LinearFilmPixel)0;
        if (pixel == 0) PathErrors.Store(0, 0);
        return;
    }
    if (Stage != CommitStage) return;
    float3 observation;
    for (uint channel = 0; channel < ColorChannels; ++channel)
        observation[channel] = SplatRound(PassRadiance, pixel * PassPixelBytes + channel * PassChannelBytes);
    if (!all(isfinite(observation))) { ReportArithmeticError(); return; }
    LinearFilmPixel film = LinearFilm[pixel];
    AccumulateFilmSample(film, observation);
    if (!all(isfinite(film.MeanRadiance)) || !isfinite(film.LuminanceM2)) { ReportArithmeticError(); return; }
    LinearFilm[pixel] = film;
    if (film.SampleCount == SamplesPerPixel)
    {
        OutputPixels[pixel] = DisplayPixel(film.MeanRadiance);
        AddStatistic(CompletedCounterWord, uint2(1, 0));
    }
}
