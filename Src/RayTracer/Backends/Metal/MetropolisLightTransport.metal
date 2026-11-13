// Independent Metal primary-sample-space MLT. Kelemen et al. (2002), "A Simple
// and Robust Mutation Strategy for the Metropolis Light Transport Algorithm".
// A fixed camera map, bootstrap distribution and persistent accept/reject chains
// own this estimator; no other integrator is included or invoked.
// https://pbr-book.org/3ed-2018/Light_Transport_III_Bidirectional_Methods/Metropolis_Light_Transport
// Small proposals use quantized, signed Laplace displacements on an integer
// torus. This symmetric proposal has the configured unwrapped standard deviation.
// Metal has no binary64 arithmetic: scaled two-float numbers retain bootstrap
// range and precision, while the film uses float CAS splats and complete passes.
// MSL 2.3 permits the ulong PCG state in device buffers (MSL specification, 2.8).
#include <metal_stdlib>
using namespace metal;
constant float Pi = 3.14159265358979323846f;
constant float RayEpsilon = 1.0e-4f, MaximumRayDistance = 1.0e30f, BasisParallelThreshold = 0.999f;
constant uint ImportanceBit = 1, NeeBit = 2, RouletteBit = 4, MisShift = 3, MisMask = 3;
constant uint MarkovMaximumReflections = 32, CameraDimensions = 4, VertexDimensions = 12;
constant uint MarkovDimensions = CameraDimensions + (MarkovMaximumReflections + 2) * VertexDimensions;
constant uint PrimaryBits = 24, RandomBits = 32, PrimaryRange = 1u << PrimaryBits,
              PrimaryMask = PrimaryRange - 1;
constant float PrimaryQuantum = 1.0f / PrimaryRange, LaplaceDeviationScale = 0.7071067811865475244f;
constant uint BootstrapDomain = 0x4d544c42u, ChainDomain = 0x4d544c43u;
constant uint NumericalFailure = 1, PassComponents = 4;
constant uint SourceDimension = 0, EmitterPositionDimension = 1, EmitterDirectionDimension = 3;
constant uint ScatterDirectionDimension = 5, ScatterEventDimension = 7, SurvivalDimension = 8;
constant uint InteriorFreeFlight = 9, InteriorPhaseDirection = 10;
struct RandomStream
{
    ulong State, Stream;
};
struct PrimaryVector
{
    uint Values[MarkovDimensions];
};
uint RandomWord(thread RandomStream& random)
{
    // PCG XSH-RR 64/32, independently expressed from O'Neill's permutation:
    // https://www.pcg-random.org/paper.html . The stream increment is always odd.
    const ulong StateMultiplier = 6364136223846793005UL;
    const uint FoldShift = 18, OutputShift = 27, RotationShift = 59, RotationMask = RandomBits - 1;
    ulong before = random.State;
    random.State = before * StateMultiplier + random.Stream;
    uint folded = uint((before ^ (before >> FoldShift)) >> OutputShift);
    uint rotation = uint(before >> RotationShift);
    return (folded >> rotation) | (folded << ((RandomBits - rotation) & RotationMask));
}
RandomStream SeedStream(ulong seed, ulong sequence)
{
    RandomStream random{0, (sequence << 1) | 1};
    RandomWord(random);
    random.State += seed;
    RandomWord(random);
    return random;
}
RandomStream MarkovStream(uint seed, uint index, uint domain)
{
    return SeedStream(seed, (ulong(domain) << RandomBits) | index);
}
float Uniform(thread RandomStream& random)
{
    return float(RandomWord(random) >> (RandomBits - PrimaryBits)) * PrimaryQuantum;
}
PrimaryVector UniformPrimary(thread RandomStream& random)
{
    PrimaryVector primary;
    for (uint dimension = 0; dimension < MarkovDimensions; ++dimension)
        primary.Values[dimension] = RandomWord(random) >> (RandomBits - PrimaryBits);
    return primary;
}
PrimaryVector MutatedPrimary(thread const PrimaryVector& current, thread RandomStream& random, bool large,
                             float scale)
{
    if (large)
        return UniformPrimary(random);
    PrimaryVector proposal;
    for (uint dimension = 0; dimension < MarkovDimensions; ++dimension)
    {
        float magnitude = -log(1 - Uniform(random)) * (scale * LaplaceDeviationScale);
        uint displacement = uint(fract(magnitude) * PrimaryRange);
        if (RandomWord(random) & 1u)
            displacement = 0u - displacement;
        proposal.Values[dimension] = (current.Values[dimension] + displacement) & PrimaryMask;
    }
    return proposal;
}
float Coordinate(thread const PrimaryVector& primary, uint dimension)
{
    return float(primary.Values[dimension]) * PrimaryQuantum;
}
float2 Coordinates(thread const PrimaryVector& primary, uint dimension)
{
    return float2(Coordinate(primary, dimension), Coordinate(primary, dimension + 1));
}
#include "SurfaceTypes.metal"
#include "SubsurfaceScattering.metal"
#include "ShadingTypes.metal"
#include "ExtendedFloat.metal"
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
struct MarkovControls
{
    packed_float3 Center;
    float Radius;
    uint Sources, BootstrapCount, ChainCount, Seed;
    uint First, Count;
    float LargeStepProbability, MutationScale;
};
struct PathValue
{
    float2 Raster;
    packed_float3 Color;
    uint Reserved;
    WideMath::Number Importance;
};
struct MarkovChain
{
    RandomStream Generator;
    PathValue Current;
    uint2 Proposals, Accepted, LargeSteps, PositiveRejected;
    PrimaryVector Coordinates;
};
struct BootstrapEntry
{
    PathValue Value;
    WideMath::Number Cumulative;
};
struct Normalization
{
    WideMath::Number Total, Mean;
};
struct ThreadStatistics
{
    uint2 Primary, Shadow, Reflection, Tests;
};
static_assert(sizeof(FrameConstants) == 240 && sizeof(MarkovControls) == 48 && sizeof(PathValue) == 40 &&
                  sizeof(MarkovChain) == 1736 && sizeof(BootstrapEntry) == 56 && sizeof(Normalization) == 32,
              "Metal MLT upload layout mismatch");
void IncrementCounter(thread uint2& value)
{
    if (++value.x == 0)
        ++value.y;
}
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
#include "LightSelection.metal"
#include "LightEmission.metal"
#include "SurfaceEmission.metal"
#include "EmitterEndpoint.metal"
#include "CameraEndpoint.metal"
struct CameraMap
{
    SceneGeometryBuffers Geometry;
    device const PackedMaterial* Materials;
    device const PackedTextureDescriptor* Textures;
    device const uint* Texels;
    device const PackedEmitter* Sources;
    constant FrameConstants* Frame;
    constant MarkovControls* Controls;
    device atomic_uint* Failure;
};
void Fail(device atomic_uint* status)
{
    atomic_store_explicit(status, NumericalFailure, memory_order_relaxed);
}
bool Nonzero(float3 color)
{
    return color.x > 0 || color.y > 0 || color.z > 0;
}
bool Nonnegative(float3 color)
{
    return color.x >= 0 && color.y >= 0 && color.z >= 0;
}
EmissionDomain WorldBounds(CameraMap map)
{
    return {float3(map.Controls->Center), map.Controls->Radius};
}
float3 OffsetPoint(float3 point, float3 normal, float3 direction)
{
    return point + normal * (dot(normal, direction) >= 0 ? RayEpsilon : -RayEpsilon);
}
float LinearChannel(uint channel)
{
    const float ByteMaximum = 255, Threshold = 0.04045f, Slope = 12.92f;
    const float Offset = 0.055f, Scale = 1.055f, Exponent = 2.4f;
    float value = float(channel) / ByteMaximum;
    return value <= Threshold ? value / Slope : pow((value + Offset) / Scale, Exponent);
}
float3 TextureColor(CameraMap map, PackedTextureDescriptor image, int x, int y)
{
    x = (x % int(image.Width) + int(image.Width)) % int(image.Width);
    y = (y % int(image.Height) + int(image.Height)) % int(image.Height);
    uint texel = map.Texels[image.OffsetTexels + uint(y) * image.Width + uint(x)];
    const uint ByteMask = 0xff, GreenShift = 8, BlueShift = 16;
    return float3(LinearChannel(texel & ByteMask), LinearChannel((texel >> GreenShift) & ByteMask),
                  LinearChannel((texel >> BlueShift) & ByteMask));
}
float3 SurfaceColor(CameraMap map, HitData hit)
{
    uint texture = map.Materials[hit.MaterialIndex].DiffuseTextureIndex;
    if (!hit.HasTexCoord || texture >= map.Frame->TextureCount || !isfinite(hit.TexCoord.x) ||
        !isfinite(hit.TexCoord.y))
        return hit.Color;
    PackedTextureDescriptor image = map.Textures[texture];
    if (!image.Width || !image.Height || image.OffsetTexels > map.Frame->TextureTexelCount ||
        image.Height > (map.Frame->TextureTexelCount - image.OffsetTexels) / image.Width)
        return hit.Color;
    float x = fract(hit.TexCoord.x) * image.Width - 0.5f, y = fract(1 - hit.TexCoord.y) * image.Height - 0.5f;
    int column = int(floor(x)), row = int(floor(y));
    float3 upper =
        mix(TextureColor(map, image, column, row), TextureColor(map, image, column + 1, row), fract(x));
    float3 lower = mix(TextureColor(map, image, column, row + 1),
                       TextureColor(map, image, column + 1, row + 1), fract(x));
    return hit.Color * mix(upper, lower, fract(y));
}
bool RectangleHit(PackedEmitter source, RayData ray, thread float& closest)
{
    if (source.Kind != PackedEmitterRectangle)
        return false;
    float3 normal = float3(source.Direction);
    float cosine = dot(normal, ray.Direction);
    if (cosine == 0)
        return false;
    float distance = dot(normal, float3(source.Origin) - ray.Origin) / cosine;
    if (!(distance > ray.MinimumDistance && distance < ray.MaximumDistance && distance <= closest))
        return false;
    float3 helper = abs(normal.y) < BasisParallelThreshold ? float3(0, 1, 0) : float3(1, 0, 0);
    float3 tangent = normalize(cross(helper, normal)), bitangent = cross(normal, tangent);
    float3 relative = ray.Origin + ray.Direction * distance - float3(source.Origin);
    if (abs(dot(relative, tangent)) > source.Width / 2 || abs(dot(relative, bitangent)) > source.Height / 2)
        return false;
    closest = distance;
    return true;
}
uint SurfaceSource(CameraMap map, uint primitive, bool infinite)
{
    for (uint index = 0; index < map.Controls->Sources; ++index)
        if (infinite ? map.Sources[index].Kind == PackedEmitterEnvironment
                     : primitive != InvalidPrimitiveIndex && map.Sources[index].Primitive == primitive)
            return index;
    return InvalidPrimitiveIndex;
}
float3 SourceRadiance(PackedEmitter source, float3 emittedDirection)
{
    if (source.Kind == PackedEmitterEnvironment)
        return mix(float3(source.BackgroundBottom), float3(source.BackgroundTop),
                   (1 - emittedDirection.y) / 2);
    if (source.Kind == PackedEmitterRectangle)
        return dot(float3(source.Direction), emittedDirection) > 0
                   ? float3(source.Strength) / (source.Width * source.Height)
                   : float3(0);
    return float3(source.Strength);
}
float StrategyWeight(uint flags, float selected, float other)
{
    const uint NoMis = 0, BalanceMis = 1;
    uint strategy = (flags >> MisShift) & MisMask;
    if (strategy == NoMis)
        return 1;
    if (!(selected > 0))
        return 0;
    if (!(other > 0))
        return 1;
    float peak = max(selected, other), first = selected / peak, second = other / peak;
    if (strategy != BalanceMis)
    {
        first *= first;
        second *= second;
    }
    return first / (first + second);
}
bool Unoccluded(CameraMap map, RayData ray, thread ThreadStatistics& statistics)
{
    IncrementCounter(statistics.Shadow);
    HitData blocker{};
    blocker.Distance = ray.MaximumDistance;
    if (IntersectScene(map.Geometry, ray, true, blocker, statistics.Tests))
        return false;
    float closest = ray.MaximumDistance;
    for (uint source = 0; source < map.Controls->Sources; ++source)
        if (RectangleHit(map.Sources[source], ray, closest))
            return false;
    return true;
}
float3 DirectContribution(CameraMap map, thread const PrimaryVector& primary, uint dimensions,
                          SurfaceBsdf surface, float3 point, float continuous, float currentInterior,
                          thread ThreadStatistics& statistics)
{
    EmitterEndpoint sampled =
        SampleEmitterEndpoint(map.Sources, map.Controls->Sources, WorldBounds(map),
                              Coordinate(primary, dimensions + SourceDimension),
                              Coordinates(primary, dimensions + EmitterPositionDimension),
                              Coordinates(primary, dimensions + EmitterDirectionDimension));
    EmissionSample light = sampled.Emission;
    if (!(light.Flags & EmissionValid))
        return float3(0);
    bool infinite = (light.Flags & EmissionInfinite) != 0;
    bool pointSource = (light.Flags & EmissionDeltaPosition) != 0;
    bool delta = (light.Flags & (EmissionDeltaPosition | EmissionDeltaDirection)) != 0;
    if (!delta && !(map.Frame->TransportFlags & NeeBit))
        return float3(0);
    RayData connection{point, -light.Direction, RayEpsilon, MaximumRayDistance};
    float geometry = 1;
    float beamDistance = MaximumRayDistance;
    if (!infinite)
    {
        float3 separation = light.Origin - point;
        float distance = length(separation);
        if (!(distance > 0) || !isfinite(distance))
            return float3(0);
        beamDistance = distance;
        connection.Direction = separation / distance;
        geometry = (pointSource ? 1 : abs(dot(light.Normal, connection.Direction))) / distance / distance;
    }
    float proposal = sampled.SelectionProbability * (infinite ? light.DirectionPdf : light.PositionPdf);
    if (!(proposal > 0))
        return float3(0);
    SurfaceEvaluation scattering = EvaluateSurface(surface, connection.Direction);
    float angular = infinite || delta ? proposal : geometry > 0 ? proposal / geometry : 0;
    float weight =
        delta ? 1 : StrategyWeight(map.Frame->TransportFlags, angular, scattering.Pdf * continuous);
    float3 value =
        scattering.Value * SourceRadiance(map.Sources[sampled.EmitterIndex], -connection.Direction) *
        (abs(dot(surface.Normal, connection.Direction)) * continuous * geometry * weight / proposal);
    float connectionMeanFreePath = surface.Material.Optical.w > 0
        ? (dot(surface.GeometricNormal, connection.Direction) < 0 ? surface.Material.Optical.w : 0)
        : currentInterior;
    if (connectionMeanFreePath > 0) value *= exp(float3(-beamDistance / connectionMeanFreePath));
    if (!all(isfinite(value)) || !Nonzero(value))
        return value;
    connection.Origin = SurfaceOffset(surface, point, connection.Direction);
    if (!infinite)
    {
        float3 target =
            pointSource ? light.Origin : OffsetPoint(light.Origin, light.Normal, -connection.Direction);
        float3 displacement = target - connection.Origin;
        float distance = length(displacement);
        if (!(distance > 2 * RayEpsilon))
            return float3(0);
        connection.Direction = displacement / distance;
        connection.MaximumDistance = distance - RayEpsilon;
    }
    return Unoccluded(map, connection, statistics) ? value : float3(0);
}
float3 InteriorContribution(CameraMap map, thread const PrimaryVector& primary, uint dimensions,
                            float3 point, float3 incoming, PackedMaterial interior,
                            thread ThreadStatistics& statistics)
{
    EmitterEndpoint sampled = SampleEmitterEndpoint(map.Sources, map.Controls->Sources, WorldBounds(map),
        Coordinate(primary, dimensions + SourceDimension),
        Coordinates(primary, dimensions + EmitterPositionDimension),
        Coordinates(primary, dimensions + EmitterDirectionDimension));
    EmissionSample light = sampled.Emission;
    if (!(light.Flags & EmissionValid)) return float3(0);
    bool infinite = (light.Flags & EmissionInfinite) != 0;
    bool pointSource = (light.Flags & EmissionDeltaPosition) != 0;
    bool delta = (light.Flags & (EmissionDeltaPosition | EmissionDeltaDirection)) != 0;
    if (!delta && !(map.Frame->TransportFlags & NeeBit)) return float3(0);
    RayData connection{point, -light.Direction, RayEpsilon, MaximumRayDistance};
    float distance = MaximumRayDistance, geometry = 1;
    if (!infinite)
    {
        float3 separation = light.Origin - point;
        float squared = dot(separation, separation);
        if (!(squared > 0) || !isfinite(squared)) return float3(0);
        distance = sqrt(squared);
        connection.Direction = separation / distance;
        geometry = (pointSource ? 1 : abs(dot(light.Normal, connection.Direction))) / squared;
    }
    float proposal = sampled.SelectionProbability * (infinite ? light.DirectionPdf : light.PositionPdf);
    if (!(proposal > 0) || !(geometry > 0)) return float3(0);
    float angular = delta ? proposal : infinite ? proposal : proposal / geometry;
    float phase = SubsurfacePhase(dot(incoming, connection.Direction), interior.ConductorIndex.w);
    float phasePdf = (map.Frame->TransportFlags & ImportanceBit) != 0 ? phase : 1 / (4 * Pi);
    float weight = delta ? 1 : StrategyWeight(map.Frame->TransportFlags, angular, phasePdf);
    float3 value = SourceRadiance(map.Sources[sampled.EmitterIndex], -connection.Direction)
        * (phase * geometry * weight / proposal) * exp(float3(-distance / interior.Optical.w));
    if (!all(isfinite(value)) || !Nonzero(value)) return value;
    if (!infinite)
    {
        float3 target = pointSource ? light.Origin : OffsetPoint(light.Origin, light.Normal, -connection.Direction);
        float3 separation = target - point;
        distance = length(separation);
        if (!(distance > RayEpsilon)) return float3(0);
        connection.Direction = separation / distance;
        connection.MaximumDistance = distance - RayEpsilon;
    }
    return Unoccluded(map, connection, statistics) ? value : float3(0);
}
float EmissionWeight(CameraMap map, uint source, float3 normal, float3 direction, float3 separation,
                     bool infinite, bool delta, float continuationPdf)
{
    uint flags = map.Frame->TransportFlags;
    if (delta || !(flags & NeeBit))
        return 1;
    if (((flags >> MisShift) & MisMask) == 0)
        return 0;
    auto pdf =
        EmitterEndpointPdf(map.Sources, map.Controls->Sources, WorldBounds(map), source, normal, direction);
    float competing = pdf.SelectionProbability * pdf.Direction;
    if (!infinite)
    {
        float cosine = abs(dot(normal, direction));
        competing =
            cosine > 0 ? pdf.SelectionProbability * pdf.Position * dot(separation, separation) / cosine : 0;
    }
    return StrategyWeight(flags, continuationPdf, competing);
}
PathValue CameraPathValue(CameraMap map, thread const PrimaryVector& primary,
                          thread ThreadStatistics& statistics)
{
    constant FrameConstants& frame = *map.Frame;
    CameraEndpointFrame camera{float3(frame.CameraPosition),
                               float3(frame.CameraForward),
                               float3(frame.CameraRight),
                               float3(frame.CameraUp),
                               frame.HalfWidth,
                               frame.HalfHeight,
                               frame.LensRadius,
                               frame.FocusDistance};
    CameraEndpointSample sensor =
        SampleCameraEndpoint(camera, Coordinates(primary, 0), Coordinates(primary, 2));
    PathValue path{};
    path.Raster = sensor.Raster;
    if (!(sensor.Flags & CameraEndpointValid))
    {
        Fail(map.Failure);
        return path;
    }
    RayData ray{sensor.Origin, sensor.Direction, RayEpsilon, MaximumRayDistance};
    float3 radiance = float3(0), throughput = float3(1), previousPosition = sensor.Origin;
    float previousDensity = 0, etaCompensation = 1;
    bool previousDelta = true;
    uint interiorMaterial = InvalidPrimitiveIndex;
    // The final endpoint may emit, but cannot scatter again. Only scattering
    // vertices consume their fixed dimension block, including terminated paths.
    for (uint depth = 0; depth <= frame.MaxReflectionBounces + 1; ++depth)
    {
        uint dimensions = CameraDimensions + depth * VertexDimensions;
        if (depth == 0)
            IncrementCounter(statistics.Primary);
        else
            IncrementCounter(statistics.Reflection);
        HitData hit{};
        hit.Distance = ray.MaximumDistance;
        bool found = IntersectScene(map.Geometry, ray, false, hit, statistics.Tests);
        float distance = found ? hit.Distance + RayEpsilon : ray.MaximumDistance;
        uint rectangle = InvalidPrimitiveIndex;
        for (uint source = 0; source < map.Controls->Sources; ++source)
            if (RectangleHit(map.Sources[source], ray, distance))
                rectangle = source;
        if (found && hit.MaterialIndex >= frame.MaterialCount)
        {
            Fail(map.Failure);
            return path;
        }
        if (depth == 0 && found && !hit.FrontFace && map.Materials[hit.MaterialIndex].Optical.w > 0)
            interiorMaterial = hit.MaterialIndex;
        if (interiorMaterial != InvalidPrimitiveIndex)
        {
            PackedMaterial interior = map.Materials[interiorMaterial];
            float endpoint = rectangle != InvalidPrimitiveIndex ? distance
                : found ? hit.Distance : ray.MaximumDistance;
            SubsurfaceFlight flight = SubsurfaceFreeFlight(interior.Optical.w,
                interior.AlbedoReflectivity.xyz * interior.ConductorExtinction.w,
                ray.Direction, endpoint, Coordinate(primary, dimensions + InteriorFreeFlight));
            throughput *= flight.Weight;
            if (!all(isfinite(throughput))) { Fail(map.Failure); return path; }
            if (!Nonzero(throughput)) break;
            if (flight.Scattered)
            {
                if (depth > frame.MaxReflectionBounces) break;
                float3 point = ray.Origin + ray.Direction * flight.Distance;
                radiance += throughput * InteriorContribution(map, primary, dimensions, point,
                    ray.Direction, interior, statistics);
                SubsurfaceDirection phase = SubsurfaceSampleDirection(ray.Direction,
                    interior.ConductorIndex.w, Coordinates(primary, dimensions + InteriorPhaseDirection),
                    (frame.TransportFlags & ImportanceBit) != 0);
                if (!(phase.Pdf > 0)) break;
                throughput *= phase.Weight;
                if (!all(isfinite(throughput))) { Fail(map.Failure); return path; }
                if (!Nonzero(throughput)) break;
                if ((frame.TransportFlags & RouletteBit) && depth + 1 >= frame.RouletteStartDepth)
                {
                    const float SurvivalMinimum = 0.05f, SurvivalMaximum = 0.95f;
                    float survival = clamp(max(throughput.x, max(throughput.y, throughput.z)) * etaCompensation,
                        SurvivalMinimum, SurvivalMaximum);
                    if (Coordinate(primary, dimensions + SurvivalDimension) >= survival) break;
                    throughput /= survival;
                }
                previousPosition = point;
                previousDensity = phase.Pdf;
                previousDelta = false;
                ray.Origin = point;
                ray.Direction = phase.Direction;
                continue;
            }
        }
        bool infinite = !found && rectangle == InvalidPrimitiveIndex;
        uint source =
            rectangle != InvalidPrimitiveIndex ? rectangle : SurfaceSource(map, hit.PrimitiveIndex, infinite);
        float3 point =
            ray.Origin + ray.Direction * (rectangle != InvalidPrimitiveIndex ? distance : hit.Distance);
        if (source != InvalidPrimitiveIndex)
        {
            float3 normal = rectangle != InvalidPrimitiveIndex ? float3(map.Sources[source].Direction)
                                                               : hit.GeometricNormal;
            radiance += throughput * SourceRadiance(map.Sources[source], -ray.Direction) *
                        EmissionWeight(map, source, normal, -ray.Direction, point - previousPosition,
                                       infinite, previousDelta, previousDensity);
        }
        if (infinite || rectangle != InvalidPrimitiveIndex || depth > frame.MaxReflectionBounces)
            break;
        PackedMaterial material = map.Materials[hit.MaterialIndex];
        float mirror =
            material.Model == MaterialLegacy ? clamp(material.AlbedoReflectivity.w, 0.0f, 1.0f) : 0;
        if (material.Model == MaterialLegacy)
            material.Model = MaterialDiffuse;
        SurfaceBsdf surface =
            MakeSurfaceBsdf(material, hit, -ray.Direction, SurfaceColor(map, hit), frame.TransportFlags);
        radiance +=
            throughput * DirectContribution(map, primary, dimensions, surface, point, 1 - mirror,
                interiorMaterial == InvalidPrimitiveIndex ? 0 : map.Materials[interiorMaterial].Optical.w, statistics);
        float choice = Coordinate(primary, dimensions + ScatterEventDimension);
        SurfaceSample scatter{};
        if (choice < mirror)
        {
            scatter.Direction = normalize(reflect(ray.Direction, surface.Normal));
            float cosine = dot(surface.Normal, scatter.Direction);
            if (!(cosine > 0) || !SurfaceDirectionValid(surface, scatter.Direction, cosine))
                break;
            scatter.Weight = float3(1);
            scatter.Pdf = mirror;
            scatter.RelativeIndex = 1;
            scatter.Event = ScatteringDelta | ScatteringReflection;
        }
        else
        {
            float remapped = mirror > 0 ? (choice - mirror) / (1 - mirror) : choice;
            scatter = SampleSurface(surface, Coordinates(primary, dimensions + ScatterDirectionDimension),
                                    remapped);
            scatter.Pdf *= 1 - mirror;
        }
        if (!scatter.Event || !(scatter.Pdf > 0))
            break;
        throughput *= scatter.Weight;
        etaCompensation *= scatter.RelativeIndex * scatter.RelativeIndex;
        if (!all(isfinite(throughput)) || !isfinite(etaCompensation))
        {
            Fail(map.Failure);
            return path;
        }
        if (!Nonzero(throughput))
            break;
        if ((frame.TransportFlags & RouletteBit) && depth + 1 >= frame.RouletteStartDepth)
        {
            const float SurvivalMinimum = 0.05f, SurvivalMaximum = 0.95f;
            float survival = clamp(max(throughput.x, max(throughput.y, throughput.z)) * etaCompensation,
                                   SurvivalMinimum, SurvivalMaximum);
            if (Coordinate(primary, dimensions + SurvivalDimension) >= survival)
                break;
            throughput /= survival;
        }
        previousPosition = point;
        previousDensity = scatter.Pdf;
        previousDelta = (scatter.Event & ScatteringDelta) != 0;
        ray.Origin = SurfaceOffset(surface, point, scatter.Direction);
        ray.Direction = scatter.Direction;
        if (material.Optical.w > 0 && (scatter.Event & ScatteringTransmission) != 0)
            interiorMaterial = hit.FrontFace ? hit.MaterialIndex : InvalidPrimitiveIndex;
    }
    if (!all(isfinite(radiance)) || !Nonnegative(radiance))
    {
        Fail(map.Failure);
        return path;
    }
    const float RedLuminance = 0.2126f, GreenLuminance = 0.7152f, BlueLuminance = 0.0722f;
    using namespace WideMath;
    path.Importance = Add(Add(Multiply(FromFloat(radiance.x), FromFloat(RedLuminance)),
                              Multiply(FromFloat(radiance.y), FromFloat(GreenLuminance))),
                          Multiply(FromFloat(radiance.z), FromFloat(BlueLuminance)));
    path.Color = packed_float3(radiance);
    return path;
}
void AtomicRadiance(device atomic_uint* component, float value, device atomic_uint* failure)
{
    uint expected = atomic_load_explicit(component, memory_order_relaxed);
    while (true)
    {
        float total = as_type<float>(expected) + value;
        if (!isfinite(total))
        {
            Fail(failure);
            return;
        }
        if (atomic_compare_exchange_weak_explicit(component, &expected, as_type<uint>(total),
                                                  memory_order_relaxed, memory_order_relaxed))
            return;
    }
}
void Splat(constant FrameConstants& frame, PathValue path, WideMath::Number mass, WideMath::Number mean,
           device atomic_uint* pass, device atomic_uint* failure)
{
    if (!(mass.High > 0 && path.Importance.High > 0))
        return;
    uint x = uint(path.Raster.x * frame.FullWidth), y = uint(path.Raster.y * frame.FullHeight);
    if (x < frame.TileX || y < frame.TileY || x - frame.TileX >= frame.Width ||
        y - frame.TileY >= frame.Height)
        return;
    using namespace WideMath;
    Number factor = Divide(Multiply(mass, mean), path.Importance);
    float3 color = float3(as_type<float>(Encode(Multiply(FromFloat(path.Color.x), factor))),
                          as_type<float>(Encode(Multiply(FromFloat(path.Color.y), factor))),
                          as_type<float>(Encode(Multiply(FromFloat(path.Color.z), factor))));
    if (!all(isfinite(color)) || !Nonnegative(color))
    {
        Fail(failure);
        return;
    }
    uint address = ((y - frame.TileY) * frame.Width + x - frame.TileX) * PassComponents;
    AtomicRadiance(pass + address, color.x, failure);
    AtomicRadiance(pass + address + 1, color.y, failure);
    AtomicRadiance(pass + address + 2, color.z, failure);
}
void RunChain(CameraMap map, thread MarkovChain& state, uint index, Normalization distribution,
              device atomic_uint* pass, thread ThreadStatistics& statistics)
{
    if (!(distribution.Mean.High > 0))
        return;
    uint count = map.Controls->ChainCount, pixels = map.Frame->FullWidth * map.Frame->FullHeight;
    uint quota = pixels / count + (index < pixels % count ? 1 : 0);
    using namespace WideMath;
    Number one = FromFloat(1);
    for (uint step = map.Controls->First; step < quota && step - map.Controls->First < map.Controls->Count;
         ++step)
    {
        bool large = Uniform(state.Generator) < map.Controls->LargeStepProbability;
        PrimaryVector candidateCoordinates =
            MutatedPrimary(state.Coordinates, state.Generator, large, map.Controls->MutationScale);
        PathValue candidate = CameraPathValue(map, candidateCoordinates, statistics);
        Number accepted = state.Current.Importance.High > 0
                              ? Minimum(one, Divide(candidate.Importance, state.Current.Importance))
                              : one;
        Splat(*map.Frame, state.Current, Subtract(one, accepted), distribution.Mean, pass, map.Failure);
        Splat(*map.Frame, candidate, accepted, distribution.Mean, pass, map.Failure);
        IncrementCounter(state.Proposals);
        if (large)
            IncrementCounter(state.LargeSteps);
        if (Compare(FromFloat(Uniform(state.Generator)), accepted) < 0)
        {
            state.Current = candidate;
            state.Coordinates = candidateCoordinates;
            IncrementCounter(state.Accepted);
        }
        else if (candidate.Importance.High > 0)
            IncrementCounter(state.PositiveRejected);
    }
}

kernel void ArvoMetropolisBootstrap(
    device const PackedAccelerationNode* nodes [[buffer(0)]], device const uint* references [[buffer(1)]],
    device const PackedTriangle* triangles [[buffer(2)]], device const PackedSphere* spheres [[buffer(3)]],
    device const PackedMaterial* materials [[buffer(4)]], device const PackedEmitter* emitters [[buffer(5)]],
    constant FrameConstants& frame [[buffer(6)]], device ThreadStatistics* statistics [[buffer(9)]],
    device const PackedTextureDescriptor* textures [[buffer(10)]], device const uint* texels [[buffer(11)]],
    device const InstanceRecord* instances [[buffer(13)]],
    device const InstanceNode* instanceNodes [[buffer(14)]],
    device atomic_uint* traversalError [[buffer(15)]], constant MarkovControls& controls [[buffer(18)]],
    device atomic_uint* error [[buffer(19)]], device BootstrapEntry* bootstrap [[buffer(20)]],
    uint local [[thread_position_in_grid]])
{
    if (local >= controls.Count || local + controls.First >= controls.BootstrapCount)
        return;
    SceneGeometryBuffers geometry{
        {nodes, references, triangles, spheres, frame.NodeCount, frame.TriangleCount, frame.Acceleration},
        instances,
        instanceNodes,
        traversalError,
        frame.InstanceNodeCount};
    CameraMap map{geometry, materials, textures, texels, emitters, &frame, &controls, error};
    RandomStream stream = MarkovStream(controls.Seed, controls.First + local, BootstrapDomain);
    PrimaryVector primary = UniformPrimary(stream);
    ThreadStatistics counters{};
    bootstrap[controls.First + local].Value = CameraPathValue(map, primary, counters);
    statistics[local] = counters;
}
kernel void ArvoMetropolisDistribution(constant MarkovControls& controls [[buffer(18)]],
                                       device atomic_uint* error [[buffer(19)]],
                                       device BootstrapEntry* bootstrap [[buffer(20)]],
                                       device Normalization* normalization [[buffer(21)]],
                                       uint index [[thread_position_in_grid]])
{
    if (index != 0)
        return;
    using namespace WideMath;
    Number total{};
    for (uint sample = 0; sample < controls.BootstrapCount; ++sample)
    {
        Number importance = bootstrap[sample].Value.Importance;
        if (!(importance.High >= 0) || !isfinite(importance.High) || !isfinite(importance.Low))
        {
            Fail(error);
            return;
        }
        total = Add(total, importance);
        bootstrap[sample].Cumulative = total;
    }
    if (!controls.BootstrapCount)
    {
        Fail(error);
        return;
    }
    *normalization = {total, Divide(total, FromFloat(float(controls.BootstrapCount)))};
}
kernel void ArvoMetropolisInitialize(device MarkovChain* chains [[buffer(16)]],
                                     constant MarkovControls& controls [[buffer(18)]],
                                     device atomic_uint* error [[buffer(19)]],
                                     device const BootstrapEntry* bootstrap [[buffer(20)]],
                                     device const Normalization* normalization [[buffer(21)]],
                                     uint index [[thread_position_in_grid]])
{
    if (index >= controls.ChainCount)
        return;
    MarkovChain state{};
    state.Generator = MarkovStream(controls.Seed, index, ChainDomain);
    if (normalization->Total.High > 0)
    {
        using namespace WideMath;
        Number target = Multiply(normalization->Total, FromFloat(Uniform(state.Generator)));
        uint first = 0, end = controls.BootstrapCount;
        while (first < end)
        {
            uint midpoint = first + (end - first) / 2;
            if (Compare(bootstrap[midpoint].Cumulative, target) <= 0)
                first = midpoint + 1;
            else
                end = midpoint;
        }
        if (first == controls.BootstrapCount)
        {
            Fail(error);
            return;
        }
        state.Current = bootstrap[first].Value;
        RandomStream source = MarkovStream(controls.Seed, first, BootstrapDomain);
        state.Coordinates = UniformPrimary(source);
    }
    chains[index] = state;
}
kernel void ArvoMetropolisMutate(
    device const PackedAccelerationNode* nodes [[buffer(0)]], device const uint* references [[buffer(1)]],
    device const PackedTriangle* triangles [[buffer(2)]], device const PackedSphere* spheres [[buffer(3)]],
    device const PackedMaterial* materials [[buffer(4)]], device const PackedEmitter* emitters [[buffer(5)]],
    constant FrameConstants& frame [[buffer(6)]], device ThreadStatistics* statistics [[buffer(9)]],
    device const PackedTextureDescriptor* textures [[buffer(10)]], device const uint* texels [[buffer(11)]],
    device const InstanceRecord* instances [[buffer(13)]],
    device const InstanceNode* instanceNodes [[buffer(14)]],
    device atomic_uint* traversalError [[buffer(15)]], device MarkovChain* chains [[buffer(16)]],
    device atomic_uint* pass [[buffer(17)]], constant MarkovControls& controls [[buffer(18)]],
    device atomic_uint* error [[buffer(19)]], device const Normalization* normalization [[buffer(21)]],
    uint index [[thread_position_in_grid]])
{
    if (index >= controls.ChainCount)
        return;
    SceneGeometryBuffers geometry{
        {nodes, references, triangles, spheres, frame.NodeCount, frame.TriangleCount, frame.Acceleration},
        instances,
        instanceNodes,
        traversalError,
        frame.InstanceNodeCount};
    CameraMap map{geometry, materials, textures, texels, emitters, &frame, &controls, error};
    ThreadStatistics counters{};
    MarkovChain state = chains[index];
    RunChain(map, state, index, *normalization, pass, counters);
    chains[index] = state;
    statistics[index] = counters;
}
kernel void ArvoMetropolisClear(constant FrameConstants& frame [[buffer(6)]],
                                device atomic_uint* pass [[buffer(17)]],
                                device atomic_uint* error [[buffer(19)]],
                                uint pixel [[thread_position_in_grid]])
{
    if (pixel >= frame.Width * frame.Height)
        return;
    for (uint component = 0; component < PassComponents; ++component)
        atomic_store_explicit(pass + pixel * PassComponents + component, 0u, memory_order_relaxed);
    if (pixel == 0)
        atomic_store_explicit(error, 0u, memory_order_relaxed);
}
kernel void ArvoMetropolisCommit(constant FrameConstants& frame [[buffer(6)]],
                                 device float4* accumulation [[buffer(7)]], device uint* counts [[buffer(8)]],
                                 device uint* targets [[buffer(12)]], device atomic_uint* pass [[buffer(17)]],
                                 device atomic_uint* error [[buffer(19)]],
                                 uint pixel [[thread_position_in_grid]])
{
    if (pixel >= frame.Width * frame.Height)
        return;
    float3 value =
        float3(as_type<float>(atomic_load_explicit(pass + pixel * PassComponents, memory_order_relaxed)),
               as_type<float>(atomic_load_explicit(pass + pixel * PassComponents + 1, memory_order_relaxed)),
               as_type<float>(atomic_load_explicit(pass + pixel * PassComponents + 2, memory_order_relaxed)));
    if (!all(isfinite(value)) || !Nonnegative(value))
    {
        Fail(error);
        return;
    }
    float4 moments = accumulation[pixel];
    uint count = counts[pixel];
    AccumulateFilmSample(moments, count, value);
    if (!all(isfinite(moments)))
    {
        Fail(error);
        return;
    }
    accumulation[pixel] = moments;
    counts[pixel] = count;
    targets[pixel] = frame.TotalSamples;
}
