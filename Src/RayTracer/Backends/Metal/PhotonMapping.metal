// Independent Metal photon mapping. Jensen (1996), "Global Illumination using
// Photon Maps". Emission and density estimation are separate native passes.
// Direct, delta-prefix caustic and indirect arrivals partition path space; NEE
// replaces only the direct class. A fixed Epanechnikov disk introduces bias.
// Camera variance excludes shared-map uncertainty: adaptive stopping and
// connection MIS are unavailable. PPM / VCM need independent estimators.
// https://pbr-book.org/3ed-2018/Light_Transport_III_Bidirectional_Methods/Stochastic_Progressive_Photon_Mapping
// Metal has no binary64. Its own WideMath leaf preserves range and about 48 bits
// for cell boundaries, kernel normalization and summation. Disable fast math.
// Interior collisions use a separate normalized three-dimensional kernel.
#include <metal_stdlib>
using namespace metal;
// The low word restores the bits lost when pi is rounded to binary32.
constant float Pi = 3.14159265358979323846f, PiResidual = -8.742278000372485e-8f;
constant float RayEpsilon = 1.0e-4f, MaximumRayDistance = 1.0e30f, BasisParallelThreshold = 0.999f;
constant uint ImportanceBit = 1, NeeBit = 2, RouletteBit = 4, MisShift = 3, MisMask = 3;
constant uint MaximumPhotonReflections = 32, InvalidPhoton = 0xffffffffu;
constant uint EmptyPhoton = 0, DirectPhoton = 1, CausticPhoton = 2, IndirectPhoton = 3;
constant uint VolumePhotonFlag = 0x80000000u, PhotonClassMask = ~VolumePhotonFlag;
constant uint CausticsEnabled = 1, IndirectEnabled = 2, NumericalError = 1, ConfigurationError = 2;
constant float NormalAgreement = 0.9f, PlaneThickness = 0.1f;
constant float VolumeKernelIntegral = 8.0f / 15.0f;
constant float CellLimit = 1048576.0f; // Below 2^24 so adjacent integer comparisons are exact.
constant uint AvalancheMultiplierA = 0x7feb352du, AvalancheMultiplierB = 0x846ca68bu;
constant uint RandomMantissaMask = 0x00ffffffu;
constant float RandomMantissaRange = 16777216.0f;
constant uint PixelSeedMultiplier = 9781, SampleSeedMultiplier = 6271, SecondDimensionScramble = 0x68bc21ebu;
constant uint EmissionDomainKey = 0x4d50484fu, CameraDomainKey = 0x43414d45u, BounceKey = 0x9e3779b9u;
constant uint SourceKey = 0x534f5552u, PositionKey = 0x504f534eu, DirectionKey = 0x44495245u;
constant uint LensKey = 0x4c454e53u, ScatterKey = 0x53434154u, EventKey = 0x45564e54u,
              RouletteKey = 0x52524f55u;
constant uint SourceDimension = 0, PositionDimension = 1, DirectionDimension = 3, RasterDimension = 0;
constant uint EmissionDimensions = 5, VertexDimensions = 4, EventOffset = 2, RouletteOffset = 3;
#include "Hash.metal"
#include "Sequence.metal"
#include "SurfaceTypes.metal"
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
struct PhotonControls
{
    packed_float3 Center;
    float WorldRadius;
    uint Sources, PhotonPaths, SlotsPerPath, Buckets;
    uint Seed, Classes, First, Count;
    float RadiusScale;
    uint Sample, Reserved0, Reserved1;
};
struct PhotonRecord
{
    packed_float3 Position;
    uint Next;
    packed_float3 Incoming;
    uint Depth;
    packed_float3 Flux;
    uint Kind;
    packed_float3 Normal;
    uint Reserved;
};
struct ThreadStatistics
{
    uint2 Primary, Shadow, Reflection, Tests;
};
static_assert(sizeof(FrameConstants) == 240 && sizeof(PhotonControls) == 64 && sizeof(PhotonRecord) == 64 &&
                  sizeof(ThreadStatistics) == 32,
              "Metal photon buffer ABI mismatch");
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
#include "SubsurfaceScattering.metal"
#include "RussianRoulette.metal"
struct PhotonScene
{
    SceneGeometryBuffers Geometry;
    device const PackedMaterial* Materials;
    device const PackedTextureDescriptor* Textures;
    device const uint* Texels;
    device const PackedEmitter* Emitters;
    constant FrameConstants* Frame;
    constant PhotonControls* Controls;
    device atomic_uint* Failure;
};
struct PhotonCell
{
    int X, Y, Z;
};
struct WideVector
{
    WideMath::Number X, Y, Z;
};
struct PhotonEndpoint
{
    HitData Hit;
    float3 Position;
    uint Source;
    bool Surface;
};
void PhotonFail(device atomic_uint* error, uint code = NumericalError)
{
    atomic_fetch_or_explicit(error, code, memory_order_relaxed);
}
bool PhotonFinite(float3 value)
{
    return all(isfinite(value)) && value.x >= 0 && value.y >= 0 && value.z >= 0;
}
bool PhotonPositive(float3 value)
{
    return value.x > 0 || value.y > 0 || value.z > 0;
}
bool PhotonWideFinite(WideMath::Number value)
{
    return isfinite(value.High) && isfinite(value.Low);
}
float PhotonFloat(WideMath::Number value)
{
    return as_type<float>(WideMath::Encode(value));
}
WideMath::Number PhotonRadius(constant PhotonControls& controls)
{
    return WideMath::Multiply(WideMath::FromFloat(controls.WorldRadius),
                              WideMath::FromFloat(controls.RadiusScale));
}
WideVector PhotonDifference(float3 a, float3 b)
{
    using namespace WideMath;
    return {Subtract(FromFloat(a.x), FromFloat(b.x)), Subtract(FromFloat(a.y), FromFloat(b.y)),
            Subtract(FromFloat(a.z), FromFloat(b.z))};
}
WideMath::Number PhotonDot(WideVector a, WideVector b)
{
    using namespace WideMath;
    return Add(Add(Multiply(a.X, b.X), Multiply(a.Y, b.Y)), Multiply(a.Z, b.Z));
}
// Rounding a quotient to float before flooring can move a point across a cell
// boundary. Correct the candidate against the full expansion on both sides.
bool PhotonCellAxis(WideMath::Number coordinate, thread int& cell)
{
    using namespace WideMath;
    if (!PhotonWideFinite(coordinate) || Compare(Absolute(coordinate), FromFloat(CellLimit)) >= 0)
        return false;
    cell = int(floor(PhotonFloat(coordinate)));
    if (Compare(coordinate, FromFloat(float(cell))) < 0)
        --cell;
    else if (Compare(coordinate, FromFloat(float(cell + 1))) >= 0)
        ++cell;
    return true;
}
bool PhotonCellAt(float3 position, constant PhotonControls& controls, device atomic_uint* error,
                  thread PhotonCell& cell)
{
    using namespace WideMath;
    Number radius = PhotonRadius(controls);
    if (!(radius.High > 0))
        return false;
    WideVector relative = PhotonDifference(position, float3(controls.Center));
    if (!PhotonCellAxis(Divide(relative.X, radius), cell.X) ||
        !PhotonCellAxis(Divide(relative.Y, radius), cell.Y) ||
        !PhotonCellAxis(Divide(relative.Z, radius), cell.Z))
    {
        PhotonFail(error);
        return false;
    }
    return true;
}
uint PhotonBucket(PhotonCell cell, uint count)
{
    const uint XMix = 0x9e3779b9u, YMix = 0x85ebca6bu, ZMix = 0xc2b2ae35u;
    return Hash(uint(cell.X) * XMix ^ uint(cell.Y) * YMix ^ uint(cell.Z) * ZMix) & (count - 1);
}
bool PhotonSameCell(PhotonCell a, PhotonCell b)
{
    return a.X == b.X && a.Y == b.Y && a.Z == b.Z;
}
uint PhotonCategory(uint depth, bool allDelta)
{
    return depth == 0 ? DirectPhoton : allDelta ? CausticPhoton : IndirectPhoton;
}
bool PhotonAccepted(PhotonRecord photon, uint cameraDepth, constant FrameConstants& frame,
                    constant PhotonControls& controls)
{
    if (photon.Kind & VolumePhotonFlag)
        return false;
    if (cameraDepth > frame.MaxReflectionBounces || photon.Depth > frame.MaxReflectionBounces - cameraDepth)
        return false;
    if (photon.Kind == DirectPhoton)
        return (frame.TransportFlags & NeeBit) == 0;
    if (photon.Kind == CausticPhoton)
        return (controls.Classes & CausticsEnabled) != 0;
    return photon.Kind == IndirectPhoton && (controls.Classes & IndirectEnabled) != 0;
}
WideMath::Number PhotonKernelWeight(WideVector separation, float3 normal, float agreement,
                                    float shadingProjection, float geometryProjection,
                                    constant PhotonControls& controls)
{
    using namespace WideMath;
    Number radius = PhotonRadius(controls);
    if (!(radius.High > 0) || !controls.PhotonPaths || !(geometryProjection > 0) ||
        agreement < NormalAgreement)
        return {};
    Number squaredRadius = Multiply(radius, radius), squaredDistance = PhotonDot(separation, separation);
    if (Compare(squaredDistance, squaredRadius) >= 0)
        return {};
    Number plane =
        Absolute(PhotonDot(separation, {FromFloat(normal.x), FromFloat(normal.y), FromFloat(normal.z)}));
    if (Compare(plane, Multiply(radius, FromFloat(PlaneThickness))) > 0)
        return {};
    Number disk = Multiply(Subtract(FromFloat(1), Divide(squaredDistance, squaredRadius)), FromFloat(2));
    Number divisor = Multiply(Multiply(Add(FromFloat(Pi), FromFloat(PiResidual)), squaredRadius),
                              FromFloat(float(controls.PhotonPaths)));
    return Divide(Multiply(Divide(disk, divisor), FromFloat(shadingProjection)),
                  FromFloat(geometryProjection));
}
WideMath::Number PhotonVolumeWeight(WideVector separation, float meanFreePath,
                                    constant PhotonControls& controls)
{
    using namespace WideMath;
    Number radius = PhotonRadius(controls);
    if (!(radius.High > 0) || !controls.PhotonPaths)
        return {};
    Number squaredRadius = Multiply(radius, radius), squaredDistance = PhotonDot(separation, separation);
    if (Compare(squaredDistance, squaredRadius) >= 0)
        return {};
    Number radial = Subtract(FromFloat(1), Divide(squaredDistance, squaredRadius));
    Number pi = Add(FromFloat(Pi), FromFloat(PiResidual));
    Number denominator = Multiply(Multiply(Multiply(pi, radius), squaredRadius),
        Multiply(FromFloat(VolumeKernelIntegral), FromFloat(float(controls.PhotonPaths))));
    return Divide(Multiply(FromFloat(meanFreePath), radial), denominator);
}
bool PhotonConfiguration(constant FrameConstants& frame, constant PhotonControls& controls)
{
    return frame.MaxReflectionBounces <= MaximumPhotonReflections &&
           controls.SlotsPerPath == frame.MaxReflectionBounces + 1 && controls.PhotonPaths > 0 &&
           controls.Buckets > 0 && (controls.Buckets & (controls.Buckets - 1)) == 0 &&
           !frame.AdaptiveEnabled && ((frame.TransportFlags >> MisShift) & MisMask) == 0 &&
           frame.ShadowStrength == 1 && controls.WorldRadius >= 0 && isfinite(controls.WorldRadius) &&
           controls.RadiusScale > 0 && isfinite(controls.RadiusScale);
}
float PhotonRandom(constant PhotonControls& controls, uint path, uint dimension)
{
    return Random01(Hash(controls.Seed ^ EmissionDomainKey) ^ Hash(path) ^ Hash(dimension + SourceKey));
}
float2 PhotonPair(constant PhotonControls& controls, uint path, uint dimension)
{
    return float2(PhotonRandom(controls, path, dimension), PhotonRandom(controls, path, dimension + 1));
}
EmissionDomain PhotonDomain(constant PhotonControls& controls)
{
    return {float3(controls.Center), controls.WorldRadius};
}
float PhotonLinear(uint channel)
{
    const float MaximumByte = 255, Threshold = 0.04045f, Slope = 12.92f, Offset = 0.055f, Scale = 1.055f,
                Exponent = 2.4f;
    float encoded = float(channel) / MaximumByte;
    return encoded <= Threshold ? encoded / Slope : pow((encoded + Offset) / Scale, Exponent);
}
float3 PhotonTexel(PhotonScene scene, PackedTextureDescriptor image, int x, int y)
{
    x = (x % int(image.Width) + int(image.Width)) % int(image.Width);
    y = (y % int(image.Height) + int(image.Height)) % int(image.Height);
    uint color = scene.Texels[image.OffsetTexels + uint(y) * image.Width + uint(x)];
    const uint ByteMask = 0xff, GreenShift = 8, BlueShift = 16;
    return float3(PhotonLinear(color & ByteMask), PhotonLinear((color >> GreenShift) & ByteMask),
                  PhotonLinear((color >> BlueShift) & ByteMask));
}
float3 PhotonColor(PhotonScene scene, HitData hit)
{
    uint index = scene.Materials[hit.MaterialIndex].DiffuseTextureIndex;
    if (!hit.HasTexCoord || index >= scene.Frame->TextureCount || !isfinite(hit.TexCoord.x) ||
        !isfinite(hit.TexCoord.y))
        return hit.Color;
    PackedTextureDescriptor image = scene.Textures[index];
    if (!image.Width || !image.Height || image.OffsetTexels > scene.Frame->TextureTexelCount ||
        image.Height > (scene.Frame->TextureTexelCount - image.OffsetTexels) / image.Width)
        return hit.Color;
    float x = fract(hit.TexCoord.x) * image.Width - 0.5f, y = fract(1 - hit.TexCoord.y) * image.Height - 0.5f;
    int column = int(floor(x)), row = int(floor(y));
    return hit.Color * mix(mix(PhotonTexel(scene, image, column, row),
                               PhotonTexel(scene, image, column + 1, row), fract(x)),
                           mix(PhotonTexel(scene, image, column, row + 1),
                               PhotonTexel(scene, image, column + 1, row + 1), fract(x)),
                           fract(y));
}
bool PhotonRectangle(PackedEmitter light, RayData ray, thread float& closest)
{
    if (light.Kind != PackedEmitterRectangle)
        return false;
    float3 normal = float3(light.Direction);
    float denominator = dot(normal, ray.Direction);
    if (denominator == 0)
        return false;
    float t = dot(normal, float3(light.Origin) - ray.Origin) / denominator;
    if (!(t > ray.MinimumDistance && t < ray.MaximumDistance && t <= closest))
        return false;
    float3 axis = abs(normal.y) < BasisParallelThreshold ? float3(0, 1, 0) : float3(1, 0, 0);
    float3 right = normalize(cross(axis, normal)), up = cross(normal, right);
    float3 relative = ray.Origin + ray.Direction * t - float3(light.Origin);
    if (abs(dot(relative, right)) > light.Width / 2 || abs(dot(relative, up)) > light.Height / 2)
        return false;
    closest = t;
    return true;
}
PhotonEndpoint PhotonTrace(PhotonScene scene, RayData ray, thread ThreadStatistics& statistics)
{
    PhotonEndpoint endpoint{};
    endpoint.Source = InvalidPhoton;
    endpoint.Hit.Distance = ray.MaximumDistance;
    endpoint.Surface = IntersectScene(scene.Geometry, ray, false, endpoint.Hit, statistics.Tests);
    // Explicit rectangles may also have a display mesh (Cornell's ceiling).
    // Let the sampled emitter win a coincident proxy hit within the ray offset;
    // ordinary occluders retain their exact distance ordering.
    bool proxy = endpoint.Surface && scene.Materials[endpoint.Hit.MaterialIndex].SampledEmitter != 0;
    float closest = endpoint.Hit.Distance + (proxy ? RayEpsilon : 0);
    for (uint i = 0; i < scene.Controls->Sources; ++i)
        if (PhotonRectangle(scene.Emitters[i], ray, closest))
        {
            endpoint.Surface = false;
            endpoint.Source = i;
        }
    if (endpoint.Source == InvalidPhoton)
        for (uint i = 0; i < scene.Controls->Sources; ++i)
            if (endpoint.Surface ? scene.Emitters[i].Primitive != InvalidPrimitiveIndex &&
                                       scene.Emitters[i].Primitive == endpoint.Hit.PrimitiveIndex
                                 : scene.Emitters[i].Kind == PackedEmitterEnvironment)
            {
                endpoint.Source = i;
                break;
            }
    endpoint.Position = ray.Origin + ray.Direction * (endpoint.Surface ? endpoint.Hit.Distance : closest);
    return endpoint;
}
float3 PhotonRadiance(PackedEmitter light, float3 outward)
{
    if (light.Kind == PackedEmitterEnvironment)
        return mix(float3(light.BackgroundBottom), float3(light.BackgroundTop), (1 - outward.y) / 2);
    if (light.Kind == PackedEmitterRectangle)
        return dot(float3(light.Direction), outward) > 0
                   ? float3(light.Strength) / (light.Width * light.Height)
                   : float3(0);
    return float3(light.Strength);
}
SurfaceSample PhotonMirror(SurfaceBsdf surface)
{
    SurfaceSample event{};
    event.Direction = SurfaceWorld(surface, float3(-surface.View.x, -surface.View.y, surface.View.z));
    if (!SurfaceDirectionValid(surface, event.Direction, surface.View.z))
        return {};
    event.Weight = float3(SurfaceAdjointCorrection(surface, event.Direction));
    event.Pdf = event.ReversePdf = event.RelativeIndex = 1;
    event.Event = ScatteringReflection | ScatteringDelta;
    return event;
}
void PhotonLightPath(PhotonScene scene, uint path, device PhotonRecord* records,
                     thread ThreadStatistics& statistics)
{
    constant PhotonControls& controls = *scene.Controls;
    uint base = path * controls.SlotsPerPath;
    for (uint i = 0; i < controls.SlotsPerPath; ++i)
        records[base + i].Kind = EmptyPhoton;
    EmitterEndpoint endpoint = SampleEmitterEndpoint(scene.Emitters, controls.Sources, PhotonDomain(controls),
                                                     PhotonRandom(controls, path, SourceDimension),
                                                     PhotonPair(controls, path, PositionDimension),
                                                     PhotonPair(controls, path, DirectionDimension));
    EmissionSample emission = endpoint.Emission;
    if (!(emission.Flags & EmissionValid))
        return;
    float3 origin =
        emission.Origin +
        emission.Normal * (dot(emission.Normal, emission.Direction) >= 0 ? RayEpsilon : -RayEpsilon);
    RayData ray{origin, emission.Direction, RayEpsilon, MaximumRayDistance};
    float3 flux = emission.Weight;
    bool allDelta = true;
    uint interiorMaterial = InvalidPhoton;
    for (uint depth = 0; depth < controls.SlotsPerPath; ++depth)
    {
        if (!PhotonFinite(flux))
        {
            PhotonFail(scene.Failure);
            break;
        }
        if (!PhotonPositive(flux))
            break;
        IncrementCounter(statistics.Reflection);
        PhotonEndpoint hit = PhotonTrace(scene, ray, statistics);
        uint dimension = EmissionDimensions + depth * VertexDimensions;
        if (hit.Surface && hit.Hit.MaterialIndex >= scene.Frame->MaterialCount)
        {
            PhotonFail(scene.Failure, ConfigurationError);
            break;
        }
        if (depth == 0 && hit.Surface && !hit.Hit.FrontFace &&
            scene.Materials[hit.Hit.MaterialIndex].Optical.w > 0)
            interiorMaterial = hit.Hit.MaterialIndex;
        if (interiorMaterial != InvalidPhoton)
        {
            PackedMaterial interior = scene.Materials[interiorMaterial];
            bool finiteSource = hit.Source != InvalidPhoton &&
                scene.Emitters[hit.Source].Kind == PackedEmitterRectangle;
            float boundary = hit.Surface || finiteSource ? length(hit.Position - ray.Origin)
                : ray.MaximumDistance;
            SubsurfaceFlight flight = SubsurfaceFreeFlight(interior.Optical.w,
                interior.AlbedoReflectivity.xyz * interior.ConductorExtinction.w,
                ray.Direction, boundary,
                PhotonRandom(controls, path, dimension + EventOffset));
            if (flight.Scattered)
                records[base + depth] = {packed_float3(ray.Origin + flight.Distance * ray.Direction),
                    InvalidPhoton, packed_float3(-ray.Direction), depth, packed_float3(flux),
                    PhotonCategory(depth, allDelta) | VolumePhotonFlag, packed_float3(float3(0)),
                    interiorMaterial};
            flux *= flight.Weight;
            if (!PhotonFinite(flux))
            {
                PhotonFail(scene.Failure);
                break;
            }
            if (!PhotonPositive(flux)) break;
            if (flight.Scattered)
            {
                if (depth == scene.Frame->MaxReflectionBounces) break;
                SubsurfaceDirection phase = SubsurfaceSampleDirection(ray.Direction,
                    interior.ConductorIndex.w, PhotonPair(controls, path, dimension),
                    (scene.Frame->TransportFlags & ImportanceBit) != 0);
                if (!(phase.Pdf > 0) || !isfinite(phase.Weight))
                {
                    PhotonFail(scene.Failure);
                    break;
                }
                flux *= phase.Weight;
                if ((scene.Frame->TransportFlags & RouletteBit) && depth + 1 >= scene.Frame->RouletteStartDepth)
                {
                    float survival = RouletteProbability(flux);
                    if (PhotonRandom(controls, path, dimension + RouletteOffset) >= survival) break;
                    flux /= survival;
                }
                allDelta = false;
                ray = {ray.Origin + flight.Distance * ray.Direction, phase.Direction,
                       RayEpsilon, MaximumRayDistance};
                continue;
            }
        }
        if (!hit.Surface) break;
        PackedMaterial material = scene.Materials[hit.Hit.MaterialIndex];
        bool legacy = material.Model == MaterialLegacy;
        float mirror = legacy ? clamp(material.AlbedoReflectivity.w, 0.0f, 1.0f) : 0;
        if (legacy)
            material.Model = MaterialDiffuse;
        SurfaceBsdf surface = MakeSurfaceBsdf(material, hit.Hit, -ray.Direction, PhotonColor(scene, hit.Hit),
                                              scene.Frame->TransportFlags, TransportImportance);
        if (!SurfaceDelta(surface) && mirror < 1)
            records[base + depth] = {packed_float3(hit.Position),
                                     InvalidPhoton,
                                     packed_float3(-ray.Direction),
                                     depth,
                                     packed_float3(flux),
                                     PhotonCategory(depth, allDelta),
                                     packed_float3(hit.Hit.GeometricNormal),
                                     hit.Hit.MaterialIndex};
        if (depth == scene.Frame->MaxReflectionBounces)
            break;
        float eventU = PhotonRandom(controls, path, dimension + EventOffset);
        SurfaceSample event;
        if (legacy && eventU < mirror)
            event = PhotonMirror(surface);
        else
            event = SampleSurface(surface, PhotonPair(controls, path, dimension), eventU);
        if (!event.Event || !(event.Pdf > 0))
            break;
        flux *= event.Weight;
        allDelta = allDelta && (event.Event & ScatteringDelta) != 0;
        if (!PhotonFinite(flux))
        {
            PhotonFail(scene.Failure);
            break;
        }
        if ((scene.Frame->TransportFlags & RouletteBit) && depth + 1 >= scene.Frame->RouletteStartDepth)
        {
            float survival = RouletteProbability(flux);
            if (PhotonRandom(controls, path, dimension + RouletteOffset) >= survival)
                break;
            flux /= survival;
        }
        ray = {SurfaceOffset(surface, hit.Position, event.Direction), event.Direction, RayEpsilon,
               MaximumRayDistance};
        if (material.Optical.w > 0 && (event.Event & ScatteringTransmission) != 0)
            interiorMaterial = hit.Hit.FrontFace ? hit.Hit.MaterialIndex : InvalidPhoton;
    }
}
float3 PhotonGather(PhotonScene scene, PhotonEndpoint hit, SurfaceBsdf surface, uint cameraDepth,
                    device const PhotonRecord* records, device atomic_uint* heads)
{
    using namespace WideMath;
    PhotonCell query;
    if (!PhotonCellAt(hit.Position, *scene.Controls, scene.Failure, query))
        return float3(0);
    Number red{}, green{}, blue{};
    for (int z = -1; z <= 1; ++z)
        for (int y = -1; y <= 1; ++y)
            for (int x = -1; x <= 1; ++x)
            {
                PhotonCell neighbor{query.X + x, query.Y + y, query.Z + z};
                uint index = atomic_load_explicit(heads + PhotonBucket(neighbor, scene.Controls->Buckets),
                                                  memory_order_relaxed),
                     visited = 0;
                while (index != InvalidPhoton)
                {
                    if (index >= scene.Controls->PhotonPaths * scene.Controls->SlotsPerPath ||
                        ++visited > scene.Controls->PhotonPaths * scene.Controls->SlotsPerPath)
                    {
                        PhotonFail(scene.Failure);
                        return float3(0);
                    }
                    PhotonRecord photon = records[index];
                    index = photon.Next;
                    if (!PhotonAccepted(photon, cameraDepth, *scene.Frame, *scene.Controls))
                        continue;
                    if (surface.Material.Optical.w > 0 &&
                        (hit.Hit.FrontFace ? dot(float3(photon.Incoming), hit.Hit.GeometricNormal) <= 0
                                           : dot(float3(photon.Incoming), hit.Hit.GeometricNormal) >= 0))
                        continue;
                    PhotonCell cell;
                    if (!PhotonCellAt(float3(photon.Position), *scene.Controls, scene.Failure, cell) ||
                        !PhotonSameCell(cell, neighbor))
                        continue;
                    float3 incoming = float3(photon.Incoming), flux = float3(photon.Flux),
                           normal = hit.Hit.GeometricNormal;
                    Number weight = PhotonKernelWeight(
                        PhotonDifference(float3(photon.Position), hit.Position), normal,
                        dot(normal, float3(photon.Normal)), abs(dot(surface.Normal, incoming)),
                        abs(dot(normal, incoming)), *scene.Controls);
                    if (!(weight.High > 0))
                        continue;
                    SurfaceEvaluation scatter = EvaluateSurface(surface, incoming);
                    if (!PhotonFinite(scatter.Value))
                    {
                        PhotonFail(scene.Failure);
                        return float3(0);
                    }
                    red = Add(red, Multiply(Multiply(FromFloat(flux.x), FromFloat(scatter.Value.x)), weight));
                    green =
                        Add(green, Multiply(Multiply(FromFloat(flux.y), FromFloat(scatter.Value.y)), weight));
                    blue =
                        Add(blue, Multiply(Multiply(FromFloat(flux.z), FromFloat(scatter.Value.z)), weight));
                }
            }
    if (!PhotonWideFinite(red) || !PhotonWideFinite(green) || !PhotonWideFinite(blue))
        PhotonFail(scene.Failure);
    return float3(PhotonFloat(red), PhotonFloat(green), PhotonFloat(blue));
}
float3 PhotonVolumeGather(PhotonScene scene, float3 position, float3 cameraDirection,
                          uint materialIndex, uint cameraDepth, device const PhotonRecord* records,
                          device atomic_uint* heads, thread ThreadStatistics& statistics)
{
    using namespace WideMath;
    PhotonCell query;
    if (!PhotonCellAt(position, *scene.Controls, scene.Failure, query))
        return float3(0);
    Number red{}, green{}, blue{};
    for (int z = -1; z <= 1; ++z)
        for (int y = -1; y <= 1; ++y)
            for (int x = -1; x <= 1; ++x)
            {
                PhotonCell neighbor{query.X + x, query.Y + y, query.Z + z};
                uint index = atomic_load_explicit(heads + PhotonBucket(neighbor, scene.Controls->Buckets),
                                                  memory_order_relaxed), visited = 0;
                while (index != InvalidPhoton)
                {
                    if (index >= scene.Controls->PhotonPaths * scene.Controls->SlotsPerPath ||
                        ++visited > scene.Controls->PhotonPaths * scene.Controls->SlotsPerPath)
                    {
                        PhotonFail(scene.Failure);
                        return float3(0);
                    }
                    PhotonRecord photon = records[index];
                    index = photon.Next;
                    if (!(photon.Kind & VolumePhotonFlag) || photon.Reserved != materialIndex ||
                        cameraDepth > scene.Frame->MaxReflectionBounces ||
                        photon.Depth > scene.Frame->MaxReflectionBounces - cameraDepth)
                        continue;
                    uint category = photon.Kind & PhotonClassMask;
                    if (category == CausticPhoton && !(scene.Controls->Classes & CausticsEnabled)) continue;
                    if (category == IndirectPhoton && !(scene.Controls->Classes & IndirectEnabled)) continue;
                    PhotonCell cell;
                    if (!PhotonCellAt(float3(photon.Position), *scene.Controls, scene.Failure, cell) ||
                        !PhotonSameCell(cell, neighbor))
                        continue;
                    float3 separation = float3(photon.Position) - position;
                    float distance = length(separation);
                    if (distance > 2 * RayEpsilon)
                    {
                        IncrementCounter(statistics.Shadow);
                        RayData connection{position, separation / distance, RayEpsilon,
                                           distance - RayEpsilon};
                        HitData blocker{};
                        blocker.Distance = connection.MaximumDistance;
                        if (IntersectScene(scene.Geometry, connection, true, blocker, statistics.Tests))
                            continue;
                    }
                    Number weight = PhotonVolumeWeight(PhotonDifference(float3(photon.Position), position),
                        scene.Materials[materialIndex].Optical.w, *scene.Controls);
                    if (!(weight.High > 0)) continue;
                    float phase = SubsurfacePhase(dot(float3(photon.Incoming), cameraDirection),
                        scene.Materials[materialIndex].ConductorIndex.w);
                    weight = Multiply(weight, FromFloat(phase));
                    float3 flux = float3(photon.Flux);
                    red = Add(red, Multiply(FromFloat(flux.x), weight));
                    green = Add(green, Multiply(FromFloat(flux.y), weight));
                    blue = Add(blue, Multiply(FromFloat(flux.z), weight));
                }
            }
    if (!PhotonWideFinite(red) || !PhotonWideFinite(green) || !PhotonWideFinite(blue))
        PhotonFail(scene.Failure);
    return float3(PhotonFloat(red), PhotonFloat(green), PhotonFloat(blue));
}
float3 PhotonDirect(PhotonScene scene, PhotonEndpoint hit, SurfaceBsdf surface, SampleContext random,
                    uint domain, bool filterTransmission, thread ThreadStatistics& statistics)
{
    using namespace WideMath;
    EmitterEndpoint endpoint =
        SampleEmitterEndpoint(scene.Emitters, scene.Controls->Sources, PhotonDomain(*scene.Controls),
                              Sample2D(random, domain ^ SourceKey).x, Sample2D(random, domain ^ PositionKey),
                              Sample2D(random, domain ^ DirectionKey));
    EmissionSample sample = endpoint.Emission;
    if (!(sample.Flags & EmissionValid))
        return float3(0);
    bool infinite = (sample.Flags & EmissionInfinite) != 0,
         deltaPosition = (sample.Flags & EmissionDeltaPosition) != 0;
    float3 direction = -sample.Direction;
    Number geometry = FromFloat(1);
    if (!infinite)
    {
        float3 segment = sample.Origin - hit.Position;
        float distance = length(segment);
        if (!(distance > 0) || !isfinite(distance))
            return float3(0);
        direction = segment / distance;
        geometry = Divide(FromFloat(deltaPosition ? 1 : abs(dot(sample.Normal, direction))),
                          Multiply(FromFloat(distance), FromFloat(distance)));
    }
    bool transmitted = filterTransmission &&
        (hit.Hit.FrontFace ? dot(direction, hit.Hit.GeometricNormal) < 0
                           : dot(direction, hit.Hit.GeometricNormal) > 0);
    if (transmitted && !deltaPosition && !(sample.Flags & EmissionDeltaDirection))
        return float3(0);
    Number probability = Multiply(FromFloat(endpoint.SelectionProbability),
                                  FromFloat(infinite ? sample.DirectionPdf : sample.PositionPdf));
    if (!(probability.High > 0))
        return float3(0);
    SurfaceEvaluation scatter = EvaluateSurface(surface, direction);
    float3 emission = PhotonRadiance(scene.Emitters[endpoint.EmitterIndex], -direction);
    if (!PhotonFinite(scatter.Value) || !PhotonFinite(emission))
    {
        PhotonFail(scene.Failure);
        return float3(0);
    }
    Number factor = Divide(Multiply(geometry, FromFloat(abs(dot(surface.Normal, direction)))), probability);
    float3 value =
        float3(PhotonFloat(Multiply(Multiply(FromFloat(scatter.Value.x), FromFloat(emission.x)), factor)),
               PhotonFloat(Multiply(Multiply(FromFloat(scatter.Value.y), FromFloat(emission.y)), factor)),
               PhotonFloat(Multiply(Multiply(FromFloat(scatter.Value.z), FromFloat(emission.z)), factor)));
    if (transmitted && hit.Hit.FrontFace && deltaPosition)
        value *= exp(float3(-length(sample.Origin - hit.Position) / surface.Material.Optical.w));
    if (!PhotonFinite(value))
    {
        PhotonFail(scene.Failure);
        return float3(0);
    }
    if (!PhotonPositive(value))
        return value;
    RayData shadow{SurfaceOffset(surface, hit.Position, direction), direction, RayEpsilon,
                   MaximumRayDistance};
    if (!infinite)
    {
        float3 segment = sample.Origin - shadow.Origin;
        float distance = length(segment);
        if (!(distance > 2 * RayEpsilon))
            return float3(0);
        shadow.Direction = segment / distance;
        shadow.MaximumDistance = distance - RayEpsilon;
    }
    IncrementCounter(statistics.Shadow);
    HitData obstruction{};
    obstruction.Distance = shadow.MaximumDistance;
    if (IntersectScene(scene.Geometry, shadow, true, obstruction, statistics.Tests))
        return float3(0);
    float closest = shadow.MaximumDistance;
    for (uint i = 0; i < scene.Controls->Sources; ++i)
        if (PhotonRectangle(scene.Emitters[i], shadow, closest))
            return float3(0);
    return value;
}
float3 PhotonCameraPath(PhotonScene scene, RayData ray, SampleContext random,
                        device const PhotonRecord* records, device atomic_uint* heads,
                        thread ThreadStatistics& statistics)
{
    float3 result = float3(0), throughput = float3(1);
    float etaScale = 1;
    uint interiorMaterial = InvalidPhoton;
    for (uint depth = 0; depth <= scene.Frame->MaxReflectionBounces + 1; ++depth)
    {
        if (!PhotonFinite(throughput) || !isfinite(etaScale))
        {
            PhotonFail(scene.Failure);
            break;
        }
        if (!PhotonPositive(throughput))
            break;
        if (depth == 0)
            IncrementCounter(statistics.Primary);
        else
            IncrementCounter(statistics.Reflection);
        PhotonEndpoint hit = PhotonTrace(scene, ray, statistics);
        uint domain = CameraDomainKey ^ (depth * BounceKey);
        if (hit.Surface && hit.Hit.MaterialIndex >= scene.Frame->MaterialCount)
        {
            PhotonFail(scene.Failure, ConfigurationError);
            break;
        }
        if (depth == 0 && hit.Surface && !hit.Hit.FrontFace &&
            scene.Materials[hit.Hit.MaterialIndex].Optical.w > 0)
            interiorMaterial = hit.Hit.MaterialIndex;
        if (interiorMaterial != InvalidPhoton)
        {
            PackedMaterial interior = scene.Materials[interiorMaterial];
            bool finiteSource = hit.Source != InvalidPhoton &&
                scene.Emitters[hit.Source].Kind == PackedEmitterRectangle;
            float boundary = hit.Surface || finiteSource ? length(hit.Position - ray.Origin)
                : ray.MaximumDistance;
            SubsurfaceFlight flight = SubsurfaceFreeFlight(interior.Optical.w,
                interior.AlbedoReflectivity.xyz * interior.ConductorExtinction.w,
                ray.Direction, boundary, Sample2D(random, domain ^ EventKey).x);
            throughput *= flight.Weight;
            if (!PhotonFinite(throughput))
            {
                PhotonFail(scene.Failure);
                break;
            }
            if (!PhotonPositive(throughput)) break;
            if (flight.Scattered)
            {
                if (depth > scene.Frame->MaxReflectionBounces) break;
                result += throughput * PhotonVolumeGather(scene,
                    ray.Origin + flight.Distance * ray.Direction, ray.Direction, interiorMaterial,
                    depth, records, heads, statistics);
                break;
            }
        }
        if (hit.Source != InvalidPhoton)
            result += throughput * PhotonRadiance(scene.Emitters[hit.Source], -ray.Direction);
        if (!hit.Surface || depth > scene.Frame->MaxReflectionBounces)
            break;
        PackedMaterial material = scene.Materials[hit.Hit.MaterialIndex];
        bool legacy = material.Model == MaterialLegacy;
        float mirror = legacy ? clamp(material.AlbedoReflectivity.w, 0.0f, 1.0f) : 0;
        if (legacy)
            material.Model = MaterialDiffuse;
        SurfaceBsdf surface = MakeSurfaceBsdf(material, hit.Hit, -ray.Direction, PhotonColor(scene, hit.Hit),
                                              scene.Frame->TransportFlags);
        bool subsurfaceTransmission = material.Optical.w > 0 && !SurfaceDelta(surface);
        if (!SurfaceDelta(surface) && mirror < 1)
        {
            float3 gathered = PhotonGather(scene, hit, surface, depth, records, heads);
            if (scene.Frame->TransportFlags & NeeBit)
                gathered += PhotonDirect(scene, hit, surface, random, domain,
                                         subsurfaceTransmission, statistics);
            result += throughput * gathered * (1 - mirror);
        }
        SurfaceSample event;
        if (legacy && mirror > 0)
        {
            event = PhotonMirror(surface);
            event.Weight *= mirror;
        }
        else if (SurfaceDelta(surface) || subsurfaceTransmission)
            event = SampleSurface(surface, Sample2D(random, domain ^ ScatterKey),
                                  Sample2D(random, domain ^ EventKey).x);
        else
            break;
        if (!event.Event || !(event.Pdf > 0))
            break;
        if (subsurfaceTransmission && !(event.Event & ScatteringTransmission))
            break;
        throughput *= event.Weight;
        etaScale *= event.RelativeIndex * event.RelativeIndex;
        if (!PhotonFinite(throughput) || !isfinite(etaScale))
        {
            PhotonFail(scene.Failure);
            break;
        }
        if ((scene.Frame->TransportFlags & RouletteBit) && depth + 1 >= scene.Frame->RouletteStartDepth)
        {
            float survival = RouletteProbability(throughput * etaScale);
            if (Sample2D(random, domain ^ RouletteKey).x >= survival)
                break;
            throughput /= survival;
        }
        ray = {SurfaceOffset(surface, hit.Position, event.Direction), event.Direction, RayEpsilon,
               MaximumRayDistance};
        if (material.Optical.w > 0 && (event.Event & ScatteringTransmission) != 0)
            interiorMaterial = hit.Hit.FrontFace ? hit.Hit.MaterialIndex : InvalidPhoton;
    }
    return result;
}

kernel void ArvoPhotonClear(device atomic_uint* heads [[buffer(17)]],
                            constant PhotonControls& controls [[buffer(18)]],
                            device atomic_uint* error [[buffer(19)]], uint lane [[thread_position_in_grid]])
{
    if (lane < controls.Buckets)
        atomic_store_explicit(heads + lane, InvalidPhoton, memory_order_relaxed);
    if (lane == 0)
        atomic_store_explicit(error, 0u, memory_order_relaxed);
}
kernel void ArvoPhotonEmit(
    device const PackedAccelerationNode* nodes [[buffer(0)]], device const uint* references [[buffer(1)]],
    device const PackedTriangle* triangles [[buffer(2)]], device const PackedSphere* spheres [[buffer(3)]],
    device const PackedMaterial* materials [[buffer(4)]], device const PackedEmitter* emitters [[buffer(5)]],
    constant FrameConstants& frame [[buffer(6)]], device ThreadStatistics* statistics [[buffer(9)]],
    device const PackedTextureDescriptor* textures [[buffer(10)]], device const uint* texels [[buffer(11)]],
    device const InstanceRecord* instances [[buffer(13)]],
    device const InstanceNode* instanceNodes [[buffer(14)]],
    device atomic_uint* traversalError [[buffer(15)]], device PhotonRecord* records [[buffer(16)]],
    constant PhotonControls& controls [[buffer(18)]], device atomic_uint* error [[buffer(19)]],
    uint lane [[thread_position_in_grid]])
{
    if (lane >= controls.Count)
        return;
    statistics[lane] = {};
    if (!PhotonConfiguration(frame, controls))
    {
        PhotonFail(error, ConfigurationError);
        return;
    }
    uint path = controls.First + lane;
    if (path >= controls.PhotonPaths)
        return;
    SceneGeometryBuffers geometry{
        {nodes, references, triangles, spheres, frame.NodeCount, frame.TriangleCount, frame.Acceleration},
        instances,
        instanceNodes,
        traversalError,
        frame.InstanceNodeCount};
    PhotonScene scene{geometry, materials, textures, texels, emitters, &frame, &controls, error};
    ThreadStatistics counters{};
    PhotonLightPath(scene, path, records, counters);
    statistics[lane] = counters;
}
kernel void ArvoPhotonIndex(device PhotonRecord* records [[buffer(16)]],
                            device atomic_uint* heads [[buffer(17)]],
                            constant PhotonControls& controls [[buffer(18)]],
                            device atomic_uint* error [[buffer(19)]], uint lane [[thread_position_in_grid]])
{
    if (lane >= controls.Count)
        return;
    uint index = controls.First + lane;
    if (index >= controls.PhotonPaths * controls.SlotsPerPath || records[index].Kind == EmptyPhoton)
        return;
    PhotonCell cell;
    if (!PhotonCellAt(float3(records[index].Position), controls, error, cell))
        return;
    records[index].Next =
        atomic_exchange_explicit(heads + PhotonBucket(cell, controls.Buckets), index, memory_order_relaxed);
    // No query overlaps insertion. Command completion publishes every link;
    // exact cell matching later removes duplicate visits from bucket collisions.
}
kernel void ArvoPhotonCamera(
    device const PackedAccelerationNode* nodes [[buffer(0)]], device const uint* references [[buffer(1)]],
    device const PackedTriangle* triangles [[buffer(2)]], device const PackedSphere* spheres [[buffer(3)]],
    device const PackedMaterial* materials [[buffer(4)]], device const PackedEmitter* emitters [[buffer(5)]],
    constant FrameConstants& frame [[buffer(6)]], device float4* accumulation [[buffer(7)]],
    device uint* counts [[buffer(8)]], device ThreadStatistics* statistics [[buffer(9)]],
    device const PackedTextureDescriptor* textures [[buffer(10)]], device const uint* texels [[buffer(11)]],
    device uint* targets [[buffer(12)]], device const InstanceRecord* instances [[buffer(13)]],
    device const InstanceNode* instanceNodes [[buffer(14)]],
    device atomic_uint* traversalError [[buffer(15)]], device const PhotonRecord* records [[buffer(16)]],
    device atomic_uint* heads [[buffer(17)]], constant PhotonControls& controls [[buffer(18)]],
    device atomic_uint* error [[buffer(19)]], uint lane [[thread_position_in_grid]])
{
    if (lane >= controls.Count)
        return;
    statistics[lane] = {};
    if (!PhotonConfiguration(frame, controls))
    {
        PhotonFail(error, ConfigurationError);
        return;
    }
    uint pixel = controls.First + lane;
    if (pixel >= frame.Width * frame.Height)
        return;
    SceneGeometryBuffers geometry{
        {nodes, references, triangles, spheres, frame.NodeCount, frame.TriangleCount, frame.Acceleration},
        instances,
        instanceNodes,
        traversalError,
        frame.InstanceNodeCount};
    PhotonScene scene{geometry, materials, textures, texels, emitters, &frame, &controls, error};
    uint x = pixel % frame.Width + frame.TileX, y = pixel / frame.Width + frame.TileY;
    SampleContext random =
        MakeSampleContext(x, y, frame.FullWidth, controls.Sample, frame.TotalSamples, frame.SamplingSequence, frame.SamplingSeed);
    CameraEndpointFrame camera{float3(frame.CameraPosition),
                               float3(frame.CameraForward),
                               float3(frame.CameraRight),
                               float3(frame.CameraUp),
                               frame.HalfWidth,
                               frame.HalfHeight,
                               frame.LensRadius,
                               frame.FocusDistance};
    float2 jitter = Sample2D(random, RasterDimension);
    CameraEndpointSample start = SampleCameraEndpoint(
        camera, CameraPixelRaster(x, y, frame.FullWidth, frame.FullHeight, jitter),
        Sample2D(random, LensKey));
    ThreadStatistics counters{};
    float3 value = float3(0);
    if (start.Flags & CameraEndpointValid)
        value = PhotonCameraPath(scene, {start.Origin, start.Direction, RayEpsilon, MaximumRayDistance},
                                 random, records, heads, counters);
    statistics[lane] = counters;
    if (!PhotonFinite(value))
    {
        PhotonFail(error);
        return;
    }
    float4 moments = accumulation[pixel];
    uint count = counts[pixel];
    AccumulateFilmSample(moments, count, value);
    if (!all(isfinite(moments)))
    {
        PhotonFail(error);
        return;
    }
    accumulation[pixel] = moments;
    counts[pixel] = count;
    targets[pixel] = frame.TotalSamples;
}
