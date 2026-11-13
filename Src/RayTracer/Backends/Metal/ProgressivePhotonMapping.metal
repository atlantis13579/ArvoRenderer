// Hachisuka and Jensen (2009), "Stochastic Progressive Photon Mapping".
// Independent Metal implementation: resample camera visible points, emit bounded
// photon batches, gather a uniform disk, then update persistent N/R/tau ONCE per
// complete light pass. Attempted emissions, including misses, normalize density.
// https://pbr-book.org/3ed-2018/Light_Transport_III_Bidirectional_Methods/Stochastic_Progressive_Photon_Mapping
// Metal's scaled two-float expansion retains about 48 significand bits and a
// separate exponent. It is not binary64; fast math must remain disabled.
// Interior collision photons retain a separate shrinking three-dimensional kernel.
#include <metal_stdlib>
using namespace metal;
constant float Pi = 3.14159265358979323846f, PiResidual = -8.742278000372485e-8f;
constant float RayEpsilon = 1e-4f, MaximumRayDistance = 1e30f, BasisParallelThreshold = 0.999f;
constant uint ImportanceBit = 1, NeeBit = 2, RouletteBit = 4, MisShift = 3, MisMask = 3;
constant uint MaximumPpmReflections = 32, InvalidPpmIndex = 0xffffffffu;
constant uint EmptyPhoton = 0, DirectPhoton = 1, CausticPhoton = 2, IndirectPhoton = 3;
constant uint VolumePhotonFlag = 0x80000000u, PhotonClassMask = ~VolumePhotonFlag;
constant uint VisibleSurface = 1, VisibleVolume = 2;
constant int VolumeDimensions = 3, SurfaceDimensions = 2;
constant uint CausticsEnabled = 1, IndirectEnabled = 2, NumericalError = 1, ConfigurationError = 2;
constant float NormalAgreement = 0.9f, PlaneRatio = 0.1f, CellLimit = 1048576.0f;
constant uint AvalancheMultiplierA = 0x7feb352du, AvalancheMultiplierB = 0x846ca68bu;
constant uint RandomMantissaMask = 0x00ffffffu;
constant float RandomMantissaRange = 16777216.0f;
constant uint PixelSeedMultiplier = 9781, SampleSeedMultiplier = 6271, SecondDimensionScramble = 0x68bc21ebu;
constant uint LightKey = 0x4d50504du, PassKey = 0x50415353u, DepthKey = 0x9e3779b9u;
constant uint LensKey = 0x4c454e53u, SelectKey = 0x53454c45u, PositionKey = 0x504f534eu;
constant uint DirectionKey = 0x44495245u, ScatterKey = 0x53434154u, LobeKey = 0x4c4f4245u,
              RouletteKey = 0x52524f55u;
constant uint VisibleKey = 0x56495349u, InteriorKey = 0x494e5445u;
constant uint RasterDimension = 0, LightChoiceDimension = 0, LightPositionDimension = 1,
              LightDirectionDimension = 3;
constant uint LightEndpointDimensions = 5, LightVertexDimensions = 4, LightLobeOffset = 2,
              LightRouletteOffset = 3;
#include "Hash.metal"
#include "Sequence.metal"
#include "SurfaceTypes.metal"
#include "ShadingTypes.metal"
#include "ExtendedFloat.metal"
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
struct PpmControls
{
    packed_float3 Center;
    float WorldRadius;
    uint Sources, PathsPerPass, SlotsPerPath, Buckets;
    uint Seed, Classes, First, Count;
    float InitialRadiusScale, Alpha;
    uint Pass, LightBegin;
    uint LightCount, Reserved[3];
};
struct PpmPhoton
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
struct PpmVisible
{
    packed_float3 Position;
    uint Active;
    packed_float3 Normal;
    uint Front;
    packed_float3 GeometryNormal;
    uint Depth;
    packed_float3 View;
    uint Material;
    packed_float3 Color;
    uint Reserved;
    packed_float3 Beta;
    uint Padding;
};
struct PpmWideVector
{
    WideMath::Number X, Y, Z;
};
struct PpmPixel
{
    WideMath::Number RadiusSquared, Count;
    PpmWideVector Flux, Direct, Incoming;
    uint Matched, Reserved[3];
    WideMath::Number VolumeRadiusSquared, VolumeCount;
    PpmWideVector VolumeFlux, VolumeIncoming;
    uint VolumeMatched, VolumeReserved[3];
};
struct ThreadStatistics
{
    uint2 Primary, Shadow, Reflection, Tests;
};
static_assert(sizeof(FrameConstants) == 240 && sizeof(PpmControls) == 80 && sizeof(PpmPhoton) == 64 &&
                  sizeof(PpmVisible) == 96 && sizeof(PpmPixel) == 336 && sizeof(ThreadStatistics) == 32,
              "Metal PPM buffer layout mismatch");
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
struct PpmScene
{
    SceneGeometryBuffers Geometry;
    device const PackedMaterial* Materials;
    device const PackedTextureDescriptor* Textures;
    device const uint* Texels;
    device const PackedEmitter* Emitters;
    constant FrameConstants* Frame;
    constant PpmControls* Controls;
    device atomic_uint* Error;
};
struct PpmCell
{
    int X, Y, Z;
};
struct PpmEndpoint
{
    HitData Hit;
    float3 Position;
    uint Source;
    bool Surface;
};
void PpmFail(device atomic_uint* error, uint bits = NumericalError)
{
    atomic_fetch_or_explicit(error, bits, memory_order_relaxed);
}
bool PpmSpectrum(float3 value)
{
    return all(isfinite(value)) && value.x >= 0 && value.y >= 0 && value.z >= 0;
}
bool PpmPositive(float3 value)
{
    return value.x > 0 || value.y > 0 || value.z > 0;
}
bool PpmFinite(WideMath::Number value)
{
    return isfinite(value.High) && isfinite(value.Low);
}
bool PpmFinite(PpmWideVector value)
{
    return PpmFinite(value.X) && PpmFinite(value.Y) && PpmFinite(value.Z);
}
float PpmFloat(WideMath::Number value)
{
    return as_type<float>(WideMath::Encode(value));
}
float3 PpmFloat(PpmWideVector value)
{
    return float3(PpmFloat(value.X), PpmFloat(value.Y), PpmFloat(value.Z));
}
WideMath::Number PpmUnsigned(uint value)
{
    using namespace WideMath;
    const uint HalfWordBits = 16, LowWordMask = 0xffffu;
    return Add(Shift(FromFloat(float(value >> HalfWordBits)), HalfWordBits),
               FromFloat(float(value & LowWordMask)));
}
PpmWideVector PpmWide(float3 value)
{
    return {WideMath::FromFloat(value.x), WideMath::FromFloat(value.y), WideMath::FromFloat(value.z)};
}
PpmWideVector PpmAdd(PpmWideVector a, PpmWideVector b)
{
    return {WideMath::Add(a.X, b.X), WideMath::Add(a.Y, b.Y), WideMath::Add(a.Z, b.Z)};
}
PpmWideVector PpmProduct(PpmWideVector a, PpmWideVector b)
{
    return {WideMath::Multiply(a.X, b.X), WideMath::Multiply(a.Y, b.Y), WideMath::Multiply(a.Z, b.Z)};
}
PpmWideVector PpmScale(PpmWideVector value, WideMath::Number scale)
{
    return {WideMath::Multiply(value.X, scale), WideMath::Multiply(value.Y, scale),
            WideMath::Multiply(value.Z, scale)};
}
PpmWideVector PpmDifference(float3 a, float3 b)
{
    using namespace WideMath;
    return {Subtract(FromFloat(a.x), FromFloat(b.x)), Subtract(FromFloat(a.y), FromFloat(b.y)),
            Subtract(FromFloat(a.z), FromFloat(b.z))};
}
WideMath::Number PpmDot(PpmWideVector a, PpmWideVector b)
{
    using namespace WideMath;
    return Add(Add(Multiply(a.X, b.X), Multiply(a.Y, b.Y)), Multiply(a.Z, b.Z));
}
WideMath::Number PpmInitialRadius(constant PpmControls& controls)
{
    return WideMath::Multiply(WideMath::FromFloat(controls.WorldRadius),
                              WideMath::FromFloat(controls.InitialRadiusScale));
}
WideMath::Number PpmRoot(WideMath::Number value, int dimension)
{
    using namespace WideMath;
    if (!(value.High > 0)) return {};
    int exponent = value.Exponent / dimension;
    if (value.Exponent < 0 && value.Exponent % dimension != 0) --exponent;
    int remainder = value.Exponent - exponent * dimension;
    Number root = Shift(FromFloat(pow(ldexp(value.High, remainder), 1.0f / float(dimension))),
                        exponent);
    for (int iteration = 0; iteration < 2; ++iteration)
        if (dimension == SurfaceDimensions)
            root = Divide(Add(root, Divide(value, root)), FromFloat(float(SurfaceDimensions)));
        else
            root = Divide(Add(Multiply(FromFloat(float(dimension - 1)), root),
                              Divide(value, Multiply(root, root))), FromFloat(float(dimension)));
    return root;
}
PpmPixel PpmAdvance(PpmPixel state, float alpha)
{
    using namespace WideMath;
    if (state.Matched)
    {
        Number arrivals = PpmUnsigned(state.Matched);
        Number nextCount = Add(state.Count, Multiply(FromFloat(alpha), arrivals));
        Number ratio = Divide(nextCount, Add(state.Count, arrivals));
        state.Count = nextCount;
        state.RadiusSquared = Multiply(state.RadiusSquared, ratio);
        state.Flux = PpmScale(PpmAdd(state.Flux, state.Incoming), ratio);
    }
    if (state.VolumeMatched)
    {
        Number arrivals = PpmUnsigned(state.VolumeMatched);
        Number nextCount = Add(state.VolumeCount, Multiply(FromFloat(alpha), arrivals));
        Number ratio = Divide(nextCount, Add(state.VolumeCount, arrivals));
        Number radiusRatio = PpmRoot(ratio, VolumeDimensions);
        state.VolumeCount = nextCount;
        state.VolumeRadiusSquared = Multiply(state.VolumeRadiusSquared,
                                             Multiply(radiusRatio, radiusRatio));
        state.VolumeFlux = PpmScale(PpmAdd(state.VolumeFlux, state.VolumeIncoming), ratio);
    }
    return state;
}
PpmWideVector PpmResolve(PpmPixel state, uint passes, uint pathsPerPass)
{
    using namespace WideMath;
    if (!passes)
        return {};
    Number inversePasses = Divide(FromFloat(1), PpmUnsigned(passes));
    PpmWideVector estimate = PpmScale(state.Direct, inversePasses);
    if (state.RadiusSquared.High > 0 && pathsPerPass)
    {
        Number denominator =
            Multiply(Multiply(Add(FromFloat(Pi), FromFloat(PiResidual)), state.RadiusSquared),
                     Multiply(PpmUnsigned(passes), PpmUnsigned(pathsPerPass)));
        estimate = PpmAdd(estimate, PpmScale(state.Flux, Divide(FromFloat(1), denominator)));
    }
    if (state.VolumeRadiusSquared.High > 0 && pathsPerPass)
    {
        Number radius = PpmRoot(state.VolumeRadiusSquared, SurfaceDimensions);
        Number sphereMeasure = Divide(FromFloat(4), FromFloat(float(VolumeDimensions)));
        Number denominator = Multiply(
            Multiply(Multiply(Add(FromFloat(Pi), FromFloat(PiResidual)), sphereMeasure),
                     Multiply(state.VolumeRadiusSquared, radius)),
            Multiply(PpmUnsigned(passes), PpmUnsigned(pathsPerPass)));
        estimate = PpmAdd(estimate, PpmScale(state.VolumeFlux, Divide(FromFloat(1), denominator)));
    }
    return estimate;
}
bool PpmAxis(WideMath::Number coordinate, thread int& cell)
{
    using namespace WideMath;
    if (!PpmFinite(coordinate) || Compare(Absolute(coordinate), FromFloat(CellLimit)) >= 0)
        return false;
    cell = int(floor(PpmFloat(coordinate)));
    if (Compare(coordinate, FromFloat(float(cell))) < 0)
        --cell;
    else if (Compare(coordinate, FromFloat(float(cell + 1))) >= 0)
        ++cell;
    return true;
}
bool PpmCellAt(float3 position, constant PpmControls& controls, device atomic_uint* error,
               thread PpmCell& cell)
{
    using namespace WideMath;
    Number radius = PpmInitialRadius(controls);
    if (!(radius.High > 0))
        return false;
    PpmWideVector offset = PpmDifference(position, float3(controls.Center));
    if (PpmAxis(Divide(offset.X, radius), cell.X) && PpmAxis(Divide(offset.Y, radius), cell.Y) &&
        PpmAxis(Divide(offset.Z, radius), cell.Z))
        return true;
    PpmFail(error);
    return false;
}
uint PpmBucket(PpmCell cell, uint count)
{
    const uint XDomain = 0x243f6a88u, YDomain = 0x85a308d3u, ZDomain = 0x13198a2eu;
    return Hash(Hash(uint(cell.X) ^ XDomain) ^ Hash(uint(cell.Y) ^ YDomain) ^ Hash(uint(cell.Z) ^ ZDomain)) &
           (count - 1);
}
bool PpmSameCell(PpmCell a, PpmCell b)
{
    return a.X == b.X && a.Y == b.Y && a.Z == b.Z;
}
uint PpmCategory(uint depth, bool deltaPrefix)
{
    return depth == 0 ? DirectPhoton : deltaPrefix ? CausticPhoton : IndirectPhoton;
}
bool PpmAccept(PpmPhoton photon, uint cameraDepth, constant FrameConstants& frame,
               constant PpmControls& controls)
{
    if (cameraDepth > frame.MaxReflectionBounces || photon.Depth > frame.MaxReflectionBounces - cameraDepth)
        return false;
    uint category = photon.Kind & PhotonClassMask;
    if (category == DirectPhoton)
        return (photon.Kind & VolumePhotonFlag) != 0 || (frame.TransportFlags & NeeBit) == 0;
    if (category == CausticPhoton)
        return (controls.Classes & CausticsEnabled) != 0;
    return category == IndirectPhoton && (controls.Classes & IndirectEnabled) != 0;
}
bool PpmMergeable(PpmWideVector offset, WideMath::Number radiusSquared, float3 normal, float agreement)
{
    using namespace WideMath;
    if (!(radiusSquared.High > 0) || agreement < NormalAgreement ||
        Compare(PpmDot(offset, offset), radiusSquared) > 0)
        return false;
    Number plane = PpmDot(offset, PpmWide(normal));
    return Compare(Multiply(plane, plane),
                   Multiply(radiusSquared, Multiply(FromFloat(PlaneRatio), FromFloat(PlaneRatio)))) <= 0;
}
bool PpmConfiguration(constant FrameConstants& frame, constant PpmControls& controls)
{
    return frame.MaxReflectionBounces <= MaximumPpmReflections &&
           controls.SlotsPerPath == frame.MaxReflectionBounces + 1 && controls.PathsPerPass > 0 &&
           controls.Buckets > 0 && (controls.Buckets & (controls.Buckets - 1)) == 0 &&
           !frame.AdaptiveEnabled && ((frame.TransportFlags >> MisShift) & MisMask) == 0 &&
           frame.ShadowStrength == 1 && controls.WorldRadius >= 0 && isfinite(controls.WorldRadius) &&
           controls.InitialRadiusScale > 0 && isfinite(controls.InitialRadiusScale) && controls.Alpha > 0 &&
           controls.Alpha < 1;
}
float PpmLightRandom(constant PpmControls& controls, uint path, uint dimension)
{
    return Random01(Hash(controls.Seed ^ LightKey) ^ Hash(controls.Pass ^ PassKey) ^ Hash(path) ^
                    Hash(dimension ^ DepthKey));
}
float2 PpmLightPair(constant PpmControls& controls, uint path, uint dimension)
{
    return float2(PpmLightRandom(controls, path, dimension), PpmLightRandom(controls, path, dimension + 1));
}
float PpmDecode(uint component)
{
    const float ByteMaximum = 255, Threshold = 0.04045f, Slope = 12.92f, Offset = 0.055f, Scale = 1.055f,
                Power = 2.4f;
    float value = component / ByteMaximum;
    return value <= Threshold ? value / Slope : pow((value + Offset) / Scale, Power);
}
float3 PpmTexel(PpmScene scene, PackedTextureDescriptor image, int column, int row)
{
    column = (column % int(image.Width) + int(image.Width)) % int(image.Width);
    row = (row % int(image.Height) + int(image.Height)) % int(image.Height);
    uint pixel = scene.Texels[image.OffsetTexels + uint(row) * image.Width + uint(column)];
    const uint ByteMask = 0xff, GreenShift = 8, BlueShift = 16;
    return float3(PpmDecode(pixel & ByteMask), PpmDecode((pixel >> GreenShift) & ByteMask),
                  PpmDecode((pixel >> BlueShift) & ByteMask));
}
float3 PpmColor(PpmScene scene, HitData hit)
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
    return hit.Color *
           mix(mix(PpmTexel(scene, image, column, row), PpmTexel(scene, image, column + 1, row), fract(x)),
               mix(PpmTexel(scene, image, column, row + 1), PpmTexel(scene, image, column + 1, row + 1),
                   fract(x)),
               fract(y));
}
bool PpmRectangle(PackedEmitter source, RayData ray, thread float& nearest)
{
    if (source.Kind != PackedEmitterRectangle)
        return false;
    float3 normal = float3(source.Direction);
    float projection = dot(ray.Direction, normal);
    if (projection == 0)
        return false;
    float distance = dot(float3(source.Origin) - ray.Origin, normal) / projection;
    if (!(distance > ray.MinimumDistance && distance < ray.MaximumDistance && distance <= nearest))
        return false;
    float3 helper = abs(normal.y) < BasisParallelThreshold ? float3(0, 1, 0) : float3(1, 0, 0);
    float3 right = normalize(cross(helper, normal)), up = cross(normal, right);
    float3 relative = ray.Origin + distance * ray.Direction - float3(source.Origin);
    if (abs(dot(relative, right)) > source.Width / 2 || abs(dot(relative, up)) > source.Height / 2)
        return false;
    nearest = distance;
    return true;
}
PpmEndpoint PpmTrace(PpmScene scene, RayData ray, thread ThreadStatistics& statistics)
{
    PpmEndpoint endpoint{};
    endpoint.Source = InvalidPpmIndex;
    endpoint.Hit.Distance = ray.MaximumDistance;
    endpoint.Surface = IntersectScene(scene.Geometry, ray, false, endpoint.Hit, statistics.Tests);
    bool proxy = endpoint.Surface && scene.Materials[endpoint.Hit.MaterialIndex].SampledEmitter != 0;
    float nearest = endpoint.Hit.Distance + (proxy ? RayEpsilon : 0);
    for (uint source = 0; source < scene.Controls->Sources; ++source)
        if (PpmRectangle(scene.Emitters[source], ray, nearest))
        {
            endpoint.Source = source;
            endpoint.Surface = false;
        }
    if (endpoint.Source == InvalidPpmIndex)
        for (uint source = 0; source < scene.Controls->Sources; ++source)
        {
            PackedEmitter light = scene.Emitters[source];
            if (endpoint.Surface ? light.Primitive != InvalidPrimitiveIndex &&
                                       light.Primitive == endpoint.Hit.PrimitiveIndex
                                 : light.Kind == PackedEmitterEnvironment)
            {
                endpoint.Source = source;
                break;
            }
        }
    endpoint.Position = ray.Origin + ray.Direction * (endpoint.Surface ? endpoint.Hit.Distance : nearest);
    return endpoint;
}
float3 PpmRadiance(PackedEmitter source, float3 outward)
{
    if (source.Kind == PackedEmitterEnvironment)
        return mix(float3(source.BackgroundBottom), float3(source.BackgroundTop), (1 - outward.y) / 2);
    if (source.Kind == PackedEmitterRectangle)
        return dot(float3(source.Direction), outward) > 0
                   ? float3(source.Strength) / (source.Width * source.Height)
                   : float3(0);
    return float3(source.Strength);
}
SurfaceSample PpmMirror(SurfaceBsdf surface)
{
    SurfaceSample sample{};
    sample.Direction = SurfaceWorld(surface, float3(-surface.View.x, -surface.View.y, surface.View.z));
    if (!SurfaceDirectionValid(surface, sample.Direction, surface.View.z))
        return {};
    sample.Weight = float3(SurfaceAdjointCorrection(surface, sample.Direction));
    sample.Pdf = sample.ReversePdf = sample.RelativeIndex = 1;
    sample.Event = ScatteringReflection | ScatteringDelta;
    return sample;
}
float3 PpmDirectLight(PpmScene scene, PpmEndpoint hit, SurfaceBsdf surface, SampleContext random, uint domain,
                      bool filterTransmission, thread ThreadStatistics& statistics)
{
    using namespace WideMath;
    EmissionDomain emissionDomain{float3(scene.Controls->Center), scene.Controls->WorldRadius};
    EmitterEndpoint source = SampleEmitterEndpoint(
        scene.Emitters, scene.Controls->Sources, emissionDomain, Sample2D(random, domain ^ SelectKey).x,
        Sample2D(random, domain ^ PositionKey), Sample2D(random, domain ^ DirectionKey));
    EmissionSample emission = source.Emission;
    if (!(emission.Flags & EmissionValid))
        return float3(0);
    bool infinite = (emission.Flags & EmissionInfinite) != 0,
         pointSource = (emission.Flags & EmissionDeltaPosition) != 0;
    float3 incoming = -emission.Direction;
    Number geometry = FromFloat(1);
    if (!infinite)
    {
        float3 difference = emission.Origin - hit.Position;
        float distance = length(difference);
        if (!(distance > 0) || !isfinite(distance))
            return float3(0);
        incoming = difference / distance;
        geometry = Divide(FromFloat(pointSource ? 1 : abs(dot(emission.Normal, incoming))),
                          Multiply(FromFloat(distance), FromFloat(distance)));
    }
    bool transmitted = filterTransmission &&
        (hit.Hit.FrontFace ? dot(incoming, hit.Hit.GeometricNormal) < 0
                           : dot(incoming, hit.Hit.GeometricNormal) > 0);
    if (transmitted && !pointSource && !(emission.Flags & EmissionDeltaDirection))
        return float3(0);
    Number density = Multiply(FromFloat(source.SelectionProbability),
                              FromFloat(infinite ? emission.DirectionPdf : emission.PositionPdf));
    if (!(density.High > 0))
        return float3(0);
    SurfaceEvaluation evaluation = EvaluateSurface(surface, incoming);
    PpmWideVector product = PpmProduct(PpmWide(evaluation.Value),
                                       PpmWide(PpmRadiance(scene.Emitters[source.EmitterIndex], -incoming)));
    float3 estimate = PpmFloat(PpmScale(
        product, Divide(Multiply(geometry, FromFloat(abs(dot(surface.Normal, incoming)))), density)));
    if (transmitted && hit.Hit.FrontFace && pointSource)
        estimate *= exp(float3(-length(emission.Origin - hit.Position) / surface.Material.Optical.w));
    if (!PpmSpectrum(estimate))
    {
        PpmFail(scene.Error);
        return float3(0);
    }
    if (!PpmPositive(estimate))
        return float3(0);
    RayData shadow{SurfaceOffset(surface, hit.Position, incoming), incoming, RayEpsilon, MaximumRayDistance};
    if (!infinite)
    {
        float3 segment = emission.Origin - shadow.Origin;
        float distance = length(segment);
        if (!(distance > 2 * RayEpsilon))
            return float3(0);
        shadow.Direction = segment / distance;
        shadow.MaximumDistance = distance - RayEpsilon;
    }
    IncrementCounter(statistics.Shadow);
    HitData blocker{};
    blocker.Distance = shadow.MaximumDistance;
    if (IntersectScene(scene.Geometry, shadow, true, blocker, statistics.Tests))
        return float3(0);
    float nearest = shadow.MaximumDistance;
    for (uint sourceIndex = 0; sourceIndex < scene.Controls->Sources; ++sourceIndex)
        if (PpmRectangle(scene.Emitters[sourceIndex], shadow, nearest))
            return float3(0);
    return estimate;
}

bool PpmSelectVisible(thread PpmVisible& visible, thread uint& candidates, float variate)
{
    ++candidates;
    if (visible.Active)
        visible.Beta = packed_float3(float3(visible.Beta) *
                                     (float(candidates) / float(candidates - 1)));
    return candidates == 1 || variate < 1.0f / float(candidates);
}
void PpmCameraPath(PpmScene scene, uint pixel, device PpmVisible* visiblePoints, device PpmPixel* pixels,
                   thread ThreadStatistics& statistics)
{
    constant FrameConstants& frame = *scene.Frame;
    constant PpmControls& controls = *scene.Controls;
    PpmPixel state = pixels[pixel];
    state.Matched = 0;
    state.Incoming = {};
    state.VolumeMatched = 0;
    state.VolumeIncoming = {};
    PpmVisible visible{};
    uint x = pixel % frame.Width + frame.TileX, y = pixel / frame.Width + frame.TileY;
    SampleContext random =
        MakeSampleContext(x, y, frame.FullWidth, controls.Pass, frame.TotalSamples, frame.SamplingSequence, frame.SamplingSeed);
    CameraEndpointFrame camera{float3(frame.CameraPosition),
                               float3(frame.CameraForward),
                               float3(frame.CameraRight),
                               float3(frame.CameraUp),
                               frame.HalfWidth,
                               frame.HalfHeight,
                               frame.LensRadius,
                               frame.FocusDistance};
    CameraEndpointSample start = SampleCameraEndpoint(
        camera, CameraPixelRaster(x, y, frame.FullWidth, frame.FullHeight, Sample2D(random, RasterDimension)),
        Sample2D(random, LensKey));
    if (!(start.Flags & CameraEndpointValid))
    {
        PpmFail(scene.Error);
        return;
    }
    RayData ray{start.Origin, start.Direction, RayEpsilon, MaximumRayDistance};
    float3 beta = float3(1);
    float etaScale = 1;
    PpmWideVector direct{};
    uint interiorMaterial = InvalidPpmIndex, candidates = 0;
    for (uint depth = 0; depth <= frame.MaxReflectionBounces + 1; ++depth)
    {
        if (!PpmSpectrum(beta) || !isfinite(etaScale))
        {
            PpmFail(scene.Error);
            break;
        }
        if (!PpmPositive(beta))
            break;
        if (depth == 0)
            IncrementCounter(statistics.Primary);
        else
            IncrementCounter(statistics.Reflection);
        PpmEndpoint hit = PpmTrace(scene, ray, statistics);
        uint domain = DepthKey * (depth + 1);
        if (hit.Surface && hit.Hit.MaterialIndex >= frame.MaterialCount)
        {
            PpmFail(scene.Error, ConfigurationError);
            break;
        }
        if (depth == 0 && hit.Surface && !hit.Hit.FrontFace &&
            scene.Materials[hit.Hit.MaterialIndex].Optical.w > 0)
            interiorMaterial = hit.Hit.MaterialIndex;
        if (interiorMaterial != InvalidPpmIndex)
        {
            PackedMaterial interior = scene.Materials[interiorMaterial];
            bool finiteSource = hit.Source != InvalidPpmIndex &&
                scene.Emitters[hit.Source].Kind == PackedEmitterRectangle;
            float boundary = hit.Surface || finiteSource
                ? length(hit.Position - ray.Origin) : ray.MaximumDistance;
            SubsurfaceFlight flight = SubsurfaceFreeFlight(interior.Optical.w,
                interior.AlbedoReflectivity.xyz * interior.ConductorExtinction.w,
                ray.Direction, boundary, Sample2D(random, domain ^ InteriorKey).x);
            beta *= flight.Weight;
            if (!PpmSpectrum(beta))
            {
                PpmFail(scene.Error);
                break;
            }
            if (!PpmPositive(beta)) break;
            if (flight.Scattered)
            {
                if (depth > frame.MaxReflectionBounces) break;
                if (PpmSelectVisible(visible, candidates, Sample2D(random, domain ^ VisibleKey).x))
                {
                    visible.Position = packed_float3(ray.Origin + flight.Distance * ray.Direction);
                    visible.Active = VisibleVolume;
                    visible.View = packed_float3(ray.Direction);
                    visible.Material = interiorMaterial;
                    visible.Depth = depth;
                    visible.Beta = packed_float3(beta * float(candidates));
                }
                break;
            }
        }
        if (hit.Source != InvalidPpmIndex)
            direct =
                PpmAdd(direct, PpmProduct(PpmWide(beta),
                                          PpmWide(PpmRadiance(scene.Emitters[hit.Source], -ray.Direction))));
        if (!hit.Surface || depth > frame.MaxReflectionBounces)
            break;
        PackedMaterial material = scene.Materials[hit.Hit.MaterialIndex];
        bool legacy = material.Model == MaterialLegacy;
        float mirror = legacy ? clamp(material.AlbedoReflectivity.w, 0.0f, 1.0f) : 0;
        if (legacy)
            material.Model = MaterialDiffuse;
        float3 color = PpmColor(scene, hit.Hit);
        SurfaceBsdf surface = MakeSurfaceBsdf(material, hit.Hit, -ray.Direction, color, frame.TransportFlags);
        bool subsurfaceTransmission = material.Optical.w > 0 && !SurfaceDelta(surface);
        bool continuous = !SurfaceDelta(surface) && mirror < 1;
        if (continuous && (frame.TransportFlags & NeeBit))
            direct = PpmAdd(
                direct, PpmScale(PpmProduct(PpmWide(beta), PpmWide(PpmDirectLight(scene, hit, surface, random,
                                                                                  domain, subsurfaceTransmission,
                                                                                  statistics))),
                                 WideMath::FromFloat(1 - mirror)));
        // Mixture selection already accounts for its probability. The diffuse
        // visible point keeps beta unchanged; direct NEE integrates both choices.
        bool legacyReflection = legacy && Sample2D(random, domain ^ LobeKey).x < mirror;
        if (continuous && !legacyReflection)
        {
            if (PpmSelectVisible(visible, candidates, Sample2D(random, domain ^ VisibleKey).x))
                visible = {packed_float3(hit.Position),
                           VisibleSurface,
                           packed_float3(hit.Hit.Normal),
                           hit.Hit.FrontFace,
                           packed_float3(hit.Hit.GeometricNormal),
                           depth,
                           packed_float3(-ray.Direction),
                           hit.Hit.MaterialIndex,
                           packed_float3(color),
                           0,
                           packed_float3(beta * float(candidates)),
                           0};
            if (!subsurfaceTransmission) break;
        }
        SurfaceSample event;
        if (legacyReflection)
            event = PpmMirror(surface);
        else
            event = SampleSurface(surface, Sample2D(random, domain ^ ScatterKey),
                                  Sample2D(random, domain ^ LobeKey).x);
        if (!event.Event || !(event.Pdf > 0))
            break;
        if (subsurfaceTransmission && !(event.Event & ScatteringTransmission)) break;
        beta *= event.Weight;
        etaScale *= event.RelativeIndex * event.RelativeIndex;
        if (!PpmSpectrum(beta) || !isfinite(etaScale))
        {
            PpmFail(scene.Error);
            break;
        }
        if ((frame.TransportFlags & RouletteBit) && depth + 1 >= frame.RouletteStartDepth)
        {
            float survival = RouletteProbability(beta * etaScale);
            if (Sample2D(random, domain ^ RouletteKey).x >= survival)
                break;
            beta /= survival;
        }
        ray = {SurfaceOffset(surface, hit.Position, event.Direction), event.Direction, RayEpsilon,
               MaximumRayDistance};
        if (material.Optical.w > 0 && (event.Event & ScatteringTransmission))
            interiorMaterial = hit.Hit.FrontFace ? hit.Hit.MaterialIndex : InvalidPpmIndex;
    }
    state.Direct = PpmAdd(state.Direct, direct);
    if (!PpmFinite(state.Direct))
        PpmFail(scene.Error);
    pixels[pixel] = state;
    visiblePoints[pixel] = visible;
}
void PpmEmitPath(PpmScene scene, uint localPath, device PpmPhoton* photons,
                 thread ThreadStatistics& statistics)
{
    constant FrameConstants& frame = *scene.Frame;
    constant PpmControls& controls = *scene.Controls;
    uint path = controls.LightBegin + localPath, base = localPath * controls.SlotsPerPath;
    for (uint depth = 0; depth < controls.SlotsPerPath; ++depth)
        photons[base + depth] = {};
    EmissionDomain domain{float3(controls.Center), controls.WorldRadius};
    EmitterEndpoint source = SampleEmitterEndpoint(scene.Emitters, controls.Sources, domain,
                                                   PpmLightRandom(controls, path, LightChoiceDimension),
                                                   PpmLightPair(controls, path, LightPositionDimension),
                                                   PpmLightPair(controls, path, LightDirectionDimension));
    EmissionSample emission = source.Emission;
    if (!(emission.Flags & EmissionValid))
        return;
    RayData ray{emission.Origin +
                    emission.Normal *
                        (dot(emission.Normal, emission.Direction) >= 0 ? RayEpsilon : -RayEpsilon),
                emission.Direction, RayEpsilon, MaximumRayDistance};
    float3 flux = emission.Weight;
    bool deltaPrefix = true;
    uint interiorMaterial = InvalidPpmIndex;
    for (uint depth = 0; depth <= frame.MaxReflectionBounces; ++depth)
    {
        if (!PpmSpectrum(flux))
        {
            PpmFail(scene.Error);
            break;
        }
        if (!PpmPositive(flux))
            break;
        IncrementCounter(statistics.Reflection);
        PpmEndpoint hit = PpmTrace(scene, ray, statistics);
        uint dimension = LightEndpointDimensions + depth * LightVertexDimensions;
        if (hit.Surface && hit.Hit.MaterialIndex >= frame.MaterialCount)
        {
            PpmFail(scene.Error, ConfigurationError);
            break;
        }
        if (depth == 0 && hit.Surface && !hit.Hit.FrontFace &&
            scene.Materials[hit.Hit.MaterialIndex].Optical.w > 0)
            interiorMaterial = hit.Hit.MaterialIndex;
        if (interiorMaterial != InvalidPpmIndex)
        {
            PackedMaterial interior = scene.Materials[interiorMaterial];
            bool finiteSource = hit.Source != InvalidPpmIndex &&
                scene.Emitters[hit.Source].Kind == PackedEmitterRectangle;
            float boundary = hit.Surface || finiteSource
                ? length(hit.Position - ray.Origin) : ray.MaximumDistance;
            SubsurfaceFlight flight = SubsurfaceFreeFlight(interior.Optical.w,
                interior.AlbedoReflectivity.xyz * interior.ConductorExtinction.w,
                ray.Direction, boundary, PpmLightRandom(controls, path, dimension + LightLobeOffset));
            float3 collisionPosition = ray.Origin + flight.Distance * ray.Direction;
            if (flight.Scattered)
                photons[base + depth] = {packed_float3(collisionPosition), InvalidPpmIndex,
                    packed_float3(-ray.Direction), depth, packed_float3(flux),
                    PpmCategory(depth, deltaPrefix) | VolumePhotonFlag,
                    packed_float3(float3(0)), interiorMaterial};
            flux *= flight.Weight;
            if (!PpmSpectrum(flux))
            {
                PpmFail(scene.Error);
                break;
            }
            if (!PpmPositive(flux)) break;
            if (flight.Scattered)
            {
                if (depth == frame.MaxReflectionBounces) break;
                SubsurfaceDirection phase = SubsurfaceSampleDirection(ray.Direction,
                    interior.ConductorIndex.w, PpmLightPair(controls, path, dimension),
                    (frame.TransportFlags & ImportanceBit) != 0);
                if (!(phase.Pdf > 0) || !isfinite(phase.Weight))
                {
                    PpmFail(scene.Error);
                    break;
                }
                flux *= phase.Weight;
                if (!PpmSpectrum(flux))
                {
                    PpmFail(scene.Error);
                    break;
                }
                if ((frame.TransportFlags & RouletteBit) && depth + 1 >= frame.RouletteStartDepth)
                {
                    float survival = RouletteProbability(flux);
                    if (PpmLightRandom(controls, path, dimension + LightRouletteOffset) >= survival) break;
                    flux /= survival;
                }
                deltaPrefix = false;
                ray = {collisionPosition, phase.Direction, RayEpsilon, MaximumRayDistance};
                continue;
            }
        }
        if (!hit.Surface)
            break;
        PackedMaterial material = scene.Materials[hit.Hit.MaterialIndex];
        bool legacy = material.Model == MaterialLegacy;
        float mirror = legacy ? clamp(material.AlbedoReflectivity.w, 0.0f, 1.0f) : 0;
        if (legacy)
            material.Model = MaterialDiffuse;
        SurfaceBsdf surface = MakeSurfaceBsdf(material, hit.Hit, -ray.Direction, PpmColor(scene, hit.Hit),
                                              frame.TransportFlags, TransportImportance);
        if (!SurfaceDelta(surface) && mirror < 1)
            photons[base + depth] = {packed_float3(hit.Position),
                                     InvalidPpmIndex,
                                     packed_float3(-ray.Direction),
                                     depth,
                                     packed_float3(flux),
                                     PpmCategory(depth, deltaPrefix),
                                     packed_float3(hit.Hit.GeometricNormal),
                                     hit.Hit.MaterialIndex};
        if (depth == frame.MaxReflectionBounces)
            break;
        float lobe = PpmLightRandom(controls, path, dimension + LightLobeOffset);
        SurfaceSample event;
        if (legacy && lobe < mirror)
            event = PpmMirror(surface);
        else
            event = SampleSurface(surface, PpmLightPair(controls, path, dimension), lobe);
        if (!event.Event || !(event.Pdf > 0))
            break;
        flux *= event.Weight;
        deltaPrefix = deltaPrefix && (event.Event & ScatteringDelta) != 0;
        if (!PpmSpectrum(flux))
        {
            PpmFail(scene.Error);
            break;
        }
        if ((frame.TransportFlags & RouletteBit) && depth + 1 >= frame.RouletteStartDepth)
        {
            float survival = RouletteProbability(flux);
            if (PpmLightRandom(controls, path, dimension + LightRouletteOffset) >= survival)
                break;
            flux /= survival;
        }
        ray = {SurfaceOffset(surface, hit.Position, event.Direction), event.Direction, RayEpsilon,
               MaximumRayDistance};
        if (material.Optical.w > 0 && (event.Event & ScatteringTransmission))
            interiorMaterial = hit.Hit.FrontFace ? hit.Hit.MaterialIndex : InvalidPpmIndex;
    }
}
void PpmGatherPixel(uint pixel, PpmScene scene, device const PpmPhoton* photons,
                    device atomic_uint* heads, device const PpmVisible* visiblePoints,
                    device PpmPixel* pixels, thread ThreadStatistics& statistics)
{
    using namespace WideMath;
    constant FrameConstants& frame = *scene.Frame;
    constant PpmControls& controls = *scene.Controls;
    device atomic_uint* error = scene.Error;
    device const PackedMaterial* materials = scene.Materials;
    PpmVisible visible = visiblePoints[pixel];
    PpmPixel state = pixels[pixel];
    PpmCell query;
    if (!visible.Active || !(visible.Active == VisibleVolume
        ? state.VolumeRadiusSquared.High > 0 : state.RadiusSquared.High > 0) ||
        !PpmCellAt(float3(visible.Position), controls, error, query))
        return;
    SurfaceBsdf surface{};
    if (visible.Active == VisibleSurface)
    {
        HitData hit{};
        hit.Normal = float3(visible.Normal);
        hit.GeometricNormal = float3(visible.GeometryNormal);
        hit.FrontFace = visible.Front;
        PackedMaterial material = materials[visible.Material];
        if (material.Model == MaterialLegacy)
            material.Model = MaterialDiffuse;
        surface = MakeSurfaceBsdf(material, hit, float3(visible.View), float3(visible.Color),
                                  frame.TransportFlags);
    }
    for (int z = -1; z <= 1; ++z)
        for (int y = -1; y <= 1; ++y)
            for (int x = -1; x <= 1; ++x)
            {
                PpmCell neighbor{query.X + x, query.Y + y, query.Z + z};
                uint index = atomic_load_explicit(heads + PpmBucket(neighbor, controls.Buckets),
                                                  memory_order_relaxed),
                     visited = 0;
                while (index != InvalidPpmIndex)
                {
                    if (index >= controls.LightCount * controls.SlotsPerPath ||
                        ++visited > controls.LightCount * controls.SlotsPerPath)
                    {
                        PpmFail(error);
                        return;
                    }
                    PpmPhoton photon = photons[index];
                    index = photon.Next;
                    if (!PpmAccept(photon, visible.Depth, frame, controls))
                        continue;
                    bool volume = (photon.Kind & VolumePhotonFlag) != 0;
                    if (volume != (visible.Active == VisibleVolume)) continue;
                    PpmCell cell;
                    if (!PpmCellAt(float3(photon.Position), controls, error, cell) ||
                        !PpmSameCell(cell, neighbor))
                        continue;
                    PpmWideVector separation =
                        PpmDifference(float3(photon.Position), float3(visible.Position));
                    if (volume)
                    {
                        if (photon.Reserved != visible.Material ||
                            Compare(PpmDot(separation, separation), state.VolumeRadiusSquared) > 0)
                            continue;
                        float3 segment = float3(photon.Position) - float3(visible.Position);
                        float distance = length(segment);
                        if (distance > 2 * RayEpsilon)
                        {
                            IncrementCounter(statistics.Shadow);
                            RayData connection{float3(visible.Position), segment / distance,
                                               RayEpsilon, distance - RayEpsilon};
                            HitData blocker{};
                            blocker.Distance = connection.MaximumDistance;
                            if (IntersectScene(scene.Geometry, connection, true, blocker, statistics.Tests))
                                continue;
                        }
                        PackedMaterial interior = materials[visible.Material];
                        float phase = SubsurfacePhase(dot(float3(photon.Incoming), float3(visible.View)),
                            interior.ConductorIndex.w);
                        PpmWideVector contribution = PpmScale(
                            PpmProduct(PpmWide(float3(visible.Beta)), PpmWide(float3(photon.Flux))),
                            Multiply(FromFloat(phase), FromFloat(interior.Optical.w)));
                        if (!PpmFinite(contribution) || contribution.X.High < 0 ||
                            contribution.Y.High < 0 || contribution.Z.High < 0)
                        {
                            PpmFail(error);
                            return;
                        }
                        state.VolumeIncoming = PpmAdd(state.VolumeIncoming, contribution);
                        ++state.VolumeMatched;
                        continue;
                    }
                    float3 normal = float3(visible.GeometryNormal), incoming = float3(photon.Incoming);
                    if (materials[visible.Material].Optical.w > 0 &&
                        (visible.Front ? dot(incoming, normal) <= 0 : dot(incoming, normal) >= 0))
                        continue;
                    if (!PpmMergeable(separation, state.RadiusSquared, normal,
                                      dot(normal, float3(photon.Normal))))
                        continue;
                    float geometryProjection = abs(dot(normal, incoming));
                    if (!(geometryProjection > 0))
                        continue;
                    SurfaceEvaluation evaluation = EvaluateSurface(surface, incoming);
                    PpmWideVector contribution =
                        PpmProduct(PpmProduct(PpmWide(float3(visible.Beta)), PpmWide(float3(photon.Flux))),
                                   PpmWide(evaluation.Value));
                    contribution =
                        PpmScale(contribution, Divide(FromFloat(abs(dot(surface.Normal, incoming))),
                                                      FromFloat(geometryProjection)));
                    if (!PpmFinite(contribution) || contribution.X.High < 0 || contribution.Y.High < 0 ||
                        contribution.Z.High < 0)
                    {
                        PpmFail(error);
                        return;
                    }
                    state.Incoming = PpmAdd(state.Incoming, contribution);
                    ++state.Matched;
                }
            }
    // This pixel owns its entire expansion. Batch completion publishes it before
    // the next gather; the radius and effective count do not change until Update.
    pixels[pixel] = state;
}

kernel void ArvoPpmClear(device atomic_uint* heads [[buffer(17)]],
                         constant PpmControls& controls [[buffer(18)]],
                         device atomic_uint* error [[buffer(19)]], uint lane [[thread_position_in_grid]])
{
    if (lane >= controls.Count)
        return;
    uint index = controls.First + lane;
    if (index < controls.Buckets)
        atomic_store_explicit(heads + index, InvalidPpmIndex, memory_order_relaxed);
    if (index == 0)
        atomic_store_explicit(error, 0u, memory_order_relaxed);
}
kernel void ArvoPpmInitialize(constant FrameConstants& frame [[buffer(6)]],
                              device float4* accumulation [[buffer(7)]], device uint* counts [[buffer(8)]],
                              device uint* targets [[buffer(12)]],
                              constant PpmControls& controls [[buffer(18)]],
                              device atomic_uint* error [[buffer(19)]],
                              device PpmVisible* visible [[buffer(20)]],
                              device PpmPixel* pixels [[buffer(21)]], uint lane [[thread_position_in_grid]])
{
    if (lane >= controls.Count)
        return;
    if (!PpmConfiguration(frame, controls))
    {
        PpmFail(error, ConfigurationError);
        return;
    }
    uint pixel = controls.First + lane;
    if (pixel >= frame.Width * frame.Height)
        return;
    PpmPixel state{};
    WideMath::Number radius = PpmInitialRadius(controls);
    state.RadiusSquared = WideMath::Multiply(radius, radius);
    state.VolumeRadiusSquared = state.RadiusSquared;
    pixels[pixel] = state;
    visible[pixel] = {};
    accumulation[pixel] = float4(0);
    counts[pixel] = 0;
    targets[pixel] = frame.TotalSamples;
}
kernel void ArvoPpmCamera(
    device const PackedAccelerationNode* nodes [[buffer(0)]], device const uint* references [[buffer(1)]],
    device const PackedTriangle* triangles [[buffer(2)]], device const PackedSphere* spheres [[buffer(3)]],
    device const PackedMaterial* materials [[buffer(4)]], device const PackedEmitter* emitters [[buffer(5)]],
    constant FrameConstants& frame [[buffer(6)]], device ThreadStatistics* statistics [[buffer(9)]],
    device const PackedTextureDescriptor* textures [[buffer(10)]], device const uint* texels [[buffer(11)]],
    device const InstanceRecord* instances [[buffer(13)]],
    device const InstanceNode* instanceNodes [[buffer(14)]],
    device atomic_uint* traversalError [[buffer(15)]], constant PpmControls& controls [[buffer(18)]],
    device atomic_uint* error [[buffer(19)]], device PpmVisible* visible [[buffer(20)]],
    device PpmPixel* pixels [[buffer(21)]], uint lane [[thread_position_in_grid]])
{
    if (lane >= controls.Count)
        return;
    statistics[lane] = {};
    if (!PpmConfiguration(frame, controls))
    {
        PpmFail(error, ConfigurationError);
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
    PpmScene scene{geometry, materials, textures, texels, emitters, &frame, &controls, error};
    ThreadStatistics counters{};
    PpmCameraPath(scene, pixel, visible, pixels, counters);
    statistics[lane] = counters;
}
kernel void ArvoPpmEmit(
    device const PackedAccelerationNode* nodes [[buffer(0)]], device const uint* references [[buffer(1)]],
    device const PackedTriangle* triangles [[buffer(2)]], device const PackedSphere* spheres [[buffer(3)]],
    device const PackedMaterial* materials [[buffer(4)]], device const PackedEmitter* emitters [[buffer(5)]],
    constant FrameConstants& frame [[buffer(6)]], device ThreadStatistics* statistics [[buffer(9)]],
    device const PackedTextureDescriptor* textures [[buffer(10)]], device const uint* texels [[buffer(11)]],
    device const InstanceRecord* instances [[buffer(13)]],
    device const InstanceNode* instanceNodes [[buffer(14)]],
    device atomic_uint* traversalError [[buffer(15)]], device PpmPhoton* photons [[buffer(16)]],
    constant PpmControls& controls [[buffer(18)]], device atomic_uint* error [[buffer(19)]],
    uint lane [[thread_position_in_grid]])
{
    if (lane >= controls.Count)
        return;
    statistics[lane] = {};
    if (!PpmConfiguration(frame, controls))
    {
        PpmFail(error, ConfigurationError);
        return;
    }
    uint localPath = controls.First + lane;
    if (localPath >= controls.LightCount)
        return;
    SceneGeometryBuffers geometry{
        {nodes, references, triangles, spheres, frame.NodeCount, frame.TriangleCount, frame.Acceleration},
        instances,
        instanceNodes,
        traversalError,
        frame.InstanceNodeCount};
    PpmScene scene{geometry, materials, textures, texels, emitters, &frame, &controls, error};
    ThreadStatistics counters{};
    PpmEmitPath(scene, localPath, photons, counters);
    statistics[lane] = counters;
}
kernel void ArvoPpmIndex(device PpmPhoton* photons [[buffer(16)]], device atomic_uint* heads [[buffer(17)]],
                         constant PpmControls& controls [[buffer(18)]],
                         device atomic_uint* error [[buffer(19)]], uint lane [[thread_position_in_grid]])
{
    if (lane >= controls.Count)
        return;
    uint index = controls.First + lane;
    if (index >= controls.LightCount * controls.SlotsPerPath || photons[index].Kind == EmptyPhoton)
        return;
    PpmCell cell;
    if (!PpmCellAt(float3(photons[index].Position), controls, error, cell))
        return;
    photons[index].Next =
        atomic_exchange_explicit(heads + PpmBucket(cell, controls.Buckets), index, memory_order_relaxed);
}
kernel void ArvoPpmGather(device const PackedAccelerationNode* nodes [[buffer(0)]],
                          device const uint* references [[buffer(1)]],
                          device const PackedTriangle* triangles [[buffer(2)]],
                          device const PackedSphere* spheres [[buffer(3)]],
                          device const PackedMaterial* materials [[buffer(4)]],
                          device const PackedEmitter* emitters [[buffer(5)]],
                          constant FrameConstants& frame [[buffer(6)]],
                          device ThreadStatistics* statistics [[buffer(9)]],
                          device const PackedTextureDescriptor* textures [[buffer(10)]],
                          device const uint* texels [[buffer(11)]],
                          device const InstanceRecord* instances [[buffer(13)]],
                          device const InstanceNode* instanceNodes [[buffer(14)]],
                          device atomic_uint* traversalError [[buffer(15)]],
                          device const PpmPhoton* photons [[buffer(16)]],
                          device atomic_uint* heads [[buffer(17)]],
                          constant PpmControls& controls [[buffer(18)]],
                          device atomic_uint* error [[buffer(19)]],
                          device const PpmVisible* visible [[buffer(20)]],
                          device PpmPixel* pixels [[buffer(21)]], uint lane [[thread_position_in_grid]])
{
    if (lane >= controls.Count)
        return;
    if (!PpmConfiguration(frame, controls))
    {
        PpmFail(error, ConfigurationError);
        return;
    }
    uint pixel = controls.First + lane;
    if (pixel < frame.Width * frame.Height)
    {
        SceneGeometryBuffers geometry{
            {nodes, references, triangles, spheres, frame.NodeCount, frame.TriangleCount, frame.Acceleration},
            instances, instanceNodes, traversalError, frame.InstanceNodeCount};
        PpmScene scene{geometry, materials, textures, texels, emitters, &frame, &controls, error};
        ThreadStatistics counters{};
        PpmGatherPixel(pixel, scene, photons, heads, visible, pixels, counters);
        statistics[lane] = counters;
    }
}
kernel void ArvoPpmUpdate(constant FrameConstants& frame [[buffer(6)]],
                          device float4* accumulation [[buffer(7)]], device uint* counts [[buffer(8)]],
                          device uint* targets [[buffer(12)]], constant PpmControls& controls [[buffer(18)]],
                          device atomic_uint* error [[buffer(19)]], device PpmPixel* pixels [[buffer(21)]],
                          uint lane [[thread_position_in_grid]])
{
    if (lane >= controls.Count)
        return;
    if (!PpmConfiguration(frame, controls))
    {
        PpmFail(error, ConfigurationError);
        return;
    }
    uint pixel = controls.First + lane;
    if (pixel >= frame.Width * frame.Height)
        return;
    PpmPixel state = PpmAdvance(pixels[pixel], controls.Alpha);
    if (!PpmFinite(state.RadiusSquared) || !PpmFinite(state.Count) || !PpmFinite(state.Flux) ||
        !PpmFinite(state.Direct) || !PpmFinite(state.VolumeRadiusSquared) ||
        !PpmFinite(state.VolumeCount) || !PpmFinite(state.VolumeFlux))
    {
        PpmFail(error);
        return;
    }
    pixels[pixel] = state;
    float3 radiance = PpmFloat(PpmResolve(state, controls.Pass + 1, controls.PathsPerPass));
    if (!PpmSpectrum(radiance))
    {
        PpmFail(error);
        return;
    }
    // Persistent densities do not have independent per-pass variance estimates.
    accumulation[pixel] = float4(radiance, 0);
    counts[pixel] = controls.Pass + 1;
    targets[pixel] = frame.TotalSamples;
}
