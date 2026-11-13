// Independent CUDA two-pass photon mapping. Jensen (1996), "Global Illumination
// using Photon Maps"; PBRT 3e, section 16.2, derives the flux-density estimator.
// https://pbr-book.org/3ed-2018/Light_Transport_III_Bidirectional_Methods/Stochastic_Progressive_Photon_Mapping
// Fixed path slots feed a collision-resolving hash grid. Arrivals at continuous
// surfaces carry importance-mode flux; camera delta events continue. Direct,
// delta-prefix caustic and other indirect histories form disjoint path classes.
// NEE replaces direct photons. K=2*(1-d^2/R^2)/(pi*R^2) is normalized over a disk;
// divide by emitted paths, including misses, never by stored photon count.
// The radius and geometric filters introduce bias. More camera samples do not
// refine the light population, so adaptive stopping and connection MIS are off.
// Double query sums limit rounding from unordered GPU bucket insertion; native
// scheduling is not promised to be bitwise reproducible. No integrator is reused.
// Interior collisions occupy the same fixed path slots and use a normalized
// three-dimensional kernel with extinction compensation at camera collisions.
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cfloat>
#include "PhotonLayout.h"
#include "InstanceLayout.h"
#include "BsdfTypes.cuh"
#include "Fresnel.cu"
#include "Ggx.cu"
#include "Diffuse.cu"
#include "Conductor.cu"
#include "Dielectric.cu"
#include "LightSelection.cu"
#include "LightEmission.cu"
#include "SurfaceEmission.cu"
#include "EmitterEndpoint.cu"
#include "CameraEndpoint.cu"
#include "SubsurfaceScattering.cu"

namespace
{
namespace P = Arvo::RayTracing::CudaPhoton;
using Vec3 = P::Vector;
using Emitter = ArvoCudaEndpoint::PackedEmitter;
constexpr unsigned SphereBit = 0x80000000u, InvalidPrimitiveIndex = P::Invalid;
constexpr float Pi = CUDART_PI_F, RayEpsilon = 1.0e-4f, MaximumRayDistance = 1.0e30f;
constexpr float BasisParallelThreshold = 0.999f, MinimumDirectionLengthSquared = 1.0e-12f;
constexpr float MinimumTriangleDeterminant = 1.0e-8f;
constexpr unsigned ImportanceBit = 1, NeeBit = 2, RouletteBit = 4, MisShift = 3, MisMask = 3;
constexpr unsigned PixelSeedMultiplier = 9781, SampleSeedMultiplier = 6271;
constexpr unsigned SecondDimensionScramble = 0x68bc21ebu;
constexpr unsigned EmissionSourceDomain = 0x50485352u, EmissionPositionDomain = 0x5048504fu;
constexpr unsigned EmissionDirectionDomain = 0x50484449u, BounceDomain = 0x9e3779b9u;
constexpr unsigned ScatterDomain = 0x53434154u, LobeDomain = 0x4c4f4245u, RouletteDomain = 0x52554c45u;
constexpr unsigned CameraDomain = 0x43414d45u, ApertureDomain = 0x41504552u;
constexpr double KernelPi = 3.1415926535897932384626433832795;
constexpr double MinimumNormalAlignment = 0.9, MaximumNormalSeparation = 0.1;
constexpr double VolumeKernelIntegral = 8.0 / 15.0;
constexpr float MinimumSurvival = 0.05f, MaximumSurvival = 0.95f;

// Scene-upload records match the existing CUDA RHI, independently of integrator.
struct alignas(16) Vec4
{
    float X, Y, Z, W;
};
struct alignas(16) PackedAccelerationNode
{
    float MinX, MinY, MinZ;
    unsigned FirstOrEscape;
    float MaxX, MaxY, MaxZ;
    unsigned PrimitiveCount;
    float SplitPosition;
    unsigned SplitAxis, AboveChild;
    float SecondPlane;
};
struct PackedPrimitiveRef
{
    unsigned TypeAndIndex;
};
struct alignas(16) PackedTriangle
{
    Vec4 Position0, Edge1, Edge2, Normal0, Normal1, Normal2, Color0, Color1, Color2, TexCoord01, TexCoord2;
    unsigned MaterialIndex, Flags, Padding0, Padding1;
};
struct alignas(16) PackedSphere
{
    Vec4 CenterRadius;
    unsigned MaterialIndex, Padding0, Padding1, Padding2;
};
#include "MaterialTypes.cuh"
struct alignas(16) PackedTexture
{
    unsigned OffsetTexels, Width, Height, Padding;
};
struct alignas(16) FrameConstants
{
    Vec3 CameraPosition;
    unsigned Width;
    Vec3 CameraForward;
    unsigned Height;
    Vec3 CameraRight;
    unsigned TotalSamples;
    Vec3 CameraUp;
    unsigned MaxReflectionBounces;
    float HalfWidth, HalfHeight;
    unsigned MaterialCount, LightCount;
    Vec3 BackgroundTop;
    float AmbientIntensity;
    Vec3 BackgroundBottom;
    unsigned NodeCount;
    unsigned PrimitiveRefCount, TriangleCount, SphereCount;
    float ShadowStrength;
    unsigned PixelBegin, PixelCount, SampleBegin, SampleCount;
    unsigned GammaCorrection, GroupCount, TextureCount, TextureTexelCount;
    unsigned FullWidth, FullHeight, TileX, TileY;
    unsigned Integrator, TransportFlags, RouletteStartDepth, SamplingSequence;
    unsigned AdaptiveEnabled, AdaptivePilotSamples, AdaptiveMinimumSamples, AdaptivePhase;
    float AdaptiveRelativeError, AdaptiveAbsoluteError;
    unsigned Acceleration, InstanceNodeCount;
    const Arvo::RayTracing::CudaInstances::Record* InstanceRecords;
    const Arvo::RayTracing::CudaInstances::Node* InstanceNodes;
    unsigned* InstanceError;
    float LensRadius, FocusDistance;
    unsigned int SamplingSeed;
    unsigned int SamplingPadding[3];
};
struct alignas(16) GroupStatistics
{
    unsigned long long PrimaryRays, ShadowRays, ReflectionRays, PrimitiveTests;
};
struct Ray
{
    Vec3 Origin, Direction;
    float MinimumDistance, MaximumDistance;
};
struct Hit
{
    float Distance;
    Vec3 Normal, GeometricNormal;
    unsigned FrontFace;
    Vec3 Color;
    unsigned MaterialIndex;
    float TexCoordX, TexCoordY;
    unsigned HasTexCoord, PrimitiveIndex;
};
static_assert(sizeof(FrameConstants) == 272 && offsetof(FrameConstants, InstanceRecords) == 224);
static_assert(sizeof(PackedTriangle) == 192 && sizeof(PackedMaterial) == 128);

__device__ Vec3 Vector(float3 v)
{
    return {v.x, v.y, v.z};
}
__device__ float3 Native(Vec3 v)
{
    return make_float3(v.X, v.Y, v.Z);
}
__device__ Vec3 ToVec3(Vec4 v)
{
    return {v.X, v.Y, v.Z};
}
__device__ Vec3 Add(Vec3 a, Vec3 b)
{
    return {a.X + b.X, a.Y + b.Y, a.Z + b.Z};
}
__device__ Vec3 Subtract(Vec3 a, Vec3 b)
{
    return {a.X - b.X, a.Y - b.Y, a.Z - b.Z};
}
__device__ Vec3 Multiply(Vec3 v, float s)
{
    return {v.X * s, v.Y * s, v.Z * s};
}
__device__ Vec3 Multiply(Vec3 a, Vec3 b)
{
    return {a.X * b.X, a.Y * b.Y, a.Z * b.Z};
}
__device__ float Dot(Vec3 a, Vec3 b)
{
    return a.X * b.X + a.Y * b.Y + a.Z * b.Z;
}
__device__ Vec3 Cross(Vec3 a, Vec3 b)
{
    return {a.Y * b.Z - a.Z * b.Y, a.Z * b.X - a.X * b.Z, a.X * b.Y - a.Y * b.X};
}
__device__ Vec3 Normalize(Vec3 v)
{
    float length = norm3df(v.X, v.Y, v.Z);
    return length > 0 ? Multiply(v, 1 / length) : Vec3{};
}
__device__ Vec3 Reflect(Vec3 d, Vec3 n)
{
    return Subtract(d, Multiply(n, 2 * Dot(d, n)));
}
__device__ float Component(Vec3 v, unsigned axis)
{
    return axis == 0 ? v.X : axis == 1 ? v.Y : v.Z;
}
__device__ bool Valid(Vec3 v)
{
    return isfinite(v.X) && isfinite(v.Y) && isfinite(v.Z) && v.X >= 0 && v.Y >= 0 && v.Z >= 0;
}
__device__ bool Positive(Vec3 v)
{
    return v.X > 0 || v.Y > 0 || v.Z > 0;
}
__device__ unsigned Hash(unsigned value)
{
    constexpr unsigned FirstAvalanche = 0x7feb352du, SecondAvalanche = 0x846ca68bu;
    constexpr unsigned HalfWord = 16, MiddleShift = 15;
    value = (value ^ (value >> HalfWord)) * FirstAvalanche;
    value = (value ^ (value >> MiddleShift)) * SecondAvalanche;
    return value ^ (value >> HalfWord);
}
__device__ float Random01(unsigned seed)
{
    constexpr unsigned MantissaMask = 0x00ffffffu;
    constexpr float MantissaRange = 16777216.0f;
    return float(Hash(seed) & MantissaMask) / MantissaRange;
}
#include "Sequence.cu"
#include "Triangle.cu"
#include "Sphere.cu"
#include "Bvh.cu"
#include "KdTree.cu"
#include "Bih.cu"
#include "Dispatch.cu"
#include "SurfaceScattering.cu"
#include "FilmMoments.cu"

struct SceneView
{
    const PackedAccelerationNode* Nodes;
    const PackedPrimitiveRef* References;
    const PackedTriangle* Triangles;
    const PackedSphere* Spheres;
    const PackedMaterial* Materials;
    const PackedTexture* Textures;
    const unsigned* Texels;
    const Emitter* Emitters;
    const FrameConstants& Frame;
    const P::Controls& Control;
    unsigned* Error;
};
__device__ bool Configuration(const FrameConstants& frame, const P::Controls& control, unsigned* error)
{
    if (frame.MaxReflectionBounces > P::MaximumReflections ||
        control.SlotsPerPath != frame.MaxReflectionBounces + 1 || !control.PhotonPaths ||
        !control.BucketCount || (control.BucketCount & (control.BucketCount - 1)) || !frame.TotalSamples ||
        frame.AdaptiveEnabled || ((frame.TransportFlags >> MisShift) & MisMask) ||
        frame.ShadowStrength != 1 || control.Radius < 0 || !isfinite(control.Radius))
    {
        atomicOr(error, P::InvalidConfiguration);
        return false;
    }
    return true;
}
__device__ Vec3 Displaced(Vec3 point, Vec3 normal, Vec3 direction)
{
    return Add(point, Multiply(normal, Dot(normal, direction) >= 0 ? RayEpsilon : -RayEpsilon));
}
__device__ float Survival(Vec3 throughput)
{
    return fminf(MaximumSurvival,
                 fmaxf(MinimumSurvival, fmaxf(throughput.X, fmaxf(throughput.Y, throughput.Z))));
}
__device__ float2 PhotonVariates(const P::Controls& control, unsigned path, unsigned domain)
{
    constexpr unsigned LightPopulationDomain = 0x43555048u;
    unsigned key = Hash(path + Hash(control.Seed ^ LightPopulationDomain)) ^ domain;
    return make_float2(Random01(key), Random01(key ^ SecondDimensionScramble));
}
__device__ float Decode(unsigned channel)
{
    constexpr float ByteMaximum = 255, Threshold = 0.04045f, LinearScale = 12.92f;
    constexpr float Offset = 0.055f, Scale = 1.055f, Exponent = 2.4f;
    float encoded = channel / ByteMaximum;
    return encoded <= Threshold ? encoded / LinearScale : powf((encoded + Offset) / Scale, Exponent);
}
__device__ Vec3 TexturePixel(const SceneView& scene, const PackedTexture& texture, long long x, long long y)
{
    x %= texture.Width;
    y %= texture.Height;
    if (x < 0)
        x += texture.Width;
    if (y < 0)
        y += texture.Height;
    constexpr unsigned ByteMask = 0xffu, GreenShift = 8, BlueShift = 16;
    unsigned value = scene.Texels[texture.OffsetTexels + y * texture.Width + x];
    return {Decode(value & ByteMask), Decode((value >> GreenShift) & ByteMask),
            Decode((value >> BlueShift) & ByteMask)};
}
__device__ Vec3 SurfaceColor(const SceneView& scene, const Hit& hit)
{
    unsigned index = scene.Materials[hit.MaterialIndex].DiffuseTextureIndex;
    if (!hit.HasTexCoord || index >= scene.Frame.TextureCount || !isfinite(hit.TexCoordX) ||
        !isfinite(hit.TexCoordY))
        return hit.Color;
    const auto& texture = scene.Textures[index];
    unsigned long long size = static_cast<unsigned long long>(texture.Width) * texture.Height;
    if (!size || texture.OffsetTexels >= scene.Frame.TextureTexelCount ||
        size > scene.Frame.TextureTexelCount - texture.OffsetTexels)
        return hit.Color;
    float flipped = 1 - hit.TexCoordY;
    float x = (hit.TexCoordX - floorf(hit.TexCoordX)) * texture.Width - 0.5f,
          y = (flipped - floorf(flipped)) * texture.Height - 0.5f;
    long long column = static_cast<long long>(floorf(x)), row = static_cast<long long>(floorf(y));
    float fx = x - floorf(x), fy = y - floorf(y);
    Vec3 lower = Add(Multiply(TexturePixel(scene, texture, column, row), 1 - fx),
                     Multiply(TexturePixel(scene, texture, column + 1, row), fx));
    Vec3 upper = Add(Multiply(TexturePixel(scene, texture, column, row + 1), 1 - fx),
                     Multiply(TexturePixel(scene, texture, column + 1, row + 1), fx));
    return Multiply(hit.Color, Add(Multiply(lower, 1 - fy), Multiply(upper, fy)));
}
__device__ bool Rectangle(const Emitter& source, const Ray& ray, float& distance)
{
    if (source.Kind != ArvoCudaEndpoint::Rectangle)
        return false;
    Vec3 normal = Vector(source.Direction);
    float cosine = Dot(normal, ray.Direction);
    if (cosine == 0)
        return false;
    float candidate = Dot(normal, Subtract(Vector(source.Origin), ray.Origin)) / cosine;
    if (!(candidate > ray.MinimumDistance && candidate < ray.MaximumDistance && candidate <= distance))
        return false;
    Vec3 helper = fabsf(normal.Y) < BasisParallelThreshold ? Vec3{0, 1, 0} : Vec3{1, 0, 0};
    Vec3 tangent = Normalize(Cross(helper, normal)), bitangent = Cross(normal, tangent);
    Vec3 relative = Subtract(Add(ray.Origin, Multiply(ray.Direction, candidate)), Vector(source.Origin));
    if (fabsf(Dot(relative, tangent)) > source.Width / 2 ||
        fabsf(Dot(relative, bitangent)) > source.Height / 2)
        return false;
    distance = candidate;
    return true;
}
struct Intersection
{
    Hit Surface;
    Vec3 Position;
    unsigned Source;
    bool Geometry;
};
__device__ Intersection Trace(const SceneView& scene, const Ray& ray, GroupStatistics& statistics)
{
    Intersection endpoint{};
    endpoint.Source = P::Invalid;
    endpoint.Surface.Distance = ray.MaximumDistance;
    endpoint.Geometry = IntersectScene(scene.Nodes, scene.References, scene.Triangles, scene.Spheres,
                                       scene.Frame, ray, false, endpoint.Surface, statistics.PrimitiveTests);
    const bool proxy = endpoint.Geometry && endpoint.Surface.MaterialIndex < scene.Frame.MaterialCount &&
                       scene.Materials[endpoint.Surface.MaterialIndex].SampledEmitter != 0;
    float nearest = endpoint.Geometry ? endpoint.Surface.Distance + (proxy ? RayEpsilon : 0)
                                     : ray.MaximumDistance;
    for (unsigned i = 0; i < scene.Control.EmitterCount; ++i)
        if (Rectangle(scene.Emitters[i], ray, nearest))
            endpoint.Source = i;
    if (endpoint.Source != P::Invalid)
    {
        endpoint.Geometry = false;
        endpoint.Position = Add(ray.Origin, Multiply(ray.Direction, nearest));
        return endpoint;
    }
    if (endpoint.Geometry)
        endpoint.Position = Add(ray.Origin, Multiply(ray.Direction, endpoint.Surface.Distance));
    for (unsigned i = 0; i < scene.Control.EmitterCount; ++i)
    {
        const auto& source = scene.Emitters[i];
        if (endpoint.Geometry
                ? source.Primitive != P::Invalid && source.Primitive == endpoint.Surface.PrimitiveIndex
                : source.Kind == ArvoCudaEndpoint::Environment)
        {
            endpoint.Source = i;
            break;
        }
    }
    return endpoint;
}
__device__ Vec3 Radiance(const Emitter& source, Vec3 direction)
{
    if (source.Kind == ArvoCudaEndpoint::Environment)
    {
        float blend = (1 - direction.Y) / 2;
        return Add(Multiply(Vector(source.BackgroundTop), blend),
                   Multiply(Vector(source.BackgroundBottom), 1 - blend));
    }
    if (source.Kind == ArvoCudaEndpoint::Rectangle)
        return Dot(Vector(source.Direction), direction) > 0
                   ? Multiply(Vector(source.Strength), 1 / (source.Width * source.Height))
                   : Vec3{};
    return Vector(source.Strength);
}
__device__ PhysicalSurface Scattering(const SceneView& scene, const Intersection& hit, Vec3 view,
                                      unsigned mode, float& mirror)
{
    PackedMaterial material = scene.Materials[hit.Surface.MaterialIndex];
    mirror = material.Model == LegacyMaterial ? fminf(1, fmaxf(0, material.AlbedoReflectivity.W)) : 0;
    if (material.Model == LegacyMaterial)
        material.Model = DiffuseMaterial;
    return MakePhysicalSurface(material, hit.Surface, view, SurfaceColor(scene, hit.Surface), mode);
}
__device__ PhysicalSample Mirror(const PhysicalSurface& surface, Vec3 view)
{
    Vec3 direction = Normalize(Reflect(Multiply(view, -1), surface.Normal));
    if (Dot(view, surface.Geometry) * Dot(direction, surface.Geometry) <= 0 ||
        Dot(direction, surface.Normal) <= 0 || surface.View.z <= 0)
        return {};
    float correction = SurfaceAdjoint(surface, direction);
    return {direction, {correction, correction, correction},
            1,         1,
            1,         ArvoCuda::ScatteringDelta | ArvoCuda::ScatteringReflection};
}
__device__ void Emit(const SceneView& scene, unsigned path, P::Photon* photons, GroupStatistics& statistics)
{
    const auto& control = scene.Control;
    const auto light = ArvoCudaEndpoint::Emit(scene.Emitters, control.EmitterCount,
                                              {Native(control.WorldCenter), control.WorldRadius},
                                              PhotonVariates(control, path, EmissionSourceDomain).x,
                                              PhotonVariates(control, path, EmissionPositionDomain),
                                              PhotonVariates(control, path, EmissionDirectionDomain));
    if (!(light.Emission.Flags & ArvoCudaEmission::Valid))
        return;
    Vec3 flux = Vector(light.Emission.Weight), direction = Vector(light.Emission.Direction);
    Ray ray{Displaced(Vector(light.Emission.Origin), Vector(light.Emission.Normal), direction), direction,
            RayEpsilon, MaximumRayDistance};
    bool deltaPrefix = true;
    unsigned interiorMaterial = P::Invalid;
    for (unsigned depth = 0; depth < control.SlotsPerPath; ++depth)
    {
        if (!Valid(flux))
        {
            atomicOr(scene.Error, P::NumericalFailure);
            return;
        }
        if (!Positive(flux))
            return;
        ++statistics.ReflectionRays;
        const auto hit = Trace(scene, ray, statistics);
        unsigned bounce = Hash(depth ^ BounceDomain);
        if (hit.Geometry && hit.Surface.MaterialIndex >= scene.Frame.MaterialCount)
        {
            atomicOr(scene.Error, P::InvalidConfiguration);
            return;
        }
        if (depth == 0 && hit.Geometry && !hit.Surface.FrontFace &&
            scene.Materials[hit.Surface.MaterialIndex].Optical.W > 0)
            interiorMaterial = hit.Surface.MaterialIndex;
        if (interiorMaterial != P::Invalid)
        {
            const PackedMaterial interior = scene.Materials[interiorMaterial];
            const bool finiteSource = hit.Source != P::Invalid &&
                scene.Emitters[hit.Source].Kind == ArvoCudaEndpoint::Rectangle;
            const float boundary = hit.Geometry || finiteSource
                ? norm3df(hit.Position.X - ray.Origin.X, hit.Position.Y - ray.Origin.Y,
                          hit.Position.Z - ray.Origin.Z) : ray.MaximumDistance;
            const auto flight = SubsurfaceFreeFlight(interior.Optical.W,
                make_float3(interior.AlbedoReflectivity.X * interior.ConductorExtinction.W,
                            interior.AlbedoReflectivity.Y * interior.ConductorExtinction.W,
                            interior.AlbedoReflectivity.Z * interior.ConductorExtinction.W),
                Native(ray.Direction), boundary,
                PhotonVariates(control, path, bounce ^ LobeDomain).x);
            if (flight.Scattered)
            {
                auto& saved = photons[path * control.SlotsPerPath + depth];
                saved.Position = Add(ray.Origin, Multiply(ray.Direction, flight.Distance));
                saved.Incoming = Multiply(ray.Direction, -1);
                saved.Flux = flux;
                saved.LightDepth = depth;
                saved.Kind = (depth == 0 ? P::Direct : deltaPrefix ? P::Caustic : P::Indirect)
                    | P::VolumeFlag;
                saved.MaterialIndex = interiorMaterial;
            }
            flux = Multiply(flux, Vector(flight.Weight));
            if (!Valid(flux))
            {
                atomicOr(scene.Error, P::NumericalFailure);
                return;
            }
            if (!Positive(flux)) return;
            if (flight.Scattered)
            {
                if (depth + 1 == control.SlotsPerPath) return;
                const auto phase = SubsurfaceSampleDirection(Native(ray.Direction), interior.ConductorIndex.W,
                    PhotonVariates(control, path, bounce ^ ScatterDomain),
                    (scene.Frame.TransportFlags & ImportanceBit) != 0);
                if (!(phase.Pdf > 0) || !isfinite(phase.Weight))
                {
                    atomicOr(scene.Error, P::NumericalFailure);
                    return;
                }
                flux = Multiply(flux, phase.Weight);
                if ((scene.Frame.TransportFlags & RouletteBit) && depth + 1 >= scene.Frame.RouletteStartDepth)
                {
                    const float survival = Survival(flux);
                    if (PhotonVariates(control, path, bounce ^ RouletteDomain).x >= survival) return;
                    flux = Multiply(flux, 1 / survival);
                }
                deltaPrefix = false;
                ray = {Add(ray.Origin, Multiply(ray.Direction, flight.Distance)), Vector(phase.Direction),
                       RayEpsilon, MaximumRayDistance};
                continue;
            }
        }
        if (!hit.Geometry)
            return;
        float mirror;
        const auto surface =
            Scattering(scene, hit, Multiply(ray.Direction, -1), ArvoCuda::TransportImportance, mirror);
        if (!PhysicalDelta(surface) && mirror < 1)
        {
            auto& saved = photons[path * control.SlotsPerPath + depth];
            saved.Position = hit.Position;
            saved.Incoming = Multiply(ray.Direction, -1);
            saved.Flux = flux;
            saved.Geometry = hit.Surface.GeometricNormal;
            saved.LightDepth = depth;
            saved.Kind = depth == 0 ? P::Direct : deltaPrefix ? P::Caustic : P::Indirect;
            saved.MaterialIndex = hit.Surface.MaterialIndex;
        }
        if (depth + 1 == control.SlotsPerPath)
            return;
        float lobe = PhotonVariates(control, path, bounce ^ LobeDomain).x;
        const auto event =
            lobe < mirror ? Mirror(surface, Multiply(ray.Direction, -1))
                          : SamplePhysical(surface, PhotonVariates(control, path, bounce ^ ScatterDomain),
                                           mirror > 0 ? (lobe - mirror) / (1 - mirror) : lobe,
                                           (scene.Frame.TransportFlags & ImportanceBit) != 0);
        if (!event.Event || !(event.Pdf > 0))
            return;
        flux = Multiply(flux, event.Weight);
        if (!Valid(flux))
        {
            atomicOr(scene.Error, P::NumericalFailure);
            return;
        }
        deltaPrefix = deltaPrefix && (event.Event & ArvoCuda::ScatteringDelta);
        if ((scene.Frame.TransportFlags & RouletteBit) && depth + 1 >= scene.Frame.RouletteStartDepth)
        {
            float survival = Survival(flux);
            if (PhotonVariates(control, path, bounce ^ RouletteDomain).x >= survival)
                return;
            flux = Multiply(flux, 1 / survival);
        }
        ray.Origin = PhysicalRayOrigin(surface, hit.Position, event.Direction);
        ray.Direction = event.Direction;
        if (surface.Material.Optical.W > 0 && (event.Event & ArvoCuda::ScatteringTransmission) != 0)
            interiorMaterial = hit.Surface.FrontFace ? hit.Surface.MaterialIndex : P::Invalid;
    }
}

struct Cell
{
    int X, Y, Z;
};
__device__ Cell GridCell(Vec3 position, const P::Controls& control, unsigned* error)
{
    constexpr double CoordinateLimit = 1073741824.0; // Leaves room for all neighboring signed cells.
    double x = floor((double(position.X) - control.WorldCenter.X) / control.Radius);
    double y = floor((double(position.Y) - control.WorldCenter.Y) / control.Radius);
    double z = floor((double(position.Z) - control.WorldCenter.Z) / control.Radius);
    if (!isfinite(x) || !isfinite(y) || !isfinite(z) || fabs(x) > CoordinateLimit ||
        fabs(y) > CoordinateLimit || fabs(z) > CoordinateLimit)
    {
        atomicOr(error, P::NumericalFailure);
        return {};
    }
    return {int(x), int(y), int(z)};
}
__device__ unsigned Bucket(Cell cell, unsigned count)
{
    constexpr unsigned AxisPrimeX = 73856093u, AxisPrimeY = 19349663u, AxisPrimeZ = 83492791u;
    return (unsigned(cell.X) * AxisPrimeX ^ unsigned(cell.Y) * AxisPrimeY ^ unsigned(cell.Z) * AxisPrimeZ) &
           (count - 1);
}
__device__ Vec3 Density(const SceneView& scene, const P::Photon* photons, const unsigned* heads,
                        const Intersection& receiver, const PhysicalSurface& surface, unsigned cameraDepth)
{
    const auto& control = scene.Control;
    if (!(control.Radius > 0))
        return {};
    const Cell center = GridCell(receiver.Position, control, scene.Error);
    const double radiusSquared = control.Radius * control.Radius;
    const double scale = 2 / (KernelPi * radiusSquared * control.PhotonPaths);
    double red = 0, green = 0, blue = 0;
    constexpr int NeighborRadius = 1;
    for (int z = -NeighborRadius; z <= NeighborRadius; ++z)
        for (int y = -NeighborRadius; y <= NeighborRadius; ++y)
            for (int x = -NeighborRadius; x <= NeighborRadius; ++x)
            {
                const Cell requested{center.X + x, center.Y + y, center.Z + z};
                unsigned cursor = heads[Bucket(requested, control.BucketCount)];
                while (cursor != P::Invalid)
                {
                    const auto& photon = photons[cursor];
                    cursor = photon.Next;
                    if (photon.Kind & P::VolumeFlag)
                        continue;
                    // Hash collisions must never count a photon more than once.
                    const Cell stored = GridCell(photon.Position, control, scene.Error);
                    if (stored.X != requested.X || stored.Y != requested.Y || stored.Z != requested.Z)
                        continue;
                    if (photon.LightDepth > scene.Frame.MaxReflectionBounces - cameraDepth)
                        continue;
                    const unsigned kind = photon.Kind & P::ClassMask;
                    if (kind == P::Direct && (scene.Frame.TransportFlags & NeeBit))
                        continue;
                    if (kind == P::Caustic && !(control.Classes & P::CausticEnabled))
                        continue;
                    if (kind == P::Indirect && !(control.Classes & P::IndirectEnabled))
                        continue;
                    if (surface.Material.Optical.W > 0 &&
                        (receiver.Surface.FrontFace ? Dot(photon.Incoming, surface.Geometry) <= 0
                                                    : Dot(photon.Incoming, surface.Geometry) >= 0))
                        continue;
                    if (Dot(photon.Geometry, surface.Geometry) < MinimumNormalAlignment)
                        continue;
                    double dx = double(photon.Position.X) - receiver.Position.X,
                           dy = double(photon.Position.Y) - receiver.Position.Y,
                           dz = double(photon.Position.Z) - receiver.Position.Z;
                    double distanceSquared = dx * dx + dy * dy + dz * dz;
                    if (distanceSquared > radiusSquared ||
                        fabs(dx * surface.Geometry.X + dy * surface.Geometry.Y + dz * surface.Geometry.Z) >
                            MaximumNormalSeparation * control.Radius)
                        continue;
                    double projected = fabs(double(Dot(photon.Incoming, surface.Geometry)));
                    if (!(projected > 0))
                        continue;
                    const auto evaluated = EvaluatePhysical(
                        surface, photon.Incoming, (scene.Frame.TransportFlags & ImportanceBit) != 0);
                    double weight = scale * (1 - distanceSquared / radiusSquared) *
                                    fabs(double(Dot(photon.Incoming, surface.Normal))) / projected;
                    red += double(photon.Flux.X) * evaluated.Value.X * weight;
                    green += double(photon.Flux.Y) * evaluated.Value.Y * weight;
                    blue += double(photon.Flux.Z) * evaluated.Value.Z * weight;
                }
            }
    return {float(red), float(green), float(blue)};
}
__device__ Vec3 VolumeDensity(const SceneView& scene, const P::Photon* photons, const unsigned* heads,
                              Vec3 point, Vec3 cameraDirection, unsigned materialIndex,
                              unsigned cameraDepth, GroupStatistics& statistics)
{
    const auto& control = scene.Control;
    if (!(control.Radius > 0))
        return {};
    const Cell center = GridCell(point, control, scene.Error);
    const double radiusSquared = control.Radius * control.Radius;
    // Collision density carries sigma_t; the camera's free-flight event has
    // already applied scattering albedo. Convert density back to incident light.
    const double scale = scene.Materials[materialIndex].Optical.W /
        (KernelPi * control.Radius * radiusSquared * VolumeKernelIntegral * control.PhotonPaths);
    double red = 0, green = 0, blue = 0;
    constexpr int NeighborRadius = 1;
    for (int z = -NeighborRadius; z <= NeighborRadius; ++z)
        for (int y = -NeighborRadius; y <= NeighborRadius; ++y)
            for (int x = -NeighborRadius; x <= NeighborRadius; ++x)
            {
                const Cell requested{center.X + x, center.Y + y, center.Z + z};
                unsigned cursor = heads[Bucket(requested, control.BucketCount)];
                while (cursor != P::Invalid)
                {
                    const auto& photon = photons[cursor];
                    cursor = photon.Next;
                    if (!(photon.Kind & P::VolumeFlag) || photon.MaterialIndex != materialIndex)
                        continue;
                    const Cell stored = GridCell(photon.Position, control, scene.Error);
                    if (stored.X != requested.X || stored.Y != requested.Y || stored.Z != requested.Z)
                        continue;
                    if (photon.LightDepth > scene.Frame.MaxReflectionBounces - cameraDepth)
                        continue;
                    const unsigned kind = photon.Kind & P::ClassMask;
                    if (kind == P::Caustic && !(control.Classes & P::CausticEnabled))
                        continue;
                    if (kind == P::Indirect && !(control.Classes & P::IndirectEnabled))
                        continue;
                    const Vec3 separation = Subtract(photon.Position, point);
                    const double squared = double(separation.X) * separation.X +
                        double(separation.Y) * separation.Y + double(separation.Z) * separation.Z;
                    if (squared > radiusSquared)
                        continue;
                    const float distance = norm3df(separation.X, separation.Y, separation.Z);
                    if (distance > 2 * RayEpsilon)
                    {
                        const Ray connection{point, Multiply(separation, 1 / distance), RayEpsilon,
                                             distance - RayEpsilon};
                        Hit blocker{};
                        blocker.Distance = connection.MaximumDistance;
                        if (IntersectScene(scene.Nodes, scene.References, scene.Triangles, scene.Spheres,
                                           scene.Frame, connection, true, blocker, statistics.PrimitiveTests))
                            continue;
                    }
                    const double phase = SubsurfacePhase(Dot(photon.Incoming, cameraDirection),
                        scene.Materials[materialIndex].ConductorIndex.W);
                    const double weight = scale * (1 - squared / radiusSquared) * phase;
                    red += photon.Flux.X * weight;
                    green += photon.Flux.Y * weight;
                    blue += photon.Flux.Z * weight;
                }
            }
    return {float(red), float(green), float(blue)};
}
__device__ Vec3 Direct(const SceneView& scene, const SampleContext& sequence, unsigned domain, Vec3 point,
                       const PhysicalSurface& surface, const Hit* subsurfaceBoundary,
                       GroupStatistics& statistics)
{
    const auto light = ArvoCudaEndpoint::Emit(scene.Emitters, scene.Control.EmitterCount,
                                              {Native(scene.Control.WorldCenter), scene.Control.WorldRadius},
                                              Sample2D(sequence, domain ^ EmissionSourceDomain).x,
                                              Sample2D(sequence, domain ^ EmissionPositionDomain),
                                              Sample2D(sequence, domain ^ EmissionDirectionDomain));
    const auto& emitted = light.Emission;
    if (!(emitted.Flags & ArvoCudaEmission::Valid))
        return {};
    bool infinite = (emitted.Flags & ArvoCudaEmission::Infinite) != 0,
         pointLight = (emitted.Flags & ArvoCudaEmission::DeltaPosition) != 0;
    Vec3 incoming = Multiply(Vector(emitted.Direction), -1);
    double geometry = 1;
    if (!infinite)
    {
        double dx = double(emitted.Origin.x) - point.X, dy = double(emitted.Origin.y) - point.Y,
               dz = double(emitted.Origin.z) - point.Z;
        double distance = norm3d(dx, dy, dz);
        if (!(distance > 0))
            return {};
        incoming = {float(dx / distance), float(dy / distance), float(dz / distance)};
        geometry =
            (pointLight ? 1 : fabs(double(Dot(Vector(emitted.Normal), incoming)))) / (distance * distance);
    }
    const bool transmitted = subsurfaceBoundary &&
        (subsurfaceBoundary->FrontFace ? Dot(incoming, subsurfaceBoundary->GeometricNormal) < 0
                                       : Dot(incoming, subsurfaceBoundary->GeometricNormal) > 0);
    if (transmitted && !pointLight && !(emitted.Flags & ArvoCudaEmission::DeltaDirection))
        return {};
    double density =
        double(light.SelectionProbability) * (infinite ? emitted.DirectionPdf : emitted.PositionPdf);
    if (!(density > 0))
        return {};
    const auto scattering =
        EvaluatePhysical(surface, incoming, (scene.Frame.TransportFlags & ImportanceBit) != 0);
    const Vec3 emission = Radiance(scene.Emitters[light.EmitterIndex], Multiply(incoming, -1));
    double weight = fabs(double(Dot(surface.Normal, incoming))) * geometry / density;
    Vec3 value{float(double(scattering.Value.X) * emission.X * weight),
               float(double(scattering.Value.Y) * emission.Y * weight),
               float(double(scattering.Value.Z) * emission.Z * weight)};
    if (transmitted && subsurfaceBoundary->FrontFace && pointLight)
    {
        const Vec3 separation = Subtract(Vector(emitted.Origin), point);
        value = Multiply(value, expf(-norm3df(separation.X, separation.Y, separation.Z) /
            surface.Material.Optical.W));
    }
    if (!Valid(value) || !Positive(value))
        return value;
    Ray shadow{PhysicalRayOrigin(surface, point, incoming), incoming, RayEpsilon, MaximumRayDistance};
    if (!infinite)
    {
        Vec3 target = pointLight
                          ? Vector(emitted.Origin)
                          : Displaced(Vector(emitted.Origin), Vector(emitted.Normal), Multiply(incoming, -1));
        Vec3 connection = Subtract(target, shadow.Origin);
        float distance = norm3df(connection.X, connection.Y, connection.Z);
        if (!(distance > 2 * RayEpsilon))
            return {};
        shadow.Direction = Multiply(connection, 1 / distance);
        shadow.MaximumDistance = distance - RayEpsilon;
    }
    ++statistics.ShadowRays;
    Hit occluder{};
    occluder.Distance = shadow.MaximumDistance;
    if (IntersectScene(scene.Nodes, scene.References, scene.Triangles, scene.Spheres, scene.Frame, shadow,
                       true, occluder, statistics.PrimitiveTests))
        return {};
    float closest = shadow.MaximumDistance;
    for (unsigned source = 0; source < scene.Control.EmitterCount; ++source)
        if (Rectangle(scene.Emitters[source], shadow, closest))
            return {};
    return value;
}
__device__ Vec3 Camera(const SceneView& scene, const P::Photon* photons, const unsigned* heads, Ray ray,
                       const SampleContext& sequence, GroupStatistics& statistics)
{
    Vec3 result{}, throughput{1, 1, 1};
    float etaScale = 1;
    unsigned interiorMaterial = P::Invalid;
    for (unsigned depth = 0; depth <= scene.Control.SlotsPerPath; ++depth)
    {
        if (!Valid(throughput) || !isfinite(etaScale))
        {
            atomicOr(scene.Error, P::NumericalFailure);
            return {};
        }
        if (!Positive(throughput))
            break;
        if (depth == 0)
            ++statistics.PrimaryRays;
        else
            ++statistics.ReflectionRays;
        const auto hit = Trace(scene, ray, statistics);
        const unsigned domain = CameraDomain ^ Hash(depth ^ BounceDomain);
        if (hit.Geometry && hit.Surface.MaterialIndex >= scene.Frame.MaterialCount)
        {
            atomicOr(scene.Error, P::InvalidConfiguration);
            return {};
        }
        if (depth == 0 && hit.Geometry && !hit.Surface.FrontFace &&
            scene.Materials[hit.Surface.MaterialIndex].Optical.W > 0)
            interiorMaterial = hit.Surface.MaterialIndex;
        if (interiorMaterial != P::Invalid)
        {
            const PackedMaterial interior = scene.Materials[interiorMaterial];
            const bool finiteSource = hit.Source != P::Invalid &&
                scene.Emitters[hit.Source].Kind == ArvoCudaEndpoint::Rectangle;
            const float boundary = hit.Geometry || finiteSource
                ? norm3df(hit.Position.X - ray.Origin.X, hit.Position.Y - ray.Origin.Y,
                          hit.Position.Z - ray.Origin.Z) : ray.MaximumDistance;
            const auto flight = SubsurfaceFreeFlight(interior.Optical.W,
                make_float3(interior.AlbedoReflectivity.X * interior.ConductorExtinction.W,
                            interior.AlbedoReflectivity.Y * interior.ConductorExtinction.W,
                            interior.AlbedoReflectivity.Z * interior.ConductorExtinction.W),
                Native(ray.Direction), boundary, Sample2D(sequence, domain ^ LobeDomain).x);
            throughput = Multiply(throughput, Vector(flight.Weight));
            if (!Valid(throughput))
            {
                atomicOr(scene.Error, P::NumericalFailure);
                return {};
            }
            if (!Positive(throughput)) break;
            if (flight.Scattered)
            {
                if (depth == scene.Control.SlotsPerPath) break;
                const Vec3 position = Add(ray.Origin, Multiply(ray.Direction, flight.Distance));
                result = Add(result, Multiply(throughput, VolumeDensity(scene, photons, heads, position,
                    ray.Direction, interiorMaterial, depth, statistics)));
                break;
            }
        }
        if (hit.Source != P::Invalid)
            result = Add(result, Multiply(throughput,
                                          Radiance(scene.Emitters[hit.Source], Multiply(ray.Direction, -1))));
        if (!hit.Geometry || depth == scene.Control.SlotsPerPath)
            break;
        float mirror;
        auto surface =
            Scattering(scene, hit, Multiply(ray.Direction, -1), ArvoCuda::TransportRadiance, mirror);
        const bool subsurfaceTransmission = surface.Material.Optical.W > 0 && !PhysicalDelta(surface);
        if (!PhysicalDelta(surface) && mirror < 1)
        {
            Vec3 illumination = Density(scene, photons, heads, hit, surface, depth);
            if (scene.Frame.TransportFlags & NeeBit)
                illumination = Add(illumination, Direct(scene, sequence, domain, hit.Position, surface,
                    subsurfaceTransmission ? &hit.Surface : nullptr, statistics));
            result = Add(result, Multiply(Multiply(throughput, illumination), 1 - mirror));
        }
        PhysicalSample event{};
        if (mirror > 0)
        {
            event = Mirror(surface, Multiply(ray.Direction, -1));
            event.Weight = Multiply(event.Weight, mirror);
        }
        else if (PhysicalDelta(surface) || subsurfaceTransmission)
            event = SamplePhysical(surface, Sample2D(sequence, domain ^ ScatterDomain),
                                   Sample2D(sequence, domain ^ LobeDomain).x,
                                   (scene.Frame.TransportFlags & ImportanceBit) != 0);
        else
            break;
        if (!event.Event || !(event.Pdf > 0))
            break;
        if (subsurfaceTransmission && !(event.Event & ArvoCuda::ScatteringTransmission))
            break;
        throughput = Multiply(throughput, event.Weight);
        etaScale *= event.RelativeIndex * event.RelativeIndex;
        if (!Valid(throughput) || !isfinite(etaScale))
        {
            atomicOr(scene.Error, P::NumericalFailure);
            return {};
        }
        if ((scene.Frame.TransportFlags & RouletteBit) && depth + 1 >= scene.Frame.RouletteStartDepth)
        {
            float survival = Survival(Multiply(throughput, etaScale));
            if (Sample2D(sequence, domain ^ RouletteDomain).x >= survival)
                break;
            throughput = Multiply(throughput, 1 / survival);
        }
        ray.Origin = PhysicalRayOrigin(surface, hit.Position, event.Direction);
        ray.Direction = event.Direction;
        if (surface.Material.Optical.W > 0 && (event.Event & ArvoCuda::ScatteringTransmission) != 0)
            interiorMaterial = hit.Surface.FrontFace ? hit.Surface.MaterialIndex : P::Invalid;
    }
    return result;
}
} // namespace

extern "C" __global__ void PhotonEmitKernel(const PackedAccelerationNode* nodes,
                                            const PackedPrimitiveRef* references,
                                            const PackedTriangle* triangles, const PackedSphere* spheres,
                                            const PackedMaterial* materials, const PackedTexture* textures,
                                            const unsigned* texels, const Emitter* emitters,
                                            const FrameConstants* frame, const P::Controls* controls,
                                            P::Photon* photons, GroupStatistics* statistics, unsigned* error)
{
    unsigned lane = blockIdx.x * blockDim.x + threadIdx.x, path = controls->Begin + lane;
    if (lane >= controls->Count || path >= controls->PhotonPaths)
        return;
    statistics[lane] = {};
    if (!Configuration(*frame, *controls, error))
        return;
    for (unsigned slot = 0; slot < controls->SlotsPerPath; ++slot)
        photons[path * controls->SlotsPerPath + slot] = {};
    const SceneView scene{nodes,  references, triangles, spheres,   materials, textures,
                          texels, emitters,   *frame,    *controls, error};
    Emit(scene, path, photons, statistics[lane]);
}
extern "C" __global__ void PhotonIndexKernel(const P::Controls* controls, P::Photon* photons, unsigned* heads,
                                             unsigned* error)
{
    unsigned lane = blockIdx.x * blockDim.x + threadIdx.x, slot = controls->Begin + lane;
    if (lane >= controls->Count || slot >= controls->PhotonPaths * controls->SlotsPerPath ||
        !photons[slot].Kind)
        return;
    Cell cell = GridCell(photons[slot].Position, *controls, error);
    // The next kernel starts after all insertions finish on the same stream.
    // No thread traverses a partially published list during this construction.
    photons[slot].Next = atomicExch(heads + Bucket(cell, controls->BucketCount), slot);
}
extern "C" __global__ void PhotonCameraKernel(const PackedAccelerationNode* nodes,
                                              const PackedPrimitiveRef* references,
                                              const PackedTriangle* triangles, const PackedSphere* spheres,
                                              const PackedMaterial* materials, const PackedTexture* textures,
                                              const unsigned* texels, const Emitter* emitters,
                                              const FrameConstants* frame, const P::Controls* controls,
                                              const P::Photon* photons, const unsigned* heads, Vec4* film,
                                              unsigned* counts, GroupStatistics* statistics, unsigned* error)
{
    unsigned lane = blockIdx.x * blockDim.x + threadIdx.x, pixel = controls->Begin + lane;
    if (lane >= controls->Count || pixel >= frame->Width * frame->Height)
        return;
    statistics[lane] = {};
    if (!Configuration(*frame, *controls, error))
        return;
    unsigned x = pixel % frame->Width + frame->TileX, y = pixel / frame->Width + frame->TileY;
    auto sequence = MakeSampleContext(x, y, frame->FullWidth, controls->Sample, frame->TotalSamples,
                                      frame->SamplingSequence, frame->SamplingSeed);
    auto jitter = Sample2D(sequence, CameraDomain);
    float2 raster = ArvoCudaCamera::PixelRaster(x, y, frame->FullWidth, frame->FullHeight, jitter);
    const ArvoCudaCamera::Frame sensor{Native(frame->CameraPosition),
                                       Native(frame->CameraForward),
                                       Native(frame->CameraRight),
                                       Native(frame->CameraUp),
                                       frame->HalfWidth,
                                       frame->HalfHeight,
                                       frame->LensRadius,
                                       frame->FocusDistance};
    auto primary = ArvoCudaCamera::Emit(sensor, raster, Sample2D(sequence, ApertureDomain));
    Vec3 radiance{};
    if (primary.Flags & ArvoCudaCamera::Valid)
    {
        SceneView scene{nodes,  references, triangles, spheres,   materials, textures,
                        texels, emitters,   *frame,    *controls, error};
        radiance = Camera(scene, photons, heads,
                          {Vector(primary.Origin), Vector(primary.Direction), RayEpsilon, MaximumRayDistance},
                          sequence, statistics[lane]);
    }
    if (!Valid(radiance))
    {
        atomicOr(error, P::NumericalFailure);
        return;
    }
    AccumulateFilmSample(film[pixel], counts[pixel], radiance);
    if (!Valid(ToVec3(film[pixel])) || !isfinite(film[pixel].W))
        atomicOr(error, P::NumericalFailure);
}
