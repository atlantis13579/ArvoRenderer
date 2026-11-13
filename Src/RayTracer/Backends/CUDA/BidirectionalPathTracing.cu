// Independent CUDA BDPT, fixed path length, with explicit camera connections.
// Veach (1997), "Robust Monte Carlo Methods for Light Transport Simulation",
// chapter 10; PBRT 3e, "Bidirectional Path Tracing". Each light/camera pair
// contributes all feasible cuts of a complete path. Lens connections splat to
// other pixels; film moments are updated only after a whole sample pass.
// https://pbr-book.org/3ed-2018/Light_Transport_III_Bidirectional_Methods/Bidirectional_Path_Tracing
// Interior scattering extends each path to volume measure: extinction enters
// destination PDFs and visibility connections carry Beer transmittance.
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cfloat>
#include "InstanceLayout.h"
#include "BidirectionalLayout.h"
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
#include "BidirectionalMis.cu"
#include "SubsurfaceScattering.cu"
#include "RadianceAccumulator.cu"

namespace
{
namespace Layout = Arvo::RayTracing::CudaBdpt;
namespace Mis = ArvoCudaBidirectional;
using Vec3 = Layout::Vector;
using Vertex = Layout::Vertex;
using Path = Layout::Path;
using Emitter = ArvoCudaEndpoint::PackedEmitter;
constexpr unsigned SphereBit = 0x80000000u, InvalidPrimitiveIndex = 0xffffffffu;
constexpr float Pi = CUDART_PI_F, RayEpsilon = 1.0e-4f, MaximumRayDistance = 1.0e30f;
constexpr float BasisParallelThreshold = 0.999f, MinimumDirectionLengthSquared = 1.0e-12f;
constexpr float MinimumTriangleDeterminant = 1.0e-8f;
constexpr unsigned PixelSeedMultiplier = 9781u, SampleSeedMultiplier = 6271u;
constexpr unsigned BounceSeedMultiplier = 0x9e3779b9u, SecondDimensionScramble = 0x68bc21ebu;
constexpr unsigned LobeSelectionScramble = 0xa511e9b3u;
constexpr unsigned CameraWalkDimension = 0x43414d50u, LightWalkDimension = 0x4c495450u;
constexpr unsigned SourceDimension = 0x53524345u, PositionDimension = 0x53504f53u;
constexpr unsigned DirectionDimension = 0x53444952u, LensDimension = 0x4c454e53u;
constexpr unsigned ConnectionLensDimension = 0x434c454eu;
constexpr unsigned InteriorFlightDimension = 0x49464c54u, InteriorPhaseDimension = 0x49504853u;
constexpr unsigned ImportanceBit = 1u, MisShift = 3u, MisMask = 3u;

// These records match the existing CUDA scene upload ABI, not another backend.
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
static_assert(Layout::MaximumVertices == Mis::MaximumVertices);

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
__device__ Vec3 Multiply(Vec3 a, float b)
{
    return {a.X * b, a.Y * b, a.Z * b};
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
    return length > 0 ? Vec3{v.X / length, v.Y / length, v.Z / length} : Vec3{};
}
__device__ Vec3 Reflect(Vec3 d, Vec3 n)
{
    return Subtract(d, Multiply(n, 2 * Dot(d, n)));
}
__device__ float Component(Vec3 v, unsigned axis)
{
    return axis == 0 ? v.X : (axis == 1 ? v.Y : v.Z);
}
__device__ bool Finite(Vec3 v)
{
    return isfinite(v.X) && isfinite(v.Y) && isfinite(v.Z);
}
__device__ bool Positive(Vec3 v)
{
    return v.X > 0 || v.Y > 0 || v.Z > 0;
}
__device__ unsigned Hash(unsigned value)
{
    constexpr unsigned AvalancheA = 0x7feb352du, AvalancheB = 0x846ca68bu;
    constexpr unsigned HalfWord = 16, MiddleShift = 15;
    value ^= value >> HalfWord;
    value *= AvalancheA;
    value ^= value >> MiddleShift;
    value *= AvalancheB;
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
    const Layout::Controls& Control;
    unsigned* Error;
};
__device__ ArvoCudaEmission::Domain Domain(const SceneView& scene)
{
    return {Native(scene.Control.WorldCenter), scene.Control.WorldRadius};
}
__device__ ArvoCudaCamera::Frame CameraFrame(const FrameConstants& frame)
{
    return {Native(frame.CameraPosition),
            Native(frame.CameraForward),
            Native(frame.CameraRight),
            Native(frame.CameraUp),
            frame.HalfWidth,
            frame.HalfHeight,
            frame.LensRadius,
            frame.FocusDistance};
}
__device__ Vec3 Offset(const Vertex& v, Vec3 direction)
{
    if (v.Type == Layout::VolumeVertex) return v.Position;
    return Add(v.Position, Multiply(v.Geometry, Dot(v.Geometry, direction) >= 0 ? RayEpsilon : -RayEpsilon));
}
__device__ float LinearChannel(unsigned byte)
{
    constexpr float ByteMaximum = 255, SrgbThreshold = 0.04045f, SrgbScale = 12.92f;
    constexpr float SrgbOffset = 0.055f, SrgbDivisor = 1.055f, SrgbExponent = 2.4f;
    float value = byte / ByteMaximum;
    return value <= SrgbThreshold ? value / SrgbScale
                                  : powf((value + SrgbOffset) / SrgbDivisor, SrgbExponent);
}
__device__ Vec3 Texel(const SceneView& scene, const PackedTexture& texture, long long x, long long y)
{
    long long column = x % texture.Width, row = y % texture.Height;
    if (column < 0)
        column += texture.Width;
    if (row < 0)
        row += texture.Height;
    unsigned packed = scene.Texels[texture.OffsetTexels + row * texture.Width + column];
    constexpr unsigned ByteMask = 0xffu, GreenShift = 8, BlueShift = 16;
    return {LinearChannel(packed & ByteMask), LinearChannel((packed >> GreenShift) & ByteMask),
            LinearChannel((packed >> BlueShift) & ByteMask)};
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
    float u = hit.TexCoordX - floorf(hit.TexCoordX), flipped = 1 - hit.TexCoordY;
    float v = flipped - floorf(flipped), x = u * texture.Width - 0.5f, y = v * texture.Height - 0.5f;
    long long ix = static_cast<long long>(floorf(x)), iy = static_cast<long long>(floorf(y));
    float fx = x - floorf(x), fy = y - floorf(y);
    Vec3 top =
        Add(Multiply(Texel(scene, texture, ix, iy), 1 - fx), Multiply(Texel(scene, texture, ix + 1, iy), fx));
    Vec3 bottom = Add(Multiply(Texel(scene, texture, ix, iy + 1), 1 - fx),
                      Multiply(Texel(scene, texture, ix + 1, iy + 1), fx));
    return Multiply(hit.Color, Add(Multiply(top, 1 - fy), Multiply(bottom, fy)));
}
__device__ PhysicalSurface ScatteringFrame(const SceneView& scene, const Vertex& vertex, Vec3 view,
                                           unsigned mode)
{
    Hit hit{};
    hit.Normal = Dot(vertex.Normal, view) < 0 ? Multiply(vertex.Normal, -1) : vertex.Normal;
    hit.GeometricNormal = vertex.Geometry;
    hit.FrontFace = Dot(vertex.Geometry, view) > 0;
    PackedMaterial material = scene.Materials[vertex.Material];
    if (material.Model == LegacyMaterial)
        material.Model = DiffuseMaterial;
    return MakePhysicalSurface(material, hit, view, vertex.Color, mode);
}
__device__ PhysicalEvaluation Evaluate(const SceneView& scene, const Vertex& vertex, Vec3 view, Vec3 outgoing,
                                       unsigned mode)
{
    if (vertex.Type == Layout::VolumeVertex)
    {
        float anisotropy = scene.Materials[vertex.Material].ConductorIndex.W;
        float phase = SubsurfacePhase(Dot(Multiply(view, -1), outgoing), anisotropy);
        float pdf = (scene.Frame.TransportFlags & ImportanceBit) ? phase : 1 / (4 * Pi);
        return {{phase, phase, phase}, pdf, pdf};
    }
    auto surface = ScatteringFrame(scene, vertex, view, mode);
    auto result = EvaluatePhysical(surface, outgoing, (scene.Frame.TransportFlags & ImportanceBit) != 0);
    const auto& material = scene.Materials[vertex.Material];
    if (material.Model == LegacyMaterial)
    {
        float diffuse = 1 - fminf(1, fmaxf(0, material.AlbedoReflectivity.W));
        result.Value = Multiply(result.Value, diffuse);
        result.Pdf *= diffuse;
        result.ReversePdf *= diffuse;
    }
    return result;
}
__device__ PhysicalSample Scatter(const SceneView& scene, const Vertex& vertex, unsigned mode, float2 u,
                                  float lobe)
{
    if (vertex.Type == Layout::VolumeVertex)
    {
        auto phase = SubsurfaceSampleDirection(Native(Multiply(vertex.Previous, -1)),
            scene.Materials[vertex.Material].ConductorIndex.W, u,
            (scene.Frame.TransportFlags & ImportanceBit) != 0);
        return {Vector(phase.Direction), {phase.Weight, phase.Weight, phase.Weight}, phase.Pdf,
                phase.Pdf, 1, ArvoCuda::ScatteringGlossy};
    }
    auto surface = ScatteringFrame(scene, vertex, vertex.Previous, mode);
    const auto& material = scene.Materials[vertex.Material];
    float mirror = material.Model == LegacyMaterial ? fminf(1, fmaxf(0, material.AlbedoReflectivity.W)) : 0;
    if (lobe < mirror)
    {
        Vec3 direction = Reflect(Multiply(vertex.Previous, -1), surface.Normal);
        if (!SurfaceHemisphere(surface, direction, false))
            return {};
        float weight = SurfaceAdjoint(surface, direction);
        return {direction, {weight, weight, weight},
                mirror,    mirror,
                1,         ArvoCuda::ScatteringReflection | ArvoCuda::ScatteringDelta};
    }
    auto result = SamplePhysical(surface, u, lobe, (scene.Frame.TransportFlags & ImportanceBit) != 0);
    if (material.Model == LegacyMaterial)
    {
        result.Pdf *= 1 - mirror;
        result.ReversePdf *= 1 - mirror;
    }
    return result;
}
__device__ unsigned FindEmitter(const SceneView& scene, unsigned primitive, bool environment)
{
    for (unsigned i = 0; i < scene.Control.EmitterCount; ++i)
        if (environment ? scene.Emitters[i].Kind == ArvoCudaEndpoint::Environment
                        : (primitive != InvalidPrimitiveIndex && scene.Emitters[i].Primitive == primitive))
            return i;
    return InvalidPrimitiveIndex;
}
__device__ bool Rectangle(const Emitter& source, const Ray& ray, float& distance)
{
    if (source.Kind != ArvoCudaEndpoint::Rectangle)
        return false;
    Vec3 normal = Vector(source.Direction);
    float denominator = Dot(normal, ray.Direction);
    if (denominator == 0)
        return false;
    float candidate = Dot(normal, Subtract(Vector(source.Origin), ray.Origin)) / denominator;
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
__device__ Vec3 Emission(const Emitter& source, Vec3 outward)
{
    if (source.Kind == ArvoCudaEndpoint::Environment)
    {
        float top = (1 - outward.Y) / 2;
        return Add(Multiply(Vector(source.BackgroundTop), top),
                   Multiply(Vector(source.BackgroundBottom), 1 - top));
    }
    if (source.Kind == ArvoCudaEndpoint::Rectangle)
        return Dot(Vector(source.Direction), outward) > 0
                   ? Multiply(Vector(source.Strength), 1 / (source.Width * source.Height))
                   : Vec3{};
    return Vector(source.Strength);
}
__device__ bool Occluded(const SceneView& scene, Ray ray, GroupStatistics& statistics)
{
    ++statistics.ShadowRays;
    Hit hit{};
    hit.Distance = ray.MaximumDistance;
    if (IntersectScene(scene.Nodes, scene.References, scene.Triangles, scene.Spheres, scene.Frame, ray, true,
                       hit, statistics.PrimitiveTests))
        return true;
    for (unsigned i = 0; i < scene.Control.EmitterCount; ++i)
        if (scene.Emitters[i].Kind == ArvoCudaEndpoint::Rectangle)
        {
            ++statistics.PrimitiveTests;
            float distance = ray.MaximumDistance;
            if (Rectangle(scene.Emitters[i], ray, distance))
                return true;
        }
    return false;
}
__device__ bool Visible(const SceneView& scene, Vec3 start, Vec3 end, GroupStatistics& statistics)
{
    Vec3 difference = Subtract(end, start);
    float distance = norm3df(difference.X, difference.Y, difference.Z);
    return distance > 2 * RayEpsilon && isfinite(distance) &&
           !Occluded(scene, {start, Multiply(difference, 1 / distance), RayEpsilon, distance - RayEpsilon},
                     statistics);
}
__device__ void Walk(const SceneView& scene, Path& path, Ray ray, Vec3 beta, SampleContext samples,
                     GroupStatistics& statistics)
{
    unsigned stream = path.Mode == ArvoCuda::TransportRadiance ? CameraWalkDimension : LightWalkDimension;
    unsigned interiorMaterial = InvalidPrimitiveIndex;
    while (path.Count < path.Limit)
    {
        if (!Finite(beta))
        {
            atomicOr(scene.Error, Layout::NumericalFailure);
            return;
        }
        if (!Positive(beta))
            return;
        if (path.Mode == ArvoCuda::TransportRadiance && path.Count == 1)
            ++statistics.PrimaryRays;
        else
            ++statistics.ReflectionRays;
        Hit hit{};
        hit.Distance = ray.MaximumDistance;
        bool found = IntersectScene(scene.Nodes, scene.References, scene.Triangles, scene.Spheres,
                                    scene.Frame, ray, false, hit, statistics.PrimitiveTests);
        if (found && hit.MaterialIndex >= scene.Frame.MaterialCount)
        {
            atomicOr(scene.Error, Layout::NumericalFailure);
            return;
        }
        unsigned rectangle = InvalidPrimitiveIndex;
        float distance = found ? hit.Distance + RayEpsilon : ray.MaximumDistance;
        for (unsigned i = 0; i < scene.Control.EmitterCount; ++i)
            if (scene.Emitters[i].Kind == ArvoCudaEndpoint::Rectangle)
            {
                ++statistics.PrimitiveTests;
                if (Rectangle(scene.Emitters[i], ray, distance))
                    rectangle = i;
            }
        Vertex& vertex = path.Vertices[path.Count++];
        vertex = {};
        vertex.Previous = Multiply(ray.Direction, -1);
        vertex.Beta = beta;
        vertex.Emitter = InvalidPrimitiveIndex;
        if (path.Count == 2 && found && !hit.FrontFace &&
            scene.Materials[hit.MaterialIndex].Optical.W > 0)
            interiorMaterial = hit.MaterialIndex;
        if (interiorMaterial != InvalidPrimitiveIndex)
        {
            const auto& interior = scene.Materials[interiorMaterial];
            float endpoint = rectangle != InvalidPrimitiveIndex ? distance :
                (found ? hit.Distance : ray.MaximumDistance);
            unsigned dimension = stream ^ ((path.Count - 1) * BounceSeedMultiplier);
            auto flight = SubsurfaceFreeFlight(interior.Optical.W,
                make_float3(interior.AlbedoReflectivity.X * interior.ConductorExtinction.W,
                            interior.AlbedoReflectivity.Y * interior.ConductorExtinction.W,
                            interior.AlbedoReflectivity.Z * interior.ConductorExtinction.W),
                Native(ray.Direction), endpoint,
                Sample2D(samples, dimension ^ InteriorFlightDimension).x);
            if (flight.Scattered)
            {
                vertex.Type = Layout::VolumeVertex;
                vertex.Position = Add(ray.Origin, Multiply(ray.Direction, flight.Distance));
                vertex.Material = interiorMaterial;
                vertex.Beta = Multiply(beta, Vector(flight.Weight));
                if (!Finite(vertex.Beta) || !Finite(vertex.Position))
                {
                    atomicOr(scene.Error, Layout::NumericalFailure);
                    return;
                }
                if (path.Count == path.Limit) return;
                auto phase = Scatter(scene, vertex, path.Mode,
                                     Sample2D(samples, dimension ^ InteriorPhaseDimension), 0);
                vertex.Pdf = phase.Pdf;
                vertex.ReversePdf = phase.ReversePdf;
                vertex.Event = phase.Event;
                if (!(phase.Pdf > 0)) return;
                beta = Multiply(vertex.Beta, phase.Weight);
                ray = {vertex.Position, phase.Direction, RayEpsilon, MaximumRayDistance};
                continue;
            }
        }
        if (rectangle != InvalidPrimitiveIndex)
        {
            vertex.Type = Layout::EmitterVertex;
            vertex.Position = Add(ray.Origin, Multiply(ray.Direction, distance));
            vertex.Geometry = Vector(scene.Emitters[rectangle].Direction);
            vertex.Emitter = rectangle;
            return;
        }
        if (!found)
        {
            vertex.Type = Layout::InfiniteVertex;
            vertex.Emitter = FindEmitter(scene, InvalidPrimitiveIndex, true);
            return;
        }
        vertex.Type = Layout::SurfaceVertex;
        vertex.Position = Add(ray.Origin, Multiply(ray.Direction, hit.Distance));
        vertex.Normal = hit.Normal;
        vertex.Geometry = hit.GeometricNormal;
        vertex.FrontFace = hit.FrontFace;
        vertex.Material = hit.MaterialIndex;
        vertex.Color = SurfaceColor(scene, hit);
        vertex.Emitter = FindEmitter(scene, hit.PrimitiveIndex, false);
        if (path.Count == path.Limit)
            return;
        unsigned dimension = stream ^ (path.Count * BounceSeedMultiplier);
        auto sample = Scatter(scene, vertex, path.Mode, Sample2D(samples, dimension),
                              Sample2D(samples, dimension ^ LobeSelectionScramble).x);
        vertex.Pdf = sample.Pdf;
        vertex.ReversePdf = sample.ReversePdf;
        vertex.Event = sample.Event;
        if (!sample.Event || !(sample.Pdf > 0))
            return;
        beta = Multiply(beta, sample.Weight);
        ray = {Offset(vertex, sample.Direction), sample.Direction, RayEpsilon, MaximumRayDistance};
        if (scene.Materials[hit.MaterialIndex].Optical.W > 0 &&
            (sample.Event & ArvoCuda::ScatteringTransmission))
            interiorMaterial = hit.FrontFace ? hit.MaterialIndex : InvalidPrimitiveIndex;
    }
}
__device__ void GeneratePair(const SceneView& scene, Layout::Pair& pair, SampleContext samples,
                             bool cameraWalk, GroupStatistics& statistics)
{
    Path& camera = pair.Camera;
    Path& light = pair.Light;
    camera.Count = light.Count = 0;
    camera.Limit = light.Limit = scene.Frame.MaxReflectionBounces + 3;
    camera.Mode = ArvoCuda::TransportRadiance;
    light.Mode = ArvoCuda::TransportImportance;
    float2 jitter = Sample2D(samples, 0);
    float2 raster = ArvoCudaCamera::PixelRaster(samples.PixelX, samples.PixelY, scene.Frame.FullWidth,
                                               scene.Frame.FullHeight, jitter);
    auto endpoint = ArvoCudaCamera::Emit(CameraFrame(scene.Frame), raster, Sample2D(samples, LensDimension));
    if (endpoint.Flags & ArvoCudaCamera::Valid)
    {
        camera.EndpointFlags = endpoint.Flags;
        camera.PositionPdf = endpoint.PositionPdf;
        camera.DirectionPdf = endpoint.DirectionPdf;
        camera.LensX = endpoint.LensPoint.x;
        camera.LensY = endpoint.LensPoint.y;
        camera.RasterX = raster.x;
        camera.RasterY = raster.y;
        camera.LaunchDirection = Vector(endpoint.Direction);
        auto& root = camera.Vertices[camera.Count++];
        root = {};
        root.Position = Vector(endpoint.Origin);
        root.Geometry = scene.Frame.CameraForward;
        root.Type = Layout::CameraVertex;
        root.Beta = {1, 1, 1};
        root.Emitter = InvalidPrimitiveIndex;
        if (cameraWalk)
            Walk(scene, camera, {root.Position, camera.LaunchDirection, RayEpsilon, MaximumRayDistance},
                 root.Beta, samples, statistics);
    }
    auto emitted = ArvoCudaEndpoint::Emit(
        scene.Emitters, scene.Control.EmitterCount, Domain(scene), Sample2D(samples, SourceDimension).x,
        Sample2D(samples, PositionDimension), Sample2D(samples, DirectionDimension));
    const auto& emission = emitted.Emission;
    if (!(emission.Flags & ArvoCudaEmission::Valid))
        return;
    light.EndpointFlags = emission.Flags;
    light.PositionPdf = emission.PositionPdf;
    light.DirectionPdf = emission.DirectionPdf;
    light.SelectionProbability = emitted.SelectionProbability;
    light.LaunchDirection = Vector(emission.Direction);
    light.LaunchWeight = Vector(emission.Weight);
    light.EndpointValue = Vector(emission.Value);
    auto& root = light.Vertices[light.Count++];
    root = {};
    root.Type = Layout::EmitterVertex;
    root.Position = Vector(emission.Origin);
    root.Geometry = Vector(emission.Normal);
    root.Emitter = emitted.EmitterIndex;
    float density =
        emission.Flags & ArvoCudaEmission::Infinite ? emission.DirectionPdf : emission.PositionPdf;
    root.Beta = Multiply(light.EndpointValue, 1 / (density * light.SelectionProbability));
    Vec3 origin = emission.Flags & ArvoCudaEmission::DeltaPosition ? root.Position
                                                                   : Offset(root, light.LaunchDirection);
    Walk(scene, light, {origin, light.LaunchDirection, RayEpsilon, MaximumRayDistance}, light.LaunchWeight,
         samples, statistics);
}
struct LightIdentity
{
    unsigned Index, Flags;
    Vec3 Direction;
};
__device__ const Vertex& CompletedVertex(const Path& light, const Path& camera, unsigned s, unsigned t,
                                         unsigned index, const Vertex& cameraRoot)
{
    if (index + 1 == s + t)
        return cameraRoot;
    return index < s ? light.Vertices[index] : camera.Vertices[s + t - 1 - index];
}
__device__ Vec3 SegmentDirection(const Vertex& from, const Vertex& to, bool targetInfinite, Vec3 launch)
{
    return targetInfinite ? Multiply(launch, -1) : Normalize(Subtract(to.Position, from.Position));
}
__device__ Mis::Density Angular(float value, unsigned event)
{
    return Mis::Make(value, event & ArvoCuda::ScatteringDelta ? Mis::Dirac : Mis::SolidAngle);
}
__device__ unsigned SegmentInterior(const SceneView& scene, const Vertex& from, const Vertex& to)
{
    // Scene validation guarantees separated, outward-facing closed SSS solids.
    if (from.Type == Layout::VolumeVertex) return from.Material;
    if (to.Type == Layout::VolumeVertex) return to.Material;
    Vec3 segment = Subtract(to.Position, from.Position);
    if (from.Type == Layout::SurfaceVertex && scene.Materials[from.Material].Optical.W > 0 &&
        Dot(from.Geometry, segment) < 0)
        return from.Material;
    if (to.Type == Layout::SurfaceVertex && scene.Materials[to.Material].Optical.W > 0 &&
        Dot(to.Geometry, Multiply(segment, -1)) < 0)
        return to.Material;
    return InvalidPrimitiveIndex;
}
__device__ float Transmittance(const SceneView& scene, const Vertex& from, const Vertex& to)
{
    unsigned interior = SegmentInterior(scene, from, to);
    if (interior == InvalidPrimitiveIndex) return 1;
    Vec3 segment = Subtract(to.Position, from.Position);
    return expf(-norm3df(segment.X, segment.Y, segment.Z) /
                scene.Materials[interior].Optical.W);
}
__device__ Mis::Density Destination(const SceneView& scene, Mis::Density angular,
                                   const Vertex& from, const Vertex& to, bool infinite)
{
    unsigned destination = infinite ? Mis::InfiniteDestination :
        (to.Type == Layout::VolumeVertex ? Mis::VolumeDestination : Mis::SurfaceDestination);
    Vec3 segment = Subtract(to.Position, from.Position);
    auto density = Mis::Convert(angular, Native(segment), Native(to.Geometry), destination);
    if (!Mis::Exists(density) || infinite) return density;
    unsigned interior = SegmentInterior(scene, from, to);
    if (interior == InvalidPrimitiveIndex)
        return destination == Mis::VolumeDestination ? Mis::Density{} : density;
    float meanFreePath = scene.Materials[interior].Optical.W;
    float distance = norm3df(segment.X, segment.Y, segment.Z);
    density.Log2Value -= distance / meanFreePath * log2f(expf(1.0f));
    if (destination == Mis::VolumeDestination) density.Log2Value -= log2f(meanFreePath);
    return isfinite(density.Log2Value) ? density : Mis::Density{};
}
__device__ ArvoCudaCamera::Evaluation CameraPdf(const Path& camera)
{
    return {make_float2(camera.RasterX, camera.RasterY), 0, camera.PositionPdf, camera.DirectionPdf,
            camera.EndpointFlags};
}
// Densities are reconstructed on the connected path, including unsampled cuts.
// The stored event at a replaced connection is discarded: a previous delta
// continuation must not suppress a newly connected, non-delta lobe.
__device__ float Weight(const SceneView& scene, const Path& light, const Path& camera, unsigned s, unsigned t,
                        const Vertex& cameraRoot, ArvoCudaCamera::Evaluation cameraPdf,
                        LightIdentity identity)
{
    unsigned count = s + t;
    if (count == 2)
        return s == 0 ? 1.0f : 0.0f;
    bool infinite = (identity.Flags & ArvoCudaEmission::Infinite) != 0;
    Mis::Vertex vertices[Mis::MaximumVertices]{};
    for (unsigned i = 1; i + 1 < count; ++i)
    {
        const auto& previous = CompletedVertex(light, camera, s, t, i - 1, cameraRoot);
        const auto& current = CompletedVertex(light, camera, s, t, i, cameraRoot);
        const auto& next = CompletedVertex(light, camera, s, t, i + 1, cameraRoot);
        float forward, reverse;
        unsigned event = 0;
        if ((s > 0 && i == s - 1) || i == s)
        {
            auto evaluated = Evaluate(
                scene, current, SegmentDirection(current, previous, i == 1 && infinite, identity.Direction),
                SegmentDirection(current, next, false, identity.Direction), ArvoCuda::TransportImportance);
            forward = evaluated.Pdf;
            reverse = evaluated.ReversePdf;
        }
        else
        {
            forward = i < s ? current.Pdf : current.ReversePdf;
            reverse = i < s ? current.ReversePdf : current.Pdf;
            event = current.Event;
        }
        vertices[i].Flags = event & ArvoCuda::ScatteringDelta ? Mis::DeltaScattering : 0;
        vertices[i + 1].Forward = Destination(scene, Angular(forward, event), current, next, false);
        vertices[i - 1].Reverse = Destination(scene, Angular(reverse, event), current, previous,
                                              i == 1 && infinite);
    }
    const auto& root = CompletedVertex(light, camera, s, t, 0, cameraRoot);
    const auto& first = CompletedVertex(light, camera, s, t, 1, cameraRoot);
    Vec3 emitted = infinite ? identity.Direction : Normalize(Subtract(first.Position, root.Position));
    auto emitterPdf = ArvoCudaEndpoint::Pdf(scene.Emitters, scene.Control.EmitterCount, Domain(scene),
                                            identity.Index, Native(root.Geometry), Native(emitted));
    bool deltaPosition = (emitterPdf.Flags & ArvoCudaEmission::DeltaPosition) != 0;
    bool deltaDirection = (emitterPdf.Flags & ArvoCudaEmission::DeltaDirection) != 0;
    auto position =
        Mis::Make(deltaPosition ? 1 : emitterPdf.Position, deltaPosition ? Mis::Dirac : Mis::Area);
    auto direction =
        Mis::Make(deltaDirection ? 1 : emitterPdf.Direction, deltaDirection ? Mis::Dirac : Mis::SolidAngle);
    vertices[1].Forward = infinite ? Mis::ProjectDisk(position, Native(first.Geometry), Native(emitted))
                                   : Destination(scene, direction, root, first, false);
    vertices[0].Flags = deltaPosition || deltaDirection ? Mis::EndpointNotIntersectable : 0;
    const auto& last = CompletedVertex(light, camera, s, t, count - 2, cameraRoot);
    vertices[count - 2].Reverse =
        Destination(scene, Mis::Make(cameraPdf.DirectionPdf, Mis::SolidAngle), cameraRoot, last, false);
    bool pinhole = (cameraPdf.Flags & ArvoCudaCamera::DeltaPosition) != 0;
    vertices[count - 1].Flags = pinhole ? Mis::EndpointNotIntersectable : 0;
    Mis::Strategy strategies[Mis::MaximumStrategies]{};
    for (unsigned cut = 0; cut < count; ++cut)
    {
        auto& strategy = strategies[cut];
        strategy.Enabled = cut <= light.Limit && count - cut <= camera.Limit;
        strategy.LightEndpoint = cut == 0 ? Mis::Make(1, Mis::Dirac) : (infinite ? direction : position);
        strategy.LightSelection = cut == 0 ? 1 : emitterPdf.SelectionProbability;
        strategy.CameraEndpoint =
            Mis::Make(pinhole ? 1 : cameraPdf.PositionPdf, pinhole ? Mis::Dirac : Mis::Area);
        strategy.SampleCount = 1;
    }
    unsigned heuristic = (scene.Frame.TransportFlags >> MisShift) & MisMask;
    return Mis::Evaluate(vertices, count, strategies, heuristic).Values[s];
}
__device__ LightIdentity Identity(const Path& light)
{
    return {light.Vertices[0].Emitter, light.EndpointFlags, light.LaunchDirection};
}
__device__ Vec3 EmitterHit(const SceneView& scene, const Path& light, const Path& camera, unsigned t)
{
    const auto& end = camera.Vertices[t - 1];
    if (end.Emitter >= scene.Control.EmitterCount)
        return {};
    const auto& emitter = scene.Emitters[end.Emitter];
    Vec3 value = Multiply(end.Beta, Emission(emitter, end.Previous));
    if (!Positive(value))
        return {};
    auto pdf = ArvoCudaEndpoint::Pdf(scene.Emitters, scene.Control.EmitterCount, Domain(scene), end.Emitter,
                                     Native(end.Geometry), Native(end.Previous));
    return Multiply(value, Weight(scene, light, camera, 0, t, camera.Vertices[0], CameraPdf(camera),
                                  {end.Emitter, pdf.Flags, end.Previous}));
}
__device__ Vec3 DirectEndpoint(const SceneView& scene, const Path& light, const Path& camera, unsigned t,
                               GroupStatistics& statistics)
{
    const auto& vertex = camera.Vertices[t - 1];
    if (vertex.Type != Layout::SurfaceVertex && vertex.Type != Layout::VolumeVertex)
        return {};
    const auto& root = light.Vertices[0];
    bool infinite = (light.EndpointFlags & ArvoCudaEmission::Infinite) != 0;
    bool point = (light.EndpointFlags & ArvoCudaEmission::DeltaPosition) != 0;
    Vec3 direction;
    float geometry = 1;
    if (infinite)
        direction = Multiply(light.LaunchDirection, -1);
    else
    {
        Vec3 segment = Subtract(root.Position, vertex.Position);
        float distanceSquared = Dot(segment, segment);
        if (!(distanceSquared > 0) || !isfinite(distanceSquared))
            return {};
        direction = Multiply(segment, rsqrtf(distanceSquared));
        geometry = (point ? 1 : fabsf(Dot(root.Geometry, direction))) / distanceSquared;
    }
    auto evaluated = Evaluate(scene, vertex, vertex.Previous, direction, ArvoCuda::TransportRadiance);
    float density = light.SelectionProbability * (infinite ? light.DirectionPdf : light.PositionPdf);
    Vec3 value = Multiply(Multiply(vertex.Beta, evaluated.Value),
                          Emission(scene.Emitters[root.Emitter], Multiply(direction, -1)));
    float cosine = vertex.Type == Layout::VolumeVertex ? 1 : fabsf(Dot(vertex.Normal, direction));
    value = Multiply(value, cosine * geometry / density *
                            (infinite ? 1 : Transmittance(scene, vertex, root)));
    if (!Positive(value))
        return {};
    if (infinite)
    {
        if (Occluded(scene, {Offset(vertex, direction), direction, RayEpsilon, MaximumRayDistance},
                     statistics))
            return {};
    }
    else if (!Visible(scene, Offset(vertex, direction),
                      point ? root.Position : Offset(root, Multiply(direction, -1)), statistics))
        return {};
    return Multiply(
        value, Weight(scene, light, camera, 1, t, camera.Vertices[0], CameraPdf(camera), Identity(light)));
}
struct Contribution
{
    Vec3 Value;
    float Reserved;
    float2 Raster;
};
__device__ Contribution Connect(const SceneView& scene, const Path& light, const Path& camera, unsigned s,
                                unsigned t, float2 lens, GroupStatistics& statistics)
{
    const auto& a = light.Vertices[s - 1];
    const auto& b = camera.Vertices[t - 1];
    if ((a.Type != Layout::SurfaceVertex && a.Type != Layout::VolumeVertex) ||
        (t > 1 && b.Type != Layout::SurfaceVertex && b.Type != Layout::VolumeVertex))
        return {};
    Vertex cameraRoot = camera.Vertices[0];
    auto cameraPdf = CameraPdf(camera);
    Vec3 towardCamera, target, value;
    float2 raster = cameraPdf.Raster;
    if (t == 1)
    {
        auto connection = ArvoCudaCamera::Connect(CameraFrame(scene.Frame), Native(a.Position), lens);
        if (!(connection.Flags & ArvoCudaCamera::Valid))
            return {};
        towardCamera = Vector(connection.Direction);
        target = Vector(connection.Position);
        auto bsdf = Evaluate(scene, a, a.Previous, towardCamera, ArvoCuda::TransportImportance);
        cameraRoot.Position = target;
        float cosine = a.Type == Layout::VolumeVertex ? 1 : fabsf(Dot(a.Normal, towardCamera));
        value = Multiply(Multiply(a.Beta, bsdf.Value),
                         cosine * connection.Weight * Transmittance(scene, a, cameraRoot));
        cameraPdf = ArvoCudaCamera::Evaluate(CameraFrame(scene.Frame), connection.LensPoint,
                                             Native(Multiply(towardCamera, -1)));
        raster = connection.Raster;
    }
    else
    {
        Vec3 segment = Subtract(b.Position, a.Position);
        float squared = Dot(segment, segment);
        if (!(squared > 0) || !isfinite(squared))
            return {};
        towardCamera = Multiply(segment, rsqrtf(squared));
        auto lightBsdf = Evaluate(scene, a, a.Previous, towardCamera, ArvoCuda::TransportImportance);
        auto cameraBsdf =
            Evaluate(scene, b, b.Previous, Multiply(towardCamera, -1), ArvoCuda::TransportRadiance);
        float cosineA = a.Type == Layout::VolumeVertex ? 1 : fabsf(Dot(a.Normal, towardCamera));
        float cosineB = b.Type == Layout::VolumeVertex ? 1 : fabsf(Dot(b.Normal, towardCamera));
        float geometry = cosineA * cosineB / squared;
        value = Multiply(Multiply(Multiply(a.Beta, lightBsdf.Value), Multiply(b.Beta, cameraBsdf.Value)),
                         geometry * Transmittance(scene, a, b));
        target = Offset(b, Multiply(towardCamera, -1));
    }
    if (!Positive(value) || !Visible(scene, Offset(a, towardCamera), target, statistics))
        return {};
    float weight = Weight(scene, light, camera, s, t, cameraRoot, cameraPdf, Identity(light));
    return {Multiply(value, weight), 0, raster};
}
__device__ bool LocalPixel(const FrameConstants& frame, unsigned x, unsigned y, unsigned& pixel)
{
    if (x < frame.TileX || y < frame.TileY || x - frame.TileX >= frame.Width ||
        y - frame.TileY >= frame.Height)
        return false;
    pixel = (y - frame.TileY) * frame.Width + x - frame.TileX;
    return true;
}
__device__ void AddContribution(unsigned* pass, unsigned pixel, Vec3 value, unsigned* error)
{
    unsigned* channels = pass + pixel * Layout::PassWords;
    const bool red = ArvoCudaRadiance::Add(channels, value.X);
    const bool green = ArvoCudaRadiance::Add(channels + ArvoCudaRadiance::WordCount, value.Y);
    const bool blue = ArvoCudaRadiance::Add(channels + 2 * ArvoCudaRadiance::WordCount, value.Z);
    if (!red || !green || !blue)
        atomicOr(error, Layout::NumericalFailure);
}
// Each batch uses independent absolute pixel/sample identities. Cropped renders
// still generate every full-film light path, preserving incoming t=1 splats.
// Exact integer sums retain the same result for every GPU scheduling order.
extern "C" __global__ void BidirectionalPairsKernel(
    const PackedAccelerationNode* nodes, const PackedPrimitiveRef* references,
    const PackedTriangle* triangles, const PackedSphere* spheres, const PackedMaterial* materials,
    const PackedTexture* textures, const unsigned* texels, const Emitter* emitters,
    const FrameConstants* frame, const Layout::Controls* controls, Layout::Pair* scratch, unsigned* pass,
    GroupStatistics* statistics, unsigned* error)
{
    unsigned local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= controls->PathCount)
        return;
    SceneView scene{nodes,  references, triangles, spheres,   materials, textures,
                    texels, emitters,   *frame,    *controls, error};
    unsigned global = controls->PathBegin + local, x = global % frame->FullWidth,
             y = global / frame->FullWidth;
    unsigned pixel = 0;
    bool owned = LocalPixel(*frame, x, y, pixel);
    SampleContext samples = MakeSampleContext(x, y, frame->FullWidth, frame->SampleBegin, frame->TotalSamples,
                                              frame->SamplingSequence, frame->SamplingSeed);
    GroupStatistics counts{};
    auto& pair = scratch[local];
    GeneratePair(scene, pair, samples, owned, counts);
    const auto& camera = pair.Camera;
    const auto& light = pair.Light;
    Vec3 cameraValue{};
    for (unsigned t = 2; t <= camera.Count; ++t)
        cameraValue = Add(cameraValue, EmitterHit(scene, light, camera, t));
    if (light.Count && camera.Count)
    {
        for (unsigned t = 2; t <= camera.Count && t < camera.Limit; ++t)
            cameraValue = Add(cameraValue, DirectEndpoint(scene, light, camera, t, counts));
        for (unsigned s = 2; s <= light.Count; ++s)
            for (unsigned t = 1; t <= camera.Count && s + t <= camera.Limit; ++t)
            {
                float2 lens = Sample2D(samples, ConnectionLensDimension ^ (s * BounceSeedMultiplier));
                auto result = Connect(scene, light, camera, s, t, lens, counts);
                if (t > 1)
                    cameraValue = Add(cameraValue, result.Value);
                else if (Positive(result.Value) || !Finite(result.Value))
                {
                    unsigned target;
                    if (ArvoCudaCamera::InSquare(result.Raster) &&
                        LocalPixel(*frame, unsigned(result.Raster.x * frame->FullWidth),
                                   unsigned(result.Raster.y * frame->FullHeight), target))
                        AddContribution(pass, target, result.Value, error);
                }
            }
    }
    if (owned)
        AddContribution(pass, pixel, cameraValue, error);
    statistics[local] = counts;
}
// A pass is a correlated collection of camera contributions and light splats.
// It is ONE film sample. A host cancellation between batches discards the pass.
extern "C" __global__ void BidirectionalCommitKernel(const FrameConstants* frame, const unsigned* pass,
                                                     Vec4* film, unsigned* sampleCounts, unsigned* error)
{
    unsigned pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= frame->Width * frame->Height)
        return;
    const unsigned* channels = pass + pixel * Layout::PassWords;
    Vec3 value{ArvoCudaRadiance::Round(channels),
               ArvoCudaRadiance::Round(channels + ArvoCudaRadiance::WordCount),
               ArvoCudaRadiance::Round(channels + 2 * ArvoCudaRadiance::WordCount)};
    if (!Finite(value))
    {
        atomicOr(error, Layout::NumericalFailure);
        return;
    }
    Vec4 moments = film[pixel];
    unsigned count = sampleCounts[pixel];
    AccumulateFilmSample(moments, count, value);
    if (!Finite(ToVec3(moments)) || !isfinite(moments.W))
    {
        atomicOr(error, Layout::NumericalFailure);
        return;
    }
    film[pixel] = moments;
    sampleCounts[pixel] = count;
}
} // namespace
