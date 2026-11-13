// Independent CUDA VCM transport. Georgiev et al. (2012), "Light Transport
// Simulation with Vertex Connection and Merging"; see the adjacent MIS source
// for the complete-path measures. This file owns both walk directions and their
// connection/merging estimators and progressive frame kernels. Other integrators are not
// included. Only CUDA geometry, sampling, camera, emitter and BSDF leaves recur.
//
// Walks retain pre-scattering throughput, actual sampled directions, conditional
// endpoint densities and generalized forward/reverse arrivals. Delta branch
// masses keep their geometry factors. Infinite-source disk projection differs
// from angular-to-area conversion. Fixed depth avoids unaccounted roulette PDFs.
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cfloat>
#include "VertexConnectionMergingMis.cu"
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
namespace V = Arvo::RayTracing::CudaVcm;
using Vec3 = V::Vector;
using Emitter = ArvoCudaEndpoint::PackedEmitter;
constexpr unsigned SphereBit = 0x80000000u, InvalidPrimitiveIndex = V::InvalidIndex;
constexpr float Pi = CUDART_PI_F, RayEpsilon = 1e-4f, MaximumRayDistance = 1e30f;
constexpr float BasisParallelThreshold = 0.999f, MinimumDirectionLengthSquared = 1e-12f,
                MinimumTriangleDeterminant = 1e-8f;
constexpr unsigned ImportanceBit = 1, RouletteBit = 4;
constexpr unsigned PixelSeedMultiplier = 9781, SampleSeedMultiplier = 6271,
                   SecondDimensionScramble = 0x68bc21ebu;
constexpr unsigned CameraDomain = 0x56434d43u, LightDomain = 0x56434d4cu, PassDomain = 0x56434d50u;
constexpr unsigned SourceDomain = 0x53524345u, PositionDomain = 0x504f5349u, DirectionDomain = 0x44495245u;
constexpr unsigned LensDomain = 0x4c454e53u, BounceDomain = 0x9e3779b9u, LobeDomain = 0xa511e9b3u;
constexpr unsigned InteriorFlightDomain = 0x5643464cu, InteriorPhaseDomain = 0x56435048u;

// Existing CUDA scene upload ABI. These records contain no transport algorithm.
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
    unsigned MaterialIndex, Flags, Reserved[2];
};
struct alignas(16) PackedSphere
{
    Vec4 CenterRadius;
    unsigned MaterialIndex, Reserved[3];
};
#include "MaterialTypes.cuh"
struct alignas(16) PackedTexture
{
    unsigned OffsetTexels, Width, Height, Reserved;
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
static_assert(sizeof(PackedAccelerationNode) == 48 && sizeof(PackedTriangle) == 192 &&
              sizeof(PackedSphere) == 32 && sizeof(PackedMaterial) == 128);
static_assert(sizeof(FrameConstants) == 272 && offsetof(FrameConstants, InstanceRecords) == 224 &&
              offsetof(FrameConstants, LensRadius) == 248 && sizeof(GroupStatistics) == 32);

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
    const float length = norm3df(v.X, v.Y, v.Z);
    return length > 0 ? Multiply(v, 1 / length) : Vec3{};
}
__device__ float Component(Vec3 v, unsigned axis)
{
    return axis == 0 ? v.X : axis == 1 ? v.Y : v.Z;
}
__device__ bool Finite(Vec3 v)
{
    return isfinite(v.X) && isfinite(v.Y) && isfinite(v.Z);
}
__device__ bool Energy(Vec3 v)
{
    return Finite(v) && v.X >= 0 && v.Y >= 0 && v.Z >= 0;
}
__device__ bool Nonzero(Vec3 v)
{
    return v.X > 0 || v.Y > 0 || v.Z > 0;
}
__device__ unsigned Hash(unsigned value)
{
    constexpr unsigned First = 0x7feb352du, Second = 0x846ca68bu, OuterShift = 16, MiddleShift = 15;
    value ^= value >> OuterShift;
    value *= First;
    value ^= value >> MiddleShift;
    value *= Second;
    return value ^ (value >> OuterShift);
}
__device__ float Random01(unsigned value)
{
    constexpr unsigned MantissaMask = 0xffffffu;
    constexpr float MantissaRange = 16777216.f;
    return float(Hash(value) & MantissaMask) / MantissaRange;
}
#include "Sequence.cu"
#include "Triangle.cu"
#include "Sphere.cu"
#include "Bvh.cu"
#include "KdTree.cu"
#include "Bih.cu"
#include "Dispatch.cu"
#include "SurfaceScattering.cu"

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
    const V::WalkControls& Control;
};
__device__ unsigned SegmentInterior(const SceneView& scene, const V::PathVertex& source,
                                    const V::PathVertex& target)
{
    if (source.Kind == V::VolumeVertex) return source.Material;
    if (target.Kind == V::VolumeVertex) return target.Material;
    if (source.Kind == V::LightVertex && source.Reserved[0] < scene.Frame.MaterialCount &&
        scene.Materials[source.Reserved[0]].Optical.W > 0)
        return source.Reserved[0];
    if (target.Kind == V::LightVertex && target.Reserved[0] < scene.Frame.MaterialCount &&
        scene.Materials[target.Reserved[0]].Optical.W > 0)
        return target.Reserved[0];
    const Vec3 edge = Subtract(target.Position, source.Position);
    if (source.Kind == V::SurfaceVertex && scene.Materials[source.Material].Optical.W > 0 &&
        Dot(source.Geometry, edge) < 0)
        return source.Material;
    if (target.Kind == V::SurfaceVertex && scene.Materials[target.Material].Optical.W > 0 &&
        Dot(target.Geometry, Multiply(edge, -1)) < 0)
        return target.Material;
    return V::InvalidIndex;
}
__device__ double SegmentTransmittance(const SceneView& scene, const V::PathVertex& source,
                                       const V::PathVertex& target)
{
    const unsigned medium = SegmentInterior(scene, source, target);
    if (medium == V::InvalidIndex) return 1;
    const Vec3 edge = Subtract(target.Position, source.Position);
    return exp(-norm3d(edge.X, edge.Y, edge.Z) / scene.Materials[medium].Optical.W);
}
__device__ V::Density EdgeDensity(const SceneView& scene, V::Density angular,
                                 const V::PathVertex& source, const V::PathVertex& target)
{
    if (!V::HasDensity(angular)) return {};
    const Vec3 edge = Subtract(target.Position, source.Position);
    const double distance = norm3d(edge.X, edge.Y, edge.Z);
    if (!(distance > 0) || !isfinite(distance)) return {};
    V::Density converted = target.Kind == V::VolumeVertex
        ? V::Density{angular.Log2Value - 2 * log2(distance), 1, 0}
        : V::AreaArrival(angular, Native(edge), Native(target.Geometry));
    if (!V::HasDensity(converted)) return {};
    const unsigned medium = SegmentInterior(scene, source, target);
    if (medium == V::InvalidIndex)
        return target.Kind == V::VolumeVertex ? V::Density{} : converted;
    const double meanFreePath = scene.Materials[medium].Optical.W;
    converted.Log2Value -= distance / meanFreePath * log2(exp(1.0));
    if (target.Kind == V::VolumeVertex)
        converted.Log2Value -= log2(meanFreePath);
    return isfinite(converted.Log2Value) ? converted : V::Density{};
}
__device__ Vec3 Depart(const V::PathVertex& vertex, Vec3 direction)
{
    if (vertex.Kind == V::VolumeVertex) return vertex.Position;
    return Add(vertex.Position,
               Multiply(vertex.Geometry, Dot(vertex.Geometry, direction) >= 0 ? RayEpsilon : -RayEpsilon));
}
__device__ float DecodeChannel(unsigned channel)
{
    constexpr float MaximumByte = 255, LinearLimit = 0.04045f, LinearSlope = 12.92f, Offset = 0.055f,
                    Scale = 1.055f, Exponent = 2.4f;
    const float value = channel / MaximumByte;
    return value <= LinearLimit ? value / LinearSlope : powf((value + Offset) / Scale, Exponent);
}
__device__ Vec3 ReadTexel(const SceneView& scene, const PackedTexture& texture, int column, int row)
{
    const unsigned x = unsigned((column % int(texture.Width) + int(texture.Width)) % int(texture.Width));
    const unsigned y = unsigned((row % int(texture.Height) + int(texture.Height)) % int(texture.Height));
    const unsigned packed = scene.Texels[texture.OffsetTexels + y * texture.Width + x];
    constexpr unsigned ByteMask = 0xff, GreenShift = 8, BlueShift = 16;
    return {DecodeChannel(packed & ByteMask), DecodeChannel((packed >> GreenShift) & ByteMask),
            DecodeChannel((packed >> BlueShift) & ByteMask)};
}
__device__ Vec3 SurfaceColor(const SceneView& scene, const Hit& hit)
{
    const unsigned textureIndex = scene.Materials[hit.MaterialIndex].DiffuseTextureIndex;
    if (!hit.HasTexCoord || textureIndex >= scene.Frame.TextureCount || !isfinite(hit.TexCoordX) ||
        !isfinite(hit.TexCoordY))
        return hit.Color;
    const auto texture = scene.Textures[textureIndex];
    if (!texture.Width || !texture.Height || texture.OffsetTexels > scene.Frame.TextureTexelCount ||
        texture.Height > (scene.Frame.TextureTexelCount - texture.OffsetTexels) / texture.Width)
        return hit.Color;
    const float wrappedU = hit.TexCoordX - floorf(hit.TexCoordX);
    const float wrappedV = 1 - hit.TexCoordY - floorf(1 - hit.TexCoordY);
    const float x = wrappedU * texture.Width - 0.5f, y = wrappedV * texture.Height - 0.5f;
    const int column = int(floorf(x)), row = int(floorf(y));
    const float horizontal = x - floorf(x), vertical = y - floorf(y);
    Vec3 color{};
    for (int j = 0; j < 2; ++j)
        for (int i = 0; i < 2; ++i)
        {
            const float weight = (i ? horizontal : 1 - horizontal) * (j ? vertical : 1 - vertical);
            color = Add(color, Multiply(ReadTexel(scene, texture, column + i, row + j), weight));
        }
    return Multiply(hit.Color, color);
}
__device__ bool IntersectRectangle(const Emitter& emitter, const Ray& ray, float& maximum)
{
    const Vec3 normal = Vector(emitter.Direction);
    const float denominator = Dot(normal, ray.Direction);
    if (denominator == 0)
        return false;
    const float distance = Dot(Subtract(Vector(emitter.Origin), ray.Origin), normal) / denominator;
    if (!(distance > ray.MinimumDistance && distance < ray.MaximumDistance && distance <= maximum))
        return false;
    const Vec3 helper = fabsf(normal.Y) < BasisParallelThreshold ? Vec3{0, 1, 0} : Vec3{1, 0, 0};
    const Vec3 horizontal = Normalize(Cross(helper, normal)), vertical = Cross(normal, horizontal);
    const Vec3 displacement =
        Subtract(Add(ray.Origin, Multiply(ray.Direction, distance)), Vector(emitter.Origin));
    if (fabsf(Dot(displacement, horizontal)) > emitter.Width / 2 ||
        fabsf(Dot(displacement, vertical)) > emitter.Height / 2)
        return false;
    maximum = distance;
    return true;
}
__device__ V::PathVertex NextVertex(const SceneView& scene, const Ray& ray, GroupStatistics& statistics)
{
    V::PathVertex vertex{};
    vertex.Emitter = vertex.Primitive = V::InvalidIndex;
    vertex.Material = V::InvalidIndex;
    vertex.Reserved[0] = V::InvalidIndex;
    vertex.ToPrevious = Multiply(ray.Direction, -1);
    Hit hit{};
    hit.Distance = ray.MaximumDistance;
    const bool found = IntersectScene(scene.Nodes, scene.References, scene.Triangles, scene.Spheres,
                                      scene.Frame, ray, false, hit, statistics.PrimitiveTests);
    if (found && !hit.FrontFace && scene.Materials[hit.MaterialIndex].Optical.W > 0)
        vertex.Reserved[0] = hit.MaterialIndex;
    const bool proxy = found && scene.Materials[hit.MaterialIndex].SampledEmitter != 0;
    float nearest = found ? hit.Distance + (proxy ? RayEpsilon : 0) : ray.MaximumDistance;
    for (unsigned source = 0; source < scene.Control.Sources; ++source)
        if (scene.Emitters[source].Kind == ArvoCudaEndpoint::Rectangle)
        {
            ++statistics.PrimitiveTests;
            if (IntersectRectangle(scene.Emitters[source], ray, nearest))
                vertex.Emitter = source;
        }
    if (vertex.Emitter != V::InvalidIndex)
    {
        vertex.Kind = V::LightVertex;
        vertex.Position = Add(ray.Origin, Multiply(ray.Direction, nearest));
        vertex.Normal = vertex.Geometry = Vector(scene.Emitters[vertex.Emitter].Direction);
        return vertex;
    }
    vertex.Kind = found ? V::SurfaceVertex : V::EnvironmentVertex;
    if (found)
    {
        // The proxy allowance is only a rectangle comparison tolerance. An
        // uncovered proxy surface keeps its actual geometric hit position.
        vertex.Position = Add(ray.Origin, Multiply(ray.Direction, hit.Distance));
        vertex.Material = hit.MaterialIndex;
        vertex.Primitive = hit.PrimitiveIndex;
        vertex.Normal = hit.Normal;
        vertex.Geometry = hit.GeometricNormal;
        vertex.FrontFace = hit.FrontFace;
        vertex.Color = SurfaceColor(scene, hit);
    }
    for (unsigned source = 0; source < scene.Control.Sources; ++source)
    {
        const auto& emitter = scene.Emitters[source];
        if (found ? emitter.Primitive != V::InvalidIndex && emitter.Primitive == hit.PrimitiveIndex
                  : emitter.Kind == ArvoCudaEndpoint::Environment)
        {
            vertex.Emitter = source;
            break;
        }
    }
    return vertex;
}
__device__ PhysicalSurface ScatteringFrame(const SceneView& scene, const V::PathVertex& vertex, Vec3 view,
                                           unsigned mode)
{
    Hit hit{};
    hit.Normal = Dot(vertex.Normal, view) < 0 ? Multiply(vertex.Normal, -1) : vertex.Normal;
    hit.GeometricNormal = vertex.Geometry;
    hit.FrontFace = Dot(vertex.Geometry, view) > 0;
    return MakePhysicalSurface(scene.Materials[vertex.Material], hit, view, vertex.Color, mode);
}
__device__ PhysicalEvaluation EvaluateScattering(const SceneView& scene, const V::PathVertex& vertex,
                                                 Vec3 view, Vec3 outgoing, unsigned mode)
{
    if (vertex.Kind == V::VolumeVertex)
    {
        const auto& material = scene.Materials[vertex.Material];
        const float phase = SubsurfacePhase(Dot(Multiply(view, -1), outgoing),
                                            material.ConductorIndex.W);
        const float pdf = (scene.Frame.TransportFlags & ImportanceBit) ? phase : 1 / (4 * Pi);
        const Vec3 albedo = Multiply(ToVec3(material.AlbedoReflectivity),
                                     material.ConductorExtinction.W * phase);
        return {albedo, pdf, pdf};
    }
    if (vertex.Kind != V::SurfaceVertex)
        return {};
    auto surface = ScatteringFrame(scene, vertex, view, mode);
    float fraction = 1;
    if (surface.Material.Model == LegacyMaterial)
    {
        fraction -= fminf(1, fmaxf(0, surface.Material.AlbedoReflectivity.W));
        surface.Material.Model = DiffuseMaterial;
    }
    auto result = EvaluatePhysical(surface, outgoing, (scene.Frame.TransportFlags & ImportanceBit) != 0);
    result.Value = Multiply(result.Value, fraction);
    result.Pdf *= fraction;
    result.ReversePdf *= fraction;
    return result;
}
__device__ PhysicalSample Scatter(const SceneView& scene, const V::PathVertex& vertex, unsigned mode,
                                  float2 directionU, float lobeU)
{
    if (vertex.Kind == V::VolumeVertex)
    {
        const auto& material = scene.Materials[vertex.Material];
        const auto phase = SubsurfaceSampleDirection(Native(Multiply(vertex.ToPrevious, -1)),
            material.ConductorIndex.W, directionU,
            (scene.Frame.TransportFlags & ImportanceBit) != 0);
        const Vec3 weight = Multiply(ToVec3(material.AlbedoReflectivity),
                                      material.ConductorExtinction.W * phase.Weight);
        return {Vector(phase.Direction), weight, phase.Pdf, phase.Pdf, 1,
                ArvoCuda::ScatteringGlossy};
    }
    auto surface = ScatteringFrame(scene, vertex, vertex.ToPrevious, mode);
    const float mirror = surface.Material.Model == LegacyMaterial
                             ? fminf(1, fmaxf(0, surface.Material.AlbedoReflectivity.W))
                             : 0;
    if (lobeU < mirror)
    {
        const Vec3 direction =
            SurfaceWorld(surface, make_float3(-surface.View.x, -surface.View.y, surface.View.z));
        if (!(surface.View.z > 0) || !SurfaceHemisphere(surface, direction, false))
            return {};
        const float adjoint = SurfaceAdjoint(surface, direction);
        return {direction, {adjoint, adjoint, adjoint},
                mirror,    mirror,
                1,         ArvoCuda::ScatteringDelta | ArvoCuda::ScatteringReflection};
    }
    if (surface.Material.Model == LegacyMaterial)
        surface.Material.Model = DiffuseMaterial;
    auto sample =
        SamplePhysical(surface, directionU, lobeU, (scene.Frame.TransportFlags & ImportanceBit) != 0);
    sample.Pdf *= 1 - mirror;
    sample.ReversePdf *= 1 - mirror;
    return sample;
}
__device__ void GrowPath(const SceneView& scene, V::Path& path, Ray ray, Vec3 beta, V::Density directionPdf,
                         SampleContext samples, GroupStatistics& statistics)
{
    const unsigned domain = path.Mode == ArvoCuda::TransportRadiance ? CameraDomain : LightDomain;
    unsigned interior = V::InvalidIndex;
    for (unsigned index = path.Count; index < path.Limit; ++index)
    {
        if (!Energy(beta) || !Finite(ray.Direction))
        {
            path.End = V::NumericalFailure;
            return;
        }
        if (!Nonzero(beta))
        {
            path.End = V::Absorbed;
            return;
        }
        if (path.Mode == ArvoCuda::TransportRadiance && index == 1)
            ++statistics.PrimaryRays;
        else
            ++statistics.ReflectionRays;
        auto vertex = NextVertex(scene, ray, statistics);
        const auto& previous = path.Vertices[index - 1];
        const bool infiniteLaunch = index == 1 && path.Mode == ArvoCuda::TransportImportance &&
                                    (path.EndpointFlags & ArvoCudaEmission::Infinite);
        if (index == 1 && vertex.Reserved[0] != V::InvalidIndex)
            interior = vertex.Reserved[0];
        const unsigned flightDimension = domain ^ (index * BounceDomain);
        if (interior != V::InvalidIndex)
        {
            const auto& material = scene.Materials[interior];
            const Vec3 endpointEdge = Subtract(vertex.Position, ray.Origin);
            const float endpoint = vertex.Kind == V::EnvironmentVertex ? ray.MaximumDistance :
                norm3df(endpointEdge.X, endpointEdge.Y, endpointEdge.Z);
            const auto flight = SubsurfaceFreeFlight(material.Optical.W,
                Native(Multiply(ToVec3(material.AlbedoReflectivity), material.ConductorExtinction.W)),
                Native(ray.Direction), endpoint,
                Sample2D(samples, flightDimension ^ InteriorFlightDomain).x);
            if (flight.Scattered)
            {
                // Incident throughput precedes the phase/albedo at this vertex.
                vertex.Kind = V::VolumeVertex;
                vertex.Position = Add(ray.Origin, Multiply(ray.Direction, flight.Distance));
                vertex.Material = interior;
                vertex.Emitter = vertex.Primitive = V::InvalidIndex;
                vertex.Normal = vertex.Geometry = {};
                vertex.FrontFace = 0;
            }
        }
        vertex.Throughput = beta;
        if (vertex.Kind == V::EnvironmentVertex)
            vertex.Arrival = directionPdf;
        else if (infiniteLaunch)
            vertex.Arrival = V::DiskProjection(V::DensityFromValue(path.PositionPdf), Native(vertex.Geometry),
                                               Native(ray.Direction));
        else
            vertex.Arrival = EdgeDensity(scene, directionPdf, previous, vertex);
        if (!V::HasDensity(vertex.Arrival) || !Finite(vertex.Position))
        {
            path.End = V::NumericalFailure;
            return;
        }
        path.Vertices[index] = vertex;
        path.Count = index + 1;
        if (vertex.Kind != V::SurfaceVertex && vertex.Kind != V::VolumeVertex)
        {
            path.End = vertex.Kind == V::EnvironmentVertex ? V::Escaped : V::Absorbed;
            return;
        }
        if (path.Count == path.Limit)
            break;
        const unsigned dimension = domain ^ (path.Count * BounceDomain);
        const auto sample = Scatter(scene, vertex, path.Mode,
                                    Sample2D(samples, dimension ^
                                        (vertex.Kind == V::VolumeVertex ? InteriorPhaseDomain : 0)),
                                    Sample2D(samples, dimension ^ LobeDomain).x);
        if (!sample.Event)
        {
            path.End = V::Absorbed;
            return;
        }
        if (!(sample.Pdf > 0 && sample.ReversePdf >= 0) || !isfinite(sample.Pdf) ||
            !isfinite(sample.ReversePdf) || !Finite(sample.Direction) || !Energy(sample.Weight))
        {
            path.End = V::NumericalFailure;
            return;
        }
        auto& stored = path.Vertices[index];
        stored.Continuation = sample.Direction;
        stored.Pdf = sample.Pdf;
        stored.ReversePdf = sample.ReversePdf;
        stored.Event = sample.Event;
        stored.RelativeIndex = sample.RelativeIndex;
        const auto reverse = V::DensityFromValue(sample.ReversePdf);
        stored.ReturnToPrevious =
            infiniteLaunch ? reverse
                           : EdgeDensity(scene, reverse, stored, previous);
        beta = Multiply(beta, sample.Weight);
        directionPdf = V::DensityFromValue(sample.Pdf);
        ray = {Depart(stored, sample.Direction), sample.Direction, RayEpsilon, MaximumRayDistance};
        if (stored.Kind == V::SurfaceVertex && scene.Materials[stored.Material].Optical.W > 0 &&
            (sample.Event & ArvoCuda::ScatteringTransmission))
            interior = stored.FrontFace ? stored.Material : V::InvalidIndex;
    }
    path.End = V::VertexLimitReached;
}
__device__ bool ConfigurePath(const SceneView& scene, V::Path& path, unsigned mode)
{
    path = {};
    path.Mode = mode;
    path.Limit = scene.Control.VertexLimit;
    path.Emitter = V::InvalidIndex;
    const auto& frame = scene.Frame;
    return path.Limit > 0 && path.Limit <= V::MaximumVertices &&
           frame.MaxReflectionBounces <= V::MaximumReflections && !(frame.TransportFlags & RouletteBit) &&
           frame.Width && frame.Height && frame.TotalSamples && frame.FullWidth >= frame.Width &&
           frame.FullHeight >= frame.Height && frame.TileX <= frame.FullWidth - frame.Width &&
           frame.TileY <= frame.FullHeight - frame.Height && frame.SamplingSequence <= BlueNoiseSequence &&
           scene.Control.Pass < frame.TotalSamples;
}
__device__ bool ValidateRoot(V::Path& path)
{
    const auto& root = path.Vertices[0];
    if (V::HasDensity(root.Arrival) && Energy(root.Throughput) && Finite(root.Position))
        return true;
    path.End = V::NumericalFailure;
    return false;
}
__device__ void TraceCamera(const SceneView& scene, unsigned pixel, V::Path& path,
                            GroupStatistics& statistics)
{
    if (!ConfigurePath(scene, path, ArvoCuda::TransportRadiance))
        return;
    const auto& frame = scene.Frame;
    if (pixel >= frame.Width * frame.Height)
        return;
    const unsigned x = pixel % frame.Width + frame.TileX, y = pixel / frame.Width + frame.TileY;
    auto samples = MakeSampleContext(x, y, frame.FullWidth, scene.Control.Pass, frame.TotalSamples,
                                     frame.SamplingSequence, frame.SamplingSeed);
    samples.PixelSeed = Hash(samples.PixelSeed ^ Hash(scene.Control.Seed ^ CameraDomain));
    const float2 raster =
        ArvoCudaCamera::PixelRaster(x, y, frame.FullWidth, frame.FullHeight, Sample2D(samples, 0));
    const ArvoCudaCamera::Frame camera{Native(frame.CameraPosition),
                                       Native(frame.CameraForward),
                                       Native(frame.CameraRight),
                                       Native(frame.CameraUp),
                                       frame.HalfWidth,
                                       frame.HalfHeight,
                                       frame.LensRadius,
                                       frame.FocusDistance};
    const auto endpoint = ArvoCudaCamera::Emit(camera, raster, Sample2D(samples, LensDomain));
    if (!(endpoint.Flags & ArvoCudaCamera::Valid))
    {
        path.End = V::NullEndpoint;
        return;
    }
    path.EndpointFlags = endpoint.Flags;
    path.LaunchDirection = Vector(endpoint.Direction);
    path.LaunchWeight = {endpoint.Weight, endpoint.Weight, endpoint.Weight};
    path.PositionPdf = endpoint.PositionPdf;
    path.DirectionPdf = endpoint.DirectionPdf;
    path.EndpointValue = {endpoint.Importance, endpoint.Importance, endpoint.Importance};
    path.EndpointNormal = frame.CameraForward;
    path.LensX = endpoint.LensPoint.x;
    path.LensY = endpoint.LensPoint.y;
    path.RasterX = raster.x;
    path.RasterY = raster.y;
    auto& root = path.Vertices[0];
    root.Kind = V::CameraVertex;
    root.Emitter = root.Primitive = root.Material = V::InvalidIndex;
    root.Reserved[0] = V::InvalidIndex;
    root.Position = Vector(endpoint.Origin);
    root.Geometry = root.Normal = frame.CameraForward;
    root.Throughput = path.LaunchWeight;
    root.Arrival = V::DensityFromValue(endpoint.PositionPdf);
    path.Count = 1;
    if (ValidateRoot(path))
        GrowPath(scene, path, {root.Position, path.LaunchDirection, RayEpsilon, MaximumRayDistance},
                 path.LaunchWeight, V::DensityFromValue(endpoint.DirectionPdf), samples, statistics);
}
__device__ void TraceLight(const SceneView& scene, unsigned index, V::Path& path, GroupStatistics& statistics)
{
    if (!ConfigurePath(scene, path, ArvoCuda::TransportImportance))
        return;
    const unsigned seed =
        Hash(scene.Control.Seed ^ LightDomain) ^ Hash(scene.Control.Pass ^ PassDomain) ^ Hash(index);
    const SampleContext samples{index, 0, seed, scene.Control.Pass, 1, IndependentSequence};
    const ArvoCudaEmission::Domain domain{Native(scene.Control.Center), scene.Control.WorldRadius};
    const auto endpoint = ArvoCudaEndpoint::Emit(
        scene.Emitters, scene.Control.Sources, domain, Sample2D(samples, SourceDomain).x,
        Sample2D(samples, PositionDomain), Sample2D(samples, DirectionDomain));
    const auto& emitted = endpoint.Emission;
    if (!(emitted.Flags & ArvoCudaEmission::Valid))
    {
        path.End = V::NullEndpoint;
        return;
    }
    path.EndpointFlags = emitted.Flags;
    path.Emitter = endpoint.EmitterIndex;
    path.LaunchDirection = Vector(emitted.Direction);
    path.LaunchWeight = Vector(emitted.Weight);
    path.EndpointValue = Vector(emitted.Value);
    path.EndpointNormal = Vector(emitted.Normal);
    path.SourceSelection = endpoint.SelectionProbability;
    path.PositionPdf = emitted.PositionPdf;
    path.DirectionPdf = emitted.DirectionPdf;
    auto& root = path.Vertices[0];
    root.Kind = V::LightVertex;
    root.Emitter = endpoint.EmitterIndex;
    root.Primitive = endpoint.Primitive;
    root.Material = V::InvalidIndex;
    root.Reserved[0] = V::InvalidIndex;
    root.Position = Vector(emitted.Origin);
    root.Geometry = root.Normal = Vector(emitted.Normal);
    const double rootPdf =
        emitted.Flags & ArvoCudaEmission::Infinite ? emitted.DirectionPdf : emitted.PositionPdf;
    root.Arrival = V::DensityFromValue(rootPdf);
    const double reciprocal = 1 / (double(endpoint.SelectionProbability) * rootPdf);
    root.Throughput = {float(double(emitted.Value.x) * reciprocal),
                       float(double(emitted.Value.y) * reciprocal),
                       float(double(emitted.Value.z) * reciprocal)};
    path.Count = 1;
    if (!ValidateRoot(path))
        return;
    const Vec3 origin =
        emitted.Flags & ArvoCudaEmission::DeltaPosition ? root.Position : Depart(root, path.LaunchDirection);
    GrowPath(scene, path, {origin, path.LaunchDirection, RayEpsilon, MaximumRayDistance}, path.LaunchWeight,
             V::DensityFromValue(emitted.DirectionPdf), samples, statistics);
}
__device__ void RecordWalkError(const V::Path& path, unsigned* errors)
{
    if (path.End == V::InvalidWalk)
        atomicOr(errors, V::ConfigurationError);
    if (path.End == V::NumericalFailure ||
        (path.Mode == ArvoCuda::TransportRadiance && path.End == V::NullEndpoint))
        atomicOr(errors, V::TransportError);
}

// Couplings keep the two original arrivals at a merged vertex. The left accessor
// retains the photon's position and the right accessor retains the eye position;
// no copied or perturbed path can accidentally replace a cached specular edge.
struct JoinedPath
{
    const SceneView& Scene;
    const V::Path* Light;
    const V::Path& Camera;
    V::PathVertex Lens;
    Vec3 Launch;
    unsigned Source, SourceFlags, LightLimit, S, T, Count;
    bool Merged;

    __device__ const V::PathVertex& Left(unsigned i) const
    {
        return i + 1 == Count ? Lens : i < S ? Light->Vertices[i] : Camera.Vertices[Count - 1 - i];
    }
    __device__ const V::PathVertex& Right(unsigned i) const
    {
        const unsigned boundary = S - unsigned(Merged);
        return i + 1 == Count ? Lens : i < boundary ? Light->Vertices[i] : Camera.Vertices[Count - 1 - i];
    }
    __device__ Vec3 Edge(unsigned from, unsigned to) const
    {
        return from < to ? Subtract(Left(to).Position, Right(from).Position)
                         : Subtract(Right(to).Position, Left(from).Position);
    }
    __device__ Vec3 Direction(unsigned from, unsigned to) const
    {
        if (SourceFlags & ArvoCudaEmission::Infinite)
        {
            if (from == 0)
                return Launch;
            if (to == 0)
                return Multiply(Launch, -1);
        }
        return Normalize(Edge(from, to));
    }
    __device__ V::Density Arrival(double pdf, unsigned from, unsigned to) const
    {
        const auto angular = V::DensityFromValue(pdf);
        if (to == 0 && (SourceFlags & ArvoCudaEmission::Infinite))
            return angular;
        return from < to ? EdgeDensity(Scene, angular, Right(from), Left(to))
                         : EdgeDensity(Scene, angular, Left(from), Right(to));
    }
};
__device__ ArvoCudaCamera::Frame CameraFrame(const SceneView& scene)
{
    const auto& f = scene.Frame;
    return {Native(f.CameraPosition),
            Native(f.CameraForward),
            Native(f.CameraRight),
            Native(f.CameraUp),
            f.HalfWidth,
            f.HalfHeight,
            f.LensRadius,
            f.FocusDistance};
}
__device__ ArvoCudaCamera::Evaluation CameraDensity(const V::Path& path)
{
    return {make_float2(path.RasterX, path.RasterY), path.EndpointValue.X, path.PositionPdf,
            path.DirectionPdf, path.EndpointFlags};
}
__device__ ArvoCudaEndpoint::Density SourceDensity(const SceneView& scene, unsigned source, Vec3 normal,
                                                   Vec3 direction)
{
    const ArvoCudaEmission::Domain domain{Native(scene.Control.Center), scene.Control.WorldRadius};
    return ArvoCudaEndpoint::Pdf(scene.Emitters, scene.Control.Sources, domain, source, Native(normal),
                                 Native(direction));
}
__device__ V::Weights CouplingWeights(const SceneView& scene, const JoinedPath& path,
                                      const ArvoCudaCamera::Evaluation& camera,
                                      const V::CouplingControls& controls)
{
    V::VertexPdf vertex[V::MaximumVertices]{};
    for (unsigned i = 1; i + 1 < path.Count; ++i)
    {
        const auto& fromLight = path.Left(i);
        const auto& fromCamera = path.Right(i);
        if (fromCamera.Kind == V::VolumeVertex)
            vertex[i].Flags |= V::VolumeScattering;
        float forward, reverse;
        const bool reconnect = path.Merged ? i + 1 == path.S : (i == path.S || i + 1 == path.S);
        if (reconnect)
        {
            const Vec3 incident =
                path.Merged || i + 1 == path.S ? fromLight.ToPrevious : path.Direction(i, i - 1);
            const Vec3 outgoing =
                path.Merged || i == path.S ? fromCamera.ToPrevious : path.Direction(i, i + 1);
            const auto bsdf =
                EvaluateScattering(scene, fromCamera, incident, outgoing, ArvoCuda::TransportImportance);
            forward = bsdf.Pdf;
            reverse = bsdf.ReversePdf;
        }
        else
        {
            forward = i < path.S ? fromCamera.Pdf : fromCamera.ReversePdf;
            reverse = i < path.S ? fromCamera.ReversePdf : fromCamera.Pdf;
            if (fromCamera.Event & ArvoCuda::ScatteringDelta)
                vertex[i].Flags |= V::DeltaScattering;
        }
        vertex[i + 1].Forward = path.Arrival(forward, i, i + 1);
        vertex[i - 1].Reverse = path.Arrival(reverse, i, i - 1);
    }
    const bool infinite = (path.SourceFlags & ArvoCudaEmission::Infinite) != 0;
    const Vec3 launch = path.S == 1 && !infinite ? path.Direction(0, 1) : path.Launch;
    const auto source = SourceDensity(scene, path.Source, path.Left(0).Geometry, launch);
    const auto position =
        V::DensityFromValue(source.Flags & ArvoCudaEmission::DeltaPosition ? 1 : source.Position);
    const double angular = source.Flags & ArvoCudaEmission::DeltaDirection ? 1 : source.Direction;
    const auto sourceEndpoint = infinite ? V::DensityFromValue(angular) : position;
    vertex[1].Forward = infinite ? V::DiskProjection(position, Native(path.Left(1).Geometry), Native(launch))
                                 : path.Arrival(angular, 0, 1);
    if (source.Flags & (ArvoCudaEmission::DeltaPosition | ArvoCudaEmission::DeltaDirection))
        vertex[0].Flags |= V::EndpointNotIntersectable;
    vertex[path.Count - 2].Reverse = path.Arrival(camera.DirectionPdf, path.Count - 1, path.Count - 2);
    vertex[path.Count - 1].Flags |= V::EndpointNotIntersectable;
    const auto lens =
        V::DensityFromValue(camera.Flags & ArvoCudaCamera::DeltaPosition ? 1 : camera.PositionPdf);
    const auto unit = V::DensityFromValue(1);
    V::Strategy connections[V::MaximumStrategies]{}, mergings[V::MaximumStrategies]{};
    for (unsigned s = 0; s < path.Count; ++s)
    {
        auto& vc = connections[s];
        vc.LightEndpoint = s ? sourceEndpoint : unit;
        vc.CameraEndpoint = lens;
        vc.LightSelection = s ? source.SelectionProbability : 1;
        vc.SampleCount = s + 1 == path.Count ? controls.LightTracingSamples : 1;
        vc.Enabled = s <= path.LightLimit && path.Count - s <= path.Camera.Limit &&
                     (s != 1 || controls.NextEventEstimation) && (path.Count > 2 || s == 0);
        if (s >= 2)
        {
            auto& vm = mergings[s];
            vm.LightEndpoint = sourceEndpoint;
            vm.CameraEndpoint = lens;
            vm.LightSelection = source.SelectionProbability;
            vm.SampleCount = controls.MergingSamples;
            vm.Enabled = s <= path.LightLimit && path.Count + 1 - s <= path.Camera.Limit;
        }
    }
    return V::EvaluateStrategies(vertex, path.Count, connections, mergings, controls.MergeRadius,
                                 controls.Heuristic);
}
struct Spectrum
{
    double X, Y, Z;
    __device__ Spectrum(Vec3 v) : X(v.X), Y(v.Y), Z(v.Z) {}
    __device__ void MultiplyBy(Vec3 factor)
    {
        X *= factor.X;
        Y *= factor.Y;
        Z *= factor.Z;
    }
    __device__ void Scale(double factor)
    {
        X *= factor;
        Y *= factor;
        Z *= factor;
    }
    __device__ bool HasEnergy() const { return X > 0 || Y > 0 || Z > 0; }
};
__device__ void FinishContribution(V::Contribution& result, Spectrum value, double weight)
{
    result.MisWeight = weight;
    value.Scale(weight);
    const Vec3 rgb{float(value.X), float(value.Y), float(value.Z)};
    if (!Energy(rgb) || !isfinite(weight) || weight < 0 || weight > 1)
        result.Flags |= V::InvalidContribution;
    else
        result.Radiance = rgb;
}
__device__ Vec3 EmittedValue(const Emitter& emitter, Vec3 outgoing)
{
    if (emitter.Kind == ArvoCudaEndpoint::Environment)
    {
        const float top = (1 - outgoing.Y) / 2;
        return Add(Multiply(Vector(emitter.BackgroundBottom), 1 - top),
                   Multiply(Vector(emitter.BackgroundTop), top));
    }
    if (emitter.Kind == ArvoCudaEndpoint::Rectangle)
        return Dot(Vector(emitter.Direction), outgoing) > 0
                   ? Multiply(Vector(emitter.Strength), 1 / (emitter.Width * emitter.Height))
                   : Vec3{};
    return Vector(emitter.Strength);
}
__device__ bool Unobstructed(const SceneView& scene, Ray ray, GroupStatistics& counters)
{
    ++counters.ShadowRays;
    Hit hit{};
    hit.Distance = ray.MaximumDistance;
    if (IntersectScene(scene.Nodes, scene.References, scene.Triangles, scene.Spheres, scene.Frame, ray, true,
                       hit, counters.PrimitiveTests))
        return false;
    for (unsigned i = 0; i < scene.Control.Sources; ++i)
        if (scene.Emitters[i].Kind == ArvoCudaEndpoint::Rectangle)
        {
            ++counters.PrimitiveTests;
            float nearest = ray.MaximumDistance;
            if (IntersectRectangle(scene.Emitters[i], ray, nearest))
                return false;
        }
    return true;
}
__device__ bool UnobstructedSegment(const SceneView& scene, Vec3 origin, Vec3 target,
                                    GroupStatistics& counters)
{
    const Vec3 edge = Subtract(target, origin);
    const float length = norm3df(edge.X, edge.Y, edge.Z);
    return length > 2 * RayEpsilon && isfinite(length) &&
           Unobstructed(scene, {origin, Multiply(edge, 1 / length), RayEpsilon, length - RayEpsilon},
                        counters);
}
__device__ bool ValidCouplingControls(const V::CouplingControls& c)
{
    return c.MaximumPathVertices >= 2 && c.MaximumPathVertices <= V::MaximumVertices && c.MergeRadius >= 0 &&
           isfinite(c.MergeRadius) && c.MergingSamples >= 0 && isfinite(c.MergingSamples) &&
           c.LightTracingSamples >= 0 && isfinite(c.LightTracingSamples) && c.Heuristic <= V::PowerMis &&
           c.NextEventEstimation <= 1;
}
__device__ bool UsablePath(const V::Path& path, unsigned mode)
{
    return path.Mode == mode && path.Count && path.Count <= path.Limit && path.Limit <= V::MaximumVertices &&
           path.End != V::InvalidWalk && path.End != V::NumericalFailure;
}

// One entry handles VC and VM queries; no integrator delegates to another one.
// Light tracing produces full-film splats. The frame estimator must divide their
// sum by LightTracingSamples = N_light / full_film_pixels, including null walks.
__device__ V::Contribution Couple(const SceneView& scene, const V::Path* light, const V::Path& camera,
                                  const V::CouplingQuery& query, const V::CouplingControls& controls,
                                  GroupStatistics& counters)
{
    V::Contribution result{};
    const unsigned s = query.LightVertices, t = query.CameraVertices;
    const bool merge = query.Merge != 0;
    if (!ValidCouplingControls(controls) || query.Merge > 1)
    {
        result.Flags = V::InvalidContribution;
        return result;
    }
    if (!UsablePath(camera, ArvoCuda::TransportRadiance) || t < 1 || t > camera.Count ||
        s > V::MaximumVertices || s + t < 2 || s + t - unsigned(merge) > controls.MaximumPathVertices ||
        (merge && (s < 2 || t < 2)))
        return result;
    result.RasterX = camera.RasterX;
    result.RasterY = camera.RasterY;
    const auto& eye = camera.Vertices[t - 1];
    auto lensPdf = CameraDensity(camera);
    JoinedPath joined{
        scene, light, camera, camera.Vertices[0], {}, V::InvalidIndex, 0,
        controls.MaximumPathVertices, s, t, s + t - unsigned(merge), merge};
    if (s == 0)
    {
        if (t < 2 || eye.Emitter >= scene.Control.Sources)
            return result;
        joined.Source = eye.Emitter;
        joined.Launch = eye.ToPrevious;
        joined.SourceFlags = SourceDensity(scene, eye.Emitter, eye.Geometry, eye.ToPrevious).Flags;
        Spectrum value(eye.Throughput);
        value.MultiplyBy(EmittedValue(scene.Emitters[eye.Emitter], eye.ToPrevious));
        const auto weights = CouplingWeights(scene, joined, lensPdf, controls);
        FinishContribution(result, value, weights.Connection[0]);
        return result;
    }
    if (!light || !UsablePath(*light, ArvoCuda::TransportImportance) || s > light->Count ||
        light->Emitter >= scene.Control.Sources || (s == 1 && (t == 1 || !controls.NextEventEstimation)))
        return result;
    joined.Source = light->Emitter;
    joined.SourceFlags = light->EndpointFlags;
    joined.Launch = light->LaunchDirection;
    joined.LightLimit = light->Limit;
    const auto& photon = light->Vertices[s - 1];
    Spectrum value(eye.Throughput);
    if (merge)
    {
        if (eye.Kind != photon.Kind ||
            (eye.Kind != V::SurfaceVertex && eye.Kind != V::VolumeVertex) ||
            !(controls.MergeRadius > 0 && controls.MergingSamples > 0))
            return result;
        const bool volume = eye.Kind == V::VolumeVertex;
        if (volume && eye.Material != photon.Material)
            return result;
        constexpr double RelativeSheetThickness = 0.1;
        constexpr float NormalAgreement = 0.9f;
        const Vec3 offset = Subtract(photon.Position, eye.Position);
        const double planeDistance =
            fabs(double(offset.X) * eye.Geometry.X + double(offset.Y) * eye.Geometry.Y +
                 double(offset.Z) * eye.Geometry.Z);
        const double distance = norm3d(offset.X, offset.Y, offset.Z);
        if (distance > controls.MergeRadius || (!volume &&
            (planeDistance > RelativeSheetThickness * controls.MergeRadius ||
             Dot(eye.Geometry, photon.Geometry) < NormalAgreement)))
            return result;
        if (volume && distance > 2 * RayEpsilon &&
            !UnobstructedSegment(scene, eye.Position, photon.Position, counters))
            return result;
        const double projectedArea = volume ? 1 : fabs(double(Dot(eye.Geometry, photon.ToPrevious)));
        if (!(projectedArea > 0))
            return result;
        const auto scatter =
            EvaluateScattering(scene, eye, eye.ToPrevious, photon.ToPrevious, ArvoCuda::TransportRadiance);
        value.MultiplyBy(photon.Throughput);
        value.MultiplyBy(scatter.Value);
        const double areaCorrection = volume ? 1 :
            fabs(double(Dot(eye.Normal, photon.ToPrevious))) / projectedArea;
        const auto measure = volume ? V::SphereVolume(controls.MergeRadius) :
                                      V::DiskArea(controls.MergeRadius);
        value.Scale(exp2(-measure.Log2Value) * areaCorrection /
                    controls.MergingSamples);
    }
    else if (s == 1)
    {
        if (eye.Kind != V::SurfaceVertex && eye.Kind != V::VolumeVertex)
            return result;
        const bool infinite = (joined.SourceFlags & ArvoCudaEmission::Infinite) != 0;
        const bool point = (joined.SourceFlags & ArvoCudaEmission::DeltaPosition) != 0;
        Vec3 toSource = Multiply(joined.Launch, -1);
        double geometry = 1;
        if (!infinite)
        {
            const Vec3 edge = Subtract(photon.Position, eye.Position);
            const double distance = norm3d(edge.X, edge.Y, edge.Z);
            if (!(distance > 0))
                return result;
            toSource = Multiply(edge, float(1 / distance));
            geometry = (point ? 1 : fabs(double(Dot(photon.Geometry, toSource)))) / (distance * distance);
        }
        const auto scatter =
            EvaluateScattering(scene, eye, eye.ToPrevious, toSource, ArvoCuda::TransportRadiance);
        value.MultiplyBy(scatter.Value);
        value.MultiplyBy(EmittedValue(scene.Emitters[joined.Source], Multiply(toSource, -1)));
        const double density =
            double(light->SourceSelection) * (infinite ? light->DirectionPdf : light->PositionPdf);
        if (!(density > 0))
            return result;
        value.Scale((eye.Kind == V::VolumeVertex ? 1 : fabs(double(Dot(eye.Normal, toSource)))) *
                    geometry * (infinite ? 1 : SegmentTransmittance(scene, eye, photon)) / density);
        if (!value.HasEnergy())
            return result;
        const Vec3 origin = Depart(eye, toSource);
        const bool visible =
            infinite ? Unobstructed(scene, {origin, toSource, RayEpsilon, MaximumRayDistance}, counters)
                     : UnobstructedSegment(scene, origin,
                                           point ? photon.Position : Depart(photon, Multiply(toSource, -1)),
                                           counters);
        if (!visible)
            return result;
    }
    else
    {
        if ((photon.Kind != V::SurfaceVertex && photon.Kind != V::VolumeVertex) ||
            (t > 1 && eye.Kind != V::SurfaceVertex && eye.Kind != V::VolumeVertex))
            return result;
        Vec3 toCamera, target;
        if (t == 1)
        {
            const auto projection = ArvoCudaCamera::Connect(CameraFrame(scene), Native(photon.Position),
                                                            make_float2(query.LensU, query.LensV));
            if (!(projection.Flags & ArvoCudaCamera::Valid) || !(controls.LightTracingSamples > 0))
                return result;
            toCamera = Vector(projection.Direction);
            target = Vector(projection.Position);
            joined.Lens.Position = target;
            joined.Lens.Geometry = scene.Frame.CameraForward;
            value = Spectrum(photon.Throughput);
            value.Scale((photon.Kind == V::VolumeVertex ? 1 :
                fabs(double(Dot(photon.Normal, toCamera)))) * projection.Weight *
                SegmentTransmittance(scene, photon, joined.Lens));
            lensPdf = ArvoCudaCamera::Evaluate(CameraFrame(scene), projection.LensPoint,
                                               Native(Multiply(toCamera, -1)));
            result.RasterX = projection.Raster.x;
            result.RasterY = projection.Raster.y;
            result.Flags = V::SplatContribution;
        }
        else
        {
            const Vec3 edge = Subtract(eye.Position, photon.Position);
            const double distance = norm3d(edge.X, edge.Y, edge.Z);
            if (!(distance > 0))
                return result;
            toCamera = Multiply(edge, float(1 / distance));
            target = Depart(eye, Multiply(toCamera, -1));
            value.MultiplyBy(photon.Throughput);
            const auto scatter = EvaluateScattering(scene, eye, eye.ToPrevious, Multiply(toCamera, -1),
                                                    ArvoCuda::TransportRadiance);
            value.MultiplyBy(scatter.Value);
            const double lightCosine = photon.Kind == V::VolumeVertex ? 1 :
                fabs(double(Dot(photon.Normal, toCamera)));
            const double eyeCosine = eye.Kind == V::VolumeVertex ? 1 :
                fabs(double(Dot(eye.Normal, toCamera)));
            value.Scale(lightCosine * eyeCosine / (distance * distance) *
                        SegmentTransmittance(scene, photon, eye));
        }
        const auto scatter =
            EvaluateScattering(scene, photon, photon.ToPrevious, toCamera, ArvoCuda::TransportImportance);
        value.MultiplyBy(scatter.Value);
        if (!value.HasEnergy() || !(lensPdf.Flags & ArvoCudaCamera::Valid) ||
            !UnobstructedSegment(scene, Depart(photon, toCamera), target, counters))
            return result;
    }
    const auto weights = CouplingWeights(scene, joined, lensPdf, controls);
    FinishContribution(result, value, merge ? weights.Merging[s] : weights.Connection[s]);
    return result;
}

// A global light population is reused by every crop. Index records are sorted
// by (bucket, path/vertex identity); lens records by (pixel, path/vertex identity).
// Each pixel reduces in that order without floating-point atomics. Bitonic
// compare/exchange passes are independent pairs, including across launch batches.
constexpr unsigned PairingDomain = 0x56435052u, SplatDomain = 0x56434c53u;
__device__ double PassRadius(const V::FrameControls& controls, unsigned pass)
{
    // Georgiev et al. (2012): shrinking bandwidth, averaging complete pass
    // estimates. This is not PPM's persistent flux/count/radius recurrence.
    return controls.InitialRadius / pow(double(pass) + 1, (1 - controls.Alpha) / 2);
}
__device__ V::CouplingControls FrameCouplings(const FrameConstants& frame, const V::WalkControls& walk,
                                              const V::FrameControls& controls)
{
    return {PassRadius(controls, walk.Pass),
            double(controls.Paths),
            double(controls.Paths) / (double(frame.FullWidth) * frame.FullHeight),
            walk.VertexLimit,
            controls.Heuristic,
            controls.NextEventEstimation,
            {}};
}
__device__ bool FrameConfiguration(const FrameConstants& frame, const V::WalkControls& walk,
                                   const V::FrameControls& controls, unsigned* error)
{
    const bool valid =
        frame.Width && frame.Height && frame.FullWidth >= frame.Width && frame.FullHeight >= frame.Height &&
        frame.TileX <= frame.FullWidth - frame.Width && frame.TileY <= frame.FullHeight - frame.Height &&
        frame.TotalSamples && walk.Pass < frame.TotalSamples &&
        frame.MaxReflectionBounces <= V::MaximumReflections &&
        walk.VertexLimit == frame.MaxReflectionBounces + 3 && !frame.AdaptiveEnabled &&
        !(frame.TransportFlags & RouletteBit) && frame.ShadowStrength == 1 &&
        frame.SamplingSequence <= BlueNoiseSequence && controls.Paths &&
        static_cast<unsigned long long>(controls.Paths) * sizeof(V::Path) <= V::MaximumPathCacheBytes &&
        controls.IndexCapacity && !(controls.IndexCapacity & (controls.IndexCapacity - 1)) &&
        static_cast<unsigned long long>(controls.Paths) * (walk.VertexLimit - 2) <= controls.IndexCapacity &&
        controls.Buckets && !(controls.Buckets & (controls.Buckets - 1)) && controls.InitialRadius >= 0 &&
        isfinite(controls.InitialRadius) && controls.Alpha > 0 && controls.Alpha < 1 &&
        controls.Heuristic <= V::PowerMis && controls.NextEventEstimation <= 1;
    if (!valid)
        atomicOr(error, V::ConfigurationError);
    return valid;
}
struct GridCell
{
    int X, Y, Z;
};
__device__ bool CellAt(Vec3 position, const V::WalkControls& walk, double radius, GridCell& cell,
                       unsigned* error)
{
    // Reserve an integer margin for the +/-1 neighborhood. Reject overflows
    // explicitly rather than silently dropping vertices or aliasing cells.
    constexpr double CoordinateLimit = 1073741824.0;
    const double x = floor((double(position.X) - walk.Center.X) / radius);
    const double y = floor((double(position.Y) - walk.Center.Y) / radius);
    const double z = floor((double(position.Z) - walk.Center.Z) / radius);
    if (!isfinite(x) || !isfinite(y) || !isfinite(z) || fabs(x) >= CoordinateLimit ||
        fabs(y) >= CoordinateLimit || fabs(z) >= CoordinateLimit)
    {
        atomicOr(error, V::TransportError);
        return false;
    }
    cell = {int(x), int(y), int(z)};
    return true;
}
__device__ unsigned Bucket(GridCell cell, unsigned buckets)
{
    return Hash(Hash(unsigned(cell.X)) ^ Hash(unsigned(cell.Y) ^ PositionDomain) ^
                Hash(unsigned(cell.Z) ^ DirectionDomain)) &
           (buckets - 1);
}
__device__ bool SameCell(GridCell a, GridCell b)
{
    return a.X == b.X && a.Y == b.Y && a.Z == b.Z;
}
__device__ void AccumulateContribution(double* sum, const V::Contribution& contribution, unsigned* error)
{
    if (contribution.Flags & V::InvalidContribution)
        atomicOr(error, V::TransportError);
    sum[0] += contribution.Radiance.X;
    sum[1] += contribution.Radiance.Y;
    sum[2] += contribution.Radiance.Z;
}
__device__ void GatherMerges(const SceneView& scene, const V::FrameControls& frame,
                             const V::CouplingControls& coupling, const V::Path* lights,
                             const V::Path& camera, unsigned t, const V::IndexEntry* index,
                             const V::BucketRange* ranges, double* sum, GroupStatistics& statistics,
                             unsigned* error)
{
    const auto& eye = camera.Vertices[t - 1];
    if ((eye.Kind != V::SurfaceVertex && eye.Kind != V::VolumeVertex) ||
        !(coupling.MergeRadius > 0))
        return;
    GridCell center;
    if (!CellAt(eye.Position, scene.Control, coupling.MergeRadius, center, error))
        return;
    const unsigned slots = scene.Control.VertexLimit - 2;
    for (int z = -1; z <= 1; ++z)
        for (int y = -1; y <= 1; ++y)
            for (int x = -1; x <= 1; ++x)
            {
                const GridCell cell{center.X + x, center.Y + y, center.Z + z};
                const auto range = ranges[Bucket(cell, frame.Buckets)];
                if (range.Begin > range.End || range.End > frame.IndexCapacity)
                {
                    atomicOr(error, V::TransportError);
                    return;
                }
                for (unsigned i = range.Begin; i < range.End; ++i)
                {
                    const unsigned reference = index[i].Reference, path = reference / slots;
                    const unsigned s = reference % slots + 2;
                    if (path >= frame.Paths)
                    {
                        atomicOr(error, V::TransportError);
                        return;
                    }
                    if (s + t - 1 > coupling.MaximumPathVertices || s > lights[path].Count)
                        continue;
                    GridCell candidate;
                    if (!CellAt(lights[path].Vertices[s - 1].Position, scene.Control, coupling.MergeRadius,
                                candidate, error))
                        return;
                    // Different neighbors can hash to the same bucket; exact cell
                    // matching makes every photon contribute at most once.
                    if (!SameCell(cell, candidate))
                        continue;
                    const V::CouplingQuery query{path, 0, s, t, 0, 0, 1, 0};
                    AccumulateContribution(
                        sum, Couple(scene, lights + path, camera, query, coupling, statistics), error);
                }
            }
}
__device__ bool EntryBefore(V::IndexEntry a, V::IndexEntry b)
{
    return a.Key < b.Key || (a.Key == b.Key && a.Reference < b.Reference);
}
} // namespace

extern "C" __global__ void VcmTraceCameraPathsKernel(
    const PackedAccelerationNode* nodes, const PackedPrimitiveRef* references,
    const PackedTriangle* triangles, const PackedSphere* spheres, const PackedMaterial* materials,
    const PackedTexture* textures, const unsigned* texels, const Emitter* emitters,
    const FrameConstants* frame, const V::WalkControls* controls, V::Path* paths, GroupStatistics* statistics,
    unsigned* errors)
{
    const unsigned local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= controls->Count)
        return;
    GroupStatistics counters{};
    const SceneView scene{nodes,    references, triangles, spheres, materials,
                          textures, texels,     emitters,  *frame,  *controls};
    TraceCamera(scene, controls->Begin + local, paths[local], counters);
    statistics[local] = counters;
    RecordWalkError(paths[local], errors);
}
extern "C" __global__ void VcmTraceLightPathsKernel(
    const PackedAccelerationNode* nodes, const PackedPrimitiveRef* references,
    const PackedTriangle* triangles, const PackedSphere* spheres, const PackedMaterial* materials,
    const PackedTexture* textures, const unsigned* texels, const Emitter* emitters,
    const FrameConstants* frame, const V::WalkControls* controls, V::Path* paths, GroupStatistics* statistics,
    unsigned* errors)
{
    const unsigned local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= controls->Count)
        return;
    GroupStatistics counters{};
    const SceneView scene{nodes,    references, triangles, spheres, materials,
                          textures, texels,     emitters,  *frame,  *controls};
    TraceLight(scene, controls->Begin + local, paths[local], counters);
    statistics[local] = counters;
    RecordWalkError(paths[local], errors);
}

// Diagnostic/batched coupling entry also compiles the same device functions
// used by the forthcoming frame scheduler. Rounded dispatches leave tails alone.
extern "C" __global__ void VcmEvaluateCouplingsKernel(
    const PackedAccelerationNode* nodes, const PackedPrimitiveRef* references,
    const PackedTriangle* triangles, const PackedSphere* spheres, const PackedMaterial* materials,
    const PackedTexture* textures, const unsigned* texels, const Emitter* emitters,
    const FrameConstants* frame, const V::WalkControls* walks, const V::CouplingControls* controls,
    const V::Path* lightPaths, unsigned lightCount, const V::Path* cameraPaths, unsigned cameraCount,
    const V::CouplingQuery* queries, unsigned queryCount, V::Contribution* contributions,
    GroupStatistics* statistics, unsigned* errors)
{
    const unsigned local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= queryCount)
        return;
    const auto query = queries[local];
    V::Contribution result{};
    GroupStatistics counters{};
    if (query.CameraPath >= cameraCount || (query.LightVertices && query.LightPath >= lightCount))
        result.Flags = V::InvalidContribution;
    else
    {
        const SceneView scene{nodes,    references, triangles, spheres, materials,
                              textures, texels,     emitters,  *frame,  *walks};
        result = Couple(scene, query.LightVertices ? lightPaths + query.LightPath : nullptr,
                        cameraPaths[query.CameraPath], query, *controls, counters);
    }
    contributions[local] = result;
    statistics[local] = counters;
    if (result.Flags & V::InvalidContribution)
        atomicOr(errors, V::TransportError);
}

extern "C" __global__ void VcmInitializeFrameKernel(const FrameConstants* frame, const V::WalkControls* walks,
                                                    const V::FrameControls* controls, V::PixelState* pixels,
                                                    V::Path* lensRoot, Vec4* film, unsigned* counts,
                                                    unsigned* targets, unsigned* errors)
{
    const unsigned local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= walks->Count || !FrameConfiguration(*frame, *walks, *controls, errors))
        return;
    const unsigned pixel = walks->Begin + local;
    if (pixel >= frame->Width * frame->Height)
        return;
    pixels[pixel] = {};
    film[pixel] = {};
    counts[pixel] = 0;
    targets[pixel] = frame->TotalSamples;
    if (pixel == 0)
    {
        *lensRoot = {};
        lensRoot->Count = 1;
        lensRoot->Limit = walks->VertexLimit;
        lensRoot->End = V::VertexLimitReached;
        lensRoot->Mode = ArvoCuda::TransportRadiance;
        lensRoot->Vertices[0].Kind = V::CameraVertex;
        lensRoot->Vertices[0].Position = frame->CameraPosition;
        lensRoot->Vertices[0].Normal = lensRoot->Vertices[0].Geometry = frame->CameraForward;
        lensRoot->Vertices[0].Throughput = {1, 1, 1};
        // Every light-to-camera query samples and evaluates a fresh aperture
        // endpoint. This root carries only topology, not an extra camera sample.
    }
}
extern "C" __global__ void VcmBuildIndexKernel(const V::WalkControls* walks, const V::FrameControls* controls,
                                               const V::Path* lights, V::IndexEntry* index,
                                               V::BucketRange* ranges, unsigned* errors)
{
    const unsigned local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= walks->Count)
        return;
    const unsigned slot = walks->Begin + local;
    if (slot < controls->Buckets)
        ranges[slot] = {};
    if (slot >= controls->IndexCapacity)
        return;
    V::IndexEntry entry{V::InvalidIndex, slot};
    const unsigned slots = walks->VertexLimit - 2, path = slot / slots, prefix = slot % slots + 2;
    const double radius = PassRadius(*controls, walks->Pass);
    if (path < controls->Paths && prefix <= lights[path].Count && radius > 0 &&
        (lights[path].Vertices[prefix - 1].Kind == V::SurfaceVertex ||
         lights[path].Vertices[prefix - 1].Kind == V::VolumeVertex))
    {
        GridCell cell;
        if (CellAt(lights[path].Vertices[prefix - 1].Position, *walks, radius, cell, errors))
            entry.Key = Bucket(cell, controls->Buckets);
    }
    index[slot] = entry;
}
extern "C" __global__ void VcmSortIndexKernel(const V::WalkControls* walks, const V::FrameControls* controls,
                                              V::IndexEntry* index, unsigned span, unsigned stride)
{
    const unsigned local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= walks->Count)
        return;
    const unsigned i = walks->Begin + local, partner = i ^ stride;
    if (i >= controls->IndexCapacity || partner >= controls->IndexCapacity || i >= partner)
        return;
    const auto a = index[i], b = index[partner];
    const bool ascending = (i & span) == 0;
    if (ascending ? EntryBefore(b, a) : EntryBefore(a, b))
    {
        index[i] = b;
        index[partner] = a;
    }
}
extern "C" __global__ void VcmBuildRangesKernel(const V::WalkControls* walks,
                                                const V::FrameControls* controls, const V::IndexEntry* index,
                                                V::BucketRange* ranges, unsigned* errors)
{
    const unsigned local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= walks->Count)
        return;
    const unsigned i = walks->Begin + local;
    if (i >= controls->IndexCapacity || index[i].Key == V::InvalidIndex)
        return;
    const unsigned key = index[i].Key;
    if (key >= controls->Buckets)
    {
        atomicOr(errors, V::TransportError);
        return;
    }
    if (!i || index[i - 1].Key != key)
        ranges[key].Begin = i;
    if (i + 1 == controls->IndexCapacity || index[i + 1].Key != key)
        ranges[key].End = i + 1;
}
extern "C" __global__ void VcmCameraEstimateKernel(
    const PackedAccelerationNode* nodes, const PackedPrimitiveRef* references,
    const PackedTriangle* triangles, const PackedSphere* spheres, const PackedMaterial* materials,
    const PackedTexture* textures, const unsigned* texels, const Emitter* emitters,
    const FrameConstants* frame, const V::WalkControls* walks, const V::FrameControls* controls,
    const V::Path* lights, const V::Path* cameras, const V::IndexEntry* index, const V::BucketRange* ranges,
    V::PixelState* pixels, GroupStatistics* statistics, unsigned* errors)
{
    const unsigned local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= walks->Count)
        return;
    const unsigned pixel = walks->Begin + local;
    if (pixel >= frame->Width * frame->Height)
        return;
    const SceneView scene{nodes,    references, triangles, spheres, materials,
                          textures, texels,     emitters,  *frame,  *walks};
    const auto coupling = FrameCouplings(*frame, *walks, *controls);
    auto random =
        MakeSampleContext(pixel % frame->Width + frame->TileX, pixel / frame->Width + frame->TileY,
                          frame->FullWidth, walks->Pass, frame->TotalSamples, frame->SamplingSequence, frame->SamplingSeed);
    random.PixelSeed = Hash(random.PixelSeed ^ Hash(walks->Seed ^ CameraDomain));
    const unsigned paired = unsigned(double(Sample2D(random, PairingDomain).x) * controls->Paths);
    const auto& light = lights[paired < controls->Paths ? paired : controls->Paths - 1];
    const auto& camera = cameras[local];
    double sum[3]{};
    GroupStatistics counters{};
    for (unsigned t = 2; t <= camera.Count; ++t)
    {
        for (unsigned s = 0; s <= light.Count && s + t <= walks->VertexLimit; ++s)
        {
            const V::CouplingQuery query{0, 0, s, t, 0, 0, 0, 0};
            AccumulateContribution(sum, Couple(scene, &light, camera, query, coupling, counters), errors);
        }
        GatherMerges(scene, *controls, coupling, lights, camera, t, index, ranges, sum, counters, errors);
    }
    for (unsigned channel = 0; channel < 3; ++channel)
        pixels[pixel].Pass[channel] = sum[channel];
    statistics[local] = counters;
}
extern "C" __global__ void VcmSplatEstimateKernel(
    const PackedAccelerationNode* nodes, const PackedPrimitiveRef* references,
    const PackedTriangle* triangles, const PackedSphere* spheres, const PackedMaterial* materials,
    const PackedTexture* textures, const unsigned* texels, const Emitter* emitters,
    const FrameConstants* frame, const V::WalkControls* walks, const V::FrameControls* controls,
    const V::Path* lights, const V::Path* lensRoot, V::IndexEntry* index, V::Contribution* splats,
    GroupStatistics* statistics, unsigned* errors)
{
    const unsigned local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= walks->Count)
        return;
    const unsigned slot = walks->Begin + local;
    if (slot >= controls->IndexCapacity)
        return;
    V::IndexEntry entry{V::InvalidIndex, slot};
    V::Contribution contribution{};
    GroupStatistics counters{};
    const unsigned slots = walks->VertexLimit - 2, path = slot / slots, prefix = slot % slots + 2;
    if (path < controls->Paths && prefix <= lights[path].Count)
    {
        const unsigned seed = Hash(walks->Seed ^ SplatDomain) ^ Hash(walks->Pass ^ PassDomain) ^ Hash(path);
        const SampleContext random{path, 0, seed, walks->Pass, 1, IndependentSequence};
        const auto lens = Sample2D(random, LensDomain ^ (prefix * BounceDomain));
        const V::CouplingQuery query{path, 0, prefix, 1, lens.x, lens.y, 0, 0};
        const SceneView scene{nodes,    references, triangles, spheres, materials,
                              textures, texels,     emitters,  *frame,  *walks};
        contribution = Couple(scene, lights + path, *lensRoot, query,
                              FrameCouplings(*frame, *walks, *controls), counters);
        if (contribution.Flags & V::InvalidContribution)
            atomicOr(errors, V::TransportError);
        if (Nonzero(contribution.Radiance))
        {
            const unsigned rasterX = unsigned(contribution.RasterX * frame->FullWidth);
            const unsigned rasterY = unsigned(contribution.RasterY * frame->FullHeight);
            const unsigned x = rasterX < frame->FullWidth ? rasterX : frame->FullWidth - 1;
            const unsigned y = rasterY < frame->FullHeight ? rasterY : frame->FullHeight - 1;
            if (x >= frame->TileX && y >= frame->TileY && x - frame->TileX < frame->Width &&
                y - frame->TileY < frame->Height)
                entry.Key = y * frame->FullWidth + x;
        }
    }
    index[slot] = entry;
    if (path < controls->Paths)
        splats[slot] = contribution;
    statistics[local] = counters;
}
extern "C" __global__ void VcmCommitPassKernel(const FrameConstants* frame, const V::WalkControls* walks,
                                               const V::FrameControls* controls, const V::IndexEntry* index,
                                               const V::Contribution* splats, V::PixelState* pixels,
                                               Vec4* film, unsigned* counts, unsigned* errors)
{
    const unsigned local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= walks->Count)
        return;
    const unsigned pixel = walks->Begin + local;
    if (pixel >= frame->Width * frame->Height)
        return;
    auto& state = pixels[pixel];
    if (state.CompletedPasses != walks->Pass)
    {
        atomicOr(errors, V::ConfigurationError);
        return;
    }
    const unsigned key =
        (pixel / frame->Width + frame->TileY) * frame->FullWidth + pixel % frame->Width + frame->TileX;
    unsigned lower = 0, upper = controls->IndexCapacity;
    while (lower < upper)
    {
        const unsigned middle = lower + (upper - lower) / 2;
        if (index[middle].Key < key)
            lower = middle + 1;
        else
            upper = middle;
    }
    const double splatScale = double(frame->FullWidth) * frame->FullHeight / controls->Paths;
    for (unsigned i = lower; i < controls->IndexCapacity && index[i].Key == key; ++i)
    {
        const auto color = splats[index[i].Reference].Radiance;
        state.Pass[0] += double(color.X) * splatScale;
        state.Pass[1] += double(color.Y) * splatScale;
        state.Pass[2] += double(color.Z) * splatScale;
    }
    const double passes = double(walks->Pass) + 1;
    float mean[3];
    for (unsigned channel = 0; channel < 3; ++channel)
    {
        state.Sum[channel] += state.Pass[channel];
        mean[channel] = float(state.Sum[channel] / passes);
        if (!isfinite(mean[channel]) || mean[channel] < 0)
        {
            atomicOr(errors, V::TransportError);
            return;
        }
    }
    // Shared populations and changing bandwidth do not supply independent M2.
    film[pixel] = {mean[0], mean[1], mean[2], 0};
    counts[pixel] = state.CompletedPasses = walks->Pass + 1;
}
