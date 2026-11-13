// Independent CUDA Point GI. Diffuse disks cache reflected radiance; closed
// dielectric solids transfer light through interior random walks between disks.
// Each direct bake evaluates source visibility against the actual scene; each
// subsequent order integrates the nearest disk's PREVIOUS reflected radiance.
// No photon density estimate or cross-backend code is used.
// Christensen (2008), "Point-Based Approximate Color Bleeding". This disk model
// is biased at patch boundaries; hemisphere integration resolves overlap before
// accumulation. Cache emission is excluded because source sampling supplies it.
// Sample windows bound kernel duration. Finish every window/surfel of an order
// before swapping the immutable previous-order buffer with the next buffer.
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cfloat>
#include "PointGiLayout.h"
#include "InstanceLayout.h"
#include "EmitterData.cuh"
#include "LightSelection.cu"
#include "LightEmission.cu"
#include "SurfaceEmission.cu"
#include "BsdfTypes.cuh"
#include "Fresnel.cu"
#include "Ggx.cu"
#include "Diffuse.cu"
#include "Conductor.cu"
#include "Dielectric.cu"
#include "SubsurfaceScattering.cu"

namespace
{
namespace G = Arvo::RayTracing::CudaPointGi;
namespace E = ArvoCudaEndpoint;
using Vec3 = G::Vector;
constexpr unsigned InvalidPrimitiveIndex = G::Invalid, SphereBit = 0x80000000u;
constexpr unsigned ImportanceBit = 1, NeeBit = 2, Channels = 3, TraversalStack = 64;
constexpr unsigned SourceDomain = 0x534f5552u, PositionDomain = 0x504f534eu, GatherDomain = 0x47415448u,
                   BackSideSeed = 0x4241434bu;
constexpr unsigned InterfaceDomain = 0x53534946u, BranchDomain = 0x53534252u,
                   FlightDomain = 0x5353464cu, PhaseDomain = 0x53535048u,
                   BounceScramble = 0x9e3779b9u;
constexpr unsigned PixelSeedMultiplier = 9781, SampleSeedMultiplier = 6271,
                   SecondDimensionScramble = 0x68bc21ebu;
constexpr float Pi = CUDART_PI_F, RayEpsilon = 1e-4f, MaximumRayDistance = 1e30f;
constexpr float BasisParallelThreshold = .999f, MinimumDirectionLengthSquared = 1e-12f,
                MinimumTriangleDeterminant = 1e-8f;
constexpr double IntegralPi = 3.1415926535897932384626433832795;
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
// Existing CUDA scene upload ABI, independent of the selected estimator.
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
static_assert(sizeof(PackedTriangle) == 192 && sizeof(PackedAccelerationNode) == 48 &&
              sizeof(GroupStatistics) == 32);
__device__ Vec3 V(float3 p)
{
    return {p.x, p.y, p.z};
}
__device__ Vec3 ToVec3(Vec4 p)
{
    return {p.X, p.Y, p.Z};
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
__device__ Vec3 Normalize(Vec3 a)
{
    float length = norm3df(a.X, a.Y, a.Z);
    return length > 0 ? Multiply(a, 1 / length) : Vec3{};
}
__device__ float Component(Vec3 v, unsigned i)
{
    return i == 0 ? v.X : i == 1 ? v.Y : v.Z;
}
__device__ unsigned Hash(unsigned value)
{
    constexpr unsigned FirstMix = 0x7feb352du, SecondMix = 0x846ca68bu, OuterShift = 16, InnerShift = 15;
    value = (value ^ (value >> OuterShift)) * FirstMix;
    value = (value ^ (value >> InnerShift)) * SecondMix;
    return value ^ (value >> OuterShift);
}
__device__ float Random01(unsigned seed)
{
    constexpr unsigned MantissaMask = 0xffffffu;
    constexpr float MantissaRange = 16777216.f;
    return float(Hash(seed) & MantissaMask) / MantissaRange;
}
#include "Sequence.cu"
#include "Importance.cu"
#include "Triangle.cu"
#include "Sphere.cu"
#include "Bvh.cu"
#include "KdTree.cu"
#include "Bih.cu"
#include "Dispatch.cu"
#include "ThinLens.cu"
#include "FilmMoments.cu"
#include "SurfaceScattering.cu"

__device__ double3 Wide(Vec3 p)
{
    return make_double3(p.X, p.Y, p.Z);
}
__device__ double3 Plus(double3 a, double3 b)
{
    return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ double3 Minus(double3 a, double3 b)
{
    return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ double3 Scale(double3 a, double b)
{
    return make_double3(a.x * b, a.y * b, a.z * b);
}
__device__ double3 Product(double3 a, double3 b)
{
    return make_double3(a.x * b.x, a.y * b.y, a.z * b.z);
}
__device__ double Inner(double3 a, double3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ double Length(double3 a)
{
    return norm3d(a.x, a.y, a.z);
}
__device__ bool Valid(double3 a)
{
    return isfinite(a.x) && isfinite(a.y) && isfinite(a.z) && a.x >= 0 && a.y >= 0 && a.z >= 0;
}
__device__ Vec3 Narrow(double3 a, unsigned* errors)
{
    if (!isfinite(a.x) || !isfinite(a.y) || !isfinite(a.z) || fabs(a.x) > FLT_MAX || fabs(a.y) > FLT_MAX ||
        fabs(a.z) > FLT_MAX)
    {
        atomicOr(errors, G::NumericalError);
        return {};
    }
    return {float(a.x), float(a.y), float(a.z)};
}
struct World
{
    const PackedAccelerationNode* Nodes;
    const PackedPrimitiveRef* References;
    const PackedTriangle* Triangles;
    const PackedSphere* Spheres;
    const PackedMaterial* Materials;
    const PackedTexture* Textures;
    const unsigned* Texels;
    const E::PackedEmitter* Sources;
    const FrameConstants& Frame;
    const G::Controls& Control;
    unsigned* Errors;
};
__device__ bool Intersect(const World& world, Ray ray, Hit& hit, unsigned long long& tests, bool any = false)
{
    // Traversal narrows an existing closest-hit distance. Both camera and
    // visibility rays start a fresh query with the ray's complete interval.
    hit = {};
    hit.Distance = ray.MaximumDistance;
    hit.PrimitiveIndex = InvalidPrimitiveIndex;
    return IntersectScene(world.Nodes, world.References, world.Triangles, world.Spheres, world.Frame, ray,
                          any, hit, tests);
}
__device__ float Decode(unsigned value)
{
    constexpr float ByteRange = 255, Threshold = .04045f, LinearSlope = 12.92f;
    constexpr float Offset = .055f, ScaleFactor = 1.055f, Exponent = 2.4f;
    const float encoded = value / ByteRange;
    return encoded <= Threshold ? encoded / LinearSlope : powf((encoded + Offset) / ScaleFactor, Exponent);
}
__device__ Vec3 Texel(const World& world, const PackedTexture& texture, int x, int y)
{
    x = (x % int(texture.Width) + int(texture.Width)) % int(texture.Width);
    y = (y % int(texture.Height) + int(texture.Height)) % int(texture.Height);
    const unsigned long long address = texture.OffsetTexels + (unsigned long long)y * texture.Width + x;
    if (address >= world.Frame.TextureTexelCount)
    {
        atomicOr(world.Errors, G::BoundsError);
        return {};
    }
    constexpr unsigned ByteMask = 0xffu, GreenShift = 8, BlueShift = 16;
    const unsigned rgba = world.Texels[address];
    return {Decode(rgba & ByteMask), Decode((rgba >> GreenShift) & ByteMask),
            Decode((rgba >> BlueShift) & ByteMask)};
}
__device__ double3 Rho(const World& world, const G::Surfel& point)
{
    if (point.Material >= world.Frame.MaterialCount)
    {
        atomicOr(world.Errors, G::BoundsError);
        return {};
    }
    const auto& material = world.Materials[point.Material];
    if (material.SampledEmitter)
        return {};
    if (material.Model == DielectricMaterial && material.Optical.W > 0)
        return {};
    if (material.Model != DiffuseMaterial &&
        !(material.Model == LegacyMaterial && material.AlbedoReflectivity.W == 0 &&
          material.Specular.X == 0 && material.Specular.Y == 0 && material.Specular.Z == 0))
    {
        atomicOr(world.Errors, G::ConfigurationError);
        return {};
    }
    Vec3 texture{1, 1, 1};
    if (point.HasTexCoord && material.DiffuseTextureIndex < world.Frame.TextureCount)
    {
        const auto& descriptor = world.Textures[material.DiffuseTextureIndex];
        if (descriptor.Width && descriptor.Height)
        {
            const float u = point.TexCoord[0] - floorf(point.TexCoord[0]);
            const float v = point.TexCoord[1] - floorf(point.TexCoord[1]);
            const float x = u * descriptor.Width - .5f, y = (1 - v) * descriptor.Height - .5f;
            const int column = int(floorf(x)), row = int(floorf(y));
            const float fx = x - column, fy = y - row;
            const Vec3 top = Add(Multiply(Texel(world, descriptor, column, row), 1 - fx),
                                 Multiply(Texel(world, descriptor, column + 1, row), fx));
            const Vec3 bottom = Add(Multiply(Texel(world, descriptor, column, row + 1), 1 - fx),
                                    Multiply(Texel(world, descriptor, column + 1, row + 1), fx));
            texture = Add(Multiply(top, 1 - fy), Multiply(bottom, fy));
        }
    }
    double3 albedo = Wide(ToVec3(material.AlbedoReflectivity)), color = Wide(point.Color),
            tex = Wide(texture);
    if (!Valid(albedo) || !Valid(color) || !Valid(tex))
    {
        atomicOr(world.Errors, G::NumericalError);
        return {};
    }
    return make_double3(fmin(1., albedo.x * color.x * tex.x), fmin(1., albedo.y * color.y * tex.y),
                        fmin(1., albedo.z * color.z * tex.z));
}
__device__ double3 Reflected(double3 rho, double3 irradiance)
{
    return make_double3(rho.x * irradiance.x / IntegralPi, rho.y * irradiance.y / IntegralPi,
                        rho.z * irradiance.z / IntegralPi);
}
struct RectHit
{
    unsigned Source = G::Invalid;
    double Distance = 0;
    bool Front = false;
};
__device__ RectHit Rectangle(const World& world, Ray ray, double maximum, unsigned long long& tests)
{
    RectHit hit;
    for (unsigned source = 0; source < world.Control.Sources; ++source)
    {
        const auto& emitter = world.Sources[source];
        if (emitter.Kind != E::Rectangle)
            continue;
        ++tests;
        const Vec3 n = Normalize(V(emitter.Direction));
        const double denominator = Inner(Wide(n), Wide(ray.Direction));
        if (denominator == 0)
            continue;
        double3 relative = Minus(Wide(ray.Origin), Wide(V(emitter.Origin)));
        double distance = -Inner(relative, Wide(n)) / denominator;
        if (!(distance > ray.MinimumDistance && distance < maximum))
            continue;
        Vec3 helper = fabsf(n.Y) < BasisParallelThreshold ? Vec3{0, 1, 0} : Vec3{1, 0, 0};
        Vec3 tangent = Normalize(Cross(helper, n)), bitangent = Cross(n, tangent);
        double3 p = Plus(relative, Scale(Wide(ray.Direction), distance));
        if (fabs(Inner(p, Wide(tangent))) <= emitter.Width / 2 &&
            fabs(Inner(p, Wide(bitangent))) <= emitter.Height / 2)
        {
            hit = {source, distance, denominator < 0};
            maximum = distance;
        }
    }
    return hit;
}
struct DiskHit
{
    unsigned Surfel = G::Invalid;
    double Distance = CUDART_INF;
    bool Front = false;
};
__device__ DiskHit Disks(const World& world, const G::Surfel* points, const G::Node* nodes,
                         const unsigned* order, Ray ray, unsigned ignored, unsigned long long& tests)
{
    DiskHit hit;
    if (!world.Control.Surfels)
        return hit;
    unsigned stack[TraversalStack], pending = 1, visited = 0;
    stack[0] = 0;
    while (pending)
    {
        unsigned slot = stack[--pending];
        if (slot >= world.Control.Nodes || ++visited > world.Control.Nodes)
        {
            atomicOr(world.Errors, G::BoundsError);
            return {};
        }
        const auto& node = nodes[slot];
        double near = ray.MinimumDistance, far = fmin(hit.Distance, double(ray.MaximumDistance));
        bool overlaps = true;
        for (unsigned axis = 0; axis < Channels; ++axis)
        {
            double origin = Component(ray.Origin, axis), direction = Component(ray.Direction, axis);
            if (direction == 0)
            {
                if (origin < node.Low[axis] || origin > node.High[axis])
                    overlaps = false;
                continue;
            }
            double a = (node.Low[axis] - origin) / direction, b = (node.High[axis] - origin) / direction;
            near = fmax(near, fmin(a, b));
            far = fmin(far, fmax(a, b));
            if (near > far)
                overlaps = false;
        }
        if (!overlaps)
            continue;
        if (node.Left != G::Invalid)
        {
            if (pending + 2 > TraversalStack)
            {
                atomicOr(world.Errors, G::BoundsError);
                return {};
            }
            stack[pending++] = node.Right;
            stack[pending++] = node.Left;
            continue;
        }
        if ((unsigned long long)node.Begin + node.Count > world.Control.Surfels)
        {
            atomicOr(world.Errors, G::BoundsError);
            return {};
        }
        for (unsigned i = 0; i < node.Count; ++i)
        {
            unsigned identity = order[node.Begin + i];
            if (identity >= world.Control.Surfels)
            {
                atomicOr(world.Errors, G::BoundsError);
                return {};
            }
            if (identity == ignored)
                continue;
            ++tests;
            const auto& point = points[identity];
            double denominator = Inner(Wide(point.GeometryNormal), Wide(ray.Direction));
            if (denominator == 0)
                continue;
            double3 relative = Minus(Wide(ray.Origin), Wide(point.Position));
            double distance = -Inner(relative, Wide(point.GeometryNormal)) / denominator;
            if (!(distance > ray.MinimumDistance && distance < ray.MaximumDistance) ||
                distance > hit.Distance || (distance == hit.Distance && identity >= hit.Surfel))
                continue;
            double3 radial = Plus(relative, Scale(Wide(ray.Direction), distance));
            if (Inner(radial, radial) <= point.Radius * point.Radius)
                hit = {identity, distance, denominator < 0};
        }
    }
    return hit;
}
__device__ SampleContext BakeSample(const World& world, unsigned identity, bool front, unsigned sample,
                                    unsigned count)
{
    return {
        identity, front ? 0u : 1u, Hash(identity ^ Hash(world.Control.Seed) ^ (front ? 0u : BackSideSeed)),
        sample,   count,           world.Frame.SamplingSequence, world.Frame.SamplingSeed};
}
__device__ double3 Direct(const World& world, const G::Surfel& point, bool front, SampleContext sample,
                          GroupStatistics& statistics)
{
    if (!world.Control.Sources)
        return {};
    const auto selected = ArvoCudaSelection::Select(world.Control.Sources, Sample2D(sample, SourceDomain).x);
    if (!(selected.Probability > 0))
    {
        atomicOr(world.Errors, G::ConfigurationError);
        return {};
    }
    const auto& emitter = world.Sources[selected.Index];
    Vec3 n = Multiply(point.ShadingNormal, front ? 1.f : -1.f),
         ng = Multiply(point.GeometryNormal, front ? 1.f : -1.f);
    float2 uv = Sample2D(sample, PositionDomain);
    Vec3 wi{}, endpoint{};
    double3 value = Wide(V(emitter.Strength));
    double weight = 1 / double(selected.Probability);
    bool infinite = emitter.Kind == E::Directional || emitter.Kind == E::Environment;
    if (emitter.Kind == E::Directional)
        wi = Normalize(V(emitter.Direction));
    else if (emitter.Kind == E::Environment)
    {
        wi = SampleDiffuseDirection(n, uv, world.Frame.TransportFlags);
        double pdf = DiffusePdf(Dot(n, wi), world.Frame.TransportFlags);
        if (!(pdf > 0))
            return {};
        weight /= pdf;
        double fraction = fmin(1., fmax(0., (double(wi.Y) + 1) / 2));
        value = Plus(Scale(Wide(V(emitter.BackgroundBottom)), 1 - fraction),
                     Scale(Wide(V(emitter.BackgroundTop)), fraction));
    }
    else
    {
        Vec3 sourceNormal{};
        float positionPdf = 1;
        unsigned valid = 1;
        if (emitter.Kind == E::Point)
            endpoint = V(emitter.Origin);
        else if (emitter.Kind == E::Rectangle)
        {
            ArvoCudaEmission::Source source{ArvoCudaEmission::Rectangle,
                                            emitter.Origin,
                                            emitter.Direction,
                                            emitter.Strength,
                                            {},
                                            {},
                                            make_float2(emitter.Width, emitter.Height)};
            auto location = ArvoCudaEmission::Emit(source, {}, uv, make_float2(.5f, .5f));
            endpoint = V(location.Origin);
            sourceNormal = V(location.Normal);
            value = Wide(V(location.Value));
            positionPdf = location.PositionPdf;
            valid = location.Flags;
        }
        else if (emitter.Kind == E::Triangle || emitter.Kind == E::Sphere)
        {
            ArvoCudaSurfaceEmission::Source source{emitter.Kind == E::Triangle
                                                       ? ArvoCudaSurfaceEmission::Triangle
                                                       : ArvoCudaSurfaceEmission::Ellipsoid,
                                                   emitter.Origin,
                                                   emitter.AxisX,
                                                   emitter.AxisY,
                                                   emitter.AxisZ,
                                                   emitter.Strength};
            auto location = ArvoCudaSurfaceEmission::Emit(source, uv, make_float2(.5f, .5f));
            endpoint = V(location.Origin);
            sourceNormal = V(location.Normal);
            value = Wide(V(location.Value));
            positionPdf = location.PositionPdf;
            valid = location.Flags;
        }
        else
        {
            atomicOr(world.Errors, G::ConfigurationError);
            return {};
        }
        if (!(valid & ArvoCudaEmission::Valid) || !(positionPdf > 0) || !isfinite(positionPdf))
        {
            atomicOr(world.Errors, G::NumericalError);
            return {};
        }
        double3 delta = Minus(Wide(endpoint), Wide(point.Position));
        double distance = Length(delta);
        if (!(distance > RayEpsilon))
            return {};
        wi = Narrow(Scale(delta, 1 / distance), world.Errors);
        weight /= distance * distance;
        if (emitter.Kind != E::Point)
        {
            double cosine = -Dot(sourceNormal, wi);
            weight *= (emitter.Kind == E::Rectangle ? fmax(0., cosine) : fabs(cosine)) / positionPdf;
        }
    }
    double cosine = Dot(n, wi);
    if (!(cosine > 0 && Dot(ng, wi) > 0 && weight > 0))
        return {};
    Ray shadow{Add(point.Position, Multiply(ng, RayEpsilon)), wi, RayEpsilon, MaximumRayDistance};
    if (!infinite)
    {
        double3 delta = Minus(Wide(endpoint), Wide(shadow.Origin));
        double distance = Length(delta);
        if (!(distance > 2 * RayEpsilon) || distance > MaximumRayDistance)
            return {};
        shadow.Direction = Narrow(Scale(delta, 1 / distance), world.Errors);
        shadow.MaximumDistance = float(distance - RayEpsilon);
    }
    Hit hit;
    ++statistics.ShadowRays;
    if (Intersect(world, shadow, hit, statistics.PrimitiveTests, true) ||
        Rectangle(world, shadow, shadow.MaximumDistance, statistics.PrimitiveTests).Source != G::Invalid)
        return {};
    return Scale(value, weight * cosine);
}
__device__ double3 Gather(const World& world, const G::Surfel* points, const G::Node* nodes,
                          const unsigned* order, const G::Radiance* previous, const G::Surfel& receiver,
                          unsigned identity, bool front, SampleContext sample, GroupStatistics& statistics)
{
    Vec3 n = Multiply(receiver.ShadingNormal, front ? 1.f : -1.f),
         ng = Multiply(receiver.GeometryNormal, front ? 1.f : -1.f);
    Vec3 wi = SampleDiffuseDirection(n, Sample2D(sample, GatherDomain), world.Frame.TransportFlags);
    double cosine = Dot(n, wi), pdf = DiffusePdf(float(cosine), world.Frame.TransportFlags);
    if (!(pdf > 0 && Dot(ng, wi) > 0))
        return {};
    Ray ray{Add(receiver.Position, Multiply(ng, RayEpsilon)), wi, RayEpsilon, MaximumRayDistance};
    ++statistics.ReflectionRays;
    auto hit = Disks(world, points, nodes, order, ray, identity, statistics.PrimitiveTests);
    if (hit.Surfel == G::Invalid ||
        Rectangle(world, ray, hit.Distance, statistics.PrimitiveTests).Source != G::Invalid)
        return {};
    const double* value = hit.Front ? previous[hit.Surfel].Front : previous[hit.Surfel].Back;
    return Scale(make_double3(value[0], value[1], value[2]), cosine / pdf);
}
struct SubsurfaceExit
{
    G::Surfel Receiver{};
    double3 Throughput{};
    bool Reached = false;
};
__device__ SubsurfaceExit WalkSubsurface(const World& world, Ray ray, Hit entry, Vec3 position,
                                         bool cameraInside, SampleContext random,
                                         GroupStatistics& statistics)
{
    SubsurfaceExit result;
    if (entry.MaterialIndex >= world.Frame.MaterialCount)
    {
        atomicOr(world.Errors, G::BoundsError);
        return result;
    }
    const PackedMaterial material = world.Materials[entry.MaterialIndex];
    double3 throughput = make_double3(1, 1, 1);
    const bool importance = (world.Frame.TransportFlags & ImportanceBit) != 0;
    if (!cameraInside)
    {
        const PhysicalSurface surface = MakePhysicalSurface(material, entry,
            Multiply(ray.Direction, -1), entry.Color);
        const PhysicalSample launch = SamplePhysical(surface, Sample2D(random, InterfaceDomain),
            Sample2D(random, BranchDomain).x, importance);
        if (!(launch.Event & ArvoCuda::ScatteringTransmission) || !(launch.Pdf > 0))
            return result;
        throughput = Product(throughput, Wide(launch.Weight));
        ray = {PhysicalRayOrigin(surface, position, launch.Direction), launch.Direction,
               RayEpsilon, MaximumRayDistance};
    }
    for (unsigned bounce = 0; bounce < G::MaximumReflections; ++bounce)
    {
        Hit boundary;
        ++statistics.ReflectionRays;
        if (!Intersect(world, ray, boundary, statistics.PrimitiveTests) ||
            boundary.MaterialIndex != entry.MaterialIndex)
            return result;
        const unsigned dimension = (bounce + 1) * BounceScramble;
        const auto flight = SubsurfaceFreeFlight(material.Optical.W,
            make_float3(material.AlbedoReflectivity.X * material.ConductorExtinction.W,
                        material.AlbedoReflectivity.Y * material.ConductorExtinction.W,
                        material.AlbedoReflectivity.Z * material.ConductorExtinction.W),
            make_float3(ray.Direction.X, ray.Direction.Y, ray.Direction.Z), boundary.Distance,
            Sample2D(random, dimension ^ FlightDomain).x);
        throughput = Product(throughput, Wide(V(flight.Weight)));
        if (!Valid(throughput) || !(throughput.x > 0 || throughput.y > 0 || throughput.z > 0))
            return result;
        if (flight.Scattered)
        {
            const auto phase = SubsurfaceSampleDirection(
                make_float3(ray.Direction.X, ray.Direction.Y, ray.Direction.Z), material.ConductorIndex.W,
                Sample2D(random, dimension ^ PhaseDomain), importance);
            if (!(phase.Pdf > 0)) return result;
            throughput = Scale(throughput, phase.Weight);
            ray = {Add(ray.Origin, Multiply(ray.Direction, flight.Distance)), V(phase.Direction),
                   RayEpsilon, MaximumRayDistance};
            continue;
        }
        const Vec3 boundaryPosition = Add(ray.Origin, Multiply(ray.Direction, boundary.Distance));
        const PhysicalSurface surface = MakePhysicalSurface(material, boundary,
            Multiply(ray.Direction, -1), boundary.Color);
        const PhysicalSample event = SamplePhysical(surface,
            Sample2D(random, dimension ^ InterfaceDomain),
            Sample2D(random, dimension ^ BranchDomain).x, importance);
        if (!event.Event || !(event.Pdf > 0)) return result;
        throughput = Product(throughput, Wide(event.Weight));
        if (!Valid(throughput) || !(throughput.x > 0 || throughput.y > 0 || throughput.z > 0))
            return result;
        if (!(event.Event & ArvoCuda::ScatteringTransmission))
        {
            ray = {PhysicalRayOrigin(surface, boundaryPosition, event.Direction), event.Direction,
                   RayEpsilon, MaximumRayDistance};
            continue;
        }
        G::Surfel receiver{};
        receiver.Position = boundaryPosition;
        receiver.GeometryNormal = boundary.GeometricNormal;
        receiver.ShadingNormal = Dot(boundary.Normal, receiver.GeometryNormal) >= 0
            ? boundary.Normal : Multiply(boundary.Normal, -1);
        receiver.Material = boundary.MaterialIndex;
        receiver.Primitive = boundary.PrimitiveIndex;
        receiver.Color = boundary.Color;
        receiver.HasTexCoord = boundary.HasTexCoord;
        receiver.TexCoord[0] = boundary.TexCoordX;
        receiver.TexCoord[1] = boundary.TexCoordY;
        result.Receiver = receiver;
        result.Throughput = throughput;
        result.Reached = true;
        return result;
    }
    return result;
}
__device__ Hit SurfelHit(const G::Surfel& point)
{
    Hit hit{};
    hit.Normal = point.ShadingNormal;
    hit.GeometricNormal = point.GeometryNormal;
    hit.FrontFace = 1;
    hit.Color = point.Color;
    hit.MaterialIndex = point.Material;
    hit.PrimitiveIndex = point.Primitive;
    hit.TexCoordX = point.TexCoord[0];
    hit.TexCoordY = point.TexCoord[1];
    hit.HasTexCoord = point.HasTexCoord;
    return hit;
}
__device__ double3 BakeSubsurface(const World& world, const G::Surfel* points,
                                   const G::Node* nodes, const unsigned* order,
                                   const G::Radiance* previous, const G::Surfel& point,
                                   SampleContext random, GroupStatistics& statistics)
{
    Ray launch{point.Position, Multiply(point.GeometryNormal, -1), RayEpsilon, MaximumRayDistance};
    const auto exit = WalkSubsurface(world, launch, SurfelHit(point), point.Position,
                                     false, random, statistics);
    if (!exit.Reached) return {};
    const double3 incident = world.Control.Order
        ? Gather(world, points, nodes, order, previous, exit.Receiver, G::Invalid, true,
                 random, statistics)
        : Direct(world, exit.Receiver, true, random, statistics);
    return Scale(Product(exit.Throughput, incident), 1 / IntegralPi);
}
__device__ void StoreRadiance(double* target, double3 value, unsigned* errors)
{
    if (!Valid(value))
    {
        atomicOr(errors, G::NumericalError);
        return;
    }
    target[0] = value.x;
    target[1] = value.y;
    target[2] = value.z;
}
__device__ bool ControlsValid(const World& world)
{
    constexpr unsigned MaximumBakeSamples = 65536, MaximumSurfels = 1048576, PointGiAlgorithm = 8;
    const auto& control = world.Control;
    if (world.Frame.Integrator != PointGiAlgorithm ||
        world.Frame.MaxReflectionBounces > G::MaximumReflections || world.Frame.AdaptiveEnabled ||
        world.Frame.ShadowStrength != 1 || (world.Frame.TransportFlags & ~ImportanceBit) != NeeBit ||
        world.Frame.SamplingSequence > BlueNoiseSequence || !control.DirectSamples ||
        control.DirectSamples > MaximumBakeSamples || !control.GatherSamples ||
        control.GatherSamples > MaximumBakeSamples || control.Surfels > MaximumSurfels ||
        !control.SampleCount || control.SampleCount > G::SamplesPerDispatch)
    {
        atomicOr(world.Errors, G::ConfigurationError);
        return false;
    }
    return true;
}
} // namespace

extern "C" __global__ void PointGiBakeKernel(
    const PackedAccelerationNode* nodes, const PackedPrimitiveRef* references,
    const PackedTriangle* triangles, const PackedSphere* spheres, const PackedMaterial* materials,
    const PackedTexture* textures, const unsigned* texels, const E::PackedEmitter* sources,
    const FrameConstants* frame, const G::Controls* controls, const G::Surfel* surfels,
    const G::Node* hierarchy, const unsigned* order, const G::Radiance* previous, G::Radiance* next,
    G::Radiance* total, GroupStatistics* statistics, unsigned* errors)
{
    unsigned lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= controls->Count)
        return;
    statistics[lane] = {};
    World world{nodes,  references, triangles, spheres,   materials, textures,
                texels, sources,    *frame,    *controls, errors};
    if (!ControlsValid(world))
        return;
    unsigned identity = controls->Begin + lane;
    const unsigned count = controls->Order ? controls->GatherSamples : controls->DirectSamples;
    if (identity < controls->Begin || identity >= controls->Surfels ||
        controls->Order >= frame->MaxReflectionBounces || controls->SampleBegin > count ||
        controls->SampleCount > count - controls->SampleBegin || next == total ||
        (controls->Order && (previous == next || previous == total)))
    {
        atomicOr(errors, G::BoundsError);
        return;
    }
    const auto& point = surfels[identity];
    double3 rho = Rho(world, point);
    const bool subsurface = point.Material < frame->MaterialCount &&
        materials[point.Material].Model == DielectricMaterial &&
        materials[point.Material].Optical.W > 0;
    G::Radiance estimate{};
    GroupStatistics work{};
    if (rho.x > 0 || rho.y > 0 || rho.z > 0 || subsurface)
        for (unsigned side = 0; side < 2; ++side)
        {
            const bool front = side == 0;
            if (subsurface && !front) continue;
            double3 irradiance{};
            for (unsigned sample = controls->SampleBegin;
                 sample < controls->SampleBegin + controls->SampleCount; ++sample)
            {
                auto random = BakeSample(world, identity, front, sample, count);
                double3 value = subsurface
                    ? BakeSubsurface(world, surfels, hierarchy, order, previous, point, random, work)
                    : controls->Order ? Gather(world, surfels, hierarchy, order, previous, point,
                                               identity, front, random, work)
                                      : Direct(world, point, front, random, work);
                irradiance = Plus(irradiance, Scale(value, 1. / count));
            }
            StoreRadiance(front ? estimate.Front : estimate.Back,
                          subsurface ? irradiance : Reflected(rho, irradiance), errors);
        }
    const G::Radiance earlier = controls->SampleBegin ? next[identity] : G::Radiance{};
    for (unsigned channel = 0; channel < Channels; ++channel)
    {
        next[identity].Front[channel] = earlier.Front[channel] + estimate.Front[channel];
        next[identity].Back[channel] = earlier.Back[channel] + estimate.Back[channel];
        if (!isfinite(next[identity].Front[channel]) || !isfinite(next[identity].Back[channel]))
            atomicOr(errors, G::NumericalError);
    }
    if (!controls->Order)
        total[identity] = next[identity];
    else
        for (unsigned channel = 0; channel < Channels; ++channel)
        {
            total[identity].Front[channel] += estimate.Front[channel];
            total[identity].Back[channel] += estimate.Back[channel];
            if (!isfinite(total[identity].Front[channel]) || !isfinite(total[identity].Back[channel]))
                atomicOr(errors, G::NumericalError);
        }
    statistics[lane] = work;
}

extern "C" __global__ void PointGiCameraKernel(
    const PackedAccelerationNode* nodes, const PackedPrimitiveRef* references,
    const PackedTriangle* triangles, const PackedSphere* spheres, const PackedMaterial* materials,
    const PackedTexture* textures, const unsigned* texels, const E::PackedEmitter* sources,
    const FrameConstants* frame, const G::Controls* controls, const G::Surfel* surfels,
    const G::Node* hierarchy, const unsigned* order, const G::Radiance* total, Vec4* film, unsigned* counts,
    GroupStatistics* statistics, unsigned* errors)
{
    unsigned lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= controls->Count)
        return;
    statistics[lane] = {};
    World world{nodes,  references, triangles, spheres,   materials, textures,
                texels, sources,    *frame,    *controls, errors};
    if (!ControlsValid(world))
        return;
    unsigned pixel = controls->Begin + lane;
    if (pixel < controls->Begin || pixel >= frame->Width * frame->Height || !frame->TotalSamples ||
        controls->SampleBegin > frame->TotalSamples ||
        controls->SampleCount > frame->TotalSamples - controls->SampleBegin)
    {
        atomicOr(errors, G::BoundsError);
        return;
    }
    GroupStatistics work{};
    Vec4 moments = controls->SampleBegin ? film[pixel] : Vec4{};
    unsigned samples = controls->SampleBegin ? counts[pixel] : 0;
    if (samples != controls->SampleBegin)
    {
        atomicOr(errors, G::ConfigurationError);
        return;
    }
    unsigned x = pixel % frame->Width + frame->TileX, y = pixel / frame->Width + frame->TileY;
    for (unsigned sample = controls->SampleBegin; sample < controls->SampleBegin + controls->SampleCount;
         ++sample)
    {
        auto random =
            MakeSampleContext(x, y, frame->FullWidth, sample, frame->TotalSamples, frame->SamplingSequence, frame->SamplingSeed);
        float2 jitter = frame->TotalSamples > 1 || frame->SamplingSequence != IndependentSequence || frame->SamplingSeed != 0
                            ? Sample2D(random, 0)
                            : make_float2(.5f, .5f);
        float sx = (2 * (x + jitter.x) / frame->FullWidth - 1) * frame->HalfWidth;
        float sy = (1 - 2 * (y + jitter.y) / frame->FullHeight) * frame->HalfHeight;
        Ray ray{frame->CameraPosition,
                Normalize(Add(frame->CameraForward,
                              Add(Multiply(frame->CameraRight, sx), Multiply(frame->CameraUp, sy)))),
                RayEpsilon, MaximumRayDistance};
        ApplyThinLens(ray, *frame, Sample2D(random, LensSampleDimension));
        random.PixelSeed = Hash(random.PixelSeed ^ Hash(controls->Seed) ^ GatherDomain);
        ++work.PrimaryRays;
        Hit hit;
        bool found = Intersect(world, ray, hit, work.PrimitiveTests);
        const bool sampledProxy = found && hit.MaterialIndex < frame->MaterialCount &&
                                  materials[hit.MaterialIndex].SampledEmitter;
        RectHit rectangle =
            Rectangle(world, ray,
                      found ? fmin(double(ray.MaximumDistance), double(hit.Distance) +
                                                                    (sampledProxy ? RayEpsilon : 0))
                            : ray.MaximumDistance,
                      work.PrimitiveTests);
        double3 value{};
        if (rectangle.Source != G::Invalid)
        {
            const auto& emitter = sources[rectangle.Source];
            if (rectangle.Front)
                value = Scale(Wide(V(emitter.Strength)), 1 / (double(emitter.Width) * emitter.Height));
        }
        else if (!found)
        {
            double latitude = fmin(1., fmax(0., (double(ray.Direction.Y) + 1) / 2));
            value = Plus(Scale(Wide(frame->BackgroundBottom), 1 - latitude),
                         Scale(Wide(frame->BackgroundTop), latitude));
        }
        else
        {
            if (hit.MaterialIndex >= frame->MaterialCount)
            {
                atomicOr(errors, G::BoundsError);
                return;
            }
            value = Wide(ToVec3(materials[hit.MaterialIndex].Emission));
            G::Surfel point{};
            point.Position = Add(ray.Origin, Multiply(ray.Direction, hit.Distance));
            point.GeometryNormal = Multiply(hit.GeometricNormal, hit.FrontFace ? 1.f : -1.f);
            point.ShadingNormal = hit.Normal;
            if (Dot(point.GeometryNormal, point.ShadingNormal) < 0)
                point.ShadingNormal = Multiply(point.ShadingNormal, -1);
            if (Dot(point.ShadingNormal, ray.Direction) >= 0)
                point.ShadingNormal = point.GeometryNormal;
            point.Material = hit.MaterialIndex;
            point.Primitive = hit.PrimitiveIndex;
            point.Color = hit.Color;
            point.HasTexCoord = hit.HasTexCoord;
            point.TexCoord[0] = hit.TexCoordX;
            point.TexCoord[1] = hit.TexCoordY;
            const auto& material = materials[hit.MaterialIndex];
            if (material.Model == DielectricMaterial && material.Optical.W > 0)
            {
                const auto exit = WalkSubsurface(world, ray, hit, point.Position,
                                                 !hit.FrontFace, random, work);
                if (exit.Reached)
                {
                    double3 irradiance = Direct(world, exit.Receiver, true, random, work);
                    if (frame->MaxReflectionBounces)
                        irradiance = Plus(irradiance, Gather(world, surfels, hierarchy, order, total,
                                                             exit.Receiver, G::Invalid, true, random, work));
                    value = Plus(value, Scale(Product(exit.Throughput, irradiance), 1 / IntegralPi));
                }
            }
            else
            {
                double3 rho = Rho(world, point);
                if ((rho.x > 0 || rho.y > 0 || rho.z > 0) && Dot(point.ShadingNormal, ray.Direction) < 0)
                {
                    double3 irradiance = Direct(world, point, true, random, work);
                    if (frame->MaxReflectionBounces)
                        irradiance = Plus(irradiance, Gather(world, surfels, hierarchy, order, total, point,
                                                             G::Invalid, true, random, work));
                    value = Plus(value, Reflected(rho, irradiance));
                }
            }
        }
        if (!Valid(value))
        {
            atomicOr(errors, G::NumericalError);
            return;
        }
        AccumulateFilmSample(moments, samples, Narrow(value, errors));
        if (!isfinite(moments.X) || !isfinite(moments.Y) || !isfinite(moments.Z) || !isfinite(moments.W))
        {
            atomicOr(errors, G::NumericalError);
            return;
        }
    }
    film[pixel] = moments;
    counts[pixel] = samples;
    statistics[lane] = work;
}
