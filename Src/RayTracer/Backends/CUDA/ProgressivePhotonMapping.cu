// Independent CUDA stochastic progressive photon mapping.
// Hachisuka and Jensen (2009), "Stochastic Progressive Photon Mapping".
// https://pbr-book.org/3ed-2018/Light_Transport_III_Bidirectional_Methods/Stochastic_Progressive_Photon_Mapping
// Camera passes replace visible points; pixel radius/count/flux survive. A
// uniform disk uses N'=N+alpha*M, q=N'/(N+M), R'^2=q*R^2, tau'=q*(tau+Phi).
// Photon deposits include camera throughput. Resolve mean(direct)+tau/(pi*R^2*Nemit).
// Direct/caustic/indirect histories are disjoint; NEE replaces direct deposits.
// No fixed-map estimator is called and no complete photon population is stored.
//
// Index visible-point centers using the initial radius as cell width. Every
// radius only decreases, so 27 neighboring cells cover every current disk.
// Exact cell checks prevent duplicate deposits from hash collisions. Double
// atomics reduce accumulation error; execution order is not bitwise reproducible.
// CUDA stream stage boundaries publish index links and finish deposits before
// progressive updates. Camera variance cannot justify adaptive stopping.
// Interior collision estimates retain a separate three-dimensional bandwidth.
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cfloat>
#include "ProgressivePhotonLayout.h"
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
namespace P = Arvo::RayTracing::CudaPpm;
using Vec3 = P::Vector;
using Emitter = ArvoCudaEndpoint::PackedEmitter;
constexpr unsigned SphereBit = 0x80000000u, InvalidPrimitiveIndex = P::Invalid;
constexpr float Pi = CUDART_PI_F, RayEpsilon = 1e-4f, MaximumRayDistance = 1e30f;
constexpr float BasisParallelThreshold = .999f, MinimumDirectionLengthSquared = 1e-12f,
                MinimumTriangleDeterminant = 1e-8f;
constexpr unsigned ImportanceBit = 1, NeeBit = 2, RouletteBit = 4, MisShift = 3, MisMask = 3;
constexpr unsigned PixelSeedMultiplier = 9781, SampleSeedMultiplier = 6271,
                   SecondDimensionScramble = 0x68bc21ebu;
constexpr unsigned CameraKey = 0x4350504du, LightKey = 0x4c50504du, PassKey = 0x50415353u;
constexpr unsigned SourceKey = 0x53524345u, PointKey = 0x504f494eu, DirectionKey = 0x44495245u;
constexpr unsigned ScatterKey = 0x53434154u, EventKey = 0x45564e54u, RouletteKey = 0x524f554cu;
constexpr unsigned VisibleKey = 0x56495349u, InteriorKey = 0x494e5445u;
constexpr unsigned LensKey = 0x4c454e53u, BounceKey = 0x9e3779b9u, RasterDimension = 0, Channels = 3;
constexpr double DensityPi = 3.1415926535897932384626433832795;
constexpr double VolumeMeasure = 4.0 / 3.0;
constexpr double NormalAgreement = .9, PlaneThickness = .1, CellCoordinateLimit = 1073741824.0;
constexpr float MinimumSurvival = .05f, MaximumSurvival = .95f;

// CUDA scene ABI; this is a transport-independent upload contract.
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
static_assert(sizeof(PackedTriangle) == 192 && sizeof(PackedMaterial) == 128 &&
              sizeof(GroupStatistics) == 32);
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
    return length > 0 ? Multiply(v, 1 / length) : Vec3{};
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
__device__ unsigned Hash(unsigned v)
{
    constexpr unsigned First = 0x7feb352du, Second = 0x846ca68bu;
    constexpr unsigned OuterShift = 16, MiddleShift = 15;
    v ^= v >> OuterShift;
    v *= First;
    v ^= v >> MiddleShift;
    v *= Second;
    v ^= v >> OuterShift;
    return v;
}
__device__ float Random01(unsigned key)
{
    constexpr unsigned MantissaMask = 0xffffffu;
    constexpr float MantissaRange = 16777216.0f;
    return float(Hash(key) & MantissaMask) / MantissaRange;
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
    const P::Controls& Control;
    unsigned* Error;
};
struct WideColor
{
    double X, Y, Z;
};
struct Cell
{
    int X, Y, Z;
};
struct Endpoint
{
    Hit Surface;
    Vec3 Position;
    unsigned Source;
    bool Found;
};
__device__ void Fail(unsigned* error, unsigned flag = P::NumericalError)
{
    atomicOr(error, flag);
}
__device__ bool Valid(WideColor v)
{
    return isfinite(v.X) && isfinite(v.Y) && isfinite(v.Z) && v.X >= 0 && v.Y >= 0 && v.Z >= 0;
}
__device__ WideColor Energy(Vec3 a, Vec3 b, double factor = 1)
{
    return {double(a.X) * b.X * factor, double(a.Y) * b.Y * factor, double(a.Z) * b.Z * factor};
}
__device__ WideColor Weighted(WideColor a, Vec3 weight)
{
    return {a.X * weight.X, a.Y * weight.Y, a.Z * weight.Z};
}
__device__ void Sum(double* target, WideColor value)
{
    target[0] += value.X;
    target[1] += value.Y;
    target[2] += value.Z;
}
__device__ float Roulette(Vec3 beta)
{
    return fminf(MaximumSurvival, fmaxf(MinimumSurvival, fmaxf(beta.X, fmaxf(beta.Y, beta.Z))));
}
__device__ bool Configuration(const FrameConstants& frame, const P::Controls& control, unsigned* error)
{
    const bool valid = frame.MaxReflectionBounces <= P::MaximumReflections && frame.TotalSamples > 0 &&
                       !frame.AdaptiveEnabled && ((frame.TransportFlags >> MisShift) & MisMask) == 0 &&
                       frame.ShadowStrength == 1 && control.PathsPerPass > 0 && control.Buckets > 0 &&
                       !(control.Buckets & (control.Buckets - 1)) && control.InitialRadius >= 0 &&
                       isfinite(control.InitialRadius) && control.WorldRadius >= 0 &&
                       isfinite(control.WorldRadius) && control.Alpha > 0 && control.Alpha < 1 &&
                       isfinite(control.Alpha);
    if (!valid)
        Fail(error, P::ConfigurationError);
    return valid;
}
__device__ bool CellAt(Vec3 position, const P::Controls& controls, Cell& cell, unsigned* error)
{
    if (!(controls.InitialRadius > 0))
        return false;
    const double x = floor((double(position.X) - controls.Center.X) / controls.InitialRadius);
    const double y = floor((double(position.Y) - controls.Center.Y) / controls.InitialRadius);
    const double z = floor((double(position.Z) - controls.Center.Z) / controls.InitialRadius);
    if (!isfinite(x) || !isfinite(y) || !isfinite(z) || fabs(x) >= CellCoordinateLimit ||
        fabs(y) >= CellCoordinateLimit || fabs(z) >= CellCoordinateLimit)
    {
        Fail(error, P::IndexError);
        return false;
    }
    cell = {int(x), int(y), int(z)};
    return true;
}
__device__ bool SameCell(Cell a, Cell b)
{
    return a.X == b.X && a.Y == b.Y && a.Z == b.Z;
}
__device__ unsigned Bucket(Cell cell, unsigned count)
{
    constexpr unsigned XMix = 0x9e3779b9u, YMix = 0x85ebca6bu, ZMix = 0xc2b2ae35u;
    return Hash(unsigned(cell.X) * XMix ^ unsigned(cell.Y) * YMix ^ unsigned(cell.Z) * ZMix) & (count - 1);
}
__device__ float2 LightSample(const P::Controls& controls, unsigned path, unsigned dimension)
{
    const unsigned seed =
        Hash(controls.Seed ^ LightKey) ^ Hash(controls.Pass ^ PassKey) ^ Hash(path) ^ dimension;
    return make_float2(Random01(seed), Random01(seed ^ SecondDimensionScramble));
}
__device__ float Decode(unsigned channel)
{
    constexpr float MaximumByte = 255, Threshold = .04045f, Slope = 12.92f, Offset = .055f, Scale = 1.055f,
                    Exponent = 2.4f;
    float v = channel / MaximumByte;
    return v <= Threshold ? v / Slope : powf((v + Offset) / Scale, Exponent);
}
__device__ Vec3 Texel(const SceneView& scene, const PackedTexture& texture, int x, int y)
{
    x = (x % int(texture.Width) + int(texture.Width)) % int(texture.Width);
    y = (y % int(texture.Height) + int(texture.Height)) % int(texture.Height);
    unsigned color = scene.Texels[texture.OffsetTexels + unsigned(y) * texture.Width + unsigned(x)];
    constexpr unsigned ByteMask = 0xff, GreenShift = 8, BlueShift = 16;
    return {Decode(color & ByteMask), Decode((color >> GreenShift) & ByteMask),
            Decode((color >> BlueShift) & ByteMask)};
}
__device__ Vec3 Color(const SceneView& scene, const Hit& hit)
{
    const unsigned index = scene.Materials[hit.MaterialIndex].DiffuseTextureIndex;
    if (!hit.HasTexCoord || index >= scene.Frame.TextureCount || !isfinite(hit.TexCoordX) ||
        !isfinite(hit.TexCoordY))
        return hit.Color;
    const PackedTexture texture = scene.Textures[index];
    if (!texture.Width || !texture.Height || texture.OffsetTexels > scene.Frame.TextureTexelCount ||
        texture.Height > (scene.Frame.TextureTexelCount - texture.OffsetTexels) / texture.Width)
        return hit.Color;
    const float u = hit.TexCoordX - floorf(hit.TexCoordX), v = 1 - hit.TexCoordY - floorf(1 - hit.TexCoordY);
    const float x = u * texture.Width - .5f, y = v * texture.Height - .5f;
    const int column = int(floorf(x)), row = int(floorf(y));
    const float fx = x - floorf(x), fy = y - floorf(y);
    const Vec3 top = Add(Multiply(Texel(scene, texture, column, row), 1 - fx),
                         Multiply(Texel(scene, texture, column + 1, row), fx));
    const Vec3 bottom = Add(Multiply(Texel(scene, texture, column, row + 1), 1 - fx),
                            Multiply(Texel(scene, texture, column + 1, row + 1), fx));
    return Multiply(hit.Color, Add(Multiply(top, 1 - fy), Multiply(bottom, fy)));
}
__device__ bool Rectangle(const Emitter& source, const Ray& ray, float& closest)
{
    if (source.Kind != ArvoCudaEndpoint::Rectangle)
        return false;
    Vec3 normal = Vector(source.Direction);
    float cosine = Dot(normal, ray.Direction);
    if (cosine == 0)
        return false;
    float distance = Dot(normal, Subtract(Vector(source.Origin), ray.Origin)) / cosine;
    if (!(distance > ray.MinimumDistance && distance < ray.MaximumDistance && distance <= closest))
        return false;
    Vec3 helper = fabsf(normal.Y) < BasisParallelThreshold ? Vec3{0, 1, 0} : Vec3{1, 0, 0};
    Vec3 right = Normalize(Cross(helper, normal)), up = Cross(normal, right);
    Vec3 relative = Subtract(Add(ray.Origin, Multiply(ray.Direction, distance)), Vector(source.Origin));
    if (fabsf(Dot(relative, right)) > source.Width / 2 || fabsf(Dot(relative, up)) > source.Height / 2)
        return false;
    closest = distance;
    return true;
}
__device__ Endpoint Trace(const SceneView& scene, const Ray& ray, GroupStatistics& statistics)
{
    Endpoint result{};
    result.Source = P::Invalid;
    result.Surface.Distance = ray.MaximumDistance;
    result.Found = IntersectScene(scene.Nodes, scene.References, scene.Triangles, scene.Spheres, scene.Frame,
                                  ray, false, result.Surface, statistics.PrimitiveTests);
    const bool proxy = result.Found && scene.Materials[result.Surface.MaterialIndex].SampledEmitter != 0;
    float closest = result.Found ? result.Surface.Distance + (proxy ? RayEpsilon : 0) : ray.MaximumDistance;
    for (unsigned source = 0; source < scene.Control.Sources; ++source)
        if (Rectangle(scene.Emitters[source], ray, closest))
            result.Source = source;
    if (result.Source != P::Invalid)
        result.Found = false;
    else
        for (unsigned source = 0; source < scene.Control.Sources; ++source)
        {
            const Emitter& emitter = scene.Emitters[source];
            if (result.Found
                    ? emitter.Primitive != P::Invalid && emitter.Primitive == result.Surface.PrimitiveIndex
                    : emitter.Kind == ArvoCudaEndpoint::Environment)
            {
                result.Source = source;
                break;
            }
        }
    result.Position =
        Add(ray.Origin, Multiply(ray.Direction, result.Found ? result.Surface.Distance : closest));
    return result;
}
__device__ Vec3 Radiance(const Emitter& source, Vec3 outgoing)
{
    if (source.Kind == ArvoCudaEndpoint::Environment)
    {
        const float blend = (1 - outgoing.Y) / 2;
        return Add(Multiply(Vector(source.BackgroundBottom), 1 - blend),
                   Multiply(Vector(source.BackgroundTop), blend));
    }
    if (source.Kind == ArvoCudaEndpoint::Rectangle)
        return Dot(Vector(source.Direction), outgoing) > 0
                   ? Multiply(Vector(source.Strength), 1 / (source.Width * source.Height))
                   : Vec3{};
    return Vector(source.Strength);
}
__device__ PhysicalSample Mirror(const PhysicalSurface& surface)
{
    PhysicalSample event{};
    event.Direction = SurfaceWorld(surface, make_float3(-surface.View.x, -surface.View.y, surface.View.z));
    if (surface.View.z <= 0 || !SurfaceHemisphere(surface, event.Direction, false))
        return {};
    const float adjoint = SurfaceAdjoint(surface, event.Direction);
    event.Weight = {adjoint, adjoint, adjoint};
    event.Pdf = event.ReversePdf = event.RelativeIndex = 1;
    event.Event = ArvoCuda::ScatteringReflection | ArvoCuda::ScatteringDelta;
    return event;
}
__device__ ArvoCudaEmission::Domain Domain(const P::Controls& controls)
{
    return {Native(controls.Center), controls.WorldRadius};
}
__device__ WideColor Direct(const SceneView& scene, const Endpoint& hit, const PhysicalSurface& surface,
                            const SampleContext& random, unsigned domain, bool filterTransmission,
                            GroupStatistics& statistics)
{
    const auto endpoint = ArvoCudaEndpoint::Emit(
        scene.Emitters, scene.Control.Sources, Domain(scene.Control), Sample2D(random, domain ^ SourceKey).x,
        Sample2D(random, domain ^ PointKey), Sample2D(random, domain ^ DirectionKey));
    const auto& light = endpoint.Emission;
    if (!(light.Flags & ArvoCudaEmission::Valid))
        return {};
    const bool infinite = (light.Flags & ArvoCudaEmission::Infinite) != 0,
               point = (light.Flags & ArvoCudaEmission::DeltaPosition) != 0;
    Vec3 direction = Multiply(Vector(light.Direction), -1);
    double geometry = 1;
    if (!infinite)
    {
        const double x = double(light.Origin.x) - hit.Position.X, y = double(light.Origin.y) - hit.Position.Y,
                     z = double(light.Origin.z) - hit.Position.Z;
        const double distance = norm3d(x, y, z);
        if (!(distance > 0) || !isfinite(distance))
            return {};
        direction = {float(x / distance), float(y / distance), float(z / distance)};
        geometry = (point ? 1 : fabs(Dot(Vector(light.Normal), direction))) / (distance * distance);
    }
    const bool transmitted = filterTransmission &&
        (hit.Surface.FrontFace ? Dot(direction, hit.Surface.GeometricNormal) < 0
                               : Dot(direction, hit.Surface.GeometricNormal) > 0);
    if (transmitted && !point && !(light.Flags & ArvoCudaEmission::DeltaDirection))
        return {};
    const double pdf =
        double(endpoint.SelectionProbability) * (infinite ? light.DirectionPdf : light.PositionPdf);
    if (!(pdf > 0))
        return {};
    const auto bsdf = EvaluatePhysical(surface, direction, (scene.Frame.TransportFlags & ImportanceBit) != 0);
    WideColor value =
        Energy(bsdf.Value, Radiance(scene.Emitters[endpoint.EmitterIndex], Multiply(direction, -1)),
               geometry * fabs(Dot(surface.Normal, direction)) / pdf);
    if (transmitted && hit.Surface.FrontFace && point)
    {
        const Vec3 segment = Subtract(Vector(light.Origin), hit.Position);
        const double attenuation = exp(-norm3d(segment.X, segment.Y, segment.Z) /
                                       surface.Material.Optical.W);
        value = Weighted(value, Vec3{float(attenuation), float(attenuation), float(attenuation)});
    }
    if (!Valid(value))
    {
        Fail(scene.Error);
        return {};
    }
    Ray shadow{PhysicalRayOrigin(surface, hit.Position, direction), direction, RayEpsilon,
               MaximumRayDistance};
    if (!infinite)
    {
        Vec3 segment = Subtract(Vector(light.Origin), shadow.Origin);
        float distance = norm3df(segment.X, segment.Y, segment.Z);
        if (!(distance > 2 * RayEpsilon))
            return {};
        shadow.Direction = Multiply(segment, 1 / distance);
        shadow.MaximumDistance = distance - RayEpsilon;
    }
    ++statistics.ShadowRays;
    Hit obstruction{};
    obstruction.Distance = shadow.MaximumDistance;
    if (IntersectScene(scene.Nodes, scene.References, scene.Triangles, scene.Spheres, scene.Frame, shadow,
                       true, obstruction, statistics.PrimitiveTests))
        return {};
    float closest = shadow.MaximumDistance;
    for (unsigned source = 0; source < scene.Control.Sources; ++source)
        if (Rectangle(scene.Emitters[source], shadow, closest))
            return {};
    return value;
}
__device__ bool SelectVisible(P::VisiblePoint& point, unsigned& candidates, float variate)
{
    ++candidates;
    if (point.Active)
        point.Throughput = Multiply(point.Throughput, float(candidates) / float(candidates - 1));
    return candidates == 1 || variate < 1.0f / float(candidates);
}
__device__ void TraceCamera(const SceneView& scene, Ray ray, const SampleContext& random,
                            P::VisiblePoint& point, P::Pixel& state, GroupStatistics& statistics)
{
    Vec3 beta{1, 1, 1};
    float etaScale = 1;
    unsigned interiorMaterial = P::Invalid, candidates = 0;
    for (unsigned depth = 0; depth <= scene.Frame.MaxReflectionBounces + 1; ++depth)
    {
        if (!Valid(beta) || !isfinite(etaScale))
        {
            Fail(scene.Error);
            return;
        }
        if (!Positive(beta))
            return;
        if (depth == 0)
            ++statistics.PrimaryRays;
        else
            ++statistics.ReflectionRays;
        const Endpoint endpoint = Trace(scene, ray, statistics);
        const unsigned domain = CameraKey ^ (depth * BounceKey);
        if (endpoint.Found && endpoint.Surface.MaterialIndex >= scene.Frame.MaterialCount)
        {
            Fail(scene.Error, P::ConfigurationError);
            return;
        }
        if (depth == 0 && endpoint.Found && !endpoint.Surface.FrontFace &&
            scene.Materials[endpoint.Surface.MaterialIndex].Optical.W > 0)
            interiorMaterial = endpoint.Surface.MaterialIndex;
        if (interiorMaterial != P::Invalid)
        {
            const PackedMaterial interior = scene.Materials[interiorMaterial];
            const bool finiteSource = endpoint.Source != P::Invalid &&
                scene.Emitters[endpoint.Source].Kind == ArvoCudaEndpoint::Rectangle;
            const float boundary = endpoint.Found || finiteSource
                ? norm3df(endpoint.Position.X - ray.Origin.X, endpoint.Position.Y - ray.Origin.Y,
                          endpoint.Position.Z - ray.Origin.Z) : ray.MaximumDistance;
            const auto flight = SubsurfaceFreeFlight(interior.Optical.W,
                make_float3(interior.AlbedoReflectivity.X * interior.ConductorExtinction.W,
                            interior.AlbedoReflectivity.Y * interior.ConductorExtinction.W,
                            interior.AlbedoReflectivity.Z * interior.ConductorExtinction.W),
                Native(ray.Direction), boundary, Sample2D(random, domain ^ InteriorKey).x);
            beta = Multiply(beta, Vector(flight.Weight));
            if (!Valid(beta))
            {
                Fail(scene.Error);
                return;
            }
            if (!Positive(beta)) return;
            if (flight.Scattered)
            {
                if (depth > scene.Frame.MaxReflectionBounces) return;
                if (SelectVisible(point, candidates, Sample2D(random, domain ^ VisibleKey).x))
                {
                    point.Position = Add(ray.Origin, Multiply(ray.Direction, flight.Distance));
                    point.View = ray.Direction;
                    point.Depth = depth;
                    point.Material = interiorMaterial;
                    point.Throughput = Multiply(beta, float(candidates));
                    point.Active = P::VisibleVolume;
                }
                return;
            }
        }
        if (endpoint.Source != P::Invalid)
            Sum(state.Direct,
                Energy(beta, Radiance(scene.Emitters[endpoint.Source], Multiply(ray.Direction, -1))));
        if (!endpoint.Found || depth > scene.Frame.MaxReflectionBounces)
            return;
        PackedMaterial material = scene.Materials[endpoint.Surface.MaterialIndex];
        const bool legacy = material.Model == LegacyMaterial;
        const float mirror = legacy ? fminf(1, fmaxf(0, material.AlbedoReflectivity.W)) : 0;
        if (legacy)
            material.Model = DiffuseMaterial;
        const Vec3 view = Multiply(ray.Direction, -1), color = Color(scene, endpoint.Surface);
        const PhysicalSurface surface = MakePhysicalSurface(material, endpoint.Surface, view, color);
        const bool subsurfaceTransmission = material.Optical.W > 0 && !PhysicalDelta(surface);
        if (!PhysicalDelta(surface) && mirror < 1 && (scene.Frame.TransportFlags & NeeBit))
            Sum(state.Direct, Weighted(Direct(scene, endpoint, surface, random, domain,
                                               subsurfaceTransmission, statistics),
                                       Multiply(beta, 1 - mirror)));
        const bool chooseMirror = legacy && Sample2D(random, domain ^ EventKey).x < mirror;
        if (!chooseMirror && !PhysicalDelta(surface))
        {
            if (SelectVisible(point, candidates, Sample2D(random, domain ^ VisibleKey).x))
            {
                point.Position = endpoint.Position;
                point.Geometry = endpoint.Surface.GeometricNormal;
                point.Normal = endpoint.Surface.Normal;
                point.FrontFace = endpoint.Surface.FrontFace;
                point.Depth = depth;
                point.Material = endpoint.Surface.MaterialIndex;
                point.Color = color;
                point.View = view;
                point.Throughput = Multiply(beta, float(candidates));
                point.Active = P::VisibleSurface;
            }
            if (!subsurfaceTransmission) return;
        }
        const auto event = chooseMirror ? Mirror(surface)
                                        : SamplePhysical(surface, Sample2D(random, domain ^ ScatterKey),
                                                         Sample2D(random, domain ^ EventKey).x,
                                                         (scene.Frame.TransportFlags & ImportanceBit) != 0);
        if (!event.Event || !(event.Pdf > 0))
            return;
        if (subsurfaceTransmission && !(event.Event & ArvoCuda::ScatteringTransmission))
            return;
        beta = Multiply(beta, event.Weight);
        etaScale *= event.RelativeIndex * event.RelativeIndex;
        if (!Valid(beta) || !isfinite(etaScale))
        {
            Fail(scene.Error);
            return;
        }
        if ((scene.Frame.TransportFlags & RouletteBit) && depth + 1 >= scene.Frame.RouletteStartDepth)
        {
            float survival = Roulette(Multiply(beta, etaScale));
            if (Sample2D(random, domain ^ RouletteKey).x >= survival)
                return;
            beta = Multiply(beta, 1 / survival);
        }
        ray = {PhysicalRayOrigin(surface, endpoint.Position, event.Direction), event.Direction, RayEpsilon,
               MaximumRayDistance};
        if (material.Optical.W > 0 && (event.Event & ArvoCuda::ScatteringTransmission) != 0)
            interiorMaterial = endpoint.Surface.FrontFace ? endpoint.Surface.MaterialIndex : P::Invalid;
    }
}
__device__ bool Accepted(unsigned lightDepth, bool deltaPrefix, bool volume,
                         unsigned cameraDepth, const SceneView& scene)
{
    if (lightDepth + cameraDepth > scene.Frame.MaxReflectionBounces)
        return false;
    if (lightDepth == 0)
        return volume || (scene.Frame.TransportFlags & NeeBit) == 0;
    return (scene.Control.Classes & (deltaPrefix ? P::Caustics : P::Indirect)) != 0;
}
__device__ void Deposit(const SceneView& scene, const Endpoint& light, Vec3 incoming, Vec3 flux,
                        unsigned depth, bool deltaPrefix, bool volume,
                        const P::VisiblePoint* points, P::Pixel* states,
                        const unsigned* heads, GroupStatistics& statistics)
{
    Cell center;
    if (!CellAt(light.Position, scene.Control, center, scene.Error))
        return;
    const unsigned pixels = scene.Frame.Width * scene.Frame.Height;
    for (int z = -1; z <= 1; ++z)
        for (int y = -1; y <= 1; ++y)
            for (int x = -1; x <= 1; ++x)
            {
                const Cell neighbor{center.X + x, center.Y + y, center.Z + z};
                unsigned link = heads[Bucket(neighbor, scene.Control.Buckets)], visited = 0;
                while (link != P::Invalid)
                {
                    if (link >= pixels || ++visited > pixels)
                    {
                        Fail(scene.Error, P::IndexError);
                        return;
                    }
                    const unsigned pixel = link;
                    const P::VisiblePoint point = points[pixel];
                    link = point.Next;
                    if (point.Active != (volume ? P::VisibleVolume : P::VisibleSurface) ||
                        !Accepted(depth, deltaPrefix, volume, point.Depth, scene))
                        continue;
                    Cell cell;
                    if (!CellAt(point.Position, scene.Control, cell, scene.Error) ||
                        !SameCell(cell, neighbor))
                        continue;
                    const double dx = double(light.Position.X) - point.Position.X,
                                 dy = double(light.Position.Y) - point.Position.Y,
                                 dz = double(light.Position.Z) - point.Position.Z;
                    const double squaredDistance = dx * dx + dy * dy + dz * dz,
                                 radiusSquared = volume ? states[pixel].VolumeRadiusSquared
                                                        : states[pixel].RadiusSquared;
                    if (squaredDistance > radiusSquared)
                        continue;
                    if (volume)
                    {
                        if (light.Surface.MaterialIndex != point.Material) continue;
                        const Vec3 separation = Subtract(light.Position, point.Position);
                        const float distance = norm3df(separation.X, separation.Y, separation.Z);
                        if (distance > 2 * RayEpsilon)
                        {
                            ++statistics.ShadowRays;
                            const Ray connection{point.Position, Multiply(separation, 1 / distance),
                                                 RayEpsilon, distance - RayEpsilon};
                            Hit obstruction{};
                            obstruction.Distance = connection.MaximumDistance;
                            if (IntersectScene(scene.Nodes, scene.References, scene.Triangles, scene.Spheres,
                                               scene.Frame, connection, true, obstruction,
                                               statistics.PrimitiveTests))
                                continue;
                        }
                        const PackedMaterial material = scene.Materials[point.Material];
                        const double phase = SubsurfacePhase(Dot(incoming, point.View),
                                                              material.ConductorIndex.W);
                        const WideColor value = Energy(flux, point.Throughput,
                                                        phase * material.Optical.W);
                        if (!Valid(value))
                        {
                            Fail(scene.Error);
                            return;
                        }
                        atomicAdd(states[pixel].VolumeIncoming, value.X);
                        atomicAdd(states[pixel].VolumeIncoming + 1, value.Y);
                        atomicAdd(states[pixel].VolumeIncoming + 2, value.Z);
                        atomicAdd(&states[pixel].VolumeArrivals, 1u);
                        continue;
                    }
                    if (Dot(light.Surface.GeometricNormal, point.Geometry) < NormalAgreement)
                        continue;
                    if (scene.Materials[point.Material].Optical.W > 0 &&
                        (point.FrontFace ? Dot(incoming, point.Geometry) <= 0
                                         : Dot(incoming, point.Geometry) >= 0))
                        continue;
                    const double plane =
                        dx * point.Geometry.X + dy * point.Geometry.Y + dz * point.Geometry.Z;
                    if (plane * plane > PlaneThickness * PlaneThickness * radiusSquared)
                        continue;
                    const double projected = fabs(Dot(point.Geometry, incoming));
                    if (!(projected > 0))
                        continue;
                    PackedMaterial material = scene.Materials[point.Material];
                    if (material.Model == LegacyMaterial)
                        material.Model = DiffuseMaterial;
                    Hit hit{};
                    hit.Normal = point.Normal;
                    hit.GeometricNormal = point.Geometry;
                    hit.FrontFace = point.FrontFace;
                    const auto surface = MakePhysicalSurface(material, hit, point.View, point.Color);
                    const auto scattering = EvaluatePhysical(
                        surface, incoming, (scene.Frame.TransportFlags & ImportanceBit) != 0);
                    const WideColor value = Weighted(
                        Energy(flux, scattering.Value, fabs(Dot(point.Normal, incoming)) / projected),
                        point.Throughput);
                    if (!Valid(value))
                    {
                        Fail(scene.Error);
                        return;
                    }
                    // SM 7.5 is the project's compilation baseline; native binary64 atomics
                    // avoid float overflow and reduce order-dependent loss of small flux.
                    atomicAdd(states[pixel].Incoming, value.X);
                    atomicAdd(states[pixel].Incoming + 1, value.Y);
                    atomicAdd(states[pixel].Incoming + 2, value.Z);
                    atomicAdd(&states[pixel].Arrivals, 1u);
                }
            }
}
__device__ void TracePhoton(const SceneView& scene, unsigned path, const P::VisiblePoint* points,
                            P::Pixel* states, const unsigned* heads, GroupStatistics& statistics)
{
    const auto endpoint = ArvoCudaEndpoint::Emit(scene.Emitters, scene.Control.Sources, Domain(scene.Control),
                                                 LightSample(scene.Control, path, SourceKey).x,
                                                 LightSample(scene.Control, path, PointKey),
                                                 LightSample(scene.Control, path, DirectionKey));
    const auto& emission = endpoint.Emission;
    if (!(emission.Flags & ArvoCudaEmission::Valid))
        return;
    Vec3 origin = Vector(emission.Origin), direction = Vector(emission.Direction),
         normal = Vector(emission.Normal);
    origin = Add(origin, Multiply(normal, Dot(normal, direction) >= 0 ? RayEpsilon : -RayEpsilon));
    Ray ray{origin, direction, RayEpsilon, MaximumRayDistance};
    Vec3 flux = Vector(emission.Weight);
    bool deltaPrefix = true;
    unsigned interiorMaterial = P::Invalid;
    for (unsigned depth = 0; depth <= scene.Frame.MaxReflectionBounces; ++depth)
    {
        if (!Valid(flux))
        {
            Fail(scene.Error);
            return;
        }
        if (!Positive(flux))
            return;
        ++statistics.ReflectionRays;
        const Endpoint vertex = Trace(scene, ray, statistics);
        const unsigned domain = LightKey ^ (depth * BounceKey);
        if (vertex.Found && vertex.Surface.MaterialIndex >= scene.Frame.MaterialCount)
        {
            Fail(scene.Error, P::ConfigurationError);
            return;
        }
        if (depth == 0 && vertex.Found && !vertex.Surface.FrontFace &&
            scene.Materials[vertex.Surface.MaterialIndex].Optical.W > 0)
            interiorMaterial = vertex.Surface.MaterialIndex;
        if (interiorMaterial != P::Invalid)
        {
            const PackedMaterial interior = scene.Materials[interiorMaterial];
            const bool finiteSource = vertex.Source != P::Invalid &&
                scene.Emitters[vertex.Source].Kind == ArvoCudaEndpoint::Rectangle;
            const float boundary = vertex.Found || finiteSource
                ? norm3df(vertex.Position.X - ray.Origin.X, vertex.Position.Y - ray.Origin.Y,
                          vertex.Position.Z - ray.Origin.Z) : ray.MaximumDistance;
            const auto flight = SubsurfaceFreeFlight(interior.Optical.W,
                make_float3(interior.AlbedoReflectivity.X * interior.ConductorExtinction.W,
                            interior.AlbedoReflectivity.Y * interior.ConductorExtinction.W,
                            interior.AlbedoReflectivity.Z * interior.ConductorExtinction.W),
                Native(ray.Direction), boundary, LightSample(scene.Control, path, domain ^ InteriorKey).x);
            const Vec3 collisionPosition = Add(ray.Origin, Multiply(ray.Direction, flight.Distance));
            if (flight.Scattered)
            {
                Endpoint collision{};
                collision.Position = collisionPosition;
                collision.Surface.MaterialIndex = interiorMaterial;
                Deposit(scene, collision, Multiply(ray.Direction, -1), flux, depth, deltaPrefix,
                        true, points, states, heads, statistics);
            }
            flux = Multiply(flux, Vector(flight.Weight));
            if (!Valid(flux))
            {
                Fail(scene.Error);
                return;
            }
            if (!Positive(flux)) return;
            if (flight.Scattered)
            {
                if (depth == scene.Frame.MaxReflectionBounces) return;
                const auto phase = SubsurfaceSampleDirection(Native(ray.Direction), interior.ConductorIndex.W,
                    LightSample(scene.Control, path, domain ^ ScatterKey),
                    (scene.Frame.TransportFlags & ImportanceBit) != 0);
                if (!(phase.Pdf > 0) || !isfinite(phase.Weight))
                {
                    Fail(scene.Error);
                    return;
                }
                flux = Multiply(flux, phase.Weight);
                if ((scene.Frame.TransportFlags & RouletteBit) && depth + 1 >= scene.Frame.RouletteStartDepth)
                {
                    const float survival = Roulette(flux);
                    if (LightSample(scene.Control, path, domain ^ RouletteKey).x >= survival) return;
                    flux = Multiply(flux, 1 / survival);
                }
                deltaPrefix = false;
                ray = {collisionPosition, Vector(phase.Direction), RayEpsilon, MaximumRayDistance};
                continue;
            }
        }
        if (!vertex.Found)
            return;
        PackedMaterial material = scene.Materials[vertex.Surface.MaterialIndex];
        const bool legacy = material.Model == LegacyMaterial;
        const float mirror = legacy ? fminf(1, fmaxf(0, material.AlbedoReflectivity.W)) : 0;
        if (legacy)
            material.Model = DiffuseMaterial;
        const auto surface = MakePhysicalSurface(material, vertex.Surface, Multiply(ray.Direction, -1),
                                                 Color(scene, vertex.Surface), ArvoCuda::TransportImportance);
        if (!PhysicalDelta(surface) && mirror < 1)
            Deposit(scene, vertex, Multiply(ray.Direction, -1), flux, depth, deltaPrefix, false,
                    points, states, heads, statistics);
        if (depth == scene.Frame.MaxReflectionBounces)
            return;
        const float choice = LightSample(scene.Control, path, domain ^ EventKey).x;
        const auto event =
            legacy && choice < mirror
                ? Mirror(surface)
                : SamplePhysical(surface, LightSample(scene.Control, path, domain ^ ScatterKey), choice,
                                 (scene.Frame.TransportFlags & ImportanceBit) != 0);
        if (!event.Event || !(event.Pdf > 0))
            return;
        flux = Multiply(flux, event.Weight);
        deltaPrefix = deltaPrefix && (event.Event & ArvoCuda::ScatteringDelta) != 0;
        if (!Valid(flux))
        {
            Fail(scene.Error);
            return;
        }
        if ((scene.Frame.TransportFlags & RouletteBit) && depth + 1 >= scene.Frame.RouletteStartDepth)
        {
            const float survival = Roulette(flux);
            if (LightSample(scene.Control, path, domain ^ RouletteKey).x >= survival)
                return;
            flux = Multiply(flux, 1 / survival);
        }
        ray = {PhysicalRayOrigin(surface, vertex.Position, event.Direction), event.Direction, RayEpsilon,
               MaximumRayDistance};
        if (material.Optical.W > 0 && (event.Event & ArvoCuda::ScatteringTransmission) != 0)
            interiorMaterial = vertex.Surface.FrontFace ? vertex.Surface.MaterialIndex : P::Invalid;
    }
}
__device__ void Advance(P::Pixel& pixel, double alpha)
{
    if (pixel.Arrivals)
    {
        const double next = pixel.EffectiveCount + alpha * pixel.Arrivals;
        const double ratio = next / (pixel.EffectiveCount + pixel.Arrivals);
        pixel.EffectiveCount = next;
        pixel.RadiusSquared *= ratio;
        for (unsigned channel = 0; channel < Channels; ++channel)
            pixel.Flux[channel] = (pixel.Flux[channel] + pixel.Incoming[channel]) * ratio;
    }
    if (pixel.VolumeArrivals)
    {
        const double next = pixel.VolumeEffectiveCount + alpha * pixel.VolumeArrivals;
        const double ratio = next / (pixel.VolumeEffectiveCount + pixel.VolumeArrivals);
        pixel.VolumeEffectiveCount = next;
        pixel.VolumeRadiusSquared *= pow(ratio, 2.0 / 3.0);
        for (unsigned channel = 0; channel < Channels; ++channel)
            pixel.VolumeFlux[channel] = (pixel.VolumeFlux[channel] + pixel.VolumeIncoming[channel]) * ratio;
    }
}
} // namespace
extern "C" __global__ void PpmInitializeKernel(const FrameConstants* frame, const P::Controls* controls,
                                               P::Pixel* states, P::VisiblePoint* points, Vec4* film,
                                               unsigned* counts, unsigned* targets, unsigned* error)
{
    const unsigned lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= controls->Count)
        return;
    if (!Configuration(*frame, *controls, error))
        return;
    const unsigned pixel = controls->Begin + lane;
    if (pixel >= frame->Width * frame->Height)
        return;
    P::Pixel state{};
    state.RadiusSquared = controls->InitialRadius * controls->InitialRadius;
    state.VolumeRadiusSquared = state.RadiusSquared;
    states[pixel] = state;
    points[pixel] = {};
    film[pixel] = {};
    counts[pixel] = 0;
    targets[pixel] = frame->TotalSamples;
}
extern "C" __global__ void PpmCameraKernel(const PackedAccelerationNode* nodes,
                                           const PackedPrimitiveRef* refs, const PackedTriangle* triangles,
                                           const PackedSphere* spheres, const PackedMaterial* materials,
                                           const PackedTexture* textures, const unsigned* texels,
                                           const Emitter* emitters, const FrameConstants* frame,
                                           const P::Controls* controls, P::VisiblePoint* points,
                                           P::Pixel* states, GroupStatistics* statistics, unsigned* error)
{
    const unsigned lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= controls->Count)
        return;
    statistics[lane] = {};
    if (!Configuration(*frame, *controls, error))
        return;
    const unsigned pixel = controls->Begin + lane;
    if (pixel >= frame->Width * frame->Height)
        return;
    auto& point = points[pixel];
    point = {};
    point.Next = P::Invalid;
    auto& state = states[pixel];
    state.Incoming[0] = state.Incoming[1] = state.Incoming[2] = 0;
    state.Arrivals = 0;
    state.VolumeIncoming[0] = state.VolumeIncoming[1] = state.VolumeIncoming[2] = 0;
    state.VolumeArrivals = 0;
    const unsigned x = pixel % frame->Width + frame->TileX, y = pixel / frame->Width + frame->TileY;
    const SampleContext random = MakeSampleContext(x, y, frame->FullWidth, controls->Pass,
                                                   frame->TotalSamples, frame->SamplingSequence, frame->SamplingSeed);
    const float2 jitter = Sample2D(random, RasterDimension);
    const ArvoCudaCamera::Frame camera{Native(frame->CameraPosition),
                                       Native(frame->CameraForward),
                                       Native(frame->CameraRight),
                                       Native(frame->CameraUp),
                                       frame->HalfWidth,
                                       frame->HalfHeight,
                                       frame->LensRadius,
                                       frame->FocusDistance};
    const auto start = ArvoCudaCamera::Emit(
        camera, ArvoCudaCamera::PixelRaster(x, y, frame->FullWidth, frame->FullHeight, jitter),
        Sample2D(random, LensKey));
    SceneView scene{nodes,  refs,     triangles, spheres,   materials, textures,
                    texels, emitters, *frame,    *controls, error};
    GroupStatistics counters{};
    if (start.Flags & ArvoCudaCamera::Valid)
        TraceCamera(scene, {Vector(start.Origin), Vector(start.Direction), RayEpsilon, MaximumRayDistance},
                    random, point, state, counters);
    else
        Fail(error);
    statistics[lane] = counters;
}
extern "C" __global__ void PpmIndexKernel(const FrameConstants* frame, const P::Controls* controls,
                                          P::VisiblePoint* points, const P::Pixel* states, unsigned* heads,
                                          unsigned* error)
{
    const unsigned lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= controls->Count)
        return;
    const unsigned pixel = controls->Begin + lane;
    if (pixel >= frame->Width * frame->Height || !points[pixel].Active ||
        !(points[pixel].Active == P::VisibleVolume ? states[pixel].VolumeRadiusSquared > 0
                                    : states[pixel].RadiusSquared > 0))
        return;
    Cell cell;
    if (!CellAt(points[pixel].Position, *controls, cell, error))
        return;
    points[pixel].Next = atomicExch(heads + Bucket(cell, controls->Buckets), pixel);
}
extern "C" __global__ void PpmPhotonKernel(const PackedAccelerationNode* nodes,
                                           const PackedPrimitiveRef* refs, const PackedTriangle* triangles,
                                           const PackedSphere* spheres, const PackedMaterial* materials,
                                           const PackedTexture* textures, const unsigned* texels,
                                           const Emitter* emitters, const FrameConstants* frame,
                                           const P::Controls* controls, const P::VisiblePoint* points,
                                           P::Pixel* states, const unsigned* heads,
                                           GroupStatistics* statistics, unsigned* error)
{
    const unsigned lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= controls->Count)
        return;
    statistics[lane] = {};
    if (!Configuration(*frame, *controls, error))
        return;
    const unsigned path = controls->Begin + lane;
    if (path >= controls->PathsPerPass)
        return;
    SceneView scene{nodes,  refs,     triangles, spheres,   materials, textures,
                    texels, emitters, *frame,    *controls, error};
    GroupStatistics counters{};
    TracePhoton(scene, path, points, states, heads, counters);
    statistics[lane] = counters;
}
extern "C" __global__ void PpmUpdateKernel(const FrameConstants* frame, const P::Controls* controls,
                                           P::Pixel* states, Vec4* film, unsigned* counts, unsigned* error)
{
    const unsigned lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= controls->Count)
        return;
    const unsigned pixel = controls->Begin + lane;
    if (pixel >= frame->Width * frame->Height)
        return;
    auto& state = states[pixel];
    Advance(state, controls->Alpha);
    const double passes = double(controls->Pass) + 1, emitted = passes * controls->PathsPerPass;
    double value[Channels]{};
    for (unsigned channel = 0; channel < Channels; ++channel)
    {
        value[channel] =
            state.Direct[channel] / passes +
            (state.RadiusSquared > 0 ? state.Flux[channel] / (DensityPi * state.RadiusSquared * emitted) : 0) +
            (state.VolumeRadiusSquared > 0
                ? state.VolumeFlux[channel] /
                    (VolumeMeasure * DensityPi * state.VolumeRadiusSquared *
                     sqrt(state.VolumeRadiusSquared) * emitted) : 0);
        if (!isfinite(value[channel]) || value[channel] < 0 || !isfinite(state.Flux[channel]) ||
            !isfinite(state.VolumeFlux[channel]))
        {
            Fail(error);
            return;
        }
    }
    Vec3 radiance{float(value[0]), float(value[1]), float(value[2])};
    if (!Valid(radiance) || !isfinite(state.RadiusSquared) || !isfinite(state.EffectiveCount) ||
        !isfinite(state.VolumeRadiusSquared) || !isfinite(state.VolumeEffectiveCount))
    {
        Fail(error);
        return;
    }
    // This is the final progressive estimate, not an average of pass estimates.
    // No independent-sample M2 is available for persistent density state.
    film[pixel] = {radiance.X, radiance.Y, radiance.Z, 0};
    counts[pixel] = controls->Pass + 1;
}
