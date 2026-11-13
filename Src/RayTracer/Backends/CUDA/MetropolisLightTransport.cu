// Independent CUDA primary-sample-space MLT: this file owns its camera-path
// estimator, bootstrap distribution, persistent chains and two-contribution splat.
// Kelemen et al. (2002), "A Simple and Robust Mutation Strategy for the Metropolis
// Light Transport Algorithm". Symmetric wrapped Gaussian / independent proposals
// cancel their transition densities. Acceptance is min(1, I(proposal)/I(current)).
// https://pbr-book.org/3ed-2018/Light_Transport_III_Bidirectional_Methods/Metropolis_Light_Transport
// Each complete pass has FullWidth*FullHeight mutations; raster-bin area cancels
// that count, leaving bootstrap mean importance as the splat scale. Film moments
// describe correlated passes and cannot justify independent-sample adaptive stops.
// No path-tracing or BDPT integrator is included or invoked.
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cfloat>
#include "MetropolisLayout.h"
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
namespace Mlt = Arvo::RayTracing::CudaMlt;
using Vec3 = Mlt::Vector;
using Emitter = ArvoCudaEndpoint::PackedEmitter;
constexpr unsigned SphereBit = 0x80000000u, InvalidPrimitiveIndex = 0xffffffffu;
constexpr float Pi = CUDART_PI_F, RayEpsilon = 1.0e-4f, MaximumRayDistance = 1.0e30f;
constexpr float BasisParallelThreshold = 0.999f, MinimumDirectionLengthSquared = 1.0e-12f;
constexpr float MinimumTriangleDeterminant = 1.0e-8f;
constexpr unsigned ImportanceBit = 1, NeeBit = 2, RouletteBit = 4, MisShift = 3, MisMask = 3;
constexpr unsigned LightChoice = 0, LightPosition = 1, LightDirection = 3;
constexpr unsigned ScatterDirection = 5, ScatterLobe = 7, RouletteChoice = 8;
constexpr unsigned InteriorFreeFlight = 9, InteriorPhaseDirection = 10;
constexpr double ImportanceRed = 0.2126, ImportanceGreen = 0.7152, ImportanceBlue = 0.0722;

// These records match CUDA scene uploads. They are data contracts, not a
// translated implementation of another transport algorithm.
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
__device__ Vec3 Multiply(Vec3 a, float s)
{
    return {a.X * s, a.Y * s, a.Z * s};
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
__device__ Vec3 Reflect(Vec3 direction, Vec3 normal)
{
    return Subtract(direction, Multiply(normal, 2 * Dot(direction, normal)));
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
#include "Triangle.cu"
#include "Sphere.cu"
#include "Bvh.cu"
#include "KdTree.cu"
#include "Bih.cu"
#include "Dispatch.cu"
#include "SurfaceScattering.cu"
#include "FilmMoments.cu"

// SplitMix64 supplies primary samples, not per-pixel sampler state. Domain keys
// make bootstrap replay and chain scheduling independent of each other.
constexpr unsigned long long RandomIncrement = 0x9e3779b97f4a7c15ull;
constexpr unsigned long long RandomMultiplierA = 0xbf58476d1ce4e5b9ull;
constexpr unsigned long long RandomMultiplierB = 0x94d049bb133111ebull;
constexpr unsigned long long BootstrapDomain = 0x43554441424f4f54ull;
constexpr unsigned long long ChainDomain = 0x4355444143484149ull;
constexpr unsigned RandomShiftA = 30, RandomShiftB = 27, RandomShiftC = 31;
constexpr unsigned WordBits = 64, MantissaBits = 53;
constexpr double UniformScale = 1.0 / double(1ull << MantissaBits);
__device__ unsigned long long MixWord(unsigned long long word)
{
    word = (word ^ (word >> RandomShiftA)) * RandomMultiplierA;
    word = (word ^ (word >> RandomShiftB)) * RandomMultiplierB;
    return word ^ (word >> RandomShiftC);
}
__device__ double Uniform(unsigned long long& state)
{
    state += RandomIncrement;
    return double(MixWord(state) >> (WordBits - MantissaBits)) * UniformScale;
}
__device__ unsigned long long BootstrapSeed(const Mlt::Controls& controls, unsigned index)
{
    return MixWord(BootstrapDomain ^ controls.Seed ^ (RandomIncrement * (index + 1ull)));
}
__device__ void IndependentPrimary(Mlt::PrimaryState& primary, unsigned long long state)
{
    for (unsigned dimension = 0; dimension < Mlt::PrimaryDimensions; ++dimension)
        primary.Values[dimension] = Uniform(state);
}
__device__ void MutatePrimary(const Mlt::PrimaryState& current, Mlt::PrimaryState& proposal,
                              unsigned long long& state, bool large, double scale)
{
    constexpr double TwoPi = 6.283185307179586476925286766559;
    for (unsigned dimension = 0; dimension < Mlt::PrimaryDimensions; ++dimension)
    {
        if (large)
            proposal.Values[dimension] = Uniform(state);
        else
        {
            double radius = sqrt(-2 * log(fmax(Uniform(state), UniformScale)));
            double offset = scale * radius * cos(TwoPi * Uniform(state));
            double value = current.Values[dimension] + offset;
            proposal.Values[dimension] = value - floor(value);
        }
    }
}
__device__ float Variate(const Mlt::PrimaryState& primary, unsigned dimension)
{
    return fminf(float(primary.Values[dimension]), nextafterf(1, 0));
}
__device__ float2 Pair(const Mlt::PrimaryState& primary, unsigned dimension)
{
    return make_float2(Variate(primary, dimension), Variate(primary, dimension + 1));
}
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
    const Mlt::Controls& Controls;
    unsigned* Error;
};
__device__ Vec3 Offset(Vec3 position, Vec3 geometry, Vec3 direction)
{
    return Add(position, Multiply(geometry, Dot(geometry, direction) >= 0 ? RayEpsilon : -RayEpsilon));
}
__device__ float LinearByte(unsigned value)
{
    constexpr float ByteMaximum = 255, SrgbThreshold = 0.04045f, SrgbLinearScale = 12.92f;
    constexpr float SrgbOffset = 0.055f, SrgbScale = 1.055f, SrgbExponent = 2.4f;
    float encoded = value / ByteMaximum;
    return encoded <= SrgbThreshold ? encoded / SrgbLinearScale
                                    : powf((encoded + SrgbOffset) / SrgbScale, SrgbExponent);
}
__device__ Vec3 ReadTexel(const SceneView& scene, const PackedTexture& image, long long x, long long y)
{
    x %= image.Width;
    y %= image.Height;
    if (x < 0)
        x += image.Width;
    if (y < 0)
        y += image.Height;
    unsigned packed = scene.Texels[image.OffsetTexels + y * image.Width + x];
    constexpr unsigned ByteMask = 0xff, GreenShift = 8, BlueShift = 16;
    return {LinearByte(packed & ByteMask), LinearByte((packed >> GreenShift) & ByteMask),
            LinearByte((packed >> BlueShift) & ByteMask)};
}
__device__ Vec3 TexturedColor(const SceneView& scene, const Hit& hit)
{
    unsigned index = scene.Materials[hit.MaterialIndex].DiffuseTextureIndex;
    if (!hit.HasTexCoord || index >= scene.Frame.TextureCount || !isfinite(hit.TexCoordX) ||
        !isfinite(hit.TexCoordY))
        return hit.Color;
    const auto& image = scene.Textures[index];
    unsigned long long count = static_cast<unsigned long long>(image.Width) * image.Height;
    if (!count || image.OffsetTexels >= scene.Frame.TextureTexelCount ||
        count > scene.Frame.TextureTexelCount - image.OffsetTexels)
        return hit.Color;
    float u = hit.TexCoordX - floorf(hit.TexCoordX), flippedV = 1 - hit.TexCoordY;
    float x = u * image.Width - 0.5f, y = (flippedV - floorf(flippedV)) * image.Height - 0.5f;
    long long column = static_cast<long long>(floorf(x)), row = static_cast<long long>(floorf(y));
    float horizontal = x - floorf(x), vertical = y - floorf(y);
    Vec3 rows[2];
    for (unsigned offset = 0; offset < 2; ++offset)
        rows[offset] = Add(Multiply(ReadTexel(scene, image, column, row + offset), 1 - horizontal),
                           Multiply(ReadTexel(scene, image, column + 1, row + offset), horizontal));
    return Multiply(hit.Color, Add(Multiply(rows[0], 1 - vertical), Multiply(rows[1], vertical)));
}

__device__ ArvoCudaEmission::Domain EmissionDomain(const SceneView& scene)
{
    return {Native(scene.Controls.WorldCenter), scene.Controls.WorldRadius};
}
__device__ Vec3 SourceRadiance(const Emitter& source, Vec3 launch)
{
    if (source.Kind == ArvoCudaEndpoint::Environment)
    {
        float top = (1 - launch.Y) / 2;
        return Add(Multiply(Vector(source.BackgroundTop), top),
                   Multiply(Vector(source.BackgroundBottom), 1 - top));
    }
    if (source.Kind == ArvoCudaEndpoint::Rectangle)
        return Dot(Vector(source.Direction), launch) > 0
                   ? Multiply(Vector(source.Strength), 1 / (source.Width * source.Height))
                   : Vec3{};
    return Vector(source.Strength);
}
__device__ unsigned SourceIndex(const SceneView& scene, unsigned primitive, bool sky)
{
    for (unsigned index = 0; index < scene.Controls.EmitterCount; ++index)
    {
        const auto& source = scene.Emitters[index];
        if (sky ? source.Kind == ArvoCudaEndpoint::Environment
                : primitive != InvalidPrimitiveIndex && source.Primitive == primitive)
            return index;
    }
    return InvalidPrimitiveIndex;
}
__device__ bool HitRectangle(const Emitter& source, const Ray& ray, float& nearest)
{
    if (source.Kind != ArvoCudaEndpoint::Rectangle)
        return false;
    Vec3 normal = Vector(source.Direction);
    float denominator = Dot(normal, ray.Direction);
    if (denominator == 0)
        return false;
    float distance = Dot(normal, Subtract(Vector(source.Origin), ray.Origin)) / denominator;
    if (!(distance > ray.MinimumDistance && distance < ray.MaximumDistance && distance <= nearest))
        return false;
    float3 tangent, bitangent;
    ArvoCudaEmission::Basis(source.Direction, tangent, bitangent);
    Vec3 relative = Subtract(Add(ray.Origin, Multiply(ray.Direction, distance)), Vector(source.Origin));
    if (fabsf(Dot(relative, Vector(tangent))) > source.Width / 2 ||
        fabsf(Dot(relative, Vector(bitangent))) > source.Height / 2)
        return false;
    nearest = distance;
    return true;
}
__device__ bool Unoccluded(const SceneView& scene, Ray ray, GroupStatistics& statistics)
{
    ++statistics.ShadowRays;
    Hit blocker{};
    blocker.Distance = ray.MaximumDistance;
    if (IntersectScene(scene.Nodes, scene.References, scene.Triangles, scene.Spheres, scene.Frame, ray, true,
                       blocker, statistics.PrimitiveTests))
        return false;
    float nearest = ray.MaximumDistance;
    for (unsigned index = 0; index < scene.Controls.EmitterCount; ++index)
        if (scene.Emitters[index].Kind == ArvoCudaEndpoint::Rectangle)
        {
            ++statistics.PrimitiveTests;
            if (HitRectangle(scene.Emitters[index], ray, nearest))
                return false;
        }
    return true;
}
__device__ float MisWeight(float sampled, float competing, unsigned flags)
{
    constexpr unsigned MisNone = 0, MisBalance = 1;
    unsigned heuristic = (flags >> MisShift) & MisMask;
    if (heuristic == MisNone)
        return 1;
    if (!(sampled > 0))
        return 0;
    if (!(competing > 0))
        return 1;
    // Scale before squaring to keep finite HDR densities within float range.
    float scale = fmaxf(sampled, competing), a = sampled / scale, b = competing / scale;
    if (heuristic != MisBalance)
    {
        a *= a;
        b *= b;
    }
    return a / (a + b);
}
__device__ float HitEmissionWeight(const SceneView& scene, unsigned source, Vec3 normal, Vec3 launch,
                                   Vec3 previous, Vec3 position, bool sky, bool delta, float previousPdf)
{
    unsigned flags = scene.Frame.TransportFlags;
    if (delta || !(flags & NeeBit) || source == InvalidPrimitiveIndex)
        return 1;
    if (((flags >> MisShift) & MisMask) == 0)
        return 0;
    const auto density = ArvoCudaEndpoint::Pdf(scene.Emitters, scene.Controls.EmitterCount,
                                               EmissionDomain(scene), source, Native(normal), Native(launch));
    float lightPdf = density.SelectionProbability * density.Direction;
    if (!sky)
    {
        Vec3 segment = Subtract(position, previous);
        float cosine = fabsf(Dot(normal, launch));
        lightPdf =
            cosine > 0 ? density.SelectionProbability * density.Position * Dot(segment, segment) / cosine : 0;
    }
    return MisWeight(previousPdf, lightPdf, flags);
}
__device__ Vec3 ConnectSource(const SceneView& scene, const Mlt::PrimaryState& primary, unsigned base,
                              const PhysicalSurface& surface, Vec3 position, float diffuseFraction,
                              float interiorMeanFreePath, GroupStatistics& statistics)
{
    const auto endpoint =
        ArvoCudaEndpoint::Emit(scene.Emitters, scene.Controls.EmitterCount, EmissionDomain(scene),
                               Variate(primary, base + LightChoice), Pair(primary, base + LightPosition),
                               Pair(primary, base + LightDirection));
    const auto& emission = endpoint.Emission;
    using namespace ArvoCudaEmission;
    if (!(emission.Flags & Valid))
        return {};
    bool infinite = (emission.Flags & Infinite) != 0, point = (emission.Flags & DeltaPosition) != 0;
    bool delta = (emission.Flags & (DeltaPosition | DeltaDirection)) != 0;
    unsigned flags = scene.Frame.TransportFlags;
    if (!delta && !(flags & NeeBit))
        return {};
    Vec3 direction = Multiply(Vector(emission.Direction), -1);
    float geometry = 1, distance = MaximumRayDistance;
    if (!infinite)
    {
        Vec3 segment = Subtract(Vector(emission.Origin), position);
        distance = norm3df(segment.X, segment.Y, segment.Z);
        if (!(distance > 0) || !isfinite(distance))
            return {};
        direction = Multiply(segment, 1 / distance);
        geometry = (point ? 1 : fabsf(Dot(Vector(emission.Normal), direction))) / distance / distance;
    }
    auto evaluation = EvaluatePhysical(surface, direction, (flags & ImportanceBit) != 0);
    float density = endpoint.SelectionProbability * (infinite ? emission.DirectionPdf : emission.PositionPdf);
    if (!(density > 0))
        return {};
    float angularPdf = infinite || delta ? density : geometry > 0 ? density / geometry : 0;
    float weight = delta ? 1 : MisWeight(angularPdf, evaluation.Pdf * diffuseFraction, flags);
    Vec3 radiance = SourceRadiance(scene.Emitters[endpoint.EmitterIndex], Multiply(direction, -1));
    Vec3 value =
        Multiply(Multiply(evaluation.Value, radiance),
                 diffuseFraction * fabsf(Dot(surface.Normal, direction)) * geometry * weight / density);
    float connectionMeanFreePath = surface.Material.Optical.W > 0
        ? (Dot(surface.Geometry, direction) < 0 ? surface.Material.Optical.W : 0)
        : interiorMeanFreePath;
    if (connectionMeanFreePath > 0)
        value = Multiply(value, expf(-distance / connectionMeanFreePath));
    if (!Finite(value) || !Positive(value))
        return value;
    Vec3 origin = PhysicalRayOrigin(surface, position, direction);
    if (!infinite)
    {
        Vec3 target = point
                          ? Vector(emission.Origin)
                          : Offset(Vector(emission.Origin), Vector(emission.Normal), Multiply(direction, -1));
        Vec3 segment = Subtract(target, origin);
        distance = norm3df(segment.X, segment.Y, segment.Z);
        if (!(distance > 2 * RayEpsilon))
            return {};
        direction = Multiply(segment, 1 / distance);
        distance -= RayEpsilon;
    }
    return Unoccluded(scene, {origin, direction, RayEpsilon, distance}, statistics) ? value : Vec3{};
}

__device__ Vec3 ConnectInteriorSource(const SceneView& scene, const Mlt::PrimaryState& primary,
                                     unsigned base, Vec3 position, Vec3 incoming,
                                     const PackedMaterial& interior, GroupStatistics& statistics)
{
    const auto endpoint = ArvoCudaEndpoint::Emit(scene.Emitters, scene.Controls.EmitterCount,
        EmissionDomain(scene), Variate(primary, base + LightChoice),
        Pair(primary, base + LightPosition), Pair(primary, base + LightDirection));
    const auto& emission = endpoint.Emission;
    using namespace ArvoCudaEmission;
    if (!(emission.Flags & Valid)) return {};
    const bool infinite = (emission.Flags & Infinite) != 0;
    const bool point = (emission.Flags & DeltaPosition) != 0;
    const bool delta = (emission.Flags & (DeltaPosition | DeltaDirection)) != 0;
    const unsigned flags = scene.Frame.TransportFlags;
    if (!delta && !(flags & NeeBit)) return {};
    Vec3 direction = Multiply(Vector(emission.Direction), -1);
    float distance = MaximumRayDistance, geometry = 1;
    if (!infinite)
    {
        const Vec3 segment = Subtract(Vector(emission.Origin), position);
        const float squared = Dot(segment, segment);
        if (!(squared > 0) || !isfinite(squared)) return {};
        distance = sqrtf(squared);
        direction = Multiply(segment, 1 / distance);
        geometry = (point ? 1 : fabsf(Dot(Vector(emission.Normal), direction))) / squared;
    }
    const float density = endpoint.SelectionProbability *
        (infinite ? emission.DirectionPdf : emission.PositionPdf);
    if (!(density > 0) || !(geometry > 0)) return {};
    const float sampledPdf = delta ? density : infinite ? density : density / geometry;
    const float phase = SubsurfacePhase(Dot(incoming, direction), interior.ConductorIndex.W);
    const float phasePdf = (flags & ImportanceBit) != 0 ? phase : 1 / (4 * Pi);
    const float mis = delta ? 1 : MisWeight(sampledPdf, phasePdf, flags);
    const Vec3 radiance = SourceRadiance(scene.Emitters[endpoint.EmitterIndex], Multiply(direction, -1));
    Vec3 value = Multiply(radiance, phase * geometry * mis / density
        * expf(-distance / interior.Optical.W));
    if (!Finite(value) || !Positive(value)) return value;
    if (!infinite)
    {
        const Vec3 target = point ? Vector(emission.Origin)
            : Offset(Vector(emission.Origin), Vector(emission.Normal), Multiply(direction, -1));
        const Vec3 segment = Subtract(target, position);
        distance = norm3df(segment.X, segment.Y, segment.Z);
        if (!(distance > RayEpsilon)) return {};
        direction = Multiply(segment, 1 / distance);
        distance -= RayEpsilon;
    }
    return Unoccluded(scene, {position, direction, RayEpsilon, distance}, statistics) ? value : Vec3{};
}

__device__ Mlt::PathValue EvaluateCameraPath(const SceneView& scene, const Mlt::PrimaryState& primary,
                                             GroupStatistics& statistics)
{
    const auto& frame = scene.Frame;
    ArvoCudaCamera::Frame camera{Native(frame.CameraPosition),
                                 Native(frame.CameraForward),
                                 Native(frame.CameraRight),
                                 Native(frame.CameraUp),
                                 frame.HalfWidth,
                                 frame.HalfHeight,
                                 frame.LensRadius,
                                 frame.FocusDistance};
    const auto initial = ArvoCudaCamera::Emit(camera, Pair(primary, 0), Pair(primary, 2));
    Mlt::PathValue result{initial.Raster.x, initial.Raster.y, {}, 0};
    if (!(initial.Flags & ArvoCudaCamera::Valid))
    {
        atomicOr(scene.Error, Mlt::NumericalFailure);
        return result;
    }
    Ray ray{Vector(initial.Origin), Vector(initial.Direction), RayEpsilon, MaximumRayDistance};
    Vec3 throughput{1, 1, 1}, previous = ray.Origin;
    float previousPdf = 0, etaScale = 1;
    bool previousDelta = true;
    unsigned interiorMaterial = InvalidPrimitiveIndex;
    for (unsigned vertex = 0; vertex <= frame.MaxReflectionBounces + 1; ++vertex)
    {
        const unsigned base = Mlt::CameraDimensions + vertex * Mlt::VertexDimensions;
        if (vertex == 0)
            ++statistics.PrimaryRays;
        else
            ++statistics.ReflectionRays;
        Hit hit{};
        hit.Distance = ray.MaximumDistance;
        bool found = IntersectScene(scene.Nodes, scene.References, scene.Triangles, scene.Spheres, frame, ray,
                                    false, hit, statistics.PrimitiveTests);
        // The demo may contain a mesh coincident with an analytic rectangle.
        // Select the analytic endpoint once, including its one-sided emission.
        float endpointDistance = found ? hit.Distance + RayEpsilon : MaximumRayDistance;
        unsigned rectangle = InvalidPrimitiveIndex;
        for (unsigned index = 0; index < scene.Controls.EmitterCount; ++index)
            if (scene.Emitters[index].Kind == ArvoCudaEndpoint::Rectangle)
            {
                ++statistics.PrimitiveTests;
                if (HitRectangle(scene.Emitters[index], ray, endpointDistance))
                    rectangle = index;
            }
        if (found && hit.MaterialIndex >= frame.MaterialCount)
        {
            atomicOr(scene.Error, Mlt::InvalidConfiguration);
            return result;
        }
        if (vertex == 0 && found && !hit.FrontFace && scene.Materials[hit.MaterialIndex].Optical.W > 0)
            interiorMaterial = hit.MaterialIndex;
        if (interiorMaterial != InvalidPrimitiveIndex)
        {
            const PackedMaterial interior = scene.Materials[interiorMaterial];
            const float endpoint = rectangle != InvalidPrimitiveIndex ? endpointDistance
                : found ? hit.Distance : ray.MaximumDistance;
            const SubsurfaceFlight flight = SubsurfaceFreeFlight(interior.Optical.W,
                make_float3(interior.AlbedoReflectivity.X * interior.ConductorExtinction.W,
                    interior.AlbedoReflectivity.Y * interior.ConductorExtinction.W,
                    interior.AlbedoReflectivity.Z * interior.ConductorExtinction.W),
                Native(ray.Direction), endpoint, Variate(primary, base + InteriorFreeFlight));
            throughput = Multiply(throughput, Vector(flight.Weight));
            if (!Finite(throughput)) { atomicOr(scene.Error, Mlt::NumericalFailure); return result; }
            if (!Positive(throughput)) break;
            if (flight.Scattered)
            {
                if (vertex > frame.MaxReflectionBounces) break;
                const Vec3 position = Add(ray.Origin, Multiply(ray.Direction, flight.Distance));
                result.Radiance = Add(result.Radiance, Multiply(throughput,
                    ConnectInteriorSource(scene, primary, base, position, ray.Direction, interior, statistics)));
                const SubsurfaceDirection phase = SubsurfaceSampleDirection(Native(ray.Direction),
                    interior.ConductorIndex.W, Pair(primary, base + InteriorPhaseDirection),
                    (frame.TransportFlags & ImportanceBit) != 0);
                if (!(phase.Pdf > 0)) break;
                throughput = Multiply(throughput, phase.Weight);
                if (!Finite(throughput)) { atomicOr(scene.Error, Mlt::NumericalFailure); return result; }
                if (!Positive(throughput)) break;
                if ((frame.TransportFlags & RouletteBit) && vertex + 1 >= frame.RouletteStartDepth)
                {
                    constexpr float MinimumSurvival = 0.05f, MaximumSurvival = 0.95f;
                    const float survival = fminf(MaximumSurvival,
                        fmaxf(MinimumSurvival, fmaxf(throughput.X,
                            fmaxf(throughput.Y, throughput.Z)) * etaScale));
                    if (Variate(primary, base + RouletteChoice) >= survival) break;
                    throughput = Multiply(throughput, 1 / survival);
                }
                previous = position;
                previousPdf = phase.Pdf;
                previousDelta = false;
                ray = {position, Vector(phase.Direction), RayEpsilon, MaximumRayDistance};
                continue;
            }
        }
        Vec3 launch = Multiply(ray.Direction, -1);
        if (rectangle != InvalidPrimitiveIndex || !found)
        {
            unsigned source = rectangle != InvalidPrimitiveIndex
                                  ? rectangle
                                  : SourceIndex(scene, InvalidPrimitiveIndex, true);
            if (source != InvalidPrimitiveIndex)
            {
                bool sky = rectangle == InvalidPrimitiveIndex;
                Vec3 position = sky ? ray.Origin : Add(ray.Origin, Multiply(ray.Direction, endpointDistance));
                Vec3 normal = sky ? launch : Vector(scene.Emitters[source].Direction);
                float weight = HitEmissionWeight(scene, source, normal, launch, previous, position, sky,
                                                 previousDelta, previousPdf);
                result.Radiance = Add(
                    result.Radiance,
                    Multiply(Multiply(throughput, SourceRadiance(scene.Emitters[source], launch)), weight));
            }
            break;
        }
        Vec3 position = Add(ray.Origin, Multiply(ray.Direction, hit.Distance));
        PackedMaterial material = scene.Materials[hit.MaterialIndex];
        unsigned source = SourceIndex(scene, hit.PrimitiveIndex, false);
        if (source != InvalidPrimitiveIndex)
        {
            float emissionWeight = HitEmissionWeight(scene, source, hit.GeometricNormal, launch, previous,
                                                     position, false, previousDelta, previousPdf);
            result.Radiance =
                Add(result.Radiance,
                    Multiply(Multiply(throughput, SourceRadiance(scene.Emitters[source], launch)),
                             emissionWeight));
        }
        if (vertex > frame.MaxReflectionBounces)
            break;
        float mirror =
            material.Model == LegacyMaterial ? fminf(1, fmaxf(0, material.AlbedoReflectivity.W)) : 0;
        if (material.Model == LegacyMaterial)
            material.Model = DiffuseMaterial;
        PhysicalSurface surface = MakePhysicalSurface(material, hit, launch, TexturedColor(scene, hit));
        result.Radiance =
            Add(result.Radiance, Multiply(throughput, ConnectSource(scene, primary, base, surface, position,
                1 - mirror, interiorMaterial == InvalidPrimitiveIndex ? 0 :
                    scene.Materials[interiorMaterial].Optical.W, statistics)));
        PhysicalSample scattered{};
        float lobe = Variate(primary, base + ScatterLobe);
        if (mirror > 0 && lobe < mirror)
        {
            scattered.Direction = Normalize(Reflect(ray.Direction, surface.Normal));
            if (!SurfaceHemisphere(surface, scattered.Direction, false))
                break;
            scattered.Weight = {1, 1, 1};
            scattered.Pdf = mirror;
            scattered.RelativeIndex = 1;
            scattered.Event = ArvoCuda::ScatteringDelta | ArvoCuda::ScatteringReflection;
        }
        else
        {
            scattered = SamplePhysical(surface, Pair(primary, base + ScatterDirection),
                                       mirror > 0 ? (lobe - mirror) / (1 - mirror) : lobe,
                                       (frame.TransportFlags & ImportanceBit) != 0);
            scattered.Pdf *= 1 - mirror;
        }
        if (!scattered.Event || !(scattered.Pdf > 0))
            break;
        throughput = Multiply(throughput, scattered.Weight);
        etaScale *= scattered.RelativeIndex * scattered.RelativeIndex;
        if (!Finite(throughput) || !isfinite(etaScale))
        {
            atomicOr(scene.Error, Mlt::NumericalFailure);
            return result;
        }
        if (!Positive(throughput))
            break;
        if ((frame.TransportFlags & RouletteBit) && vertex + 1 >= frame.RouletteStartDepth)
        {
            constexpr float MinimumSurvival = 0.05f, MaximumSurvival = 0.95f;
            float survival = fminf(
                MaximumSurvival,
                fmaxf(MinimumSurvival, fmaxf(throughput.X, fmaxf(throughput.Y, throughput.Z)) * etaScale));
            if (Variate(primary, base + RouletteChoice) >= survival)
                break;
            throughput = Multiply(throughput, 1 / survival);
        }
        previous = position;
        previousPdf = scattered.Pdf;
        previousDelta = (scattered.Event & ArvoCuda::ScatteringDelta) != 0;
        ray = {PhysicalRayOrigin(surface, position, scattered.Direction), scattered.Direction, RayEpsilon,
               MaximumRayDistance};
        if (material.Optical.W > 0 && (scattered.Event & ArvoCuda::ScatteringTransmission) != 0)
            interiorMaterial = hit.FrontFace ? hit.MaterialIndex : InvalidPrimitiveIndex;
    }
    result.Importance = ImportanceRed * result.Radiance.X + ImportanceGreen * result.Radiance.Y +
                        ImportanceBlue * result.Radiance.Z;
    if (!Finite(result.Radiance) || result.Radiance.X < 0 || result.Radiance.Y < 0 || result.Radiance.Z < 0 ||
        !isfinite(result.Importance) || result.Importance < 0)
        atomicOr(scene.Error, Mlt::NumericalFailure);
    return result;
}
__device__ void Splat(const FrameConstants& frame, const Mlt::PathValue& value, double probability,
                      double normalization, Mlt::PassPixel* film)
{
    if (!(probability > 0 && value.Importance > 0))
        return;
    unsigned x = static_cast<unsigned>(value.RasterX * frame.FullWidth);
    unsigned y = static_cast<unsigned>(value.RasterY * frame.FullHeight);
    if (x < frame.TileX || y < frame.TileY || x - frame.TileX >= frame.Width ||
        y - frame.TileY >= frame.Height)
        return;
    double weight = normalization * probability / value.Importance;
    auto& pixel = film[(y - frame.TileY) * frame.Width + x - frame.TileX];
    atomicAdd(&pixel.Red, weight * value.Radiance.X);
    atomicAdd(&pixel.Green, weight * value.Radiance.Y);
    atomicAdd(&pixel.Blue, weight * value.Radiance.Z);
}
} // namespace

extern "C" __global__ void MetropolisBootstrapKernel(
    const PackedAccelerationNode* nodes, const PackedPrimitiveRef* references,
    const PackedTriangle* triangles, const PackedSphere* spheres, const PackedMaterial* materials,
    const PackedTexture* textures, const unsigned* texels, const Emitter* emitters,
    const FrameConstants* frame, const Mlt::Controls* controls, Mlt::PathValue* bootstrap,
    GroupStatistics* statistics, unsigned* error)
{
    unsigned slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= controls->Count || controls->Begin + slot >= controls->BootstrapCount)
        return;
    SceneView scene{nodes,  references, triangles, spheres,   materials, textures,
                    texels, emitters,   *frame,    *controls, error};
    GroupStatistics counted{};
    Mlt::PrimaryState primary;
    IndependentPrimary(primary, BootstrapSeed(*controls, controls->Begin + slot));
    bootstrap[controls->Begin + slot] = EvaluateCameraPath(scene, primary, counted);
    statistics[slot] = counted;
}
extern "C" __global__ void MetropolisDistributionKernel(const Mlt::Controls* controls,
                                                        const Mlt::PathValue* bootstrap, double* cdf,
                                                        Mlt::Distribution* distribution, unsigned* error)
{
    if (blockIdx.x || threadIdx.x)
        return;
    // One bounded, fixed-order scan avoids scheduling-dependent bootstrap choices.
    double total = 0;
    for (unsigned index = 0; index < controls->BootstrapCount; ++index)
    {
        double importance = bootstrap[index].Importance;
        if (!isfinite(importance) || importance < 0)
            atomicOr(error, Mlt::NumericalFailure);
        total += importance;
        cdf[index] = total;
    }
    if (!controls->BootstrapCount || !isfinite(total))
        atomicOr(error, Mlt::NumericalFailure);
    *distribution = {controls->BootstrapCount ? total / controls->BootstrapCount : 0, total};
}
extern "C" __global__ void MetropolisInitializeKernel(const Mlt::Controls* controls,
                                                      const Mlt::PathValue* bootstrap, const double* cdf,
                                                      const Mlt::Distribution* distribution,
                                                      Mlt::Chain* chains)
{
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= controls->ChainCount)
        return;
    auto& chain = chains[index];
    chain = {};
    chain.RandomState = MixWord(ChainDomain ^ controls->Seed ^ (RandomIncrement * (index + 1ull)));
    if (!(distribution->TotalImportance > 0))
        return;
    double target = Uniform(chain.RandomState) * distribution->TotalImportance;
    target = fmin(target, nextafter(distribution->TotalImportance, 0.0));
    unsigned lower = 0, upper = controls->BootstrapCount;
    while (lower < upper)
    {
        unsigned middle = lower + (upper - lower) / 2;
        if (cdf[middle] <= target)
            lower = middle + 1;
        else
            upper = middle;
    }
    IndependentPrimary(chain.Primary, BootstrapSeed(*controls, lower));
    chain.Current = bootstrap[lower];
}
extern "C" __global__ void MetropolisMutateKernel(
    const PackedAccelerationNode* nodes, const PackedPrimitiveRef* references,
    const PackedTriangle* triangles, const PackedSphere* spheres, const PackedMaterial* materials,
    const PackedTexture* textures, const unsigned* texels, const Emitter* emitters,
    const FrameConstants* frame, const Mlt::Controls* controls, const Mlt::Distribution* distribution,
    Mlt::Chain* chains, Mlt::PassPixel* film, GroupStatistics* statistics, unsigned* error)
{
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= controls->ChainCount)
        return;
    GroupStatistics counted{};
    unsigned fullPixels = frame->FullWidth * frame->FullHeight;
    unsigned quota = fullPixels / controls->ChainCount + (index < fullPixels % controls->ChainCount ? 1 : 0);
    SceneView scene{nodes,  references, triangles, spheres,   materials, textures,
                    texels, emitters,   *frame,    *controls, error};
    auto& chain = chains[index];
    if (distribution->Normalization > 0)
        for (unsigned step = controls->Begin; step < quota && step - controls->Begin < controls->Count;
             ++step)
        {
            bool large = Uniform(chain.RandomState) < controls->LargeStepProbability;
            Mlt::PrimaryState proposal;
            MutatePrimary(chain.Primary, proposal, chain.RandomState, large, controls->MutationScale);
            Mlt::PathValue value = EvaluateCameraPath(scene, proposal, counted);
            double acceptance =
                chain.Current.Importance > 0 ? fmin(1.0, value.Importance / chain.Current.Importance) : 1;
            Splat(*frame, chain.Current, 1 - acceptance, distribution->Normalization, film);
            Splat(*frame, value, acceptance, distribution->Normalization, film);
            ++chain.Proposals;
            chain.LargeSteps += large;
            if (Uniform(chain.RandomState) < acceptance)
            {
                chain.Primary = proposal;
                chain.Current = value;
                ++chain.Accepted;
            }
            else if (value.Importance > 0)
                ++chain.RejectedPositive;
        }
    statistics[index] = counted;
}
extern "C" __global__ void MetropolisCommitKernel(const FrameConstants* frame, const Mlt::PassPixel* pass,
                                                  Vec4* accumulation, unsigned* counts, unsigned* error)
{
    unsigned pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= frame->Width * frame->Height)
        return;
    const auto value = pass[pixel];
    if (!isfinite(value.Red) || !isfinite(value.Green) || !isfinite(value.Blue) || value.Red < 0 ||
        value.Green < 0 || value.Blue < 0 || value.Red > FLT_MAX || value.Green > FLT_MAX ||
        value.Blue > FLT_MAX)
    {
        atomicOr(error, Mlt::NumericalFailure);
        return;
    }
    AccumulateFilmSample(accumulation[pixel], counts[pixel],
                         {float(value.Red), float(value.Green), float(value.Blue)});
    const Vec4 moments = accumulation[pixel];
    if (!Finite(ToVec3(moments)) || !isfinite(moments.W))
        atomicOr(error, Mlt::NumericalFailure);
}
