#pragma once

#include "Renderer/RayTracingBackendCommon.h"

namespace Arvo::RayTracing::Cpu
{
using Detail::CancellationProbe;

inline constexpr float Pi = 3.14159265358979323846f;
inline constexpr float RayEpsilon = 1.0e-4f;
inline constexpr float MaximumRayDistance = 1.0e30f;
inline constexpr float MinimumDirectionLengthSquared = 1.0e-12f;
inline constexpr float BasisParallelThreshold = 0.999f;
inline constexpr uint32_t AreaLightSamples = 4;
inline constexpr uint32_t AreaLightStrataPerAxis = 2;
inline constexpr uint32_t PixelSeedMultiplier = 9781;
inline constexpr uint32_t SampleSeedMultiplier = 6271;
inline constexpr uint32_t BounceSeedMultiplier = 0x9e3779b9u;
inline constexpr uint32_t LightSeedMultiplier = 0x85ebca6bu;
inline constexpr uint32_t LightSampleSeedMultiplier = 0xc2b2ae35u;
inline constexpr uint32_t SecondDimensionScramble = 0x68bc21ebu;
inline constexpr uint32_t LobeSelectionScramble = 0xa511e9b3u;
inline constexpr uint32_t DirectBsdfScramble = 0xd1b54a35u;
inline constexpr uint32_t RouletteScramble = 0x94d049bbu;

inline Float3 Multiply(const Float3& a, const Float3& b) { return {a.X * b.X, a.Y * b.Y, a.Z * b.Z}; }
inline float LengthSquared(const Float3& value) { return Dot(value, value); }
inline Float3 Reflect(const Float3& direction, const Float3& normal)
{
    return direction - normal * (2.0f * Dot(direction, normal));
}

struct SurfaceHit
{
    float Distance = std::numeric_limits<float>::max();
    Float3 Normal;
    Float3 GeometricNormal;
    Float3 Color = Float3(1.0f, 1.0f, 1.0f);
    Float2 TexCoord;
    // Keep source colors separate from their RGB interpolation. Spectral
    // shading evaluates each vertex at the path wavelength before mixing it.
    const Triangle* SourceTriangle = nullptr;
    Float2 Barycentric;
    uint32_t MaterialIndex = 0;
    uint32_t PrimitiveIndex = InvalidRayPrimitiveIndex;
    bool HasTexCoord = false;
    bool FrontFace = false;
};

bool IntersectTriangle(const Triangle& triangle, uint32_t primitiveIndex, const Ray& ray, SurfaceHit& hit);
bool IntersectSphere(const Sphere& sphere, uint32_t primitiveIndex, const Ray& ray, SurfaceHit& hit);

class SceneAccelerator
{
public:
    virtual ~SceneAccelerator() = default;
    virtual bool WasCancelled() const = 0;
    // Any-hit queries guarantee only the boolean result, not populated attributes.
    virtual bool Intersect(const Ray& ray, SurfaceHit& hit, uint64_t& primitiveTests, bool anyHit = false) const = 0;
    virtual bool Occluded(const Ray& ray, uint64_t& primitiveTests) const = 0;
    // Static BLASes ignore time. Motion-aware scene traversal overrides this
    // entry without changing the public diagnostic ray's packed ABI.
    virtual bool IntersectAtTime(const Ray& ray, SurfaceHit& hit, uint64_t& tests,
                                 float sceneTime, bool anyHit = false) const;
    bool OccludedAtTime(const Ray& ray, uint64_t& tests, float sceneTime) const;
};

class BVH final : public SceneAccelerator
{
public:
    using Hit = SurfaceHit;
    BVH(const SceneGeometry& scene, CancellationProbe cancellation);
    ~BVH() override;
    bool WasCancelled() const override;
    bool Intersect(const Ray& ray, Hit& hit, uint64_t& primitiveTests, bool anyHit = false) const override;
    bool Occluded(const Ray& ray, uint64_t& primitiveTests) const override;
private:
    struct Impl;
    std::unique_ptr<Impl> Implementation;
};

class KdTree final : public SceneAccelerator
{
public:
    KdTree(const SceneGeometry& scene, CancellationProbe cancellation);
    ~KdTree() override;
    bool WasCancelled() const override;
    bool Intersect(const Ray& ray, SurfaceHit& hit, uint64_t& tests, bool anyHit = false) const override;
    bool Occluded(const Ray& ray, uint64_t& tests) const override;
private:
    struct Impl;
    std::unique_ptr<Impl> Implementation;
};

class Bih final : public SceneAccelerator
{
public:
    Bih(const SceneGeometry& scene, CancellationProbe cancellation);
    ~Bih() override;
    bool WasCancelled() const override;
    bool Intersect(const Ray& ray, SurfaceHit& hit, uint64_t& tests, bool anyHit = false) const override;
    bool Occluded(const Ray& ray, uint64_t& tests) const override;
private:
    struct Impl;
    std::unique_ptr<Impl> Implementation;
};

class SpatialBvh final : public SceneAccelerator
{
public:
    SpatialBvh(const SceneGeometry& scene, CancellationProbe cancellation);
    ~SpatialBvh() override;
    bool WasCancelled() const override;
    bool Intersect(const Ray& ray, SurfaceHit& hit, uint64_t& tests, bool anyHit = false) const override;
    bool Occluded(const Ray& ray, uint64_t& tests) const override;
    uint64_t SpatialSplitCount() const;
    size_t ReferenceCount() const;
private:
    struct Impl;
    std::unique_ptr<Impl> Implementation;
};

std::unique_ptr<SceneAccelerator> BuildAccelerator(const Scene& scene, AccelerationOptions options,
    CancellationProbe cancellation);
std::unique_ptr<SceneAccelerator> BuildGeometryAccelerator(const SceneGeometry& geometry,
    AccelerationOptions options, CancellationProbe cancellation);

bool ValidateInstances(const Scene& scene, std::string& error);
class InstancedScene final : public SceneAccelerator
{
public:
    InstancedScene(const Scene& scene, AccelerationOptions options, CancellationProbe cancellation);
    ~InstancedScene() override;
    bool WasCancelled() const override;
    bool Intersect(const Ray& ray, SurfaceHit& hit, uint64_t& tests, bool anyHit = false) const override;
    bool Occluded(const Ray& ray, uint64_t& tests) const override;
    bool IntersectAtTime(const Ray&, SurfaceHit&, uint64_t&, float sceneTime,
                         bool anyHit = false) const override;
    size_t GeometryBuildCount() const;
    size_t StoredPrimitiveCount() const;
private:
    struct Impl;
    std::unique_ptr<Impl> Implementation;
};

struct AtomicStatistics
{
    std::atomic<uint64_t> PrimaryRays{0};
    std::atomic<uint64_t> PilotRays{0};
    std::atomic<uint64_t> ShadowRays{0};
    std::atomic<uint64_t> ReflectionRays{0};
    std::atomic<uint64_t> PrimitiveTests{0};
    std::atomic<uint64_t> CompletedPixels{0};
    RenderStatistics Resolve(double buildMilliseconds, double renderMilliseconds) const;
};

struct CameraFrame
{
    Float3 Position;
    Float3 Forward;
    Float3 Right;
    Float3 Up;
    float HalfWidth = 1.0f;
    float HalfHeight = 1.0f;
    CameraLens Lens;
    MotionData::Exposure Motion;
};

using Detail::SampleDiffuseTexture;
Float3 Background(const Scene& scene, const Ray& ray);
void AccumulateFilmSample(FilmPixel& pixel, const Float3& radiance);

struct PreparedSpectralFrame;
class SpectralPath;
struct SampleContext
{
    uint32_t PixelX, PixelY, PixelSeed;
    uint32_t SampleIndex, SampleCount;
    SampleSequence Sequence;
    uint32_t SamplingSeed = 0;
    const PreparedSpectralFrame* SpectralFrame = nullptr;
    float SceneTime = 0;
};

uint32_t Hash(uint32_t value);
float Random01(uint32_t seed);
SampleContext MakeSampleContext(const RenderSettings& settings, uint32_t x, uint32_t y, uint32_t sample);
SampleContext MakePilotSampleContext(const RenderSettings& settings, uint32_t x, uint32_t y, uint32_t sample);
uint32_t AdaptiveSampleCount(const FilmPixel& pilot, const RenderSettings& settings);
Float2 Sample2D(const SampleContext& context, uint32_t dimension);
Float2 IndependentSample(const SampleContext& context, uint32_t dimension);
Float2 StratifiedSample(const SampleContext& context, uint32_t dimension);
Float2 SobolSample(const SampleContext& context, uint32_t dimension);
Float2 BlueNoiseSample(const SampleContext& context, uint32_t dimension);
float MultipleImportance(float sampledDensity, float competingDensity, MisHeuristic heuristic);
float RouletteProbability(const Float3& throughput);
Float3 SampleDiffuseDirection(const Float3& normal, float u, float v, bool importance);
float DiffusePdf(float cosine, bool importance);
Ray MakeCameraRay(const CameraFrame& camera, const RenderSettings& settings, const SampleContext& samples);
void ApplyThinLens(Ray& ray, const CameraFrame& camera, const Float2& lensSample);
inline constexpr uint32_t LensSampleDimension = 0x4c454e53u; // "LENS", separate from raster/transport dimensions.
inline constexpr uint32_t TimeSampleDimension = 0x54494d45u;

enum class DirectLightingModel { LegacyBlinnPhong, Lambertian };
Float3 DirectLighting(const Scene& scene, const SceneAccelerator& accelerator, const Ray& ray,
    const SurfaceHit& hit, const Float3& surface, const RenderSettings& settings,
    AtomicStatistics& statistics, const CancellationProbe& cancellation,
    const SampleContext& samples, uint32_t bounce, DirectLightingModel model, bool stochasticLights = false,
    const SpectralPath* spectral = nullptr);

using Integrator = Float3 (*)(const Scene&, const SceneAccelerator&, Ray, const RenderSettings&,
    AtomicStatistics&, const CancellationProbe&, const SampleContext&);
Float3 Whitted(const Scene&, const SceneAccelerator&, Ray, const RenderSettings&, AtomicStatistics&, const CancellationProbe&, const SampleContext&);
Float3 PathTracing(const Scene&, const SceneAccelerator&, Ray, const RenderSettings&, AtomicStatistics&, const CancellationProbe&, const SampleContext&);
Float3 DistributedRayTracing(const Scene&, const SceneAccelerator&, Ray, const RenderSettings&, AtomicStatistics&, const CancellationProbe&, const SampleContext&);
}
