#pragma once
#include "Renderer/RayTracingBackendCommon.h"
#include "Renderer/EmitterData.h"
#include <array>

namespace Arvo::RayTracing::Cpu
{
// A disk is a quadrature approximation to a surface patch, not a photon.
// Area is world-space differential area; its two sides share the same measure.
struct PointGiSurfel
{
    Float3 Position, GeometryNormal, ShadingNormal, Color;
    Float2 TexCoord;
    uint32_t Material = 0, Primitive = InvalidRayPrimitiveIndex, Patch = 0;
    bool HasTexCoord = false;
    double Area = 0, Radius = 0;
};
struct PointGiNode
{
    std::array<double, 3> Low{}, High{}, Center{};
    double Area = 0, Radius = 0;
    Float3 NormalAxis;
    double MinimumNormalCosine = -1;
    uint32_t Begin = 0, Count = 0;
    uint32_t Left = InvalidRayPrimitiveIndex, Right = InvalidRayPrimitiveIndex;
};
struct PointGiSurfaceCache
{
    // Surfel order remains stable. Only the hierarchy's permutation is sorted.
    std::vector<PointGiSurfel> Surfels;
    std::vector<uint32_t> Order;
    std::vector<PointGiNode> Nodes;
};
struct PointGiBakeGeometry
{
    double MaximumArea = 1;
    uint32_t MaximumSurfels = 1048576;
};
bool BuildPointGiSurfaceCache(const Scene&, const PointGiBakeGeometry&, PointGiSurfaceCache&,
                              const Detail::CancellationProbe&, std::string& error);
class SceneAccelerator;
struct AtomicStatistics;
struct CameraFrame;
struct PointGiRadiance
{
    std::array<double, 3> Front{}, Back{};
};
struct PointGiLightingOptions
{
    // Additional diffuse orders after the directly reflected order. These bake
    // samples are independent of camera samples and adaptive pixel statistics.
    uint32_t DirectSamples = 16, GatherSamples = 64, IndirectBounces = 0, Seed = 1;
    SampleSequence Sequence = SampleSequence::Sobol;
    bool ImportanceSampling = true;
    uint32_t SamplingSeed = 0;
};
struct PointGiLighting
{
    // Reflected radiance only. Source emission is evaluated separately.
    std::vector<PointGiRadiance> Direct, Total;
    uint32_t CompletedOrders = 0;
};
struct PointGiDiskHit
{
    uint32_t Surfel = InvalidRayPrimitiveIndex;
    double Distance = std::numeric_limits<double>::infinity();
    bool Front = false;
};
// Cache input must come from BuildPointGiSurfaceCache and remain immutable.
// Returns completion status; a completed miss has an invalid Surfel identity.
bool IntersectPointGiDisks(const PointGiSurfaceCache&, const Ray&, uint32_t ignoredSurfel, PointGiDiskHit&,
                           uint64_t& diskTests, std::string& error);
// Sources come from PrepareEmitters. Use an invalid receiver index for a query
// outside the cache. Input radiance excludes intrinsic emission; misses are dark.
bool GatherPointGiIrradiance(const PointGiSurfaceCache&, const std::vector<PointGiRadiance>&,
                             const PreparedEmissionScene&, const PointGiSurfel& receiver,
                             uint32_t receiverIndex, bool front, const PointGiLightingOptions&,
                             std::array<double, 3>& irradiance, AtomicStatistics&,
                             const Detail::CancellationProbe&, std::string& error);
bool BakePointGiDiffuse(const Scene&, const SceneAccelerator&, const PointGiSurfaceCache&,
                        const PointGiLightingOptions&, PointGiLighting&, AtomicStatistics&,
                        const Detail::CancellationProbe&, std::string& error);
using PointGiDispatcher = std::function<void(uint32_t, const std::function<void(uint32_t)>&)>;
bool RenderPointGiFrame(const Scene&, const SceneAccelerator&, const CameraFrame&, const RenderSettings&,
                        LinearFilm&, AtomicStatistics&, const Detail::CancellationProbe&,
                        const PointGiDispatcher&, std::string& error);
} // namespace Arvo::RayTracing::Cpu
