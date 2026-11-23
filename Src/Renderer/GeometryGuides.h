#pragma once
#include "Renderer/RayTracingBackendCommon.h"

namespace Arvo::RayTracing
{
namespace GeometryGuideData
{
// Upload ABI, not shared camera or guide-generation arithmetic.
struct alignas(16) EncodedRay
{
    float OriginX, OriginY, OriginZ, Minimum;
    float DirectionX, DirectionY, DirectionZ, Maximum;
};
struct alignas(16) Controls
{
    uint32_t Width, Height, FullWidth, FullHeight;
    uint32_t TileX, TileY, Begin, Count;
    float PositionX, PositionY, PositionZ, FieldOfView;
    float TargetX, TargetY, TargetZ, MinimumDistance;
    float UpX, UpY, UpZ, MaximumDistance;
    uint32_t Reserved[4]{};
};
static_assert(sizeof(EncodedRay) == 32 && sizeof(Controls) == 96);
static_assert(offsetof(Controls, PositionX) == 32 && offsetof(Controls, UpX) == 64);
Controls PackControls(const Camera&, const RenderSettings&);
} // namespace GeometryGuideData
bool ValidateGeometryGuideRequest(const Camera&, const RenderSettings&, std::string& error);
namespace Detail
{
using GenerateGuideRays = std::function<bool(const GeometryGuideData::Controls&,
                                             std::vector<GeometryGuideData::EncodedRay>&, std::string&)>;
using TraceGuideRays = std::function<RayQueryResult(const std::vector<Ray>&)>;
using ResolveGeometryGuides =
    std::function<bool(const GeometryGuideData::Controls&, const std::vector<GeometryGuideData::EncodedRay>&,
                       const std::vector<RayQueryHit>&, std::vector<GeometryGuide>&, std::string&)>;
GeometryGuideResult CaptureGeometryGuides(const Scene&, const Camera&, const RenderSettings&, bool spatialAlbedo,
                                          const CancellationProbe&,
                                          const GenerateGuideRays&, const TraceGuideRays&,
                                          const ResolveGeometryGuides&);
} // namespace Detail
} // namespace Arvo::RayTracing
