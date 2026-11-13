#pragma once
#include "Renderer/RayTracing.h"
#include "Renderer/RayTracingBackendCommon.h"

namespace Arvo::RayTracing
{
namespace ReconstructionData
{
// Shared upload schema only. All filter arithmetic belongs to backend files.
inline constexpr uint32_t MomentGuidance = 1, GeometryGuidance = 2;
struct alignas(16) Pixel
{
    float Red, Green, Blue, LuminanceM2;
    uint32_t Samples, GuideFlags = 0, GuideMaterial = 0, Reserved = 0;
    Float3 GuidePosition;
    float GuideDepth = 0;
    Float3 GuideNormal;
    float ReservedNormal = 0;
};
struct alignas(16) Controls
{
    uint32_t Width, Height, Pass, Begin;
    uint32_t Count, Flags;
    float NoiseScale, RelativeColorScale;
    float AbsoluteColorScale;
    uint32_t GammaCorrection;
    float NormalScale = 0.25f, RelativePlaneScale = 0.01f;
};
static_assert(sizeof(Pixel) == 64 && offsetof(Pixel, Samples) == 16 && offsetof(Pixel, GuidePosition) == 32);
static_assert(sizeof(Controls) == 48 && offsetof(Controls, AbsoluteColorScale) == 32);
void PackGuide(Pixel &, const GeometryGuide &);
} // namespace ReconstructionData
bool ValidateReconstructionInput(const LinearFilm&, const ReconstructionOptions&, std::string& error);
bool ValidateReconstructionSupport(const Camera &, bool sceneMotion, const ReconstructionOptions &,
                                   const MediumOptions &, std::string &error);
bool ValidateReconstructionGuides(const LinearFilm &, const ReconstructionOptions &,
                                  const GeometryGuideImage *, std::string &error);
namespace Detail
{
// Scheduling/cropping only. Each callback invokes its own native implementation.
using RawRenderFunction = std::function<RenderResult(const RenderSettings&)>;
using ReconstructionFunction =
    std::function<bool(const RenderSettings &, const LinearFilm &, ReconstructedImage &, std::string &)>;
// One render owns this cache. Equal pilot rasters reuse geometry, never noisy
// radiance; nested halos carry their actual full-film coordinates to capture.
class SpatialGuideCache
{
  public:
    const GeometryGuideImage *Get(const RenderSettings &, const std::function<GeometryGuideResult()> &capture,
                                  std::string &error);

  private:
    GeometryGuideImage Image;
};
RenderResult RenderWithReconstruction(const RenderSettings&, const CancellationProbe&,
                                      const RawRenderFunction&, const ReconstructionFunction&);
} // namespace Detail
} // namespace Arvo::RayTracing
