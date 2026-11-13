#pragma once
#include "Renderer/GeometryGuides.h"

namespace Arvo::RayTracing
{
struct TemporalOptions
{
    // Depth tolerance is absolute world distance plus a relative axial term.
    // Bilinear support and RGB neighborhood expansion are unitless fractions.
    uint32_t MaximumHistory = 32;
    float MinimumCurrentWeight = 0.05f;
    float AbsoluteDepthTolerance = 0.01f;
    float RelativeDepthTolerance = 0.01f;
    float MinimumNormalCosine = 0.9f;
    float MinimumHistoryWeight = 0.25f;
    float NeighborhoodExpansion = 0.1f;
    bool GammaCorrection = true;
    bool operator==(const TemporalOptions& b) const
    {
        return MaximumHistory == b.MaximumHistory && MinimumCurrentWeight == b.MinimumCurrentWeight &&
               AbsoluteDepthTolerance == b.AbsoluteDepthTolerance &&
               RelativeDepthTolerance == b.RelativeDepthTolerance &&
               MinimumNormalCosine == b.MinimumNormalCosine &&
               MinimumHistoryWeight == b.MinimumHistoryWeight &&
               NeighborhoodExpansion == b.NeighborhoodExpansion && GammaCorrection == b.GammaCorrection;
    }
};
struct TemporalIdentity
{
    // The owner changes Content whenever geometry, materials, lighting or the
    // transport estimator changes. Camera pose alone may retain correspondence.
    // Frame is strictly increasing within Sequence. Samples identifies actual
    // transport sample ranges/seeds; relabeling identical paths is not a fresh
    // observation. Repeated sample identities reset temporal age to one.
    uint64_t Sequence = 0, Content = 0, Frame = 0, Samples = 0;
};
struct TemporalPixel
{
    Float3 Radiance;
    // Accepted history frames, never a Monte Carlo sample count or confidence.
    uint32_t Age = 0;
};
static_assert(sizeof(TemporalPixel) == 16 && offsetof(TemporalPixel, Age) == 12);
struct TemporalFrame
{
    // Borrowed only for the duration of the synchronous RHI call. The film is
    // immutable, complete and linear; a worker's unfinished crop is not a frame.
    const LinearFilm& Film;
    const GeometryGuideImage& Geometry;
    Camera View;
    TemporalIdentity Identity;
};
struct TemporalHistory
{
    TemporalIdentity Identity;
    Camera View;
    TemporalOptions Options;
    BackendType Backend = BackendType::CPU;
    GeometryGuideImage Geometry;
    std::vector<TemporalPixel> Pixels;
};
enum class TemporalReset
{
    None,
    NoHistory,
    Sequence,
    Content,
    FrameGap,
    RepeatedSamples,
    Projection,
    Resolution,
    Backend,
    Options
};
struct TemporalResult
{
    RenderStatus Status = RenderStatus::Failed;
    TemporalHistory History;
    Image Display;
    TemporalReset Reset = TemporalReset::NoHistory;
    uint64_t AcceptedHistoryPixels = 0;
    std::string Error;
};
bool ValidateTemporalOptions(const TemporalOptions&, std::string& error);
namespace TemporalData
{
// Upload ABI only; all reprojection, visibility and color arithmetic is native.
struct alignas(16) Controls
{
    uint32_t Width, Height, Begin, Count;
    uint32_t HasHistory, MaximumHistory, GammaCorrection, Reserved = 0;
    float MinimumCurrentWeight, AbsoluteDepthTolerance, RelativeDepthTolerance, MinimumNormalCosine;
    float MinimumHistoryWeight, NeighborhoodExpansion, ReservedScales[2]{};
    float EyeX, EyeY, EyeZ, FieldOfView;
    float TargetX, TargetY, TargetZ, ReservedTarget = 0;
    float UpX, UpY, UpZ, ReservedUp = 0;
};
static_assert(sizeof(Controls) == 112 && offsetof(Controls, EyeX) == 64 && offsetof(Controls, UpX) == 96);
} // namespace TemporalData
namespace Detail
{
// Shared input contract only. Reprojection and filtering remain native.
bool ValidateTemporalRequest(BackendType, const TemporalFrame&, const TemporalHistory*,
                             const TemporalOptions&, TemporalReset&, std::string& error);
using NativeTemporalReconstruction =
    std::function<bool(const TemporalData::Controls&, const std::vector<TemporalPixel>&,
                       const std::vector<GeometryGuide>&, const std::vector<TemporalPixel>&,
                       const std::vector<GeometryGuide>&, std::vector<TemporalPixel>&, Image&, std::string&)>;
TemporalResult ReconstructTemporal(BackendType, const TemporalFrame&, const TemporalHistory*,
                                   const TemporalOptions&, const CancellationProbe&,
                                   const NativeTemporalReconstruction&);
} // namespace Detail
} // namespace Arvo::RayTracing
