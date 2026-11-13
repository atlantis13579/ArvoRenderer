#pragma once
#include <cstddef>
#include <cstdint>
namespace Arvo::RayTracing::MetalPpm
{
// Native Metal resource ABI only; transport lives in ProgressivePhotonMapping.metal.
constexpr uint32_t MaximumReflections = 32, ThreadsPerGroup = 64;
constexpr uint32_t LightBatch = 1024, PixelBatch = 1024, LinearBatch = 65536;
constexpr uint32_t PhotonsBinding = 16, HeadsBinding = 17, ControlsBinding = 18, ErrorBinding = 19;
constexpr uint32_t VisibleBinding = 20, PixelsBinding = 21;
constexpr uint32_t ClearStage = 0, InitializeStage = 1, CameraStage = 2, EmitStage = 3;
constexpr uint32_t IndexStage = 4, GatherStage = 5, UpdateStage = 6, StageCount = 7;
constexpr uint32_t PhotonStride = 64, VisibleStride = 96, PixelStride = 336;
struct Controls
{
    float Center[3], WorldRadius;
    uint32_t Sources, PathsPerPass, SlotsPerPath, Buckets;
    uint32_t Seed, Classes, First, Count;
    float InitialRadiusScale, Alpha;
    uint32_t Pass, LightBegin;
    uint32_t LightCount, Reserved[3];
};
static_assert(sizeof(Controls) == 80 && offsetof(Controls, InitialRadiusScale) == 48 &&
              offsetof(Controls, LightCount) == 64);
} // namespace Arvo::RayTracing::MetalPpm
