#pragma once
#include <cstddef>
#include <cstdint>
namespace Arvo::RayTracing::DxrPpm
{
// Offline DXR progressive-photon library. Scene roots retain their ABI.
constexpr uint32_t MaximumReflections = 32, LightBatch = 1024, PixelBatch = 1024, LinearBatch = 65536;
constexpr uint32_t EmitterRoot = 19, PhotonsRoot = 20, HeadsRoot = 21, ErrorsRoot = 22;
constexpr uint32_t VisibleRoot = 23, PixelsRoot = 24, ControlsRoot = 25;
constexpr uint32_t Clear = 0, Initialize = 1, Camera = 2, Emit = 3, Index = 4, Gather = 5, Update = 6;
constexpr uint32_t PhotonStride = 64, VisibleStride = 96, PixelStride = 176;
struct Controls
{
    float Center[3], WorldRadius;
    double InitialRadius, Alpha;
    uint32_t EmitterCount, PathsPerPass, Seed, Classes;
    uint32_t SlotsPerPath, BucketCount, Begin, Count;
    uint32_t Pass, LightBegin, LightCount, Stage;
};
static_assert(sizeof(Controls) == 80 && offsetof(Controls, InitialRadius) == 16 &&
              offsetof(Controls, EmitterCount) == 32 && offsetof(Controls, Pass) == 64);
} // namespace Arvo::RayTracing::DxrPpm
