#pragma once
#include <cstdint>
namespace Arvo::RayTracing::DxrPhoton
{
// Raw buffer / root ABI; all transport and indexing are native DXR HLSL.
constexpr uint32_t MaximumReflections = 32, TransportBatch = 1024, IndexBatch = 65536;
constexpr uint32_t EmitterRoot = 19, RecordsRoot = 20, HeadsRoot = 21, ErrorRoot = 22, ControlsRoot = 23;
constexpr uint32_t Clear = 0, Emit = 1, Index = 2, Camera = 3;
constexpr uint32_t RecordStride = 64, PositionOffset = 0, NextOffset = 12;
constexpr uint32_t IncomingOffset = 16, DepthOffset = 28, FluxOffset = 32, ClassOffset = 44;
constexpr uint32_t NormalOffset = 48, ReservedOffset = 60;
struct Controls
{
    float Center[3], WorldRadius;
    double Radius;
    uint32_t EmitterCount, PhotonPaths;
    uint32_t SlotsPerPath, BucketCount, Seed, Classes;
    uint32_t Begin, Count, Stage, Sample;
};
static_assert(sizeof(Controls) == 64);
} // namespace Arvo::RayTracing::DxrPhoton
