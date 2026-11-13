#pragma once
#include <cuda_runtime.h>
namespace ArvoCudaEndpoint
{
// Data ABI only. float3 occupies 12 bytes; each scalar completes a 16-byte lane.
constexpr unsigned Directional = 0, Point = 1, Rectangle = 2, Environment = 3, Triangle = 4, Sphere = 5;
struct alignas(16) PackedEmitter
{
    float3 Origin;
    unsigned Kind;
    float3 Direction;
    float Width;
    float3 Strength;
    float Height;
    float3 BackgroundTop;
    unsigned Primitive;
    float3 BackgroundBottom;
    unsigned SourceIndex;
    float3 AxisX;
    unsigned Instance;
    float3 AxisY;
    unsigned Flags;
    float3 AxisZ;
    unsigned Reserved;
};
} // namespace ArvoCudaEndpoint
