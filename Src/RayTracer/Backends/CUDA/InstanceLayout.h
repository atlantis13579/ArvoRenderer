#pragma once
#include "../../../Renderer/MotionData.h"
#include <cstddef>
#include <cstdint>

namespace Arvo::RayTracing::CudaInstances
{
struct alignas(16) Record
{
    double WorldToObject[12]{};
    uint32_t NodeOffset = 0, NodeCount = 0, ReferenceOffset = 0, TriangleOffset = 0;
    uint32_t SphereOffset = 0, TriangleCount = 0, FirstPrimitive = 0, ObjectIndex = 0;
    float Minimum[3]{};
    uint32_t Padding0 = 0;
    float Maximum[3]{};
    uint32_t Padding1 = 0;
    MotionData::RigidMotion Motion;
};
struct alignas(16) Node
{
    float Minimum[3]{};
    uint32_t FirstOrEscape = 0;
    float Maximum[3]{};
    uint32_t Count = 0;
};
inline constexpr uint32_t InvalidObjectRay = 1;
static_assert(sizeof(Record) == 240 && offsetof(Record, NodeOffset) == 96 &&
              offsetof(Record, FirstPrimitive) == 120);
static_assert(offsetof(Record, Minimum) == 128 && offsetof(Record, Maximum) == 144);
static_assert(sizeof(Node) == 32 && offsetof(Node, FirstOrEscape) == 12 && offsetof(Node, Count) == 28);
} // namespace Arvo::RayTracing::CudaInstances

static_assert(offsetof(Arvo::RayTracing::CudaInstances::Record, Motion) == 160);
