#pragma once
#include "Renderer/RayTracing.h"
#include <functional>

namespace Arvo::RayTracing::ComputeAcceleration
{
struct SpatialPrimitive
{
    Float3 Minimum, Maximum;
    uint32_t Reference = 0;
};
struct SpatialNode
{
    Float3 Minimum;
    uint32_t FirstOrEscape = 0;
    Float3 Maximum;
    uint32_t PrimitiveCount = 0;
};
static_assert(sizeof(SpatialNode) == 32 && offsetof(SpatialNode, FirstOrEscape) == 12 &&
                  offsetof(SpatialNode, PrimitiveCount) == 28,
              "Compute spatial BVH node ABI");
struct SpatialTree
{
    std::vector<SpatialNode> Nodes;
    std::vector<uint32_t> References;
    uint64_t SpatialSplits = 0;
};
bool BuildSpatialBvh(const std::vector<SpatialPrimitive> &primitives, const std::function<bool()> &cancelled,
                     SpatialTree &tree, std::string &error);
} // namespace Arvo::RayTracing::ComputeAcceleration
