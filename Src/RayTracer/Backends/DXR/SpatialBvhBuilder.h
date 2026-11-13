#pragma once
#include "Renderer/RayTracing.h"
#include <functional>

namespace Arvo::RayTracing::Dxr
{
constexpr uint32_t SpatialBvhMaximumDepth = 48;
struct SpatialBvhNode
{
    Float3 Minimum;
    uint32_t FirstOrEscape = 0;
    Float3 Maximum;
    uint32_t Count = 0;
};
static_assert(sizeof(SpatialBvhNode) == 32 && offsetof(SpatialBvhNode, FirstOrEscape) == 12 &&
                  offsetof(SpatialBvhNode, Count) == 28,
              "DXR spatial BVH node ABI");
struct SpatialBvhData
{
    Float3 Minimum, Maximum;
    std::vector<SpatialBvhNode> Nodes;
    std::vector<uint32_t> PrimitiveIndices;
    uint64_t SpatialSplits = 0;
};
bool BuildSpatialBvh(const SceneGeometry &scene, const std::function<bool()> &cancelled, SpatialBvhData &tree,
                     std::string &error);
} // namespace Arvo::RayTracing::Dxr
