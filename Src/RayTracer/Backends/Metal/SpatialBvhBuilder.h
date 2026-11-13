#pragma once
#include <array>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace Arvo::RayTracing::MetalSpatial
{
struct Reference
{
    std::array<float, 3> Lower, Upper;
    uint32_t EncodedIndex = 0;
};
struct Node
{
    std::array<float, 3> Lower, Upper;
    uint32_t FirstOrEscape = 0, Count = 0;
};
struct Hierarchy
{
    std::vector<Node> Nodes;
    std::vector<uint32_t> References;
    uint64_t SpatialSplits = 0;
};
bool Build(const std::vector<Reference> &references, const std::function<bool()> &cancelled,
           Hierarchy &hierarchy, std::string &error);
} // namespace Arvo::RayTracing::MetalSpatial
