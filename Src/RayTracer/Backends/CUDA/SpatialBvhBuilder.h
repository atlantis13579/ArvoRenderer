#pragma once
#include <array>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace Arvo::RayTracing::CudaSpatial
{
struct Primitive
{
    std::array<float, 3> Minimum, Maximum;
    uint32_t Reference = 0;
};
struct Node
{
    std::array<float, 3> Minimum, Maximum;
    uint32_t FirstOrEscape = 0, PrimitiveCount = 0;
};
struct Tree
{
    std::vector<Node> Nodes;
    std::vector<uint32_t> References;
    uint64_t SpatialSplits = 0;
};
bool Build(const std::vector<Primitive> &primitives, const std::function<bool()> &cancelled, Tree &tree,
           std::string &error);
} // namespace Arvo::RayTracing::CudaSpatial
