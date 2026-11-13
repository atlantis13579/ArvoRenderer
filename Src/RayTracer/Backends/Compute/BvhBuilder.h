#pragma once
#include <array>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace Arvo::RayTracing::ComputeBvh
{
struct Primitive
{
    std::array<float, 3> Minimum, Maximum, Centroid;
    uint32_t EncodedIndex = 0;
};
struct Node
{
    float MinX = 0, MinY = 0, MinZ = 0;
    uint32_t FirstOrEscape = 0;
    float MaxX = 0, MaxY = 0, MaxZ = 0;
    uint32_t PrimitiveCount = 0;
};
struct Tree
{
    std::vector<Node> Nodes;
    std::vector<uint32_t> References;
};
bool Build(const std::vector<Primitive> &input, const std::function<bool()> &cancelled, Tree &tree,
           std::string &error);
} // namespace Arvo::RayTracing::ComputeBvh
