#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace Arvo::RayTracing::MetalKd
{
    struct PrimitiveBounds
    {
        std::array<float, 3> Lower;
        std::array<float, 3> Upper;
        std::uint32_t EncodedIndex = 0;
    };

    struct SpatialNode
    {
        std::array<float, 3> Lower;
        std::array<float, 3> Upper;
        float Plane = 0.0f;
        std::uint32_t Axis = 3;
        std::uint32_t UpperChild = 0;
        std::uint32_t ReferenceBegin = 0;
        std::uint32_t ReferenceCount = 0;
    };

    bool Build(const std::vector<PrimitiveBounds>& primitives,
        const std::function<bool()>& cancelled, std::vector<SpatialNode>& nodes,
        std::vector<std::uint32_t>& references, std::string& error);
}
