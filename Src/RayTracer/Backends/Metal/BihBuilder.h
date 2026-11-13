#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace Arvo::RayTracing::MetalBih
{
    struct Object
    {
        std::array<float, 3> Lower;
        std::array<float, 3> Upper;
        std::uint32_t EncodedIndex = 0;
    };

    struct IntervalNode
    {
        std::array<float, 3> Lower;
        std::array<float, 3> Upper;
        std::uint32_t Axis = 3;
        std::uint32_t RightChild = 0;
        std::uint32_t ReferenceBegin = 0;
        std::uint32_t ReferenceCount = 0;
        float LeftMaximum = 0.0f;
        float RightMinimum = 0.0f;
    };

    bool Build(const std::vector<Object>& objects, const std::function<bool()>& cancelled,
        std::vector<IntervalNode>& nodes, std::vector<std::uint32_t>& references,
        std::string& error);
}
