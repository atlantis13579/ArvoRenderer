#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace Arvo::RayTracing::CudaBih
{
    struct Primitive
    {
        std::array<float, 3> Minimum;
        std::array<float, 3> Maximum;
        std::uint32_t Reference = 0;
    };

    struct Node
    {
        std::array<float, 3> Minimum;
        std::array<float, 3> Maximum;
        std::uint32_t FirstReference = 0;
        std::uint32_t ReferenceCount = 0;
        std::uint32_t Axis = 3;
        std::uint32_t AboveChild = 0;
        float LeftMaximum = 0.0f;
        float RightMinimum = 0.0f;
    };

    bool Build(const std::vector<Primitive>& primitives, std::vector<Node>& nodes,
        std::vector<std::uint32_t>& references, const std::function<bool()>& cancelled,
        std::string& error);
}
