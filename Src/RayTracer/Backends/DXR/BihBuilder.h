#pragma once

#include "Renderer/RayTracing.h"

#include <functional>
#include <string>
#include <vector>

namespace Arvo::RayTracing::Dxr
{
    constexpr uint32_t BihMaximumDepth = 48;
    constexpr uint32_t BihLeafAxis = 3;

    // Explicit fields keep the two clipping planes readable. Interior children
    // occupy First and First+1; leaves address Count entries starting at First.
    struct BihNode
    {
        float LeftMaximum = 0;
        float RightMinimum = 0;
        uint32_t First = 0;
        uint32_t Count = 0;
        uint32_t Axis = BihLeafAxis;
        uint32_t Padding[3]{};
    };
    static_assert(sizeof(BihNode) == 32 && offsetof(BihNode, Axis) == 16,
        "DXR BIH node layout must match its native byte-buffer loader.");

    struct BihData
    {
        Float3 Minimum;
        Float3 Maximum;
        std::vector<BihNode> Nodes;
        // A permutation of the original IDs: no spatial-reference duplication.
        std::vector<uint32_t> PrimitiveIndices;
    };

    bool BuildBih(const SceneGeometry& scene, const std::function<bool()>& cancelled,
        BihData& hierarchy, std::string& error);
}
