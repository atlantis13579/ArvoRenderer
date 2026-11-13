#pragma once

#include "Renderer/RayTracingBackendCommon.h"

namespace Arvo::RayTracing::ComputeAcceleration
{
    struct BihBounds
    {
        Float3 Minimum;
        uint32_t Padding0 = 0;
        Float3 Maximum;
        uint32_t Padding1 = 0;
    };

    // Interior: left child follows this node; RightChildOrFirstReference points
    // to its sibling. Leaf: that field and PrimitiveCount address references.
    struct BihNode
    {
        float LeftMaximum = 0.0f;
        float RightMinimum = 0.0f;
        uint32_t Axis = 3;
        uint32_t RightChildOrFirstReference = 0;
        uint32_t PrimitiveCount = 0;
        uint32_t Padding0 = 0;
        uint32_t Padding1 = 0;
        uint32_t Padding2 = 0;
    };

    struct BihPrimitive
    {
        Float3 Minimum;
        Float3 Maximum;
        uint32_t EncodedReference = 0;
    };

    struct BihTree
    {
        BihBounds Bounds;
        std::vector<BihNode> Nodes;
        std::vector<uint32_t> References;
    };

    static_assert(sizeof(BihBounds) == 32 && offsetof(BihBounds, Maximum) == 16
        && sizeof(BihNode) == 32 && offsetof(BihNode, PrimitiveCount) == 16,
        "Compute BIH raw-buffer layout mismatch");

    bool BuildBih(const std::vector<BihPrimitive>& primitives,
        const Detail::CancellationProbe& cancellation, BihTree& output, std::string& error);
}
