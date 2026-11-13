#pragma once

#include "Renderer/RayTracingBackendCommon.h"

namespace Arvo::RayTracing::ComputeAcceleration
{
    // The raw GPU node buffer begins with these bounds, then KdNode records.
    struct KdBounds
    {
        Float3 Minimum;
        uint32_t Padding0 = 0;
        Float3 Maximum;
        uint32_t Padding1 = 0;
    };

    struct KdNode
    {
        float SplitPosition = 0.0f;
        uint32_t SplitAxis = 3;
        uint32_t AboveChildOrFirstReference = 0;
        uint32_t PrimitiveCount = 0;
    };

    struct KdPrimitive
    {
        Float3 Minimum;
        Float3 Maximum;
        uint32_t EncodedReference = 0;
    };

    struct KdTree
    {
        KdBounds Bounds;
        std::vector<KdNode> Nodes;
        std::vector<uint32_t> References;
    };

    static_assert(sizeof(KdBounds) == 32 && sizeof(KdNode) == 16,
        "Compute kd-tree raw-buffer layout mismatch");

    bool BuildKdTree(const std::vector<KdPrimitive>& primitives,
        const Detail::CancellationProbe& cancellation, KdTree& output, std::string& error);
}
