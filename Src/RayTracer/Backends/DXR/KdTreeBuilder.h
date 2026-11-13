#pragma once

#include "Renderer/RayTracing.h"

#include <functional>
#include <string>
#include <vector>

namespace Arvo::RayTracing::Dxr
{
    // These limits agree with the bounded native stack in KdTree.hlsl.
    constexpr uint32_t KdMaximumDepth = 48;
    constexpr uint32_t KdLeafAxis = 3;

    struct KdNode
    {
        float Split = 0;
        uint32_t First = 0;
        uint32_t Count = 0;
        uint32_t Axis = KdLeafAxis;
    };
    static_assert(sizeof(KdNode) == 16, "Native DXR kd-node ABI mismatch.");

    struct KdTreeData
    {
        Float3 Minimum;
        Float3 Maximum;
        std::vector<KdNode> Nodes;
        std::vector<uint32_t> PrimitiveIndices;
    };

    bool BuildKdTree(const SceneGeometry& scene, const std::function<bool()>& cancelled,
        KdTreeData& tree, std::string& error);
}
