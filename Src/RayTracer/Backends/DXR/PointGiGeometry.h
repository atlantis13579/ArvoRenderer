#pragma once
#include "PointGiLayout.h"
#include "Renderer/RayTracingBackendCommon.h"

namespace Arvo::RayTracing::DxrPointGi
{
struct SurfaceCache
{
    std::vector<Surface> Surfaces;
    std::vector<Node> Nodes;
    std::vector<uint32_t> Indices;
};
bool BuildSurfaceCache(const Scene&, const PointGiOptions&, const Detail::CancellationProbe&, SurfaceCache&,
                       std::string& error);
} // namespace Arvo::RayTracing::DxrPointGi
