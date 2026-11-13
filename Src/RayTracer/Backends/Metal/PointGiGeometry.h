#pragma once
#include "PointGiLayout.h"
#include "Renderer/RayTracingBackendCommon.h"

namespace Arvo::RayTracing::MetalPointGi
{
struct Geometry
{
    std::vector<Surface> Surfaces;
    std::vector<Node> Nodes;
    std::vector<uint32_t> Indices;
};
bool BuildGeometry(const Scene&, const PointGiOptions&, const Detail::CancellationProbe&, Geometry&,
                   std::string& error);
} // namespace Arvo::RayTracing::MetalPointGi
