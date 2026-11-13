#pragma once
#include "PointGiLayout.h"
#include "Renderer/RayTracingBackendCommon.h"

namespace Arvo::RayTracing::CudaPointGi
{
struct GeometryCache
{
    std::vector<Surfel> Surfels;
    std::vector<Node> Nodes;
    std::vector<uint32_t> Order;
};
bool BuildGeometry(const Scene&, const PointGiOptions&, GeometryCache&, const Detail::CancellationProbe&,
                   std::string& error);
} // namespace Arvo::RayTracing::CudaPointGi
