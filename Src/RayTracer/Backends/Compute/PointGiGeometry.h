#pragma once
#include "PointGiLayout.h"
#include "Renderer/RayTracingBackendCommon.h"

namespace Arvo::RayTracing::ComputePointGi
{
struct Geometry
{
    std::vector<Surface> Surfaces;
    std::vector<Bounds> Hierarchy;
    std::vector<uint32_t> Indices;
};
bool BuildGeometry(const Scene& scene, const PointGiOptions& options,
                   const Detail::CancellationProbe& cancelled, Geometry& result, std::string& error);
} // namespace Arvo::RayTracing::ComputePointGi
