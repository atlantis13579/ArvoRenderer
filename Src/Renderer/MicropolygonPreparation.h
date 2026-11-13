#pragma once
#include "Renderer/MicropolygonData.h"
#include "Renderer/RayTracingBackendCommon.h"

namespace Arvo::RayTracing
{
struct MicropolygonResult
{
    RenderStatus Status = RenderStatus::Failed;
    std::shared_ptr<const Scene> Geometry;
    std::shared_ptr<const MicropolygonPlan> Plan;
    uint32_t InputTriangles = 0, OutputTriangles = 0, ClampedComponents = 0;
    // Each output instance identifies its source instance for preview selection
    // and reference transforms. Geometry-only consumers may ignore this map.
    std::vector<uint32_t> InstanceSources;
    std::string Error;
};
namespace Detail
{
// Shared work is layout/topology/allocation only. Projection, subdivision and
// displacement are executed by the selected backend's independent program.
using NativeMicropolygonPass =
    std::function<bool(const MicropolygonData::Controls&, const std::vector<MicropolygonData::Patch>&,
                       const std::vector<MicropolygonData::Range>&, std::vector<uint32_t>&, std::string&)>;
MicropolygonResult PrepareMicropolygons(const Scene&, const Camera&, const RenderSettings&,
                                        const CancellationProbe&, const NativeMicropolygonPass&);
} // namespace Detail
} // namespace Arvo::RayTracing
