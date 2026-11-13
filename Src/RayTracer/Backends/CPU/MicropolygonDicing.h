#pragma once
#include "Renderer/MicropolygonData.h"
#include <functional>
#include <vector>

namespace Arvo::RayTracing::Cpu
{
bool MicropolygonPass(const MicropolygonData::Controls&, const std::vector<MicropolygonData::Patch>&,
                      const std::vector<MicropolygonData::Range>&, std::vector<uint32_t>&,
                      const std::function<bool()>& cancelled);
}
