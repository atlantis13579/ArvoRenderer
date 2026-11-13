#pragma once
#include "Renderer/TemporalReconstruction.h"
namespace Arvo::RayTracing::Cpu
{
bool TemporalDenoise(const TemporalData::Controls&, const std::vector<TemporalPixel>&,
                     const std::vector<GeometryGuide>&, const std::vector<TemporalPixel>&,
                     const std::vector<GeometryGuide>&, const Detail::CancellationProbe&,
                     std::vector<TemporalPixel>&, Image&, std::string&);
}
