#pragma once
#include "Renderer/GeometryGuides.h"
namespace Arvo::RayTracing::Cpu
{
bool GenerateGeometryGuideRays(const GeometryGuideData::Controls&, const Detail::CancellationProbe&,
                               std::vector<GeometryGuideData::EncodedRay>&, std::string&);
bool ResolveGeometryGuideHits(const GeometryGuideData::Controls&,
                              const std::vector<GeometryGuideData::EncodedRay>&,
                              const std::vector<RayQueryHit>&, const Detail::CancellationProbe&,
                              std::vector<GeometryGuide>&, std::string&);
} // namespace Arvo::RayTracing::Cpu
