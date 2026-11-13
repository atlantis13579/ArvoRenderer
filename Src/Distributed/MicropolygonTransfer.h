#pragma once
#include "Renderer/MicropolygonOptions.h"
#include <atomic>
#include <map>
#include <string>

namespace Arvo::Distributed
{
struct RenderConfig;
inline constexpr const char* MicropolygonWireVersion = "1";
inline constexpr size_t MaximumMicropolygonReplyBytes = 3u * 1024u * 1024u;
std::string EncodeMicropolygons(const RayTracing::MicropolygonOptions&);
RayTracing::MicropolygonOptions ParseMicropolygons(const std::string&);
std::string EncodeDicingPlan(const RayTracing::MicropolygonPlan&);
std::shared_ptr<const RayTracing::MicropolygonPlan> ParseDicingPlan(const std::string&);
void RequireMicropolygonWorker(const std::map<std::string, std::string>&);
} // namespace Arvo::Distributed
