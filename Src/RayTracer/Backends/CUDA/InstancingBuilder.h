#pragma once
#include "InstanceLayout.h"
#include "Renderer/RayTracing.h"
#include <functional>

namespace Arvo::RayTracing::CudaInstances
{
struct Hierarchy
{
    std::vector<const SceneGeometry *> Objects;
    std::vector<Record> Records;
    std::vector<Node> Nodes;
};
bool Validate(const Scene &scene, std::string &error);
bool Build(const Scene &scene, const std::function<bool()> &cancelled, Hierarchy &hierarchy,
           std::string &error);
} // namespace Arvo::RayTracing::CudaInstances
