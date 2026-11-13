#pragma once
#include "Renderer/RayTracing.h"
#include <cstddef>
#include <functional>

namespace Arvo::RayTracing::ComputeInstances
{
// Data-only native layout. NodeByteOffset includes the BLAS root-bounds header
// used by kd-trees and BIHs; all child and primitive indices remain local.
struct GeometryRange
{
    uint32_t NodeByteOffset = 0, NodeCount = 0, ReferenceOffset = 0, TriangleOffset = 0;
    uint32_t SphereOffset = 0, TriangleCount = 0, FirstPrimitive = 0, Reserved = 0;
};
struct alignas(16) Instance
{
    double Inverse[12]{};
    GeometryRange Geometry;
    float Minimum[3]{};
    uint32_t ObjectIndex = 0;
    float Maximum[3]{};
    uint32_t Padding = 0;
    MotionData::RigidMotion Motion;
};
struct alignas(16) Node
{
    float Minimum[3]{};
    uint32_t FirstOrEscape = 0;
    float Maximum[3]{};
    uint32_t Count = 0;
};
struct Hierarchy
{
    std::vector<const SceneGeometry *> Objects;
    std::vector<Instance> Instances;
    std::vector<Node> Nodes;
};
static_assert(sizeof(GeometryRange) == 32 && sizeof(Instance) == 240 && sizeof(Node) == 32);
static_assert(offsetof(Instance, Geometry) == 96 && offsetof(Instance, Minimum) == 128 &&
              offsetof(Instance, Maximum) == 144);
bool Validate(const Scene &scene, std::string &error);
bool Build(const Scene &scene, const std::function<bool()> &cancelled, Hierarchy &result, std::string &error);
} // namespace Arvo::RayTracing::ComputeInstances

static_assert(offsetof(Arvo::RayTracing::ComputeInstances::Instance, Motion) == 160);
