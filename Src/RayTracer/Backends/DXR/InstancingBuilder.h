#pragma once
#include "Renderer/RayTracing.h"
#include <cstddef>
#include <functional>

namespace Arvo::RayTracing::DxrInstances
{
struct GeometryRange
{
    uint32_t NodeBytes = 0, NodeCount = 0, ReferenceOffset = 0, TriangleOffset = 0;
    uint32_t SphereOffset = 0, TriangleCount = 0, FirstPrimitive = 0, ObjectIndex = 0;
    float Minimum[3]{};
    uint32_t SphereCount = 0;
    float Maximum[3]{};
    uint32_t Padding = 0;
};
struct alignas(16) Record
{
    double Inverse[12]{};
    GeometryRange Geometry;
    float Minimum[3]{};
    uint32_t Padding0 = 0;
    float Maximum[3]{};
    uint32_t Padding1 = 0;
    MotionData::RigidMotion Motion;
};
struct Node
{
    float Minimum[3]{};
    uint32_t FirstOrEscape = 0;
    float Maximum[3]{};
    uint32_t Count = 0;
};
struct Hierarchy
{
    std::vector<const SceneGeometry *> Objects;
    std::vector<GeometryRange> Ranges;
    std::vector<Record> Records;
    std::vector<std::array<float, 12>> Transforms;
    std::vector<Node> Nodes;
};
static_assert(sizeof(GeometryRange) == 64 && sizeof(Record) == 272 && sizeof(Node) == 32);
static_assert(offsetof(Record, Geometry) == 96 && offsetof(Record, Minimum) == 160 &&
              offsetof(Record, Maximum) == 176);
bool Validate(const Scene &scene, std::string &error);
bool Build(const Scene &scene, const std::function<bool()> &cancelled, Hierarchy &output, std::string &error);
} // namespace Arvo::RayTracing::DxrInstances

static_assert(offsetof(Arvo::RayTracing::DxrInstances::Record, Motion) == 192);
