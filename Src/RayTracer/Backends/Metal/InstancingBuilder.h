#pragma once
#include "Renderer/RayTracing.h"
#include <cstddef>
#include <functional>

namespace Arvo::RayTracing::MetalInstances
{
struct alignas(16) Coefficient
{
    float High = 0, Low = 0;
    int32_t Exponent = 0;
    uint32_t Reserved = 0;
};
struct GeometryRange
{
    uint32_t NodeBegin = 0, NodeCount = 0, ReferenceBegin = 0, TriangleBegin = 0;
    uint32_t SphereBegin = 0, TriangleCount = 0, FirstPrimitive = 0, Reserved = 0;
};
struct alignas(16) Record
{
    Coefficient Inverse[12]{};
    GeometryRange Geometry;
    float Minimum[3]{};
    uint32_t ObjectIndex = 0;
    float Maximum[3]{};
    uint32_t Reserved = 0;
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
    // Borrowed immutable geometry; the scene must outlive packing and upload.
    std::vector<const SceneGeometry *> Objects;
    std::vector<Record> Records;
    std::vector<Node> Nodes;
};
static_assert(sizeof(Coefficient) == 16 && sizeof(GeometryRange) == 32);
static_assert(sizeof(Record) == 336 && sizeof(Node) == 32);
static_assert(offsetof(Record, Geometry) == 192 && offsetof(Record, Minimum) == 224 &&
              offsetof(Record, Maximum) == 240);
bool Validate(const Scene &scene, std::string &error);
bool Build(const Scene &scene, const std::function<bool()> &cancelled, Hierarchy &output, std::string &error);
} // namespace Arvo::RayTracing::MetalInstances

static_assert(offsetof(Arvo::RayTracing::MetalInstances::Record, Motion) == 256);
