#pragma once
#include "InstancingBuilder.h"
#include "Renderer/RayTracingBackendCommon.h"

namespace Arvo::RayTracing::MetalScene
{
inline constexpr std::uint32_t SphereBit = 0x80000000u;
inline constexpr std::uint32_t LightTypeDirectional = 0u, LightTypePoint = 1u, LightTypeRectArea = 2u;
struct Vec3
{
    float X = 0.0f;
    float Y = 0.0f;
    float Z = 0.0f;
};

struct alignas(16) Vec4
{
    float X = 0.0f;
    float Y = 0.0f;
    float Z = 0.0f;
    float W = 0.0f;
};

struct alignas(16) PackedAccelerationNode
{
    float MinX = 0.0f;
    float MinY = 0.0f;
    float MinZ = 0.0f;
    std::uint32_t FirstOrEscape = 0;
    float MaxX = 0.0f;
    float MaxY = 0.0f;
    float MaxZ = 0.0f;
    std::uint32_t PrimitiveCount = 0;
    float SplitPosition = 0.0f;
    std::uint32_t SplitAxis = 3;
    std::uint32_t AboveChild = 0;
    float SecondPlane = 0.0f;
};

struct PackedPrimitiveRef
{
    std::uint32_t TypeAndIndex = 0;
};

struct alignas(16) PackedTriangle
{
    Vec4 Position0;
    Vec4 Edge1;
    Vec4 Edge2;
    Vec4 Normal0;
    Vec4 Normal1;
    Vec4 Normal2;
    Vec4 Color0;
    Vec4 Color1;
    Vec4 Color2;
    Vec4 TexCoord01;
    Vec4 TexCoord2;
    std::uint32_t MaterialIndex = 0;
    std::uint32_t Flags = 0;
    std::uint32_t Padding0 = 0;
    std::uint32_t Padding1 = 0;
};

struct alignas(16) PackedSphere
{
    Vec4 CenterRadius;
    std::uint32_t MaterialIndex = 0;
    std::uint32_t Padding0 = 0;
    std::uint32_t Padding1 = 0;
    std::uint32_t Padding2 = 0;
};

struct alignas(16) PackedMaterial
{
    Vec4 AlbedoReflectivity;
    Vec4 AmbientShininess;
    Vec4 Specular;
    Vec4 Emission;
    std::uint32_t DiffuseTextureIndex = InvalidRayTextureIndex;
    std::uint32_t SampledEmitter = 0;
    std::uint32_t Model = 0;
    std::uint32_t Padding = 0;
    Vec4 Optical;
    Vec4 ConductorIndex;
    Vec4 ConductorExtinction;
};

struct alignas(16) PackedLight
{
    Vec3 Position;
    std::uint32_t Type = LightTypeDirectional;
    Vec3 Direction;
    float Intensity = 0.0f;
    Vec3 Color;
    std::uint32_t CastShadows = 0u;
    float Width = 0.0f;
    float Height = 0.0f;
    std::uint32_t Padding0 = 0u;
    std::uint32_t Padding1 = 0u;
};

struct PackedScene
{
    std::vector<PackedAccelerationNode> Nodes;
    std::vector<PackedPrimitiveRef> PrimitiveRefs;
    std::vector<PackedTriangle> Triangles;
    std::vector<PackedSphere> Spheres;
    std::vector<PackedMaterial> Materials;
    std::vector<PackedLight> Lights;
    Detail::PackedRayTextureAtlas TextureAtlas;
    std::vector<MetalInstances::Record> Instances;
    std::vector<MetalInstances::Node> InstanceNodes;
};

static_assert(sizeof(PackedAccelerationNode) == 48 && sizeof(PackedTriangle) == 192 &&
              sizeof(PackedSphere) == 32);
static_assert(sizeof(PackedMaterial) == 128 && offsetof(PackedMaterial, Model) == 72 &&
              offsetof(PackedMaterial, Optical) == 80 && offsetof(PackedMaterial, ConductorIndex) == 96 &&
              offsetof(PackedMaterial, ConductorExtinction) == 112,
              "Metal shading upload must match ShadingTypes.metal.");
bool ValidateGeometry(const SceneGeometry &geometry, size_t materialCount, std::string &error);
bool Pack(const Scene &scene, AccelerationOptions acceleration, const Detail::CancellationProbe &cancellation,
          PackedScene &output, std::string &error);
} // namespace Arvo::RayTracing::MetalScene
