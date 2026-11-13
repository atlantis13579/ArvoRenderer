#pragma once
#include "InstancingBuilder.h"
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <array>
#include <cstdint>
#include <d3d12.h>
#include <functional>
#include <string>
#include <wrl/client.h>

namespace Arvo::RayTracing::Dxr
{
constexpr size_t HardwareGeometryGroupCount = 2;
struct HardwareBvhResources
{
    using GeometryGroups = std::array<Microsoft::WRL::ComPtr<ID3D12Resource>, HardwareGeometryGroupCount>;
    std::vector<GeometryGroups> BottomLevel, BottomScratch;
    // A distinct motion track owns its procedural BLAS and swept AABB upload.
    std::vector<GeometryGroups> MotionBottomLevel, MotionScratch, MotionBounds;
    Microsoft::WRL::ComPtr<ID3D12Resource> TopLevel, TopScratch, Instances;
};
struct HardwareBvhServices
{
    ID3D12Device5 *Device = nullptr;
    std::function<bool()> Cancelled;
    std::function<bool(uint64_t, D3D12_HEAP_TYPE, D3D12_RESOURCE_STATES, D3D12_RESOURCE_FLAGS,
                       Microsoft::WRL::ComPtr<ID3D12Resource> &, std::string &)>
        Allocate;
    std::function<bool(const void *, uint64_t, Microsoft::WRL::ComPtr<ID3D12Resource> &, std::string &)>
        Upload;
};
bool BuildHardwareBvh(ID3D12GraphicsCommandList4 *commandList, uint32_t vertexCount, uint32_t sphereCount,
                      uint64_t vertexStride, ID3D12Resource *vertices, ID3D12Resource *sphereBounds,
                      const HardwareBvhServices &services, HardwareBvhResources &resources,
                      std::string &error, const DxrInstances::Hierarchy *instances = nullptr);
} // namespace Arvo::RayTracing::Dxr
#endif
