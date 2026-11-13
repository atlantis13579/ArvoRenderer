#include "HardwareBvh.h"
#include "MotionPreparation.h"
#include <algorithm>
#include <cmath>
#include <limits>
#if defined(_WIN32)
#include <vector>

namespace Arvo::RayTracing::Dxr
{
namespace
{
constexpr uint32_t AllInstanceVisibilityBits = 0xff;
constexpr uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    return (value + alignment - 1) & ~(alignment - 1);
}
D3D12_RESOURCE_BARRIER UavBarrier(ID3D12Resource *resource)
{
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    barrier.UAV.pResource = resource;
    return barrier;
}
} // namespace
// The RHI supplies buffer allocation, while this file owns geometry groups,
// build flags, BLAS/TLAS dependencies and their UAV barriers. The returned
// resources must remain alive until the recorded command list has completed.
bool BuildHardwareBvh(ID3D12GraphicsCommandList4 *commandList, uint32_t vertexCount, uint32_t sphereCount,
                      uint64_t vertexStride, ID3D12Resource *vertices, ID3D12Resource *sphereBounds,
                      const HardwareBvhServices &services, HardwareBvhResources &resources,
                      std::string &error, const DxrInstances::Hierarchy *instancePlan)
{
    ID3D12Device5 *Device = services.Device;
    const auto &CreateBuffer = services.Allocate;
    const auto &CreateUploadBuffer = services.Upload;
    const auto stopped = [&] {
        if (!services.Cancelled || !services.Cancelled())
            return false;
        error = "DXR hardware instance construction was cancelled.";
        return true;
    };
    if (stopped())
        return false;
    if (!Device || !commandList || !CreateBuffer || !CreateUploadBuffer ||
        (vertexCount && (!vertices || vertexStride == 0)) || (sphereCount && !sphereBounds))
    {
        error = "DXR hardware BVH construction requires valid device, buffers and allocation services";
        return false;
    }
    // DXR BLASes may only contain one geometry type. Build each object's two
    // groups once, then reference them from transformed TLAS instances. SBT
    // contributions select triangle/sphere shaders; InstanceID selects metadata.
    // https://microsoft.github.io/DirectX-Specs/d3d/Raytracing.html#geometry-and-acceleration-structures
    constexpr uint32_t TriangleGroup = 0;
    constexpr uint32_t SphereGroup = 1;
    constexpr uint64_t AccelerationAlignment = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT;
    const auto buildBottom = [&](const D3D12_RAYTRACING_GEOMETRY_DESC& geometry,
                                  Microsoft::WRL::ComPtr<ID3D12Resource>& scratch,
                                  Microsoft::WRL::ComPtr<ID3D12Resource>& bottom) {
            D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS inputs{};
            inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
            inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
            inputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
            inputs.NumDescs = 1;
            inputs.pGeometryDescs = &geometry;
            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO info{};
            Device->GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &info);
            if (info.ResultDataMaxSizeInBytes == 0)
            {
                error = "DXR returned an empty BLAS prebuild description.";
                return false;
            }
            if (!CreateBuffer(AlignUp(info.ScratchDataSizeInBytes, AccelerationAlignment),
                              D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                              D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                              scratch, error) ||
                !CreateBuffer(AlignUp(info.ResultDataMaxSizeInBytes, AccelerationAlignment),
                              D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
                              D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                              bottom, error))
                return false;
            D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC build{};
            build.Inputs = inputs;
            build.ScratchAccelerationStructureData =
                scratch->GetGPUVirtualAddress();
            build.DestAccelerationStructureData =
                bottom->GetGPUVirtualAddress();
            commandList->BuildRaytracingAccelerationStructure(&build, 0, nullptr);
            const D3D12_RESOURCE_BARRIER barrier = UavBarrier(bottom.Get());
            commandList->ResourceBarrier(1, &barrier);
            return true;
    };
    std::vector<D3D12_RAYTRACING_INSTANCE_DESC> instances;
    const size_t objectCount = instancePlan ? instancePlan->Objects.size() : 1;
    resources.BottomLevel.resize(objectCount);
    resources.BottomScratch.resize(objectCount);
    for (size_t object = 0; object < objectCount; ++object)
    {
        if (stopped())
            return false;
        const uint32_t objectVertices =
            instancePlan ? instancePlan->Ranges[object].TriangleCount * 3 : vertexCount;
        const uint32_t objectSpheres = instancePlan ? instancePlan->Ranges[object].SphereCount : sphereCount;
        const uint64_t vertexOffset =
            instancePlan ? uint64_t(instancePlan->Ranges[object].TriangleOffset) * 3 * vertexStride : 0;
        const uint64_t sphereOffset =
            instancePlan ? uint64_t(instancePlan->Ranges[object].SphereOffset) * sizeof(D3D12_RAYTRACING_AABB)
                         : 0;
        for (uint32_t group : {TriangleGroup, SphereGroup})
        {
            if (group == TriangleGroup ? objectVertices == 0 : objectSpheres == 0)
                continue;
            D3D12_RAYTRACING_GEOMETRY_DESC geometry{};
            geometry.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
            if (group == TriangleGroup)
            {
                geometry.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
                geometry.Triangles.VertexBuffer.StartAddress =
                    vertices->GetGPUVirtualAddress() + vertexOffset;
                geometry.Triangles.VertexBuffer.StrideInBytes = vertexStride;
                geometry.Triangles.VertexCount = objectVertices;
                geometry.Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
                geometry.Triangles.IndexFormat = DXGI_FORMAT_UNKNOWN;
            }
            else
            {
                geometry.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_PROCEDURAL_PRIMITIVE_AABBS;
                geometry.AABBs.AABBCount = objectSpheres;
                geometry.AABBs.AABBs.StartAddress = sphereBounds->GetGPUVirtualAddress() + sphereOffset;
                geometry.AABBs.AABBs.StrideInBytes = sizeof(D3D12_RAYTRACING_AABB);
            }
            if (!buildBottom(geometry, resources.BottomScratch[object][group], resources.BottomLevel[object][group]))
                return false;
        }
    }
    const size_t placementCount = instancePlan ? instancePlan->Records.size() : 1;
    resources.MotionBottomLevel.resize(placementCount);
    resources.MotionScratch.resize(placementCount);
    resources.MotionBounds.resize(placementCount);
    for (size_t placement = 0; placement < placementCount; ++placement)
    {
        if (stopped())
            return false;
        const size_t object = instancePlan ? instancePlan->Records[placement].Geometry.ObjectIndex : 0;
        for (uint32_t group : {TriangleGroup, SphereGroup})
        {
            auto* bottom = resources.BottomLevel[object][group].Get();
            if (!bottom)
                continue;
            const bool moving = instancePlan && instancePlan->Records[placement].Motion.Enabled;
            if (moving)
            {
                const auto& motion = instancePlan->Records[placement].Motion;
                const auto& source = *instancePlan->Objects[object];
                const size_t count = group == TriangleGroup ? source.Triangles.size() : source.Spheres.size();
                std::vector<D3D12_RAYTRACING_AABB> swept(count);
                constexpr size_t CancellationPeriod = 256;
                for (size_t primitive = 0; primitive < count; ++primitive)
                {
                    if (primitive % CancellationPeriod == 0 && stopped())
                        return false;
                    std::array<double, 3> low{}, high{};
                    if (group == TriangleGroup)
                    {
                        const auto& triangle = source.Triangles[primitive];
                        const std::array<Float3, 3> corners{triangle.Position0, triangle.Position1, triangle.Position2};
                        low.fill(INFINITY);
                        high.fill(-INFINITY);
                        for (const auto p : corners)
                        {
                            const double coordinates[]{p.X, p.Y, p.Z};
                            for (size_t axis = 0; axis < low.size(); ++axis)
                            {
                                low[axis] = std::min(low[axis], coordinates[axis]);
                                high[axis] = std::max(high[axis], coordinates[axis]);
                            }
                        }
                    }
                    else
                    {
                        const auto& sphere = source.Spheres[primitive];
                        const double center[]{sphere.Center.X, sphere.Center.Y, sphere.Center.Z};
                        for (size_t axis = 0; axis < low.size(); ++axis)
                        {
                            low[axis] = center[axis] - sphere.Radius;
                            high[axis] = center[axis] + sphere.Radius;
                        }
                    }
                    std::array<double, 3> motionLow{}, motionHigh{};
                    if (!DxrInstances::SweptLocalBounds(motion, low, high, motionLow, motionHigh))
                    {
                        error = "DXR moving primitive has invalid swept bounds.";
                        return false;
                    }
                    std::array<float, 3> minimum{}, maximum{};
                    for (size_t axis = 0; axis < minimum.size(); ++axis)
                    {
                        minimum[axis] = std::nextafter(float(motionLow[axis]), -INFINITY);
                        maximum[axis] = std::nextafter(float(motionHigh[axis]), INFINITY);
                        if (!std::isfinite(minimum[axis]) || !std::isfinite(maximum[axis]))
                        {
                            error = "DXR procedural motion bounds exceed the finite object-space float range.";
                            return false;
                        }
                    }
                    swept[primitive] = {minimum[0], minimum[1], minimum[2], maximum[0], maximum[1], maximum[2]};
                }
                if (!CreateUploadBuffer(swept.data(), swept.size() * sizeof(swept.front()),
                                        resources.MotionBounds[placement][group], error))
                    return false;
                D3D12_RAYTRACING_GEOMETRY_DESC geometry{};
                geometry.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_PROCEDURAL_PRIMITIVE_AABBS;
                geometry.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
                geometry.AABBs.AABBCount = swept.size();
                geometry.AABBs.AABBs.StartAddress = resources.MotionBounds[placement][group]->GetGPUVirtualAddress();
                geometry.AABBs.AABBs.StrideInBytes = sizeof(D3D12_RAYTRACING_AABB);
                if (!buildBottom(geometry, resources.MotionScratch[placement][group], resources.MotionBottomLevel[placement][group]))
                    return false;
                bottom = resources.MotionBottomLevel[placement][group].Get();
            }
            D3D12_RAYTRACING_INSTANCE_DESC instance{};
            if (instancePlan)
                for (uint32_t row = 0; row < 3; ++row)
                    for (uint32_t column = 0; column < 4; ++column)
                        instance.Transform[row][column] =
                            instancePlan->Transforms[placement][row * 4 + column];
            else
                instance.Transform[0][0] = instance.Transform[1][1] = instance.Transform[2][2] = 1.0f;
            instance.InstanceID = uint32_t(placement);
            instance.InstanceMask = AllInstanceVisibilityBits;
            constexpr uint32_t MotionTriangleHitGroup = 2;
            instance.InstanceContributionToHitGroupIndex = moving && group == TriangleGroup ? MotionTriangleHitGroup : group;
            instance.AccelerationStructure = bottom->GetGPUVirtualAddress();
            instances.push_back(instance);
        }
    }
    if (instances.empty())
        return true;
    if (!CreateUploadBuffer(instances.data(), instances.size() * sizeof(instances.front()),
                            resources.Instances, error))
        return false;
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS inputs{};
    inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
    inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    inputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
    inputs.NumDescs = static_cast<UINT>(instances.size());
    inputs.InstanceDescs = resources.Instances->GetGPUVirtualAddress();
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO info{};
    Device->GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &info);
    if (info.ResultDataMaxSizeInBytes == 0)
    {
        error = "DXR returned an empty TLAS prebuild description.";
        return false;
    }
    if (!CreateBuffer(AlignUp(info.ScratchDataSizeInBytes, AccelerationAlignment), D3D12_HEAP_TYPE_DEFAULT,
                      D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                      resources.TopScratch, error) ||
        !CreateBuffer(AlignUp(info.ResultDataMaxSizeInBytes, AccelerationAlignment), D3D12_HEAP_TYPE_DEFAULT,
                      D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
                      D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, resources.TopLevel, error))
        return false;
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC build{};
    build.Inputs = inputs;
    build.ScratchAccelerationStructureData = resources.TopScratch->GetGPUVirtualAddress();
    build.DestAccelerationStructureData = resources.TopLevel->GetGPUVirtualAddress();
    commandList->BuildRaytracingAccelerationStructure(&build, 0, nullptr);
    const D3D12_RESOURCE_BARRIER barrier = UavBarrier(resources.TopLevel.Get());
    commandList->ResourceBarrier(1, &barrier);
    return true;
}
} // namespace Arvo::RayTracing::Dxr
#endif
