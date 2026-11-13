#include <RayTracer/RayTracer.h>

#include "ComputeShaderRayTracingRHI.h"
#include "CpuRayTracingRHI.h"
#include "CudaRayTracingRHI.h"
#include "MetalRayTracingRHI.h"
#include "Renderer/DxrRayTracingRHI.h"

#include <memory>

namespace Arvo
{
namespace RayTracing
{
    std::shared_ptr<IRayTracingRHI> CreateRayTracingRHI(BackendType type)
    {
        switch (type)
        {
        case BackendType::CPU:
            return std::make_shared<CpuRayTracingRHI>();
        case BackendType::DXR:
            // Offline jobs and the viewport share the same implementation and HLSL.
            return Renderer::CreateDxrRHI();
        case BackendType::CUDA:
            return std::make_shared<CudaRayTracingRHI>();
        case BackendType::ComputeShader:
            return std::make_shared<ComputeShaderRayTracingRHI>();
        case BackendType::Metal:
            return std::make_shared<MetalRayTracingRHI>();
        }
        return nullptr;
    }

    const char* ToString(BackendType type)
    {
        switch (type)
        {
        case BackendType::CPU: return "CPU";
        case BackendType::DXR: return "DXR";
        case BackendType::CUDA: return "CUDA";
        case BackendType::ComputeShader: return "ComputeShader";
        case BackendType::Metal: return "Metal";
        }
        return "Unknown";
    }
}
}
