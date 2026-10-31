#pragma once

#include <Renderer/RayTracing.h>

#if defined(_WIN32) && !defined(ARVO_RAYTRACER_STATIC)
#  if defined(ARVO_RAYTRACER_EXPORTS)
#    define ARVO_RAYTRACER_API __declspec(dllexport)
#  else
#    define ARVO_RAYTRACER_API __declspec(dllimport)
#  endif
#else
#  define ARVO_RAYTRACER_API
#endif

namespace Arvo
{
namespace RayTracing
{
    class CpuRayTracingRHI;
    class CudaRayTracingRHI;
    class ComputeShaderRayTracingRHI;
    class MetalRayTracingRHI;

    ARVO_RAYTRACER_API std::shared_ptr<IRayTracingRHI> CreateRayTracingRHI(
        BackendType type);
    ARVO_RAYTRACER_API const char* ToString(BackendType type);
}
}
