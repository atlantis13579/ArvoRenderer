#include "Renderer/SubsurfaceValidation.h"
#include "DxrRayTracingRHI.h"

#include "Renderer/RayTracingBackendCommon.h"
#include "RayTracer/Backends/DXR/KdTreeBuilder.h"
#include "RayTracer/Backends/DXR/BihBuilder.h"
#include "RayTracer/Backends/DXR/SpatialBvhBuilder.h"
#include "RayTracer/Backends/DXR/HardwareBvh.h"
#include "RayTracer/Backends/DXR/InstancingBuilder.h"
#include "RayTracer/Backends/DXR/Limits.h"
#include "RayTracer/Backends/DXR/BidirectionalLayout.h"
#include "RayTracer/Backends/DXR/MetropolisLayout.h"
#include "RayTracer/Backends/DXR/PhotonLayout.h"
#include "RayTracer/Backends/DXR/ProgressivePhotonLayout.h"
#include "RayTracer/Backends/DXR/PointGiGeometry.h"
#include "RayTracer/Backends/DXR/VertexConnectionMergingDispatch.h"
#include "Renderer/EmitterData.h"
#include "Renderer/Reconstruction.h"
#include "Renderer/AdaptiveReconstruction.h"
#include "Renderer/PathFrameData.h"
#include "Renderer/SpectralSceneData.h"
#include "Renderer/GeometryGuides.h"
#include "Renderer/TemporalReconstruction.h"
#include "Renderer/MicropolygonPreparation.h"
#include <type_traits>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <functional>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32)
#  if !defined(NOMINMAX)
#    define NOMINMAX
#  endif
#  include <Windows.h>
#  include <d3d12.h>
#  include <d3d12sdklayers.h>
#  include "DxrShaderBytecode.h"
#  include <dxgi1_6.h>
#  include <wrl/client.h>
#endif

namespace Arvo
{
namespace RayTracing
{
    namespace
    {
#if defined(_WIN32)
        using Microsoft::WRL::ComPtr;

        constexpr std::uint32_t DirectionalLightType = 0;
        constexpr std::uint32_t PointLightType = 1;
        constexpr std::uint32_t RectAreaLightType = 2;
        constexpr DWORD GpuWaitTimeoutMilliseconds = 5000;
        constexpr UINT InstanceRecordsRootParameter = 16;
        constexpr UINT InstanceNodesRootParameter = 17;
        constexpr UINT InstanceErrorsRootParameter = 18;
        constexpr UINT MotionTriangleRecord = 4;
        constexpr UINT HitGroupRecordCount = 3;
        constexpr UINT TraversalCallableRecord = MotionTriangleRecord + 1;
        constexpr UINT TraversalCallableCount = 1;
        // PathTracing.hlsl uses the same order after software traversal. These
        // pure stages keep shading/sampling out of the main visibility program.
        constexpr const wchar_t* PathCallableNames[] = {L"PathSurface", L"PathSampling", L"PathSpectrum"};
        constexpr UINT PathCallableCount = TraversalCallableCount + static_cast<UINT>(std::size(PathCallableNames));
        constexpr UINT VcmRayGenerationRecord = TraversalCallableRecord + PathCallableCount;
        constexpr UINT ShaderRecordCount = VcmRayGenerationRecord +
            static_cast<UINT>(std::size(DxrVcm::StageEntryPoints)) - 1;
        enum class IntegratorLibrary { Standard, PathTracing, VertexConnectionMerging };

        struct DynamicModule
        {
            DynamicModule() = default;
            ~DynamicModule()
            {
                if (Handle) FreeLibrary(Handle);
            }

            DynamicModule(const DynamicModule&) = delete;
            DynamicModule& operator=(const DynamicModule&) = delete;

            bool Load(const wchar_t* name)
            {
                Handle = LoadLibraryW(name);
                return Handle != nullptr;
            }

            template <typename Function>
            Function Get(const char* name) const
            {
                return reinterpret_cast<Function>(GetProcAddress(Handle, name));
            }

            HMODULE Handle = nullptr;
        };

        struct EventHandle
        {
            ~EventHandle()
            {
                if (Handle) CloseHandle(Handle);
            }

            EventHandle(const EventHandle&) = delete;
            EventHandle& operator=(const EventHandle&) = delete;
            EventHandle() = default;

            HANDLE Handle = nullptr;
        };

        struct GpuVertex
        {
            float X;
            float Y;
            float Z;
            float Padding;
        };

        struct GpuTriangle
        {
            float Normal0[4];
            float Normal1[4];
            float Normal2[4];
            float Color0[4];
            float Color1[4];
            float Color2[4];
            float TexCoord01[4];
            float TexCoord2[4];
            uint32_t MaterialIndex;
            uint32_t SmoothNormals;
            uint32_t HasTexCoord;
            uint32_t Padding;
        };

        struct GpuSphere
        {
            float Center[3];
            float Radius;
            uint32_t MaterialIndex;
            uint32_t Padding[3];
        };

        struct GpuQueryRay
        {
            float Origin[3];
            float MinimumDistance;
            float Direction[3];
            float MaximumDistance;
        };
        static_assert(sizeof(GpuSphere) == 32 && sizeof(GpuQueryRay) == 32,
            "DXR geometry and query layouts must match native HLSL.");
        static_assert(sizeof(RayQueryHit) == 64 && offsetof(RayQueryHit, PrimitiveIndex) == 28
            && offsetof(RayQueryHit, Flags) == 44, "DXR query payload ABI mismatch.");

        struct GpuMaterial
        {
            float AlbedoReflectivity[4];
            float AmbientShininess[4];
            float Specular[4];
            float Emission[4];
            uint32_t DiffuseTextureIndex;
            uint32_t SampledEmitter;
            uint32_t Model;
            uint32_t Padding;
            float Optical[4];
            float ConductorIndex[4];
            float ConductorExtinction[4];
        };

        struct GpuLight
        {
            float Position[3];
            uint32_t Type;
            float Direction[3];
            float Intensity;
            float Color[3];
            uint32_t CastShadows;
            float Width;
            float Height;
            uint32_t Padding0;
            uint32_t Padding1;
        };

        struct GpuConstants
        {
            float CameraPosition[3];
            uint32_t Width;
            float CameraForward[3];
            uint32_t Height;
            float CameraRight[3];
            uint32_t SamplesPerPixel;
            float CameraUp[3];
            uint32_t MaxReflectionBounces;
            float HalfWidth;
            float HalfHeight;
            uint32_t MaterialCount;
            uint32_t LightCount;
            float BackgroundTop[3];
            float AmbientIntensity;
            float BackgroundBottom[3];
            float ShadowStrength;
            uint32_t GammaCorrection;
            uint32_t PrimitiveCount;
            uint32_t TextureCount;
            uint32_t TextureTexelCount;
            uint32_t FullWidth;
            uint32_t FullHeight;
            uint32_t TileX;
            uint32_t TileY;
            uint32_t Integrator;
            uint32_t TransportFlags;
            uint32_t RouletteStartDepth;
            uint32_t SamplingSequence;
            uint32_t AdaptiveEnabled;
            uint32_t AdaptivePilotSamples;
            uint32_t AdaptiveMinimumSamples;
            uint32_t AdaptivePhase;
            float AdaptiveRelativeError;
            float AdaptiveAbsoluteError;
            uint32_t InstanceNodeCount;
            uint32_t InstanceCount;
            uint32_t TriangleCount;
            uint32_t SphereCount;
            uint32_t Acceleration;
            uint32_t BvhPolicy;
            float SoftwareBoundsMinimum[3];
            uint32_t SoftwareNodeCount;
            float SoftwareBoundsMaximum[3];
            uint32_t SoftwarePrimitiveIndexCount;
            float LensRadius;
            float FocusDistance;
            uint32_t SamplingSeed, LensPadding1;
        };

        static_assert(offsetof(GpuConstants, LensRadius) == 240 && offsetof(GpuConstants, FocusDistance) == 244,
            "DXR lens constant ABI mismatch.");

        struct GpuCounter
        {
            uint32_t Low;
            uint32_t High;

            uint64_t Value() const { return uint64_t(Low) | (uint64_t(High) << 32u); }
        };

        struct GpuStatistics
        {
            GpuCounter Primary;
            GpuCounter Shadow;
            GpuCounter Reflection;
            GpuCounter Completed;
        };
        static_assert(sizeof(GpuCounter) == 8 && sizeof(GpuStatistics) == 32,
            "DXR counters must match the low/high uint pairs in Statistics.hlsl.");

        static_assert(sizeof(GpuVertex) == 16, "DXR vertex layout must match HLSL.");
        static_assert(sizeof(GpuTriangle) == 144, "DXR triangle layout must match HLSL.");
        static_assert(offsetof(GpuTriangle, TexCoord01) == 96, "DXR triangle field layout must match HLSL.");
        static_assert(offsetof(GpuTriangle, MaterialIndex) == 128, "DXR triangle field layout must match HLSL.");
        static_assert(sizeof(GpuMaterial) == 128, "DXR material layout must match HLSL.");
        static_assert(
            offsetof(GpuMaterial, AmbientShininess) == 16
                && offsetof(GpuMaterial, Specular) == 32
                && offsetof(GpuMaterial, Emission) == 48
                && offsetof(GpuMaterial, DiffuseTextureIndex) == 64
                && offsetof(GpuMaterial, SampledEmitter) == 68
                && offsetof(GpuMaterial, Model) == 72
                && offsetof(GpuMaterial, Optical) == 80
                && offsetof(GpuMaterial, ConductorIndex) == 96
                && offsetof(GpuMaterial, ConductorExtinction) == 112,
            "DXR material field layout must match HLSL.");
        static_assert(sizeof(GpuLight) == 64, "DXR light layout must match HLSL.");
        static_assert(
            offsetof(GpuLight, Type) == 12
                && offsetof(GpuLight, Direction) == 16
                && offsetof(GpuLight, Intensity) == 28
                && offsetof(GpuLight, Color) == 32
                && offsetof(GpuLight, CastShadows) == 44
                && offsetof(GpuLight, Width) == 48,
            "DXR light field layout must match HLSL.");
        static_assert(
            sizeof(GpuConstants) == 256
                && offsetof(GpuConstants, ShadowStrength) == 108,
            "DXR constant layout must match HLSL.");
        static_assert(offsetof(GpuConstants, AdaptiveEnabled) == 160, "DXR adaptive constant ABI mismatch.");
        static_assert(offsetof(GpuConstants, InstanceNodeCount) == 184
            && offsetof(GpuConstants, InstanceCount) == 188, "DXR instance constant ABI mismatch.");
        static_assert(offsetof(GpuConstants, Acceleration) == 200
            && offsetof(GpuConstants, SoftwareBoundsMinimum) == 208, "DXR acceleration constant ABI mismatch.");

        bool IsFinite(const Float3& value)
        {
            return std::isfinite(value.X) && std::isfinite(value.Y) && std::isfinite(value.Z);
        }

        bool IsNonNegative(const Float3& value)
        {
            return value.X >= 0.0f && value.Y >= 0.0f && value.Z >= 0.0f;
        }

        static_assert(offsetof(GpuConstants, FullWidth) == 128, "DXR tile constant ABI mismatch.");
        static_assert(offsetof(GpuConstants, Integrator) == 144, "DXR integrator constant ABI mismatch.");

        bool CheckedMultiply(std::uint64_t left, std::uint64_t right, std::uint64_t& result)
        {
            if (left != 0u && right > std::numeric_limits<std::uint64_t>::max() / left)
            {
                return false;
            }

            result = left * right;
            return true;
        }

        bool CheckedAdd(std::uint64_t left, std::uint64_t right, std::uint64_t& result)
        {
            if (right > std::numeric_limits<std::uint64_t>::max() - left)
            {
                return false;
            }

            result = left + right;
            return true;
        }

        bool ValidateDxrScene(const Scene& scene, std::string& error)
        {
            if (!DxrInstances::Validate(scene, error)) return false;
            if (scene.Materials.empty() && (!scene.Triangles.empty() || !scene.Spheres.empty()))
            {
                error = "Scene geometry requires at least one material.";
                return false;
            }

            if (!IsFinite(scene.BackgroundBottom) || !IsFinite(scene.BackgroundTop) ||
                !std::isfinite(scene.AmbientIntensity) || scene.AmbientIntensity < 0.0f)
            {
                error = "Scene environment parameters must be finite and non-negative.";
                return false;
            }

            const std::uint64_t materialCount = static_cast<std::uint64_t>(scene.Materials.size());
            std::uint64_t lightCount =
                static_cast<std::uint64_t>(scene.DirectionalLights.size());
            if (!CheckedAdd(
                    lightCount,
                    static_cast<std::uint64_t>(scene.PointLights.size()),
                    lightCount)
                || !CheckedAdd(
                    lightCount,
                    static_cast<std::uint64_t>(scene.RectAreaLights.size()),
                    lightCount)
                || materialCount > std::numeric_limits<std::uint32_t>::max()
                || lightCount > std::numeric_limits<std::uint32_t>::max())
            {
                error = "DXR material or light count exceeds 32-bit shader indexing limits.";
                return false;
            }

            const uint64_t totalTriangleCount = scene.Triangles.size();
            uint64_t primitiveCount = 0;
            uint64_t vertexCount = 0;
            if (!CheckedAdd(totalTriangleCount, scene.Spheres.size(), primitiveCount)
                || primitiveCount >= InvalidRayPrimitiveIndex
                || !CheckedMultiply(totalTriangleCount, 3u, vertexCount)
                || vertexCount > std::numeric_limits<uint32_t>::max())
            {
                error = "DXR geometry count exceeds 32-bit acceleration structure indexing limits.";
                return false;
            }

            std::uint64_t vertexBytes = 0u;
            std::uint64_t triangleBytes = 0u;
            std::uint64_t materialBytes = 0u;
            std::uint64_t lightBytes = 0u;
            if (!CheckedMultiply(vertexCount, sizeof(GpuVertex), vertexBytes) ||
                !CheckedMultiply(totalTriangleCount, sizeof(GpuTriangle), triangleBytes) ||
                !CheckedMultiply(materialCount, sizeof(GpuMaterial), materialBytes) ||
                !CheckedMultiply(lightCount, sizeof(GpuLight), lightBytes) ||
                vertexBytes > std::numeric_limits<std::size_t>::max() ||
                triangleBytes > std::numeric_limits<std::size_t>::max() ||
                materialBytes > std::numeric_limits<std::size_t>::max() ||
                lightBytes > std::numeric_limits<std::size_t>::max())
            {
                error = "DXR scene buffer size overflow.";
                return false;
            }

            for (const Material& material : scene.Materials)
            {
                if (!IsFinite(material.Albedo) || !IsFinite(material.AmbientColor) ||
                    !IsFinite(material.SpecularColor) || !IsFinite(material.Emission) ||
                    !std::isfinite(material.Shininess) ||
                    !std::isfinite(material.Reflectivity) ||
                    !IsNonNegative(material.AmbientColor) ||
                    !IsNonNegative(material.SpecularColor) ||
                    material.Shininess < 0.0f)
                {
                    error = "Material parameters must be finite; ambient, specular, and shininess must be non-negative.";
                    return false;
                }
            }

            std::vector<const SceneGeometry*> geometries{&scene};
            for (const auto& geometry : scene.Geometries) geometries.push_back(geometry.get());
            for (const auto* geometry : geometries)
            {
                for (const Triangle& triangle : geometry->Triangles)
                {
                    if (triangle.MaterialIndex >= scene.Materials.size())
                    {
                        error = "Triangle material index is out of range.";
                        return false;
                    }

                    if (!IsFinite(triangle.Position0) || !IsFinite(triangle.Position1) ||
                        !IsFinite(triangle.Position2) || !IsFinite(triangle.Normal0) ||
                        !IsFinite(triangle.Normal1) || !IsFinite(triangle.Normal2) ||
                        !IsFinite(triangle.Color0) || !IsFinite(triangle.Color1) ||
                        !IsFinite(triangle.Color2) ||
                        !std::isfinite(triangle.TexCoord0.X) ||
                        !std::isfinite(triangle.TexCoord0.Y) ||
                        !std::isfinite(triangle.TexCoord1.X) ||
                        !std::isfinite(triangle.TexCoord1.Y) ||
                        !std::isfinite(triangle.TexCoord2.X) ||
                        !std::isfinite(triangle.TexCoord2.Y))
                    {
                        error = "Triangle geometry must be finite.";
                        return false;
                    }

                    const Float3 edge1 = {
                        triangle.Position1.X - triangle.Position0.X,
                        triangle.Position1.Y - triangle.Position0.Y,
                        triangle.Position1.Z - triangle.Position0.Z};
                    const Float3 edge2 = {
                        triangle.Position2.X - triangle.Position0.X,
                        triangle.Position2.Y - triangle.Position0.Y,
                        triangle.Position2.Z - triangle.Position0.Z};
                    const Float3 areaVector = {
                        edge1.Y * edge2.Z - edge1.Z * edge2.Y,
                        edge1.Z * edge2.X - edge1.X * edge2.Z,
                        edge1.X * edge2.Y - edge1.Y * edge2.X};
                    const float edge1LengthSquared = edge1.X * edge1.X + edge1.Y * edge1.Y + edge1.Z * edge1.Z;
                    const float edge2LengthSquared = edge2.X * edge2.X + edge2.Y * edge2.Y + edge2.Z * edge2.Z;
                    const float areaSquared = areaVector.X * areaVector.X + areaVector.Y * areaVector.Y +
                        areaVector.Z * areaVector.Z;
                    if (!IsFinite(edge1) || !IsFinite(edge2) || !IsFinite(areaVector) ||
                        !std::isfinite(edge1LengthSquared) || !std::isfinite(edge2LengthSquared) ||
                        !std::isfinite(areaSquared) || edge1LengthSquared < 1.0e-16f ||
                        edge2LengthSquared < 1.0e-16f || areaSquared < 1.0e-16f)
                    {
                        error = "Triangle geometry must be non-degenerate and within finite float range.";
                        return false;
                    }
                }

                for (const Sphere& sphere : geometry->Spheres)
                {
                    if (sphere.MaterialIndex >= scene.Materials.size())
                    {
                        error = "Sphere material index is out of range.";
                        return false;
                    }

                    if (!IsFinite(sphere.Center) || !std::isfinite(sphere.Radius) || sphere.Radius <= 0.0f)
                    {
                        error = "Sphere center and radius must be finite, and radius must be positive.";
                        return false;
                    }

                    const Float3 minimum = {
                        sphere.Center.X - sphere.Radius,
                        sphere.Center.Y - sphere.Radius,
                        sphere.Center.Z - sphere.Radius};
                    const Float3 maximum = {
                        sphere.Center.X + sphere.Radius,
                        sphere.Center.Y + sphere.Radius,
                        sphere.Center.Z + sphere.Radius};
                    if (!IsFinite(minimum) || !IsFinite(maximum))
                    {
                        error = "Sphere bounds exceed finite float range.";
                        return false;
                    }
                }

            }
            for (const DirectionalLight& light : scene.DirectionalLights)
            {
                if (!IsFinite(light.DirectionToLight) || !IsFinite(light.Color) ||
                    !std::isfinite(light.Intensity) ||
                    light.Intensity < 0.0f)
                {
                    error = "Directional light parameters must be finite and intensity must be non-negative.";
                    return false;
                }

                const float directionLengthSquared = light.DirectionToLight.X * light.DirectionToLight.X +
                    light.DirectionToLight.Y * light.DirectionToLight.Y +
                    light.DirectionToLight.Z * light.DirectionToLight.Z;
                if (!std::isfinite(directionLengthSquared) || directionLengthSquared < 1.0e-12f)
                {
                    error = "Directional light direction must be finite and non-zero.";
                    return false;
                }
            }
            for (const PointLight& light : scene.PointLights)
            {
                if (!IsFinite(light.Position) || !IsFinite(light.Color)
                    || !std::isfinite(light.Intensity)
                    || light.Intensity < 0.0f)
                {
                    error =
                        "Point light parameters must be finite and intensity "
                        "must be non-negative.";
                    return false;
                }
            }
            for (const RectAreaLight& light : scene.RectAreaLights)
            {
                const float directionLengthSquared =
                    light.Direction.X * light.Direction.X
                    + light.Direction.Y * light.Direction.Y
                    + light.Direction.Z * light.Direction.Z;
                if (!IsFinite(light.Position) || !IsFinite(light.Direction)
                    || !IsFinite(light.Color)
                    || !std::isfinite(light.Size.X)
                    || !std::isfinite(light.Size.Y)
                    || !std::isfinite(light.Intensity)
                    || light.Intensity < 0.0f
                    || !std::isfinite(directionLengthSquared)
                    || directionLengthSquared < 1.0e-12f
                    || light.Size.X <= 0.0f || light.Size.Y <= 0.0f)
                {
                    error =
                        "Rect area light parameters must be finite with a "
                        "non-zero direction, positive size, and non-negative "
                        "intensity.";
                    return false;
                }
                const Float3 emissionDirection = Normalize(light.Direction);
                const Float3 helper =
                    std::fabs(emissionDirection.Y) < 0.999f
                    ? Float3(0.0f, 1.0f, 0.0f)
                    : Float3(1.0f, 0.0f, 0.0f);
                const Float3 tangent =
                    Normalize(Cross(helper, emissionDirection));
                const Float3 bitangent =
                    Normalize(Cross(emissionDirection, tangent));
                const Float3 halfTangent =
                    tangent * (light.Size.X * 0.5f);
                const Float3 halfBitangent =
                    bitangent * (light.Size.Y * 0.5f);
                if (!IsFinite(light.Position + halfTangent + halfBitangent)
                    || !IsFinite(light.Position + halfTangent - halfBitangent)
                    || !IsFinite(light.Position - halfTangent + halfBitangent)
                    || !IsFinite(light.Position - halfTangent - halfBitangent))
                {
                    error = "Rect area light bounds exceed finite float range.";
                    return false;
                }
            }

            return true;
        }

        bool ValidateDxrInput(
            const Scene& scene,
            const Camera& camera,
            const RenderSettings& settings,
            std::string& error, bool immutableSceneValidated = false)
        {
            if (!ValidateCameraLens(camera.Lens, error) ||
                !ValidateSubsurfaceScene(scene, settings, error) ||
                !ValidateCameraMotionSupport(settings.Integrator, camera.Motion, error) ||
                !ValidateSceneMotionSupport(settings.Integrator, scene, error) ||
                !ValidateCameraSamplePlan(camera, settings, error))
                return false;
            if (!ValidateTileSettings(settings, error)) return false;
            if (!ValidateAcceleration({settings.Acceleration, settings.BvhPolicy}, error)) return false;
            if (!ValidateIntegrator(BackendType::DXR, settings.Integrator, error)) return false;
            if (settings.Width == 0u || settings.Height == 0u)
            {
                error = "Render dimensions must be non-zero.";
                return false;
            }

            constexpr std::uint32_t MaxRenderDimension = 16384u;
            if (settings.Width > MaxRenderDimension || settings.Height > MaxRenderDimension)
            {
                error = "Render dimensions exceed the supported limit.";
                return false;
            }

            if (settings.SamplesPerPixel == 0u || settings.SamplesPerPixel > 4096u)
            {
                error = "Samples per pixel must be in the range [1, 4096].";
                return false;
            }

            if (settings.MaxReflectionBounces > DxrLimits::MaximumReflectionBounces)
            {
                error = "Reflection bounce count exceeds the supported limit.";
                return false;
            }

            if (!std::isfinite(settings.ShadowStrength) ||
                settings.ShadowStrength < 0.0f || settings.ShadowStrength > 1.0f)
            {
                error = "Shadow strength must be finite and in the range [0, 1].";
                return false;
            }

            if (!IsFinite(camera.Position) || !IsFinite(camera.Target) || !IsFinite(camera.Up) ||
                !std::isfinite(camera.VerticalFieldOfViewDegrees))
            {
                error = "Camera parameters must be finite.";
                return false;
            }

            if (camera.VerticalFieldOfViewDegrees <= 1.0f || camera.VerticalFieldOfViewDegrees >= 179.0f)
            {
                error = "Camera vertical field of view must be in the range (1, 179) degrees.";
                return false;
            }

            const Float3 cameraForward = {
                camera.Target.X - camera.Position.X,
                camera.Target.Y - camera.Position.Y,
                camera.Target.Z - camera.Position.Z};
            const float forwardLengthSquared = cameraForward.X * cameraForward.X +
                cameraForward.Y * cameraForward.Y + cameraForward.Z * cameraForward.Z;
            const float upLengthSquared = camera.Up.X * camera.Up.X + camera.Up.Y * camera.Up.Y +
                camera.Up.Z * camera.Up.Z;
            if (!IsFinite(cameraForward) || !std::isfinite(forwardLengthSquared) ||
                !std::isfinite(upLengthSquared) || forwardLengthSquared < 1.0e-10f ||
                upLengthSquared < 1.0e-10f)
            {
                error = "Camera direction and up vectors must be finite and non-zero.";
                return false;
            }

            const float inverseForwardLength = 1.0f / std::sqrt(forwardLengthSquared);
            const float inverseUpLength = 1.0f / std::sqrt(upLengthSquared);
            const Float3 normalizedForward = {
                cameraForward.X * inverseForwardLength,
                cameraForward.Y * inverseForwardLength,
                cameraForward.Z * inverseForwardLength};
            const Float3 normalizedUp = {
                camera.Up.X * inverseUpLength,
                camera.Up.Y * inverseUpLength,
                camera.Up.Z * inverseUpLength};
            const Float3 cameraRight = {
                normalizedForward.Y * normalizedUp.Z - normalizedForward.Z * normalizedUp.Y,
                normalizedForward.Z * normalizedUp.X - normalizedForward.X * normalizedUp.Z,
                normalizedForward.X * normalizedUp.Y - normalizedForward.Y * normalizedUp.X};
            const float rightLengthSquared = cameraRight.X * cameraRight.X + cameraRight.Y * cameraRight.Y +
                cameraRight.Z * cameraRight.Z;
            if (!std::isfinite(rightLengthSquared) || rightLengthSquared < 1.0e-10f)
            {
                error = "Camera direction and up vectors must not be parallel.";
                return false;
            }

            std::uint64_t pixelCount = 0u;
            std::uint64_t outputBytes = 0u;
            std::uint64_t primaryRayCount = 0u;
            std::uint64_t maximumPathRayCount = 0u;
            const bool splitsIdealBranches = settings.Integrator == Algorithm::RayTracing &&
                std::any_of(scene.Materials.begin(), scene.Materials.end(), [](const Material& material) {
                    return material.Model == MaterialModel::Dielectric &&
                           material.ExteriorIndex != material.InteriorIndex;
                });
            // Each Whitted dielectric vertex can create two children. The local
            // stack is linear in depth, but the conservative ray-count bound is
            // a binary tree. The validated depth makes this 64-bit shift safe.
            const std::uint64_t maximumRaysPerSample = splitsIdealBranches
                ? (std::uint64_t{1} << (settings.MaxReflectionBounces + 1u)) - 1u
                : settings.MaxReflectionBounces + 1u;
            const std::uint64_t maximumSamples = static_cast<std::uint64_t>(settings.SamplesPerPixel) +
                (settings.Adaptive.Enabled && !settings.ProductionBudgets ? settings.Adaptive.PilotSamples : 0u);
            if (!CheckedMultiply(settings.Width, settings.Height, pixelCount) ||
                !CheckedMultiply(pixelCount, sizeof(std::uint32_t), outputBytes) ||
                !CheckedMultiply(pixelCount, maximumSamples, primaryRayCount) ||
                !CheckedMultiply(primaryRayCount, maximumRaysPerSample, maximumPathRayCount) ||
                outputBytes > std::numeric_limits<std::size_t>::max())
            {
                error = "DXR output or dispatch size overflow.";
                return false;
            }

            return immutableSceneValidated || ValidateDxrScene(scene, error);
        }

        std::string HResultMessage(const char* operation, HRESULT result)
        {
            std::ostringstream stream;
            stream << operation << " failed with HRESULT 0x" << std::hex
                << static_cast<uint32_t>(result) << '.';
            return stream.str();
        }

        std::string Narrow(const wchar_t* value)
        {
            if (!value || !*value) return {};
            const int required = WideCharToMultiByte(
                CP_UTF8, 0, value, -1, nullptr, 0, nullptr, nullptr);
            if (required <= 1) return {};
            std::string result(static_cast<size_t>(required), '\0');
            WideCharToMultiByte(
                CP_UTF8, 0, value, -1, result.data(), required, nullptr, nullptr);
            result.pop_back();
            return result;
        }

        D3D12_HEAP_PROPERTIES HeapProperties(D3D12_HEAP_TYPE type)
        {
            D3D12_HEAP_PROPERTIES properties{};
            properties.Type = type;
            properties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
            properties.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
            properties.CreationNodeMask = 1;
            properties.VisibleNodeMask = 1;
            return properties;
        }

        D3D12_RESOURCE_DESC BufferDescription(uint64_t size, D3D12_RESOURCE_FLAGS flags)
        {
            D3D12_RESOURCE_DESC description{};
            description.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
            description.Width = std::max<uint64_t>(size, 4);
            description.Height = 1;
            description.DepthOrArraySize = 1;
            description.MipLevels = 1;
            description.Format = DXGI_FORMAT_UNKNOWN;
            description.SampleDesc.Count = 1;
            description.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            description.Flags = flags;
            return description;
        }

        D3D12_RESOURCE_BARRIER TransitionBarrier(
            ID3D12Resource* resource,
            D3D12_RESOURCE_STATES before,
            D3D12_RESOURCE_STATES after)
        {
            D3D12_RESOURCE_BARRIER barrier{};
            barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barrier.Transition.pResource = resource;
            barrier.Transition.StateBefore = before;
            barrier.Transition.StateAfter = after;
            barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            return barrier;
        }

        D3D12_RESOURCE_BARRIER UavBarrier(ID3D12Resource* resource)
        {
            D3D12_RESOURCE_BARRIER barrier{};
            barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
            barrier.UAV.pResource = resource;
            return barrier;
        }

        void StoreFloat3(float destination[3], const Float3& value)
        {
            destination[0] = value.X;
            destination[1] = value.Y;
            destination[2] = value.Z;
        }

        void StoreFloat4(float destination[4], const Float3& value, float w)
        {
            destination[0] = value.X;
            destination[1] = value.Y;
            destination[2] = value.Z;
            destination[3] = w;
        }
#endif
    }

    struct DxrRayTracingRHI::Impl
    {
#if defined(_WIN32)
        struct Pipeline
        {
            ComPtr<ID3D12StateObject> StateObject;
            ComPtr<ID3D12StateObjectProperties> Properties;
            std::array<uint8_t, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES * ShaderRecordCount> ShaderIdentifiers{};
            UINT CallableCount = TraversalCallableCount;
        };
        struct SpectralSnapshot
        {
            ComPtr<ID3D12Resource> Coefficients, References;
            std::map<std::array<float, 9>, uint32_t> TriangleReferences;
            uint32_t SceneSourceCount = 0;
            MediumOptions Medium;
        };
#endif
        Impl()
        {
#if defined(_WIN32)
            Initialize();
#else
            AvailabilityError = "DXR is available only on Windows.";
#endif
        }

        ~Impl()
        {
            RHIQueue.WaitIdle();
#if defined(_WIN32)
            if (DiagnosticInfo) DiagnosticInfo->UnregisterMessageCallback(DiagnosticCookie);
#endif
        }

        RenderJob Submit(
            const Scene* scene,
            std::shared_ptr<const Scene> owner,
            Camera camera,
            RenderSettings settings,
            CancellationToken external = CancellationToken()) const
        {
            return RHIQueue.Submit(
                [this, scene, owner = std::move(owner), camera, settings, external]
                (const CancellationToken& queued) mutable
                {
                    if (!scene)
                    {
                        return Detail::MakeFailedResult("The DXR submission has no scene.");
                    }
                    const Detail::CancellationProbe cancellation =
                        Detail::CombineCancellation(queued, external);
                    if (cancellation()) return Detail::MakeCancelledResult();
                    if (!Available)
                    {
                        return Detail::MakeFailedResult(
                            std::string("DXR is unavailable: ") + AvailabilityError);
                    }
                    ++DispatchRaysSequence;
#if defined(_WIN32)
                    if (settings.Micropolygons.Enabled)
                    {
                        auto diced = Detail::PrepareMicropolygons(
                            *scene, camera, settings, cancellation,
                            [&](const auto& controls, const auto& patches, const auto& ranges, auto& words,
                                auto& error) {
                                return MicropolygonPassNative(controls, patches, ranges, words, cancellation,
                                                              error);
                            });
                        if (diced.Status != RenderStatus::Completed)
                            return diced.Status == RenderStatus::Cancelled
                                       ? Detail::MakeCancelledResult()
                                       : Detail::MakeFailedResult(diced.Error);
                        owner = std::move(diced.Geometry);
                        scene = owner.get();
                        settings.Micropolygons.Enabled = false;
                        settings.DicingPlan.reset();
                    }
                    Detail::SpatialGuideCache spatialGuides;
                    return Detail::RenderWithDenoisedSampling(
                        *scene, camera, settings, cancellation, [&](const RenderSettings &expanded)
                        { return RenderNative(*scene, owner, camera, expanded, cancellation); },
                        [&](const RenderSettings &raster, const LinearFilm &film, ReconstructedImage &image,
                            std::string &error)
                        {
                            const GeometryGuideImage *guides = nullptr;
                            if (settings.Reconstruction.GeometryGuidance)
                            {
                                guides = spatialGuides.Get(
                                    raster,
                                    [&] { return CaptureGuidesNative(*scene, camera, raster, cancellation, true); },
                                    error);
                                if (!guides)
                                    return false;
                            }
                            return ReconstructNative(film, settings, guides, cancellation, image, error);
                        },
                        [&](const auto &input, auto controls, auto &counts, auto &error)
                        { return AllocateDenoisedNative(input, controls, cancellation, counts, error); });
#else
                    return Detail::MakeFailedResult("DXR is available only on Windows.");
#endif
                });
        }

#if defined(_WIN32)
        using CreateFactoryFunction = HRESULT(WINAPI*)(UINT, REFIID, void**);
        using CreateDeviceFunction = HRESULT(WINAPI*)(IUnknown*, D3D_FEATURE_LEVEL, REFIID, void**);

        bool Initialize()
        {
            const bool gpuValidation = GetEnvironmentVariableW(L"ARVO_D3D12_GPU_VALIDATION", nullptr, 0) != 0;
            const bool diagnostics = gpuValidation ||
                GetEnvironmentVariableW(L"ARVO_D3D12_DEBUG", nullptr, 0) != 0;
            if (diagnostics)
            {
                const DWORD pathLength = GetEnvironmentVariableW(L"ARVO_DXR_PATH_LIBRARY", nullptr, 0);
                if (pathLength != 0)
                {
                    std::vector<wchar_t> path(pathLength);
                    if (GetEnvironmentVariableW(L"ARVO_DXR_PATH_LIBRARY", path.data(), pathLength) >= pathLength)
                    {
                        AvailabilityError = "The diagnostic DXR path library name changed while reading it.";
                        return false;
                    }
                    std::ifstream library(std::filesystem::path(path.data()), std::ios::binary);
                    DiagnosticPathBytecode.assign(std::istreambuf_iterator<char>(library), {});
                    if (library.bad() || DiagnosticPathBytecode.empty())
                    {
                        AvailabilityError = "The diagnostic DXR path library could not be read.";
                        return false;
                    }
                }
            }
            if (!DxgiModule.Load(L"dxgi.dll"))
            {
                AvailabilityError = "dxgi.dll could not be loaded.";
                return false;
            }
            if (!D3D12Module.Load(L"d3d12.dll"))
            {
                AvailabilityError = "d3d12.dll could not be loaded.";
                return false;
            }

            const auto createFactory = DxgiModule.Get<CreateFactoryFunction>("CreateDXGIFactory2");
            CreateDevice = D3D12Module.Get<CreateDeviceFunction>("D3D12CreateDevice");
            if (diagnostics)
            {
                using DebugInterface = HRESULT(WINAPI*)(REFIID, void**);
                const auto getDebug = D3D12Module.Get<DebugInterface>("D3D12GetDebugInterface");
                ComPtr<ID3D12Debug> debug;
                if (getDebug && SUCCEEDED(getDebug(IID_PPV_ARGS(&debug))))
                {
                    debug->EnableDebugLayer();
                    ComPtr<ID3D12Debug1> validation;
                    if (gpuValidation && SUCCEEDED(debug.As(&validation)))
                        validation->SetEnableGPUBasedValidation(TRUE);
                }
                ComPtr<ID3D12DeviceRemovedExtendedDataSettings> dred;
                if (getDebug && SUCCEEDED(getDebug(IID_PPV_ARGS(&dred))))
                {
                    dred->SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
                    dred->SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
                }
            }
            if (!createFactory || !CreateDevice)
            {
                AvailabilityError = "Required DXGI or D3D12 entry points are missing.";
                return false;
            }

            HRESULT result = createFactory(0, IID_PPV_ARGS(&Factory));
            if (FAILED(result))
            {
                AvailabilityError = HResultMessage("CreateDXGIFactory2", result);
                return false;
            }

            for (UINT adapterIndex = 0; ; ++adapterIndex)
            {
                ComPtr<IDXGIAdapter1> candidate;
                if (Factory->EnumAdapters1(adapterIndex, &candidate) == DXGI_ERROR_NOT_FOUND) break;
                DXGI_ADAPTER_DESC1 description{};
                candidate->GetDesc1(&description);
                if ((description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0) continue;

                ComPtr<ID3D12Device> baseDevice;
                result = CreateDevice(
                    candidate.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&baseDevice));
                if (FAILED(result)) continue;
                ComPtr<ID3D12Device5> rayTracingDevice;
                if (FAILED(baseDevice.As(&rayTracingDevice))) continue;

                D3D12_FEATURE_DATA_D3D12_OPTIONS5 options{};
                if (FAILED(rayTracingDevice->CheckFeatureSupport(
                    D3D12_FEATURE_D3D12_OPTIONS5, &options, sizeof(options))))
                {
                    continue;
                }
                if (options.RaytracingTier == D3D12_RAYTRACING_TIER_NOT_SUPPORTED) continue;
                D3D12_FEATURE_DATA_D3D12_OPTIONS arithmetic{};
                if (FAILED(rayTracingDevice->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS,
                    &arithmetic,sizeof(arithmetic))) || !arithmetic.DoublePrecisionFloatShaderOps) continue;

                Adapter = std::move(candidate);
                Device = std::move(rayTracingDevice);
                AdapterName = Narrow(description.Description);
                RayTracingTier = options.RaytracingTier;
                break;
            }

            if (!Device)
            {
                AvailabilityError = "No hardware adapter exposes DirectX Raytracing with double shader operations.";
                return false;
            }

            if (diagnostics && SUCCEEDED(Device.As(&DiagnosticInfo)))
                DiagnosticInfo->RegisterMessageCallback(
                    [](D3D12_MESSAGE_CATEGORY, D3D12_MESSAGE_SEVERITY severity, D3D12_MESSAGE_ID id,
                       LPCSTR message, void*) {
                        if (severity <= D3D12_MESSAGE_SEVERITY_WARNING)
                            std::fprintf(stderr, "DXR diagnostic %u: %s\n", static_cast<unsigned>(id), message);
                    }, D3D12_MESSAGE_CALLBACK_FLAG_NONE, nullptr, &DiagnosticCookie);

            D3D12_COMMAND_QUEUE_DESC queueDescription{};
            queueDescription.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
            result = Device->CreateCommandQueue(&queueDescription, IID_PPV_ARGS(&CommandQueue));
            if (FAILED(result))
            {
                AvailabilityError = HResultMessage("ID3D12Device::CreateCommandQueue", result);
                return false;
            }
            result = Device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&Fence));
            if (FAILED(result))
            {
                AvailabilityError = HResultMessage("ID3D12Device::CreateFence", result);
                return false;
            }
            FenceEvent.Handle = CreateEventW(nullptr, FALSE, FALSE, nullptr);
            if (!FenceEvent.Handle)
            {
                AvailabilityError = "The DXR fence event could not be created.";
                return false;
            }

            if (!CreateGlobalRootSignature()) return false;
            Available = true;
            return true;
        }

        bool CreateGlobalRootSignature()
        {
            const HRESULT result = Device->CreateRootSignature(
                0, Detail::DxrRootSignature, sizeof(Detail::DxrRootSignature), IID_PPV_ARGS(&RootSignature));
            if (FAILED(result))
            {
                AvailabilityError = HResultMessage("ID3D12Device::CreateRootSignature", result);
                return false;
            }
            return true;
        }

        bool CreateRayTracingStateObject(const void* shaderBytecode,
            size_t shaderBytecodeSize, Pipeline& pipeline, std::string& error,
            ID3D12RootSignature* integratorRoot = nullptr,
            IntegratorLibrary kind = IntegratorLibrary::Standard) const
        {
            const bool pathTransport = kind == IntegratorLibrary::PathTracing;
            const bool vcm = kind == IntegratorLibrary::VertexConnectionMerging;
            std::vector<D3D12_EXPORT_DESC> libraryExports = {
                {L"RayGen", nullptr, D3D12_EXPORT_FLAG_NONE},
                {L"Miss", nullptr, D3D12_EXPORT_FLAG_NONE},
                {L"ClosestHit", nullptr, D3D12_EXPORT_FLAG_NONE},
                {L"SphereClosestHit", nullptr, D3D12_EXPORT_FLAG_NONE},
                {L"SphereIntersection", nullptr, D3D12_EXPORT_FLAG_NONE},
                {L"MotionTriangleIntersection", nullptr, D3D12_EXPORT_FLAG_NONE},
                {L"TriangleTieHit", nullptr, D3D12_EXPORT_FLAG_NONE},
                {L"SphereTieHit", nullptr, D3D12_EXPORT_FLAG_NONE},
                {L"SoftwareTraversal", nullptr, D3D12_EXPORT_FLAG_NONE}};
            if (pathTransport)
                for (const auto* name : PathCallableNames)
                    libraryExports.push_back({name, nullptr, D3D12_EXPORT_FLAG_NONE});
            if (vcm)
                for (UINT stage = DxrVcm::LightStage; stage <= DxrVcm::CommitStage; ++stage)
                    libraryExports.push_back({DxrVcm::StageEntryPoints[stage], nullptr, D3D12_EXPORT_FLAG_NONE});
            D3D12_DXIL_LIBRARY_DESC libraryDescription{};
            libraryDescription.DXILLibrary.pShaderBytecode = shaderBytecode;
            libraryDescription.DXILLibrary.BytecodeLength = shaderBytecodeSize;
            libraryDescription.NumExports = static_cast<UINT>(std::size(libraryExports));
            libraryDescription.pExports = libraryExports.data();

            D3D12_HIT_GROUP_DESC hitGroup{};
            hitGroup.HitGroupExport = L"HitGroup";
            hitGroup.ClosestHitShaderImport = L"ClosestHit";
            hitGroup.AnyHitShaderImport = L"TriangleTieHit";
            hitGroup.Type = D3D12_HIT_GROUP_TYPE_TRIANGLES;

            D3D12_HIT_GROUP_DESC sphereHitGroup{};
            sphereHitGroup.HitGroupExport = L"SphereHitGroup";
            sphereHitGroup.ClosestHitShaderImport = L"SphereClosestHit";
            sphereHitGroup.AnyHitShaderImport = L"SphereTieHit";
            sphereHitGroup.IntersectionShaderImport = L"SphereIntersection";
            sphereHitGroup.Type = D3D12_HIT_GROUP_TYPE_PROCEDURAL_PRIMITIVE;

            D3D12_HIT_GROUP_DESC motionHitGroup = hitGroup;
            motionHitGroup.HitGroupExport = L"MotionTriangleHitGroup";
            motionHitGroup.IntersectionShaderImport = L"MotionTriangleIntersection";
            motionHitGroup.Type = D3D12_HIT_GROUP_TYPE_PROCEDURAL_PRIMITIVE;

            D3D12_RAYTRACING_SHADER_CONFIG shaderConfig{};
            constexpr uint32_t SpectralSurfaceAttributeBytes = sizeof(uint32_t) + sizeof(float) * 2;
            shaderConfig.MaxPayloadSizeInBytes =
                sizeof(RayQueryHit) +
                (pathTransport ? SpectralSurfaceAttributeBytes : 0);
            shaderConfig.MaxAttributeSizeInBytes = D3D12_RAYTRACING_MAX_ATTRIBUTE_SIZE_IN_BYTES;

            D3D12_GLOBAL_ROOT_SIGNATURE globalRoot{};
            globalRoot.pGlobalRootSignature = integratorRoot ? integratorRoot : RootSignature.Get();

            D3D12_RAYTRACING_PIPELINE_CONFIG pipelineConfig{};
            pipelineConfig.MaxTraceRecursionDepth = 1;

            std::array<D3D12_STATE_SUBOBJECT, 8> subobjects{};
            subobjects[0] = {D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY, &libraryDescription};
            subobjects[1] = {D3D12_STATE_SUBOBJECT_TYPE_HIT_GROUP, &hitGroup};
            subobjects[2] = {D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_SHADER_CONFIG, &shaderConfig};
            std::vector<const wchar_t*> associatedExports = {
                L"RayGen", L"Miss", L"HitGroup", L"SphereHitGroup", L"MotionTriangleHitGroup", L"SoftwareTraversal"};
            if (pathTransport)
                for (const auto* name : PathCallableNames)
                    associatedExports.push_back(name);
            if (vcm)
                for (UINT stage = DxrVcm::LightStage; stage <= DxrVcm::CommitStage; ++stage)
                    associatedExports.push_back(DxrVcm::StageEntryPoints[stage]);
            D3D12_SUBOBJECT_TO_EXPORTS_ASSOCIATION association{};
            association.pSubobjectToAssociate = &subobjects[2];
            association.NumExports = static_cast<UINT>(std::size(associatedExports));
            association.pExports = associatedExports.data();
            subobjects[3] = {D3D12_STATE_SUBOBJECT_TYPE_SUBOBJECT_TO_EXPORTS_ASSOCIATION, &association};
            subobjects[4] = {D3D12_STATE_SUBOBJECT_TYPE_GLOBAL_ROOT_SIGNATURE, &globalRoot};
            subobjects[5] = {D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_PIPELINE_CONFIG, &pipelineConfig};
            subobjects[6] = {D3D12_STATE_SUBOBJECT_TYPE_HIT_GROUP, &sphereHitGroup};
            subobjects[7] = {D3D12_STATE_SUBOBJECT_TYPE_HIT_GROUP, &motionHitGroup};

            D3D12_STATE_OBJECT_DESC description{};
            description.Type = D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE;
            description.NumSubobjects = static_cast<UINT>(subobjects.size());
            description.pSubobjects = subobjects.data();
            if (DiagnosticInfo) std::fprintf(stderr, "DXR pipeline begin: %zu bytes\n", shaderBytecodeSize);
            HRESULT result = Device->CreateStateObject(&description, IID_PPV_ARGS(&pipeline.StateObject));
            if (DiagnosticInfo) std::fprintf(stderr, "DXR pipeline end: 0x%08lx\n", static_cast<unsigned long>(result));
            if (FAILED(result))
            {
                error = HResultMessage("ID3D12Device5::CreateStateObject", result);
                return false;
            }
            result = pipeline.StateObject.As(&pipeline.Properties);
            if (FAILED(result))
            {
                error = HResultMessage("ID3D12StateObject::QueryInterface", result);
                return false;
            }

            if (DiagnosticInfo)
                std::fprintf(stderr, "DXR stack bytes: raygen=%llu callable=%llu pipeline=%llu\n",
                    static_cast<unsigned long long>(pipeline.Properties->GetShaderStackSize(L"RayGen")),
                    static_cast<unsigned long long>(pipeline.Properties->GetShaderStackSize(L"SoftwareTraversal")),
                    static_cast<unsigned long long>(pipeline.Properties->GetPipelineStackSize()));

            const void* rayGeneration = pipeline.Properties->GetShaderIdentifier(L"RayGen");
            const void* miss = pipeline.Properties->GetShaderIdentifier(L"Miss");
            const void* hit = pipeline.Properties->GetShaderIdentifier(L"HitGroup");
            const void* sphereHit = pipeline.Properties->GetShaderIdentifier(L"SphereHitGroup");
            const void* motionHit = pipeline.Properties->GetShaderIdentifier(L"MotionTriangleHitGroup");
            const void* traversal = pipeline.Properties->GetShaderIdentifier(L"SoftwareTraversal");
            if (!rayGeneration || !miss || !hit || !sphereHit || !motionHit || !traversal)
            {
                error = "The DXR state object did not expose all shader identifiers.";
                return false;
            }
            std::memcpy(pipeline.ShaderIdentifiers.data(), rayGeneration, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
            std::memcpy(
                pipeline.ShaderIdentifiers.data() + D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES,
                miss,
                D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
            std::memcpy(
                pipeline.ShaderIdentifiers.data() + D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES * 2,
                hit,
                D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
            std::memcpy(
                pipeline.ShaderIdentifiers.data() + D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES * 3,
                sphereHit, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
            std::memcpy(pipeline.ShaderIdentifiers.data() + D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES * MotionTriangleRecord,
                        motionHit, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
            std::memcpy(
                pipeline.ShaderIdentifiers.data() + D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES * TraversalCallableRecord,
                traversal, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
            if (pathTransport)
            {
                pipeline.CallableCount = PathCallableCount;
                for (UINT component = 0; component < std::size(PathCallableNames); ++component)
                {
                    const void* identifier = pipeline.Properties->GetShaderIdentifier(PathCallableNames[component]);
                    if (!identifier)
                    {
                        error = "The DXR path state object did not expose all callable components.";
                        return false;
                    }
                    const UINT record = TraversalCallableRecord + TraversalCallableCount + component;
                    std::memcpy(pipeline.ShaderIdentifiers.data() + D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES * record,
                                identifier, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
                }
            }
            if (vcm)
            {
                for (UINT stage = DxrVcm::LightStage; stage <= DxrVcm::CommitStage; ++stage)
                {
                    const void* identifier = pipeline.Properties->GetShaderIdentifier(DxrVcm::StageEntryPoints[stage]);
                    if (!identifier)
                    {
                        error = "The DXR VCM state object did not expose every dispatch stage.";
                        return false;
                    }
                    const UINT record = VcmRayGenerationRecord + stage - DxrVcm::LightStage;
                    std::memcpy(pipeline.ShaderIdentifiers.data() + D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES * record,
                                identifier, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
                }
            }
            return true;
        }

        bool EnsurePipeline(const void* bytecode, size_t size, Pipeline& pipeline, std::string& error,
                            ID3D12RootSignature* integratorRoot = nullptr,
                            IntegratorLibrary kind = IntegratorLibrary::Standard) const
        {
            // Calls run on the serialized RHI queue. Compile each native pipeline
            // on first use, keeping device discovery and the editor thread free
            // of compilation for unused integrators. Publish only a complete PSO.
            if (pipeline.StateObject) return true;
            Pipeline compiled;
            if (!CreateRayTracingStateObject(bytecode, size, compiled, error, integratorRoot, kind)) return false;
            pipeline = std::move(compiled);
            return true;
        }

        bool EnsureIntegratorPipeline(Algorithm integrator, const Pipeline*& pipeline, std::string& error) const
        {
            switch (integrator)
            {
            case Algorithm::PointBasedGlobalIllumination:
                if (!PointGiRoot)
                {
                    const HRESULT status = Device->CreateRootSignature(0, Detail::DxrPointGiRootSignature,
                        sizeof(Detail::DxrPointGiRootSignature), IID_PPV_ARGS(&PointGiRoot));
                    if (FAILED(status))
                    { error = HResultMessage("Creating the DXR Point GI root signature", status); return false; }
                }
                pipeline = &PointGiPipeline;
                return EnsurePipeline(Detail::DxrPointGiDxil, sizeof(Detail::DxrPointGiDxil), PointGiPipeline,
                                      error, PointGiRoot.Get());
            case Algorithm::RayTracing:
                pipeline = &WhittedPipeline;
                return EnsurePipeline(Detail::DxrWhittedDxil, sizeof(Detail::DxrWhittedDxil), WhittedPipeline, error);
            case Algorithm::PathTracing:
                pipeline = &PathTracingPipeline;
                // Explicit diagnostic bytecode keeps compiler experiments on the
                // same native RHI, callable exports and payload ABI. Normal jobs
                // use the embedded HLSL shared by offline rendering and preview.
                if (!DiagnosticPathBytecode.empty())
                    return EnsurePipeline(DiagnosticPathBytecode.data(), DiagnosticPathBytecode.size(),
                                          PathTracingPipeline, error, nullptr, IntegratorLibrary::PathTracing);
                return EnsurePipeline(Detail::DxrPathTracingDxil, sizeof(Detail::DxrPathTracingDxil),
                                      PathTracingPipeline, error, nullptr, IntegratorLibrary::PathTracing);
            case Algorithm::DistributedRayTracing:
                pipeline = &DistributedRayPipeline;
                return EnsurePipeline(Detail::DxrDistributedRayDxil, sizeof(Detail::DxrDistributedRayDxil), DistributedRayPipeline, error);
            case Algorithm::BidirectionalPathTracing:
                pipeline = &BidirectionalPipeline;
                return EnsurePipeline(Detail::DxrBidirectionalDxil, sizeof(Detail::DxrBidirectionalDxil), BidirectionalPipeline, error);
            case Algorithm::MetropolisLightTransport:
                if (!MetropolisRoot)
                {
                    const HRESULT result = Device->CreateRootSignature(
                        0, Detail::DxrMetropolisRootSignature, sizeof(Detail::DxrMetropolisRootSignature),
                        IID_PPV_ARGS(&MetropolisRoot));
                    if (FAILED(result))
                    {
                        error = HResultMessage("Creating the DXR MLT root signature", result);
                        return false;
                    }
                }
                pipeline = &MetropolisPipeline;
                return EnsurePipeline(Detail::DxrMetropolisDxil, sizeof(Detail::DxrMetropolisDxil),
                                      MetropolisPipeline, error, MetropolisRoot.Get());
            case Algorithm::ProgressivePhotonMapping:
                if (!ProgressiveRoot)
                {
                    const HRESULT result = Device->CreateRootSignature(
                        0, Detail::DxrPpmRootSignature, sizeof(Detail::DxrPpmRootSignature),
                        IID_PPV_ARGS(&ProgressiveRoot));
                    if (FAILED(result))
                    {
                        error = HResultMessage("Creating the DXR PPM root signature", result);
                        return false;
                    }
                }
                pipeline = &ProgressivePipeline;
                return EnsurePipeline(Detail::DxrPpmDxil, sizeof(Detail::DxrPpmDxil),
                                      ProgressivePipeline, error, ProgressiveRoot.Get());
            case Algorithm::VertexConnectionMerging:
                if (!VcmRoot)
                {
                    const HRESULT result = Device->CreateRootSignature(
                        0, Detail::DxrVcmRootSignature, sizeof(Detail::DxrVcmRootSignature), IID_PPV_ARGS(&VcmRoot));
                    if (FAILED(result))
                    {
                        error = HResultMessage("Creating the DXR VCM root signature", result);
                        return false;
                    }
                }
                pipeline = &VcmPipeline;
                return EnsurePipeline(Detail::DxrVcmDxil, sizeof(Detail::DxrVcmDxil), VcmPipeline, error,
                                      VcmRoot.Get(), IntegratorLibrary::VertexConnectionMerging);
            case Algorithm::PhotonMapping:
                if (!PhotonRoot)
                {
                    const HRESULT result = Device->CreateRootSignature(
                        0, Detail::DxrPhotonRootSignature, sizeof(Detail::DxrPhotonRootSignature),
                        IID_PPV_ARGS(&PhotonRoot));
                    if (FAILED(result))
                    {
                        error = HResultMessage("Creating the DXR photon root signature", result);
                        return false;
                    }
                }
                pipeline = &PhotonPipeline;
                return EnsurePipeline(Detail::DxrPhotonDxil, sizeof(Detail::DxrPhotonDxil),
                                      PhotonPipeline, error, PhotonRoot.Get());
            default:
                error = "No native DXR pipeline for the integrator";
                return false;
            }
        }

        bool CreateBuffer(
            uint64_t size,
            D3D12_HEAP_TYPE heapType,
            D3D12_RESOURCE_STATES initialState,
            D3D12_RESOURCE_FLAGS flags,
            ComPtr<ID3D12Resource>& resource,
            std::string& error) const
        {
            const D3D12_HEAP_PROPERTIES heap = HeapProperties(heapType);
            const D3D12_RESOURCE_DESC description = BufferDescription(size, flags);
            const HRESULT result = Device->CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &description,
                initialState,
                nullptr,
                IID_PPV_ARGS(&resource));
            if (FAILED(result))
            {
                error = HResultMessage("ID3D12Device::CreateCommittedResource", result);
                return false;
            }
            return true;
        }

        bool CreateUploadBuffer(
            const void* data,
            uint64_t size,
            ComPtr<ID3D12Resource>& resource,
            std::string& error) const
        {
            if (!CreateBuffer(
                std::max<uint64_t>(size, 4),
                D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_STATE_GENERIC_READ,
                D3D12_RESOURCE_FLAG_NONE,
                resource,
                error))
            {
                return false;
            }
            if (data && size != 0)
            {
                void* destination = nullptr;
                const D3D12_RANGE noRead{0, 0};
                const HRESULT result = resource->Map(0, &noRead, &destination);
                if (FAILED(result))
                {
                    error = HResultMessage("ID3D12Resource::Map", result);
                    return false;
                }
                std::memcpy(destination, data, static_cast<size_t>(size));
                resource->Unmap(0, nullptr);
            }
            return true;
        }

        bool ExecuteAndWait(ID3D12CommandList* commandList, std::string& error) const
        {
            ID3D12CommandList* commandLists[] = {commandList};
            CommandQueue->ExecuteCommandLists(1, commandLists);
            const uint64_t value = ++FenceValue;
            HRESULT result = CommandQueue->Signal(Fence.Get(), value);
            if (FAILED(result))
            {
                error = HResultMessage("ID3D12CommandQueue::Signal", result);
                Device->RemoveDevice();
                return false;
            }
            if (Fence->GetCompletedValue() < value)
            {
                result = Fence->SetEventOnCompletion(value, FenceEvent.Handle);
                if (FAILED(result))
                {
                    error = HResultMessage("ID3D12Fence::SetEventOnCompletion", result);
                    Device->RemoveDevice();
                    return false;
                }
                const DWORD waitResult =
                    WaitForSingleObject(FenceEvent.Handle, GpuWaitTimeoutMilliseconds);
                if (waitResult != WAIT_OBJECT_0)
                {
                    const HRESULT removedReason = Device->GetDeviceRemovedReason();
                    ReportDeviceRemoval();
                    error = FAILED(removedReason)
                        ? HResultMessage("DXR device removal", removedReason)
                        : waitResult == WAIT_TIMEOUT
                        ? "Waiting for the DXR command queue timed out."
                        : "Waiting for the DXR command queue failed.";
                    // A timeout does not end GPU access to temporary buffers.
                    // Retire this logical device before unwinding them; this
                    // does not restart the physical display adapter.
                    Device->RemoveDevice();
                    return false;
                }
            }
            result = Device->GetDeviceRemovedReason();
            if (FAILED(result))
            {
                error = HResultMessage("DXR device removal", result);
                ReportDeviceRemoval();
                return false;
            }
            return true;
        }

        void ReportDeviceRemoval() const
        {
            if (!DiagnosticInfo) return;
            ComPtr<ID3D12DeviceRemovedExtendedData> dred;
            D3D12_DRED_PAGE_FAULT_OUTPUT fault{};
            if (SUCCEEDED(Device.As(&dred)) && SUCCEEDED(dred->GetPageFaultAllocationOutput(&fault)))
                std::fprintf(stderr, "DXR page fault VA: 0x%llx\n",
                    static_cast<unsigned long long>(fault.PageFaultVA));
        }

        void AppendTriangle(
            const Float3& position0,
            const Float3& position1,
            const Float3& position2,
            const Float3& normal0,
            const Float3& normal1,
            const Float3& normal2,
            const Float3& color0,
            const Float3& color1,
            const Float3& color2,
            uint32_t materialIndex,
            bool smoothNormals,
            std::vector<GpuVertex>& vertices,
            std::vector<GpuTriangle>& triangles,
            bool hasTexCoord = false,
            const Float2& texCoord0 = Float2(),
            const Float2& texCoord1 = Float2(),
            const Float2& texCoord2 = Float2()) const
        {
            vertices.push_back({position0.X, position0.Y, position0.Z, 0.0f});
            vertices.push_back({position1.X, position1.Y, position1.Z, 0.0f});
            vertices.push_back({position2.X, position2.Y, position2.Z, 0.0f});
            GpuTriangle triangle{};
            StoreFloat4(triangle.Normal0, normal0, 0.0f);
            StoreFloat4(triangle.Normal1, normal1, 0.0f);
            StoreFloat4(triangle.Normal2, normal2, 0.0f);
            StoreFloat4(triangle.Color0, color0, 1.0f);
            StoreFloat4(triangle.Color1, color1, 1.0f);
            StoreFloat4(triangle.Color2, color2, 1.0f);
            triangle.TexCoord01[0] = texCoord0.X;
            triangle.TexCoord01[1] = texCoord0.Y;
            triangle.TexCoord01[2] = texCoord1.X;
            triangle.TexCoord01[3] = texCoord1.Y;
            triangle.TexCoord2[0] = texCoord2.X;
            triangle.TexCoord2[1] = texCoord2.Y;
            triangle.MaterialIndex = materialIndex;
            triangle.SmoothNormals = smoothNormals ? 1u : 0u;
            triangle.HasTexCoord = hasTexCoord ? 1u : 0u;
            triangles.push_back(triangle);
        }

        bool BuildGeometryData(
            const SceneGeometry& scene,
            std::vector<GpuVertex>& vertices,
            std::vector<GpuTriangle>& triangles,
            std::vector<GpuSphere>& spheres,
            std::vector<D3D12_RAYTRACING_AABB>& sphereBounds,
            std::string& error) const
        {
            if (scene.Triangles.size() > std::numeric_limits<uint32_t>::max() / 3u
                || scene.Spheres.size() >= InvalidRayPrimitiveIndex - scene.Triangles.size())
            {
                error = "DXR geometry exceeds 32-bit primitive indexing.";
                return false;
            }
            vertices.reserve(vertices.size() + scene.Triangles.size() * 3);
            triangles.reserve(triangles.size() + scene.Triangles.size());
            spheres.reserve(spheres.size() + scene.Spheres.size());
            sphereBounds.reserve(sphereBounds.size() + scene.Spheres.size());

            for (const Triangle& source : scene.Triangles)
            {
                AppendTriangle(
                    source.Position0,
                    source.Position1,
                    source.Position2,
                    source.Normal0,
                    source.Normal1,
                    source.Normal2,
                    source.Color0,
                    source.Color1,
                    source.Color2,
                    source.MaterialIndex,
                    source.SmoothNormals,
                    vertices,
                    triangles,
                    true,
                    source.TexCoord0,
                    source.TexCoord1,
                    source.TexCoord2);
            }

            for (const Sphere& source : scene.Spheres)
            {
                GpuSphere sphere{};
                StoreFloat3(sphere.Center, source.Center);
                sphere.Radius = source.Radius;
                sphere.MaterialIndex = source.MaterialIndex;
                spheres.push_back(sphere);
                // Keep procedural AABBs conservative under float rounding;
                // the analytic shader still tests the original sphere radius.
                // Expanding the radius first also protects bounds at zero;
                // merely nextafter(0) could be flushed to zero by the GPU.
                const float finiteLimit = std::numeric_limits<float>::max();
                const float expandedRadius = std::nextafter(source.Radius, finiteLimit);
                const auto outwardBound = [expandedRadius, finiteLimit](float center, bool upper)
                {
                    const double exact = static_cast<double>(center)
                        + (upper ? static_cast<double>(expandedRadius) : -static_cast<double>(expandedRadius));
                    const float rounded = static_cast<float>(std::clamp(
                        exact, -static_cast<double>(finiteLimit), static_cast<double>(finiteLimit)));
                    return std::nextafter(rounded, upper ? finiteLimit : -finiteLimit);
                };
                sphereBounds.push_back({
                    outwardBound(source.Center.X, false), outwardBound(source.Center.Y, false),
                    outwardBound(source.Center.Z, false), outwardBound(source.Center.X, true),
                    outwardBound(source.Center.Y, true), outwardBound(source.Center.Z, true)});
            }
            return true;
        }

        bool BuildSceneGeometry(const Scene& scene, const Detail::CancellationProbe& cancellation,
                                DxrInstances::Hierarchy& plan, std::vector<GpuVertex>& vertices,
                                std::vector<GpuTriangle>& triangles, std::vector<GpuSphere>& spheres,
                                std::vector<D3D12_RAYTRACING_AABB>& sphereBounds, std::string& error) const
        {
            if (scene.Instances.empty())
                return BuildGeometryData(scene, vertices, triangles, spheres, sphereBounds, error);
            if (!DxrInstances::Build(scene, cancellation, plan, error))
                return false;
            for (size_t object = 0; object < plan.Objects.size(); ++object)
            {
                if (cancellation())
                {
                    error = "DXR instance packing was cancelled.";
                    return false;
                }
                const auto& source = *plan.Objects[object];
                if (uint64_t(triangles.size()) + source.Triangles.size() >
                        std::numeric_limits<uint32_t>::max() / 3 ||
                    uint64_t(triangles.size()) + source.Triangles.size() + spheres.size() +
                            source.Spheres.size() >=
                        InvalidRayPrimitiveIndex)
                {
                    error = "DXR shared object buffers exceed native indexing.";
                    return false;
                }
                plan.Ranges[object].TriangleOffset = uint32_t(triangles.size());
                plan.Ranges[object].SphereOffset = uint32_t(spheres.size());
                if (!BuildGeometryData(source, vertices, triangles, spheres, sphereBounds, error))
                    return false;
            }
            return true;
        }

        void BuildShadingData(const Scene& scene,
            std::vector<GpuMaterial>& materials, std::vector<GpuLight>& lights) const
        {
            materials.reserve(scene.Materials.size());
            for (const Material& source : scene.Materials)
            {
                GpuMaterial material{};
                StoreFloat4(material.AlbedoReflectivity, source.Albedo, source.Reflectivity);
                StoreFloat4(material.AmbientShininess, source.AmbientColor, source.Shininess);
                StoreFloat4(material.Specular, source.SpecularColor, 0.0f);
                StoreFloat4(material.Emission, source.Emission, 0.0f);
                material.DiffuseTextureIndex = source.DiffuseTextureIndex;
                material.SampledEmitter = source.SampledEmitter ? 1u : 0u;
                material.Model = static_cast<uint32_t>(source.Model);
                material.Optical[0] = source.MicrofacetSlope;
                material.Optical[1] = source.ExteriorIndex;
                material.Optical[2] = source.InteriorIndex;
                material.Optical[3] = source.SubsurfaceMeanFreePath;
                StoreFloat4(material.ConductorIndex, source.ConductorIndex, source.SubsurfaceAnisotropy);
                StoreFloat4(material.ConductorExtinction, source.ConductorExtinction, source.SubsurfaceAlbedoScale);
                materials.push_back(material);
            }
            lights.reserve(
                scene.DirectionalLights.size()
                + scene.PointLights.size()
                + scene.RectAreaLights.size());
            for (const DirectionalLight& source : scene.DirectionalLights)
            {
                GpuLight light{};
                light.Type = DirectionalLightType;
                StoreFloat3(light.Direction, Normalize(source.DirectionToLight));
                light.Intensity = source.Intensity;
                StoreFloat3(light.Color, source.Color);
                light.CastShadows = source.CastShadows ? 1u : 0u;
                lights.push_back(light);
            }
            for (const PointLight& source : scene.PointLights)
            {
                GpuLight light{};
                StoreFloat3(light.Position, source.Position);
                light.Type = PointLightType;
                light.Intensity = source.Intensity;
                StoreFloat3(light.Color, source.Color);
                light.CastShadows = source.CastShadows ? 1u : 0u;
                lights.push_back(light);
            }
            for (const RectAreaLight& source : scene.RectAreaLights)
            {
                GpuLight light{};
                StoreFloat3(light.Position, source.Position);
                light.Type = RectAreaLightType;
                StoreFloat3(light.Direction, Normalize(source.Direction));
                light.Intensity = source.Intensity;
                StoreFloat3(light.Color, source.Color);
                light.CastShadows = source.CastShadows ? 1u : 0u;
                light.Width = source.Size.X;
                light.Height = source.Size.Y;
                lights.push_back(light);
            }
        }

        using AccelerationResources = Dxr::HardwareBvhResources;

        bool BuildAcceleration(ID3D12GraphicsCommandList4* commandList,
            uint32_t vertexCount, uint32_t sphereCount, ID3D12Resource* vertices,
            ID3D12Resource* sphereBounds, AccelerationResources& resources, std::string& error,
            const DxrInstances::Hierarchy& plan, const Detail::CancellationProbe& cancellation) const
        {
            Dxr::HardwareBvhServices services;
            services.Device = Device.Get();
            services.Cancelled = cancellation;
            services.Allocate = [this](uint64_t bytes, D3D12_HEAP_TYPE heap,
                D3D12_RESOURCE_STATES state, D3D12_RESOURCE_FLAGS flags,
                ComPtr<ID3D12Resource>& buffer, std::string& allocationError)
            {
                return CreateBuffer(bytes, heap, state, flags, buffer, allocationError);
            };
            services.Upload = [this](const void* data, uint64_t bytes,
                ComPtr<ID3D12Resource>& buffer, std::string& uploadError)
            {
                return CreateUploadBuffer(data, bytes, buffer, uploadError);
            };
            return Dxr::BuildHardwareBvh(commandList, vertexCount, sphereCount, sizeof(GpuVertex),
                vertices, sphereBounds, services, resources, error, plan.Records.empty() ? nullptr : &plan);
        }

        static constexpr uint64_t ShaderRecordSize = D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT;

        bool CreateShaderTable(const Pipeline& pipeline, ComPtr<ID3D12Resource>& table,
            std::string& error) const
        {
            std::array<uint8_t, ShaderRecordSize * ShaderRecordCount> data{};
            for (uint32_t record = 0; record < ShaderRecordCount; ++record)
                std::memcpy(data.data() + record * ShaderRecordSize,
                    pipeline.ShaderIdentifiers.data() + record * D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES,
                    D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
            return CreateUploadBuffer(data.data(), data.size(), table, error);
        }

        D3D12_DISPATCH_RAYS_DESC MakeDispatch(ID3D12Resource* table, UINT width, UINT height,
                                             UINT callableCount = TraversalCallableCount) const
        {
            D3D12_DISPATCH_RAYS_DESC dispatch{};
            const D3D12_GPU_VIRTUAL_ADDRESS address = table->GetGPUVirtualAddress();
            dispatch.RayGenerationShaderRecord = {address, ShaderRecordSize};
            dispatch.MissShaderTable = {address + ShaderRecordSize, ShaderRecordSize, ShaderRecordSize};
            dispatch.HitGroupTable = {address + ShaderRecordSize * 2, ShaderRecordSize * HitGroupRecordCount, ShaderRecordSize};
            dispatch.CallableShaderTable = {
                address + ShaderRecordSize * TraversalCallableRecord,
                ShaderRecordSize * callableCount, ShaderRecordSize};
            dispatch.Width = width;
            dispatch.Height = height;
            dispatch.Depth = 1;
            return dispatch;
        }

        struct SoftwareAccelerationResources
        {
            Float3 Minimum;
            Float3 Maximum;
            uint32_t NodeCount = 0;
            uint32_t PrimitiveIndexCount = 0;
            ComPtr<ID3D12Resource> Nodes;
            ComPtr<ID3D12Resource> PrimitiveIndices;
        };

        struct SoftwareData
        {
            Float3 Minimum, Maximum;
            uint32_t NodeCount = 0;
            std::vector<uint8_t> Nodes;
            std::vector<uint32_t> References;
        };
        template <typename Tree>
        bool PackSoftwareData(const Tree& tree, SoftwareData& data, std::string& error) const
        {
            const uint64_t bytes = tree.Nodes.size() * sizeof(tree.Nodes.front());
            if (bytes > std::numeric_limits<uint32_t>::max() ||
                tree.PrimitiveIndices.size() > std::numeric_limits<uint32_t>::max())
            {
                error = "DXR object acceleration exceeds raw buffer addressing.";
                return false;
            }
            data.Minimum = tree.Minimum;
            data.Maximum = tree.Maximum;
            data.NodeCount = uint32_t(tree.Nodes.size());
            data.Nodes.resize(size_t(bytes));
            if (bytes)
                std::memcpy(data.Nodes.data(), tree.Nodes.data(), size_t(bytes));
            data.References = tree.PrimitiveIndices;
            return true;
        }
        bool BuildSoftwareData(const SceneGeometry& geometry, AccelerationOptions options,
                               const Detail::CancellationProbe& cancellation, SoftwareData& data,
                               std::string& error) const
        {
            if (options.Type == AccelerationType::Bvh)
            {
                if (options.BvhPolicy == BvhBuildPolicy::Default)
                    return true;
                Dxr::SpatialBvhData tree;
                return Dxr::BuildSpatialBvh(geometry, cancellation, tree, error) &&
                       PackSoftwareData(tree, data, error);
            }
            if (options.Type == AccelerationType::KdTree)
            {
                Dxr::KdTreeData tree;
                return Dxr::BuildKdTree(geometry, cancellation, tree, error) &&
                       PackSoftwareData(tree, data, error);
            }
            if (options.Type == AccelerationType::Bih)
            {
                Dxr::BihData tree;
                return Dxr::BuildBih(geometry, cancellation, tree, error) &&
                       PackSoftwareData(tree, data, error);
            }
            error = "DXR acceleration has no native implementation.";
            return false;
        }
        bool PrepareSoftwareAcceleration(const Scene& scene, AccelerationOptions options,
                                         const Detail::CancellationProbe& cancellation,
                                         SoftwareAccelerationResources& resources, std::string& error,
                                         DxrInstances::Hierarchy& plan) const
        {
            SoftwareData packed;
            if (plan.Records.empty())
            {
                if (!BuildSoftwareData(scene, options, cancellation, packed, error))
                    return false;
            }
            else
            {
                for (size_t object = 0; object < plan.Objects.size(); ++object)
                {
                    if (cancellation())
                    {
                        error = "DXR instance acceleration was cancelled.";
                        return false;
                    }
                    SoftwareData local;
                    if (!BuildSoftwareData(*plan.Objects[object], options, cancellation, local, error))
                        return false;
                    if (uint64_t(packed.Nodes.size()) + local.Nodes.size() >
                            std::numeric_limits<uint32_t>::max() ||
                        uint64_t(packed.References.size()) + local.References.size() >
                            std::numeric_limits<uint32_t>::max())
                    {
                        error = "DXR instance acceleration exceeds raw buffer addressing.";
                        return false;
                    }
                    auto& range = plan.Ranges[object];
                    range.NodeBytes = uint32_t(packed.Nodes.size());
                    range.NodeCount = local.NodeCount;
                    range.ReferenceOffset = uint32_t(packed.References.size());
                    if (local.NodeCount)
                    {
                        StoreFloat3(range.Minimum, local.Minimum);
                        StoreFloat3(range.Maximum, local.Maximum);
                    }
                    packed.Nodes.insert(packed.Nodes.end(), local.Nodes.begin(), local.Nodes.end());
                    packed.References.insert(packed.References.end(), local.References.begin(),
                                             local.References.end());
                }
                for (auto& record : plan.Records)
                {
                    const uint32_t first = record.Geometry.FirstPrimitive;
                    record.Geometry = plan.Ranges[record.Geometry.ObjectIndex];
                    record.Geometry.FirstPrimitive = first;
                }
            }
            resources.Minimum = packed.Minimum;
            resources.Maximum = packed.Maximum;
            resources.NodeCount = packed.NodeCount;
            resources.PrimitiveIndexCount = uint32_t(packed.References.size());
            return CreateUploadBuffer(packed.Nodes.data(), packed.Nodes.size(), resources.Nodes, error) &&
                   CreateUploadBuffer(packed.References.data(), packed.References.size() * sizeof(uint32_t),
                                      resources.PrimitiveIndices, error);
        }

        struct InstanceResources
        {
            ComPtr<ID3D12Resource> Records, Nodes, Error, Readback, Zero;
        };
        struct ResidentScene
        {
            std::vector<GpuVertex> Vertices;
            std::vector<GpuTriangle> Triangles;
            std::vector<GpuSphere> Spheres;
            std::vector<D3D12_RAYTRACING_AABB> SphereBounds;
            std::vector<GpuMaterial> Materials;
            std::vector<GpuLight> Lights;
            Detail::PackedRayTextureAtlas Textures;
            DxrInstances::Hierarchy Instances;
            ComPtr<ID3D12Resource> VertexBuffer, TriangleBuffer, SphereBuffer, SphereBoundsBuffer;
            ComPtr<ID3D12Resource> MaterialBuffer, LightBuffer, TextureDescriptors, TextureTexels;
            AccelerationResources Acceleration;
        };
        bool PrepareInstances(const DxrInstances::Hierarchy& plan, InstanceResources& buffers,
                              ID3D12GraphicsCommandList4* commands, std::string& error) const
        {
            constexpr uint32_t ClearError = 0;
            if (!CreateUploadBuffer(plan.Records.data(), plan.Records.size() * sizeof(DxrInstances::Record),
                                    buffers.Records, error) ||
                !CreateUploadBuffer(plan.Nodes.data(), plan.Nodes.size() * sizeof(DxrInstances::Node),
                                    buffers.Nodes, error) ||
                !CreateUploadBuffer(&ClearError, sizeof(ClearError), buffers.Zero, error) ||
                !CreateBuffer(sizeof(uint32_t), D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST,
                              D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, buffers.Error, error) ||
                !CreateBuffer(sizeof(uint32_t), D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST,
                              D3D12_RESOURCE_FLAG_NONE, buffers.Readback, error))
                return false;
            commands->CopyBufferRegion(buffers.Error.Get(), 0, buffers.Zero.Get(), 0, sizeof(uint32_t));
            auto transition = TransitionBarrier(buffers.Error.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                                                D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            commands->ResourceBarrier(1, &transition);
            return true;
        }
        void BindInstances(ID3D12GraphicsCommandList4* commands, const InstanceResources& buffers) const
        {
            commands->SetComputeRootShaderResourceView(
                InstanceRecordsRootParameter, buffers.Records->GetGPUVirtualAddress());
            commands->SetComputeRootShaderResourceView(
                InstanceNodesRootParameter, buffers.Nodes->GetGPUVirtualAddress());
            commands->SetComputeRootUnorderedAccessView(
                InstanceErrorsRootParameter, buffers.Error->GetGPUVirtualAddress());
        }
        void CopyInstanceError(ID3D12GraphicsCommandList4* commands, const InstanceResources& buffers) const
        {
            const D3D12_RESOURCE_BARRIER barriers[] = {
                UavBarrier(buffers.Error.Get()),
                TransitionBarrier(buffers.Error.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                  D3D12_RESOURCE_STATE_COPY_SOURCE)};
            commands->ResourceBarrier(UINT(std::size(barriers)), barriers);
            commands->CopyBufferRegion(buffers.Readback.Get(), 0, buffers.Error.Get(), 0, sizeof(uint32_t));
        }
        bool CheckInstanceError(const InstanceResources& buffers, std::string& error) const
        {
            void* mapped = nullptr;
            const D3D12_RANGE read{0, sizeof(uint32_t)}, noWrite{0, 0};
            const HRESULT result = buffers.Readback->Map(0, &read, &mapped);
            if (FAILED(result))
            {
                error = HResultMessage("Reading DXR instance errors", result);
                return false;
            }
            uint32_t flags;
            std::memcpy(&flags, mapped, sizeof(flags));
            buffers.Readback->Unmap(0, &noWrite);
            if (!flags)
                return true;
            error = "DXR instance ray exceeds native object-space coordinate or direction range.";
            return false;
        }

        void StoreAccelerationConstants(GpuConstants& constants, AccelerationOptions acceleration,
            const SoftwareAccelerationResources& resources) const
        {
            constants.Acceleration = static_cast<uint32_t>(acceleration.Type);
            constants.BvhPolicy = static_cast<uint32_t>(acceleration.BvhPolicy);
            StoreFloat3(constants.SoftwareBoundsMinimum, resources.Minimum);
            StoreFloat3(constants.SoftwareBoundsMaximum, resources.Maximum);
            constants.SoftwareNodeCount = resources.NodeCount;
            constants.SoftwarePrimitiveIndexCount = resources.PrimitiveIndexCount;
        }

        RayQueryResult QueryNative(const Scene& scene, const std::vector<Ray>& rays,
            AccelerationOptions accelerationType, const Detail::CancellationProbe& cancellation) const
        {
            RayQueryResult query;
            if (!ValidateAcceleration(accelerationType, query.Error)
                || !ValidateRayQueries(rays, query.Error) || !ValidateDxrScene(scene, query.Error))
                return query;
            if (cancellation()) { query.Status = RenderStatus::Cancelled; return query; }
            if (rays.empty()) { query.Status = RenderStatus::Completed; return query; }
            if (!EnsurePipeline(Detail::DxrRayQueriesDxil, sizeof(Detail::DxrRayQueriesDxil), RayQueryPipeline, query.Error))
                return query;
            if (cancellation()) { query.Status = RenderStatus::Cancelled; return query; }

            std::vector<GpuVertex> vertices;
            std::vector<GpuTriangle> triangles;
            std::vector<GpuSphere> spheres;
            std::vector<D3D12_RAYTRACING_AABB> sphereBounds;
            DxrInstances::Hierarchy instancePlan;
            if (!BuildSceneGeometry(scene,cancellation,instancePlan,vertices,triangles,spheres,sphereBounds,query.Error))
            {
                if (cancellation()) { query.Status = RenderStatus::Cancelled; query.Error.clear(); }
                return query;
            }
            std::vector<GpuQueryRay> packedRays(rays.size());
            for (size_t index = 0; index < rays.size(); ++index)
            {
                StoreFloat3(packedRays[index].Origin, rays[index].Origin);
                StoreFloat3(packedRays[index].Direction, rays[index].Direction);
                packedRays[index].MinimumDistance = rays[index].MinDistance;
                packedRays[index].MaximumDistance = rays[index].MaxDistance;
            }
            ComPtr<ID3D12Resource> vertexBuffer;
            ComPtr<ID3D12Resource> triangleBuffer;
            ComPtr<ID3D12Resource> sphereBuffer;
            ComPtr<ID3D12Resource> sphereBoundsBuffer;
            ComPtr<ID3D12Resource> rayBuffer;
            ComPtr<ID3D12Resource> hitBuffer;
            ComPtr<ID3D12Resource> readback;
            const uint64_t outputBytes = rays.size() * sizeof(RayQueryHit);
            if (!CreateUploadBuffer(vertices.data(), vertices.size() * sizeof(GpuVertex), vertexBuffer, query.Error)
                || !CreateUploadBuffer(triangles.data(), triangles.size() * sizeof(GpuTriangle), triangleBuffer, query.Error)
                || !CreateUploadBuffer(spheres.data(), spheres.size() * sizeof(GpuSphere), sphereBuffer, query.Error)
                || !CreateUploadBuffer(sphereBounds.data(), sphereBounds.size() * sizeof(D3D12_RAYTRACING_AABB),
                    sphereBoundsBuffer, query.Error)
                || !CreateUploadBuffer(packedRays.data(), packedRays.size() * sizeof(GpuQueryRay), rayBuffer, query.Error)
                || !CreateBuffer(outputBytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, hitBuffer, query.Error)
                || !CreateBuffer(outputBytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_FLAG_NONE, readback, query.Error)) return query;

            ComPtr<ID3D12CommandAllocator> allocator;
            ComPtr<ID3D12GraphicsCommandList> baseCommandList;
            ComPtr<ID3D12GraphicsCommandList4> commandList;
            HRESULT result = Device->CreateCommandAllocator(
                D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator));
            if (SUCCEEDED(result)) result = Device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                allocator.Get(), nullptr, IID_PPV_ARGS(&baseCommandList));
            if (SUCCEEDED(result)) result = baseCommandList.As(&commandList);
            if (FAILED(result))
            {
                query.Error = HResultMessage("Creating the DXR query command list", result);
                return query;
            }
            AccelerationResources acceleration;
            SoftwareAccelerationResources software;
            if (!PrepareSoftwareAcceleration(scene, accelerationType, cancellation, software, query.Error, instancePlan))
            {
                if (cancellation()) query.Status = RenderStatus::Cancelled;
                return query;
            }
            InstanceResources instanceBuffers;
            if (!PrepareInstances(instancePlan,instanceBuffers,commandList.Get(),query.Error)) return query;
            if (accelerationType.Type == AccelerationType::Bvh
                && accelerationType.BvhPolicy == BvhBuildPolicy::Default
                && !BuildAcceleration(commandList.Get(), static_cast<uint32_t>(vertices.size()),
                static_cast<uint32_t>(spheres.size()), vertexBuffer.Get(), sphereBoundsBuffer.Get(),
                acceleration, query.Error, instancePlan, cancellation))
            {
                if (cancellation()) { query.Status = RenderStatus::Cancelled; query.Error.clear(); }
                return query;
            }
            result = commandList->Close();
            if (FAILED(result))
            {
                query.Error = HResultMessage("Closing the DXR query build command list", result);
                return query;
            }
            if (!ExecuteAndWait(commandList.Get(), query.Error)) return query;
            if (cancellation()) { query.Status = RenderStatus::Cancelled; return query; }

            GpuConstants constants{};
            constants.InstanceCount = uint32_t(instancePlan.Records.size());
            constants.InstanceNodeCount = uint32_t(instancePlan.Nodes.size());
            StoreAccelerationConstants(constants, accelerationType, software);
            constants.TriangleCount = static_cast<uint32_t>(triangles.size());
            constants.SphereCount = static_cast<uint32_t>(spheres.size());
            constants.PrimitiveCount = constants.TriangleCount + constants.SphereCount;
            std::array<uint8_t, D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT> constantData{};
            std::memcpy(constantData.data(), &constants, sizeof(constants));
            ComPtr<ID3D12Resource> constantBuffer;
            ComPtr<ID3D12Resource> shaderTable;
            if (!CreateUploadBuffer(constantData.data(), constantData.size(), constantBuffer, query.Error)
                || !CreateShaderTable(RayQueryPipeline, shaderTable, query.Error)) return query;
            result = allocator->Reset();
            if (SUCCEEDED(result)) result = commandList->Reset(allocator.Get(), nullptr);
            if (FAILED(result))
            {
                query.Error = HResultMessage("Resetting the DXR query command list", result);
                return query;
            }
            commandList->SetPipelineState1(RayQueryPipeline.StateObject.Get());
            commandList->SetComputeRootSignature(RootSignature.Get());
            BindInstances(commandList.Get(),instanceBuffers);
            commandList->SetComputeRootShaderResourceView(0,
                acceleration.TopLevel ? acceleration.TopLevel->GetGPUVirtualAddress() : 0);
            commandList->SetComputeRootShaderResourceView(3, vertexBuffer->GetGPUVirtualAddress());
            commandList->SetComputeRootShaderResourceView(4, triangleBuffer->GetGPUVirtualAddress());
            commandList->SetComputeRootConstantBufferView(9, constantBuffer->GetGPUVirtualAddress());
            commandList->SetComputeRootShaderResourceView(11, sphereBuffer->GetGPUVirtualAddress());
            commandList->SetComputeRootShaderResourceView(12, rayBuffer->GetGPUVirtualAddress());
            commandList->SetComputeRootUnorderedAccessView(13, hitBuffer->GetGPUVirtualAddress());
            commandList->SetComputeRootShaderResourceView(14, software.Nodes->GetGPUVirtualAddress());
            commandList->SetComputeRootShaderResourceView(15, software.PrimitiveIndices->GetGPUVirtualAddress());
            const D3D12_DISPATCH_RAYS_DESC dispatch =
                MakeDispatch(shaderTable.Get(), static_cast<UINT>(rays.size()), 1);
            commandList->DispatchRays(&dispatch);
            CopyInstanceError(commandList.Get(),instanceBuffers);
            const D3D12_RESOURCE_BARRIER barriers[] = {UavBarrier(hitBuffer.Get()),
                TransitionBarrier(hitBuffer.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COPY_SOURCE)};
            commandList->ResourceBarrier(static_cast<UINT>(std::size(barriers)), barriers);
            commandList->CopyResource(readback.Get(), hitBuffer.Get());
            result = commandList->Close();
            if (FAILED(result))
            {
                query.Error = HResultMessage("Closing the DXR query dispatch command list", result);
                return query;
            }
            if (!ExecuteAndWait(commandList.Get(), query.Error)) return query;
            if (cancellation()) { query.Status = RenderStatus::Cancelled; return query; }
            if (!CheckInstanceError(instanceBuffers,query.Error)) return query;
            void* data = nullptr;
            const D3D12_RANGE range{0, static_cast<SIZE_T>(outputBytes)};
            result = readback->Map(0, &range, &data);
            if (FAILED(result))
            {
                query.Error = HResultMessage("Reading DXR ray-query hits", result);
                return query;
            }
            query.Hits.resize(rays.size());
            std::memcpy(query.Hits.data(), data, static_cast<size_t>(outputBytes));
            const D3D12_RANGE noWrite{0, 0};
            readback->Unmap(0, &noWrite);
            query.Status = RenderStatus::Completed;
            return query;
        }

        bool DispatchBidirectional(const PreparedEmissionScene& emission, const GpuConstants& constants,
                                   ID3D12CommandAllocator* allocator, ID3D12GraphicsCommandList4* commands,
                                   ID3D12Resource* shaderTable, const InstanceResources& instances,
                                   const std::function<void()>& bindScene,
                                   const Detail::CancellationProbe& cancellation, std::string& error) const
        {
            const uint32_t fullPixels = constants.FullWidth * constants.FullHeight;
            const uint32_t capacity = std::min(DxrBdpt::BatchCapacity, fullPixels);
            const uint64_t vertexBytes = uint64_t(capacity) * DxrBdpt::SubpathsPerPair *
                                         DxrBdpt::MaximumVertices * sizeof(DxrBdpt::Vertex);
            constexpr uint64_t PassPixelBytes = DxrBdpt::PassPixelBytes;
            const uint64_t passBytes = uint64_t(constants.Width) * constants.Height * PassPixelBytes;
            ComPtr<ID3D12Resource> sources, vertices, pass, failures, readback;
            if (!CreateUploadBuffer(emission.Sources.data(), emission.Sources.size() * sizeof(PackedEmitter),
                                    sources, error) ||
                !CreateBuffer(vertexBytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                              D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, vertices, error) ||
                !CreateBuffer(passBytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                              D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, pass, error) ||
                !CreateBuffer(sizeof(uint32_t), D3D12_HEAP_TYPE_DEFAULT,
                              D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                              D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, failures, error) ||
                !CreateBuffer(sizeof(uint32_t), D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST,
                              D3D12_RESOURCE_FLAG_NONE, readback, error))
                return false;
            DxrBdpt::Controls controls{{emission.Center.X, emission.Center.Y, emission.Center.Z},
                                       emission.Radius,
                                       uint32_t(emission.Sources.size()),
                                       0,
                                       0,
                                       DxrBdpt::ClearPass};
            auto reset = [&] {
                HRESULT result = allocator->Reset();
                if (SUCCEEDED(result))
                    result = commands->Reset(allocator, nullptr);
                if (FAILED(result))
                {
                    error = HResultMessage("Resetting DXR BDPT commands", result);
                    return false;
                }
                bindScene();
                return true;
            };
            auto dispatch = [&](uint32_t stage, uint32_t width, uint32_t height) {
                controls.Stage = stage;
                commands->SetComputeRootShaderResourceView(DxrBdpt::EmitterRoot,
                                                           sources->GetGPUVirtualAddress());
                commands->SetComputeRootUnorderedAccessView(DxrBdpt::VerticesRoot,
                                                            vertices->GetGPUVirtualAddress());
                commands->SetComputeRootUnorderedAccessView(DxrBdpt::PassRoot, pass->GetGPUVirtualAddress());
                commands->SetComputeRootUnorderedAccessView(DxrBdpt::ErrorRoot,
                                                            failures->GetGPUVirtualAddress());
                commands->SetComputeRoot32BitConstants(DxrBdpt::ControlsRoot,
                                                       sizeof(controls) / sizeof(uint32_t), &controls, 0);
                auto description = MakeDispatch(shaderTable, width, height);
                commands->DispatchRays(&description);
                // Publish splats, path scratch, film and counters before the next
                // stage. Each trace dispatch owns disjoint path scratch records.
                const D3D12_RESOURCE_BARRIER barriers[] = {
                    UavBarrier(nullptr),
                    TransitionBarrier(failures.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                      D3D12_RESOURCE_STATE_COPY_SOURCE)};
                commands->ResourceBarrier(UINT(std::size(barriers)), barriers);
                commands->CopyResource(readback.Get(), failures.Get());
                const auto restoreError = TransitionBarrier(failures.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
                                                            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                commands->ResourceBarrier(1, &restoreError);
                CopyInstanceError(commands, instances);
                const auto restoreInstances =
                    TransitionBarrier(instances.Error.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
                                      D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                commands->ResourceBarrier(1, &restoreInstances);
                HRESULT result = commands->Close();
                if (FAILED(result))
                {
                    error = HResultMessage("Closing DXR BDPT commands", result);
                    return false;
                }
                if (!ExecuteAndWait(commands, error) || !CheckInstanceError(instances, error))
                    return false;
                void* mapped = nullptr;
                const D3D12_RANGE read{0, sizeof(uint32_t)}, noWrite{0, 0};
                result = readback->Map(0, &read, &mapped);
                if (FAILED(result))
                {
                    error = HResultMessage("Mapping DXR BDPT numerical status", result);
                    return false;
                }
                uint32_t status;
                std::memcpy(&status, mapped, sizeof(status));
                readback->Unmap(0, &noWrite);
                if (status)
                {
                    error = "DXR BDPT encountered nonfinite path or film arithmetic.";
                    return false;
                }
                return reset();
            };
            // Cropped jobs still launch every global light path. The sensor walk
            // is omitted outside the tile, but incoming lens splats are retained.
            // No partly accumulated pass is exposed when cancellation interrupts.
            for (uint32_t sample = 0; sample < constants.SamplesPerPixel; ++sample)
            {
                if (cancellation())
                    return false;
                controls.SampleIndex = sample;
                if (!dispatch(DxrBdpt::ClearPass, constants.Width, constants.Height))
                    return false;
                for (uint32_t begin = 0; begin < fullPixels; begin += capacity)
                {
                    if (cancellation())
                        return false;
                    controls.FirstPath = begin;
                    if (!dispatch(DxrBdpt::ConnectPaths, std::min(capacity, fullPixels - begin), 1))
                        return false;
                }
                if (cancellation())
                    return false;
                if (!dispatch(DxrBdpt::CommitPass, constants.Width, constants.Height))
                    return false;
            }
            return true;
        }

        bool DispatchMetropolis(const PreparedEmissionScene& emission, const GpuConstants& constants,
                                const MetropolisOptions& options, ID3D12CommandAllocator* allocator,
                                ID3D12GraphicsCommandList4* commands, ID3D12Resource* shaderTable,
                                const InstanceResources& instances, const std::function<void()>& bindScene,
                                const Detail::CancellationProbe& cancellation, std::string& error) const
        {
            const uint32_t fullPixels = constants.FullWidth * constants.FullHeight;
            const uint32_t chains = std::min(options.ChainCount, fullPixels);
            constexpr uint32_t PassPixelBytes = sizeof(float) * 4;
            const uint64_t passBytes = uint64_t(constants.Width) * constants.Height * PassPixelBytes;
            ComPtr<ID3D12Resource> emitters, arena, pass, failures, readback;
            if (!CreateUploadBuffer(emission.Sources.data(), emission.Sources.size() * sizeof(PackedEmitter),
                                    emitters, error) ||
                !CreateBuffer(DxrMlt::ArenaBytes(chains, options.BootstrapSamples), D3D12_HEAP_TYPE_DEFAULT,
                              D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                              D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, arena, error) ||
                !CreateBuffer(passBytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                              D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, pass, error) ||
                !CreateBuffer(sizeof(uint32_t), D3D12_HEAP_TYPE_DEFAULT,
                              D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                              D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, failures, error) ||
                !CreateBuffer(sizeof(uint32_t), D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST,
                              D3D12_RESOURCE_FLAG_NONE, readback, error))
                return false;
            DxrMlt::Controls controls{};
            controls.Center[0] = emission.Center.X;
            controls.Center[1] = emission.Center.Y;
            controls.Center[2] = emission.Center.Z;
            controls.Radius = emission.Radius;
            controls.EmitterCount = uint32_t(emission.Sources.size());
            controls.BootstrapCount = options.BootstrapSamples;
            controls.ChainCount = chains;
            controls.Seed = options.Seed ^ constants.SamplingSeed;
            controls.LargeStepProbability = options.LargeStepProbability;
            controls.MutationScale = options.MutationScale;
            auto reset = [&] {
                HRESULT result = allocator->Reset();
                if (SUCCEEDED(result))
                    result = commands->Reset(allocator, nullptr);
                if (FAILED(result))
                {
                    error = HResultMessage("Resetting DXR MLT commands", result);
                    return false;
                }
                bindScene();
                return true;
            };
            auto dispatch = [&](uint32_t stage, uint32_t width, uint32_t height) {
                if (cancellation())
                    return false;
                controls.Stage = stage;
                commands->SetComputeRootShaderResourceView(DxrMlt::EmitterRoot,
                                                           emitters->GetGPUVirtualAddress());
                commands->SetComputeRootUnorderedAccessView(DxrMlt::ArenaRoot, arena->GetGPUVirtualAddress());
                commands->SetComputeRootUnorderedAccessView(DxrMlt::PassRoot, pass->GetGPUVirtualAddress());
                commands->SetComputeRootUnorderedAccessView(DxrMlt::ErrorRoot,
                                                            failures->GetGPUVirtualAddress());
                commands->SetComputeRoot32BitConstants(DxrMlt::ControlsRoot,
                                                       sizeof(controls) / sizeof(uint32_t), &controls, 0);
                auto description = MakeDispatch(shaderTable, width, height);
                commands->DispatchRays(&description);
                // The arena contains disjoint chain records. A global UAV barrier
                // publishes both chain state and colliding film splats between stages.
                const D3D12_RESOURCE_BARRIER barriers[] = {
                    UavBarrier(nullptr),
                    TransitionBarrier(failures.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                      D3D12_RESOURCE_STATE_COPY_SOURCE)};
                commands->ResourceBarrier(UINT(std::size(barriers)), barriers);
                commands->CopyResource(readback.Get(), failures.Get());
                const auto restoreError = TransitionBarrier(failures.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
                                                            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                commands->ResourceBarrier(1, &restoreError);
                CopyInstanceError(commands, instances);
                const auto restoreInstances =
                    TransitionBarrier(instances.Error.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
                                      D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                commands->ResourceBarrier(1, &restoreInstances);
                HRESULT result = commands->Close();
                if (FAILED(result))
                {
                    error = HResultMessage("Closing DXR MLT commands", result);
                    return false;
                }
                if (!ExecuteAndWait(commands, error) || !CheckInstanceError(instances, error))
                    return false;
                void* mapped = nullptr;
                const D3D12_RANGE read{0, sizeof(uint32_t)}, noWrite{0, 0};
                result = readback->Map(0, &read, &mapped);
                if (FAILED(result))
                {
                    error = HResultMessage("Mapping DXR MLT numerical status", result);
                    return false;
                }
                uint32_t status;
                std::memcpy(&status, mapped, sizeof(status));
                readback->Unmap(0, &noWrite);
                if (status)
                {
                    error = "DXR MLT encountered invalid path, state or film arithmetic.";
                    return false;
                }
                return reset();
            };
            if (!dispatch(DxrMlt::Clear, constants.Width, constants.Height))
                return false;
            for (uint32_t first = 0; first < options.BootstrapSamples; first += DxrMlt::BootstrapBatch)
            {
                controls.Begin = first;
                controls.Count = std::min(DxrMlt::BootstrapBatch, options.BootstrapSamples - first);
                if (!dispatch(DxrMlt::Bootstrap, controls.Count, 1))
                    return false;
            }
            if (!dispatch(DxrMlt::Distribution, 1, 1) || !dispatch(DxrMlt::Initialize, chains, 1))
                return false;
            // Every crop replays the global bootstrap and chains, then accepts
            // only splats landing in its film. A sample is one complete mutation
            // pass, not one deposit; Welford moments include zero-valued pixels.
            const uint32_t maximumChainSteps = fullPixels / chains + (fullPixels % chains != 0);
            for (uint32_t sample = 0; sample < constants.SamplesPerPixel; ++sample)
            {
                controls.PassIndex = sample;
                if (sample && !dispatch(DxrMlt::Clear, constants.Width, constants.Height))
                    return false;
                for (uint32_t first = 0; first < maximumChainSteps; first += DxrMlt::MutationBatch)
                {
                    controls.Begin = first;
                    controls.Count = std::min(DxrMlt::MutationBatch, maximumChainSteps - first);
                    if (!dispatch(DxrMlt::Mutate, chains, 1))
                        return false;
                }
                if (!dispatch(DxrMlt::Commit, constants.Width, constants.Height))
                    return false;
            }
            return true;
        }

        bool EnsureMicropolygons(std::string& error) const
        {
            if (DicingState)
                return true;
            ComPtr<ID3D12RootSignature> root;
            HRESULT status = Device->CreateRootSignature(0, Detail::DxrMicropolygonRootSignature,
                                                         sizeof(Detail::DxrMicropolygonRootSignature),
                                                         IID_PPV_ARGS(&root));
            if (FAILED(status))
            {
                error = HResultMessage("Creating DXR dicing root", status);
                return false;
            }
            const D3D12_EXPORT_DESC entry{L"MicropolygonRayGeneration", nullptr, D3D12_EXPORT_FLAG_NONE};
            const D3D12_DXIL_LIBRARY_DESC library{
                {Detail::DxrMicropolygonDxil, sizeof(Detail::DxrMicropolygonDxil)}, 1, &entry};
            const D3D12_RAYTRACING_SHADER_CONFIG shader{0, 0};
            const D3D12_GLOBAL_ROOT_SIGNATURE global{root.Get()};
            const D3D12_RAYTRACING_PIPELINE_CONFIG recursion{1};
            const D3D12_STATE_SUBOBJECT parts[]{
                {D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY, &library},
                {D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_SHADER_CONFIG, &shader},
                {D3D12_STATE_SUBOBJECT_TYPE_GLOBAL_ROOT_SIGNATURE, &global},
                {D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_PIPELINE_CONFIG, &recursion}};
            const D3D12_STATE_OBJECT_DESC description{D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE,
                                                      UINT(std::size(parts)), parts};
            ComPtr<ID3D12StateObject> state;
            status = Device->CreateStateObject(&description, IID_PPV_ARGS(&state));
            if (FAILED(status))
            {
                error = HResultMessage("Creating DXR dicing state", status);
                return false;
            }
            ComPtr<ID3D12StateObjectProperties> properties;
            status = state.As(&properties);
            if (FAILED(status))
            {
                error = HResultMessage("Inspecting DXR dicing state", status);
                return false;
            }
            const void* identifier = properties->GetShaderIdentifier(entry.Name);
            if (!identifier)
            {
                error = "DXR dicing shader identifier is unavailable";
                return false;
            }
            std::array<uint8_t, ShaderRecordSize> record{};
            std::memcpy(record.data(), identifier, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
            ComPtr<ID3D12Resource> table;
            if (!CreateUploadBuffer(record.data(), record.size(), table, error))
                return false;
            DicingRoot = std::move(root);
            DicingState = std::move(state);
            DicingTable = std::move(table);
            return true;
        }
        bool MicropolygonPassNative(MicropolygonData::Controls controls,
                                    const std::vector<MicropolygonData::Patch>& patches,
                                    const std::vector<MicropolygonData::Range>& ranges,
                                    std::vector<uint32_t>& words, const Detail::CancellationProbe& cancelled,
                                    std::string& error) const
        {
            namespace M = MicropolygonData;
            if (!EnsureMicropolygons(error))
                return false;
            const M::Range empty{};
            const uint64_t bytes = words.size() * sizeof(uint32_t);
            ComPtr<ID3D12Resource> input, layout, output, clear, readback;
            // The plan pass uses atomic maximum; start from zero every time.
            if (!CreateUploadBuffer(patches.data(), patches.size() * sizeof(M::Patch), input, error) ||
                !CreateUploadBuffer(ranges.empty() ? &empty : ranges.data(),
                                    std::max(size_t(1), ranges.size()) * sizeof(M::Range), layout, error) ||
                !CreateBuffer(bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST,
                              D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, output, error) ||
                !CreateUploadBuffer(words.data(), size_t(bytes), clear, error) ||
                !CreateBuffer(bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST,
                              D3D12_RESOURCE_FLAG_NONE, readback, error))
                return false;
            ComPtr<ID3D12CommandAllocator> allocator;
            ComPtr<ID3D12GraphicsCommandList4> commands;
            HRESULT status =
                Device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator));
            if (SUCCEEDED(status))
                status = Device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(),
                                                   nullptr, IID_PPV_ARGS(&commands));
            if (FAILED(status))
            {
                error = HResultMessage("Creating DXR dicing commands", status);
                return false;
            }
            const auto finish = [&] {
                const HRESULT hr = commands->Close();
                if (FAILED(hr))
                {
                    error = HResultMessage("Closing DXR dicing commands", hr);
                    return false;
                }
                return ExecuteAndWait(commands.Get(), error);
            };
            const auto begin = [&] {
                HRESULT hr = allocator->Reset();
                if (SUCCEEDED(hr))
                    hr = commands->Reset(allocator.Get(), nullptr);
                if (FAILED(hr))
                {
                    error = HResultMessage("Resetting DXR dicing commands", hr);
                    return false;
                }
                return true;
            };
            commands->CopyBufferRegion(output.Get(), 0, clear.Get(), 0, bytes);
            const auto initial = TransitionBarrier(output.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                                                   D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            commands->ResourceBarrier(1, &initial);
            if (cancelled() || !finish())
                return false;
            constexpr uint32_t BatchItems = 16384;
            const uint32_t population = controls.Dispatch.Y;
            for (controls.Dispatch.X = 0; controls.Dispatch.X < population; controls.Dispatch.X += BatchItems)
            {
                if (cancelled() || !begin())
                    return false;
                controls.Dispatch.Y = std::min(BatchItems, population - controls.Dispatch.X);
                commands->SetComputeRootSignature(DicingRoot.Get());
                commands->SetComputeRoot32BitConstants(0, sizeof(controls) / sizeof(uint32_t), &controls, 0);
                commands->SetComputeRootShaderResourceView(1, input->GetGPUVirtualAddress());
                commands->SetComputeRootShaderResourceView(2, layout->GetGPUVirtualAddress());
                commands->SetComputeRootUnorderedAccessView(3, output->GetGPUVirtualAddress());
                commands->SetPipelineState1(DicingState.Get());
                D3D12_DISPATCH_RAYS_DESC dispatch{};
                dispatch.RayGenerationShaderRecord = {DicingTable->GetGPUVirtualAddress(), ShaderRecordSize};
                dispatch.Width = controls.Dispatch.Y;
                dispatch.Height = dispatch.Depth = 1;
                commands->DispatchRays(&dispatch);
                const auto barrier = UavBarrier(output.Get());
                commands->ResourceBarrier(1, &barrier);
                if (!finish())
                    return false;
            }
            if (cancelled() || !begin())
                return false;
            const auto ready = TransitionBarrier(output.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                                 D3D12_RESOURCE_STATE_COPY_SOURCE);
            commands->ResourceBarrier(1, &ready);
            commands->CopyBufferRegion(readback.Get(), 0, output.Get(), 0, bytes);
            if (!finish())
                return false;
            void* mapped = nullptr;
            const D3D12_RANGE read{0, size_t(bytes)}, noWrite{0, 0};
            status = readback->Map(0, &read, &mapped);
            if (FAILED(status))
            {
                error = HResultMessage("Reading DXR micropolygons", status);
                return false;
            }
            std::memcpy(words.data(), mapped, size_t(bytes));
            readback->Unmap(0, &noWrite);
            return !cancelled();
        }

        bool EnsureTemporalReconstruction(std::string& error) const
        {
            if (TemporalState)
                return true;
            ComPtr<ID3D12RootSignature> root;
            HRESULT status = Device->CreateRootSignature(0, Detail::DxrTemporalDenoiseRootSignature,
                                                         sizeof(Detail::DxrTemporalDenoiseRootSignature),
                                                         IID_PPV_ARGS(&root));
            if (FAILED(status))
            {
                error = HResultMessage("Creating DXR temporal root", status);
                return false;
            }
            const D3D12_EXPORT_DESC entry{L"TemporalDenoiseRayGen", nullptr, D3D12_EXPORT_FLAG_NONE};
            const D3D12_DXIL_LIBRARY_DESC library{
                {Detail::DxrTemporalDenoiseDxil, sizeof(Detail::DxrTemporalDenoiseDxil)}, 1, &entry};
            const D3D12_RAYTRACING_SHADER_CONFIG shader{0, 0};
            const D3D12_GLOBAL_ROOT_SIGNATURE global{root.Get()};
            const D3D12_RAYTRACING_PIPELINE_CONFIG recursion{1};
            const D3D12_STATE_SUBOBJECT parts[] = {
                {D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY, &library},
                {D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_SHADER_CONFIG, &shader},
                {D3D12_STATE_SUBOBJECT_TYPE_GLOBAL_ROOT_SIGNATURE, &global},
                {D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_PIPELINE_CONFIG, &recursion}};
            const D3D12_STATE_OBJECT_DESC description{D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE,
                                                      UINT(std::size(parts)), parts};
            ComPtr<ID3D12StateObject> state;
            status = Device->CreateStateObject(&description, IID_PPV_ARGS(&state));
            if (FAILED(status))
            {
                error = HResultMessage("Creating DXR temporal pipeline", status);
                return false;
            }
            ComPtr<ID3D12StateObjectProperties> properties;
            status = state.As(&properties);
            if (FAILED(status))
            {
                error = HResultMessage("Reading DXR temporal state", status);
                return false;
            }
            const void* identifier = properties->GetShaderIdentifier(entry.Name);
            if (!identifier)
            {
                error = "DXR temporal shader identifier is unavailable";
                return false;
            }
            std::array<uint8_t, ShaderRecordSize> record{};
            std::memcpy(record.data(), identifier, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
            ComPtr<ID3D12Resource> table;
            if (!CreateUploadBuffer(record.data(), record.size(), table, error))
                return false;
            TemporalRoot = std::move(root);
            TemporalTable = std::move(table);
            TemporalState = std::move(state);
            return true;
        }
        bool ReconstructTemporalNative(
            TemporalData::Controls controls, const std::vector<TemporalPixel>& current,
            const std::vector<GeometryGuide>& guides, const std::vector<TemporalPixel>& previous,
            const std::vector<GeometryGuide>& oldGuides, const Detail::CancellationProbe& cancelled,
            std::vector<TemporalPixel>& result, Image& display, std::string& error) const
        {
            if (!EnsureTemporalReconstruction(error))
                return false;
            constexpr uint32_t InputCount = 4, OutputCount = 3, ColorOutput = 0, DisplayOutput = 1,
                               ErrorOutput = 2;
            constexpr UINT InputRoot = 1, OutputRoot = InputRoot + InputCount;
            const uint32_t count = controls.Width * controls.Height;
            const uint64_t colorBytes = uint64_t(count) * sizeof(TemporalPixel),
                           displayBytes = uint64_t(count) * sizeof(uint32_t);
            const uint64_t statusOffset = colorBytes + displayBytes;
            ComPtr<IDXGIAdapter3> memoryAdapter;
            if (SUCCEEDED(Adapter.As(&memoryAdapter)))
            {
                DXGI_QUERY_VIDEO_MEMORY_INFO budget{};
                if (SUCCEEDED(
                        memoryAdapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &budget)))
                {
                    const uint64_t available =
                        budget.Budget > budget.CurrentUsage ? budget.Budget - budget.CurrentUsage : 0;
                    if (statusOffset + sizeof(uint32_t) > available)
                    {
                        error = "DXR temporal output exceeds video-memory budget";
                        return false;
                    }
                }
            }
            const TemporalPixel emptyColor{};
            const GeometryGuide emptyGuide{};
            const void* data[] = {current.data(), guides.data(),
                                  previous.empty() ? &emptyColor : previous.data(),
                                  oldGuides.empty() ? &emptyGuide : oldGuides.data()};
            const uint64_t bytes[] = {
                current.size() * sizeof(current[0]), guides.size() * sizeof(guides[0]),
                previous.empty() ? sizeof(emptyColor) : previous.size() * sizeof(previous[0]),
                oldGuides.empty() ? sizeof(emptyGuide) : oldGuides.size() * sizeof(oldGuides[0])};
            ComPtr<ID3D12Resource> inputs[InputCount], outputs[OutputCount], clear, readback;
            for (uint32_t i = 0; i < InputCount; ++i)
                if (!CreateUploadBuffer(data[i], bytes[i], inputs[i], error))
                    return false;
            const uint64_t sizes[] = {colorBytes, displayBytes, sizeof(uint32_t)};
            for (uint32_t i = 0; i < OutputCount; ++i)
                if (!CreateBuffer(sizes[i], D3D12_HEAP_TYPE_DEFAULT,
                                  i == ErrorOutput ? D3D12_RESOURCE_STATE_COPY_DEST
                                                   : D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                  D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, outputs[i], error))
                    return false;
            const uint32_t zero = 0;
            if (!CreateUploadBuffer(&zero, sizeof(zero), clear, error) ||
                !CreateBuffer(statusOffset + sizeof(zero), D3D12_HEAP_TYPE_READBACK,
                              D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE, readback, error))
                return false;
            ComPtr<ID3D12CommandAllocator> allocator;
            ComPtr<ID3D12GraphicsCommandList4> commands;
            HRESULT status =
                Device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator));
            if (SUCCEEDED(status))
                status = Device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(),
                                                   nullptr, IID_PPV_ARGS(&commands));
            if (FAILED(status))
            {
                error = HResultMessage("Creating DXR temporal commands", status);
                return false;
            }
            const auto finish = [&] {
                const HRESULT closed = commands->Close();
                if (FAILED(closed))
                {
                    error = HResultMessage("Closing DXR temporal commands", closed);
                    return false;
                }
                return ExecuteAndWait(commands.Get(), error);
            };
            const auto begin = [&] {
                HRESULT reset = allocator->Reset();
                if (SUCCEEDED(reset))
                    reset = commands->Reset(allocator.Get(), nullptr);
                if (FAILED(reset))
                {
                    error = HResultMessage("Resetting DXR temporal commands", reset);
                    return false;
                }
                return true;
            };
            commands->CopyBufferRegion(outputs[ErrorOutput].Get(), 0, clear.Get(), 0, sizeof(zero));
            const auto initialized =
                TransitionBarrier(outputs[ErrorOutput].Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                                  D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            commands->ResourceBarrier(1, &initialized);
            if (!finish())
                return false;
            clear.Reset();
            constexpr uint32_t Batch = 16384;
            for (controls.Begin = 0; controls.Begin < count; controls.Begin += Batch)
            {
                if (cancelled() || !begin())
                    return false;
                controls.Count = std::min(Batch, count - controls.Begin);
                commands->SetComputeRootSignature(TemporalRoot.Get());
                commands->SetComputeRoot32BitConstants(0, sizeof(controls) / sizeof(uint32_t), &controls, 0);
                for (uint32_t i = 0; i < InputCount; ++i)
                    commands->SetComputeRootShaderResourceView(InputRoot + i,
                                                               inputs[i]->GetGPUVirtualAddress());
                for (uint32_t i = 0; i < OutputCount; ++i)
                    commands->SetComputeRootUnorderedAccessView(OutputRoot + i,
                                                                outputs[i]->GetGPUVirtualAddress());
                commands->SetPipelineState1(TemporalState.Get());
                D3D12_DISPATCH_RAYS_DESC dispatch{};
                dispatch.RayGenerationShaderRecord = {TemporalTable->GetGPUVirtualAddress(),
                                                      ShaderRecordSize};
                dispatch.Width = controls.Count;
                dispatch.Height = 1;
                dispatch.Depth = 1;
                commands->DispatchRays(&dispatch);
                const D3D12_RESOURCE_BARRIER complete[] = {
                    UavBarrier(nullptr),
                    TransitionBarrier(outputs[ErrorOutput].Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                      D3D12_RESOURCE_STATE_COPY_SOURCE)};
                commands->ResourceBarrier(UINT(std::size(complete)), complete);
                commands->CopyBufferRegion(readback.Get(), statusOffset, outputs[ErrorOutput].Get(), 0,
                                           sizeof(zero));
                const auto restore =
                    TransitionBarrier(outputs[ErrorOutput].Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
                                      D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                commands->ResourceBarrier(1, &restore);
                if (!finish())
                    return false;
                void* mapped = nullptr;
                const D3D12_RANGE read{SIZE_T(statusOffset), SIZE_T(statusOffset + sizeof(zero))},
                    noWrite{0, 0};
                status = readback->Map(0, &read, &mapped);
                if (FAILED(status))
                {
                    error = HResultMessage("Reading DXR temporal diagnostics", status);
                    return false;
                }
                uint32_t failure = 0;
                std::memcpy(&failure, static_cast<const uint8_t*>(mapped) + statusOffset, sizeof(failure));
                readback->Unmap(0, &noWrite);
                if (failure)
                {
                    error = "DXR temporal reconstruction reported invalid configuration or radiance";
                    return false;
                }
            }
            if (cancelled() || !begin())
                return false;
            const D3D12_RESOURCE_BARRIER readable[] = {
                TransitionBarrier(outputs[ColorOutput].Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                  D3D12_RESOURCE_STATE_COPY_SOURCE),
                TransitionBarrier(outputs[DisplayOutput].Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                  D3D12_RESOURCE_STATE_COPY_SOURCE)};
            commands->ResourceBarrier(UINT(std::size(readable)), readable);
            commands->CopyBufferRegion(readback.Get(), 0, outputs[ColorOutput].Get(), 0, colorBytes);
            commands->CopyBufferRegion(readback.Get(), colorBytes, outputs[DisplayOutput].Get(), 0,
                                       displayBytes);
            if (!finish())
                return false;
            std::vector<TemporalPixel> filtered(count);
            Image encoded(controls.Width, controls.Height);
            void* mapped = nullptr;
            const D3D12_RANGE read{0, SIZE_T(statusOffset)}, noWrite{0, 0};
            status = readback->Map(0, &read, &mapped);
            if (FAILED(status))
            {
                error = HResultMessage("Reading DXR temporal output", status);
                return false;
            }
            std::memcpy(filtered.data(), mapped, size_t(colorBytes));
            std::memcpy(encoded.GetPixelsRGBA8().data(), static_cast<const uint8_t*>(mapped) + colorBytes,
                        size_t(displayBytes));
            readback->Unmap(0, &noWrite);
            if (cancelled())
                return false;
            result = std::move(filtered);
            display = std::move(encoded);
            return true;
        }

        bool EnsureGeometryGuides(std::string& error) const
        {
            if (GeometryGuidesState)
                return true;
            ComPtr<ID3D12RootSignature> root;
            HRESULT status = Device->CreateRootSignature(0, Detail::DxrGeometryGuidesRootSignature,
                                                         sizeof(Detail::DxrGeometryGuidesRootSignature),
                                                         IID_PPV_ARGS(&root));
            if (FAILED(status))
            {
                error = HResultMessage("Creating DXR geometry guide root", status);
                return false;
            }
            const D3D12_EXPORT_DESC exports[] = {
                {L"GenerateGeometryGuideRays", nullptr, D3D12_EXPORT_FLAG_NONE},
                {L"ResolveGeometryGuideHits", nullptr, D3D12_EXPORT_FLAG_NONE}};
            const D3D12_DXIL_LIBRARY_DESC library{
                {Detail::DxrGeometryGuidesDxil, sizeof(Detail::DxrGeometryGuidesDxil)},
                UINT(std::size(exports)),
                exports};
            const D3D12_RAYTRACING_SHADER_CONFIG shader{0, 0};
            const D3D12_GLOBAL_ROOT_SIGNATURE global{root.Get()};
            const D3D12_RAYTRACING_PIPELINE_CONFIG recursion{1};
            const D3D12_STATE_SUBOBJECT parts[] = {
                {D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY, &library},
                {D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_SHADER_CONFIG, &shader},
                {D3D12_STATE_SUBOBJECT_TYPE_GLOBAL_ROOT_SIGNATURE, &global},
                {D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_PIPELINE_CONFIG, &recursion}};
            const D3D12_STATE_OBJECT_DESC description{D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE,
                                                      UINT(std::size(parts)), parts};
            ComPtr<ID3D12StateObject> state;
            status = Device->CreateStateObject(&description, IID_PPV_ARGS(&state));
            if (FAILED(status))
            {
                error = HResultMessage("Creating DXR geometry guide pipeline", status);
                return false;
            }
            ComPtr<ID3D12StateObjectProperties> properties;
            status = state.As(&properties);
            if (FAILED(status))
            {
                error = HResultMessage("Reading DXR geometry guide properties", status);
                return false;
            }
            std::array<uint8_t, ShaderRecordSize * std::size(exports)> records{};
            for (size_t stage = 0; stage < std::size(exports); ++stage)
            {
                const void* id = properties->GetShaderIdentifier(exports[stage].Name);
                if (!id)
                {
                    error = "DXR geometry guide shader identifier is missing";
                    return false;
                }
                std::memcpy(records.data() + stage * ShaderRecordSize, id,
                            D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
            }
            ComPtr<ID3D12Resource> table;
            if (!CreateUploadBuffer(records.data(), records.size(), table, error))
                return false;
            GeometryGuidesRoot = std::move(root);
            GeometryGuidesTable = std::move(table);
            GeometryGuidesState = std::move(state);
            return true;
        }

        template <class Output>
        bool GeometryGuideStage(GeometryGuideData::Controls controls,
                                const std::vector<GeometryGuideData::EncodedRay>& rays,
                                const std::vector<RayQueryHit>& hits,
                                const Detail::CancellationProbe& cancelled, std::vector<Output>& result,
                                std::string& error) const
        {
            constexpr bool Generate = std::is_same<Output, GeometryGuideData::EncodedRay>::value;
            if (!EnsureGeometryGuides(error))
                return false;
            const uint32_t count = controls.Width * controls.Height;
            const uint64_t bytes = uint64_t(count) * sizeof(Output);
            ComPtr<IDXGIAdapter3> memoryAdapter;
            if (SUCCEEDED(Adapter.As(&memoryAdapter)))
            {
                DXGI_QUERY_VIDEO_MEMORY_INFO budget{};
                if (SUCCEEDED(
                        memoryAdapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &budget)))
                {
                    const uint64_t available =
                        budget.Budget > budget.CurrentUsage ? budget.Budget - budget.CurrentUsage : 0;
                    if (bytes + sizeof(uint32_t) > available)
                    {
                        error = "DXR geometry guides exceed video-memory budget";
                        return false;
                    }
                }
            }
            ComPtr<ID3D12Resource> inputRays, inputHits, output, failures, clear, readback;
            if constexpr (!Generate)
            {
                if (rays.size() != count || hits.size() != count)
                {
                    error = "DXR geometry guide input extent mismatch";
                    return false;
                }
                if (!CreateUploadBuffer(rays.data(), rays.size() * sizeof(rays[0]), inputRays, error) ||
                    !CreateUploadBuffer(hits.data(), hits.size() * sizeof(hits[0]), inputHits, error))
                    return false;
            }
            const uint32_t zero = 0;
            if (!CreateUploadBuffer(&zero, sizeof(zero), clear, error) ||
                !CreateBuffer(bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                              D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, output, error) ||
                !CreateBuffer(sizeof(zero), D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST,
                              D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, failures, error) ||
                !CreateBuffer(bytes + sizeof(zero), D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST,
                              D3D12_RESOURCE_FLAG_NONE, readback, error))
                return false;
            ComPtr<ID3D12CommandAllocator> allocator;
            ComPtr<ID3D12GraphicsCommandList4> commands;
            HRESULT status =
                Device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator));
            if (SUCCEEDED(status))
                status = Device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(),
                                                   nullptr, IID_PPV_ARGS(&commands));
            if (FAILED(status))
            {
                error = HResultMessage("Creating DXR geometry guide commands", status);
                return false;
            }
            const auto finish = [&] {
                const HRESULT closed = commands->Close();
                if (FAILED(closed))
                {
                    error = HResultMessage("Closing DXR geometry guide commands", closed);
                    return false;
                }
                return ExecuteAndWait(commands.Get(), error);
            };
            const auto begin = [&] {
                HRESULT reset = allocator->Reset();
                if (SUCCEEDED(reset))
                    reset = commands->Reset(allocator.Get(), nullptr);
                if (FAILED(reset))
                {
                    error = HResultMessage("Resetting DXR geometry guide commands", reset);
                    return false;
                }
                return true;
            };
            commands->CopyBufferRegion(failures.Get(), 0, clear.Get(), 0, sizeof(zero));
            const auto initialized = TransitionBarrier(failures.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                                                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            commands->ResourceBarrier(1, &initialized);
            if (!finish())
                return false;
            clear.Reset();
            constexpr uint32_t Batch = 16384;
            constexpr UINT ControlsRoot = 0, RayRoot = 1, HitRoot = 2, GeneratedRayRoot = 3, GuideRoot = 4,
                           ErrorRoot = 5;
            for (controls.Begin = 0; controls.Begin < count; controls.Begin += Batch)
            {
                if (cancelled() || !begin())
                    return false;
                controls.Count = std::min(Batch, count - controls.Begin);
                commands->SetComputeRootSignature(GeometryGuidesRoot.Get());
                commands->SetComputeRoot32BitConstants(ControlsRoot, sizeof(controls) / sizeof(uint32_t),
                                                       &controls, 0);
                commands->SetComputeRootShaderResourceView(RayRoot,
                                                           Generate ? 0 : inputRays->GetGPUVirtualAddress());
                commands->SetComputeRootShaderResourceView(HitRoot,
                                                           Generate ? 0 : inputHits->GetGPUVirtualAddress());
                commands->SetComputeRootUnorderedAccessView(GeneratedRayRoot,
                                                            Generate ? output->GetGPUVirtualAddress() : 0);
                commands->SetComputeRootUnorderedAccessView(GuideRoot,
                                                            Generate ? 0 : output->GetGPUVirtualAddress());
                commands->SetComputeRootUnorderedAccessView(ErrorRoot, failures->GetGPUVirtualAddress());
                commands->SetPipelineState1(GeometryGuidesState.Get());
                D3D12_DISPATCH_RAYS_DESC dispatch{};
                dispatch.RayGenerationShaderRecord = {GeometryGuidesTable->GetGPUVirtualAddress() +
                                                          (Generate ? 0 : ShaderRecordSize),
                                                      ShaderRecordSize};
                dispatch.Width = controls.Count;
                dispatch.Height = 1;
                dispatch.Depth = 1;
                commands->DispatchRays(&dispatch);
                const D3D12_RESOURCE_BARRIER completed[] = {
                    UavBarrier(nullptr),
                    TransitionBarrier(failures.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                      D3D12_RESOURCE_STATE_COPY_SOURCE)};
                commands->ResourceBarrier(UINT(std::size(completed)), completed);
                commands->CopyBufferRegion(readback.Get(), bytes, failures.Get(), 0, sizeof(zero));
                const auto restore = TransitionBarrier(failures.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
                                                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                commands->ResourceBarrier(1, &restore);
                if (!finish())
                    return false;
                void* mapped = nullptr;
                const D3D12_RANGE read{SIZE_T(bytes), SIZE_T(bytes + sizeof(zero))}, noWrite{0, 0};
                status = readback->Map(0, &read, &mapped);
                if (FAILED(status))
                {
                    error = HResultMessage("Reading DXR geometry guide diagnostics", status);
                    return false;
                }
                uint32_t failure = 0;
                std::memcpy(&failure, static_cast<const uint8_t*>(mapped) + bytes, sizeof(failure));
                readback->Unmap(0, &noWrite);
                if (failure)
                {
                    error = "DXR geometry guides encountered invalid dispatch or geometry";
                    return false;
                }
            }
            if (cancelled() || !begin())
                return false;
            const auto readable = TransitionBarrier(output.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                                    D3D12_RESOURCE_STATE_COPY_SOURCE);
            commands->ResourceBarrier(1, &readable);
            commands->CopyBufferRegion(readback.Get(), 0, output.Get(), 0, bytes);
            if (!finish())
                return false;
            std::vector<Output> decoded(count);
            void* mapped = nullptr;
            const D3D12_RANGE read{0, SIZE_T(bytes)}, noWrite{0, 0};
            status = readback->Map(0, &read, &mapped);
            if (FAILED(status))
            {
                error = HResultMessage("Reading DXR geometry guide output", status);
                return false;
            }
            std::memcpy(decoded.data(), mapped, size_t(bytes));
            readback->Unmap(0, &noWrite);
            if (cancelled())
                return false;
            result = std::move(decoded);
            return true;
        }

        bool EnsureAdaptiveDenoise(std::string& error) const
        {
            if (AdaptiveDenoiseState) return true;
            ComPtr<ID3D12RootSignature> root;
            HRESULT status = Device->CreateRootSignature(0, Detail::DxrAdaptiveDenoiseRootSignature,
                sizeof(Detail::DxrAdaptiveDenoiseRootSignature), IID_PPV_ARGS(&root));
            if (FAILED(status)) { error = HResultMessage("Creating DXR allocation root", status); return false; }
            const D3D12_EXPORT_DESC entry{L"AdaptiveDenoiseRayGen", nullptr, D3D12_EXPORT_FLAG_NONE};
            const D3D12_DXIL_LIBRARY_DESC library{{Detail::DxrAdaptiveDenoiseDxil,
                sizeof(Detail::DxrAdaptiveDenoiseDxil)}, 1, &entry};
            const D3D12_RAYTRACING_SHADER_CONFIG shader{0, 0};
            const D3D12_GLOBAL_ROOT_SIGNATURE global{root.Get()};
            const D3D12_RAYTRACING_PIPELINE_CONFIG recursion{1};
            const D3D12_STATE_SUBOBJECT parts[]{
                {D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY, &library},
                {D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_SHADER_CONFIG, &shader},
                {D3D12_STATE_SUBOBJECT_TYPE_GLOBAL_ROOT_SIGNATURE, &global},
                {D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_PIPELINE_CONFIG, &recursion}};
            const D3D12_STATE_OBJECT_DESC description{D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE, UINT(std::size(parts)), parts};
            ComPtr<ID3D12StateObject> state;
            status = Device->CreateStateObject(&description, IID_PPV_ARGS(&state));
            if (FAILED(status)) { error = HResultMessage("Creating DXR allocation pipeline", status); return false; }
            ComPtr<ID3D12StateObjectProperties> properties;
            status = state.As(&properties);
            if (FAILED(status)) { error = HResultMessage("Reading DXR allocation properties", status); return false; }
            const void* identifier = properties->GetShaderIdentifier(entry.Name);
            if (!identifier) { error = "DXR allocation ray-generation identifier is missing."; return false; }
            std::array<uint8_t, ShaderRecordSize> record{};
            std::memcpy(record.data(), identifier, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
            ComPtr<ID3D12Resource> table;
            if (!CreateUploadBuffer(record.data(), record.size(), table, error)) return false;
            AdaptiveDenoiseRoot = std::move(root);
            AdaptiveDenoiseTable = std::move(table);
            AdaptiveDenoiseState = std::move(state);
            return true;
        }

        bool AllocateDenoisedNative(const std::vector<AdaptiveReconstructionData::Pixel>& input,
            AdaptiveReconstructionData::Controls controls, const Detail::CancellationProbe& cancellation,
            std::vector<uint32_t>& result, std::string& error) const
        {
            if (!EnsureAdaptiveDenoise(error)) return false;
            const size_t bytes = input.size() * sizeof(uint32_t);
            ComPtr<ID3D12Resource> raw, counts, readback;
            if (!CreateUploadBuffer(input.data(), input.size() * sizeof(input.front()), raw, error) ||
                !CreateBuffer(bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, counts, error) ||
                !CreateBuffer(bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_FLAG_NONE, readback, error)) return false;
            ComPtr<ID3D12CommandAllocator> allocator;
            ComPtr<ID3D12GraphicsCommandList4> commands;
            HRESULT status = Device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator));
            if (SUCCEEDED(status)) status = Device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                allocator.Get(), nullptr, IID_PPV_ARGS(&commands));
            if (FAILED(status)) { error = HResultMessage("Creating DXR allocation commands", status); return false; }
            constexpr uint32_t Batch = 16384;
            const auto pixels = uint32_t(input.size());
            for (controls.Begin = 0; controls.Begin < pixels; controls.Begin += Batch)
            {
                if (cancellation()) return false;
                if (controls.Begin)
                {
                    status = allocator->Reset();
                    if (SUCCEEDED(status)) status = commands->Reset(allocator.Get(), nullptr);
                    if (FAILED(status)) { error = HResultMessage("Resetting DXR allocation commands", status); return false; }
                }
                controls.Count = std::min(Batch, pixels - controls.Begin);
                commands->SetComputeRootSignature(AdaptiveDenoiseRoot.Get());
                commands->SetComputeRoot32BitConstants(0, sizeof(controls) / sizeof(uint32_t), &controls, 0);
                commands->SetComputeRootShaderResourceView(1, raw->GetGPUVirtualAddress());
                commands->SetComputeRootUnorderedAccessView(2, counts->GetGPUVirtualAddress());
                commands->SetPipelineState1(AdaptiveDenoiseState.Get());
                D3D12_DISPATCH_RAYS_DESC dispatch{};
                dispatch.RayGenerationShaderRecord = {AdaptiveDenoiseTable->GetGPUVirtualAddress(), ShaderRecordSize};
                dispatch.Width = controls.Count; dispatch.Height = dispatch.Depth = 1;
                commands->DispatchRays(&dispatch);
                auto barrier = UavBarrier(counts.Get());
                commands->ResourceBarrier(1, &barrier);
                const bool last = controls.Begin + controls.Count == pixels;
                if (last)
                {
                    auto toCopy = TransitionBarrier(counts.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                                   D3D12_RESOURCE_STATE_COPY_SOURCE);
                    commands->ResourceBarrier(1, &toCopy);
                    commands->CopyResource(readback.Get(), counts.Get());
                }
                status = commands->Close();
                if (FAILED(status)) { error = HResultMessage("Closing DXR allocation commands", status); return false; }
                if (!ExecuteAndWait(commands.Get(), error)) return false;
            }
            if (cancellation()) return false;
            std::vector<uint32_t> candidate(input.size());
            const D3D12_RANGE range{0, bytes}, noWrite{0, 0};
            void* mapped = nullptr;
            status = readback->Map(0, &range, &mapped);
            if (FAILED(status)) { error = HResultMessage("Reading DXR production budgets", status); return false; }
            std::memcpy(candidate.data(), mapped, bytes);
            readback->Unmap(0, &noWrite);
            if (cancellation()) return false;
            result = std::move(candidate);
            return true;
        }

        bool EnsureSpatialDenoise(std::string& error) const
        {
            if (SpatialDenoiseState)
                return true;
            ComPtr<ID3D12RootSignature> root;
            HRESULT status = Device->CreateRootSignature(0, Detail::DxrSpatialDenoiseRootSignature,
                                                         sizeof(Detail::DxrSpatialDenoiseRootSignature),
                                                         IID_PPV_ARGS(&root));
            if (FAILED(status))
            {
                error = HResultMessage("Creating DXR reconstruction root", status);
                return false;
            }
            const D3D12_EXPORT_DESC entry{L"SpatialDenoiseRayGen", nullptr, D3D12_EXPORT_FLAG_NONE};
            const D3D12_DXIL_LIBRARY_DESC library{
                {Detail::DxrSpatialDenoiseDxil, sizeof(Detail::DxrSpatialDenoiseDxil)}, 1, &entry};
            // This ray-generation program reads the film directly; it traces no scene rays.
            const D3D12_RAYTRACING_SHADER_CONFIG shader{0, 0};
            const D3D12_GLOBAL_ROOT_SIGNATURE global{root.Get()};
            const D3D12_RAYTRACING_PIPELINE_CONFIG recursion{1};
            const D3D12_STATE_SUBOBJECT parts[] = {
                {D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY, &library},
                {D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_SHADER_CONFIG, &shader},
                {D3D12_STATE_SUBOBJECT_TYPE_GLOBAL_ROOT_SIGNATURE, &global},
                {D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_PIPELINE_CONFIG, &recursion}};
            const D3D12_STATE_OBJECT_DESC description{D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE,
                                                      UINT(std::size(parts)), parts};
            ComPtr<ID3D12StateObject> state;
            status = Device->CreateStateObject(&description, IID_PPV_ARGS(&state));
            if (FAILED(status))
            {
                error = HResultMessage("Creating DXR reconstruction pipeline", status);
                return false;
            }
            ComPtr<ID3D12StateObjectProperties> properties;
            status = state.As(&properties);
            if (FAILED(status))
            {
                error = HResultMessage("Reading DXR reconstruction properties", status);
                return false;
            }
            const void* identifier = properties->GetShaderIdentifier(entry.Name);
            if (!identifier)
            {
                error = "DXR reconstruction ray-generation identifier is missing.";
                return false;
            }
            std::array<uint8_t, ShaderRecordSize> record{};
            std::memcpy(record.data(), identifier, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
            ComPtr<ID3D12Resource> table;
            if (!CreateUploadBuffer(record.data(), record.size(), table, error))
                return false;
            SpatialDenoiseRoot = std::move(root);
            SpatialDenoiseTable = std::move(table);
            SpatialDenoiseState = std::move(state);
            return true;
        }

        GeometryGuideResult CaptureGuidesNative(const Scene &scene, const Camera &camera,
                                                const RenderSettings &settings,
                                                const Detail::CancellationProbe &stopped, bool spatialAlbedo = false) const
        {
            return Detail::CaptureGeometryGuides(
                scene, camera, settings, spatialAlbedo, stopped, [&](const auto &controls, auto &rays, auto &error)
                { return GeometryGuideStage(controls, {}, {}, stopped, rays, error); }, [&](const auto &rays)
                { return QueryNative(scene, rays, {settings.Acceleration, settings.BvhPolicy}, stopped); },
                [&](const auto &controls, const auto &rays, const auto &hits, auto &guides, auto &error)
                { return GeometryGuideStage(controls, rays, hits, stopped, guides, error); });
        }

        bool ReconstructNative(const LinearFilm &film, const RenderSettings &settings,
                               const GeometryGuideImage *guides,
                               const Detail::CancellationProbe &cancellation, ReconstructedImage &result,
                               std::string &error) const
        {
            namespace R = ReconstructionData;
            if (!ValidateReconstructionInput(film, settings.Reconstruction, error) ||
                !ValidateReconstructionGuides(film, settings.Reconstruction, guides, error) ||
                !EnsureSpatialDenoise(error))
                return false;
            const uint32_t pixels = film.Width * film.Height;
            using ColorRecord = std::array<float, 4>;
            const uint64_t colorBytes = uint64_t(pixels) * sizeof(ColorRecord);
            const uint64_t displayBytes = uint64_t(pixels) * sizeof(uint32_t);
            const uint64_t statusOffset = colorBytes + displayBytes;
            std::vector<R::Pixel> input(pixels);
            for (uint32_t i = 0; i < pixels; ++i)
            {
                if (cancellation())
                    return false;
                const auto& p = film.Pixels[i];
                input[i] = {p.MeanRadiance.X, p.MeanRadiance.Y, p.MeanRadiance.Z,
                            p.LuminanceM2,    p.SampleCount,    {}};
                if (guides)
                    R::PackGuide(input[i], *guides, i);
            }
            ComPtr<IDXGIAdapter3> memoryAdapter;
            if (SUCCEEDED(Adapter.As(&memoryAdapter)))
            {
                DXGI_QUERY_VIDEO_MEMORY_INFO budget{};
                if (SUCCEEDED(
                        memoryAdapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &budget)))
                {
                    const uint64_t available =
                        budget.Budget > budget.CurrentUsage ? budget.Budget - budget.CurrentUsage : 0;
                    if (2 * colorBytes + displayBytes + sizeof(uint32_t) > available)
                    {
                        error = "DXR reconstruction exceeds the video-memory budget.";
                        return false;
                    }
                }
            }
            ComPtr<ID3D12Resource> raw, previous, next, display, failures, clear, readback;
            const uint32_t zero = 0;
            if (!CreateUploadBuffer(input.data(), input.size() * sizeof(R::Pixel), raw, error) ||
                !CreateUploadBuffer(&zero, sizeof(zero), clear, error))
                return false;
            for (auto* buffer : {std::addressof(previous), std::addressof(next)})
                if (!CreateBuffer(colorBytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                  D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, *buffer, error))
                    return false;
            if (!CreateBuffer(displayBytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                              D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, display, error) ||
                !CreateBuffer(sizeof(zero), D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST,
                              D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, failures, error) ||
                !CreateBuffer(statusOffset + sizeof(zero), D3D12_HEAP_TYPE_READBACK,
                              D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE, readback, error))
                return false;
            ComPtr<ID3D12CommandAllocator> allocator;
            ComPtr<ID3D12GraphicsCommandList4> commands;
            HRESULT status =
                Device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator));
            if (SUCCEEDED(status))
                status = Device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(),
                                                   nullptr, IID_PPV_ARGS(&commands));
            if (FAILED(status))
            {
                error = HResultMessage("Creating DXR reconstruction commands", status);
                return false;
            }
            const auto finish = [&] {
                const HRESULT closed = commands->Close();
                if (FAILED(closed))
                {
                    error = HResultMessage("Closing DXR reconstruction commands", closed);
                    return false;
                }
                return ExecuteAndWait(commands.Get(), error);
            };
            const auto begin = [&] {
                HRESULT reset = allocator->Reset();
                if (SUCCEEDED(reset))
                    reset = commands->Reset(allocator.Get(), nullptr);
                if (FAILED(reset))
                {
                    error = HResultMessage("Resetting DXR reconstruction commands", reset);
                    return false;
                }
                return true;
            };
            commands->CopyBufferRegion(failures.Get(), 0, clear.Get(), 0, sizeof(zero));
            auto initialized = TransitionBarrier(failures.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                                                 D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            commands->ResourceBarrier(1, &initialized);
            if (!finish())
                return false;
            clear.Reset();
            R::Controls controls{};
            controls.Width = film.Width;
            controls.Height = film.Height;
            controls.NoiseScale = settings.Reconstruction.NoiseScale;
            controls.RelativeColorScale = settings.Reconstruction.RelativeColorScale;
            controls.AbsoluteColorScale = settings.Reconstruction.AbsoluteColorScale;
            controls.Flags = ReconstructionUsesMoments(settings.Integrator) ? R::MomentGuidance : 0;
            controls.GammaCorrection = settings.GammaCorrection ? 1 : 0;
            controls.NormalScale = settings.Reconstruction.NormalScale;
            controls.RelativePlaneScale = settings.Reconstruction.RelativePlaneScale;
            if (guides)
                controls.Flags |= R::GeometryGuidance;
            constexpr uint32_t BatchPixels = 16384;
            constexpr UINT ControlsRoot = 0, RawRoot = 1, PreviousRoot = 2, NextRoot = 3, ErrorRoot = 4,
                           DisplayRoot = 5;
            for (uint32_t pass = 0; pass < settings.Reconstruction.Passes; ++pass)
            {
                controls.Pass = pass;
                for (uint32_t first = 0; first < pixels; first += BatchPixels)
                {
                    if (cancellation() || !begin())
                        return false;
                    // Restore both scratch buffers to UAV after each level, so their roles can swap.
                    const auto readable =
                        TransitionBarrier(previous.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                          D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                    commands->ResourceBarrier(1, &readable);
                    controls.Begin = first;
                    controls.Count = std::min(BatchPixels, pixels - first);
                    commands->SetComputeRootSignature(SpatialDenoiseRoot.Get());
                    commands->SetComputeRoot32BitConstants(ControlsRoot, sizeof(controls) / sizeof(uint32_t),
                                                           &controls, 0);
                    commands->SetComputeRootShaderResourceView(RawRoot, raw->GetGPUVirtualAddress());
                    commands->SetComputeRootShaderResourceView(PreviousRoot,
                                                               previous->GetGPUVirtualAddress());
                    commands->SetComputeRootUnorderedAccessView(NextRoot, next->GetGPUVirtualAddress());
                    commands->SetComputeRootUnorderedAccessView(ErrorRoot, failures->GetGPUVirtualAddress());
                    commands->SetComputeRootUnorderedAccessView(DisplayRoot, display->GetGPUVirtualAddress());
                    commands->SetPipelineState1(SpatialDenoiseState.Get());
                    D3D12_DISPATCH_RAYS_DESC dispatch{};
                    dispatch.RayGenerationShaderRecord = {SpatialDenoiseTable->GetGPUVirtualAddress(),
                                                          ShaderRecordSize};
                    dispatch.Width = controls.Count;
                    dispatch.Height = 1;
                    dispatch.Depth = 1;
                    commands->DispatchRays(&dispatch);
                    const D3D12_RESOURCE_BARRIER completed[] = {
                        UavBarrier(nullptr),
                        TransitionBarrier(previous.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                          D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
                        TransitionBarrier(failures.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                          D3D12_RESOURCE_STATE_COPY_SOURCE)};
                    commands->ResourceBarrier(UINT(std::size(completed)), completed);
                    commands->CopyBufferRegion(readback.Get(), statusOffset, failures.Get(), 0, sizeof(zero));
                    const auto restore = TransitionBarrier(failures.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
                                                           D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                    commands->ResourceBarrier(1, &restore);
                    if (!finish())
                        return false;
                    void* data = nullptr;
                    const D3D12_RANGE read{SIZE_T(statusOffset), SIZE_T(statusOffset + sizeof(zero))},
                        noWrite{0, 0};
                    status = readback->Map(0, &read, &data);
                    if (FAILED(status))
                    {
                        error = HResultMessage("Reading DXR reconstruction diagnostics", status);
                        return false;
                    }
                    uint32_t flags = 0;
                    std::memcpy(&flags, static_cast<const uint8_t*>(data) + statusOffset, sizeof(flags));
                    readback->Unmap(0, &noWrite);
                    if (flags)
                    {
                        error = "DXR reconstruction encountered invalid controls or nonfinite radiance.";
                        return false;
                    }
                }
                std::swap(previous, next);
            }
            if (cancellation() || !begin())
                return false;
            const D3D12_RESOURCE_BARRIER copies[] = {
                TransitionBarrier(previous.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                  D3D12_RESOURCE_STATE_COPY_SOURCE),
                TransitionBarrier(display.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                  D3D12_RESOURCE_STATE_COPY_SOURCE)};
            commands->ResourceBarrier(UINT(std::size(copies)), copies);
            commands->CopyBufferRegion(readback.Get(), 0, previous.Get(), 0, colorBytes);
            commands->CopyBufferRegion(readback.Get(), colorBytes, display.Get(), 0, displayBytes);
            if (!finish())
                return false;
            ReconstructedImage output;
            output.Width = film.Width;
            output.Height = film.Height;
            output.Pixels.resize(pixels);
            output.Display = Image(film.Width, film.Height);
            void* data = nullptr;
            const D3D12_RANGE read{0, SIZE_T(statusOffset)}, noWrite{0, 0};
            status = readback->Map(0, &read, &data);
            if (FAILED(status))
            {
                error = HResultMessage("Reading DXR reconstructed film", status);
                return false;
            }
            const auto* colors = static_cast<const ColorRecord*>(data);
            for (uint32_t row = 0; row < film.Height; ++row)
            {
                if (cancellation())
                {
                    readback->Unmap(0, &noWrite);
                    return false;
                }
                for (uint32_t x = 0; x < film.Width; ++x)
                {
                    const uint32_t i = row * film.Width + x;
                    output.Pixels[i] = {colors[i][0], colors[i][1], colors[i][2]};
                }
            }
            std::memcpy(output.Display.GetPixelsRGBA8().data(),
                        static_cast<const uint8_t*>(data) + colorBytes, size_t(displayBytes));
            readback->Unmap(0, &noWrite);
            result = std::move(output);
            return true;
        }

        bool DispatchPointGi(const DxrPointGi::SurfaceCache& geometry, const PreparedEmissionScene& emission,
                             const GpuConstants& constants, const PointGiOptions& options,
                             ID3D12CommandAllocator* allocator, ID3D12GraphicsCommandList4* commands,
                             ID3D12Resource* shaderTable, const InstanceResources& instances,
                             const std::function<void()>& bindScene,
                             const Detail::CancellationProbe& cancellation, std::string& error) const
        {
            namespace G = DxrPointGi;
            const uint32_t pixelCount = constants.Width * constants.Height;
            G::Controls controls{};
            controls.Surfaces = uint32_t(geometry.Surfaces.size());
            controls.Nodes = uint32_t(geometry.Nodes.size());
            controls.Sources = uint32_t(emission.Sources.size());
            controls.Seed = options.Seed ^ constants.SamplingSeed;
            controls.DirectSamples = options.DirectSamples;
            controls.GatherSamples = options.GatherSamples;
            const uint64_t radianceBytes = uint64_t(controls.Surfaces) * sizeof(G::Radiance);
            ComPtr<ID3D12Resource> surfaces, nodes, indices, sources, previous, next, total, failures,
                readback, batch;
            struct Upload
            {
                ComPtr<ID3D12Resource>* Buffer;
                const void* Data;
                uint64_t Bytes;
            };
            const Upload uploads[] = {
                {&surfaces, geometry.Surfaces.data(), uint64_t(controls.Surfaces) * sizeof(G::Surface)},
                {&nodes, geometry.Nodes.data(), uint64_t(controls.Nodes) * sizeof(G::Node)},
                {&indices, geometry.Indices.data(), uint64_t(geometry.Indices.size()) * sizeof(uint32_t)},
                {&sources, emission.Sources.data(),
                 uint64_t(emission.Sources.size()) * sizeof(PackedEmitter)}};
            uint64_t memory = 3 * radianceBytes;
            for (const auto& upload : uploads)
                memory += upload.Bytes;
            ComPtr<IDXGIAdapter3> memoryAdapter;
            if (SUCCEEDED(Adapter.As(&memoryAdapter)))
            {
                DXGI_QUERY_VIDEO_MEMORY_INFO budget{};
                if (SUCCEEDED(
                        memoryAdapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &budget)))
                {
                    const uint64_t available =
                        budget.Budget > budget.CurrentUsage ? budget.Budget - budget.CurrentUsage : 0;
                    if (memory > available)
                    {
                        error = "DXR Point GI buffers exceed the video-memory budget.";
                        return false;
                    }
                }
            }
            for (const auto& upload : uploads)
                if (!CreateUploadBuffer(upload.Data, upload.Bytes, *upload.Buffer, error))
                    return false;
            const std::array<uint8_t, D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT> initialControls{};
            if (!CreateUploadBuffer(initialControls.data(), initialControls.size(), batch, error))
                return false;
            for (auto* buffer : {std::addressof(previous), std::addressof(next), std::addressof(total)})
                if (!CreateBuffer(std::max<uint64_t>(radianceBytes, sizeof(G::Radiance)),
                                  D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                  D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, *buffer, error))
                    return false;
            if (!CreateBuffer(sizeof(uint32_t), D3D12_HEAP_TYPE_DEFAULT,
                              D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                              D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, failures, error) ||
                !CreateBuffer(sizeof(uint32_t), D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST,
                              D3D12_RESOURCE_FLAG_NONE, readback, error))
                return false;
            const auto dispatch = [&] {
                if (cancellation())
                    return false;
                void* data = nullptr;
                const D3D12_RANGE noRead{0, 0}, written{0, sizeof(controls)};
                HRESULT status = batch->Map(0, &noRead, &data);
                if (FAILED(status))
                {
                    error = HResultMessage("Mapping DXR Point GI controls", status);
                    return false;
                }
                std::memcpy(data, &controls, sizeof(controls));
                batch->Unmap(0, &written);
                commands->SetComputeRootShaderResourceView(G::SourceRoot, sources->GetGPUVirtualAddress());
                commands->SetComputeRootShaderResourceView(G::SurfaceRoot, surfaces->GetGPUVirtualAddress());
                commands->SetComputeRootShaderResourceView(G::BoundsRoot, nodes->GetGPUVirtualAddress());
                commands->SetComputeRootShaderResourceView(G::IndexRoot, indices->GetGPUVirtualAddress());
                commands->SetComputeRootUnorderedAccessView(G::PreviousRoot,
                                                            previous->GetGPUVirtualAddress());
                commands->SetComputeRootUnorderedAccessView(G::NextRoot, next->GetGPUVirtualAddress());
                commands->SetComputeRootUnorderedAccessView(G::TotalRoot, total->GetGPUVirtualAddress());
                commands->SetComputeRootUnorderedAccessView(G::ErrorRoot, failures->GetGPUVirtualAddress());
                commands->SetComputeRootConstantBufferView(G::ControlsRoot, batch->GetGPUVirtualAddress());
                const auto rays = MakeDispatch(shaderTable, controls.Count, 1);
                commands->DispatchRays(&rays);
                const D3D12_RESOURCE_BARRIER completed[] = {
                    UavBarrier(nullptr),
                    TransitionBarrier(failures.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                      D3D12_RESOURCE_STATE_COPY_SOURCE)};
                commands->ResourceBarrier(UINT(std::size(completed)), completed);
                commands->CopyResource(readback.Get(), failures.Get());
                const auto restore = TransitionBarrier(failures.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
                                                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                commands->ResourceBarrier(1, &restore);
                CopyInstanceError(commands, instances);
                const auto restoreInstances =
                    TransitionBarrier(instances.Error.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
                                      D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                commands->ResourceBarrier(1, &restoreInstances);
                status = commands->Close();
                if (FAILED(status))
                {
                    error = HResultMessage("Closing DXR Point GI batch", status);
                    return false;
                }
                // Every submitted batch retires before cancellation can release its storage.
                if (!ExecuteAndWait(commands, error) || !CheckInstanceError(instances, error))
                    return false;
                const D3D12_RANGE read{0, sizeof(uint32_t)}, noWrite{0, 0};
                status = readback->Map(0, &read, &data);
                if (FAILED(status))
                {
                    error = HResultMessage("Mapping DXR Point GI diagnostics", status);
                    return false;
                }
                uint32_t flags;
                std::memcpy(&flags, data, sizeof(flags));
                readback->Unmap(0, &noWrite);
                if (flags)
                {
                    error = "DXR Point GI encountered invalid controls, disk indices or nonfinite transport.";
                    return false;
                }
                status = allocator->Reset();
                if (SUCCEEDED(status))
                    status = commands->Reset(allocator, nullptr);
                if (FAILED(status))
                {
                    error = HResultMessage("Resetting DXR Point GI batch", status);
                    return false;
                }
                bindScene();
                return true;
            };
            const auto population = [&](uint32_t count) {
                for (uint32_t begin = 0; begin < count; begin += G::PopulationBatch)
                {
                    controls.Begin = begin;
                    controls.Count = std::min(G::PopulationBatch, count - begin);
                    if (!dispatch())
                        return false;
                }
                return true;
            };
            const auto sampledPopulation = [&](uint32_t count, uint32_t samples) {
                for (uint32_t begin = 0; begin < samples; begin += G::SampleBatch)
                {
                    controls.SampleBegin = begin;
                    controls.SampleCount = std::min(G::SampleBatch, samples - begin);
                    if (!population(count))
                        return false;
                }
                return true;
            };
            controls.Stage = G::InitializeStage;
            if (!population(pixelCount))
                return false;
            controls.Stage = G::BakeStage;
            for (uint32_t order = 0; order < constants.MaxReflectionBounces; ++order)
            {
                controls.Order = order;
                if (!sampledPopulation(controls.Surfaces,
                                       order == 0 ? options.DirectSamples : options.GatherSamples))
                    return false;
                previous.Swap(next);
            }
            controls.Stage = G::CameraStage;
            return sampledPopulation(pixelCount, constants.SamplesPerPixel);
        }

        bool DispatchProgressivePhotons(const PreparedEmissionScene& emission, const GpuConstants& constants,
                                        const ProgressivePhotonOptions& options,
                                        ID3D12CommandAllocator* allocator,
                                        ID3D12GraphicsCommandList4* commands, ID3D12Resource* shaderTable,
                                        const InstanceResources& instances,
                                        const std::function<void()>& bindScene,
                                        const Detail::CancellationProbe& cancellation,
                                        std::string& error) const
        {
            const uint32_t pixelCount = constants.Width * constants.Height;
            const uint32_t slots = constants.MaxReflectionBounces + 1;
            const uint32_t lightCapacity = std::min(options.PathsPerPass, DxrPpm::LightBatch);
            uint32_t bucketCount = 1;
            while (bucketCount < lightCapacity)
                bucketCount *= 2;
            const uint64_t photonBytes = uint64_t(lightCapacity) * slots * DxrPpm::PhotonStride;
            const uint64_t visibleBytes = uint64_t(pixelCount) * DxrPpm::VisibleStride;
            const uint64_t pixelBytes = uint64_t(pixelCount) * DxrPpm::PixelStride;
            const uint64_t headBytes = uint64_t(bucketCount) * sizeof(uint32_t);
            ComPtr<IDXGIAdapter3> memoryAdapter;
            if (SUCCEEDED(Adapter.As(&memoryAdapter)))
            {
                DXGI_QUERY_VIDEO_MEMORY_INFO memory{};
                if (SUCCEEDED(
                        memoryAdapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &memory)))
                {
                    const uint64_t remaining =
                        memory.Budget > memory.CurrentUsage ? memory.Budget - memory.CurrentUsage : 0;
                    if (photonBytes + visibleBytes + pixelBytes + headBytes > remaining)
                    {
                        error = "DXR PPM buffers exceed the available video-memory budget.";
                        return false;
                    }
                }
            }
            ComPtr<ID3D12Resource> sources, photons, heads, visible, pixels, failures, readback, batchBuffer;
            const auto createUav = [&](uint64_t bytes, ComPtr<ID3D12Resource>& resource) {
                return CreateBuffer(bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, resource, error);
            };
            // A root CBV is two DWORDs regardless of the control block's size.
            // Reuse its upload allocation only after the preceding batch fence.
            const std::array<uint8_t, D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT> initialControls{};
            if (!CreateUploadBuffer(emission.Sources.data(), emission.Sources.size() * sizeof(PackedEmitter),
                                    sources, error) ||
                !CreateUploadBuffer(initialControls.data(), initialControls.size(), batchBuffer, error) ||
                !createUav(photonBytes, photons) || !createUav(headBytes, heads) ||
                !createUav(visibleBytes, visible) || !createUav(pixelBytes, pixels) ||
                !createUav(sizeof(uint32_t), failures) ||
                !CreateBuffer(sizeof(uint32_t), D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST,
                              D3D12_RESOURCE_FLAG_NONE, readback, error))
                return false;
            constexpr uint32_t CausticsEnabled = 1, IndirectEnabled = 2;
            DxrPpm::Controls controls{};
            controls.Center[0] = emission.Center.X;
            controls.Center[1] = emission.Center.Y;
            controls.Center[2] = emission.Center.Z;
            controls.WorldRadius = emission.Radius;
            controls.InitialRadius = double(emission.Radius) * double(options.InitialRadiusScale);
            controls.Alpha = options.Alpha;
            controls.EmitterCount = uint32_t(emission.Sources.size());
            controls.PathsPerPass = options.PathsPerPass;
            controls.Seed = options.Seed ^ constants.SamplingSeed;
            controls.Classes =
                (options.Caustics ? CausticsEnabled : 0u) | (options.Indirect ? IndirectEnabled : 0u);
            controls.SlotsPerPath = slots;
            controls.BucketCount = bucketCount;
            const auto dispatch = [&](uint32_t stage, uint32_t first, uint32_t count) {
                if (cancellation())
                    return false;
                controls.Stage = stage;
                controls.Begin = first;
                controls.Count = count;
                void* mapping = nullptr;
                const D3D12_RANGE noRead{0, 0}, written{0, sizeof(controls)};
                HRESULT status = batchBuffer->Map(0, &noRead, &mapping);
                if (FAILED(status))
                {
                    error = HResultMessage("Mapping DXR PPM controls", status);
                    return false;
                }
                std::memcpy(mapping, &controls, sizeof(controls));
                batchBuffer->Unmap(0, &written);
                commands->SetComputeRootShaderResourceView(DxrPpm::EmitterRoot,
                                                           sources->GetGPUVirtualAddress());
                commands->SetComputeRootUnorderedAccessView(DxrPpm::PhotonsRoot,
                                                            photons->GetGPUVirtualAddress());
                commands->SetComputeRootUnorderedAccessView(DxrPpm::HeadsRoot, heads->GetGPUVirtualAddress());
                commands->SetComputeRootUnorderedAccessView(DxrPpm::ErrorsRoot,
                                                            failures->GetGPUVirtualAddress());
                commands->SetComputeRootUnorderedAccessView(DxrPpm::VisibleRoot,
                                                            visible->GetGPUVirtualAddress());
                commands->SetComputeRootUnorderedAccessView(DxrPpm::PixelsRoot,
                                                            pixels->GetGPUVirtualAddress());
                commands->SetComputeRootConstantBufferView(DxrPpm::ControlsRoot,
                                                           batchBuffer->GetGPUVirtualAddress());
                const auto rays = MakeDispatch(shaderTable, count, 1);
                commands->DispatchRays(&rays);
                const D3D12_RESOURCE_BARRIER barriers[] = {
                    UavBarrier(nullptr),
                    TransitionBarrier(failures.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                      D3D12_RESOURCE_STATE_COPY_SOURCE)};
                commands->ResourceBarrier(UINT(std::size(barriers)), barriers);
                commands->CopyResource(readback.Get(), failures.Get());
                const auto restore = TransitionBarrier(failures.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
                                                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                commands->ResourceBarrier(1, &restore);
                CopyInstanceError(commands, instances);
                const auto restoreInstances =
                    TransitionBarrier(instances.Error.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
                                      D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                commands->ResourceBarrier(1, &restoreInstances);
                status = commands->Close();
                if (FAILED(status))
                {
                    error = HResultMessage("Closing DXR PPM batch", status);
                    return false;
                }
                if (!ExecuteAndWait(commands, error) || !CheckInstanceError(instances, error))
                    return false;
                const D3D12_RANGE readRange{0, sizeof(uint32_t)}, noWrite{0, 0};
                status = readback->Map(0, &readRange, &mapping);
                if (FAILED(status))
                {
                    error = HResultMessage("Mapping DXR PPM status", status);
                    return false;
                }
                uint32_t flags;
                std::memcpy(&flags, mapping, sizeof(flags));
                readback->Unmap(0, &noWrite);
                if (flags)
                {
                    error = "DXR PPM encountered invalid controls or nonfinite transport arithmetic.";
                    return false;
                }
                status = allocator->Reset();
                if (SUCCEEDED(status))
                    status = commands->Reset(allocator, nullptr);
                if (FAILED(status))
                {
                    error = HResultMessage("Resetting DXR PPM batch", status);
                    return false;
                }
                bindScene();
                return true;
            };
            const auto linear = [&](uint32_t stage, uint32_t total, uint32_t batch) {
                for (uint32_t first = 0; first < total; first += batch)
                    if (!dispatch(stage, first, std::min(batch, total - first)))
                        return false;
                return true;
            };
            if (!linear(DxrPpm::Clear, bucketCount, DxrPpm::LinearBatch) ||
                !linear(DxrPpm::Initialize, pixelCount, DxrPpm::LinearBatch))
                return false;
            for (uint32_t pass = 0; pass < constants.SamplesPerPixel; ++pass)
            {
                controls.Pass = pass;
                if (!linear(DxrPpm::Camera, pixelCount, DxrPpm::PixelBatch))
                    return false;
                for (uint32_t first = 0; first < options.PathsPerPass; first += lightCapacity)
                {
                    controls.LightBegin = first;
                    controls.LightCount = std::min(lightCapacity, options.PathsPerPass - first);
                    if (!linear(DxrPpm::Clear, bucketCount, DxrPpm::LinearBatch) ||
                        !dispatch(DxrPpm::Emit, 0, controls.LightCount) ||
                        !linear(DxrPpm::Index, controls.LightCount * slots, DxrPpm::LinearBatch) ||
                        !linear(DxrPpm::Gather, pixelCount, DxrPpm::PixelBatch))
                        return false;
                }
                // Gather all batches before shrinking R or updating N and tau.
                if (!linear(DxrPpm::Update, pixelCount, DxrPpm::LinearBatch))
                    return false;
            }
            return true;
        }

        bool DispatchPhotons(const PreparedEmissionScene& emission, const GpuConstants& constants,
                             const PhotonMappingOptions& options, ID3D12CommandAllocator* allocator,
                             ID3D12GraphicsCommandList4* commands, ID3D12Resource* shaderTable,
                             const InstanceResources& instances, const std::function<void()>& bindScene,
                             const Detail::CancellationProbe& cancellation, std::string& error) const
        {
            const uint32_t pixels = constants.Width * constants.Height;
            const uint32_t slotsPerPath = constants.MaxReflectionBounces + 1;
            const uint32_t slots = options.PhotonPaths * slotsPerPath;
            uint32_t buckets = 1;
            while (buckets < options.PhotonPaths)
                buckets *= 2;
            const uint64_t recordBytes = uint64_t(slots) * DxrPhoton::RecordStride;
            const uint64_t headBytes = uint64_t(buckets) * sizeof(uint32_t);
            ComPtr<IDXGIAdapter3> memoryAdapter;
            if (SUCCEEDED(Adapter.As(&memoryAdapter)))
            {
                DXGI_QUERY_VIDEO_MEMORY_INFO memory{};
                if (SUCCEEDED(
                        memoryAdapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &memory)))
                {
                    const uint64_t available =
                        memory.Budget > memory.CurrentUsage ? memory.Budget - memory.CurrentUsage : 0;
                    if (recordBytes + headBytes > available)
                    {
                        error = "DXR photon buffers exceed the available video-memory budget.";
                        return false;
                    }
                }
            }
            ComPtr<ID3D12Resource> sources, records, heads, failures, readback;
            if (!CreateUploadBuffer(emission.Sources.data(), emission.Sources.size() * sizeof(PackedEmitter),
                                    sources, error) ||
                !CreateBuffer(recordBytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                              D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, records, error) ||
                !CreateBuffer(headBytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                              D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, heads, error) ||
                !CreateBuffer(sizeof(uint32_t), D3D12_HEAP_TYPE_DEFAULT,
                              D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                              D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, failures, error) ||
                !CreateBuffer(sizeof(uint32_t), D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST,
                              D3D12_RESOURCE_FLAG_NONE, readback, error))
                return false;
            constexpr uint32_t CausticsEnabled = 1, IndirectEnabled = 2;
            DxrPhoton::Controls controls{{emission.Center.X, emission.Center.Y, emission.Center.Z},
                                         emission.Radius,
                                         double(emission.Radius) * double(options.RadiusScale),
                                         uint32_t(emission.Sources.size()),
                                         options.PhotonPaths,
                                         slotsPerPath,
                                         buckets,
                                         options.Seed ^ constants.SamplingSeed,
                                         (options.Caustics ? CausticsEnabled : 0u) |
                                             (options.Indirect ? IndirectEnabled : 0u),
                                         0,
                                         0,
                                         0,
                                         0};
            const auto dispatch = [&](uint32_t stage, uint32_t begin, uint32_t count) {
                if (cancellation())
                    return false;
                controls.Stage = stage;
                controls.Begin = begin;
                controls.Count = count;
                commands->SetComputeRootShaderResourceView(DxrPhoton::EmitterRoot,
                                                           sources->GetGPUVirtualAddress());
                commands->SetComputeRootUnorderedAccessView(DxrPhoton::RecordsRoot,
                                                            records->GetGPUVirtualAddress());
                commands->SetComputeRootUnorderedAccessView(DxrPhoton::HeadsRoot,
                                                            heads->GetGPUVirtualAddress());
                commands->SetComputeRootUnorderedAccessView(DxrPhoton::ErrorRoot,
                                                            failures->GetGPUVirtualAddress());
                commands->SetComputeRoot32BitConstants(DxrPhoton::ControlsRoot,
                                                       sizeof(controls) / sizeof(uint32_t), &controls, 0);
                const auto rays = MakeDispatch(shaderTable, count, 1);
                commands->DispatchRays(&rays);
                // Global UAV publication for this offline photon stage includes
                // photon links, film and statistics.
                const D3D12_RESOURCE_BARRIER publish[] = {
                    UavBarrier(nullptr),
                    TransitionBarrier(failures.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                      D3D12_RESOURCE_STATE_COPY_SOURCE)};
                commands->ResourceBarrier(UINT(std::size(publish)), publish);
                commands->CopyResource(readback.Get(), failures.Get());
                const auto restore = TransitionBarrier(failures.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
                                                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                commands->ResourceBarrier(1, &restore);
                CopyInstanceError(commands, instances);
                const auto restoreInstances =
                    TransitionBarrier(instances.Error.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
                                      D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                commands->ResourceBarrier(1, &restoreInstances);
                HRESULT status = commands->Close();
                if (FAILED(status))
                {
                    error = HResultMessage("Closing DXR photon batch", status);
                    return false;
                }
                if (!ExecuteAndWait(commands, error) || !CheckInstanceError(instances, error))
                    return false;
                void* data = nullptr;
                const D3D12_RANGE range{0, sizeof(uint32_t)}, noWrite{0, 0};
                status = readback->Map(0, &range, &data);
                if (FAILED(status))
                {
                    error = HResultMessage("Mapping DXR photon status", status);
                    return false;
                }
                uint32_t flags;
                std::memcpy(&flags, data, sizeof(flags));
                readback->Unmap(0, &noWrite);
                if (flags)
                {
                    error =
                        "DXR photon mapping encountered invalid controls or nonfinite transport arithmetic.";
                    return false;
                }
                status = allocator->Reset();
                if (SUCCEEDED(status))
                    status = commands->Reset(allocator, nullptr);
                if (FAILED(status))
                {
                    error = HResultMessage("Resetting DXR photon batch", status);
                    return false;
                }
                bindScene();
                return true;
            };
            const uint32_t clearCount = std::max(pixels, buckets);
            for (uint32_t begin = 0; begin < clearCount; begin += DxrPhoton::IndexBatch)
                if (!dispatch(DxrPhoton::Clear, begin, std::min(DxrPhoton::IndexBatch, clearCount - begin)))
                    return false;
            for (uint32_t begin = 0; begin < options.PhotonPaths; begin += DxrPhoton::TransportBatch)
                if (!dispatch(DxrPhoton::Emit, begin,
                              std::min(DxrPhoton::TransportBatch, options.PhotonPaths - begin)))
                    return false;
            for (uint32_t begin = 0; begin < slots; begin += DxrPhoton::IndexBatch)
                if (!dispatch(DxrPhoton::Index, begin, std::min(DxrPhoton::IndexBatch, slots - begin)))
                    return false;
            for (uint32_t sample = 0; sample < constants.SamplesPerPixel; ++sample)
            {
                controls.Sample = sample;
                for (uint32_t begin = 0; begin < pixels; begin += DxrPhoton::TransportBatch)
                    if (!dispatch(DxrPhoton::Camera, begin,
                                  std::min(DxrPhoton::TransportBatch, pixels - begin)))
                        return false;
            }
            return true;
        }

        bool DispatchVcm(const PreparedEmissionScene& emission, const GpuConstants& constants,
                         const RenderSettings& settings, ID3D12CommandAllocator* allocator,
                         ID3D12GraphicsCommandList4* commands, ID3D12Resource* shaderTable,
                         const InstanceResources& instances, const std::function<void()>& bindScene,
                         const Detail::CancellationProbe& cancellation, RenderStatistics& totals,
                         std::string& error) const
        {
            namespace V = DxrVcm;
            V::DispatchPlan plan;
            if (!V::PlanDispatch(constants.Width, constants.Height, constants.FullWidth, constants.FullHeight,
                                 settings.Vcm.LightPathsPerPass, constants.MaxReflectionBounces, plan, error))
                return false;
            constexpr uint64_t ControlAlignment = D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT;
            constexpr uint64_t WalkOffset = 0, CouplingOffset = ControlAlignment, FrameOffset = 2 * ControlAlignment;
            constexpr uint64_t ControlBytes = 3 * ControlAlignment;
            constexpr uint64_t StatisticsBytes = V::TransportBatch * sizeof(V::WalkStatistics);
            constexpr uint64_t ReadbackStatisticsOffset = sizeof(uint32_t);
            static_assert(sizeof(V::WalkControls) <= ControlAlignment && sizeof(V::CouplingControls) <= ControlAlignment &&
                          sizeof(V::FrameControls) <= ControlAlignment, "VCM control blocks must not overlap.");
            const std::array<uint8_t, ControlBytes> initialControls{};
            const V::CouplingQuery unusedQuery{};
            ComPtr<ID3D12Resource> sources, controls, queries, contributions, vertices, paths, failures;
            ComPtr<ID3D12Resource> statistics, index, pixels, splats, readback;
            auto uav = [&](uint64_t bytes, ComPtr<ID3D12Resource>& buffer) {
                return CreateBuffer(bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, buffer, error);
            };
            if (!CreateUploadBuffer(emission.Sources.data(), emission.Sources.size() * sizeof(PackedEmitter), sources, error) ||
                !CreateUploadBuffer(initialControls.data(), initialControls.size(), controls, error) ||
                !CreateUploadBuffer(&unusedQuery, sizeof(unusedQuery), queries, error) ||
                !uav(sizeof(V::Contribution), contributions) ||
                !uav(uint64_t(plan.VertexCapacity) * sizeof(V::PathVertex), vertices) ||
                !uav(uint64_t(plan.PathCapacity) * sizeof(V::Path), paths) ||
                !uav(StatisticsBytes, statistics) ||
                !uav(uint64_t(plan.IndexCapacity) * sizeof(V::IndexEntry), index) ||
                !uav(uint64_t(plan.Pixels) * sizeof(V::FramePixel), pixels) ||
                !uav(uint64_t(plan.IndexCapacity) * sizeof(V::Splat), splats) ||
                !CreateBuffer(sizeof(uint32_t), D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST,
                              D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, failures, error) ||
                !CreateBuffer(ReadbackStatisticsOffset + StatisticsBytes, D3D12_HEAP_TYPE_READBACK,
                              D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE, readback, error))
                return false;
            // This upload remains immutable until its copy completes. Controls
            // are rewritten per batch, so they cannot double as the zero source.
            commands->CopyBufferRegion(failures.Get(), 0, queries.Get(), 0, sizeof(uint32_t));
            auto initialized = TransitionBarrier(failures.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                                                 D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            commands->ResourceBarrier(1, &initialized);
            V::WalkControls walk{};
            StoreFloat3(walk.EmissionCenter, emission.Center);
            walk.EmissionRadius = emission.Radius;
            walk.EmitterCount = walk.SourceCapacity = uint32_t(emission.Sources.size());
            walk.PopulationSeed = settings.Vcm.Seed ^ constants.SamplingSeed;
            V::CouplingControls coupling{};
            coupling.MaximumVertices = plan.VertexLimit;
            coupling.Heuristic = static_cast<uint32_t>(settings.Transport.Mis);
            coupling.NextEventEstimation = settings.Transport.NextEventEstimation ? 1u : 0u;
            coupling.QueryCapacity = coupling.ContributionCapacity = 1;
            V::FrameControls frame{};
            frame.InitialRadius = double(emission.Radius) * double(settings.Vcm.InitialRadiusScale);
            frame.Alpha = settings.Vcm.Alpha;
            const auto submit = [&](const V::DispatchStep& step, uint32_t& unfinished) {
                if (cancellation()) return false;
                void* mapped = nullptr;
                const D3D12_RANGE noRead{0, 0}, noWrite{0, 0};
                HRESULT status = controls->Map(0, &noRead, &mapped);
                if (FAILED(status))
                {
                    error = HResultMessage("Mapping DXR VCM controls", status);
                    return false;
                }
                auto* bytes = static_cast<uint8_t*>(mapped);
                std::memcpy(bytes + WalkOffset, &step.Walk, sizeof(step.Walk));
                std::memcpy(bytes + CouplingOffset, &coupling, sizeof(coupling));
                std::memcpy(bytes + FrameOffset, &step.Frame, sizeof(step.Frame));
                const D3D12_RANGE written{0, FrameOffset + sizeof(frame)};
                controls->Unmap(0, &written);
                commands->SetComputeRootShaderResourceView(V::SourceRoot, sources->GetGPUVirtualAddress());
                commands->SetComputeRootUnorderedAccessView(V::VerticesRoot, vertices->GetGPUVirtualAddress());
                commands->SetComputeRootUnorderedAccessView(V::PathsRoot, paths->GetGPUVirtualAddress());
                commands->SetComputeRootUnorderedAccessView(V::ErrorsRoot, failures->GetGPUVirtualAddress());
                commands->SetComputeRootUnorderedAccessView(V::StatisticsRoot, statistics->GetGPUVirtualAddress());
                commands->SetComputeRootConstantBufferView(V::ControlsRoot, controls->GetGPUVirtualAddress() + WalkOffset);
                commands->SetComputeRootShaderResourceView(V::QueriesRoot, queries->GetGPUVirtualAddress());
                commands->SetComputeRootUnorderedAccessView(V::ContributionsRoot, contributions->GetGPUVirtualAddress());
                commands->SetComputeRootConstantBufferView(V::CouplingControlsRoot, controls->GetGPUVirtualAddress() + CouplingOffset);
                commands->SetComputeRootUnorderedAccessView(V::IndexRoot, index->GetGPUVirtualAddress());
                commands->SetComputeRootUnorderedAccessView(V::FramePixelsRoot, pixels->GetGPUVirtualAddress());
                commands->SetComputeRootUnorderedAccessView(V::SplatsRoot, splats->GetGPUVirtualAddress());
                commands->SetComputeRootConstantBufferView(V::FrameControlsRoot, controls->GetGPUVirtualAddress() + FrameOffset);
                auto dispatch = MakeDispatch(shaderTable, step.Count, 1);
                const UINT record = step.Walk.Stage == V::CameraStage ? 0 :
                    VcmRayGenerationRecord + step.Walk.Stage - V::LightStage;
                dispatch.RayGenerationShaderRecord.StartAddress += uint64_t(record) * ShaderRecordSize;
                commands->DispatchRays(&dispatch);
                const D3D12_RESOURCE_BARRIER publish[] = {UavBarrier(nullptr),
                    TransitionBarrier(failures.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE)};
                commands->ResourceBarrier(UINT(std::size(publish)), publish);
                commands->CopyBufferRegion(readback.Get(), 0, failures.Get(), 0, sizeof(uint32_t));
                auto restore = TransitionBarrier(failures.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
                                                 D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                commands->ResourceBarrier(1, &restore);
                if (step.ReadsStatistics)
                {
                    auto copy = TransitionBarrier(statistics.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                                  D3D12_RESOURCE_STATE_COPY_SOURCE);
                    commands->ResourceBarrier(1, &copy);
                    commands->CopyBufferRegion(readback.Get(), ReadbackStatisticsOffset, statistics.Get(), 0,
                                              uint64_t(step.Count) * sizeof(V::WalkStatistics));
                    restore = TransitionBarrier(statistics.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
                                                D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                    commands->ResourceBarrier(1, &restore);
                }
                CopyInstanceError(commands, instances);
                restore = TransitionBarrier(instances.Error.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
                                            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                commands->ResourceBarrier(1, &restore);
                status = commands->Close();
                if (FAILED(status))
                {
                    error = HResultMessage("Closing DXR VCM batch", status);
                    return false;
                }
                if (!ExecuteAndWait(commands, error) || !CheckInstanceError(instances, error))
                {
                    error = "DXR VCM stage " + std::to_string(step.Walk.Stage) + " pass " +
                            std::to_string(step.Walk.PassIndex) + " batch " +
                            std::to_string(step.Frame.WorkBegin) + " count " + std::to_string(step.Count) +
                            ": " + error;
                    return false;
                }
                const SIZE_T readBytes = step.ReadsStatistics
                    ? ReadbackStatisticsOffset + SIZE_T(step.Count) * sizeof(V::WalkStatistics) : sizeof(uint32_t);
                const D3D12_RANGE range{0, readBytes};
                status = readback->Map(0, &range, &mapped);
                if (FAILED(status))
                {
                    error = HResultMessage("Mapping DXR VCM status", status);
                    return false;
                }
                uint32_t flags;
                std::memcpy(&flags, mapped, sizeof(flags));
                bool valid = flags == 0;
                if (step.ReadsStatistics && valid)
                {
                    for (uint32_t lane = 0; lane < step.Count; ++lane)
                    {
                        V::WalkStatistics sample;
                        std::memcpy(&sample, static_cast<uint8_t*>(mapped) + ReadbackStatisticsOffset +
                                               lane * sizeof(sample), sizeof(sample));
                        const auto count = [](const uint32_t* words) {
                            return uint64_t(words[0]) | (uint64_t(words[1]) << 32);
                        };
                        totals.PrimaryRays += count(sample.Primary);
                        totals.ShadowRays += count(sample.Shadow);
                        totals.ReflectionRays += count(sample.Reflection);
                        valid &= sample.Completed[1] == 0 && sample.Completed[0] <= 1;
                        if (step.Walk.Stage == V::MergeStage || step.Walk.Stage == V::CommitStage)
                            unfinished += sample.Completed[0] == 0 ? 1u : 0u;
                        else if (step.Walk.Stage != V::EstimateStage)
                            valid &= sample.Completed[0] == 1;
                    }
                }
                readback->Unmap(0, &noWrite);
                if (!valid)
                {
                    error = "DXR VCM stage " + std::to_string(step.Walk.Stage) +
                            " reported invalid bounds, progress or nonfinite transport (flags " + std::to_string(flags) + ").";
                    return false;
                }
                // Fence completion precedes every upload overwrite, allocator
                // reset and cancellation exit, preserving native buffer lifetimes.
                status = allocator->Reset();
                if (SUCCEEDED(status)) status = commands->Reset(allocator, nullptr);
                if (FAILED(status))
                {
                    error = HResultMessage("Resetting DXR VCM commands", status);
                    return false;
                }
                bindScene();
                return true;
            };
            if (!V::RunDispatches(plan, constants.SamplesPerPixel, walk, frame, submit, error)) return false;
            totals.CompletedPixels = plan.Pixels;
            return true;
        }

        bool EnsureSpectralPreparation(std::string& error) const
        {
            if (SpectralState)
                return true;
            ComPtr<ID3D12RootSignature> root;
            HRESULT status =
                Device->CreateRootSignature(0, Detail::DxrSpectralRootSignature,
                                            sizeof(Detail::DxrSpectralRootSignature), IID_PPV_ARGS(&root));
            if (FAILED(status))
            {
                error = HResultMessage("Creating DXR spectral preparation root", status);
                return false;
            }
            const D3D12_EXPORT_DESC entries[]{{L"SpectralPrepareResponse", nullptr, D3D12_EXPORT_FLAG_NONE},
                                              {L"SpectralPreparePalette", nullptr, D3D12_EXPORT_FLAG_NONE}};
            const D3D12_DXIL_LIBRARY_DESC library{
                {Detail::DxrSpectralPrepareDxil, sizeof(Detail::DxrSpectralPrepareDxil)},
                UINT(std::size(entries)),
                entries};
            const D3D12_RAYTRACING_SHADER_CONFIG shader{0, 0};
            const D3D12_GLOBAL_ROOT_SIGNATURE global{root.Get()};
            const D3D12_RAYTRACING_PIPELINE_CONFIG recursion{1};
            const D3D12_STATE_SUBOBJECT parts[]{
                {D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY, &library},
                {D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_SHADER_CONFIG, &shader},
                {D3D12_STATE_SUBOBJECT_TYPE_GLOBAL_ROOT_SIGNATURE, &global},
                {D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_PIPELINE_CONFIG, &recursion}};
            const D3D12_STATE_OBJECT_DESC description{D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE,
                                                      UINT(std::size(parts)), parts};
            ComPtr<ID3D12StateObject> state;
            status = Device->CreateStateObject(&description, IID_PPV_ARGS(&state));
            if (FAILED(status))
            {
                error = HResultMessage("Creating DXR spectral preparation pipeline", status);
                return false;
            }
            ComPtr<ID3D12StateObjectProperties> properties;
            status = state.As(&properties);
            if (FAILED(status))
            {
                error = HResultMessage("Reading DXR spectral shader identifiers", status);
                return false;
            }
            std::array<uint8_t, std::size(entries) * ShaderRecordSize> records{};
            for (size_t index = 0; index < std::size(entries); ++index)
            {
                const void* identifier = properties->GetShaderIdentifier(entries[index].Name);
                if (!identifier)
                {
                    error = "A DXR spectral preparation ray-generation identifier is missing.";
                    return false;
                }
                std::memcpy(records.data() + index * ShaderRecordSize, identifier,
                            D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
            }
            ComPtr<ID3D12Resource> table;
            if (!CreateUploadBuffer(records.data(), records.size(), table, error))
                return false;
            SpectralRoot = std::move(root);
            SpectralTable = std::move(table);
            SpectralState = std::move(state);
            return true;
        }

        bool PrepareSpectralCoefficients(const SpectralSceneData::Request* source, uint32_t count,
                                         const Detail::CancellationProbe& cancelled,
                                         ComPtr<ID3D12Resource>& output, std::string& error) const
        {
            namespace S = SpectralSceneData;
            constexpr uint32_t QuadratureEntries = 256, BatchCapacity = 8192, Converged = 0;
            constexpr UINT SourceRoot = 0, ResponseRoot = 1, ResponseUavRoot = 5, CoefficientsUavRoot = 6;
            constexpr uint64_t ResponseRecord = 0, PaletteRecord = 1;
            if (count == 0)
            {
                error = "DXR spectral preparation needs a nonempty palette.";
                return false;
            }
            if (!EnsureSpectralPreparation(error))
                return false;
            ComPtr<ID3D12Resource> sources, response, coefficients, readback;
            if (!CreateUploadBuffer(source, uint64_t(count) * sizeof(S::Request), sources, error) ||
                !CreateBuffer(QuadratureEntries * sizeof(float) * 4, D3D12_HEAP_TYPE_DEFAULT,
                              D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                              D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, response, error) ||
                !CreateBuffer(uint64_t(count) * sizeof(S::GpuSpectrum), D3D12_HEAP_TYPE_DEFAULT,
                              D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                              D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, coefficients, error) ||
                !CreateBuffer(uint64_t(std::min(count, BatchCapacity)) * sizeof(S::GpuSpectrum),
                              D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST,
                              D3D12_RESOURCE_FLAG_NONE, readback, error))
                return false;
            ComPtr<ID3D12CommandAllocator> allocator;
            ComPtr<ID3D12GraphicsCommandList4> commands;
            HRESULT status =
                Device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator));
            if (SUCCEEDED(status))
                status = Device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(),
                                                   nullptr, IID_PPV_ARGS(&commands));
            if (FAILED(status))
            {
                error = HResultMessage("Creating DXR spectral commands", status);
                return false;
            }
            if (cancelled())
            {
                error = "DXR spectral preparation cancelled.";
                return false;
            }
            const auto dispatch = [&](uint64_t record, uint32_t width) {
                commands->SetPipelineState1(SpectralState.Get());
                D3D12_DISPATCH_RAYS_DESC rays{};
                rays.RayGenerationShaderRecord = {
                    SpectralTable->GetGPUVirtualAddress() + record * ShaderRecordSize, ShaderRecordSize};
                rays.Width = width;
                rays.Height = rays.Depth = 1;
                commands->DispatchRays(&rays);
            };
            commands->SetComputeRootSignature(SpectralRoot.Get());
            commands->SetComputeRootUnorderedAccessView(ResponseUavRoot, response->GetGPUVirtualAddress());
            dispatch(ResponseRecord, QuadratureEntries);
            auto responseReady = TransitionBarrier(response.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            commands->ResourceBarrier(1, &responseReady);
            status = commands->Close();
            if (FAILED(status))
            {
                error = HResultMessage("Closing DXR spectral quadrature commands", status);
                return false;
            }
            if (!ExecuteAndWait(commands.Get(), error))
                return false;

            for (uint32_t begin = 0; begin < count;)
            {
                if (cancelled())
                {
                    error = "DXR spectral fitting cancelled.";
                    return false;
                }
                const uint32_t batch = std::min(BatchCapacity, count - begin);
                status = allocator->Reset();
                if (SUCCEEDED(status))
                    status = commands->Reset(allocator.Get(), nullptr);
                if (FAILED(status))
                {
                    error = HResultMessage("Resetting DXR spectral fit commands", status);
                    return false;
                }
                commands->SetComputeRootSignature(SpectralRoot.Get());
                commands->SetComputeRootShaderResourceView(
                    SourceRoot, sources->GetGPUVirtualAddress() + uint64_t(begin) * sizeof(S::Request));
                commands->SetComputeRootShaderResourceView(ResponseRoot, response->GetGPUVirtualAddress());
                commands->SetComputeRootUnorderedAccessView(CoefficientsUavRoot,
                                                            coefficients->GetGPUVirtualAddress() +
                                                                uint64_t(begin) * sizeof(S::GpuSpectrum));
                dispatch(PaletteRecord, batch);
                auto copy = TransitionBarrier(coefficients.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                              D3D12_RESOURCE_STATE_COPY_SOURCE);
                commands->ResourceBarrier(1, &copy);
                const size_t bytes = size_t(batch) * sizeof(S::GpuSpectrum);
                commands->CopyBufferRegion(readback.Get(), 0, coefficients.Get(),
                                           uint64_t(begin) * sizeof(S::GpuSpectrum), bytes);
                const auto next = begin + batch == count ? D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
                                                         : D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
                auto restore = TransitionBarrier(coefficients.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE, next);
                commands->ResourceBarrier(1, &restore);
                status = commands->Close();
                if (FAILED(status))
                {
                    error = HResultMessage("Closing DXR spectral fit commands", status);
                    return false;
                }
                if (!ExecuteAndWait(commands.Get(), error))
                    return false;
                void* mapped = nullptr;
                const D3D12_RANGE range{0, bytes}, noWrite{0, 0};
                status = readback->Map(0, &range, &mapped);
                if (FAILED(status))
                {
                    error = HResultMessage("Reading DXR spectral coefficient status", status);
                    return false;
                }
                const auto* fits = static_cast<const S::GpuSpectrum*>(mapped);
                uint32_t failed = batch;
                for (uint32_t index = 0; index < batch; ++index)
                {
                    const auto& value = fits[index];
                    if (value.Status != Converged || !std::isfinite(value.Quadratic) ||
                        !std::isfinite(value.Linear) || !std::isfinite(value.Offset) ||
                        !std::isfinite(value.Scale) || !std::isfinite(value.Constant) ||
                        !std::isfinite(value.Error) || value.Scale < 0 || value.Constant < 0 ||
                        value.Error < 0)
                    {
                        failed = index;
                        break;
                    }
                }
                readback->Unmap(0, &noWrite);
                if (failed != batch)
                {
                    error = "DXR spectral fit failed at entry " + std::to_string(begin + failed);
                    return false;
                }
                begin += batch;
            }
            if (cancelled())
            {
                error = "DXR spectral coefficient publication cancelled.";
                return false;
            }
            output = std::move(coefficients);
            return true;
        }

        bool UpdateSpectralMedium(SpectralSnapshot& snapshot, const MediumOptions& medium,
                                  const Detail::CancellationProbe& cancelled, std::string& error) const
        {
            if (snapshot.Medium == medium)
                return true;
            namespace S = SpectralSceneData;
            const S::Request sources[]{{medium.Enabled ? medium.Absorption : Float3(), S::Role::Nonnegative},
                                       {medium.Enabled ? medium.Scattering : Float3(), S::Role::Nonnegative}};
            ComPtr<ID3D12Resource> candidate;
            if (!PrepareSpectralCoefficients(sources, uint32_t(std::size(sources)), cancelled, candidate,
                                             error))
                return false;
            if (cancelled())
            {
                error = "DXR spectral medium publication cancelled.";
                return false;
            }
            ComPtr<ID3D12CommandAllocator> allocator;
            ComPtr<ID3D12GraphicsCommandList4> commands;
            HRESULT status =
                Device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator));
            if (SUCCEEDED(status))
                status = Device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(),
                                                   nullptr, IID_PPV_ARGS(&commands));
            if (FAILED(status))
            {
                error = HResultMessage("Creating DXR spectral medium publication", status);
                return false;
            }
            const D3D12_RESOURCE_BARRIER transitions[]{
                TransitionBarrier(candidate.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                  D3D12_RESOURCE_STATE_COPY_SOURCE),
                TransitionBarrier(snapshot.Coefficients.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                  D3D12_RESOURCE_STATE_COPY_DEST)};
            commands->ResourceBarrier(UINT(std::size(transitions)), transitions);
            commands->CopyBufferRegion(snapshot.Coefficients.Get(),
                                       uint64_t(snapshot.SceneSourceCount) * sizeof(S::GpuSpectrum),
                                       candidate.Get(), 0, std::size(sources) * sizeof(S::GpuSpectrum));
            const auto ready = TransitionBarrier(snapshot.Coefficients.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                                                 D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            commands->ResourceBarrier(1, &ready);
            status = commands->Close();
            if (FAILED(status))
            {
                error = HResultMessage("Closing DXR spectral medium publication", status);
                return false;
            }
            if (!ExecuteAndWait(commands.Get(), error))
                return false;
            snapshot.Medium = medium;
            return true;
        }

        bool PrepareSpectralScene(const Scene& scene, const std::shared_ptr<const Scene>& owner,
                                  const MediumOptions& medium, const Detail::CancellationProbe& cancelled,
                                  std::shared_ptr<SpectralSnapshot>& output, std::string& error) const
        {
            if (owner && SpectralOwner.lock() == owner && SpectralCache)
            {
                if (!UpdateSpectralMedium(*SpectralCache, medium, cancelled, error))
                    return false;
                output = SpectralCache;
                return true;
            }
            SpectralSceneData::GpuScene inputs;
            if (DxrSpectralHost::BuildGpuInputs(scene, medium, cancelled, inputs, error) !=
                RenderStatus::Completed)
                return false;
            auto candidate = std::make_shared<SpectralSnapshot>();
            if (!PrepareSpectralCoefficients(inputs.Sources.data(), uint32_t(inputs.Sources.size()),
                                             cancelled, candidate->Coefficients, error) ||
                !CreateUploadBuffer(inputs.References.data(), inputs.References.size() * sizeof(uint32_t),
                                    candidate->References, error))
                return false;
            candidate->SceneSourceCount = inputs.SceneSourceCount;
            candidate->TriangleReferences = std::move(inputs.TriangleReferences);
            candidate->Medium = medium;
            if (cancelled())
            {
                error = "DXR spectral scene publication cancelled.";
                return false;
            }
            if (owner)
            {
                SpectralCache = candidate;
                SpectralOwner = owner;
            }
            output = std::move(candidate);
            return true;
        }

        RenderResult RenderNative(const Scene& scene, const std::shared_ptr<const Scene>& owner,
                                  const Camera& camera, const RenderSettings& settings,
                                  const Detail::CancellationProbe& cancellation) const
        {
            RenderResult renderResult;
            const bool knownScene = owner && Resident && ResidentOwner.lock() == owner;
            if (!ValidateDxrInput(scene, camera, settings, renderResult.Error, knownScene))
            {
                return renderResult;
            }
            if (!Detail::ValidateMaterialTransport(scene, BackendType::DXR,
                    settings.Integrator, renderResult.Error)) return renderResult;
            if (cancellation()) return Detail::MakeCancelledResult();

            std::shared_ptr<SpectralSnapshot> spectral;
            if (settings.Radiometry == RadiometryMode::Spectral &&
                !PrepareSpectralScene(scene, owner, settings.Medium, cancellation, spectral,
                                      renderResult.Error))
            {
                if (cancellation())
                    return Detail::MakeCancelledResult();
                return renderResult;
            }

            const bool bidirectional = settings.Integrator == Algorithm::BidirectionalPathTracing;
            const bool metropolis = settings.Integrator == Algorithm::MetropolisLightTransport;
            const bool photon = settings.Integrator == Algorithm::PhotonMapping;
            const bool progressive = settings.Integrator == Algorithm::ProgressivePhotonMapping;
            const bool vcm = settings.Integrator == Algorithm::VertexConnectionMerging;
            const bool pointGi = settings.Integrator == Algorithm::PointBasedGlobalIllumination;
            PreparedEmissionScene emission;
            if (bidirectional || metropolis || photon || progressive || vcm || pointGi)
            {
                if (bidirectional && (settings.Adaptive.Enabled || settings.Transport.RussianRoulette ||
                    !settings.Transport.NextEventEstimation || !settings.ShadowsEnabled || settings.ShadowStrength != 1))
                    return Detail::MakeFailedResult("DXR BDPT requires fixed samples/depth, NEE and full physical visibility.");
                if (metropolis && (settings.Adaptive.Enabled ||
                    settings.MaxReflectionBounces > DxrMlt::MaximumReflections ||
                    !settings.ShadowsEnabled || settings.ShadowStrength != 1))
                    return Detail::MakeFailedResult("DXR MLT requires fixed samples, at most 32 reflections and full physical visibility.");
                emission = DxrEmission::PrepareEmitters(scene);
                if (photon && (settings.Adaptive.Enabled || settings.MaxReflectionBounces > DxrPhoton::MaximumReflections ||
                    settings.Transport.Mis != MisHeuristic::None || !settings.ShadowsEnabled || settings.ShadowStrength != 1))
                    return Detail::MakeFailedResult("DXR photon mapping requires bounded depth, fixed samples, no MIS and full visibility.");
                if (progressive && (settings.Adaptive.Enabled || settings.MaxReflectionBounces > DxrPpm::MaximumReflections ||
                    settings.Transport.Mis != MisHeuristic::None || !settings.ShadowsEnabled || settings.ShadowStrength != 1))
                    return Detail::MakeFailedResult("DXR PPM requires bounded depth, fixed passes, no MIS and full visibility.");
                if (vcm && (settings.Adaptive.Enabled || settings.Transport.RussianRoulette ||
                    settings.MaxReflectionBounces > DxrVcm::MaximumReflections ||
                    !settings.ShadowsEnabled || settings.ShadowStrength != 1))
                    return Detail::MakeFailedResult("DXR VCM requires fixed passes/depth, at most 32 reflections and full physical visibility.");
                if (pointGi && (settings.Adaptive.Enabled || settings.MaxReflectionBounces > DxrPointGi::MaximumDepth ||
                    !settings.ShadowsEnabled || settings.ShadowStrength != 1))
                    return Detail::MakeFailedResult("DXR Point GI requires bounded depth, fixed samples and full visibility.");
                constexpr uint32_t PointGiSourceLimit = 1u << 24;
                if (pointGi && emission.Sources.size() > PointGiSourceLimit)
                    return Detail::MakeFailedResult("DXR Point GI source count exceeds uniform sampling precision.");
                for (const auto& source : emission.Sources)
                    if (!(source.Flags & EmitterCastsShadows))
                        return Detail::MakeFailedResult("DXR bidirectional, photon and point GI integrators require shadow casting on every emitter.");
            }

            DxrPointGi::SurfaceCache pointGiGeometry;
            if (pointGi && settings.MaxReflectionBounces > 0 &&
                !DxrPointGi::BuildSurfaceCache(scene, settings.PointGi, cancellation, pointGiGeometry, renderResult.Error))
            {
                if (cancellation()) return Detail::MakeCancelledResult();
                return renderResult;
            }

            const Pipeline* pipeline = nullptr;
            if (!EnsureIntegratorPipeline(settings.Integrator, pipeline, renderResult.Error)) return renderResult;
            if (cancellation()) return Detail::MakeCancelledResult();

            // An immutable shared owner is the scene identity. Camera/sample
            // changes reuse upload buffers and hardware BLAS/TLAS; a new owner
            // invalidates them. Spectral packing and software builders retain
            // their independent paths. Mutable error/scratch buffers stay per job.
            const bool residentEligible = owner && !spectral &&
                settings.Acceleration == AccelerationType::Bvh &&
                settings.BvhPolicy == BvhBuildPolicy::Default;
            const bool reuseResident = residentEligible && Resident && ResidentOwner.lock() == owner;
            auto resident = reuseResident ? Resident : std::make_shared<ResidentScene>();
            auto& vertices = resident->Vertices;
            auto& triangles = resident->Triangles;
            auto& spheres = resident->Spheres;
            auto& sphereBounds = resident->SphereBounds;
            auto& materials = resident->Materials;
            auto& lights = resident->Lights;
            auto& textureAtlas = resident->Textures;
            auto& instancePlan = resident->Instances;
            auto& vertexBuffer = resident->VertexBuffer;
            auto& triangleBuffer = resident->TriangleBuffer;
            auto& sphereBuffer = resident->SphereBuffer;
            auto& sphereBoundsBuffer = resident->SphereBoundsBuffer;
            auto& materialBuffer = resident->MaterialBuffer;
            auto& lightBuffer = resident->LightBuffer;
            auto& textureDescriptorBuffer = resident->TextureDescriptors;
            auto& textureTexelBuffer = resident->TextureTexels;
            if (!reuseResident)
            {
                if (!BuildSceneGeometry(scene,cancellation,instancePlan,vertices,triangles,spheres,sphereBounds,renderResult.Error))
                {
                    if (cancellation()) return Detail::MakeCancelledResult();
                    return renderResult;
                }
                if (spectral)
                {
                    for (auto& triangle : triangles)
                    {
                        const auto* a = triangle.Color0;
                        const auto* b = triangle.Color1;
                        const auto* c = triangle.Color2;
                        const auto link = spectral->TriangleReferences.find(
                            {a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]});
                        if (link == spectral->TriangleReferences.end())
                        {
                            renderResult.Error = "DXR packed triangle colors do not match the spectral scene.";
                            return renderResult;
                        }
                        triangle.Padding = link->second;
                    }
                }
                BuildShadingData(scene, materials, lights);
                if (!Detail::BuildRayTextureAtlas(
                        scene,
                        textureAtlas,
                        renderResult.Error))
                {
                    return renderResult;
                }

                if (!CreateUploadBuffer(
                        vertices.data(), vertices.size() * sizeof(GpuVertex), vertexBuffer, renderResult.Error)
                    || !CreateUploadBuffer(
                        triangles.data(), triangles.size() * sizeof(GpuTriangle), triangleBuffer, renderResult.Error)
                    || !CreateUploadBuffer(
                        spheres.data(), spheres.size() * sizeof(GpuSphere), sphereBuffer, renderResult.Error)
                    || !CreateUploadBuffer(
                        sphereBounds.data(), sphereBounds.size() * sizeof(D3D12_RAYTRACING_AABB),
                        sphereBoundsBuffer, renderResult.Error)
                    || !CreateUploadBuffer(
                        materials.data(), materials.size() * sizeof(GpuMaterial), materialBuffer, renderResult.Error)
                    || !CreateUploadBuffer(
                        lights.data(), lights.size() * sizeof(GpuLight), lightBuffer, renderResult.Error)
                    || !CreateUploadBuffer(
                        textureAtlas.Descriptors.data(),
                        textureAtlas.Descriptors.size()
                            * sizeof(Detail::PackedRayTextureDescriptor),
                        textureDescriptorBuffer,
                        renderResult.Error)
                    || !CreateUploadBuffer(
                        textureAtlas.Texels.data(),
                        textureAtlas.Texels.size() * sizeof(std::uint32_t),
                        textureTexelBuffer,
                        renderResult.Error))
                {
                    return renderResult;
                }
            }

            ComPtr<ID3D12CommandAllocator> allocator;
            ComPtr<ID3D12GraphicsCommandList> baseCommandList;
            ComPtr<ID3D12GraphicsCommandList4> commandList;
            HRESULT result = Device->CreateCommandAllocator(
                D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator));
            if (FAILED(result))
            {
                renderResult.Error = HResultMessage("ID3D12Device::CreateCommandAllocator", result);
                return renderResult;
            }
            result = Device->CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                allocator.Get(),
                nullptr,
                IID_PPV_ARGS(&baseCommandList));
            if (SUCCEEDED(result)) result = baseCommandList.As(&commandList);
            if (FAILED(result))
            {
                renderResult.Error = HResultMessage("ID3D12Device::CreateCommandList", result);
                return renderResult;
            }

            auto& acceleration = resident->Acceleration;
            const auto buildBegin = std::chrono::steady_clock::now();
            SoftwareAccelerationResources software;
            if (!PrepareSoftwareAcceleration(scene, {settings.Acceleration, settings.BvhPolicy}, cancellation, software, renderResult.Error, instancePlan))
            {
                if (cancellation()) return Detail::MakeCancelledResult();
                return renderResult;
            }
            InstanceResources instanceBuffers;
            if (!PrepareInstances(instancePlan,instanceBuffers,commandList.Get(),renderResult.Error)) return renderResult;
            if (settings.Acceleration == AccelerationType::Bvh
                && settings.BvhPolicy == BvhBuildPolicy::Default
                && !reuseResident
                && !BuildAcceleration(commandList.Get(), static_cast<uint32_t>(vertices.size()),
                static_cast<uint32_t>(spheres.size()), vertexBuffer.Get(), sphereBoundsBuffer.Get(),
                acceleration, renderResult.Error, instancePlan, cancellation))
            {
                if (cancellation()) return Detail::MakeCancelledResult();
                return renderResult;
            }
            const D3D12_GPU_VIRTUAL_ADDRESS topLevelAddress = acceleration.TopLevel
                ? acceleration.TopLevel->GetGPUVirtualAddress() : 0;
            result = commandList->Close();
            if (FAILED(result) || !ExecuteAndWait(commandList.Get(), renderResult.Error))
            {
                if (renderResult.Error.empty())
                {
                    renderResult.Error = HResultMessage("ID3D12GraphicsCommandList::Close", result);
                }
                return renderResult;
            }
            const auto buildEnd = std::chrono::steady_clock::now();
            if (cancellation()) return Detail::MakeCancelledResult();

            result = allocator->Reset();
            if (SUCCEEDED(result)) result = commandList->Reset(allocator.Get(), nullptr);
            if (FAILED(result))
            {
                renderResult.Error = HResultMessage("Resetting the DXR command list", result);
                return renderResult;
            }

            const uint64_t pixelCount = static_cast<uint64_t>(settings.Width) * settings.Height;
            const uint64_t outputSize = uint64_t(pixelCount) * sizeof(uint32_t);
            static_assert(sizeof(FilmPixel) == 20 && offsetof(FilmPixel, SampleCount) == 16,
                "DXR linear film layout must match LinearFilmPixel in native HLSL.");
            const uint64_t filmSize = pixelCount * sizeof(FilmPixel);
            constexpr uint64_t StatisticsSize = sizeof(GpuStatistics);
            ComPtr<ID3D12Resource> outputBuffer;
            ComPtr<ID3D12Resource> outputReadback;
            ComPtr<ID3D12Resource> filmBuffer;
            ComPtr<ID3D12Resource> filmReadback;
            ComPtr<ID3D12Resource> statisticsBuffer;
            ComPtr<ID3D12Resource> statisticsReadback;
            ComPtr<ID3D12Resource> zeroUpload;
            ComPtr<ID3D12Resource> budgetUpload;
            const std::array<uint8_t, StatisticsSize> zeroData{};
            if (!CreateBuffer(
                    outputSize,
                    D3D12_HEAP_TYPE_DEFAULT,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                    outputBuffer,
                    renderResult.Error)
                || !CreateBuffer(
                    outputSize,
                    D3D12_HEAP_TYPE_READBACK,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_FLAG_NONE,
                    outputReadback,
                    renderResult.Error)
                || !CreateBuffer(
                    filmSize, D3D12_HEAP_TYPE_DEFAULT,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, filmBuffer, renderResult.Error)
                || !CreateBuffer(
                    filmSize, D3D12_HEAP_TYPE_READBACK,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_FLAG_NONE, filmReadback, renderResult.Error)
                || !CreateBuffer(
                    StatisticsSize,
                    D3D12_HEAP_TYPE_DEFAULT,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                    statisticsBuffer,
                    renderResult.Error)
                || !CreateBuffer(
                    StatisticsSize,
                    D3D12_HEAP_TYPE_READBACK,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_FLAG_NONE,
                    statisticsReadback,
                    renderResult.Error)
                || !CreateUploadBuffer(
                    zeroData.data(), zeroData.size(), zeroUpload, renderResult.Error))
            {
                return renderResult;
            }

            if (settings.ProductionBudgets)
            {
                std::vector<FilmPixel> initial(static_cast<size_t>(pixelCount));
                for (size_t index = 0; index < initial.size(); ++index)
                    initial[index].SampleCount = settings.ProductionBudgets->Counts[index];
                if (!CreateUploadBuffer(initial.data(), size_t(filmSize), budgetUpload, renderResult.Error))
                    return renderResult;
                auto toCopy = TransitionBarrier(filmBuffer.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                               D3D12_RESOURCE_STATE_COPY_DEST);
                commandList->ResourceBarrier(1, &toCopy);
                commandList->CopyBufferRegion(filmBuffer.Get(), 0, budgetUpload.Get(), 0, filmSize);
                auto toTrace = TransitionBarrier(filmBuffer.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                                                D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                commandList->ResourceBarrier(1, &toTrace);
            }
            GpuConstants constants{};
            constants.InstanceCount = uint32_t(instancePlan.Records.size());
            constants.InstanceNodeCount = uint32_t(instancePlan.Nodes.size());
            StoreAccelerationConstants(constants, {settings.Acceleration, settings.BvhPolicy}, software);
            const Float3 forward = Normalize(camera.Target - camera.Position);
            const Float3 right = Normalize(Cross(forward, camera.Up));
            const Float3 up = Normalize(Cross(right, forward));
            constexpr float Pi = 3.14159265358979323846f;
            const float halfHeight = std::tan(camera.VerticalFieldOfViewDegrees * Pi / 360.0f);
            StoreFloat3(constants.CameraPosition, camera.Position);
            StoreFloat3(constants.CameraForward, forward);
            constants.LensRadius = camera.Lens.Radius;
            constants.FocusDistance = camera.Lens.FocusDistance;
            StoreFloat3(constants.CameraRight, right);
            StoreFloat3(constants.CameraUp, up);
            constants.Width = settings.Width;
            constants.FullWidth = FullImageWidth(settings);
            constants.FullHeight = FullImageHeight(settings);
            constants.TileX = settings.TileX;
            constants.TileY = settings.TileY;
            constants.Integrator = static_cast<uint32_t>(settings.Integrator);
            constants.TransportFlags = PackTransportFlags(settings.Transport);
            constexpr uint32_t SpectralTransportBit = 1u << 5;
            if (spectral)
                constants.TransportFlags |= SpectralTransportBit;
            constants.RouletteStartDepth = settings.Transport.RouletteStartDepth;
            constants.SamplingSequence = static_cast<uint32_t>(settings.Sampling);
            constants.SamplingSeed = settings.SamplingSeed;
            constants.AdaptiveEnabled = settings.Adaptive.Enabled ? 1u : 0u;
            constexpr uint32_t PreparedBudgetPhase = 4;
            constants.AdaptivePhase = settings.ProductionBudgets ? PreparedBudgetPhase : uint32_t(settings.SampleStream);
            constants.AdaptivePilotSamples = settings.Adaptive.PilotSamples;
            constants.AdaptiveMinimumSamples = settings.Adaptive.MinimumSamples;
            constants.AdaptiveRelativeError = settings.Adaptive.RelativeError;
            constants.AdaptiveAbsoluteError = settings.Adaptive.AbsoluteError;
            constants.Height = settings.Height;
            constants.SamplesPerPixel = settings.SamplesPerPixel;
            constants.MaxReflectionBounces = settings.MaxReflectionBounces;
            constants.HalfHeight = halfHeight;
            constants.HalfWidth = halfHeight * static_cast<float>(FullImageWidth(settings)) / FullImageHeight(settings);
            constants.MaterialCount = static_cast<uint32_t>(materials.size());
            constants.LightCount = static_cast<uint32_t>(lights.size());
            StoreFloat3(constants.BackgroundTop, scene.BackgroundTop);
            constants.AmbientIntensity = scene.AmbientIntensity;
            StoreFloat3(constants.BackgroundBottom, scene.BackgroundBottom);
            constants.ShadowStrength = settings.ShadowsEnabled
                ? std::clamp(settings.ShadowStrength, 0.0f, 1.0f)
                : 0.0f;
            constants.GammaCorrection = settings.GammaCorrection ? 1u : 0u;
            constants.TriangleCount = static_cast<uint32_t>(triangles.size());
            constants.SphereCount = static_cast<uint32_t>(spheres.size());
            constants.PrimitiveCount = constants.TriangleCount + constants.SphereCount;
            constants.TextureCount = static_cast<uint32_t>(
                textureAtlas.Descriptors.size());
            constants.TextureTexelCount = static_cast<uint32_t>(
                textureAtlas.Texels.size());
            std::array<uint8_t, 256> constantData{};
            std::memcpy(constantData.data(), &constants, sizeof(constants));
            ComPtr<ID3D12Resource> constantBuffer;
            if (!CreateUploadBuffer(
                constantData.data(), constantData.size(), constantBuffer, renderResult.Error))
            {
                return renderResult;
            }

            ComPtr<ID3D12Resource> shaderTable;
            if (!CreateShaderTable(*pipeline, shaderTable, renderResult.Error)) return renderResult;

            ComPtr<ID3D12Resource> mediumBuffer, pathTimeBuffer;
            if (settings.Integrator == Algorithm::PathTracing || settings.Integrator == Algorithm::DistributedRayTracing)
            {
                const PathFrameData medium{PackMedium(settings.Medium), {camera.Motion, HasSceneMotion(scene)}};
                if (!CreateUploadBuffer(&medium, sizeof(medium), mediumBuffer, renderResult.Error)) return renderResult;
                // Every dispatch lane owns one clock, reused only after its
                // previous path has completed. Exposure-off paths publish zero.
                if (!CreateBuffer(pixelCount * sizeof(float), D3D12_HEAP_TYPE_DEFAULT,
                                  D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                  D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                  pathTimeBuffer, renderResult.Error))
                    return renderResult;
            }

            commandList->CopyBufferRegion(statisticsBuffer.Get(), 0, zeroUpload.Get(), 0, StatisticsSize);
            D3D12_RESOURCE_BARRIER statisticsTransition = TransitionBarrier(
                statisticsBuffer.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            commandList->ResourceBarrier(1, &statisticsTransition);
            auto bindScene = [&] {
                commandList->SetPipelineState1(pipeline->StateObject.Get());
                commandList->SetComputeRootSignature(pointGi ? PointGiRoot.Get() : vcm ? VcmRoot.Get() : progressive ? ProgressiveRoot.Get() : photon ? PhotonRoot.Get() :
                                                     metropolis ? MetropolisRoot.Get() : RootSignature.Get());
                BindInstances(commandList.Get(), instanceBuffers);
                commandList->SetComputeRootShaderResourceView(0, topLevelAddress);
                commandList->SetComputeRootUnorderedAccessView(1, outputBuffer->GetGPUVirtualAddress());
                commandList->SetComputeRootUnorderedAccessView(2, statisticsBuffer->GetGPUVirtualAddress());
                commandList->SetComputeRootShaderResourceView(3, vertexBuffer->GetGPUVirtualAddress());
                commandList->SetComputeRootShaderResourceView(4, triangleBuffer->GetGPUVirtualAddress());
                commandList->SetComputeRootShaderResourceView(5, materialBuffer->GetGPUVirtualAddress());
                commandList->SetComputeRootShaderResourceView(6, lightBuffer->GetGPUVirtualAddress());
                commandList->SetComputeRootShaderResourceView(
                    7, textureDescriptorBuffer->GetGPUVirtualAddress());
                commandList->SetComputeRootShaderResourceView(8, textureTexelBuffer->GetGPUVirtualAddress());
                commandList->SetComputeRootConstantBufferView(9, constantBuffer->GetGPUVirtualAddress());
                if (mediumBuffer)
                {
                    constexpr UINT MediumRoot = 24;
                    commandList->SetComputeRootConstantBufferView(MediumRoot, mediumBuffer->GetGPUVirtualAddress());
                }
                if (pathTimeBuffer)
                {
                    constexpr UINT PathTimeRoot = 27;
                    commandList->SetComputeRootUnorderedAccessView(PathTimeRoot,
                                                                  pathTimeBuffer->GetGPUVirtualAddress());
                }
                if (settings.Integrator == Algorithm::PathTracing)
                {
                    constexpr UINT SpectralPaletteRoot = 25, SpectralReferenceRoot = 26;
                    commandList->SetComputeRootShaderResourceView(
                        SpectralPaletteRoot, spectral ? spectral->Coefficients->GetGPUVirtualAddress() : 0);
                    commandList->SetComputeRootShaderResourceView(
                        SpectralReferenceRoot, spectral ? spectral->References->GetGPUVirtualAddress() : 0);
                }
                commandList->SetComputeRootUnorderedAccessView(10, filmBuffer->GetGPUVirtualAddress());
                commandList->SetComputeRootShaderResourceView(11, sphereBuffer->GetGPUVirtualAddress());

                commandList->SetComputeRootShaderResourceView(14, software.Nodes->GetGPUVirtualAddress());
                commandList->SetComputeRootShaderResourceView(
                    15, software.PrimitiveIndices->GetGPUVirtualAddress());
            };
            bindScene();

            const D3D12_DISPATCH_RAYS_DESC dispatch =
                MakeDispatch(shaderTable.Get(), settings.Width, settings.Height, pipeline->CallableCount);
            const auto renderBegin = std::chrono::steady_clock::now();
            RenderStatistics vcmStatistics;
            if (pointGi)
            {
                if (!DispatchPointGi(pointGiGeometry, emission, constants, settings.PointGi, allocator.Get(),
                    commandList.Get(), shaderTable.Get(), instanceBuffers, bindScene, cancellation, renderResult.Error))
                {
                    if (cancellation()) return Detail::MakeCancelledResult();
                    return renderResult;
                }
            }
            else if (vcm)
            {
                if (!DispatchVcm(emission, constants, settings, allocator.Get(), commandList.Get(), shaderTable.Get(),
                                 instanceBuffers, bindScene, cancellation, vcmStatistics, renderResult.Error))
                {
                    if (cancellation()) return Detail::MakeCancelledResult();
                    return renderResult;
                }
            }
            else if (bidirectional)
            {
                if (!DispatchBidirectional(emission, constants, allocator.Get(), commandList.Get(), shaderTable.Get(),
                                           instanceBuffers, bindScene, cancellation, renderResult.Error))
                {
                    if (cancellation()) return Detail::MakeCancelledResult();
                    return renderResult;
                }
            }
            else if (metropolis)
            {
                if (!DispatchMetropolis(emission, constants, settings.Metropolis, allocator.Get(), commandList.Get(),
                                        shaderTable.Get(), instanceBuffers, bindScene, cancellation, renderResult.Error))
                {
                    if (cancellation()) return Detail::MakeCancelledResult();
                    return renderResult;
                }
            }
            else if (progressive)
            {
                if (!DispatchProgressivePhotons(emission, constants, settings.ProgressivePhotons, allocator.Get(),
                                               commandList.Get(), shaderTable.Get(), instanceBuffers, bindScene,
                                               cancellation, renderResult.Error))
                {
                    if (cancellation()) return Detail::MakeCancelledResult();
                    return renderResult;
                }
            }
            else if (photon)
            {
                if (!DispatchPhotons(emission, constants, settings.Photons, allocator.Get(), commandList.Get(),
                                     shaderTable.Get(), instanceBuffers, bindScene, cancellation, renderResult.Error))
                {
                    if (cancellation()) return Detail::MakeCancelledResult();
                    return renderResult;
                }
            }
            else commandList->DispatchRays(&dispatch);
            CopyInstanceError(commandList.Get(),instanceBuffers);

            D3D12_RESOURCE_BARRIER dispatchBarriers[] = {
                UavBarrier(outputBuffer.Get()),
                UavBarrier(statisticsBuffer.Get()),
                UavBarrier(filmBuffer.Get()),
                TransitionBarrier(
                    filmBuffer.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COPY_SOURCE),
                TransitionBarrier(
                    outputBuffer.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COPY_SOURCE),
                TransitionBarrier(
                    statisticsBuffer.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COPY_SOURCE)};
            commandList->ResourceBarrier(static_cast<UINT>(std::size(dispatchBarriers)), dispatchBarriers);
            commandList->CopyResource(outputReadback.Get(), outputBuffer.Get());
            commandList->CopyResource(filmReadback.Get(), filmBuffer.Get());
            commandList->CopyResource(statisticsReadback.Get(), statisticsBuffer.Get());
            result = commandList->Close();
            if (FAILED(result) || !ExecuteAndWait(commandList.Get(), renderResult.Error))
            {
                if (renderResult.Error.empty())
                {
                    renderResult.Error = HResultMessage("ID3D12GraphicsCommandList::Close", result);
                }
                return renderResult;
            }
            const auto renderEnd = std::chrono::steady_clock::now();
            if (!CheckInstanceError(instanceBuffers,renderResult.Error)) return renderResult;

            renderResult.Output = Image(settings.Width, settings.Height);
            void* mappedOutput = nullptr;
            const D3D12_RANGE outputReadRange{0, static_cast<SIZE_T>(outputSize)};
            result = outputReadback->Map(0, &outputReadRange, &mappedOutput);
            if (FAILED(result))
            {
                renderResult.Output = Image();
                renderResult.Error = HResultMessage("Mapping the DXR output readback", result);
                return renderResult;
            }
            std::memcpy(
                renderResult.Output.GetPixelsRGBA8().data(),
                mappedOutput,
                static_cast<size_t>(pixelCount) * sizeof(uint32_t));
            const D3D12_RANGE noOutputWrite{0, 0};
            outputReadback->Unmap(0, &noOutputWrite);

            void* mappedFilm = nullptr;
            const D3D12_RANGE filmReadRange{0, static_cast<SIZE_T>(filmSize)};
            result = filmReadback->Map(0, &filmReadRange, &mappedFilm);
            if (FAILED(result))
            {
                renderResult.Output = Image();
                renderResult.Error = HResultMessage("Mapping the DXR linear film readback", result);
                return renderResult;
            }
            renderResult.Film = LinearFilm(settings.Width, settings.Height);
            std::memcpy(renderResult.Film.Pixels.data(), mappedFilm, static_cast<size_t>(filmSize));
            const D3D12_RANGE noFilmWrite{0, 0};
            filmReadback->Unmap(0, &noFilmWrite);
            const uint32_t minimumSamples = settings.Adaptive.Enabled
                ? settings.Adaptive.MinimumSamples : settings.SamplesPerPixel;
            for (const auto& pixel : renderResult.Film.Pixels)
                if (pixel.SampleCount < minimumSamples || pixel.SampleCount > settings.SamplesPerPixel)
                    return Detail::MakeFailedResult("DXR film sample count is outside the requested limits.");

            void* mappedStatisticsData = nullptr;
            const D3D12_RANGE statisticsReadRange{0, StatisticsSize};
            result = statisticsReadback->Map(0, &statisticsReadRange, &mappedStatisticsData);
            if (FAILED(result))
            {
                renderResult.Output = Image();
                renderResult.Error = HResultMessage("Mapping the DXR statistics readback", result);
                return renderResult;
            }
            const auto counters = *static_cast<const GpuStatistics*>(mappedStatisticsData);
            const D3D12_RANGE noStatisticsWrite{0, 0};
            statisticsReadback->Unmap(0, &noStatisticsWrite);
            renderResult.Statistics.PilotRays = settings.Adaptive.Enabled && !settings.ProductionBudgets
                ? pixelCount * settings.Adaptive.PilotSamples : 0u;
            uint64_t expectedPrimaryRays = renderResult.Statistics.PilotRays;
            for (const auto& pixel : renderResult.Film.Pixels)
                expectedPrimaryRays += pixel.SampleCount;
            renderResult.Statistics.PrimaryRays = vcm ? vcmStatistics.PrimaryRays : counters.Primary.Value();
            renderResult.Statistics.ShadowRays = vcm ? vcmStatistics.ShadowRays : counters.Shadow.Value();
            renderResult.Statistics.ReflectionRays = vcm ? vcmStatistics.ReflectionRays : counters.Reflection.Value();
            renderResult.Statistics.CompletedPixels = vcm ? vcmStatistics.CompletedPixels : counters.Completed.Value();
            renderResult.Statistics.PrimitiveTests = 0;
            if (metropolis)
            {
                // MLT traces a bootstrap population, then one proposal per full-film
                // pixel/pass. A zero bootstrap integral skips all mutations; crops
                // still replay the full population rather than just their local film.
                const uint64_t bootstrapRays = settings.Metropolis.BootstrapSamples;
                const uint64_t mutatedRays = bootstrapRays + uint64_t(constants.FullWidth) *
                    constants.FullHeight * settings.SamplesPerPixel;
                const bool nonblack = std::any_of(renderResult.Film.Pixels.begin(), renderResult.Film.Pixels.end(),
                    [](const FilmPixel& pixel) { return pixel.MeanRadiance.X > 0 ||
                        pixel.MeanRadiance.Y > 0 || pixel.MeanRadiance.Z > 0; });
                if ((renderResult.Statistics.PrimaryRays != bootstrapRays &&
                     renderResult.Statistics.PrimaryRays != mutatedRays) ||
                    (nonblack && renderResult.Statistics.PrimaryRays != mutatedRays))
                    return Detail::MakeFailedResult("DXR MLT primary ray count differs from its bootstrap and mutation populations.");
            }
            else if (renderResult.Statistics.PrimaryRays != expectedPrimaryRays)
                return Detail::MakeFailedResult("DXR primary ray count differs from the completed film and pilot work.");
            if (renderResult.Statistics.CompletedPixels != pixelCount)
                return Detail::MakeFailedResult("DXR dispatch did not complete all requested pixels.");
            renderResult.Statistics.BuildMilliseconds =
                std::chrono::duration<double, std::milli>(buildEnd - buildBegin).count();
            renderResult.Statistics.RenderMilliseconds =
                std::chrono::duration<double, std::milli>(renderEnd - renderBegin).count();
            renderResult.Status = cancellation() ? RenderStatus::Cancelled : RenderStatus::Completed;
            if (residentEligible && renderResult.Status == RenderStatus::Completed)
            {
                ResidentOwner = owner;
                Resident = std::move(resident);
            }
            return renderResult;
        }

        DynamicModule DxgiModule;
        DynamicModule D3D12Module;
        EventHandle FenceEvent;
        CreateDeviceFunction CreateDevice = nullptr;
        ComPtr<IDXGIFactory4> Factory;
        ComPtr<IDXGIAdapter1> Adapter;
        ComPtr<ID3D12Device5> Device;
        ComPtr<ID3D12InfoQueue1> DiagnosticInfo;
        DWORD DiagnosticCookie = 0;
        std::vector<unsigned char> DiagnosticPathBytecode;
        ComPtr<ID3D12CommandQueue> CommandQueue;
        ComPtr<ID3D12Fence> Fence;
        ComPtr<ID3D12RootSignature> RootSignature;
        mutable ComPtr<ID3D12RootSignature> MetropolisRoot;
        mutable ComPtr<ID3D12RootSignature> PhotonRoot;
        mutable ComPtr<ID3D12RootSignature> ProgressiveRoot;
        mutable ComPtr<ID3D12RootSignature> VcmRoot;
        mutable ComPtr<ID3D12RootSignature> PointGiRoot;
        mutable ComPtr<ID3D12RootSignature> SpatialDenoiseRoot;
        mutable ComPtr<ID3D12RootSignature> AdaptiveDenoiseRoot;
        mutable ComPtr<ID3D12StateObject> AdaptiveDenoiseState;
        mutable ComPtr<ID3D12Resource> AdaptiveDenoiseTable;
        mutable ComPtr<ID3D12RootSignature> GeometryGuidesRoot;
        mutable ComPtr<ID3D12RootSignature> TemporalRoot;
        mutable ComPtr<ID3D12RootSignature> DicingRoot;
        mutable ComPtr<ID3D12StateObject> DicingState;
        mutable ComPtr<ID3D12Resource> DicingTable;
        mutable ComPtr<ID3D12StateObject> TemporalState;
        mutable ComPtr<ID3D12Resource> TemporalTable;
        mutable ComPtr<ID3D12StateObject> GeometryGuidesState;
        mutable ComPtr<ID3D12Resource> GeometryGuidesTable;
        mutable ComPtr<ID3D12StateObject> SpatialDenoiseState;
        mutable ComPtr<ID3D12Resource> SpatialDenoiseTable;
        mutable ComPtr<ID3D12RootSignature> SpectralRoot;
        mutable ComPtr<ID3D12StateObject> SpectralState;
        mutable ComPtr<ID3D12Resource> SpectralTable;
        mutable std::weak_ptr<const Scene> SpectralOwner;
        mutable std::shared_ptr<SpectralSnapshot> SpectralCache;
        mutable std::weak_ptr<const Scene> ResidentOwner;
        mutable std::shared_ptr<ResidentScene> Resident;
        mutable Pipeline WhittedPipeline;
        mutable Pipeline PathTracingPipeline;
        mutable Pipeline DistributedRayPipeline;
        mutable Pipeline BidirectionalPipeline;
        mutable Pipeline MetropolisPipeline;
        mutable Pipeline PhotonPipeline;
        mutable Pipeline ProgressivePipeline;
        mutable Pipeline VcmPipeline;
        mutable Pipeline PointGiPipeline;
        mutable Pipeline RayQueryPipeline;
        D3D12_RAYTRACING_TIER RayTracingTier = D3D12_RAYTRACING_TIER_NOT_SUPPORTED;
        mutable uint64_t FenceValue = 0;
        std::string AdapterName;
#endif
        bool Available = false;
        std::string AvailabilityError;
        mutable uint64_t DispatchRaysSequence = 0;
        mutable Detail::PersistentRenderQueue RHIQueue;
    };

    DxrRayTracingRHI::DxrRayTracingRHI()
        : m_Impl(std::make_unique<Impl>())
    {
    }

    DxrRayTracingRHI::~DxrRayTracingRHI() = default;

    BackendInfo DxrRayTracingRHI::GetBackendInfo() const
    {
        BackendInfo info;
        info.Type = BackendType::DXR;
        info.Name = "DXR";
        info.IsNative = m_Impl->Available;
        info.IsHardwareAccelerated = m_Impl->Available;
        info.IsAvailable = m_Impl->Available;
        info.Description = m_Impl->Available
#if defined(_WIN32)
            ? std::string("Native D3D12 DXR backend on ") + m_Impl->AdapterName
#else
            ? "Native D3D12 DXR backend"
#endif
            : std::string("Unavailable: ") + m_Impl->AvailabilityError;
        return info;
    }

    RayQueryResult DxrRayTracingRHI::QueryRays(
        const Scene& scene,
        const std::vector<Ray>& rays,
        AccelerationOptions accelerationType,
        const CancellationToken& cancellation) const
    {
        // Rendering and diagnostics share the D3D queue, fence, and event.
        // Wait for this queued task before allowing the borrowed inputs to die.
        RayQueryResult query;
        const RenderResult completion = m_Impl->RHIQueue.Submit(
            [this, &scene, &rays, &query, accelerationType, cancellation](const CancellationToken& queued)
            {
                const auto cancelled = Detail::CombineCancellation(queued, cancellation);
                if (cancelled()) return Detail::MakeCancelledResult();
                if (!m_Impl->Available)
                    return Detail::MakeFailedResult("DXR is unavailable: " + m_Impl->AvailabilityError);
#if defined(_WIN32)
                query = m_Impl->QueryNative(scene, rays, accelerationType, cancelled);
                RenderResult result;
                result.Status = query.Status;
                result.Error = query.Error;
                return result;
#else
                return Detail::MakeFailedResult("DXR is available only on Windows.");
#endif
            }).GetResult();
        query.Status = completion.Status;
        query.Error = completion.Error;
        if (query.Status != RenderStatus::Completed) query.Hits.clear();
        return query;
    }

    MicropolygonResult DxrRayTracingRHI::PrepareMicropolygons(const Scene& scene, const Camera& camera,
                                                              const RenderSettings& settings,
                                                              const CancellationToken& cancellation) const
    {
        MicropolygonResult result;
        const auto task =
            m_Impl->RHIQueue
                .Submit([&](const CancellationToken& queued) {
                    const auto stopped = Detail::CombineCancellation(queued, cancellation);
                    if (stopped())
                        return Detail::MakeCancelledResult();
                    if (!m_Impl->Available)
                        return Detail::MakeFailedResult("DXR is unavailable: " + m_Impl->AvailabilityError);
#if defined(_WIN32)
                    result = Detail::PrepareMicropolygons(scene, camera, settings, stopped,
                                                          [&](const auto& controls, const auto& patches,
                                                              const auto& ranges, auto& words, auto& error) {
                                                              return m_Impl->MicropolygonPassNative(
                                                                  controls, patches, ranges, words, stopped,
                                                                  error);
                                                          });
                    RenderResult completion;
                    completion.Status = result.Status;
                    completion.Error = result.Error;
                    return completion;
#else
                    return Detail::MakeFailedResult("Micropolygons require native DXR");
#endif
                })
                .GetResult();
        result.Status = task.Status;
        result.Error = task.Error;
        return result;
    }

    TemporalResult DxrRayTracingRHI::ReconstructTemporal(const TemporalFrame& frame,
                                                         const TemporalHistory* history,
                                                         const TemporalOptions& options,
                                                         const CancellationToken& cancellation) const
    {
        TemporalResult result;
        const auto task =
            m_Impl->RHIQueue
                .Submit([&](const CancellationToken& queued) {
                    const auto stopped = Detail::CombineCancellation(queued, cancellation);
                    if (stopped())
                        return Detail::MakeCancelledResult();
                    if (!m_Impl->Available)
                        return Detail::MakeFailedResult("DXR is unavailable: " + m_Impl->AvailabilityError);
#if defined(_WIN32)
                    result = Detail::ReconstructTemporal(
                        BackendType::DXR, frame, history, options, stopped,
                        [&](const auto& controls, const auto& current, const auto& guides,
                            const auto& previous, const auto& oldGuides, auto& output, auto& display,
                            auto& error) {
                            return m_Impl->ReconstructTemporalNative(controls, current, guides, previous,
                                                                     oldGuides, stopped, output, display,
                                                                     error);
                        });
                    RenderResult completion;
                    completion.Status = result.Status;
                    completion.Error = result.Error;
                    return completion;
#else
                    return Detail::MakeFailedResult("Temporal reconstruction requires native DXR");
#endif
                })
                .GetResult();
        result.Status = task.Status;
        result.Error = task.Error;
        if (result.Status != RenderStatus::Completed)
        {
            result.History = {};
            result.Display = {};
            result.AcceptedHistoryPixels = 0;
        }
        return result;
    }

    GeometryGuideResult DxrRayTracingRHI::CaptureGeometryGuides(const Scene& scene, const Camera& camera,
                                                                const RenderSettings& settings,
                                                                const CancellationToken& cancellation) const
    {
        GeometryGuideResult result;
        if (settings.Micropolygons.Enabled)
        {
            auto diced = PrepareMicropolygons(scene, camera, settings, cancellation);
            if (diced.Status != RenderStatus::Completed)
                return {diced.Status, {}, diced.Error};
            auto prepared = settings;
            prepared.Micropolygons.Enabled = false;
            prepared.DicingPlan.reset();
            return CaptureGeometryGuides(*diced.Geometry, camera, prepared, cancellation);
        }
        const auto task =
            m_Impl->RHIQueue
                .Submit([&](const CancellationToken& queued) {
                    const auto stopped = Detail::CombineCancellation(queued, cancellation);
                    if (stopped())
                        return Detail::MakeCancelledResult();
                    if (!m_Impl->Available)
                        return Detail::MakeFailedResult("DXR is unavailable: " + m_Impl->AvailabilityError);
#if defined(_WIN32)
                    result = m_Impl->CaptureGuidesNative(scene, camera, settings, stopped);
                    RenderResult completion;
                    completion.Status = result.Status;
                    completion.Error = result.Error;
                    return completion;
#else
                    return Detail::MakeFailedResult("Geometry guides require native DXR on Windows");
#endif
                })
                .GetResult();
        result.Status = task.Status;
        result.Error = task.Error;
        if (result.Status != RenderStatus::Completed)
            result.Image = {};
        return result;
    }

    RenderResult DxrRayTracingRHI::Render(
        const Scene& scene,
        const Camera& camera,
        const RenderSettings& settings,
        const CancellationToken& cancellation) const
    {
        return m_Impl->Submit(&scene, nullptr, camera, settings, cancellation).GetResult();
    }

    RenderJob DxrRayTracingRHI::RenderAsync(
        Scene scene,
        Camera camera,
        RenderSettings settings) const
    {
        return RenderAsync(std::make_shared<Scene>(std::move(scene)), camera, settings);
    }

    RenderJob DxrRayTracingRHI::RenderAsync(
        std::shared_ptr<const Scene> scene,
        Camera camera,
        RenderSettings settings) const
    {
        const Scene* pointer = scene.get();
        return m_Impl->Submit(pointer, std::move(scene), camera, settings);
    }

    void DxrRayTracingRHI::WaitIdle() const
    {
        m_Impl->RHIQueue.WaitIdle();
    }
}
}

namespace Arvo::Renderer
{
std::shared_ptr<RayTracing::IRayTracingRHI> CreateDxrRHI()
{
    return std::make_shared<RayTracing::DxrRayTracingRHI>();
}
}
