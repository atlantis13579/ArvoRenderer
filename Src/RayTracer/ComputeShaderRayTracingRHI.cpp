#include "Renderer/SubsurfaceValidation.h"
#include "ComputeShaderRayTracingRHI.h"
#include "Renderer/Reconstruction.h"
#include "Renderer/AdaptiveReconstruction.h"
#include "Renderer/PathFrameData.h"
#include "Renderer/SpectralSceneData.h"
#include "Renderer/GeometryGuides.h"
#include "Renderer/TemporalReconstruction.h"
#include "Renderer/MicropolygonPreparation.h"
#include <type_traits>
#include "Backends/Compute/Limits.h"
#include "Backends/Compute/BidirectionalLayout.h"
#include "Backends/Compute/MetropolisLayout.h"
#include "Backends/Compute/PhotonLayout.h"
#include "Backends/Compute/ProgressivePhotonLayout.h"
#include "Backends/Compute/VertexConnectionMergingDispatch.h"
#include "Backends/Compute/PointGiGeometry.h"
#include "Renderer/EmitterData.h"

#include "Renderer/RayTracingBackendCommon.h"
#include "Backends/Compute/BihBuilder.h"
#include "Backends/Compute/InstancingBuilder.h"
#include "Backends/Compute/SpatialBvhBuilder.h"
#include "Backends/Compute/BvhBuilder.h"
#include "Backends/Compute/KdTreeBuilder.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32)
#  include "ComputeShaderBytecode.h"
#  if !defined(NOMINMAX)
#    define NOMINMAX
#  endif
#  include <Windows.h>
#  include <d3d12.h>
#  include <d3d12sdklayers.h>
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

        constexpr std::uint32_t SphereBit = 0x80000000u;

        constexpr uint32_t SpectralPaletteDescriptor = 10, SpectralReferenceDescriptor = 11;
        constexpr uint32_t SceneDescriptorCount = 12, FrameDescriptorBegin = SceneDescriptorCount;
        constexpr uint32_t InstanceErrorDescriptor = FrameDescriptorBegin + 5;
        constexpr uint32_t DescriptorCount = InstanceErrorDescriptor + 1;
        constexpr std::uint32_t ThreadsPerGroup = 256;
        constexpr std::uint32_t PixelChunkSize = 16384;
        constexpr std::uint64_t MaximumRawNodeBufferBytes =
            static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()) + 1;
        constexpr std::uint32_t DirectionalLightType = 0;
        constexpr std::uint32_t PointLightType = 1;
        constexpr std::uint32_t RectAreaLightType = 2;
        constexpr std::size_t ReadbackChunkSize = 32u * 1024u * 1024u;
        constexpr float Pi = 3.14159265358979323846f;
        constexpr DWORD GpuWaitTimeoutMilliseconds = 5000;
        constexpr DWORD WarpWaitTimeoutMilliseconds = 60000;

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

        struct alignas(16) PackedBvhNode
        {
            float MinX = 0.0f;
            float MinY = 0.0f;
            float MinZ = 0.0f;
            std::uint32_t FirstOrEscape = 0;
            float MaxX = 0.0f;
            float MaxY = 0.0f;
            float MaxZ = 0.0f;
            std::uint32_t PrimitiveCount = 0;
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
            std::uint32_t Type = DirectionalLightType;
            Vec3 Direction;
            float Intensity = 0.0f;
            Vec3 Color;
            std::uint32_t CastShadows = 0;
            float Width = 0.0f;
            float Height = 0.0f;
            std::uint32_t Padding0 = 0;
            std::uint32_t Padding1 = 0;
        };

        // Packed shader fields; upload records have the required CBV alignment.
        struct FrameConstants
        {
            Vec3 CameraPosition;
            std::uint32_t Width = 0;
            Vec3 CameraForward;
            std::uint32_t Height = 0;
            Vec3 CameraRight;
            std::uint32_t TotalSamples = 0;
            Vec3 CameraUp;
            std::uint32_t MaxReflectionBounces = 0;
            float HalfWidth = 0.0f;
            float HalfHeight = 0.0f;
            std::uint32_t MaterialCount = 0;
            std::uint32_t LightCount = 0;
            Vec3 BackgroundTop;
            float AmbientIntensity = 0.0f;
            Vec3 BackgroundBottom;
            std::uint32_t NodeCount = 0;
            std::uint32_t PrimitiveRefCount = 0;
            std::uint32_t TriangleCount = 0;
            std::uint32_t SphereCount = 0;
            float ShadowStrength = 0.0f;
            std::uint32_t PixelBegin = 0;
            std::uint32_t PixelCount = 0;
            std::uint32_t SampleBegin = 0;
            std::uint32_t SampleCount = 0;
            std::uint32_t GammaCorrection = 0;
            std::uint32_t GroupCount = 0;
            std::uint32_t TextureCount = 0;
            std::uint32_t TextureTexelCount = 0;
            std::uint32_t FullWidth = 0;
            std::uint32_t FullHeight = 0;
            std::uint32_t TileX = 0;
            std::uint32_t TileY = 0;
            std::uint32_t Integrator = 0;
            std::uint32_t TransportFlags = 0;
            std::uint32_t RouletteStartDepth = 0;
            std::uint32_t SamplingSequence = 0;
            std::uint32_t AdaptiveEnabled = 0;
            std::uint32_t AdaptivePilotSamples = 0;
            std::uint32_t AdaptiveMinimumSamples = 0;
            std::uint32_t AdaptivePhase = 0;
            float AdaptiveRelativeError = 0.0f;
            float AdaptiveAbsoluteError = 0.0f;
            std::uint32_t Acceleration = 0;
            std::uint32_t InstanceNodeCount = 0;
            float LensRadius = 0;
            float FocusDistance = 0;
            std::uint32_t SamplingSeed = 0;
        };

        struct Counter64Parts
        {
            std::uint32_t Low = 0;
            std::uint32_t High = 0;

            std::uint64_t Value() const
            {
                return static_cast<std::uint64_t>(Low)
                    | (static_cast<std::uint64_t>(High) << 32);
            }
        };

        struct GroupStatistics
        {
            Counter64Parts PrimaryRays;
            Counter64Parts ShadowRays;
            Counter64Parts ReflectionRays;
            Counter64Parts PrimitiveTests;
        };

        static_assert(sizeof(Vec4) == 16, "Compute Vec4 ABI mismatch.");
        static_assert(sizeof(PackedBvhNode) == 32, "Compute BVH ABI mismatch.");
        static_assert(sizeof(PackedPrimitiveRef) == 4, "Compute primitive ABI mismatch.");
        static_assert(sizeof(PackedTriangle) == 192, "Compute triangle ABI mismatch.");
        static_assert(offsetof(PackedTriangle, TexCoord01) == 144, "Compute triangle field ABI mismatch.");
        static_assert(offsetof(PackedTriangle, MaterialIndex) == 176, "Compute triangle field ABI mismatch.");
        static_assert(sizeof(PackedSphere) == 32, "Compute sphere ABI mismatch.");
        static_assert(sizeof(PackedMaterial) == 128, "Compute material ABI mismatch.");
        static_assert(
            offsetof(PackedMaterial, AmbientShininess) == 16
                && offsetof(PackedMaterial, Specular) == 32
                && offsetof(PackedMaterial, Emission) == 48
                && offsetof(PackedMaterial, DiffuseTextureIndex) == 64
                && offsetof(PackedMaterial, Model) == 72
                && offsetof(PackedMaterial, Optical) == 80
                && offsetof(PackedMaterial, ConductorIndex) == 96
                && offsetof(PackedMaterial, ConductorExtinction) == 112,
            "Compute material field ABI mismatch.");
        static_assert(sizeof(PackedLight) == 64, "Compute light ABI mismatch.");
        static_assert(
            offsetof(PackedLight, Type) == 12
                && offsetof(PackedLight, Direction) == 16
                && offsetof(PackedLight, Intensity) == 28
                && offsetof(PackedLight, Color) == 32
                && offsetof(PackedLight, CastShadows) == 44
                && offsetof(PackedLight, Width) == 48,
            "Compute light field ABI mismatch.");
        static_assert(
            sizeof(FrameConstants) == 236
                && offsetof(FrameConstants, ShadowStrength) == 124,
            "Compute constants ABI mismatch.");
        static_assert(offsetof(FrameConstants, Acceleration) == 216, "Acceleration constant ABI mismatch.");
        static_assert(offsetof(FrameConstants, InstanceNodeCount) == 220, "Instance constant ABI mismatch.");
        static_assert(offsetof(FrameConstants, LensRadius) == 224 && offsetof(FrameConstants, FocusDistance) == 228,
            "Compute lens constant ABI mismatch.");
        static_assert(offsetof(FrameConstants, SamplingSeed) == 232, "Compute sampling seed ABI mismatch.");
        constexpr uint32_t RootTableCost = 2, RootBufferCost = 4, RootFrameCost = 4;
        static_assert(RootFrameCost + RootTableCost + RootBufferCost <= D3D12_MAX_ROOT_COST,
                      "Compute root signature exceeds the hardware budget.");
        constexpr uint64_t MediumConstantOffset = D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT;
        constexpr uint64_t FrameConstantStride = MediumConstantOffset * 2;
        constexpr uint64_t FrameConstantPageBytes = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
        constexpr uint64_t FrameConstantsPerPage = FrameConstantPageBytes / FrameConstantStride;
        static_assert(sizeof(FrameConstants) <= MediumConstantOffset, "Frame constants overlap medium data.");
        static_assert(sizeof(PathFrameData) <= FrameConstantStride - MediumConstantOffset,
                      "Grow the medium CBV slot.");
        constexpr uint32_t MediumRootParameter = 5;
        static_assert(sizeof(Counter64Parts) == 8, "Compute counter ABI mismatch.");
        static_assert(offsetof(FrameConstants, FullWidth) == 160, "Tile constant ABI mismatch.");
        static_assert(offsetof(FrameConstants, Integrator) == 176, "Integrator constant ABI mismatch.");
        static_assert(offsetof(FrameConstants, AdaptiveEnabled) == 192, "Adaptive constant ABI mismatch.");
        static_assert(sizeof(Ray) == 32 && offsetof(Ray, Direction) == 12
            && offsetof(Ray, MinDistance) == 24, "Compute query-ray layout must match native RayData.");
        static_assert(sizeof(RayQueryHit) == 64 && offsetof(RayQueryHit, PrimitiveIndex) == 28
            && offsetof(RayQueryHit, Flags) == 44, "Compute query-hit layout must match native QueryHit.");
        static_assert(sizeof(GroupStatistics) == 32, "Compute statistics ABI mismatch.");

        Vec3 ToVec3(const Float3& value)
        {
            return {value.X, value.Y, value.Z};
        }

        Vec4 ToVec4(const Float3& value, float w = 0.0f)
        {
            return {value.X, value.Y, value.Z, w};
        }

        Vec3 Add(const Vec3& left, const Vec3& right)
        {
            return {left.X + right.X, left.Y + right.Y, left.Z + right.Z};
        }

        Vec3 Subtract(const Vec3& left, const Vec3& right)
        {
            return {left.X - right.X, left.Y - right.Y, left.Z - right.Z};
        }

        Vec3 Multiply(const Vec3& value, float scalar)
        {
            return {value.X * scalar, value.Y * scalar, value.Z * scalar};
        }

        float Dot(const Vec3& left, const Vec3& right)
        {
            return left.X * right.X + left.Y * right.Y + left.Z * right.Z;
        }

        Vec3 Cross(const Vec3& left, const Vec3& right)
        {
            return {
                left.Y * right.Z - left.Z * right.Y,
                left.Z * right.X - left.X * right.Z,
                left.X * right.Y - left.Y * right.X};
        }

        Vec3 NormalizeVector(const Vec3& value)
        {
            const float lengthSquared = Dot(value, value);
            if (lengthSquared <= 1.0e-20f) return {};
            return Multiply(value, 1.0f / std::sqrt(lengthSquared));
        }

        bool IsFinite(const Float3& value)
        {
            return std::isfinite(value.X) && std::isfinite(value.Y) && std::isfinite(value.Z);
        }

        bool IsFinite(const Vec3& value)
        {
            return std::isfinite(value.X) && std::isfinite(value.Y) && std::isfinite(value.Z);
        }

        struct Bounds
        {
            Vec3 Minimum = {
                std::numeric_limits<float>::max(),
                std::numeric_limits<float>::max(),
                std::numeric_limits<float>::max()};
            Vec3 Maximum = {
                -std::numeric_limits<float>::max(),
                -std::numeric_limits<float>::max(),
                -std::numeric_limits<float>::max()};

            void Expand(const Vec3& point)
            {
                Minimum.X = std::min(Minimum.X, point.X);
                Minimum.Y = std::min(Minimum.Y, point.Y);
                Minimum.Z = std::min(Minimum.Z, point.Z);
                Maximum.X = std::max(Maximum.X, point.X);
                Maximum.Y = std::max(Maximum.Y, point.Y);
                Maximum.Z = std::max(Maximum.Z, point.Z);
            }

            void Expand(const Bounds& bounds)
            {
                Expand(bounds.Minimum);
                Expand(bounds.Maximum);
            }

            Vec3 Center() const
            {
                return Multiply(Add(Minimum, Maximum), 0.5f);
            }

        };

        struct BuildPrimitive
        {
            Bounds BoundingBox;
            Vec3 Centroid;
            PackedPrimitiveRef Reference;
        };

        struct PackedScene
        {
            std::vector<uint8_t> ObjectNodes;
            std::vector<ComputeInstances::Instance> Instances;
            std::vector<ComputeInstances::Node> InstanceNodes;
            std::vector<PackedBvhNode> Nodes;
            std::vector<uint8_t> KdNodes;
            uint32_t KdNodeCount = 0;
            std::vector<uint8_t> BihNodes;
            uint32_t BihNodeCount = 0;
            std::vector<PackedPrimitiveRef> PrimitiveRefs;
            std::vector<PackedTriangle> Triangles;
            std::vector<PackedSphere> Spheres;
            std::vector<PackedMaterial> Materials;
            std::vector<PackedLight> Lights;
            Detail::PackedRayTextureAtlas TextureAtlas;
        };

        bool ValidateInput(
            const Scene& scene,
            const Camera& camera,
            const RenderSettings& settings,
            std::string& error, bool validateMotionSchedule = true)
        {
            if (!ComputeInstances::Validate(scene, error)) return false;
            if (!ValidateCameraLens(camera.Lens, error) ||
                (validateMotionSchedule && !ValidateSubsurfaceScene(scene, settings, error)) ||
                !ValidateCameraMotionSupport(settings.Integrator, camera.Motion, error) ||
                (validateMotionSchedule && !ValidateSceneMotionSupport(settings.Integrator, scene, error)) ||
                !ValidateCameraSamplePlan(camera, settings, error))
                return false;
            if (!ValidateTileSettings(settings, error)) return false;
            if (!ValidateAcceleration(settings.Acceleration, error)) return false;
            if (!ValidateIntegrator(BackendType::ComputeShader, settings.Integrator, error)) return false;
            if (settings.Width == 0 || settings.Height == 0
                || settings.Width > 16384 || settings.Height > 16384)
            {
                error = "Compute image dimensions must be in the range [1, 16384].";
                return false;
            }
            if (settings.SamplesPerPixel == 0 || settings.SamplesPerPixel > 4096
                || settings.MaxReflectionBounces > ComputeLimits::MaximumReflectionBounces
                || !std::isfinite(settings.ShadowStrength)
                || settings.ShadowStrength < 0.0f
                || settings.ShadowStrength > 1.0f)
            {
                error = "Compute sample, reflection, or shadow settings exceed their safety limits.";
                return false;
            }
            if (!IsFinite(camera.Position) || !IsFinite(camera.Target) || !IsFinite(camera.Up)
                || !std::isfinite(camera.VerticalFieldOfViewDegrees)
                || camera.VerticalFieldOfViewDegrees <= 1.0f
                || camera.VerticalFieldOfViewDegrees >= 179.0f)
            {
                error = "Compute camera values are invalid.";
                return false;
            }
            const Vec3 cameraDirection = Subtract(ToVec3(camera.Target), ToVec3(camera.Position));
            const Vec3 cameraUp = ToVec3(camera.Up);
            const Vec3 frameCross = Cross(
                NormalizeVector(cameraDirection), NormalizeVector(cameraUp));
            if (!std::isfinite(Dot(cameraDirection, cameraDirection))
                || !std::isfinite(Dot(cameraUp, cameraUp))
                || Dot(cameraDirection, cameraDirection) < 1.0e-10f
                || Dot(cameraUp, cameraUp) < 1.0e-10f
                || Dot(frameCross, frameCross) < 1.0e-10f)
            {
                error = "Compute camera vectors do not form a valid frame.";
                return false;
            }
            if ((!scene.Triangles.empty() || !scene.Spheres.empty()) && scene.Materials.empty())
            {
                error = "Compute geometry requires at least one material.";
                return false;
            }
            if (scene.Triangles.size() >= SphereBit || scene.Spheres.size() >= SphereBit
                || scene.Triangles.size() + scene.Spheres.size() >= SphereBit
                || scene.Materials.size() > std::numeric_limits<std::uint32_t>::max()
                || scene.DirectionalLights.size() > std::numeric_limits<std::uint32_t>::max()
                || scene.PointLights.size() > std::numeric_limits<std::uint32_t>::max()
                || scene.RectAreaLights.size() > std::numeric_limits<std::uint32_t>::max()
                || static_cast<std::uint64_t>(scene.DirectionalLights.size())
                        + scene.PointLights.size()
                        + scene.RectAreaLights.size()
                    > std::numeric_limits<std::uint32_t>::max())
            {
                error = "Compute scene counts exceed the 31-bit primitive layout.";
                return false;
            }
            if (!IsFinite(scene.BackgroundTop) || !IsFinite(scene.BackgroundBottom)
                || !std::isfinite(scene.AmbientIntensity) || scene.AmbientIntensity < 0.0f)
            {
                error = "Compute background and ambient values are invalid.";
                return false;
            }
            for (const Material& material : scene.Materials)
            {
                if (!IsFinite(material.Albedo)
                    || !IsFinite(material.AmbientColor)
                    || !IsFinite(material.SpecularColor)
                    || !IsFinite(material.Emission)
                    || !std::isfinite(material.Shininess)
                    || !std::isfinite(material.Reflectivity))
                {
                    error = "Compute material values must be finite.";
                    return false;
                }
                if (material.AmbientColor.X < 0.0f
                    || material.AmbientColor.Y < 0.0f
                    || material.AmbientColor.Z < 0.0f
                    || material.SpecularColor.X < 0.0f
                    || material.SpecularColor.Y < 0.0f
                    || material.SpecularColor.Z < 0.0f
                    || material.Shininess < 0.0f)
                {
                    error = "Compute material ambient and specular values must be non-negative.";
                    return false;
                }
            }
            std::vector<const SceneGeometry*> geometries{&scene};
            for (const auto& geometry : scene.Geometries) geometries.push_back(geometry.get());
            for (const auto* geometry : geometries)
            {
                for (const Triangle& triangle : geometry->Triangles)
                {
                    if (triangle.MaterialIndex >= scene.Materials.size()
                        || !IsFinite(triangle.Position0) || !IsFinite(triangle.Position1)
                        || !IsFinite(triangle.Position2) || !IsFinite(triangle.Normal0)
                        || !IsFinite(triangle.Normal1) || !IsFinite(triangle.Normal2)
                        || !IsFinite(triangle.Color0) || !IsFinite(triangle.Color1)
                        || !IsFinite(triangle.Color2)
                        || !std::isfinite(triangle.TexCoord0.X)
                        || !std::isfinite(triangle.TexCoord0.Y)
                        || !std::isfinite(triangle.TexCoord1.X)
                        || !std::isfinite(triangle.TexCoord1.Y)
                        || !std::isfinite(triangle.TexCoord2.X)
                        || !std::isfinite(triangle.TexCoord2.Y))
                    {
                        error = "Compute triangle data is invalid.";
                        return false;
                    }
                    const Vec3 edge1 = Subtract(ToVec3(triangle.Position1), ToVec3(triangle.Position0));
                    const Vec3 edge2 = Subtract(ToVec3(triangle.Position2), ToVec3(triangle.Position0));
                    const Vec3 area = Cross(edge1, edge2);
                    if (!std::isfinite(Dot(edge1, edge1)) || !std::isfinite(Dot(edge2, edge2))
                        || !std::isfinite(Dot(area, area)) || Dot(area, area) < 1.0e-16f)
                    {
                        error = "Compute does not accept degenerate or numerically unsafe triangles.";
                        return false;
                    }
                }
                for (const Sphere& sphere : geometry->Spheres)
                {
                    const float radiusSquared = sphere.Radius * sphere.Radius;
                    const Float3 minimum = {
                        sphere.Center.X - sphere.Radius,
                        sphere.Center.Y - sphere.Radius,
                        sphere.Center.Z - sphere.Radius};
                    const Float3 maximum = {
                        sphere.Center.X + sphere.Radius,
                        sphere.Center.Y + sphere.Radius,
                        sphere.Center.Z + sphere.Radius};
                    if (sphere.MaterialIndex >= scene.Materials.size() || !IsFinite(sphere.Center)
                        || !std::isfinite(sphere.Radius) || sphere.Radius <= 0.0f
                        || !std::isfinite(radiusSquared)
                        || !IsFinite(minimum) || !IsFinite(maximum))
                    {
                        error = "Compute sphere data is invalid.";
                        return false;
                    }
                }
            }
            for (const DirectionalLight& light : scene.DirectionalLights)
            {
                const Vec3 direction = ToVec3(light.DirectionToLight);
                if (!IsFinite(light.DirectionToLight) || !IsFinite(light.Color)
                    || !std::isfinite(light.Intensity) || light.Intensity < 0.0f
                    || !std::isfinite(Dot(direction, direction))
                    || Dot(direction, direction) < 1.0e-12f)
                {
                    error = "Compute directional light data is invalid.";
                    return false;
                }
            }
            for (const PointLight& light : scene.PointLights)
            {
                if (!IsFinite(light.Position) || !IsFinite(light.Color)
                    || !std::isfinite(light.Intensity) || light.Intensity < 0.0f)
                {
                    error = "Compute point light data is invalid.";
                    return false;
                }
            }
            for (const RectAreaLight& light : scene.RectAreaLights)
            {
                const Vec3 direction = ToVec3(light.Direction);
                const float directionLengthSquared = Dot(direction, direction);
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
                    error = "Compute rect area light data is invalid.";
                    return false;
                }
                const Vec3 emissionDirection = NormalizeVector(direction);
                const Vec3 helper = std::fabs(emissionDirection.Y) < 0.999f
                    ? Vec3{0.0f, 1.0f, 0.0f}
                    : Vec3{1.0f, 0.0f, 0.0f};
                const Vec3 tangent =
                    NormalizeVector(Cross(helper, emissionDirection));
                const Vec3 bitangent =
                    NormalizeVector(Cross(emissionDirection, tangent));
                const Vec3 position = ToVec3(light.Position);
                const Vec3 halfTangent =
                    Multiply(tangent, light.Size.X * 0.5f);
                const Vec3 halfBitangent =
                    Multiply(bitangent, light.Size.Y * 0.5f);
                if (!IsFinite(Add(Add(position, halfTangent), halfBitangent))
                    || !IsFinite(Subtract(Add(position, halfTangent), halfBitangent))
                    || !IsFinite(Add(Subtract(position, halfTangent), halfBitangent))
                    || !IsFinite(Subtract(
                        Subtract(position, halfTangent),
                        halfBitangent)))
                {
                    error = "Compute rect area light bounds are invalid.";
                    return false;
                }
            }
            return true;
        }

        class ScenePacker
        {
        public:
            explicit ScenePacker(const Detail::CancellationProbe& cancellation)
                : Cancellation(cancellation)
            {
            }

            bool Pack(const Scene& scene, AccelerationOptions acceleration, PackedScene& output, std::string& error)
            {
                output = {};
                if (!Detail::BuildRayTextureAtlas(
                        scene,
                        output.TextureAtlas,
                        error))
                {
                    return false;
                }
                output.Materials.reserve(scene.Materials.size());
                output.Lights.reserve(
                    scene.DirectionalLights.size()
                    + scene.PointLights.size()
                    + scene.RectAreaLights.size());

                for (std::uint32_t index = 0; index < scene.Materials.size(); ++index)
                {
                    if ((index & 4095u) == 0 && IsCancelled()) return Cancel(error);
                    const Material& source = scene.Materials[index];
                    PackedMaterial material;
                    material.AlbedoReflectivity = ToVec4(source.Albedo, source.Reflectivity);
                    material.AmbientShininess = ToVec4(source.AmbientColor, source.Shininess);
                    material.Specular = ToVec4(source.SpecularColor);
                    material.Emission = ToVec4(source.Emission);
                    material.DiffuseTextureIndex = source.DiffuseTextureIndex;
                    material.SampledEmitter = source.SampledEmitter ? 1u : 0u;
                    material.Model = static_cast<std::uint32_t>(source.Model);
                    material.Optical = {source.MicrofacetSlope, source.ExteriorIndex, source.InteriorIndex, source.SubsurfaceMeanFreePath};
                    material.ConductorIndex = ToVec4(source.ConductorIndex, source.SubsurfaceAnisotropy);
                    material.ConductorExtinction = ToVec4(source.ConductorExtinction, source.SubsurfaceAlbedoScale);
                    output.Materials.push_back(material);
                }
                for (std::uint32_t index = 0; index < scene.DirectionalLights.size(); ++index)
                {
                    if ((index & 4095u) == 0 && IsCancelled()) return Cancel(error);
                    const DirectionalLight& source = scene.DirectionalLights[index];
                    const Vec3 direction = NormalizeVector(ToVec3(source.DirectionToLight));
                    PackedLight light;
                    light.Type = DirectionalLightType;
                    light.Direction = direction;
                    light.Intensity = source.Intensity;
                    light.Color = ToVec3(source.Color);
                    light.CastShadows = source.CastShadows ? 1u : 0u;
                    output.Lights.push_back(light);
                }
                for (std::uint32_t index = 0; index < scene.PointLights.size(); ++index)
                {
                    if ((index & 4095u) == 0 && IsCancelled()) return Cancel(error);
                    const PointLight& source = scene.PointLights[index];
                    PackedLight light;
                    light.Position = ToVec3(source.Position);
                    light.Type = PointLightType;
                    light.Intensity = source.Intensity;
                    light.Color = ToVec3(source.Color);
                    light.CastShadows = source.CastShadows ? 1u : 0u;
                    output.Lights.push_back(light);
                }
                for (std::uint32_t index = 0; index < scene.RectAreaLights.size(); ++index)
                {
                    if ((index & 4095u) == 0 && IsCancelled()) return Cancel(error);
                    const RectAreaLight& source = scene.RectAreaLights[index];
                    PackedLight light;
                    light.Position = ToVec3(source.Position);
                    light.Type = RectAreaLightType;
                    light.Direction =
                        NormalizeVector(ToVec3(source.Direction));
                    light.Intensity = source.Intensity;
                    light.Color = ToVec3(source.Color);
                    light.CastShadows = source.CastShadows ? 1u : 0u;
                    light.Width = source.Size.X;
                    light.Height = source.Size.Y;
                    output.Lights.push_back(light);
                }

                if (scene.Instances.empty()) return PackGeometry(scene, acceleration, output, error);
                ComputeInstances::Hierarchy hierarchy;
                if (!ComputeInstances::Build(scene, Cancellation, hierarchy, error)) return false;
                std::vector<ComputeInstances::GeometryRange> ranges(hierarchy.Objects.size());
                for (size_t object = 0; object < hierarchy.Objects.size(); ++object)
                {
                    if (IsCancelled()) return Cancel(error);
                    PackedScene local;
                    if (!PackGeometry(*hierarchy.Objects[object], acceleration, local, error)) return false;
                    const void* nodeData = local.Nodes.data();
                    size_t nodeBytes = local.Nodes.size() * sizeof(PackedBvhNode);
                    uint32_t nodeCount = uint32_t(local.Nodes.size());
                    if (acceleration.Type == AccelerationType::KdTree)
                    {
                        nodeData = local.KdNodes.data();
                        nodeBytes = local.KdNodes.size();
                        nodeCount = local.KdNodeCount;
                    }
                    else if (acceleration.Type == AccelerationType::Bih)
                    {
                        nodeData = local.BihNodes.data();
                        nodeBytes = local.BihNodes.size();
                        nodeCount = local.BihNodeCount;
                    }
                    const auto fits = [](size_t count, size_t added) {
                        return uint64_t(count) + added <= std::numeric_limits<uint32_t>::max();
                    };
                    if (!fits(output.ObjectNodes.size(), nodeBytes)
                        || !fits(output.PrimitiveRefs.size(), local.PrimitiveRefs.size())
                        || !fits(output.Triangles.size(), local.Triangles.size())
                        || !fits(output.Spheres.size(), local.Spheres.size()))
                    {
                        error = "Compute object buffers exceed 32-bit addressing.";
                        return false;
                    }
                    auto& range = ranges[object];
                    range.NodeByteOffset = uint32_t(output.ObjectNodes.size());
                    range.NodeCount = nodeCount;
                    range.ReferenceOffset = uint32_t(output.PrimitiveRefs.size());
                    range.TriangleOffset = uint32_t(output.Triangles.size());
                    range.SphereOffset = uint32_t(output.Spheres.size());
                    range.TriangleCount = uint32_t(local.Triangles.size());
                    const auto* bytes = static_cast<const uint8_t*>(nodeData);
                    if (nodeBytes) output.ObjectNodes.insert(output.ObjectNodes.end(), bytes, bytes + nodeBytes);
                    output.PrimitiveRefs.insert(output.PrimitiveRefs.end(),
                        local.PrimitiveRefs.begin(), local.PrimitiveRefs.end());
                    output.Triangles.insert(output.Triangles.end(), local.Triangles.begin(), local.Triangles.end());
                    output.Spheres.insert(output.Spheres.end(), local.Spheres.begin(), local.Spheres.end());
                }
                for (auto& instance : hierarchy.Instances)
                {
                    if (IsCancelled()) return Cancel(error);
                    const uint32_t first = instance.Geometry.FirstPrimitive;
                    instance.Geometry = ranges[instance.ObjectIndex];
                    instance.Geometry.FirstPrimitive = first;
                }
                output.Instances = std::move(hierarchy.Instances);
                output.InstanceNodes = std::move(hierarchy.Nodes);
                return !IsCancelled() || Cancel(error);
            }

            bool PackGeometry(const SceneGeometry& scene, AccelerationOptions acceleration,
                PackedScene& output, std::string& error)
            {
                Primitives.clear();
                Primitives.reserve(scene.Triangles.size() + scene.Spheres.size());
                output.Triangles.reserve(scene.Triangles.size());
                output.Spheres.reserve(scene.Spheres.size());
                for (std::uint32_t index = 0; index < scene.Triangles.size(); ++index)
                {
                    if ((index & 4095u) == 0 && IsCancelled()) return Cancel(error);
                    const Triangle& source = scene.Triangles[index];
                    const Vec3 position0 = ToVec3(source.Position0);
                    const Vec3 edge1 = Subtract(ToVec3(source.Position1), position0);
                    const Vec3 edge2 = Subtract(ToVec3(source.Position2), position0);
                    PackedTriangle triangle;
                    triangle.Position0 = {position0.X, position0.Y, position0.Z, 0.0f};
                    triangle.Edge1 = {edge1.X, edge1.Y, edge1.Z, 0.0f};
                    triangle.Edge2 = {edge2.X, edge2.Y, edge2.Z, 0.0f};
                    triangle.Normal0 = ToVec4(source.Normal0);
                    triangle.Normal1 = ToVec4(source.Normal1);
                    triangle.Normal2 = ToVec4(source.Normal2);
                    triangle.Color0 = ToVec4(source.Color0, 1.0f);
                    triangle.Color1 = ToVec4(source.Color1, 1.0f);
                    triangle.Color2 = ToVec4(source.Color2, 1.0f);
                    triangle.TexCoord01 = {
                        source.TexCoord0.X,
                        source.TexCoord0.Y,
                        source.TexCoord1.X,
                        source.TexCoord1.Y};
                    triangle.TexCoord2 = {
                        source.TexCoord2.X,
                        source.TexCoord2.Y,
                        0.0f,
                        0.0f};
                    triangle.MaterialIndex = source.MaterialIndex;
                    triangle.Flags = source.SmoothNormals ? 1u : 0u;
                    output.Triangles.push_back(triangle);

                    BuildPrimitive primitive;
                    primitive.BoundingBox.Expand(position0);
                    primitive.BoundingBox.Expand(Add(position0, edge1));
                    primitive.BoundingBox.Expand(Add(position0, edge2));
                    primitive.Centroid = primitive.BoundingBox.Center();
                    primitive.Reference.TypeAndIndex = index;
                    Primitives.push_back(primitive);
                }
                for (std::uint32_t index = 0; index < scene.Spheres.size(); ++index)
                {
                    if ((index & 4095u) == 0 && IsCancelled()) return Cancel(error);
                    const Sphere& source = scene.Spheres[index];
                    PackedSphere sphere;
                    sphere.CenterRadius = ToVec4(source.Center, source.Radius);
                    sphere.MaterialIndex = source.MaterialIndex;
                    output.Spheres.push_back(sphere);
                    const Vec3 center = ToVec3(source.Center);
                    const Vec3 radius = {source.Radius, source.Radius, source.Radius};
                    BuildPrimitive primitive;
                    primitive.BoundingBox.Expand(Subtract(center, radius));
                    primitive.BoundingBox.Expand(Add(center, radius));
                    primitive.Centroid = center;
                    primitive.Reference.TypeAndIndex = SphereBit | index;
                    Primitives.push_back(primitive);
                }
                if (!Primitives.empty() && acceleration.Type == AccelerationType::Bvh
                    && acceleration.BvhPolicy == BvhBuildPolicy::SpatialSplits)
                {
                    std::vector<ComputeAcceleration::SpatialPrimitive> primitives;
                    constexpr size_t CancellationStride = 1024;
                    primitives.reserve(Primitives.size());
                    for (size_t index = 0; index < Primitives.size(); ++index)
                    {
                        if ((index & 4095u) == 0 && IsCancelled()) return Cancel(error);
                        const auto& primitive = Primitives[index];
                        const auto& low = primitive.BoundingBox.Minimum;
                        const auto& high = primitive.BoundingBox.Maximum;
                        primitives.push_back({{low.X, low.Y, low.Z}, {high.X, high.Y, high.Z}, primitive.Reference.TypeAndIndex});
                    }
                    ComputeAcceleration::SpatialTree tree;
                    if (!ComputeAcceleration::BuildSpatialBvh(primitives, Cancellation, tree, error)) return false;
                    output.Nodes.reserve(tree.Nodes.size());
                    for (size_t index = 0; index < tree.Nodes.size(); ++index)
                    {
                        if (index % CancellationStride == 0 && IsCancelled()) return Cancel(error);
                        const auto& node = tree.Nodes[index];
                        output.Nodes.push_back({node.Minimum.X, node.Minimum.Y, node.Minimum.Z, node.FirstOrEscape,
                            node.Maximum.X, node.Maximum.Y, node.Maximum.Z, node.PrimitiveCount});
                    }
                    output.PrimitiveRefs.reserve(tree.References.size());
                    for (size_t index = 0; index < tree.References.size(); ++index)
                    {
                        if (index % CancellationStride == 0 && IsCancelled()) return Cancel(error);
                        output.PrimitiveRefs.push_back({tree.References[index]});
                    }
                }
                else if (!Primitives.empty() && acceleration.Type == AccelerationType::KdTree)
                {
                    std::vector<ComputeAcceleration::KdPrimitive> kdPrimitives;
                    kdPrimitives.reserve(Primitives.size());
                    for (size_t index = 0; index < Primitives.size(); ++index)
                    {
                        if ((index & 4095u) == 0 && IsCancelled()) return Cancel(error);
                        const auto& primitive = Primitives[index];
                        const auto& minimum = primitive.BoundingBox.Minimum;
                        const auto& maximum = primitive.BoundingBox.Maximum;
                        kdPrimitives.push_back({Float3(minimum.X, minimum.Y, minimum.Z),
                            Float3(maximum.X, maximum.Y, maximum.Z), primitive.Reference.TypeAndIndex});
                    }
                    ComputeAcceleration::KdTree tree;
                    if (!ComputeAcceleration::BuildKdTree(kdPrimitives, Cancellation, tree, error)) return false;
                    output.KdNodeCount = static_cast<uint32_t>(tree.Nodes.size());
                    const size_t headerBytes = sizeof(tree.Bounds);
                    const size_t nodeBytes = tree.Nodes.size() * sizeof(ComputeAcceleration::KdNode);
                    output.KdNodes.resize(headerBytes + nodeBytes);
                    std::memcpy(output.KdNodes.data(), &tree.Bounds, headerBytes);
                    std::memcpy(output.KdNodes.data() + headerBytes, tree.Nodes.data(), nodeBytes);
                    output.PrimitiveRefs.reserve(tree.References.size());
                    for (const uint32_t reference : tree.References)
                        output.PrimitiveRefs.push_back({reference});
                }
                else if (!Primitives.empty() && acceleration.Type == AccelerationType::Bih)
                {
                    std::vector<ComputeAcceleration::BihPrimitive> bihPrimitives;
                    bihPrimitives.reserve(Primitives.size());
                    for (size_t index = 0; index < Primitives.size(); ++index)
                    {
                        if ((index & 4095u) == 0 && IsCancelled()) return Cancel(error);
                        const auto& primitive = Primitives[index];
                        const auto& minimum = primitive.BoundingBox.Minimum;
                        const auto& maximum = primitive.BoundingBox.Maximum;
                        bihPrimitives.push_back({Float3(minimum.X, minimum.Y, minimum.Z),
                            Float3(maximum.X, maximum.Y, maximum.Z), primitive.Reference.TypeAndIndex});
                    }
                    ComputeAcceleration::BihTree tree;
                    if (!ComputeAcceleration::BuildBih(bihPrimitives, Cancellation, tree, error)) return false;
                    output.BihNodeCount = static_cast<uint32_t>(tree.Nodes.size());
                    const size_t headerBytes = sizeof(tree.Bounds);
                    const size_t nodeBytes = tree.Nodes.size() * sizeof(ComputeAcceleration::BihNode);
                    output.BihNodes.resize(headerBytes + nodeBytes);
                    std::memcpy(output.BihNodes.data(), &tree.Bounds, headerBytes);
                    std::memcpy(output.BihNodes.data() + headerBytes, tree.Nodes.data(), nodeBytes);
                    output.PrimitiveRefs.reserve(tree.References.size());
                    for (size_t index = 0; index < tree.References.size(); ++index)
                    {
                        if ((index & 4095u) == 0 && IsCancelled()) return Cancel(error);
                        output.PrimitiveRefs.push_back({tree.References[index]});
                    }
                }
                else if (!Primitives.empty())
                {
                    constexpr size_t CancellationStride = 1024;
                    std::vector<ComputeBvh::Primitive> input;
                    input.reserve(Primitives.size());
                    for (size_t index = 0; index < Primitives.size(); ++index)
                    {
                        if (index % CancellationStride == 0 && IsCancelled()) return Cancel(error);
                        const auto& primitive = Primitives[index];
                        const auto& box = primitive.BoundingBox;
                        const auto& center = primitive.Centroid;
                        input.push_back({{box.Minimum.X, box.Minimum.Y, box.Minimum.Z},
                            {box.Maximum.X, box.Maximum.Y, box.Maximum.Z}, {center.X, center.Y, center.Z},
                            primitive.Reference.TypeAndIndex});
                    }
                    ComputeBvh::Tree tree;
                    if (!ComputeBvh::Build(input, Cancellation, tree, error)) return false;
                    output.Nodes.resize(tree.Nodes.size());
                    for (size_t index = 0; index < tree.Nodes.size(); ++index)
                    {
                        if (index % CancellationStride == 0 && IsCancelled()) return Cancel(error);
                        const auto& source = tree.Nodes[index];
                        auto& target = output.Nodes[index];
                        target.MinX = source.MinX;
                        target.MinY = source.MinY;
                        target.MinZ = source.MinZ;
                        target.MaxX = source.MaxX;
                        target.MaxY = source.MaxY;
                        target.MaxZ = source.MaxZ;
                        target.FirstOrEscape = source.FirstOrEscape;
                        target.PrimitiveCount = source.PrimitiveCount;
                    }
                    output.PrimitiveRefs.reserve(tree.References.size());
                    for (size_t index = 0; index < tree.References.size(); ++index)
                    {
                        if (index % CancellationStride == 0 && IsCancelled()) return Cancel(error);
                        output.PrimitiveRefs.push_back({tree.References[index]});
                    }
                }
                return true;
            }

        private:

            bool IsCancelled() const
            {
                return Cancellation && Cancellation();
            }

            bool Cancel(std::string& error) const
            {
                error = "Compute scene packing was cancelled.";
                return false;
            }

            Detail::CancellationProbe Cancellation;
            std::vector<BuildPrimitive> Primitives;
        };

        struct DynamicModule
        {
            ~DynamicModule()
            {
                if (Handle) FreeLibrary(Handle);
            }

            DynamicModule(const DynamicModule&) = delete;
            DynamicModule& operator=(const DynamicModule&) = delete;
            DynamicModule() = default;

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

        struct GpuBuffer
        {
            ComPtr<ID3D12Resource> Resource;
            std::uint64_t Size = 0;
            D3D12_RESOURCE_STATES State = D3D12_RESOURCE_STATE_COMMON;

            void Reset()
            {
                Resource.Reset();
                Size = 0;
                State = D3D12_RESOURCE_STATE_COMMON;
            }
        };

        std::string HResultError(const char* operation, HRESULT result)
        {
            char code[32]{};
            std::snprintf(code, sizeof(code), "0x%08lx", static_cast<unsigned long>(result));
            return std::string(operation) + " failed (" + code + ").";
        }

        std::string Narrow(const wchar_t* text)
        {
            if (!text || !*text) return {};
            const int required = WideCharToMultiByte(
                CP_UTF8, 0, text, -1, nullptr, 0, nullptr, nullptr);
            if (required <= 1) return {};
            std::string result(static_cast<std::size_t>(required), '\0');
            WideCharToMultiByte(
                CP_UTF8, 0, text, -1, result.data(), required, nullptr, nullptr);
            result.pop_back();
            return result;
        }
#endif
    }

    struct ComputeShaderRayTracingRHI::Impl
    {
        explicit Impl(ComputeDeviceSelection selection) : Selection(selection)
        {
#if defined(_WIN32)
            Initialize();
#else
            AvailabilityError = "D3D12 compute is available only on Windows.";
#endif
        }

        ~Impl()
        {
            ComputeQueue.WaitIdle();
#if defined(_WIN32)
            if (Queue && Fence && FenceEvent.Handle)
            {
                std::string ignored;
                SignalAndWait(ignored);
            }
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
            return ComputeQueue.Submit(
                [this, scene, owner = std::move(owner), camera, settings, external]
                (const CancellationToken& queued) mutable
                {
                    if (!scene)
                    {
                        return Detail::MakeFailedResult(
                            "The compute shader submission has no scene.");
                    }
                    const Detail::CancellationProbe cancellation =
                        Detail::CombineCancellation(queued, external);
                    if (cancellation()) return Detail::MakeCancelledResult();
                    if (!Available)
                    {
                        return Detail::MakeFailedResult(
                            std::string("D3D12 compute is unavailable: ") + AvailabilityError);
                    }
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
                                    [&] { return CaptureGuidesNative(*scene, camera, raster, cancellation); },
                                    error);
                                if (!guides)
                                    return false;
                            }
                            return ReconstructNative(film, settings, guides, cancellation, image, error);
                        },
                        [&](const auto &input, auto controls, auto &counts, auto &error)
                        { return AllocateDenoisedNative(input, controls, cancellation, counts, error); });
#else
                    return Detail::MakeFailedResult("D3D12 compute is available only on Windows.");
#endif
                });
        }

        RayQueryResult Query(const Scene& scene, const std::vector<Ray>& rays,
            AccelerationOptions acceleration, const CancellationToken& external) const
        {
            // Reuse the backend queue: scene caches, descriptors and the command
            // allocator must never race a render submitted by the editor.
            auto result = std::make_shared<RayQueryResult>();
            auto job = ComputeQueue.Submit([this, &scene, &rays, acceleration, external, result]
                (const CancellationToken& queued)
            {
                const auto cancellation = Detail::CombineCancellation(queued, external);
                if (cancellation()) return Detail::MakeCancelledResult();
                if (!Available) return Detail::MakeFailedResult("D3D12 compute is unavailable: " + AvailabilityError);
#if defined(_WIN32)
                *result = QueryNative(scene, rays, acceleration, cancellation);
                RenderResult completion;
                completion.Status = result->Status;
                completion.Error = result->Error;
                return completion;
#else
                return Detail::MakeFailedResult("D3D12 compute is available only on Windows.");
#endif
            });
            const auto completion = job.GetResult();
            result->Status = completion.Status;
            result->Error = completion.Error;
            return std::move(*result);
        }

#if defined(_WIN32)
        using D3D12CreateDeviceFunction = HRESULT(WINAPI*)(
            IUnknown*, D3D_FEATURE_LEVEL, REFIID, void**);
        using CreateDxgiFactory2Function = HRESULT(WINAPI*)(UINT, REFIID, void**);

        bool Check(HRESULT result, const char* operation, std::string& error) const
        {
            if (SUCCEEDED(result)) return true;
            error = HResultError(operation, result);
            return false;
        }

        bool Initialize()
        {
            const bool gpuValidation = GetEnvironmentVariableW(L"ARVO_D3D12_GPU_VALIDATION", nullptr, 0) != 0;
            const bool diagnostics = Selection == ComputeDeviceSelection::WarpDiagnostic || gpuValidation ||
                GetEnvironmentVariableW(L"ARVO_D3D12_DEBUG", nullptr, 0) != 0;
            if (!D3D12Module.Load(L"d3d12.dll") || !DxgiModule.Load(L"dxgi.dll"))
            {
                AvailabilityError = "The D3D12 or DXGI runtime could not be loaded.";
                return false;
            }
            CreateDevice = D3D12Module.Get<D3D12CreateDeviceFunction>("D3D12CreateDevice");
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
            }
            CreateFactory = DxgiModule.Get<CreateDxgiFactory2Function>("CreateDXGIFactory2");
            if (!CreateDevice || !CreateFactory)
            {
                AvailabilityError = "The D3D12 runtime does not export the required functions.";
                return false;
            }

            ComPtr<IDXGIFactory4> factory;
            if (!Check(CreateFactory(0, IID_PPV_ARGS(&factory)),
                    "CreateDXGIFactory2", AvailabilityError))
            {
                return false;
            }
            std::size_t bestMemory = 0;
            for (UINT ordinal = 0;; ++ordinal)
            {
                ComPtr<IDXGIAdapter1> adapter;
                if (Selection == ComputeDeviceSelection::WarpDiagnostic)
                {
                    if (ordinal != 0 || FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)))) break;
                }
                else if (factory->EnumAdapters1(ordinal, &adapter) == DXGI_ERROR_NOT_FOUND) break;
                DXGI_ADAPTER_DESC1 description{};
                if (FAILED(adapter->GetDesc1(&description))
                    || (Selection == ComputeDeviceSelection::Hardware
                        && (description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0))
                {
                    continue;
                }
                ComPtr<ID3D12Device> candidate;
                if (FAILED(CreateDevice(
                        adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&candidate))))
                {
                    continue;
                }
                // Affine ray transforms use double operations before conditioning
                // the object-space float ray. D3D12 exposes extended doubles here:
                // https://learn.microsoft.com/windows/win32/api/d3d12/ns-d3d12-d3d12_feature_data_d3d12_options
                D3D12_FEATURE_DATA_D3D12_OPTIONS options{};
                if (FAILED(candidate->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &options, sizeof(options)))
                    || !options.DoublePrecisionFloatShaderOps) continue;
                if (!Device || description.DedicatedVideoMemory >= bestMemory)
                {
                    bestMemory = description.DedicatedVideoMemory;
                    Device = std::move(candidate);
                    Adapter = std::move(adapter);
                    AdapterName = Narrow(description.Description);
                }
            }
            if (!Device || !Adapter)
            {
                AvailabilityError = Selection == ComputeDeviceSelection::WarpDiagnostic
                    ? "No WARP diagnostic device with double-precision shader operations is available."
                    : "No hardware D3D12 adapter with double-precision shader operations is available.";
                return false;
            }
            Adapter.As(&Adapter3);
            if (diagnostics
                && SUCCEEDED(Device.As(&DiagnosticInfo)))
            {
                DiagnosticInfo->RegisterMessageCallback(
                    [](D3D12_MESSAGE_CATEGORY, D3D12_MESSAGE_SEVERITY severity, D3D12_MESSAGE_ID id,
                       LPCSTR message, void*) {
                        if (severity <= D3D12_MESSAGE_SEVERITY_ERROR)
                            std::fprintf(stderr, "D3D12 diagnostic %u: %s\n", static_cast<unsigned>(id), message);
                    },
                    D3D12_MESSAGE_CALLBACK_FLAG_NONE, nullptr, &DiagnosticCookie);
            }
            D3D12_FEATURE_DATA_SHADER_MODEL shaderModel{D3D_SHADER_MODEL_6_0};
            if (!Check(Device->CheckFeatureSupport(
                    D3D12_FEATURE_SHADER_MODEL, &shaderModel, sizeof(shaderModel)),
                    "CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL)", AvailabilityError)
                || shaderModel.HighestShaderModel < D3D_SHADER_MODEL_6_0)
            {
                if (AvailabilityError.empty())
                {
                    AvailabilityError = "The D3D12 adapter does not support shader model 6.0.";
                }
                return false;
            }

            D3D12_COMMAND_QUEUE_DESC queueDescription{};
            const auto commandType = D3D12_COMMAND_LIST_TYPE_COMPUTE;
            queueDescription.Type = commandType;
            if (!Check(Device->CreateCommandQueue(
                    &queueDescription, IID_PPV_ARGS(&Queue)),
                    "ID3D12Device::CreateCommandQueue", AvailabilityError)
                || !Check(Device->CreateCommandAllocator(
                    commandType, IID_PPV_ARGS(&Allocator)),
                    "ID3D12Device::CreateCommandAllocator", AvailabilityError)
                || !Check(Device->CreateCommandList(
                    0, commandType, Allocator.Get(), nullptr,
                    IID_PPV_ARGS(&CommandList)),
                    "ID3D12Device::CreateCommandList", AvailabilityError)
                || !Check(CommandList->Close(), "ID3D12GraphicsCommandList::Close", AvailabilityError)
                || !Check(Device->CreateFence(
                    0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&Fence)),
                    "ID3D12Device::CreateFence", AvailabilityError))
            {
                return false;
            }
            FenceEvent.Handle = CreateEventW(nullptr, FALSE, FALSE, nullptr);
            if (!FenceEvent.Handle)
            {
                AvailabilityError = "CreateEventW failed for the D3D12 compute fence.";
                return false;
            }

            D3D12_DESCRIPTOR_HEAP_DESC heapDescription{};
            heapDescription.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
            heapDescription.NumDescriptors = DescriptorCount;
            heapDescription.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
            if (!Check(Device->CreateDescriptorHeap(
                    &heapDescription, IID_PPV_ARGS(&DescriptorHeap)),
                    "ID3D12Device::CreateDescriptorHeap", AvailabilityError))
            {
                return false;
            }
            DescriptorSize = Device->GetDescriptorHandleIncrementSize(
                D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            heapDescription.NumDescriptors = 1;
            heapDescription.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
            if (!Check(Device->CreateDescriptorHeap(&heapDescription, IID_PPV_ARGS(&ErrorClearHeap)),
                "CreateDescriptorHeap(instance error clear)", AvailabilityError)) return false;
            if (!CreateRootSignature(AvailabilityError)) return false;
            Available = true;
            return true;
        }

        bool CreateRootSignature(std::string& error)
        {
            return Check(Device->CreateRootSignature(
                    0, Detail::ComputeRootSignature, sizeof(Detail::ComputeRootSignature),
                    IID_PPV_ARGS(&RootSignature)),
                "ID3D12Device::CreateRootSignature", error);
        }

        bool CreatePipeline(
            const void* bytecode,
            std::size_t bytecodeSize,
            ComPtr<ID3D12PipelineState>& pipeline,
            const char* operation,
            std::string& error, ID3D12RootSignature* signature = nullptr) const
        {
            if (pipeline) return true;
            D3D12_COMPUTE_PIPELINE_STATE_DESC description{};
            description.pRootSignature = signature ? signature : RootSignature.Get();
            description.CS = {bytecode, bytecodeSize};
            if (Selection == ComputeDeviceSelection::WarpDiagnostic)
                std::fprintf(stderr, "WARP pipeline begin: %s\n", operation);
            const bool ready = Check(Device->CreateComputePipelineState(
                    &description, IID_PPV_ARGS(&pipeline)), operation, error);
            if (Selection == ComputeDeviceSelection::WarpDiagnostic)
                std::fprintf(stderr, "WARP pipeline end: %s (%s)\n", operation, ready ? "ready" : "failed");
            return ready;
        }

        bool EnsureRenderPipelines(Algorithm algorithm, std::string& error) const
        {
            // Called only on ComputeQueue. Driver compilation can be expensive;
            // backend discovery and GUI construction must never wait for it.
            // Each handwritten integrator retains its own cached pipeline.
            bool ready = false;
            switch (algorithm)
            {
            case Algorithm::PointBasedGlobalIllumination:
                ready = EnsurePointGiPipelines(error);
                break;
            case Algorithm::VertexConnectionMerging:
                ready = EnsureVcmPipelines(error);
                break;
            case Algorithm::ProgressivePhotonMapping:
                ready = EnsureProgressivePhotonPipelines(error);
                break;
            case Algorithm::PhotonMapping:
                ready = EnsurePhotonPipelines(error);
                break;
            case Algorithm::MetropolisLightTransport:
                ready = EnsureMetropolisPipelines(error);
                break;
            case Algorithm::BidirectionalPathTracing:
                ready = EnsureBidirectionalPipelines(error);
                break;
            case Algorithm::RayTracing:
                ready = CreatePipeline(Detail::ComputeWhittedDxil, sizeof(Detail::ComputeWhittedDxil),
                    WhittedPipeline, "CreateComputePipelineState(Whitted)", error);
                break;
            case Algorithm::PathTracing:
                ready = CreatePipeline(Detail::ComputePathTracingDxil, sizeof(Detail::ComputePathTracingDxil),
                    PathTracingPipeline, "CreateComputePipelineState(PathTracing)", error);
                break;
            case Algorithm::DistributedRayTracing:
                ready = CreatePipeline(Detail::ComputeDistributedRayDxil, sizeof(Detail::ComputeDistributedRayDxil),
                    DistributedRayPipeline, "CreateComputePipelineState(DistributedRayTracing)", error);
                break;
            default: error = "No native Compute pipeline for the integrator."; return false;
            }
            return ready
                && CreatePipeline(Detail::ComputeClearPixelsDxil, sizeof(Detail::ComputeClearPixelsDxil),
                    ClearPixelsPipeline, "CreateComputePipelineState(ClearPixels)", error)
                && CreatePipeline(Detail::ComputeClearStatisticsDxil, sizeof(Detail::ComputeClearStatisticsDxil),
                    ClearStatisticsPipeline, "CreateComputePipelineState(ClearStatistics)", error)
                && CreatePipeline(Detail::ComputeResolvePixelsDxil, sizeof(Detail::ComputeResolvePixelsDxil),
                    ResolvePixelsPipeline, "CreateComputePipelineState(ResolvePixels)", error);
        }

        bool BeginCommands(std::string& error) const
        {
            if (Fence->GetCompletedValue() < FenceValue)
            {
                error = "The previous compute submission has not retired.";
                return false;
            }
            if (!Check(Allocator->Reset(), "ID3D12CommandAllocator::Reset", error)
                || !Check(CommandList->Reset(Allocator.Get(), nullptr),
                    "ID3D12GraphicsCommandList::Reset", error))
            {
                return false;
            }
            ID3D12DescriptorHeap* heaps[] = {DescriptorHeap.Get()};
            CommandList->SetDescriptorHeaps(1, heaps);
            CommandList->SetComputeRootSignature(RootSignature.Get());
            D3D12_GPU_DESCRIPTOR_HANDLE handle =
                DescriptorHeap->GetGPUDescriptorHandleForHeapStart();
            CommandList->SetComputeRootDescriptorTable(0, handle);
            handle.ptr += static_cast<std::uint64_t>(FrameDescriptorBegin) * DescriptorSize;
            CommandList->SetComputeRootDescriptorTable(1, handle);
            FrameConstantCursor = 0;
            return true;
        }

        void RetireFailedDevice() const
        {
            // Submission and event-registration failures do not prove completion.
            // Request logical-device removal before temporary buffers unwind;
            // removal signals its fences without restarting the display adapter.
            // https://learn.microsoft.com/windows/win32/api/d3d12/nf-d3d12-id3d12device5-removedevice
            ComPtr<ID3D12Device5> failedDevice;
            if (SUCCEEDED(Device.As(&failedDevice))) failedDevice->RemoveDevice();
        }

        bool SignalAndWait(std::string& error) const
        {
            const std::uint64_t value = ++FenceValue;
            if (!Check(Queue->Signal(Fence.Get(), value), "ID3D12CommandQueue::Signal", error))
            {
                RetireFailedDevice();
                return false;
            }
            if (Fence->GetCompletedValue() < value)
            {
                if (!Check(Fence->SetEventOnCompletion(value, FenceEvent.Handle),
                        "ID3D12Fence::SetEventOnCompletion", error))
                {
                    RetireFailedDevice();
                    return false;
                }
                const DWORD waitResult =
                    WaitForSingleObject(FenceEvent.Handle, Selection == ComputeDeviceSelection::WarpDiagnostic
                        ? WarpWaitTimeoutMilliseconds : GpuWaitTimeoutMilliseconds);
                if (waitResult != WAIT_OBJECT_0)
                {
                    const HRESULT removedReason = Device->GetDeviceRemovedReason();
                    error = FAILED(removedReason)
                        ? HResultError("The D3D12 compute device was removed", removedReason)
                        : waitResult == WAIT_TIMEOUT
                        ? "Waiting for the D3D12 compute fence timed out."
                        : "Waiting for the D3D12 compute fence failed.";
                    RetireFailedDevice();
                    return false;
                }
            }
            const HRESULT removedReason = Device->GetDeviceRemovedReason();
            if (FAILED(removedReason))
            {
                error = HResultError("The D3D12 compute device was removed", removedReason);
                return false;
            }
            return true;
        }

        bool ExecuteCommands(std::string& error) const
        {
            if (!Check(CommandList->Close(), "ID3D12GraphicsCommandList::Close", error))
            {
                return false;
            }
            ID3D12CommandList* lists[] = {CommandList.Get()};
            Queue->ExecuteCommandLists(1, lists);
            return SignalAndWait(error);
        }

        bool BindDispatch(
            ID3D12PipelineState* pipeline,
            const FrameConstants& constants, std::string& error) const
        {
            // A clear/resolve list may record many dispatches before execution.
            // Each one needs immutable constants until its fence completes. Use
            // aligned records in reusable upload pages, not one allocation per tile.
            const uint64_t page = FrameConstantCursor / FrameConstantsPerPage;
            const uint64_t offset = (FrameConstantCursor % FrameConstantsPerPage) * FrameConstantStride;
            if (page == FrameConstantPages.size())
            {
                GpuBuffer upload;
                if (!CreateBuffer(upload, FrameConstantPageBytes, D3D12_HEAP_TYPE_UPLOAD,
                                  D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_FLAG_NONE, error))
                {
                    CommandList->Close();
                    return false;
                }
                FrameConstantPages.push_back(std::move(upload));
            }
            auto* resource = FrameConstantPages[page].Resource.Get();
            void* mapped = nullptr;
            const D3D12_RANGE noRead{0, 0},
                written{SIZE_T(offset), SIZE_T(offset + MediumConstantOffset + sizeof(PathFrameData))};
            if (!Check(resource->Map(0, &noRead, &mapped), "Map(compute dispatch constants)", error))
            {
                CommandList->Close();
                return false;
            }
            std::memcpy(static_cast<uint8_t*>(mapped) + offset, &constants, sizeof(constants));
            std::memcpy(static_cast<uint8_t*>(mapped) + offset + MediumConstantOffset, &CurrentMedium, sizeof(CurrentMedium));
            resource->Unmap(0, &written);
            ++FrameConstantCursor;
            CommandList->SetPipelineState(pipeline);
            CommandList->SetComputeRootConstantBufferView(2, resource->GetGPUVirtualAddress() + offset);
            CommandList->SetComputeRootConstantBufferView(MediumRootParameter, resource->GetGPUVirtualAddress() + offset + MediumConstantOffset);
            return true;
        }

        void Transition(GpuBuffer& buffer, D3D12_RESOURCE_STATES state) const
        {
            if (!buffer.Resource || buffer.State == state) return;
            D3D12_RESOURCE_BARRIER barrier{};
            barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barrier.Transition.pResource = buffer.Resource.Get();
            barrier.Transition.StateBefore = buffer.State;
            barrier.Transition.StateAfter = state;
            barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            CommandList->ResourceBarrier(1, &barrier);
            buffer.State = state;
        }

        void UavBarrier(const GpuBuffer& buffer) const
        {
            if (!buffer.Resource) return;
            D3D12_RESOURCE_BARRIER barrier{};
            barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
            barrier.UAV.pResource = buffer.Resource.Get();
            CommandList->ResourceBarrier(1, &barrier);
        }

        bool CreateBuffer(
            GpuBuffer& buffer,
            std::uint64_t size,
            D3D12_HEAP_TYPE heapType,
            D3D12_RESOURCE_STATES initialState,
            D3D12_RESOURCE_FLAGS flags,
            std::string& error) const
        {
            buffer.Reset();
            if (size == 0) return true;
            D3D12_HEAP_PROPERTIES heapProperties{};
            heapProperties.Type = heapType;
            heapProperties.CreationNodeMask = 1;
            heapProperties.VisibleNodeMask = 1;
            D3D12_RESOURCE_DESC description{};
            description.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
            description.Width = size;
            description.Height = 1;
            description.DepthOrArraySize = 1;
            description.MipLevels = 1;
            description.Format = DXGI_FORMAT_UNKNOWN;
            description.SampleDesc.Count = 1;
            description.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            description.Flags = flags;
            if (!Check(Device->CreateCommittedResource(
                    &heapProperties, D3D12_HEAP_FLAG_NONE, &description,
                    initialState, nullptr, IID_PPV_ARGS(&buffer.Resource)),
                "ID3D12Device::CreateCommittedResource", error))
            {
                return false;
            }
            buffer.Size = size;
            buffer.State = initialState;
            return true;
        }

        bool CheckVideoMemory(
            std::uint64_t required,
            const char* label,
            std::string& error) const
        {
            if (!Adapter3) return true;
            DXGI_QUERY_VIDEO_MEMORY_INFO info{};
            if (FAILED(Adapter3->QueryVideoMemoryInfo(
                    0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info)))
            {
                return true;
            }
            const std::uint64_t available =
                info.Budget > info.CurrentUsage ? info.Budget - info.CurrentUsage : 0;
            if (required > available - available / 10)
            {
                error = std::string(label)
                    + " exceed 90 percent of the available D3D12 video-memory budget.";
                return false;
            }
            return true;
        }

        bool PrepareUpload(
            GpuBuffer& destination,
            const void* data,
            std::size_t size,
            ComPtr<ID3D12Resource>& staging,
            std::string& error) const
        {
            if (size == 0)
            {
                destination.Reset();
                staging.Reset();
                return true;
            }
            if (!CreateBuffer(
                    destination, size, D3D12_HEAP_TYPE_DEFAULT,
                    D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE, error))
            {
                return false;
            }
            GpuBuffer upload;
            if (!CreateBuffer(
                    upload, size, D3D12_HEAP_TYPE_UPLOAD,
                    D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_FLAG_NONE, error))
            {
                return false;
            }
            void* mapped = nullptr;
            D3D12_RANGE noRead{0, 0};
            if (!Check(upload.Resource->Map(0, &noRead, &mapped),
                    "ID3D12Resource::Map(upload)", error))
            {
                return false;
            }
            std::memcpy(mapped, data, size);
            upload.Resource->Unmap(0, nullptr);
            staging = std::move(upload.Resource);
            return true;
        }

        void ReleaseSceneBuffers() const
        {
            Instances.Reset();
            InstanceNodes.Reset();
            InstanceError.Reset();
            InstanceErrorReadback.Reset();
            InstanceCount = InstanceNodeCount = 0;
            Nodes.Reset();
            PrimitiveRefs.Reset();
            Triangles.Reset();
            Spheres.Reset();
            Materials.Reset();
            Lights.Reset();
            TextureDescriptors.Reset();
            TextureTexels.Reset();
            SpectralPalette.Reset();
            SpectralReferences.Reset();
            NodeCount = PrimitiveRefCount = TriangleCount = SphereCount = 0;
            MaterialCount = LightCount = 0;
            TextureCount = TextureTexelCount = 0;
            CachedOwner.reset();
        }

        void ReleaseFrameBuffers() const
        {
            Accumulation.Reset();
            SampleCounts.Reset();
            SampleTargets.Reset();
            Output.Reset();
            Statistics.Reset();
            OutputReadback.Reset();
            FilmReadback.Reset();
            SamplesReadback.Reset();
            TargetsReadback.Reset();
            StatisticsReadback.Reset();
            FramePixelCapacity = 0;
        }

        bool OwnerMatches(const std::shared_ptr<const Scene>& owner, AccelerationOptions acceleration) const
        {
            if (!owner || CachedAcceleration != acceleration) return false;
            const std::shared_ptr<const Scene> cached = CachedOwner.lock();
            return cached && cached.get() == owner.get();
        }

        void CreateRawSrv(uint32_t descriptorIndex, const GpuBuffer& buffer,
                          ID3D12DescriptorHeap* heap = nullptr) const
        {
            D3D12_SHADER_RESOURCE_VIEW_DESC description{};
            description.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            description.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            description.Format = DXGI_FORMAT_R32_TYPELESS;
            description.Buffer.NumElements = static_cast<UINT>(std::max<uint64_t>(buffer.Size / sizeof(uint32_t), 1));
            description.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
            auto handle = (heap ? heap : DescriptorHeap.Get())->GetCPUDescriptorHandleForHeapStart();
            handle.ptr += static_cast<uint64_t>(descriptorIndex) * DescriptorSize;
            Device->CreateShaderResourceView(buffer.Resource.Get(), &description, handle);
        }

        void CreateStructuredSrv(
            std::uint32_t descriptorIndex,
            const GpuBuffer& buffer,
            std::uint32_t count,
            std::uint32_t stride, ID3D12DescriptorHeap* heap = nullptr) const
        {
            D3D12_SHADER_RESOURCE_VIEW_DESC description{};
            description.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            description.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            description.Format = DXGI_FORMAT_UNKNOWN;
            description.Buffer.NumElements = std::max(count, 1u);
            description.Buffer.StructureByteStride = stride;
            D3D12_CPU_DESCRIPTOR_HANDLE handle =
                (heap ? heap : DescriptorHeap.Get())->GetCPUDescriptorHandleForHeapStart();
            handle.ptr += static_cast<std::uint64_t>(descriptorIndex) * DescriptorSize;
            Device->CreateShaderResourceView(buffer.Resource.Get(), &description, handle);
        }

        void CreateStructuredUav(
            std::uint32_t descriptorIndex,
            const GpuBuffer& buffer,
            std::uint32_t count,
            std::uint32_t stride, ID3D12DescriptorHeap* heap = nullptr) const
        {
            D3D12_UNORDERED_ACCESS_VIEW_DESC description{};
            description.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
            description.Format = DXGI_FORMAT_UNKNOWN;
            description.Buffer.NumElements = std::max(count, 1u);
            description.Buffer.StructureByteStride = stride;
            D3D12_CPU_DESCRIPTOR_HANDLE handle =
                (heap ? heap : DescriptorHeap.Get())->GetCPUDescriptorHandleForHeapStart();
            handle.ptr += static_cast<std::uint64_t>(descriptorIndex) * DescriptorSize;
            Device->CreateUnorderedAccessView(
                buffer.Resource.Get(), nullptr, &description, handle);
        }

        void CreateRawUav(
            std::uint32_t descriptorIndex,
            const GpuBuffer& buffer,
            std::uint32_t dwordCount,
            ID3D12DescriptorHeap* heap = nullptr) const
        {
            D3D12_UNORDERED_ACCESS_VIEW_DESC description{};
            description.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
            description.Format = DXGI_FORMAT_R32_TYPELESS;
            description.Buffer.NumElements = dwordCount;
            description.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
            D3D12_CPU_DESCRIPTOR_HANDLE handle =
                (heap ? heap : DescriptorHeap.Get())->GetCPUDescriptorHandleForHeapStart();
            handle.ptr += static_cast<std::uint64_t>(descriptorIndex) * DescriptorSize;
            Device->CreateUnorderedAccessView(
                buffer.Resource.Get(), nullptr, &description, handle);
        }

        void RefreshSceneDescriptors() const
        {
            CreateStructuredSrv(SpectralPaletteDescriptor, SpectralPalette,
                                uint32_t(SpectralPalette.Size / sizeof(SpectralSceneData::GpuSpectrum)),
                                sizeof(SpectralSceneData::GpuSpectrum));
            CreateStructuredSrv(SpectralReferenceDescriptor, SpectralReferences,
                                uint32_t(SpectralReferences.Size / sizeof(uint32_t)), sizeof(uint32_t));
            CreateStructuredSrv(8, Instances, InstanceCount, sizeof(ComputeInstances::Instance));
            CreateStructuredSrv(9, InstanceNodes, InstanceNodeCount, sizeof(ComputeInstances::Node));
            CreateRawUav(InstanceErrorDescriptor, InstanceError, 1);
            CreateRawUav(0, InstanceError, 1, ErrorClearHeap.Get());
            CreateRawSrv(0, Nodes);
            CreateStructuredSrv(1, PrimitiveRefs, PrimitiveRefCount, sizeof(PackedPrimitiveRef));
            CreateStructuredSrv(2, Triangles, TriangleCount, sizeof(PackedTriangle));
            CreateStructuredSrv(3, Spheres, SphereCount, sizeof(PackedSphere));
            CreateStructuredSrv(4, Materials, MaterialCount, sizeof(PackedMaterial));
            CreateStructuredSrv(5, Lights, LightCount, sizeof(PackedLight));
            CreateStructuredSrv(
                6,
                TextureDescriptors,
                TextureCount,
                sizeof(Detail::PackedRayTextureDescriptor));
            CreateStructuredSrv(
                7,
                TextureTexels,
                TextureTexelCount,
                sizeof(std::uint32_t));
        }

        void RefreshFrameDescriptors(std::uint32_t pixelCount) const
        {
            CreateStructuredUav(FrameDescriptorBegin, Accumulation, pixelCount, sizeof(Vec4));
            CreateStructuredUav(FrameDescriptorBegin + 1, SampleCounts, pixelCount, sizeof(std::uint32_t));
            CreateStructuredUav(FrameDescriptorBegin + 2, Output, FramePixelCapacity, sizeof(std::uint32_t));
            CreateRawUav(FrameDescriptorBegin + 3, Statistics, sizeof(GroupStatistics) / sizeof(std::uint32_t));
            CreateStructuredUav(FrameDescriptorBegin + 4, SampleTargets, pixelCount, sizeof(std::uint32_t));
        }

        bool PrepareSpectralCoefficients(const SpectralSceneData::Request* sources, uint32_t count,
                                         const Detail::CancellationProbe& cancellation, GpuBuffer& output,
                                         std::string& error) const
        {
            namespace S = SpectralSceneData;
            constexpr uint32_t ResponseEntries = 256, FitBatchSize = 8192, FitGroupSize = 64;
            constexpr uint32_t SourceRoot = 0, ResponseRoot = 1, ResponseOutputRoot = 5;
            constexpr uint32_t CoefficientOutputRoot = 6, ControlsRoot = 8, FitConverged = 0;
            if (count == 0)
            {
                error = "Compute spectral preparation requires at least one source.";
                return false;
            }
            if (!SpectralRoot &&
                !Check(Device->CreateRootSignature(0, Detail::ComputeSpectralRootSignature,
                                                   sizeof(Detail::ComputeSpectralRootSignature),
                                                   IID_PPV_ARGS(&SpectralRoot)),
                       "Creating the compute spectral preparation root", error))
                return false;
            if (!CreatePipeline(Detail::ComputeSpectralResponseDxil,
                                sizeof(Detail::ComputeSpectralResponseDxil), SpectralResponsePipeline,
                                "Creating the compute spectral quadrature pipeline", error,
                                SpectralRoot.Get()) ||
                !CreatePipeline(Detail::ComputeSpectralPaletteDxil,
                                sizeof(Detail::ComputeSpectralPaletteDxil), SpectralPalettePipeline,
                                "Creating the compute spectral fit pipeline", error, SpectralRoot.Get()))
                return false;
            const uint64_t sourceBytes = uint64_t(count) * sizeof(S::Request);
            const uint64_t coefficientBytes = uint64_t(count) * sizeof(S::GpuSpectrum);
            if (!CheckVideoMemory(sourceBytes + coefficientBytes + ResponseEntries * sizeof(Vec4),
                                  "Compute spectral preparation", error))
                return false;
            GpuBuffer input, response, coefficients, readback;
            ComPtr<ID3D12Resource> staging;
            if (!PrepareUpload(input, sources, size_t(sourceBytes), staging, error) ||
                !CreateBuffer(response, ResponseEntries * sizeof(Vec4), D3D12_HEAP_TYPE_DEFAULT,
                              D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                              D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, error) ||
                !CreateBuffer(coefficients, coefficientBytes, D3D12_HEAP_TYPE_DEFAULT,
                              D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                              D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, error) ||
                !CreateBuffer(readback, uint64_t(std::min(count, FitBatchSize)) * sizeof(S::GpuSpectrum),
                              D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST,
                              D3D12_RESOURCE_FLAG_NONE, error))
                return false;
            if (cancellation())
            {
                error = "Compute spectral preparation cancelled.";
                return false;
            }
            if (!BeginCommands(error))
                return false;
            CommandList->CopyBufferRegion(input.Resource.Get(), 0, staging.Get(), 0, input.Size);
            Transition(input, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            CommandList->SetComputeRootSignature(SpectralRoot.Get());
            CommandList->SetPipelineState(SpectralResponsePipeline.Get());
            CommandList->SetComputeRootUnorderedAccessView(ResponseOutputRoot,
                                                           response.Resource->GetGPUVirtualAddress());
            CommandList->Dispatch((ResponseEntries + FitGroupSize - 1) / FitGroupSize, 1, 1);
            Transition(response, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            if (!ExecuteCommands(error))
                return false;

            // Each fence bounds latency and cancellation. Publication requires
            // status readback for every entry, not merely a completed dispatch.
            for (uint32_t first = 0; first < count;)
            {
                if (cancellation())
                {
                    error = "Compute spectral fitting cancelled.";
                    return false;
                }
                const uint32_t batch = std::min(count - first, FitBatchSize);
                const uint32_t controls[]{batch, 0};
                if (!BeginCommands(error))
                    return false;
                Transition(coefficients, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                CommandList->SetComputeRootSignature(SpectralRoot.Get());
                CommandList->SetPipelineState(SpectralPalettePipeline.Get());
                CommandList->SetComputeRootShaderResourceView(SourceRoot,
                                                              input.Resource->GetGPUVirtualAddress() +
                                                                  uint64_t(first) * sizeof(S::Request));
                CommandList->SetComputeRootShaderResourceView(ResponseRoot,
                                                              response.Resource->GetGPUVirtualAddress());
                CommandList->SetComputeRootUnorderedAccessView(CoefficientOutputRoot,
                                                               coefficients.Resource->GetGPUVirtualAddress() +
                                                                   uint64_t(first) * sizeof(S::GpuSpectrum));
                CommandList->SetComputeRoot32BitConstants(ControlsRoot, uint32_t(std::size(controls)),
                                                          controls, 0);
                CommandList->Dispatch((batch + FitGroupSize - 1) / FitGroupSize, 1, 1);
                Transition(coefficients, D3D12_RESOURCE_STATE_COPY_SOURCE);
                const size_t bytes = size_t(batch) * sizeof(S::GpuSpectrum);
                CommandList->CopyBufferRegion(readback.Resource.Get(), 0, coefficients.Resource.Get(),
                                              uint64_t(first) * sizeof(S::GpuSpectrum), bytes);
                if (!ExecuteCommands(error))
                    return false;
                D3D12_RANGE range{0, bytes};
                const S::GpuSpectrum* values = nullptr;
                void* mapped = nullptr;
                if (!Check(readback.Resource->Map(0, &range, &mapped), "Mapping compute spectral fit status",
                           error))
                    return false;
                values = static_cast<const S::GpuSpectrum*>(mapped);
                uint32_t failed = batch;
                for (uint32_t index = 0; index < batch; ++index)
                {
                    const auto& fit = values[index];
                    if (fit.Status != FitConverged || !std::isfinite(fit.Quadratic) ||
                        !std::isfinite(fit.Linear) || !std::isfinite(fit.Offset) ||
                        !std::isfinite(fit.Scale) || !std::isfinite(fit.Constant) ||
                        !std::isfinite(fit.Error) || fit.Scale < 0 || fit.Constant < 0 || fit.Error < 0)
                    {
                        failed = index;
                        break;
                    }
                }
                D3D12_RANGE noWrite{0, 0};
                readback.Resource->Unmap(0, &noWrite);
                if (failed != batch)
                {
                    error = "Compute spectral fit failed at palette entry " + std::to_string(first + failed);
                    return false;
                }
                first += batch;
            }
            if (cancellation())
            {
                error = "Compute spectral publication cancelled.";
                return false;
            }
            if (!BeginCommands(error))
                return false;
            Transition(coefficients, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            if (!ExecuteCommands(error))
                return false;
            output = std::move(coefficients);
            return true;
        }

        bool UpdateSpectralMedium(const MediumOptions& medium, const Detail::CancellationProbe& cancellation,
                                  std::string& error) const
        {
            if (medium == SpectralMedium)
                return true;
            namespace S = SpectralSceneData;
            const S::Request sources[]{{medium.Enabled ? medium.Absorption : Float3(), S::Role::Nonnegative},
                                       {medium.Enabled ? medium.Scattering : Float3(), S::Role::Nonnegative}};
            GpuBuffer candidate;
            if (!PrepareSpectralCoefficients(sources, uint32_t(std::size(sources)), cancellation, candidate,
                                             error))
                return false;
            if (cancellation())
            {
                error = "Compute spectral medium update cancelled.";
                return false;
            }
            if (!BeginCommands(error))
                return false;
            Transition(candidate, D3D12_RESOURCE_STATE_COPY_SOURCE);
            Transition(SpectralPalette, D3D12_RESOURCE_STATE_COPY_DEST);
            CommandList->CopyBufferRegion(SpectralPalette.Resource.Get(),
                                          uint64_t(SpectralSceneSourceCount) * sizeof(S::GpuSpectrum),
                                          candidate.Resource.Get(), 0, candidate.Size);
            Transition(SpectralPalette, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            if (!ExecuteCommands(error))
                return false;
            SpectralMedium = medium;
            return true;
        }

        bool EnsureScene(const Scene& scene, const std::shared_ptr<const Scene>& owner,
                         AccelerationOptions acceleration, const Detail::CancellationProbe& cancellation,
                         std::string& error, const MediumOptions* spectralMedium = nullptr) const
        {
            if (OwnerMatches(owner, acceleration) && (!spectralMedium || SpectralPalette.Resource))
                return !spectralMedium || UpdateSpectralMedium(*spectralMedium, cancellation, error);
            PackedScene packed;
            ScenePacker packer(cancellation);
            if (!packer.Pack(scene, acceleration, packed, error)) return false;
            SpectralSceneData::GpuScene spectral;
            GpuBuffer palette, spectralReferences;
            if (spectralMedium)
            {
                if (ComputeSpectralHost::BuildGpuInputs(scene, *spectralMedium, cancellation, spectral,
                                                        error) != RenderStatus::Completed)
                    return false;
                for (auto& triangle : packed.Triangles)
                {
                    const auto& a = triangle.Color0;
                    const auto& b = triangle.Color1;
                    const auto& c = triangle.Color2;
                    const auto location =
                        spectral.TriangleReferences.find({a.X, a.Y, a.Z, b.X, b.Y, b.Z, c.X, c.Y, c.Z});
                    if (location == spectral.TriangleReferences.end())
                    {
                        error = "Compute packed vertex colors do not match the spectral scene.";
                        return false;
                    }
                    triangle.Padding0 = location->second;
                }
                if (!PrepareSpectralCoefficients(spectral.Sources.data(), uint32_t(spectral.Sources.size()),
                                                 cancellation, palette, error))
                    return false;
                ComPtr<ID3D12Resource> spectralStaging;
                if (!PrepareUpload(spectralReferences, spectral.References.data(),
                                   spectral.References.size() * sizeof(uint32_t), spectralStaging, error) ||
                    !BeginCommands(error))
                    return false;
                CommandList->CopyBufferRegion(spectralReferences.Resource.Get(), 0, spectralStaging.Get(), 0,
                                              spectralReferences.Size);
                Transition(spectralReferences, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                if (!ExecuteCommands(error))
                    return false;
            }
            if (cancellation())
            {
                error = "Compute scene construction was cancelled.";
                return false;
            }
            size_t nodeBytes = packed.Nodes.size() * sizeof(PackedBvhNode);
            const void* nodeData = packed.Nodes.data();
            uint32_t packedNodeCount = static_cast<uint32_t>(packed.Nodes.size());
            if (acceleration.Type == AccelerationType::KdTree)
            {
                nodeBytes = packed.KdNodes.size();
                nodeData = packed.KdNodes.data();
                packedNodeCount = packed.KdNodeCount;
            }
            else if (acceleration.Type == AccelerationType::Bih)
            {
                nodeBytes = packed.BihNodes.size();
                nodeData = packed.BihNodes.data();
                packedNodeCount = packed.BihNodeCount;
            }
            if (!packed.InstanceNodes.empty())
            {
                nodeBytes = packed.ObjectNodes.size();
                nodeData = packed.ObjectNodes.data();
                packedNodeCount = 0; // Individual BLAS counts live in instance records.
            }
            // Every native traversal uses 32-bit ByteAddressBuffer byte offsets.
            if (nodeBytes > MaximumRawNodeBufferBytes)
            {
                error = "Compute acceleration nodes exceed the 4 GiB raw-buffer addressing limit.";
                return false;
            }
            const std::uint64_t requiredBytes =
                nodeBytes + packed.Instances.size() * sizeof(ComputeInstances::Instance)
                + packed.InstanceNodes.size() * sizeof(ComputeInstances::Node) + sizeof(uint32_t)
                + packed.PrimitiveRefs.size() * sizeof(PackedPrimitiveRef)
                + packed.Triangles.size() * sizeof(PackedTriangle)
                + packed.Spheres.size() * sizeof(PackedSphere)
                + packed.Materials.size() * sizeof(PackedMaterial)
                + packed.Lights.size() * sizeof(PackedLight)
                + packed.TextureAtlas.Descriptors.size()
                    * sizeof(Detail::PackedRayTextureDescriptor)
                + packed.TextureAtlas.Texels.size() * sizeof(std::uint32_t);
            if (!CheckVideoMemory(requiredBytes, "Compute scene buffers", error)) return false;

            ReleaseSceneBuffers();
            std::array<ComPtr<ID3D12Resource>, 10> staging;
            if (!PrepareUpload(Nodes, nodeData, nodeBytes, staging[0], error)
                || !PrepareUpload(PrimitiveRefs, packed.PrimitiveRefs.data(),
                    packed.PrimitiveRefs.size() * sizeof(PackedPrimitiveRef), staging[1], error)
                || !PrepareUpload(Triangles, packed.Triangles.data(),
                    packed.Triangles.size() * sizeof(PackedTriangle), staging[2], error)
                || !PrepareUpload(Spheres, packed.Spheres.data(),
                    packed.Spheres.size() * sizeof(PackedSphere), staging[3], error)
                || !PrepareUpload(Materials, packed.Materials.data(),
                    packed.Materials.size() * sizeof(PackedMaterial), staging[4], error)
                || !PrepareUpload(Lights, packed.Lights.data(),
                    packed.Lights.size() * sizeof(PackedLight), staging[5], error)
                || !PrepareUpload(
                    TextureDescriptors,
                    packed.TextureAtlas.Descriptors.data(),
                    packed.TextureAtlas.Descriptors.size()
                        * sizeof(Detail::PackedRayTextureDescriptor),
                    staging[6],
                    error)
                || !PrepareUpload(
                    TextureTexels,
                    packed.TextureAtlas.Texels.data(),
                    packed.TextureAtlas.Texels.size() * sizeof(std::uint32_t),
                    staging[7],
                    error)
                || !PrepareUpload(Instances, packed.Instances.data(),
                    packed.Instances.size() * sizeof(ComputeInstances::Instance), staging[8], error)
                || !PrepareUpload(InstanceNodes, packed.InstanceNodes.data(),
                    packed.InstanceNodes.size() * sizeof(ComputeInstances::Node), staging[9], error)
                || !CreateBuffer(InstanceError, sizeof(uint32_t), D3D12_HEAP_TYPE_DEFAULT,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, error)
                || !CreateBuffer(InstanceErrorReadback, sizeof(uint32_t), D3D12_HEAP_TYPE_READBACK,
                    D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE, error))
            {
                ReleaseSceneBuffers();
                return false;
            }
            if (cancellation())
            {
                ReleaseSceneBuffers();
                error = "Compute scene upload was cancelled.";
                return false;
            }
            if (!BeginCommands(error))
            {
                ReleaseSceneBuffers();
                return false;
            }
            std::array<GpuBuffer*, 10> destinations = {
                &Nodes,
                &PrimitiveRefs,
                &Triangles,
                &Spheres,
                &Materials,
                &Lights,
                &TextureDescriptors,
                &TextureTexels,
                &Instances,
                &InstanceNodes};
            for (std::size_t index = 0; index < destinations.size(); ++index)
            {
                if (!destinations[index]->Resource) continue;
                CommandList->CopyBufferRegion(
                    destinations[index]->Resource.Get(), 0, staging[index].Get(), 0,
                    destinations[index]->Size);
                Transition(*destinations[index], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            }
            if (!ExecuteCommands(error))
            {
                ReleaseSceneBuffers();
                return false;
            }
            InstanceCount = uint32_t(packed.Instances.size());
            InstanceNodeCount = uint32_t(packed.InstanceNodes.size());
            NodeCount = packedNodeCount;
            PrimitiveRefCount = static_cast<std::uint32_t>(packed.PrimitiveRefs.size());
            TriangleCount = static_cast<std::uint32_t>(packed.Triangles.size());
            SphereCount = static_cast<std::uint32_t>(packed.Spheres.size());
            MaterialCount = static_cast<std::uint32_t>(packed.Materials.size());
            LightCount = static_cast<std::uint32_t>(packed.Lights.size());
            TextureCount = static_cast<std::uint32_t>(
                packed.TextureAtlas.Descriptors.size());
            TextureTexelCount = static_cast<std::uint32_t>(
                packed.TextureAtlas.Texels.size());
            CachedOwner = owner;
            CachedAcceleration = acceleration;
            SpectralPalette = std::move(palette);
            SpectralReferences = std::move(spectralReferences);
            SpectralSceneSourceCount = spectral.SceneSourceCount;
            if (spectralMedium)
                SpectralMedium = *spectralMedium;
            RefreshSceneDescriptors();
            return true;
        }

        bool EnsureFrameBuffers(std::uint32_t pixelCount, std::string& error) const
        {
            if (FramePixelCapacity >= pixelCount) return true;
            const std::uint64_t defaultBytes =
                static_cast<std::uint64_t>(pixelCount)
                    * (sizeof(Vec4) + sizeof(std::uint32_t) * 2)
                + uint64_t(pixelCount) * sizeof(uint32_t) + sizeof(GroupStatistics);
            if (!CheckVideoMemory(defaultBytes, "Compute frame buffers", error)) return false;
            ReleaseFrameBuffers();
            if (!CreateBuffer(Accumulation,
                    static_cast<std::uint64_t>(pixelCount) * sizeof(Vec4),
                    D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, error)
                || !CreateBuffer(SampleCounts,
                    static_cast<std::uint64_t>(pixelCount) * sizeof(std::uint32_t),
                    D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, error)
                || !CreateBuffer(Output,
                    static_cast<std::uint64_t>(pixelCount) * sizeof(std::uint32_t),
                    D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, error)
                || !CreateBuffer(SampleTargets,
                    static_cast<std::uint64_t>(pixelCount) * sizeof(std::uint32_t),
                    D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, error)
                || !CreateBuffer(Statistics, sizeof(GroupStatistics),
                    D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, error)
                || !CreateBuffer(OutputReadback,
                    static_cast<std::uint64_t>(pixelCount) * sizeof(std::uint32_t),
                    D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_FLAG_NONE, error)
                || !CreateBuffer(SamplesReadback,
                    static_cast<std::uint64_t>(pixelCount) * sizeof(std::uint32_t),
                    D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_FLAG_NONE, error)
                || !CreateBuffer(FilmReadback,
                    static_cast<std::uint64_t>(pixelCount) * sizeof(Vec4),
                    D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_FLAG_NONE, error)
                || !CreateBuffer(TargetsReadback,
                    static_cast<std::uint64_t>(pixelCount) * sizeof(std::uint32_t),
                    D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_FLAG_NONE, error)
                || !CreateBuffer(StatisticsReadback, sizeof(GroupStatistics),
                    D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_FLAG_NONE, error))
            {
                ReleaseFrameBuffers();
                return false;
            }
            FramePixelCapacity = pixelCount;
            RefreshFrameDescriptors(pixelCount);
            return true;
        }

        bool ClearFrame(
            FrameConstants constants,
            std::uint32_t pixelCount,
            const Detail::CancellationProbe& cancellation,
            std::string& error) const
        {
            if (!BeginCommands(error)) return false;
            for (std::uint32_t begin = 0; begin < pixelCount; begin += PixelChunkSize)
            {
                if (cancellation())
                {
                    CommandList->Close();
                    error = "Compute frame clear was cancelled.";
                    return false;
                }
                constants.PixelBegin = begin;
                constants.PixelCount = std::min(PixelChunkSize, pixelCount - begin);
                if (!BindDispatch(ClearPixelsPipeline.Get(), constants, error)) return false;
                CommandList->Dispatch(
                    (constants.PixelCount + ThreadsPerGroup - 1) / ThreadsPerGroup, 1, 1);
            }
            UavBarrier(Accumulation);
            UavBarrier(SampleCounts);
            UavBarrier(Output);
            UavBarrier(SampleTargets);
            return ExecuteCommands(error);
        }

        bool ReadStatistics(GroupStatistics& statistics, std::string& error) const
        {
            void* mapped = nullptr;
            D3D12_RANGE readRange{0, sizeof(GroupStatistics)};
            if (!Check(StatisticsReadback.Resource->Map(
                    0, &readRange, &mapped),
                "ID3D12Resource::Map(statistics readback)", error))
            {
                return false;
            }
            std::memcpy(&statistics, mapped, sizeof(statistics));
            D3D12_RANGE noWrite{0, 0};
            StatisticsReadback.Resource->Unmap(0, &noWrite);
            return true;
        }

        void ClearInstanceError() const
        {
            if (!InstanceNodeCount) return;
            auto gpu = DescriptorHeap->GetGPUDescriptorHandleForHeapStart();
            const auto cpu = ErrorClearHeap->GetCPUDescriptorHandleForHeapStart();
            gpu.ptr += uint64_t(InstanceErrorDescriptor) * DescriptorSize;
            const UINT zeros[4]{};
            CommandList->ClearUnorderedAccessViewUint(gpu, cpu, InstanceError.Resource.Get(), zeros, 0, nullptr);
            UavBarrier(InstanceError);
        }

        void CopyInstanceError() const
        {
            if (!InstanceNodeCount) return;
            UavBarrier(InstanceError);
            Transition(InstanceError, D3D12_RESOURCE_STATE_COPY_SOURCE);
            CommandList->CopyBufferRegion(InstanceErrorReadback.Resource.Get(), 0,
                InstanceError.Resource.Get(), 0, sizeof(uint32_t));
            Transition(InstanceError, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }

        bool CheckInstanceError(std::string& error) const
        {
            if (!InstanceNodeCount) return true;
            void* mapped = nullptr;
            const D3D12_RANGE read{0, sizeof(uint32_t)}, noWrite{0, 0};
            if (!Check(InstanceErrorReadback.Resource->Map(0, &read, &mapped),
                "Map(instance error readback)", error)) return false;
            uint32_t flags;
            std::memcpy(&flags, mapped, sizeof(flags));
            InstanceErrorReadback.Resource->Unmap(0, &noWrite);
            if (!flags) return true;
            error = "Compute instance ray cannot be represented in object-space float coordinates.";
            return false;
        }

        bool DispatchTrace(
            const FrameConstants& constants,
            GroupStatistics& statistics,
            std::string& error) const
        {
            if (DiagnosticInfo)
                std::fprintf(stderr, "Compute trace: nodes=%u instances=%u pixels=%u samples=%u depth=%u medium=%u exposure=%u sceneMotion=%u\n",
                    constants.NodeCount, constants.InstanceNodeCount, constants.PixelCount, constants.SampleCount,
                    constants.MaxReflectionBounces, CurrentMedium.Medium.Enabled,
                    CurrentMedium.CameraExposure.Enabled, CurrentMedium.CameraExposure.SceneEnabled);
            if (!BeginCommands(error)) return false;
            ClearInstanceError();
            if (!BindDispatch(ClearStatisticsPipeline.Get(), constants, error)) return false;
            CommandList->Dispatch(1, 1, 1);
            UavBarrier(Statistics);
            ID3D12PipelineState* pipeline = nullptr;
            switch (static_cast<Algorithm>(constants.Integrator))
            {
            case Algorithm::RayTracing: pipeline = WhittedPipeline.Get(); break;
            case Algorithm::PathTracing: pipeline = PathTracingPipeline.Get(); break;
            case Algorithm::DistributedRayTracing: pipeline = DistributedRayPipeline.Get(); break;
            default: error = "No native compute pipeline for the integrator"; return false;
            }
            if (!BindDispatch(pipeline, constants, error)) return false;
            CommandList->Dispatch(constants.GroupCount, 1, 1);
            UavBarrier(Accumulation);
            UavBarrier(SampleCounts);
            UavBarrier(Statistics);
            Transition(Statistics, D3D12_RESOURCE_STATE_COPY_SOURCE);
            UavBarrier(SampleTargets);
            CommandList->CopyBufferRegion(
                StatisticsReadback.Resource.Get(), 0, Statistics.Resource.Get(), 0,
                sizeof(GroupStatistics));
            Transition(Statistics, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            CopyInstanceError();
            if (!ExecuteCommands(error)) return false;
            return ReadStatistics(statistics, error) && CheckInstanceError(error);
        }

        bool ReadSampleTargets(std::uint32_t pixelCount,
            std::vector<std::uint32_t>& targets, std::string& error) const
        {
            if (!BeginCommands(error)) return false;
            const size_t byteCount = static_cast<size_t>(pixelCount) * sizeof(std::uint32_t);
            UavBarrier(SampleTargets);
            Transition(SampleTargets, D3D12_RESOURCE_STATE_COPY_SOURCE);
            CommandList->CopyBufferRegion(
                TargetsReadback.Resource.Get(), 0, SampleTargets.Resource.Get(), 0, byteCount);
            Transition(SampleTargets, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            if (!ExecuteCommands(error)) return false;
            void* mapped = nullptr;
            const D3D12_RANGE readRange{0, byteCount};
            if (!Check(TargetsReadback.Resource->Map(0, &readRange, &mapped),
                    "ID3D12Resource::Map(adaptive targets)", error)) return false;
            targets.resize(pixelCount);
            std::memcpy(targets.data(), mapped, byteCount);
            const D3D12_RANGE noWrite{0, 0};
            TargetsReadback.Resource->Unmap(0, &noWrite);
            return true;
        }

        bool ResolveAndReadback(
            FrameConstants constants,
            std::uint32_t pixelCount,
            const Detail::CancellationProbe& cancellation,
            std::vector<std::uint32_t>& packedOutput,
            std::vector<std::uint32_t>& completedSamples,
            std::vector<FilmPixel>& film,
            std::string& error) const
        {
            if (!BeginCommands(error)) return false;
            UavBarrier(Accumulation);
            UavBarrier(SampleCounts);
            for (std::uint32_t begin = 0; begin < pixelCount; begin += PixelChunkSize)
            {
                if (cancellation())
                {
                    CommandList->Close();
                    error = "Compute resolve was cancelled.";
                    return false;
                }
                constants.PixelBegin = begin;
                constants.PixelCount = std::min(PixelChunkSize, pixelCount - begin);
                if (!BindDispatch(ResolvePixelsPipeline.Get(), constants, error)) return false;
                CommandList->Dispatch(
                    (constants.PixelCount + ThreadsPerGroup - 1) / ThreadsPerGroup, 1, 1);
            }
            UavBarrier(Output);
            Transition(Output, D3D12_RESOURCE_STATE_COPY_SOURCE);
            Transition(SampleCounts, D3D12_RESOURCE_STATE_COPY_SOURCE);
            Transition(Accumulation, D3D12_RESOURCE_STATE_COPY_SOURCE);
            CommandList->CopyBufferRegion(
                FilmReadback.Resource.Get(), 0, Accumulation.Resource.Get(), 0,
                static_cast<std::uint64_t>(pixelCount) * sizeof(Vec4));
            CommandList->CopyBufferRegion(
                OutputReadback.Resource.Get(), 0, Output.Resource.Get(), 0,
                static_cast<std::uint64_t>(pixelCount) * sizeof(std::uint32_t));
            CommandList->CopyBufferRegion(
                SamplesReadback.Resource.Get(), 0, SampleCounts.Resource.Get(), 0,
                static_cast<std::uint64_t>(pixelCount) * sizeof(std::uint32_t));
            Transition(Output, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            Transition(SampleCounts, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            Transition(Accumulation, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            if (!ExecuteCommands(error)) return false;
            if (cancellation())
            {
                error = "Compute readback was cancelled.";
                return false;
            }

            packedOutput.resize(pixelCount);
            completedSamples.resize(pixelCount);
            const std::size_t byteCount =
                static_cast<std::size_t>(pixelCount) * sizeof(std::uint32_t);
            for (std::size_t offset = 0; offset < byteCount; offset += ReadbackChunkSize)
            {
                if (cancellation())
                {
                    error = "Compute readback was cancelled.";
                    return false;
                }
                const std::size_t chunk = std::min(ReadbackChunkSize, byteCount - offset);
                D3D12_RANGE readRange{offset, offset + chunk};
                void* outputMapped = nullptr;
                void* samplesMapped = nullptr;
                void* filmMapped = nullptr;
                const size_t firstPixel = offset / sizeof(std::uint32_t);
                const size_t chunkPixels = chunk / sizeof(std::uint32_t);
                const D3D12_RANGE filmRange{firstPixel * sizeof(Vec4), (firstPixel + chunkPixels) * sizeof(Vec4)};
                if (!Check(OutputReadback.Resource->Map(0, &readRange, &outputMapped),
                        "ID3D12Resource::Map(output readback)", error)
                    || !Check(SamplesReadback.Resource->Map(0, &readRange, &samplesMapped),
                        "ID3D12Resource::Map(sample readback)", error)
                    || !Check(FilmReadback.Resource->Map(0, &filmRange, &filmMapped),
                        "ID3D12Resource::Map(linear film readback)", error))
                {
                    if (outputMapped)
                    {
                        D3D12_RANGE noWrite{0, 0};
                        OutputReadback.Resource->Unmap(0, &noWrite);
                    }
                    if (samplesMapped)
                    {
                        const D3D12_RANGE noWrite{0, 0};
                        SamplesReadback.Resource->Unmap(0, &noWrite);
                    }
                    return false;
                }
                std::memcpy(
                    reinterpret_cast<std::uint8_t*>(packedOutput.data()) + offset,
                    static_cast<const std::uint8_t*>(outputMapped) + offset, chunk);
                std::memcpy(
                    reinterpret_cast<std::uint8_t*>(completedSamples.data()) + offset,
                    static_cast<const std::uint8_t*>(samplesMapped) + offset, chunk);
                const auto* moments = static_cast<const Vec4*>(filmMapped);
                for (size_t pixel = firstPixel; pixel < firstPixel + chunkPixels; ++pixel)
                {
                    const Vec4& value = moments[pixel];
                    film[pixel] = {Float3(value.X, value.Y, value.Z), value.W, completedSamples[pixel]};
                }
                D3D12_RANGE noWrite{0, 0};
                OutputReadback.Resource->Unmap(0, &noWrite);
                SamplesReadback.Resource->Unmap(0, &noWrite);
                FilmReadback.Resource->Unmap(0, &noWrite);
            }
            return true;
        }

        RayQueryResult QueryNative(const Scene& scene, const std::vector<Ray>& rays,
            AccelerationOptions acceleration, const Detail::CancellationProbe& cancellation) const
        {
            RayQueryResult result;
            if (cancellation()) { result.Status = RenderStatus::Cancelled; return result; }
            if (!ValidateRayQueries(rays, result.Error) || !ValidateAcceleration(acceleration, result.Error)) return result;
            RenderSettings validationSettings;
            validationSettings.Acceleration = acceleration.Type;
            validationSettings.BvhPolicy = acceleration.BvhPolicy;
            validationSettings.Width = validationSettings.Height = 1;
            if (!ValidateInput(scene, Camera{}, validationSettings, result.Error, false)) return result;
            if (rays.empty()) { result.Status = RenderStatus::Completed; return result; }
            if (!CreatePipeline(Detail::ComputeQueryRaysDxil, sizeof(Detail::ComputeQueryRaysDxil),
                QueryRaysPipeline, "CreateComputePipelineState(QueryRays)", result.Error)) return result;
            if (cancellation()) { result.Status = RenderStatus::Cancelled; return result; }
            if (!EnsureScene(scene, nullptr, acceleration, cancellation, result.Error))
            {
                if (cancellation()) { result.Status = RenderStatus::Cancelled; result.Error.clear(); }
                return result;
            }

            const auto count = static_cast<std::uint32_t>(rays.size());
            const size_t inputBytes = rays.size() * sizeof(Ray);
            const size_t outputBytes = rays.size() * sizeof(RayQueryHit);
            if (!CheckVideoMemory(inputBytes + outputBytes, "Compute ray-query buffers", result.Error)) return result;
            GpuBuffer input, output, readback;
            ComPtr<ID3D12Resource> staging;
            if (!PrepareUpload(input, rays.data(), inputBytes, staging, result.Error)
                || !CreateBuffer(output, outputBytes, D3D12_HEAP_TYPE_DEFAULT,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, result.Error)
                || !CreateBuffer(readback, outputBytes, D3D12_HEAP_TYPE_READBACK,
                    D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE, result.Error)) return result;

            if (!BeginCommands(result.Error)) return result;
            CommandList->CopyBufferRegion(input.Resource.Get(), 0, staging.Get(), 0, inputBytes);
            Transition(input, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            if (!ExecuteCommands(result.Error)) return result;

            FrameConstants constants;
            constants.InstanceNodeCount = InstanceNodeCount;
            constants.Acceleration = static_cast<uint32_t>(acceleration.Type);
            constants.NodeCount = NodeCount;
            constants.PrimitiveRefCount = PrimitiveRefCount;
            constants.TriangleCount = TriangleCount;
            constants.SphereCount = SphereCount;
            for (std::uint32_t begin = 0; begin < count; begin += PixelChunkSize)
            {
                if (cancellation()) { result.Status = RenderStatus::Cancelled; return result; }
                constants.PixelBegin = begin;
                constants.PixelCount = std::min(PixelChunkSize, count - begin);
                if (!BeginCommands(result.Error)) return result;
                ClearInstanceError();
                CommandList->SetComputeRootShaderResourceView(3, input.Resource->GetGPUVirtualAddress());
                CommandList->SetComputeRootUnorderedAccessView(4, output.Resource->GetGPUVirtualAddress());
                if (!BindDispatch(QueryRaysPipeline.Get(), constants, result.Error)) return result;
                CommandList->Dispatch((constants.PixelCount + ThreadsPerGroup - 1) / ThreadsPerGroup, 1, 1);
                UavBarrier(output);
                CopyInstanceError();
                if (!ExecuteCommands(result.Error) || !CheckInstanceError(result.Error)) return result;
            }
            if (cancellation()) { result.Status = RenderStatus::Cancelled; return result; }
            if (!BeginCommands(result.Error)) return result;
            Transition(output, D3D12_RESOURCE_STATE_COPY_SOURCE);
            CommandList->CopyBufferRegion(readback.Resource.Get(), 0, output.Resource.Get(), 0, outputBytes);
            if (!ExecuteCommands(result.Error)) return result;
            if (cancellation()) { result.Status = RenderStatus::Cancelled; return result; }

            result.Hits.resize(rays.size());
            void* mapped = nullptr;
            const D3D12_RANGE readRange{0, outputBytes};
            if (!Check(readback.Resource->Map(0, &readRange, &mapped),
                    "ID3D12Resource::Map(ray-query hits)", result.Error)) return result;
            std::memcpy(result.Hits.data(), mapped, outputBytes);
            const D3D12_RANGE noWrite{0, 0};
            readback.Resource->Unmap(0, &noWrite);
            result.Status = RenderStatus::Completed;
            return result;
        }

        bool EnsureBidirectionalPipelines(std::string& error) const
        {
            if (!BidirectionalRoot &&
                !Check(Device->CreateRootSignature(0, Detail::ComputeBidirectionalRootSignature,
                                                   sizeof(Detail::ComputeBidirectionalRootSignature),
                                                   IID_PPV_ARGS(&BidirectionalRoot)),
                       "CreateRootSignature(BDPT)", error))
                return false;
            return CreatePipeline(Detail::ComputeBidirectionalPairsDxil,
                                  sizeof(Detail::ComputeBidirectionalPairsDxil), BidirectionalPairs,
                                  "CreateComputePipelineState(BDPT pairs)", error, BidirectionalRoot.Get()) &&
                   CreatePipeline(Detail::ComputeBidirectionalClearDxil,
                                  sizeof(Detail::ComputeBidirectionalClearDxil), BidirectionalClear,
                                  "CreateComputePipelineState(BDPT clear)", error, BidirectionalRoot.Get()) &&
                   CreatePipeline(Detail::ComputeBidirectionalCommitDxil,
                                  sizeof(Detail::ComputeBidirectionalCommitDxil), BidirectionalCommit,
                                  "CreateComputePipelineState(BDPT commit)", error, BidirectionalRoot.Get());
        }

        bool RenderBidirectional(const Scene& scene, FrameConstants constants, const RenderSettings& settings,
                                 const Detail::CancellationProbe& cancellation, RenderResult& result) const
        {
            if (settings.Adaptive.Enabled || settings.Transport.RussianRoulette ||
                !settings.Transport.NextEventEstimation || !settings.ShadowsEnabled ||
                settings.ShadowStrength != 1)
            {
                result.Error = "Compute BDPT requires fixed samples/depth, NEE and full physical visibility.";
                return false;
            }
            const auto emitters = ComputeEmission::PrepareEmitters(scene);
            for (const auto& emitter : emitters.Sources)
                if (!(emitter.Flags & EmitterCastsShadows))
                {
                    result.Error = "Compute BDPT requires shadow casting on every emitter.";
                    return false;
                }
            const uint32_t fullPixels = constants.FullWidth * constants.FullHeight;
            const uint32_t capacity = std::min(ComputeBdpt::BatchCapacity, fullPixels);
            const uint32_t pixels = constants.Width * constants.Height;
            const uint32_t vertexCount = capacity * ComputeBdpt::EndpointCount * ComputeBdpt::MaximumVertices;
            const uint64_t statisticsBytes = uint64_t(capacity) * sizeof(GroupStatistics);
            const uint64_t readbackBytes = statisticsBytes + sizeof(uint32_t);
            const uint64_t passBytes = uint64_t(pixels) * ComputeBdpt::PassPixelBytes;
            const uint64_t vertexBytes = uint64_t(vertexCount) * sizeof(ComputeBdpt::Vertex);
            if (!CheckVideoMemory(passBytes + vertexBytes + statisticsBytes +
                                      emitters.Sources.size() * sizeof(PackedEmitter),
                                  "Compute BDPT buffers", result.Error))
                return false;
            GpuBuffer sources, vertices, pass, statistics, error, readback, frame;
            ComPtr<ID3D12Resource> staging;
            if (!PrepareUpload(sources, emitters.Sources.data(),
                               emitters.Sources.size() * sizeof(PackedEmitter), staging, result.Error) ||
                !CreateBuffer(vertices, vertexBytes, D3D12_HEAP_TYPE_DEFAULT,
                              D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                              D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, result.Error) ||
                !CreateBuffer(pass, passBytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                              D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, result.Error) ||
                !CreateBuffer(statistics, statisticsBytes, D3D12_HEAP_TYPE_DEFAULT,
                              D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                              D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, result.Error) ||
                !CreateBuffer(error, sizeof(uint32_t), D3D12_HEAP_TYPE_DEFAULT,
                              D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                              D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, result.Error) ||
                !CreateBuffer(readback, readbackBytes, D3D12_HEAP_TYPE_READBACK,
                              D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE, result.Error) ||
                !CreateBuffer(frame, D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT, D3D12_HEAP_TYPE_UPLOAD,
                              D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_FLAG_NONE, result.Error))
                return false;
            if (sources.Resource)
            {
                if (!BeginCommands(result.Error))
                    return false;
                CommandList->CopyBufferRegion(sources.Resource.Get(), 0, staging.Get(), 0, sources.Size);
                Transition(sources, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                if (!ExecuteCommands(result.Error))
                    return false;
                staging.Reset();
            }
            ComPtr<ID3D12DescriptorHeap> heap;
            D3D12_DESCRIPTOR_HEAP_DESC description{};
            description.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
            description.NumDescriptors = ComputeBdpt::DescriptorCount;
            description.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
            if (!Check(Device->CreateDescriptorHeap(&description, IID_PPV_ARGS(&heap)),
                       "CreateDescriptorHeap(BDPT)", result.Error))
                return false;
            CreateRawSrv(0, Nodes, heap.Get());
            CreateStructuredSrv(1, PrimitiveRefs, PrimitiveRefCount, sizeof(PackedPrimitiveRef), heap.Get());
            CreateStructuredSrv(2, Triangles, TriangleCount, sizeof(PackedTriangle), heap.Get());
            CreateStructuredSrv(3, Spheres, SphereCount, sizeof(PackedSphere), heap.Get());
            CreateStructuredSrv(4, Materials, MaterialCount, sizeof(PackedMaterial), heap.Get());
            CreateStructuredSrv(5, sources, uint32_t(emitters.Sources.size()), sizeof(PackedEmitter),
                                heap.Get());
            CreateStructuredSrv(6, TextureDescriptors, TextureCount,
                                sizeof(Detail::PackedRayTextureDescriptor), heap.Get());
            CreateStructuredSrv(7, TextureTexels, TextureTexelCount, sizeof(uint32_t), heap.Get());
            CreateStructuredSrv(8, Instances, InstanceCount, sizeof(ComputeInstances::Instance), heap.Get());
            CreateStructuredSrv(9, InstanceNodes, InstanceNodeCount, sizeof(ComputeInstances::Node),
                                heap.Get());
            constexpr uint32_t outputBegin = ComputeBdpt::SceneDescriptors;
            CreateStructuredUav(outputBegin, Accumulation, pixels, sizeof(Vec4), heap.Get());
            CreateStructuredUav(outputBegin + 1, SampleCounts, pixels, sizeof(uint32_t), heap.Get());
            CreateRawUav(outputBegin + 2, pass, uint32_t(passBytes / sizeof(uint32_t)), heap.Get());
            CreateStructuredUav(outputBegin + 3, statistics, capacity, sizeof(GroupStatistics), heap.Get());
            CreateStructuredUav(outputBegin + 4, SampleTargets, pixels, sizeof(uint32_t), heap.Get());
            CreateStructuredUav(outputBegin + 5, vertices, vertexCount, sizeof(ComputeBdpt::Vertex),
                                heap.Get());
            CreateRawUav(outputBegin + 6, InstanceError, 1, heap.Get());
            CreateRawUav(outputBegin + 7, error, 1, heap.Get());
            ComputeBdpt::Controls controls{{emitters.Center.X, emitters.Center.Y, emitters.Center.Z},
                                           emitters.Radius,
                                           uint32_t(emitters.Sources.size()),
                                           0,
                                           0,
                                           0};
            auto bind = [&](ID3D12PipelineState* pipeline) {
                ID3D12DescriptorHeap* heaps[] = {heap.Get()};
                CommandList->SetDescriptorHeaps(1, heaps);
                CommandList->SetComputeRootSignature(BidirectionalRoot.Get());
                auto handle = heap->GetGPUDescriptorHandleForHeapStart();
                CommandList->SetComputeRootDescriptorTable(0, handle);
                handle.ptr += uint64_t(outputBegin) * DescriptorSize;
                CommandList->SetComputeRootDescriptorTable(1, handle);
                CommandList->SetComputeRootConstantBufferView(2, frame.Resource->GetGPUVirtualAddress());
                CommandList->SetComputeRoot32BitConstants(3, sizeof(controls) / sizeof(uint32_t), &controls,
                                                          0);
                CommandList->SetPipelineState(pipeline);
            };
            auto copyError = [&] {
                UavBarrier(error);
                Transition(error, D3D12_RESOURCE_STATE_COPY_SOURCE);
                CommandList->CopyBufferRegion(readback.Resource.Get(), statisticsBytes, error.Resource.Get(),
                                              0, sizeof(uint32_t));
                Transition(error, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            };
            auto consume = [&](uint32_t count) {
                void* mapped = nullptr;
                const D3D12_RANGE range{0, size_t(readbackBytes)}, noWrite{0, 0};
                if (!Check(readback.Resource->Map(0, &range, &mapped), "Map(BDPT readback)", result.Error))
                    return false;
                const auto* rows = static_cast<const GroupStatistics*>(mapped);
                for (uint32_t i = 0; i < count; ++i)
                {
                    result.Statistics.PrimaryRays += rows[i].PrimaryRays.Value();
                    result.Statistics.ShadowRays += rows[i].ShadowRays.Value();
                    result.Statistics.ReflectionRays += rows[i].ReflectionRays.Value();
                    result.Statistics.PrimitiveTests += rows[i].PrimitiveTests.Value();
                }
                uint32_t flags;
                std::memcpy(&flags, static_cast<const uint8_t*>(mapped) + statisticsBytes, sizeof(flags));
                readback.Resource->Unmap(0, &noWrite);
                if (flags)
                    result.Error = "Compute BDPT encountered nonfinite path or film arithmetic.";
                return flags == 0;
            };
            const uint32_t filmGroupsX =
                (constants.Width + ComputeBdpt::FilmGroupWidth - 1) / ComputeBdpt::FilmGroupWidth;
            const uint32_t filmGroupsY =
                (constants.Height + ComputeBdpt::FilmGroupWidth - 1) / ComputeBdpt::FilmGroupWidth;
            for (uint32_t sample = 0; sample < constants.TotalSamples; ++sample)
            {
                if (cancellation())
                    return false;
                constants.SampleBegin = sample;
                void* mapped = nullptr;
                const D3D12_RANGE noRead{0, 0}, written{0, sizeof(constants)};
                if (!Check(frame.Resource->Map(0, &noRead, &mapped), "Map(BDPT constants)", result.Error))
                    return false;
                std::memcpy(mapped, &constants, sizeof(constants));
                frame.Resource->Unmap(0, &written);
                if (!BeginCommands(result.Error))
                    return false;
                bind(BidirectionalClear.Get());
                CommandList->Dispatch(filmGroupsX, filmGroupsY, 1);
                UavBarrier(pass);
                UavBarrier(error);
                UavBarrier(InstanceError);
                if (!ExecuteCommands(result.Error))
                    return false;
                for (uint32_t begin = 0; begin < fullPixels; begin += capacity)
                {
                    if (cancellation())
                        return false;
                    controls.PathBegin = begin;
                    controls.PathCount = std::min(capacity, fullPixels - begin);
                    if (!BeginCommands(result.Error))
                        return false;
                    bind(BidirectionalPairs.Get());
                    CommandList->Dispatch((controls.PathCount + ComputeBdpt::ThreadsPerGroup - 1) /
                                              ComputeBdpt::ThreadsPerGroup,
                                          1, 1);
                    UavBarrier(pass);
                    UavBarrier(vertices);
                    UavBarrier(statistics);
                    Transition(statistics, D3D12_RESOURCE_STATE_COPY_SOURCE);
                    CommandList->CopyBufferRegion(readback.Resource.Get(), 0, statistics.Resource.Get(), 0,
                                                  uint64_t(controls.PathCount) * sizeof(GroupStatistics));
                    Transition(statistics, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                    copyError();
                    CopyInstanceError();
                    if (!ExecuteCommands(result.Error) || !consume(controls.PathCount) ||
                        !CheckInstanceError(result.Error))
                        return false;
                }
                // Incomplete passes are discarded on cancellation; correlated
                // subpath connections never masquerade as independent samples.
                if (cancellation())
                    return false;
                if (!BeginCommands(result.Error))
                    return false;
                bind(BidirectionalCommit.Get());
                CommandList->Dispatch(filmGroupsX, filmGroupsY, 1);
                UavBarrier(Accumulation);
                UavBarrier(SampleCounts);
                UavBarrier(SampleTargets);
                copyError();
                if (!ExecuteCommands(result.Error) || !consume(0))
                    return false;
            }
            return true;
        }

        bool EnsurePhotonPipelines(std::string& error) const
        {
            if (!PhotonRoot && !Check(Device->CreateRootSignature(0, Detail::ComputePhotonRootSignature,
                                                                  sizeof(Detail::ComputePhotonRootSignature),
                                                                  IID_PPV_ARGS(&PhotonRoot)),
                                      "CreateRootSignature(photon)", error))
                return false;
            return CreatePipeline(Detail::ComputePhotonClearDxil, sizeof(Detail::ComputePhotonClearDxil),
                                  PhotonClear, "CreateComputePipelineState(photon clear)", error,
                                  PhotonRoot.Get()) &&
                   CreatePipeline(Detail::ComputePhotonEmitDxil, sizeof(Detail::ComputePhotonEmitDxil),
                                  PhotonEmit, "CreateComputePipelineState(photon emission)", error,
                                  PhotonRoot.Get()) &&
                   CreatePipeline(Detail::ComputePhotonIndexDxil, sizeof(Detail::ComputePhotonIndexDxil),
                                  PhotonIndex, "CreateComputePipelineState(photon index)", error,
                                  PhotonRoot.Get()) &&
                   CreatePipeline(Detail::ComputePhotonCameraDxil, sizeof(Detail::ComputePhotonCameraDxil),
                                  PhotonCamera, "CreateComputePipelineState(photon camera)", error,
                                  PhotonRoot.Get());
        }

        bool RenderPhotons(const Scene& scene, const FrameConstants& constants,
                           const RenderSettings& settings, const Detail::CancellationProbe& cancellation,
                           RenderResult& result) const
        {
            if (settings.MaxReflectionBounces > ComputePhoton::MaximumReflections ||
                settings.Adaptive.Enabled || settings.Transport.Mis != MisHeuristic::None ||
                !settings.ShadowsEnabled || settings.ShadowStrength != 1)
            {
                result.Error = "Compute photon mapping requires bounded depth, fixed budgets, no MIS and "
                               "full visibility.";
                return false;
            }
            const auto emitters = ComputeEmission::PrepareEmitters(scene);
            for (const auto& emitter : emitters.Sources)
                if (!(emitter.Flags & EmitterCastsShadows))
                {
                    result.Error = "Compute photon mapping requires every emitter to cast shadows.";
                    return false;
                }
            const auto& options = settings.Photons;
            const uint32_t pixels = constants.Width * constants.Height;
            const uint32_t slotsPerPath = settings.MaxReflectionBounces + 1;
            const uint32_t slots = options.PhotonPaths * slotsPerPath;
            uint32_t buckets = 1;
            while (buckets < options.PhotonPaths)
                buckets *= 2;
            const uint64_t statisticsBytes =
                uint64_t(ComputePhoton::TransportBatch) * sizeof(GroupStatistics);
            const uint64_t readbackBytes = statisticsBytes + sizeof(uint32_t);
            GpuBuffer photons, heads, statistics, error, sources, frame, readback;
            struct Allocation
            {
                GpuBuffer* Buffer;
                uint64_t Bytes;
            };
            const Allocation storage[] = {{&photons, uint64_t(slots) * sizeof(ComputePhoton::Photon)},
                                          {&heads, uint64_t(buckets) * sizeof(uint32_t)},
                                          {&statistics, statisticsBytes},
                                          {&error, sizeof(uint32_t)}};
            uint64_t memory = emitters.Sources.size() * sizeof(PackedEmitter);
            for (const auto& allocation : storage)
                memory += allocation.Bytes;
            if (!CheckVideoMemory(memory, "Compute photon buffers", result.Error))
                return false;
            for (const auto& allocation : storage)
                if (!CreateBuffer(*allocation.Buffer, allocation.Bytes, D3D12_HEAP_TYPE_DEFAULT,
                                  D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                  D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, result.Error))
                    return false;
            ComPtr<ID3D12Resource> staging;
            if (!PrepareUpload(sources, emitters.Sources.data(),
                               emitters.Sources.size() * sizeof(PackedEmitter), staging, result.Error) ||
                !CreateBuffer(readback, readbackBytes, D3D12_HEAP_TYPE_READBACK,
                              D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE, result.Error) ||
                !CreateBuffer(frame, D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT, D3D12_HEAP_TYPE_UPLOAD,
                              D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_FLAG_NONE, result.Error))
                return false;
            const D3D12_RANGE noRead{0, 0}, written{0, sizeof(constants)};
            void* mapped = nullptr;
            if (!Check(frame.Resource->Map(0, &noRead, &mapped), "Map(photon frame)", result.Error))
                return false;
            std::memcpy(mapped, &constants, sizeof(constants));
            frame.Resource->Unmap(0, &written);
            if (sources.Resource)
            {
                if (!BeginCommands(result.Error))
                    return false;
                CommandList->CopyBufferRegion(sources.Resource.Get(), 0, staging.Get(), 0, sources.Size);
                Transition(sources, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                if (!ExecuteCommands(result.Error))
                    return false;
                staging.Reset();
            }
            D3D12_DESCRIPTOR_HEAP_DESC description{};
            description.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
            description.NumDescriptors = ComputePhoton::DescriptorCount;
            description.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
            ComPtr<ID3D12DescriptorHeap> heap;
            if (!Check(Device->CreateDescriptorHeap(&description, IID_PPV_ARGS(&heap)),
                       "CreateDescriptorHeap(photon)", result.Error))
                return false;
            CreateRawSrv(0, Nodes, heap.Get());
            CreateStructuredSrv(1, PrimitiveRefs, PrimitiveRefCount, sizeof(PackedPrimitiveRef), heap.Get());
            CreateStructuredSrv(2, Triangles, TriangleCount, sizeof(PackedTriangle), heap.Get());
            CreateStructuredSrv(3, Spheres, SphereCount, sizeof(PackedSphere), heap.Get());
            CreateStructuredSrv(4, Materials, MaterialCount, sizeof(PackedMaterial), heap.Get());
            CreateStructuredSrv(5, sources, uint32_t(emitters.Sources.size()), sizeof(PackedEmitter),
                                heap.Get());
            CreateStructuredSrv(6, TextureDescriptors, TextureCount,
                                sizeof(Detail::PackedRayTextureDescriptor), heap.Get());
            CreateStructuredSrv(7, TextureTexels, TextureTexelCount, sizeof(uint32_t), heap.Get());
            CreateStructuredSrv(8, Instances, InstanceCount, sizeof(ComputeInstances::Instance), heap.Get());
            CreateStructuredSrv(9, InstanceNodes, InstanceNodeCount, sizeof(ComputeInstances::Node),
                                heap.Get());
            constexpr uint32_t output = ComputePhoton::SceneDescriptors;
            CreateStructuredUav(output, Accumulation, pixels, sizeof(Vec4), heap.Get());
            CreateStructuredUav(output + 1, SampleCounts, pixels, sizeof(uint32_t), heap.Get());
            CreateStructuredUav(output + 2, heads, buckets, sizeof(uint32_t), heap.Get());
            CreateStructuredUav(output + 3, statistics, ComputePhoton::TransportBatch,
                                sizeof(GroupStatistics), heap.Get());
            CreateStructuredUav(output + 4, SampleTargets, pixels, sizeof(uint32_t), heap.Get());
            CreateStructuredUav(output + 5, photons, slots, sizeof(ComputePhoton::Photon), heap.Get());
            CreateRawUav(output + 6, InstanceError, 1, heap.Get());
            CreateRawUav(output + 7, error, 1, heap.Get());
            constexpr uint32_t CausticsBit = 1, IndirectBit = 2;
            ComputePhoton::Controls controls{{emitters.Center.X, emitters.Center.Y, emitters.Center.Z},
                                             emitters.Radius,
                                             double(emitters.Radius) * double(options.RadiusScale),
                                             uint32_t(emitters.Sources.size()),
                                             options.PhotonPaths,
                                             slotsPerPath,
                                             buckets,
                                             options.Seed ^ settings.SamplingSeed,
                                             (options.Caustics ? CausticsBit : 0u) |
                                                 (options.Indirect ? IndirectBit : 0u),
                                             0,
                                             0,
                                             0,
                                             0};
            const auto dispatch = [&](ID3D12PipelineState* pipeline, uint32_t lanes, uint32_t statisticRows) {
                if (cancellation() || !BeginCommands(result.Error))
                    return false;
                ID3D12DescriptorHeap* heaps[] = {heap.Get()};
                CommandList->SetDescriptorHeaps(1, heaps);
                CommandList->SetComputeRootSignature(PhotonRoot.Get());
                auto handle = heap->GetGPUDescriptorHandleForHeapStart();
                CommandList->SetComputeRootDescriptorTable(0, handle);
                handle.ptr += uint64_t(output) * DescriptorSize;
                CommandList->SetComputeRootDescriptorTable(1, handle);
                CommandList->SetComputeRootConstantBufferView(2, frame.Resource->GetGPUVirtualAddress());
                CommandList->SetComputeRoot32BitConstants(3, sizeof(controls) / sizeof(uint32_t), &controls,
                                                          0);
                CommandList->SetPipelineState(pipeline);
                CommandList->Dispatch(
                    (lanes + ComputePhoton::ThreadsPerGroup - 1) / ComputePhoton::ThreadsPerGroup, 1, 1);
                // Emission -> insertion -> gathering are separate dispatches.
                // Publish both atomic heads and non-atomic links before any query.
                for (const auto& allocation : storage)
                    UavBarrier(*allocation.Buffer);
                UavBarrier(Accumulation);
                UavBarrier(SampleCounts);
                UavBarrier(SampleTargets);
                if (statisticRows)
                {
                    Transition(statistics, D3D12_RESOURCE_STATE_COPY_SOURCE);
                    CommandList->CopyBufferRegion(readback.Resource.Get(), 0, statistics.Resource.Get(), 0,
                                                  uint64_t(statisticRows) * sizeof(GroupStatistics));
                    Transition(statistics, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                }
                Transition(error, D3D12_RESOURCE_STATE_COPY_SOURCE);
                CommandList->CopyBufferRegion(readback.Resource.Get(), statisticsBytes, error.Resource.Get(),
                                              0, sizeof(uint32_t));
                Transition(error, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                CopyInstanceError();
                if (!ExecuteCommands(result.Error) || !CheckInstanceError(result.Error))
                    return false;
                void* data = nullptr;
                const D3D12_RANGE range{0, size_t(readbackBytes)}, noWrite{0, 0};
                if (!Check(readback.Resource->Map(0, &range, &data), "Map(photon diagnostics)", result.Error))
                    return false;
                uint32_t flags;
                std::memcpy(&flags, static_cast<const uint8_t*>(data) + statisticsBytes, sizeof(flags));
                const auto* rows = static_cast<const GroupStatistics*>(data);
                for (uint32_t index = 0; index < statisticRows; ++index)
                {
                    result.Statistics.PrimaryRays += rows[index].PrimaryRays.Value();
                    result.Statistics.ShadowRays += rows[index].ShadowRays.Value();
                    result.Statistics.ReflectionRays += rows[index].ReflectionRays.Value();
                    result.Statistics.PrimitiveTests += rows[index].PrimitiveTests.Value();
                }
                readback.Resource->Unmap(0, &noWrite);
                if (flags)
                    result.Error = "Compute photon mapping encountered invalid configuration or nonfinite "
                                   "transport arithmetic.";
                return flags == 0;
            };
            if (!dispatch(PhotonClear.Get(), buckets, 0))
                return false;
            for (uint32_t begin = 0; begin < options.PhotonPaths; begin += ComputePhoton::TransportBatch)
            {
                controls.Begin = begin;
                controls.Count = std::min(ComputePhoton::TransportBatch, options.PhotonPaths - begin);
                if (!dispatch(PhotonEmit.Get(), controls.Count, controls.Count))
                    return false;
            }
            for (uint32_t begin = 0; begin < slots; begin += ComputePhoton::IndexBatch)
            {
                controls.Begin = begin;
                controls.Count = std::min(ComputePhoton::IndexBatch, slots - begin);
                if (!dispatch(PhotonIndex.Get(), controls.Count, 0))
                    return false;
            }
            for (uint32_t sample = 0; sample < constants.TotalSamples; ++sample)
                for (uint32_t begin = 0; begin < pixels; begin += ComputePhoton::TransportBatch)
                {
                    controls.Begin = begin;
                    controls.Count = std::min(ComputePhoton::TransportBatch, pixels - begin);
                    controls.Sample = sample;
                    if (!dispatch(PhotonCamera.Get(), controls.Count, controls.Count))
                        return false;
                }
            return true;
        }

        bool EnsureProgressivePhotonPipelines(std::string& error) const
        {
            if (!ProgressivePhotonRoot &&
                !Check(Device->CreateRootSignature(0, Detail::ComputePpmRootSignature,
                                                   sizeof(Detail::ComputePpmRootSignature),
                                                   IID_PPV_ARGS(&ProgressivePhotonRoot)),
                       "CreateRootSignature(PPM)", error))
                return false;
            struct Program
            {
                const void* Bytes;
                size_t Size;
                ComPtr<ID3D12PipelineState>* Pipeline;
            };
            const Program programs[] = {
                {Detail::ComputePpmClearDxil, sizeof(Detail::ComputePpmClearDxil), &ProgressivePhotonClear},
                {Detail::ComputePpmInitializeDxil, sizeof(Detail::ComputePpmInitializeDxil),
                 &ProgressivePhotonInitialize},
                {Detail::ComputePpmCameraDxil, sizeof(Detail::ComputePpmCameraDxil),
                 &ProgressivePhotonCamera},
                {Detail::ComputePpmEmitDxil, sizeof(Detail::ComputePpmEmitDxil), &ProgressivePhotonEmit},
                {Detail::ComputePpmIndexDxil, sizeof(Detail::ComputePpmIndexDxil), &ProgressivePhotonIndex},
                {Detail::ComputePpmGatherDxil, sizeof(Detail::ComputePpmGatherDxil),
                 &ProgressivePhotonGather},
                {Detail::ComputePpmUpdateDxil, sizeof(Detail::ComputePpmUpdateDxil),
                 &ProgressivePhotonUpdate}};
            for (const auto& program : programs)
                if (!CreatePipeline(program.Bytes, program.Size, *program.Pipeline,
                                    "CreateComputePipelineState(PPM)", error, ProgressivePhotonRoot.Get()))
                    return false;
            return true;
        }

        bool RenderProgressivePhotons(const Scene& scene, const FrameConstants& constants,
                                      const RenderSettings& settings,
                                      const Detail::CancellationProbe& cancellation,
                                      RenderResult& result) const
        {
            namespace P = ComputePpm;
            if (settings.MaxReflectionBounces > P::MaximumReflections || settings.Adaptive.Enabled ||
                settings.Transport.Mis != MisHeuristic::None || !settings.ShadowsEnabled ||
                settings.ShadowStrength != 1)
            {
                result.Error = "Compute PPM requires bounded depth, fixed pass budgets, no connection MIS "
                               "and full visibility.";
                return false;
            }
            const auto emitters = ComputeEmission::PrepareEmitters(scene);
            for (const auto& source : emitters.Sources)
                if (!(source.Flags & EmitterCastsShadows))
                {
                    result.Error = "Compute PPM requires shadows on every emitter.";
                    return false;
                }
            const auto& options = settings.ProgressivePhotons;
            const uint32_t pixels = constants.Width * constants.Height;
            const uint32_t lightBatch = std::min(P::LightBatch, options.PathsPerPass);
            const uint32_t slotsPerPath = settings.MaxReflectionBounces + 1;
            const uint32_t slots = lightBatch * slotsPerPath;
            uint32_t buckets = 1;
            while (buckets < slots)
                buckets *= 2;
            constexpr uint32_t StatisticRows = std::max(P::LightBatch, P::CameraBatch);
            constexpr uint64_t StatisticsBytes = uint64_t(StatisticRows) * sizeof(GroupStatistics);
            constexpr uint64_t ReadbackBytes = StatisticsBytes + sizeof(uint32_t);
            GpuBuffer points, states, photons, heads, statistics, error, sources, frame, readback;
            struct Allocation
            {
                GpuBuffer* Buffer;
                uint64_t Bytes;
            };
            const Allocation allocations[] = {{&points, uint64_t(pixels) * sizeof(P::VisiblePoint)},
                                              {&states, uint64_t(pixels) * sizeof(P::Pixel)},
                                              {&photons, uint64_t(slots) * sizeof(P::Photon)},
                                              {&heads, uint64_t(buckets) * sizeof(uint32_t)},
                                              {&statistics, StatisticsBytes},
                                              {&error, sizeof(uint32_t)}};
            uint64_t memory = emitters.Sources.size() * sizeof(PackedEmitter);
            for (const auto& allocation : allocations)
                memory += allocation.Bytes;
            if (!CheckVideoMemory(memory, "Compute progressive photon buffers", result.Error))
                return false;
            for (const auto& allocation : allocations)
                if (!CreateBuffer(*allocation.Buffer, allocation.Bytes, D3D12_HEAP_TYPE_DEFAULT,
                                  D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                  D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, result.Error))
                    return false;
            ComPtr<ID3D12Resource> staging;
            if (!PrepareUpload(sources, emitters.Sources.data(),
                               emitters.Sources.size() * sizeof(PackedEmitter), staging, result.Error) ||
                !CreateBuffer(readback, ReadbackBytes, D3D12_HEAP_TYPE_READBACK,
                              D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE, result.Error) ||
                !CreateBuffer(frame, D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT, D3D12_HEAP_TYPE_UPLOAD,
                              D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_FLAG_NONE, result.Error))
                return false;
            void* mapped = nullptr;
            const D3D12_RANGE noRead{0, 0}, written{0, sizeof(constants)};
            if (!Check(frame.Resource->Map(0, &noRead, &mapped), "Map(PPM frame)", result.Error))
                return false;
            std::memcpy(mapped, &constants, sizeof(constants));
            frame.Resource->Unmap(0, &written);
            if (sources.Resource)
            {
                if (!BeginCommands(result.Error))
                    return false;
                CommandList->CopyBufferRegion(sources.Resource.Get(), 0, staging.Get(), 0, sources.Size);
                Transition(sources, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                if (!ExecuteCommands(result.Error))
                    return false;
                staging.Reset();
            }
            D3D12_DESCRIPTOR_HEAP_DESC description{};
            description.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
            description.NumDescriptors = P::DescriptorCount;
            description.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
            ComPtr<ID3D12DescriptorHeap> heap;
            if (!Check(Device->CreateDescriptorHeap(&description, IID_PPV_ARGS(&heap)),
                       "CreateDescriptorHeap(PPM)", result.Error))
                return false;
            CreateRawSrv(0, Nodes, heap.Get());
            CreateStructuredSrv(1, PrimitiveRefs, PrimitiveRefCount, sizeof(PackedPrimitiveRef), heap.Get());
            CreateStructuredSrv(2, Triangles, TriangleCount, sizeof(PackedTriangle), heap.Get());
            CreateStructuredSrv(3, Spheres, SphereCount, sizeof(PackedSphere), heap.Get());
            CreateStructuredSrv(4, Materials, MaterialCount, sizeof(PackedMaterial), heap.Get());
            CreateStructuredSrv(5, sources, uint32_t(emitters.Sources.size()), sizeof(PackedEmitter),
                                heap.Get());
            CreateStructuredSrv(6, TextureDescriptors, TextureCount,
                                sizeof(Detail::PackedRayTextureDescriptor), heap.Get());
            CreateStructuredSrv(7, TextureTexels, TextureTexelCount, sizeof(uint32_t), heap.Get());
            CreateStructuredSrv(8, Instances, InstanceCount, sizeof(ComputeInstances::Instance), heap.Get());
            CreateStructuredSrv(9, InstanceNodes, InstanceNodeCount, sizeof(ComputeInstances::Node),
                                heap.Get());
            constexpr uint32_t OutputBase = P::SceneDescriptors;
            CreateStructuredUav(OutputBase, Accumulation, pixels, sizeof(Vec4), heap.Get());
            CreateStructuredUav(OutputBase + 1, SampleCounts, pixels, sizeof(uint32_t), heap.Get());
            CreateStructuredUav(OutputBase + 2, heads, buckets, sizeof(uint32_t), heap.Get());
            CreateStructuredUav(OutputBase + 3, statistics, StatisticRows, sizeof(GroupStatistics), heap.Get());
            CreateStructuredUav(OutputBase + 4, SampleTargets, pixels, sizeof(uint32_t), heap.Get());
            CreateStructuredUav(OutputBase + 5, photons, slots, sizeof(P::Photon), heap.Get());
            CreateRawUav(OutputBase + 6, InstanceError, 1, heap.Get());
            CreateRawUav(OutputBase + 7, error, 1, heap.Get());
            CreateStructuredUav(OutputBase + 8, points, pixels, sizeof(P::VisiblePoint), heap.Get());
            CreateStructuredUav(OutputBase + 9, states, pixels, sizeof(P::Pixel), heap.Get());
            constexpr uint32_t CausticsBit = 1, IndirectBit = 2;
            P::Controls controls{{emitters.Center.X, emitters.Center.Y, emitters.Center.Z},
                                 emitters.Radius,
                                 double(emitters.Radius) * options.InitialRadiusScale,
                                 options.Alpha,
                                 uint32_t(emitters.Sources.size()),
                                 options.PathsPerPass,
                                 options.Seed ^ settings.SamplingSeed,
                                 (options.Caustics ? CausticsBit : 0u) |
                                     (options.Indirect ? IndirectBit : 0u),
                                 slotsPerPath,
                                 buckets};
            const auto dispatch = [&](ID3D12PipelineState* pipeline, uint32_t rows) {
                if (cancellation() || !BeginCommands(result.Error))
                    return false;
                ID3D12DescriptorHeap* heaps[] = {heap.Get()};
                CommandList->SetDescriptorHeaps(1, heaps);
                CommandList->SetComputeRootSignature(ProgressivePhotonRoot.Get());
                auto handle = heap->GetGPUDescriptorHandleForHeapStart();
                CommandList->SetComputeRootDescriptorTable(0, handle);
                handle.ptr += uint64_t(OutputBase) * DescriptorSize;
                CommandList->SetComputeRootDescriptorTable(1, handle);
                CommandList->SetComputeRootConstantBufferView(2, frame.Resource->GetGPUVirtualAddress());
                CommandList->SetComputeRoot32BitConstants(3, sizeof(controls) / sizeof(uint32_t), &controls,
                                                          0);
                CommandList->SetPipelineState(pipeline);
                CommandList->Dispatch((controls.Count + P::ThreadsPerGroup - 1) / P::ThreadsPerGroup, 1, 1);
                for (const auto& allocation : allocations)
                    UavBarrier(*allocation.Buffer);
                UavBarrier(Accumulation);
                UavBarrier(SampleCounts);
                UavBarrier(SampleTargets);
                if (rows)
                {
                    Transition(statistics, D3D12_RESOURCE_STATE_COPY_SOURCE);
                    CommandList->CopyBufferRegion(readback.Resource.Get(), 0, statistics.Resource.Get(), 0,
                                                  uint64_t(rows) * sizeof(GroupStatistics));
                    Transition(statistics, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                }
                Transition(error, D3D12_RESOURCE_STATE_COPY_SOURCE);
                CommandList->CopyBufferRegion(readback.Resource.Get(), StatisticsBytes, error.Resource.Get(),
                                              0, sizeof(uint32_t));
                Transition(error, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                CopyInstanceError();
                // ExecuteCommands waits for retirement even when cancellation is
                // requested during a dispatch, preserving resource lifetimes.
                if (!ExecuteCommands(result.Error) || !CheckInstanceError(result.Error))
                    return false;
                void* data = nullptr;
                const D3D12_RANGE read{0, size_t(ReadbackBytes)}, noWrite{0, 0};
                if (!Check(readback.Resource->Map(0, &read, &data), "Map(PPM diagnostics)", result.Error))
                    return false;
                uint32_t flags;
                std::memcpy(&flags, static_cast<const uint8_t*>(data) + StatisticsBytes, sizeof(flags));
                const auto* counters = static_cast<const GroupStatistics*>(data);
                for (uint32_t row = 0; row < rows; ++row)
                {
                    result.Statistics.PrimaryRays += counters[row].PrimaryRays.Value();
                    result.Statistics.ShadowRays += counters[row].ShadowRays.Value();
                    result.Statistics.ReflectionRays += counters[row].ReflectionRays.Value();
                    result.Statistics.PrimitiveTests += counters[row].PrimitiveTests.Value();
                }
                readback.Resource->Unmap(0, &noWrite);
                if (flags)
                    result.Error =
                        "Compute PPM encountered invalid configuration, indexing or nonfinite transport.";
                return flags == 0;
            };
            const auto stage = [&](ID3D12PipelineState* pipeline, uint32_t total, uint32_t batch,
                                   bool statistics) {
                for (uint32_t begin = 0; begin < total; begin += batch)
                {
                    controls.Begin = begin;
                    controls.Count = std::min(batch, total - begin);
                    if (!dispatch(pipeline, statistics ? controls.Count : 0))
                        return false;
                }
                return true;
            };
            if (!stage(ProgressivePhotonClear.Get(), buckets, P::LinearBatch, false) ||
                !stage(ProgressivePhotonInitialize.Get(), pixels, P::LinearBatch, false))
                return false;
            for (uint32_t pass = 0; pass < constants.TotalSamples; ++pass)
            {
                controls.Pass = pass;
                if (!stage(ProgressivePhotonCamera.Get(), pixels, P::CameraBatch, true))
                    return false;
                for (uint32_t begin = 0; begin < options.PathsPerPass; begin += lightBatch)
                {
                    controls.LightBegin = begin;
                    controls.LightCount = std::min(lightBatch, options.PathsPerPass - begin);
                    if (!stage(ProgressivePhotonClear.Get(), buckets, P::LinearBatch, false) ||
                        !stage(ProgressivePhotonEmit.Get(), controls.LightCount, lightBatch, true) ||
                        !stage(ProgressivePhotonIndex.Get(), controls.LightCount * slotsPerPath,
                               P::LinearBatch, false) ||
                        !stage(ProgressivePhotonGather.Get(), pixels, P::CameraBatch, false))
                        return false;
                }
                // Incoming flux/count span ALL light batches; N/R/tau advance
                // exactly once per camera pass, with all attempted paths in Nemit.
                if (!stage(ProgressivePhotonUpdate.Get(), pixels, P::LinearBatch, false))
                    return false;
            }
            return true;
        }

        bool MicropolygonPassNative(MicropolygonData::Controls controls,
                                    const std::vector<MicropolygonData::Patch>& patches,
                                    const std::vector<MicropolygonData::Range>& ranges,
                                    std::vector<uint32_t>& words, const Detail::CancellationProbe& cancelled,
                                    std::string& error) const
        {
            namespace M = MicropolygonData;
            if (!DicingRoot &&
                !Check(Device->CreateRootSignature(0, Detail::ComputeMicropolygonRootSignature,
                                                   sizeof(Detail::ComputeMicropolygonRootSignature),
                                                   IID_PPV_ARGS(&DicingRoot)),
                       "Creating Compute dicing root", error))
                return false;
            if (!CreatePipeline(Detail::ComputeMicropolygonDxil, sizeof(Detail::ComputeMicropolygonDxil),
                                DicingPipeline, "Creating Compute dicing pipeline", error, DicingRoot.Get()))
                return false;
            constexpr uint32_t InputCount = 2;
            const M::Range empty{};
            const void* sources[]{patches.data(), ranges.empty() ? &empty : ranges.data()};
            const uint32_t counts[]{uint32_t(patches.size()), uint32_t(std::max(size_t(1), ranges.size()))};
            const uint32_t strides[]{sizeof(M::Patch), sizeof(M::Range)};
            const uint64_t outputBytes = words.size() * sizeof(uint32_t);
            if (!CheckVideoMemory(outputBytes + uint64_t(counts[0]) * strides[0] +
                                      uint64_t(counts[1]) * strides[1],
                                  "Compute micropolygons", error))
                return false;
            GpuBuffer inputs[InputCount], output, readback, clear;
            ComPtr<ID3D12Resource> uploads[InputCount];
            for (uint32_t i = 0; i < InputCount; ++i)
                if (!PrepareUpload(inputs[i], sources[i], size_t(counts[i]) * strides[i], uploads[i], error))
                    return false;
            if (!CreateBuffer(output, outputBytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST,
                              D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, error) ||
                !CreateBuffer(readback, outputBytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST,
                              D3D12_RESOURCE_FLAG_NONE, error) ||
                !CreateBuffer(clear, outputBytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ,
                              D3D12_RESOURCE_FLAG_NONE, error))
                return false;
            void* mapped = nullptr;
            const D3D12_RANGE noAccess{0, 0}, allBytes{0, size_t(outputBytes)};
            if (!Check(clear.Resource->Map(0, &noAccess, &mapped), "Mapping Compute dicing clear", error))
                return false;
            std::memset(mapped, 0, size_t(outputBytes));
            clear.Resource->Unmap(0, &allBytes);
            D3D12_DESCRIPTOR_HEAP_DESC description{};
            description.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
            description.NumDescriptors = InputCount + 1;
            description.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
            ComPtr<ID3D12DescriptorHeap> heap;
            if (!Check(Device->CreateDescriptorHeap(&description, IID_PPV_ARGS(&heap)),
                       "Creating Compute dicing descriptors", error))
                return false;
            for (uint32_t i = 0; i < InputCount; ++i)
                CreateStructuredSrv(i, inputs[i], counts[i], strides[i], heap.Get());
            CreateStructuredUav(InputCount, output, uint32_t(words.size()), sizeof(uint32_t), heap.Get());
            if (cancelled() || !BeginCommands(error))
                return false;
            for (uint32_t i = 0; i < InputCount; ++i)
            {
                CommandList->CopyBufferRegion(inputs[i].Resource.Get(), 0, uploads[i].Get(), 0,
                                              uint64_t(counts[i]) * strides[i]);
                Transition(inputs[i], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            }
            CommandList->CopyBufferRegion(output.Resource.Get(), 0, clear.Resource.Get(), 0, outputBytes);
            Transition(output, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            if (!ExecuteCommands(error))
                return false;
            constexpr uint32_t BatchItems = 16384, Threads = 64;
            const uint32_t population = controls.Dispatch.Y;
            for (controls.Dispatch.X = 0; controls.Dispatch.X < population; controls.Dispatch.X += BatchItems)
            {
                if (cancelled() || !BeginCommands(error))
                    return false;
                controls.Dispatch.Y = std::min(BatchItems, population - controls.Dispatch.X);
                ID3D12DescriptorHeap* heaps[]{heap.Get()};
                CommandList->SetDescriptorHeaps(1, heaps);
                CommandList->SetComputeRootSignature(DicingRoot.Get());
                CommandList->SetComputeRoot32BitConstants(0, sizeof(controls) / sizeof(uint32_t), &controls,
                                                          0);
                auto handle = heap->GetGPUDescriptorHandleForHeapStart();
                CommandList->SetComputeRootDescriptorTable(1, handle);
                handle.ptr += uint64_t(InputCount) * DescriptorSize;
                CommandList->SetComputeRootDescriptorTable(2, handle);
                CommandList->SetPipelineState(DicingPipeline.Get());
                CommandList->Dispatch((controls.Dispatch.Y + Threads - 1) / Threads, 1, 1);
                UavBarrier(output);
                if (!ExecuteCommands(error))
                    return false;
            }
            if (cancelled() || !BeginCommands(error))
                return false;
            Transition(output, D3D12_RESOURCE_STATE_COPY_SOURCE);
            CommandList->CopyBufferRegion(readback.Resource.Get(), 0, output.Resource.Get(), 0, outputBytes);
            if (!ExecuteCommands(error) ||
                !Check(readback.Resource->Map(0, &allBytes, &mapped), "Reading Compute micropolygons", error))
                return false;
            std::memcpy(words.data(), mapped, size_t(outputBytes));
            readback.Resource->Unmap(0, &noAccess);
            return !cancelled();
        }

        bool ReconstructTemporalNative(
            TemporalData::Controls controls, const std::vector<TemporalPixel>& current,
            const std::vector<GeometryGuide>& guides, const std::vector<TemporalPixel>& previous,
            const std::vector<GeometryGuide>& oldGuides, const Detail::CancellationProbe& cancelled,
            std::vector<TemporalPixel>& result, Image& display, std::string& error) const
        {
            if (!TemporalRoot &&
                !Check(Device->CreateRootSignature(0, Detail::ComputeTemporalDenoiseRootSignature,
                                                   sizeof(Detail::ComputeTemporalDenoiseRootSignature),
                                                   IID_PPV_ARGS(&TemporalRoot)),
                       "Creating Compute temporal root", error))
                return false;
            if (!CreatePipeline(Detail::ComputeTemporalDenoiseDxil,
                                sizeof(Detail::ComputeTemporalDenoiseDxil), TemporalPipeline,
                                "Creating Compute temporal pipeline", error, TemporalRoot.Get()))
                return false;
            constexpr uint32_t InputCount = 4, OutputCount = 3, ColorOutput = 0, DisplayOutput = 1,
                               ErrorOutput = 2;
            const uint32_t count = controls.Width * controls.Height;
            const uint64_t colorBytes = uint64_t(count) * sizeof(TemporalPixel),
                           displayBytes = uint64_t(count) * sizeof(uint32_t);
            const uint64_t statusOffset = colorBytes + displayBytes;
            const TemporalPixel emptyColor{};
            const GeometryGuide emptyGuide{};
            const void* sources[] = {current.data(), guides.data(),
                                     previous.empty() ? &emptyColor : previous.data(),
                                     oldGuides.empty() ? &emptyGuide : oldGuides.data()};
            const uint32_t populations[] = {count, count, previous.empty() ? 1 : count,
                                            oldGuides.empty() ? 1 : count};
            const uint32_t strides[] = {sizeof(TemporalPixel), sizeof(GeometryGuide), sizeof(TemporalPixel),
                                        sizeof(GeometryGuide)};
            uint64_t allocation = statusOffset + sizeof(uint32_t);
            for (uint32_t i = 0; i < InputCount; ++i)
                allocation += uint64_t(populations[i]) * strides[i];
            if (!CheckVideoMemory(allocation, "Compute temporal reconstruction", error))
                return false;
            GpuBuffer inputs[InputCount], outputs[OutputCount], readback, clear;
            ComPtr<ID3D12Resource> uploads[InputCount];
            for (uint32_t i = 0; i < InputCount; ++i)
                if (!PrepareUpload(inputs[i], sources[i], size_t(populations[i]) * strides[i], uploads[i],
                                   error))
                    return false;
            const uint64_t outputBytes[] = {colorBytes, displayBytes, sizeof(uint32_t)};
            for (uint32_t i = 0; i < OutputCount; ++i)
                if (!CreateBuffer(outputs[i], outputBytes[i], D3D12_HEAP_TYPE_DEFAULT,
                                  i == ErrorOutput ? D3D12_RESOURCE_STATE_COPY_DEST
                                                   : D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                  D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, error))
                    return false;
            if (!CreateBuffer(readback, statusOffset + sizeof(uint32_t), D3D12_HEAP_TYPE_READBACK,
                              D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE, error) ||
                !CreateBuffer(clear, sizeof(uint32_t), D3D12_HEAP_TYPE_UPLOAD,
                              D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_FLAG_NONE, error))
                return false;
            void* mapped = nullptr;
            const D3D12_RANGE noRead{0, 0}, noWrite{0, 0}, clearWrite{0, sizeof(uint32_t)};
            if (!Check(clear.Resource->Map(0, &noRead, &mapped), "Mapping Compute temporal clear", error))
                return false;
            std::memset(mapped, 0, sizeof(uint32_t));
            clear.Resource->Unmap(0, &clearWrite);
            D3D12_DESCRIPTOR_HEAP_DESC description{};
            description.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
            description.NumDescriptors = InputCount + OutputCount;
            description.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
            ComPtr<ID3D12DescriptorHeap> heap;
            if (!Check(Device->CreateDescriptorHeap(&description, IID_PPV_ARGS(&heap)),
                       "Creating Compute temporal descriptors", error))
                return false;
            for (uint32_t i = 0; i < InputCount; ++i)
                CreateStructuredSrv(i, inputs[i], populations[i], strides[i], heap.Get());
            CreateStructuredUav(InputCount + ColorOutput, outputs[ColorOutput], count, sizeof(TemporalPixel),
                                heap.Get());
            CreateStructuredUav(InputCount + DisplayOutput, outputs[DisplayOutput], count, sizeof(uint32_t),
                                heap.Get());
            CreateStructuredUav(InputCount + ErrorOutput, outputs[ErrorOutput], 1, sizeof(uint32_t),
                                heap.Get());
            if (cancelled() || !BeginCommands(error))
                return false;
            for (uint32_t i = 0; i < InputCount; ++i)
            {
                CommandList->CopyBufferRegion(inputs[i].Resource.Get(), 0, uploads[i].Get(), 0,
                                              uint64_t(populations[i]) * strides[i]);
                Transition(inputs[i], D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            }
            CommandList->CopyBufferRegion(outputs[ErrorOutput].Resource.Get(), 0, clear.Resource.Get(), 0,
                                          sizeof(uint32_t));
            Transition(outputs[ErrorOutput], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            if (!ExecuteCommands(error))
                return false;
            clear.Reset();
            for (auto& upload : uploads)
                upload.Reset();
            constexpr uint32_t Batch = 16384, Threads = 64;
            ID3D12DescriptorHeap* heaps[] = {heap.Get()};
            for (controls.Begin = 0; controls.Begin < count; controls.Begin += Batch)
            {
                if (cancelled() || !BeginCommands(error))
                    return false;
                controls.Count = std::min(Batch, count - controls.Begin);
                Transition(outputs[ErrorOutput], D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                CommandList->SetDescriptorHeaps(1, heaps);
                CommandList->SetComputeRootSignature(TemporalRoot.Get());
                CommandList->SetComputeRoot32BitConstants(0, sizeof(controls) / sizeof(uint32_t), &controls,
                                                          0);
                auto descriptor = heap->GetGPUDescriptorHandleForHeapStart();
                CommandList->SetComputeRootDescriptorTable(1, descriptor);
                descriptor.ptr += uint64_t(InputCount) * DescriptorSize;
                CommandList->SetComputeRootDescriptorTable(2, descriptor);
                CommandList->SetPipelineState(TemporalPipeline.Get());
                CommandList->Dispatch((controls.Count + Threads - 1) / Threads, 1, 1);
                for (auto& output : outputs)
                    UavBarrier(output);
                Transition(outputs[ErrorOutput], D3D12_RESOURCE_STATE_COPY_SOURCE);
                CommandList->CopyBufferRegion(readback.Resource.Get(), statusOffset,
                                              outputs[ErrorOutput].Resource.Get(), 0, sizeof(uint32_t));
                if (!ExecuteCommands(error))
                    return false;
                const D3D12_RANGE read{size_t(statusOffset), size_t(statusOffset + sizeof(uint32_t))};
                if (!Check(readback.Resource->Map(0, &read, &mapped), "Reading Compute temporal status",
                           error))
                    return false;
                uint32_t failure = 0;
                std::memcpy(&failure, static_cast<const uint8_t*>(mapped) + statusOffset, sizeof(failure));
                readback.Resource->Unmap(0, &noWrite);
                if (failure)
                {
                    error = "Compute temporal reconstruction reported invalid configuration or radiance";
                    return false;
                }
            }
            if (cancelled() || !BeginCommands(error))
                return false;
            Transition(outputs[ColorOutput], D3D12_RESOURCE_STATE_COPY_SOURCE);
            Transition(outputs[DisplayOutput], D3D12_RESOURCE_STATE_COPY_SOURCE);
            CommandList->CopyBufferRegion(readback.Resource.Get(), 0, outputs[ColorOutput].Resource.Get(), 0,
                                          colorBytes);
            CommandList->CopyBufferRegion(readback.Resource.Get(), colorBytes,
                                          outputs[DisplayOutput].Resource.Get(), 0, displayBytes);
            if (!ExecuteCommands(error))
                return false;
            std::vector<TemporalPixel> filtered(count);
            Image encoded(controls.Width, controls.Height);
            const D3D12_RANGE read{0, size_t(statusOffset)};
            if (!Check(readback.Resource->Map(0, &read, &mapped), "Reading Compute temporal output", error))
                return false;
            std::memcpy(filtered.data(), mapped, size_t(colorBytes));
            std::memcpy(encoded.GetPixelsRGBA8().data(), static_cast<const uint8_t*>(mapped) + colorBytes,
                        size_t(displayBytes));
            readback.Resource->Unmap(0, &noWrite);
            if (cancelled())
                return false;
            result = std::move(filtered);
            display = std::move(encoded);
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
            if (!GeometryGuidesRoot &&
                !Check(Device->CreateRootSignature(0, Detail::ComputeGeometryGuidesRootSignature,
                                                   sizeof(Detail::ComputeGeometryGuidesRootSignature),
                                                   IID_PPV_ARGS(&GeometryGuidesRoot)),
                       "Creating Compute geometry guide root", error))
                return false;
            auto& pipeline = Generate ? GeometryGuideGenerate : GeometryGuideResolve;
            const void* code = Generate ? Detail::ComputeGenerateGeometryGuideRaysDxil
                                        : Detail::ComputeResolveGeometryGuideHitsDxil;
            const size_t codeBytes = Generate ? sizeof(Detail::ComputeGenerateGeometryGuideRaysDxil)
                                              : sizeof(Detail::ComputeResolveGeometryGuideHitsDxil);
            if (!CreatePipeline(code, codeBytes, pipeline, "Creating Compute geometry guide pipeline", error,
                                GeometryGuidesRoot.Get()))
                return false;
            const uint32_t count = controls.Width * controls.Height;
            const uint64_t outputBytes = uint64_t(count) * sizeof(Output),
                           rayBytes = uint64_t(rays.size()) * sizeof(rays[0]),
                           hitBytes = uint64_t(hits.size()) * sizeof(hits[0]);
            if (!CheckVideoMemory(outputBytes + rayBytes + hitBytes + sizeof(uint32_t),
                                  "Compute geometry guides", error))
                return false;
            GpuBuffer rayInput, hitInput, output, failures, readback, zero, nullView;
            ComPtr<ID3D12Resource> rayUpload, hitUpload;
            if constexpr (!Generate)
            {
                if (rays.size() != count || hits.size() != count)
                {
                    error = "Compute geometry guide input extent mismatch";
                    return false;
                }
                if (!PrepareUpload(rayInput, rays.data(), size_t(rayBytes), rayUpload, error) ||
                    !PrepareUpload(hitInput, hits.data(), size_t(hitBytes), hitUpload, error))
                    return false;
            }
            if (!CreateBuffer(output, outputBytes, D3D12_HEAP_TYPE_DEFAULT,
                              D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                              D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, error) ||
                !CreateBuffer(failures, sizeof(uint32_t), D3D12_HEAP_TYPE_DEFAULT,
                              D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                              error) ||
                !CreateBuffer(readback, outputBytes + sizeof(uint32_t), D3D12_HEAP_TYPE_READBACK,
                              D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE, error) ||
                !CreateBuffer(zero, sizeof(uint32_t), D3D12_HEAP_TYPE_UPLOAD,
                              D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_FLAG_NONE, error))
                return false;
            void* mapped = nullptr;
            const D3D12_RANGE noRead{0, 0}, noWrite{0, 0}, clearWrite{0, sizeof(uint32_t)};
            if (!Check(zero.Resource->Map(0, &noRead, &mapped), "Mapping geometry guide clear", error))
                return false;
            std::memset(mapped, 0, sizeof(uint32_t));
            zero.Resource->Unmap(0, &clearWrite);
            constexpr uint32_t SrvCount = 2, UavCount = 3, GuideDescriptors = SrvCount + UavCount;
            constexpr uint32_t RaySrv = 0, HitSrv = 1, RayUav = 2, GuideUav = 3, ErrorUav = 4;
            D3D12_DESCRIPTOR_HEAP_DESC description{};
            description.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
            description.NumDescriptors = GuideDescriptors;
            description.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
            ComPtr<ID3D12DescriptorHeap> heap;
            if (!Check(Device->CreateDescriptorHeap(&description, IID_PPV_ARGS(&heap)),
                       "Creating geometry guide descriptors", error))
                return false;
            CreateStructuredSrv(RaySrv, rayInput, Generate ? 1 : count, sizeof(GeometryGuideData::EncodedRay),
                                heap.Get());
            CreateStructuredSrv(HitSrv, hitInput, Generate ? 1 : count, sizeof(RayQueryHit), heap.Get());
            CreateStructuredUav(RayUav, Generate ? output : nullView, Generate ? count : 1,
                                sizeof(GeometryGuideData::EncodedRay), heap.Get());
            CreateStructuredUav(GuideUav, Generate ? nullView : output, Generate ? 1 : count,
                                sizeof(GeometryGuide), heap.Get());
            CreateStructuredUav(ErrorUav, failures, 1, sizeof(uint32_t), heap.Get());
            if (cancelled() || !BeginCommands(error))
                return false;
            if constexpr (!Generate)
            {
                CommandList->CopyBufferRegion(rayInput.Resource.Get(), 0, rayUpload.Get(), 0, rayBytes);
                CommandList->CopyBufferRegion(hitInput.Resource.Get(), 0, hitUpload.Get(), 0, hitBytes);
                Transition(rayInput, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                Transition(hitInput, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            }
            CommandList->CopyBufferRegion(failures.Resource.Get(), 0, zero.Resource.Get(), 0,
                                          sizeof(uint32_t));
            Transition(failures, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            if (!ExecuteCommands(error))
                return false;
            zero.Reset();
            rayUpload.Reset();
            hitUpload.Reset();
            constexpr uint32_t Threads = 64, Batch = 16384;
            ID3D12DescriptorHeap* heaps[]{heap.Get()};
            for (controls.Begin = 0; controls.Begin < count; controls.Begin += Batch)
            {
                if (cancelled() || !BeginCommands(error))
                    return false;
                controls.Count = std::min(Batch, count - controls.Begin);
                Transition(failures, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                CommandList->SetDescriptorHeaps(1, heaps);
                CommandList->SetComputeRootSignature(GeometryGuidesRoot.Get());
                CommandList->SetComputeRoot32BitConstants(0, sizeof(controls) / sizeof(uint32_t), &controls,
                                                          0);
                auto descriptor = heap->GetGPUDescriptorHandleForHeapStart();
                CommandList->SetComputeRootDescriptorTable(1, descriptor);
                descriptor.ptr += uint64_t(SrvCount) * DescriptorSize;
                CommandList->SetComputeRootDescriptorTable(2, descriptor);
                CommandList->SetPipelineState(pipeline.Get());
                CommandList->Dispatch((controls.Count + Threads - 1) / Threads, 1, 1);
                UavBarrier(output);
                UavBarrier(failures);
                Transition(failures, D3D12_RESOURCE_STATE_COPY_SOURCE);
                CommandList->CopyBufferRegion(readback.Resource.Get(), outputBytes, failures.Resource.Get(),
                                              0, sizeof(uint32_t));
                if (!ExecuteCommands(error))
                    return false;
                const D3D12_RANGE statusRead{size_t(outputBytes), size_t(outputBytes + sizeof(uint32_t))};
                if (!Check(readback.Resource->Map(0, &statusRead, &mapped),
                           "Reading geometry guide diagnostics", error))
                    return false;
                uint32_t failure = 0;
                std::memcpy(&failure, static_cast<const uint8_t*>(mapped) + outputBytes, sizeof(failure));
                readback.Resource->Unmap(0, &noWrite);
                if (failure)
                {
                    error = "Compute geometry guides encountered invalid dispatch or geometry";
                    return false;
                }
            }
            if (cancelled() || !BeginCommands(error))
                return false;
            Transition(output, D3D12_RESOURCE_STATE_COPY_SOURCE);
            CommandList->CopyBufferRegion(readback.Resource.Get(), 0, output.Resource.Get(), 0, outputBytes);
            if (!ExecuteCommands(error))
                return false;
            std::vector<Output> decoded(count);
            const D3D12_RANGE outputRead{0, size_t(outputBytes)};
            if (!Check(readback.Resource->Map(0, &outputRead, &mapped), "Reading Compute geometry guides",
                       error))
                return false;
            std::memcpy(decoded.data(), mapped, size_t(outputBytes));
            readback.Resource->Unmap(0, &noWrite);
            if (cancelled())
                return false;
            result = std::move(decoded);
            return true;
        }

        bool AllocateDenoisedNative(const std::vector<AdaptiveReconstructionData::Pixel>& input,
            AdaptiveReconstructionData::Controls controls, const Detail::CancellationProbe& cancellation,
            std::vector<uint32_t>& result, std::string& error) const
        {
            if (!AdaptiveDenoiseRoot &&
                !Check(Device->CreateRootSignature(0, Detail::ComputeAdaptiveDenoiseRootSignature,
                    sizeof(Detail::ComputeAdaptiveDenoiseRootSignature), IID_PPV_ARGS(&AdaptiveDenoiseRoot)),
                    "CreateRootSignature(adaptive denoising)", error)) return false;
            if (!CreatePipeline(Detail::ComputeAdaptiveDenoiseDxil, sizeof(Detail::ComputeAdaptiveDenoiseDxil),
                AdaptiveDenoisePipeline, "CreateComputePipelineState(adaptive denoising)", error,
                AdaptiveDenoiseRoot.Get())) return false;
            const size_t inputBytes = input.size() * sizeof(input.front()), outputBytes = input.size() * sizeof(uint32_t);
            if (!CheckVideoMemory(inputBytes + outputBytes, "Compute adaptive allocation buffers", error)) return false;
            GpuBuffer raw, output, readback;
            ComPtr<ID3D12Resource> staging;
            if (!PrepareUpload(raw, input.data(), inputBytes, staging, error) ||
                !CreateBuffer(output, outputBytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, error) ||
                !CreateBuffer(readback, outputBytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_FLAG_NONE, error)) return false;
            if (cancellation() || !BeginCommands(error)) return false;
            CommandList->CopyBufferRegion(raw.Resource.Get(), 0, staging.Get(), 0, inputBytes);
            Transition(raw, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            if (!ExecuteCommands(error)) return false;
            staging.Reset();
            constexpr uint32_t Batch = 16384, Threads = 64;
            const auto pixels = uint32_t(input.size());
            for (controls.Begin = 0; controls.Begin < pixels; controls.Begin += Batch)
            {
                if (cancellation() || !BeginCommands(error)) return false;
                controls.Count = std::min(Batch, pixels - controls.Begin);
                CommandList->SetComputeRootSignature(AdaptiveDenoiseRoot.Get());
                CommandList->SetComputeRoot32BitConstants(0, sizeof(controls) / sizeof(uint32_t), &controls, 0);
                CommandList->SetComputeRootShaderResourceView(1, raw.Resource->GetGPUVirtualAddress());
                CommandList->SetComputeRootUnorderedAccessView(2, output.Resource->GetGPUVirtualAddress());
                CommandList->SetPipelineState(AdaptiveDenoisePipeline.Get());
                CommandList->Dispatch((controls.Count + Threads - 1) / Threads, 1, 1);
                UavBarrier(output);
                if (!ExecuteCommands(error)) return false;
            }
            if (cancellation() || !BeginCommands(error)) return false;
            Transition(output, D3D12_RESOURCE_STATE_COPY_SOURCE);
            CommandList->CopyBufferRegion(readback.Resource.Get(), 0, output.Resource.Get(), 0, outputBytes);
            if (!ExecuteCommands(error)) return false;
            std::vector<uint32_t> candidate(input.size());
            void* mapped = nullptr;
            const D3D12_RANGE range{0, outputBytes}, noWrite{0, 0};
            if (!Check(readback.Resource->Map(0, &range, &mapped), "Map(adaptive allocation)", error)) return false;
            std::memcpy(candidate.data(), mapped, outputBytes);
            readback.Resource->Unmap(0, &noWrite);
            if (cancellation()) return false;
            result = std::move(candidate);
            return true;
        }

        GeometryGuideResult CaptureGuidesNative(const Scene &scene, const Camera &camera,
                                                const RenderSettings &settings,
                                                const Detail::CancellationProbe &stopped) const
        {
            return Detail::CaptureGeometryGuides(
                scene, camera, settings, stopped, [&](const auto &controls, auto &rays, auto &error)
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
            namespace Data = ReconstructionData;
            if (!ValidateReconstructionInput(film, settings.Reconstruction, error) ||
                !ValidateReconstructionGuides(film, settings.Reconstruction, guides, error))
                return false;
            if (!SpatialDenoiseRoot &&
                !Check(Device->CreateRootSignature(0, Detail::ComputeSpatialDenoiseRootSignature,
                                                   sizeof(Detail::ComputeSpatialDenoiseRootSignature),
                                                   IID_PPV_ARGS(&SpatialDenoiseRoot)),
                       "CreateRootSignature(spatial denoising)", error))
                return false;
            if (!CreatePipeline(Detail::ComputeSpatialDenoiseDxil, sizeof(Detail::ComputeSpatialDenoiseDxil),
                                SpatialDenoisePipeline, "CreateComputePipelineState(spatial denoising)",
                                error, SpatialDenoiseRoot.Get()))
                return false;
            std::vector<Data::Pixel> input;
            input.reserve(film.Pixels.size());
            for (const auto& pixel : film.Pixels)
            {
                if (cancellation())
                    return false;
                input.push_back({pixel.MeanRadiance.X, pixel.MeanRadiance.Y, pixel.MeanRadiance.Z,
                                 pixel.LuminanceM2, pixel.SampleCount});
                if (guides)
                    Data::PackGuide(input.back(), guides->Pixels[input.size() - 1]);
            }
            const uint32_t pixels = uint32_t(input.size());
            const uint64_t inputBytes = uint64_t(pixels) * sizeof(Data::Pixel);
            const uint64_t colorBytes = uint64_t(pixels) * sizeof(Vec4),
                           displayBytes = uint64_t(pixels) * sizeof(uint32_t);
            const uint64_t statusOffset = colorBytes + displayBytes;
            if (!CheckVideoMemory(inputBytes + 2 * colorBytes + displayBytes + sizeof(uint32_t),
                                  "Compute reconstruction buffers", error))
                return false;
            GpuBuffer raw, previous, next, display, failures, readback, zeroUpload;
            ComPtr<ID3D12Resource> staging;
            if (!PrepareUpload(raw, input.data(), size_t(inputBytes), staging, error))
                return false;
            for (auto* buffer : {&previous, &next, &display, &failures})
            {
                const uint64_t bytes = buffer == &failures  ? sizeof(uint32_t)
                                       : buffer == &display ? displayBytes
                                                            : colorBytes;
                if (!CreateBuffer(*buffer, bytes, D3D12_HEAP_TYPE_DEFAULT,
                                  D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                  D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, error))
                    return false;
            }
            if (!CreateBuffer(readback, statusOffset + sizeof(uint32_t), D3D12_HEAP_TYPE_READBACK,
                              D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE, error))
                return false;
            if (!CreateBuffer(zeroUpload, sizeof(uint32_t), D3D12_HEAP_TYPE_UPLOAD,
                              D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_FLAG_NONE, error))
                return false;
            void* zeroData = nullptr;
            const D3D12_RANGE noRead{0, 0}, zeroWritten{0, sizeof(uint32_t)};
            if (!Check(zeroUpload.Resource->Map(0, &noRead, &zeroData), "Map(reconstruction clear)", error))
                return false;
            std::memset(zeroData, 0, sizeof(uint32_t));
            zeroUpload.Resource->Unmap(0, &zeroWritten);
            constexpr uint32_t SrvCount = 2, UavCount = 3, ReconstructionDescriptors = SrvCount + UavCount;
            constexpr uint32_t RawSlot = 0, PreviousSlot = 1, NextSlot = 2, FailureSlot = 3, DisplaySlot = 4;
            D3D12_DESCRIPTOR_HEAP_DESC description{};
            description.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
            description.NumDescriptors = ReconstructionDescriptors;
            description.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
            ComPtr<ID3D12DescriptorHeap> heap;
            if (!Check(Device->CreateDescriptorHeap(&description, IID_PPV_ARGS(&heap)),
                       "CreateDescriptorHeap(spatial denoising)", error))
                return false;
            CreateStructuredSrv(RawSlot, raw, pixels, sizeof(Data::Pixel), heap.Get());
            CreateStructuredUav(FailureSlot, failures, 1, sizeof(uint32_t), heap.Get());
            CreateStructuredUav(DisplaySlot, display, pixels, sizeof(uint32_t), heap.Get());
            if (cancellation() || !BeginCommands(error))
                return false;
            CommandList->CopyBufferRegion(raw.Resource.Get(), 0, staging.Get(), 0, inputBytes);
            Transition(raw, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            ID3D12DescriptorHeap* heaps[]{heap.Get()};
            CommandList->SetDescriptorHeaps(1, heaps);
            Transition(failures, D3D12_RESOURCE_STATE_COPY_DEST);
            CommandList->CopyBufferRegion(failures.Resource.Get(), 0, zeroUpload.Resource.Get(), 0,
                                          sizeof(uint32_t));
            Transition(failures, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            if (!ExecuteCommands(error))
                return false;
            staging.Reset();
            zeroUpload.Reset();
            const auto& options = settings.Reconstruction;
            Data::Controls controls{film.Width,
                                    film.Height,
                                    0,
                                    0,
                                    0,
                                    ReconstructionUsesMoments(settings.Integrator) ? Data::MomentGuidance : 0,
                                    options.NoiseScale,
                                    options.RelativeColorScale,
                                    options.AbsoluteColorScale,
                                    settings.GammaCorrection ? 1u : 0u,
                                    options.NormalScale,
                                    options.RelativePlaneScale};
            if (guides)
                controls.Flags |= Data::GeometryGuidance;
            constexpr uint32_t Batch = 16384, Threads = 64;
            for (controls.Pass = 0; controls.Pass < options.Passes; ++controls.Pass)
            {
                CreateStructuredSrv(PreviousSlot, previous, pixels, sizeof(Vec4), heap.Get());
                CreateStructuredUav(NextSlot, next, pixels, sizeof(Vec4), heap.Get());
                for (controls.Begin = 0; controls.Begin < pixels; controls.Begin += Batch)
                {
                    if (cancellation() || !BeginCommands(error))
                        return false;
                    controls.Count = std::min(Batch, pixels - controls.Begin);
                    Transition(previous, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                    Transition(next, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                    Transition(failures, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                    CommandList->SetDescriptorHeaps(1, heaps);
                    CommandList->SetComputeRootSignature(SpatialDenoiseRoot.Get());
                    CommandList->SetComputeRoot32BitConstants(0, sizeof(controls) / sizeof(uint32_t),
                                                              &controls, 0);
                    auto handle = heap->GetGPUDescriptorHandleForHeapStart();
                    CommandList->SetComputeRootDescriptorTable(1, handle);
                    handle.ptr += uint64_t(SrvCount) * DescriptorSize;
                    CommandList->SetComputeRootDescriptorTable(2, handle);
                    CommandList->SetPipelineState(SpatialDenoisePipeline.Get());
                    CommandList->Dispatch((controls.Count + Threads - 1) / Threads, 1, 1);
                    UavBarrier(next);
                    UavBarrier(display);
                    UavBarrier(failures);
                    Transition(failures, D3D12_RESOURCE_STATE_COPY_SOURCE);
                    CommandList->CopyBufferRegion(readback.Resource.Get(), statusOffset,
                                                  failures.Resource.Get(), 0, sizeof(uint32_t));
                    if (!ExecuteCommands(error))
                        return false;
                    const D3D12_RANGE read{size_t(statusOffset), size_t(statusOffset + sizeof(uint32_t))},
                        noWrite{0, 0};
                    void* data = nullptr;
                    if (!Check(readback.Resource->Map(0, &read, &data), "Map(reconstruction status)", error))
                        return false;
                    uint32_t failure = 0;
                    std::memcpy(&failure, static_cast<const uint8_t*>(data) + statusOffset, sizeof(failure));
                    readback.Resource->Unmap(0, &noWrite);
                    if (failure)
                    {
                        error = "Compute spatial denoising reported invalid state";
                        return false;
                    }
                }
                std::swap(previous, next);
            }
            if (cancellation() || !BeginCommands(error))
                return false;
            Transition(previous, D3D12_RESOURCE_STATE_COPY_SOURCE);
            Transition(display, D3D12_RESOURCE_STATE_COPY_SOURCE);
            CommandList->CopyBufferRegion(readback.Resource.Get(), 0, previous.Resource.Get(), 0, colorBytes);
            CommandList->CopyBufferRegion(readback.Resource.Get(), colorBytes, display.Resource.Get(), 0,
                                          displayBytes);
            if (!ExecuteCommands(error))
                return false;
            ReconstructedImage output{film.Width, film.Height, {}, Image(film.Width, film.Height)};
            output.Pixels.resize(pixels);
            const D3D12_RANGE read{0, size_t(statusOffset)}, noWrite{0, 0};
            void* data = nullptr;
            if (!Check(readback.Resource->Map(0, &read, &data), "Map(reconstructed image)", error))
                return false;
            const auto* colors = static_cast<const Vec4*>(data);
            bool stopped = false;
            for (uint32_t row = 0; row < film.Height && !stopped; ++row)
            {
                stopped = cancellation();
                const size_t first = size_t(row) * film.Width;
                for (uint32_t x = 0; x < film.Width; ++x)
                    output.Pixels[first + x] = {colors[first + x].X, colors[first + x].Y,
                                                colors[first + x].Z};
                std::memcpy(output.Display.GetPixelsRGBA8().data() + first * sizeof(uint32_t),
                            static_cast<const uint8_t*>(data) + colorBytes + first * sizeof(uint32_t),
                            size_t(film.Width) * sizeof(uint32_t));
            }
            readback.Resource->Unmap(0, &noWrite);
            if (stopped)
                return false;
            result = std::move(output);
            return true;
        }

        bool EnsurePointGiPipelines(std::string& error) const
        {
            if (!PointGiRoot &&
                !Check(Device->CreateRootSignature(0, Detail::ComputePointGiRootSignature,
                                                   sizeof(Detail::ComputePointGiRootSignature),
                                                   IID_PPV_ARGS(&PointGiRoot)),
                       "CreateRootSignature(Point GI)", error))
                return false;
            return CreatePipeline(Detail::ComputePointGiInitializeDxil,
                                  sizeof(Detail::ComputePointGiInitializeDxil), PointGiInitialize,
                                  "CreateComputePipelineState(Point GI initialize)", error,
                                  PointGiRoot.Get()) &&
                   CreatePipeline(Detail::ComputePointGiBakeDxil, sizeof(Detail::ComputePointGiBakeDxil),
                                  PointGiBake, "CreateComputePipelineState(Point GI bake)", error,
                                  PointGiRoot.Get()) &&
                   CreatePipeline(Detail::ComputePointGiCameraDxil, sizeof(Detail::ComputePointGiCameraDxil),
                                  PointGiCamera, "CreateComputePipelineState(Point GI camera)", error,
                                  PointGiRoot.Get());
        }

        bool RenderPointGi(const Scene& scene, const FrameConstants& constants,
                           const RenderSettings& settings, const Detail::CancellationProbe& cancellation,
                           RenderResult& result) const
        {
            namespace G = ComputePointGi;
            if (!ValidatePointGiOptions(settings.PointGi, result.Error) ||
                !ValidateTransportOptions(settings.Integrator, settings.Transport, result.Error))
                return false;
            if (settings.MaxReflectionBounces > G::MaximumReflections || settings.Adaptive.Enabled ||
                !settings.ShadowsEnabled || settings.ShadowStrength != 1)
            {
                result.Error = "Compute Point GI requires bounded depth, fixed samples and full visibility.";
                return false;
            }
            const auto emitters = ComputeEmission::PrepareEmitters(scene);
            constexpr uint32_t UniformSourceLimit = 1u << 24;
            if (emitters.Sources.size() > UniformSourceLimit)
            {
                result.Error = "Compute Point GI source count exceeds the uniform sampler precision.";
                return false;
            }
            for (const auto& emitter : emitters.Sources)
                if (!(emitter.Flags & EmitterCastsShadows))
                {
                    result.Error = "Compute Point GI requires shadows on every source.";
                    return false;
                }
            G::Geometry geometry;
            if (settings.MaxReflectionBounces > 0 &&
                !G::BuildGeometry(scene, settings.PointGi, cancellation, geometry, result.Error))
                return false;
            G::Controls controls{uint32_t(geometry.Surfaces.size()), uint32_t(geometry.Hierarchy.size()),
                                 uint32_t(emitters.Sources.size()),  settings.PointGi.Seed ^ settings.SamplingSeed,
                                 settings.PointGi.DirectSamples,     settings.PointGi.GatherSamples};
            const uint32_t pixels = constants.Width * constants.Height;
            constexpr uint64_t StatisticsBytes = uint64_t(G::WorkBatch) * sizeof(GroupStatistics);
            constexpr uint64_t DiagnosticBytes = StatisticsBytes + sizeof(uint32_t);
            const uint64_t radianceBytes = uint64_t(controls.SurfaceCount) * sizeof(G::Radiance);
            GpuBuffer surfaces, bounds, indices, sources, previous, next, total, statistics, error, frame,
                readback;
            struct Upload
            {
                GpuBuffer* Buffer;
                const void* Data;
                uint64_t Bytes;
            };
            const Upload uploads[] = {
                {&surfaces, geometry.Surfaces.data(),
                 uint64_t(geometry.Surfaces.size()) * sizeof(G::Surface)},
                {&bounds, geometry.Hierarchy.data(), uint64_t(geometry.Hierarchy.size()) * sizeof(G::Bounds)},
                {&indices, geometry.Indices.data(), uint64_t(geometry.Indices.size()) * sizeof(uint32_t)},
                {&sources, emitters.Sources.data(),
                 uint64_t(emitters.Sources.size()) * sizeof(PackedEmitter)}};
            struct Storage
            {
                GpuBuffer* Buffer;
                uint64_t Bytes;
            };
            const Storage storage[] = {{&previous, radianceBytes},
                                       {&next, radianceBytes},
                                       {&total, radianceBytes},
                                       {&statistics, StatisticsBytes},
                                       {&error, sizeof(uint32_t)}};
            uint64_t memory = 0;
            for (const auto& upload : uploads)
                memory += upload.Bytes;
            for (const auto& allocation : storage)
                memory += allocation.Bytes;
            if (!CheckVideoMemory(memory, "Compute Point GI buffers", result.Error))
                return false;
            for (const auto& allocation : storage)
                if (!CreateBuffer(*allocation.Buffer, allocation.Bytes, D3D12_HEAP_TYPE_DEFAULT,
                                  D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                  D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, result.Error))
                    return false;
            std::array<ComPtr<ID3D12Resource>, std::size(uploads)> staging;
            for (size_t i = 0; i < std::size(uploads); ++i)
                if (!PrepareUpload(*uploads[i].Buffer, uploads[i].Data, uploads[i].Bytes, staging[i],
                                   result.Error))
                    return false;
            if (!CreateBuffer(frame, D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT, D3D12_HEAP_TYPE_UPLOAD,
                              D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_FLAG_NONE, result.Error) ||
                !CreateBuffer(readback, DiagnosticBytes, D3D12_HEAP_TYPE_READBACK,
                              D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE, result.Error))
                return false;
            void* mapped = nullptr;
            const D3D12_RANGE noRead{0, 0}, written{0, sizeof(constants)};
            if (!Check(frame.Resource->Map(0, &noRead, &mapped), "Map(Point GI frame)", result.Error))
                return false;
            std::memcpy(mapped, &constants, sizeof(constants));
            frame.Resource->Unmap(0, &written);
            if (cancellation() || !BeginCommands(result.Error))
                return false;
            for (size_t i = 0; i < std::size(uploads); ++i)
            {
                auto& buffer = *uploads[i].Buffer;
                if (!buffer.Resource)
                    continue;
                CommandList->CopyBufferRegion(buffer.Resource.Get(), 0, staging[i].Get(), 0, buffer.Size);
                Transition(buffer, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            }
            if (!ExecuteCommands(result.Error))
                return false;
            for (auto& resource : staging)
                resource.Reset();
            D3D12_DESCRIPTOR_HEAP_DESC description{};
            description.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
            description.NumDescriptors = G::DescriptorCount;
            description.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
            ComPtr<ID3D12DescriptorHeap> heap;
            if (!Check(Device->CreateDescriptorHeap(&description, IID_PPV_ARGS(&heap)),
                       "CreateDescriptorHeap(Point GI)", result.Error))
                return false;
            CreateRawSrv(0, Nodes, heap.Get());
            CreateStructuredSrv(1, PrimitiveRefs, PrimitiveRefCount, sizeof(PackedPrimitiveRef), heap.Get());
            CreateStructuredSrv(2, Triangles, TriangleCount, sizeof(PackedTriangle), heap.Get());
            CreateStructuredSrv(3, Spheres, SphereCount, sizeof(PackedSphere), heap.Get());
            CreateStructuredSrv(4, Materials, MaterialCount, sizeof(PackedMaterial), heap.Get());
            CreateStructuredSrv(5, sources, controls.SourceCount, sizeof(PackedEmitter), heap.Get());
            CreateStructuredSrv(6, TextureDescriptors, TextureCount,
                                sizeof(Detail::PackedRayTextureDescriptor), heap.Get());
            CreateStructuredSrv(7, TextureTexels, TextureTexelCount, sizeof(uint32_t), heap.Get());
            CreateStructuredSrv(8, Instances, InstanceCount, sizeof(ComputeInstances::Instance), heap.Get());
            CreateStructuredSrv(9, InstanceNodes, InstanceNodeCount, sizeof(ComputeInstances::Node),
                                heap.Get());
            CreateStructuredSrv(10, surfaces, controls.SurfaceCount, sizeof(G::Surface), heap.Get());
            CreateStructuredSrv(11, bounds, controls.BoundsCount, sizeof(G::Bounds), heap.Get());
            CreateStructuredSrv(12, indices, controls.SurfaceCount, sizeof(uint32_t), heap.Get());
            constexpr uint32_t FrameBase = G::SceneDescriptors;
            CreateStructuredUav(FrameBase, Accumulation, pixels, sizeof(Vec4), heap.Get());
            CreateStructuredUav(FrameBase + 1, SampleCounts, pixels, sizeof(uint32_t), heap.Get());
            CreateStructuredUav(FrameBase + 3, statistics, G::WorkBatch, sizeof(GroupStatistics), heap.Get());
            CreateStructuredUav(FrameBase + 4, SampleTargets, pixels, sizeof(uint32_t), heap.Get());
            CreateRawUav(FrameBase + 6, InstanceError, 1, heap.Get());
            CreateRawUav(FrameBase + 7, error, 1, heap.Get());
            CreateStructuredUav(FrameBase + 8, total, controls.SurfaceCount, sizeof(G::Radiance), heap.Get());
            const auto bindOrders = [&] {
                CreateStructuredUav(FrameBase + 2, previous, controls.SurfaceCount, sizeof(G::Radiance),
                                    heap.Get());
                CreateStructuredUav(FrameBase + 5, next, controls.SurfaceCount, sizeof(G::Radiance),
                                    heap.Get());
            };
            bindOrders();
            const auto dispatch = [&](ID3D12PipelineState* pipeline, bool collectStatistics) {
                if (cancellation() || !BeginCommands(result.Error))
                    return false;
                ID3D12DescriptorHeap* heaps[]{heap.Get()};
                CommandList->SetDescriptorHeaps(1, heaps);
                CommandList->SetComputeRootSignature(PointGiRoot.Get());
                auto handle = heap->GetGPUDescriptorHandleForHeapStart();
                CommandList->SetComputeRootDescriptorTable(0, handle);
                handle.ptr += uint64_t(FrameBase) * DescriptorSize;
                CommandList->SetComputeRootDescriptorTable(1, handle);
                CommandList->SetComputeRootConstantBufferView(2, frame.Resource->GetGPUVirtualAddress());
                CommandList->SetComputeRoot32BitConstants(3, sizeof(controls) / sizeof(uint32_t), &controls,
                                                          0);
                CommandList->SetPipelineState(pipeline);
                CommandList->Dispatch((controls.Count + G::Threads - 1) / G::Threads, 1, 1);
                for (const auto& allocation : storage)
                    if (allocation.Buffer->Resource)
                        UavBarrier(*allocation.Buffer);
                UavBarrier(Accumulation);
                UavBarrier(SampleCounts);
                UavBarrier(SampleTargets);
                if (collectStatistics)
                {
                    Transition(statistics, D3D12_RESOURCE_STATE_COPY_SOURCE);
                    CommandList->CopyBufferRegion(readback.Resource.Get(), 0, statistics.Resource.Get(), 0,
                                                  uint64_t(controls.Count) * sizeof(GroupStatistics));
                    Transition(statistics, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                }
                Transition(error, D3D12_RESOURCE_STATE_COPY_SOURCE);
                CommandList->CopyBufferRegion(readback.Resource.Get(), StatisticsBytes, error.Resource.Get(),
                                              0, sizeof(uint32_t));
                Transition(error, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                CopyInstanceError();
                // Retire submitted commands before destroying or rebinding their resources,
                // including when a cancellation request arrives during GPU work.
                if (!ExecuteCommands(result.Error) || !CheckInstanceError(result.Error))
                    return false;
                void* data = nullptr;
                const D3D12_RANGE range{0, size_t(DiagnosticBytes)}, noWrite{0, 0};
                if (!Check(readback.Resource->Map(0, &range, &data), "Map(Point GI diagnostics)",
                           result.Error))
                    return false;
                uint32_t flags;
                std::memcpy(&flags, static_cast<const uint8_t*>(data) + StatisticsBytes, sizeof(flags));
                if (collectStatistics)
                {
                    const auto* rows = static_cast<const GroupStatistics*>(data);
                    for (uint32_t i = 0; i < controls.Count; ++i)
                    {
                        result.Statistics.PrimaryRays += rows[i].PrimaryRays.Value();
                        result.Statistics.ShadowRays += rows[i].ShadowRays.Value();
                        result.Statistics.ReflectionRays += rows[i].ReflectionRays.Value();
                        result.Statistics.PrimitiveTests += rows[i].PrimitiveTests.Value();
                    }
                }
                readback.Resource->Unmap(0, &noWrite);
                if (flags)
                    result.Error = "Compute Point GI encountered invalid configuration, indexing or "
                                   "nonfinite transport.";
                return flags == 0;
            };
            const auto population = [&](ID3D12PipelineState* pipeline, uint32_t count, bool statistics) {
                for (uint32_t begin = 0; begin < count; begin += G::WorkBatch)
                {
                    controls.Begin = begin;
                    controls.Count = std::min(G::WorkBatch, count - begin);
                    if (!dispatch(pipeline, statistics))
                        return false;
                }
                return true;
            };
            const auto sampleWindows = [&](ID3D12PipelineState* pipeline, uint32_t count, uint32_t samples) {
                for (uint32_t first = 0; first < samples; first += G::SampleBatch)
                {
                    controls.FirstSample = first;
                    controls.Samples = std::min(G::SampleBatch, samples - first);
                    if (!population(pipeline, count, true))
                        return false;
                }
                return true;
            };
            if (!population(PointGiInitialize.Get(), pixels, false))
                return false;
            for (uint32_t order = 0; order < settings.MaxReflectionBounces; ++order)
            {
                controls.Order = order;
                if (!sampleWindows(PointGiBake.Get(), controls.SurfaceCount,
                                   order == 0 ? controls.DirectSamples : controls.GatherSamples))
                    return false;
                std::swap(previous, next);
                bindOrders();
            }
            return sampleWindows(PointGiCamera.Get(), pixels, constants.TotalSamples);
        }

        bool EnsureVcmPipelines(std::string& error) const
        {
            std::array<std::vector<char>, 2> diagnosticCamera;
            if (GetEnvironmentVariableW(L"ARVO_D3D12_DEBUG", nullptr, 0) != 0)
            {
                if (GetEnvironmentVariableA("ARVO_COMPUTE_VCM_CAMERA_LIBRARY", nullptr, 0) != 0)
                {
                    error = "The fused VCM camera override is obsolete; use a CONNECTION or MERGE library.";
                    return false;
                }
                const char* names[] = {"ARVO_COMPUTE_VCM_CONNECTION_LIBRARY", "ARVO_COMPUTE_VCM_MERGE_LIBRARY"};
                for (size_t phase = 0; phase < diagnosticCamera.size(); ++phase)
                {
                    const DWORD length = GetEnvironmentVariableA(names[phase], nullptr, 0);
                    if (length == 0) continue;
                    std::vector<char> path(length);
                    if (GetEnvironmentVariableA(names[phase], path.data(), length) >= length)
                    {
                        error = "The diagnostic VCM camera library name changed while reading it.";
                        return false;
                    }
                    std::ifstream input(path.data(), std::ios::binary);
                    diagnosticCamera[phase].assign(std::istreambuf_iterator<char>(input), {});
                    if (input.bad() || diagnosticCamera[phase].empty())
                    {
                        error = "Cannot read the diagnostic VCM camera library.";
                        return false;
                    }
                }
            }
            namespace V = ComputeVcm;
            if (!VcmRoot && !Check(Device->CreateRootSignature(0, Detail::ComputeVcmRootSignature,
                                                               sizeof(Detail::ComputeVcmRootSignature),
                                                               IID_PPV_ARGS(&VcmRoot)),
                                   "CreateRootSignature(VCM)", error))
                return false;
            struct Program
            {
                V::Stage Stage;
                const void* Bytes;
                size_t Size;
            };
            const Program programs[] = {
                {V::Stage::Initialize, Detail::ComputeVcmInitializeDxil,
                 sizeof(Detail::ComputeVcmInitializeDxil)},
                {V::Stage::LightWalk, Detail::ComputeVcmLightWalksDxil,
                 sizeof(Detail::ComputeVcmLightWalksDxil)},
                {V::Stage::CameraWalk, Detail::ComputeVcmCameraWalksDxil,
                 sizeof(Detail::ComputeVcmCameraWalksDxil)},
                {V::Stage::BuildGrid, Detail::ComputeVcmBuildGridDxil,
                 sizeof(Detail::ComputeVcmBuildGridDxil)},
                {V::Stage::SortIndex, Detail::ComputeVcmSortIndexDxil,
                 sizeof(Detail::ComputeVcmSortIndexDxil)},
                {V::Stage::CameraEstimate, Detail::ComputeVcmCameraEstimateDxil,
                 sizeof(Detail::ComputeVcmCameraEstimateDxil)},
                {V::Stage::CameraMerge, Detail::ComputeVcmCameraMergeDxil,
                 sizeof(Detail::ComputeVcmCameraMergeDxil)},
                {V::Stage::SplatEstimate, Detail::ComputeVcmSplatEstimateDxil,
                 sizeof(Detail::ComputeVcmSplatEstimateDxil)},
                {V::Stage::Commit, Detail::ComputeVcmCommitDxil, sizeof(Detail::ComputeVcmCommitDxil)}};
            for (const auto& program : programs)
            {
                const auto* diagnostic = program.Stage == V::Stage::CameraEstimate ? &diagnosticCamera[0] :
                    program.Stage == V::Stage::CameraMerge ? &diagnosticCamera[1] : nullptr;
                const bool overrideCamera = diagnostic && !diagnostic->empty();
                if (!CreatePipeline(overrideCamera ? diagnostic->data() : program.Bytes,
                                    overrideCamera ? diagnostic->size() : program.Size,
                                    VcmPipelines[size_t(program.Stage)],
                                    "CreateComputePipelineState(VCM)", error, VcmRoot.Get()))
                    return false;
            }
            return true;
        }

        bool RenderVcm(const Scene& scene, const FrameConstants& constants, const RenderSettings& settings,
                       const Detail::CancellationProbe& cancellation, RenderResult& result) const
        {
            namespace V = ComputeVcm;
            if (settings.MaxReflectionBounces > V::MaximumVertices - 3 || settings.Adaptive.Enabled ||
                settings.Transport.RussianRoulette || !settings.ShadowsEnabled ||
                settings.ShadowStrength != 1)
            {
                result.Error = "Compute VCM requires fixed passes/depth, at most 32 reflections, no adaptive "
                               "stopping or roulette, and full visibility.";
                return false;
            }
            if (!ValidateVertexConnectionMergingOptions(settings.Vcm, result.Error))
                return false;
            const auto emitters = ComputeEmission::PrepareEmitters(scene);
            for (const auto& emitter : emitters.Sources)
                if (!(emitter.Flags & EmitterCastsShadows))
                {
                    result.Error = "Compute VCM requires shadow casting on every emitter.";
                    return false;
                }
            const auto& options = settings.Vcm;
            V::DispatchPlan plan;
            if (!V::MakeDispatchPlan(
                    {constants.Width, constants.Height, constants.FullWidth, constants.FullHeight,
                     constants.TileX, constants.TileY},
                    constants.TotalSamples, settings.MaxReflectionBounces + 3, options.LightPathsPerPass,
                    options.Seed ^ settings.SamplingSeed, {emitters.Center.X, emitters.Center.Y, emitters.Center.Z}, emitters.Radius,
                    uint32_t(emitters.Sources.size()), double(emitters.Radius) * options.InitialRadiusScale,
                    options.Alpha, uint32_t(settings.Transport.Mis), settings.Transport.NextEventEstimation,
                    {}, plan, result.Error))
                return false;
            const uint64_t sourceBytes = uint64_t(emitters.Sources.size()) * sizeof(PackedEmitter);
            if (!CheckVideoMemory(plan.DeviceBytes + sourceBytes, "Compute VCM buffers", result.Error))
                return false;
            GpuBuffer vertices, metadata, index, splats, film, statistics, error, sources, upload, readback;
            struct Allocation
            {
                GpuBuffer* Buffer;
                uint64_t Bytes;
            };
            const Allocation allocations[] = {
                {&vertices, plan.VertexBytes}, {&metadata, plan.MetadataBytes},
                {&index, plan.IndexBytes},     {&splats, plan.SplatBytes},
                {&film, plan.FilmBytes},       {&statistics, plan.StatisticsBytes},
                {&error, sizeof(uint32_t)}};
            for (const auto& allocation : allocations)
                if (!CreateBuffer(*allocation.Buffer, allocation.Bytes, D3D12_HEAP_TYPE_DEFAULT,
                                  D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                  D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, result.Error))
                    return false;
            constexpr uint64_t ConstantAlignment = D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT;
            constexpr uint64_t CouplingOffset = ConstantAlignment, EmptyQueryOffset = 2 * ConstantAlignment;
            constexpr uint64_t UploadBytes = 3 * ConstantAlignment;
            const uint64_t counterReadbackBytes = uint64_t(plan.Limits.Transport) * sizeof(V::Statistics);
            const uint64_t readbackBytes = counterReadbackBytes + sizeof(uint32_t);
            static_assert(sizeof(FrameConstants) <= ConstantAlignment &&
                          sizeof(V::CouplingControls) <= ConstantAlignment);
            static_assert(sizeof(GroupStatistics) == sizeof(V::Statistics));
            ComPtr<ID3D12Resource> sourceUpload;
            if (!PrepareUpload(sources, emitters.Sources.data(), size_t(sourceBytes), sourceUpload,
                               result.Error) ||
                !CreateBuffer(upload, UploadBytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ,
                              D3D12_RESOURCE_FLAG_NONE, result.Error) ||
                !CreateBuffer(readback, readbackBytes, D3D12_HEAP_TYPE_READBACK,
                              D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE, result.Error))
                return false;
            void* mapped = nullptr;
            const D3D12_RANGE noRead{0, 0}, written{0, size_t(UploadBytes)};
            if (!Check(upload.Resource->Map(0, &noRead, &mapped), "Map(VCM constants)", result.Error))
                return false;
            // The shader's 240-byte frame CBV includes eight bytes of tail padding;
            // The existing frame fields keep their original offsets.
            std::memset(mapped, 0, size_t(UploadBytes));
            std::memcpy(mapped, &constants, sizeof(constants));
            std::memcpy(static_cast<uint8_t*>(mapped) + CouplingOffset, &plan.Coupling,
                        sizeof(plan.Coupling));
            upload.Resource->Unmap(0, &written);
            if (cancellation() || !BeginCommands(result.Error))
                return false;
            ClearInstanceError();
            if (sources.Resource)
            {
                CommandList->CopyBufferRegion(sources.Resource.Get(), 0, sourceUpload.Get(), 0, sourceBytes);
                Transition(sources, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            }
            // Error initialization is a separate ordered command, so no shader
            // lane can clear a failure reported concurrently by another lane.
            Transition(error, D3D12_RESOURCE_STATE_COPY_DEST);
            CommandList->CopyBufferRegion(error.Resource.Get(), 0, upload.Resource.Get(), EmptyQueryOffset,
                                          sizeof(uint32_t));
            Transition(error, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            if (!ExecuteCommands(result.Error))
                return false;
            sourceUpload.Reset();
            D3D12_DESCRIPTOR_HEAP_DESC heapDescription{};
            heapDescription.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
            heapDescription.NumDescriptors = V::DescriptorCount;
            heapDescription.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
            ComPtr<ID3D12DescriptorHeap> heap;
            if (!Check(Device->CreateDescriptorHeap(&heapDescription, IID_PPV_ARGS(&heap)),
                       "CreateDescriptorHeap(VCM)", result.Error))
                return false;
            CreateRawSrv(0, Nodes, heap.Get());
            CreateStructuredSrv(1, PrimitiveRefs, PrimitiveRefCount, sizeof(PackedPrimitiveRef), heap.Get());
            CreateStructuredSrv(2, Triangles, TriangleCount, sizeof(PackedTriangle), heap.Get());
            CreateStructuredSrv(3, Spheres, SphereCount, sizeof(PackedSphere), heap.Get());
            CreateStructuredSrv(4, Materials, MaterialCount, sizeof(PackedMaterial), heap.Get());
            CreateStructuredSrv(5, sources, uint32_t(emitters.Sources.size()), sizeof(PackedEmitter),
                                heap.Get());
            CreateStructuredSrv(6, TextureDescriptors, TextureCount,
                                sizeof(Detail::PackedRayTextureDescriptor), heap.Get());
            CreateStructuredSrv(7, TextureTexels, TextureTexelCount, sizeof(uint32_t), heap.Get());
            CreateStructuredSrv(8, Instances, InstanceCount, sizeof(ComputeInstances::Instance), heap.Get());
            CreateStructuredSrv(9, InstanceNodes, InstanceNodeCount, sizeof(ComputeInstances::Node),
                                heap.Get());
            constexpr uint32_t OutputBase = V::SceneDescriptors;
            GpuBuffer unusedCouplingOutput;
            CreateStructuredUav(OutputBase, vertices, plan.VertexCount, sizeof(V::PathVertex), heap.Get());
            CreateStructuredUav(OutputBase + 1, metadata, plan.MetadataCount, sizeof(V::Path), heap.Get());
            // u2/t11 belong to the standalone coupling probe. Frame kernels bind
            // a null output descriptor and a readable zero query, never dispatching it.
            CreateStructuredUav(OutputBase + 2, unusedCouplingOutput, 1, sizeof(V::Contribution), heap.Get());
            CreateStructuredUav(OutputBase + 3, statistics, plan.MetadataCount, sizeof(V::Statistics),
                                heap.Get());
            CreateStructuredUav(OutputBase + 4, index, plan.Frame.IndexCapacity, sizeof(V::IndexEntry),
                                heap.Get());
            CreateStructuredUav(OutputBase + 5, film, plan.Pixels, sizeof(V::FilmPixel), heap.Get());
            CreateRawUav(OutputBase + 6, InstanceError, 1, heap.Get());
            CreateRawUav(OutputBase + 7, error, 1, heap.Get());
            CreateStructuredUav(OutputBase + 8, splats, plan.Frame.IndexCapacity, sizeof(V::Splat),
                                heap.Get());
            CreateStructuredUav(OutputBase + 9, Accumulation, plan.Pixels, sizeof(Vec4), heap.Get());
            CreateStructuredUav(OutputBase + 10, SampleCounts, plan.Pixels, sizeof(uint32_t), heap.Get());
            CreateStructuredUav(OutputBase + 11, SampleTargets, plan.Pixels, sizeof(uint32_t), heap.Get());
            bool capturedCameraInputs = false;
            const auto captureCameraInputs = [&](const V::Dispatch& command) {
                const DWORD length = GetEnvironmentVariableA("ARVO_COMPUTE_VCM_TRACE", nullptr, 0);
                if (length == 0) return true;
                std::vector<char> prefix(length);
                if (GetEnvironmentVariableA("ARVO_COMPUTE_VCM_TRACE", prefix.data(), length) >= length)
                {
                    result.Error = "The VCM diagnostic prefix changed while reading it.";
                    return false;
                }
                const auto write = [&](const char* name, const void* data, size_t bytes) {
                    std::ofstream output(std::string(prefix.data()) + "-" + name + ".bin", std::ios::binary);
                    if (bytes) output.write(static_cast<const char*>(data), std::streamsize(bytes));
                    if (!output) result.Error = "Writing Compute VCM diagnostic input failed.";
                    return bool(output);
                };
                const auto capture = [&](const char* name, GpuBuffer& input) {
                    if (!input.Resource) return write(name, nullptr, 0);
                    GpuBuffer copy;
                    if (!CreateBuffer(copy, input.Size, D3D12_HEAP_TYPE_READBACK,
                                      D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE, result.Error) ||
                        !BeginCommands(result.Error)) return false;
                    const auto prior = input.State;
                    const bool transition = (prior & D3D12_RESOURCE_STATE_COPY_SOURCE) == 0;
                    if (transition) Transition(input, D3D12_RESOURCE_STATE_COPY_SOURCE);
                    CommandList->CopyBufferRegion(copy.Resource.Get(), 0, input.Resource.Get(), 0, input.Size);
                    if (transition) Transition(input, prior);
                    if (!ExecuteCommands(result.Error)) return false;
                    void* data = nullptr;
                    const D3D12_RANGE read{0, size_t(input.Size)}, noWrite{0, 0};
                    if (!Check(copy.Resource->Map(0, &read, &data), "Mapping VCM diagnostic input", result.Error))
                        return false;
                    const bool saved = write(name, data, size_t(input.Size));
                    copy.Resource->Unmap(0, &noWrite);
                    return saved;
                };
                // Exact native bytes, captured after grid sorting and before
                // the first camera estimate; no estimator work is replaced.
                return write("command", &command, sizeof(command)) &&
                       write("globals-cpu", &constants, sizeof(constants)) &&
                       write("coupling-cpu", &plan.Coupling, sizeof(plan.Coupling)) &&
                       capture("paths", metadata) && capture("vertices", vertices) &&
                       capture("index", index) && capture("upload", upload) &&
                       capture("sources", sources) && capture("materials", Materials) &&
                       capture("nodes", Nodes) && capture("references", PrimitiveRefs) &&
                       capture("triangles", Triangles) && capture("spheres", Spheres);
            };
            const auto execute = [&](const V::Dispatch& command) {
                if (!capturedCameraInputs && command.Program == V::Stage::CameraEstimate)
                {
                    if (!captureCameraInputs(command)) return false;
                    capturedCameraInputs = true;
                }
                if (cancellation() || !BeginCommands(result.Error))
                    return false;
                ID3D12DescriptorHeap* heaps[] = {heap.Get()};
                CommandList->SetDescriptorHeaps(1, heaps);
                CommandList->SetComputeRootSignature(VcmRoot.Get());
                auto handle = heap->GetGPUDescriptorHandleForHeapStart();
                CommandList->SetComputeRootDescriptorTable(0, handle);
                handle.ptr += uint64_t(OutputBase) * DescriptorSize;
                CommandList->SetComputeRootDescriptorTable(1, handle);
                const auto address = upload.Resource->GetGPUVirtualAddress();
                CommandList->SetComputeRootConstantBufferView(2, address);
                CommandList->SetComputeRoot32BitConstants(3, sizeof(command.Walk) / sizeof(uint32_t),
                                                          &command.Walk, 0);
                CommandList->SetComputeRootShaderResourceView(4, address + EmptyQueryOffset);
                CommandList->SetComputeRootConstantBufferView(5, address + CouplingOffset);
                CommandList->SetComputeRoot32BitConstants(6, sizeof(command.Frame) / sizeof(uint32_t),
                                                          &command.Frame, 0);
                CommandList->SetPipelineState(VcmPipelines[size_t(command.Program)].Get());
                CommandList->Dispatch((command.Threads + V::WalkThreads - 1) / V::WalkThreads, 1, 1);
                for (const auto& allocation : allocations)
                    UavBarrier(*allocation.Buffer);
                UavBarrier(Accumulation);
                UavBarrier(SampleCounts);
                UavBarrier(SampleTargets);
                if (command.StatisticsCount)
                {
                    Transition(statistics, D3D12_RESOURCE_STATE_COPY_SOURCE);
                    CommandList->CopyBufferRegion(readback.Resource.Get(), 0, statistics.Resource.Get(),
                                                  uint64_t(command.StatisticsBegin) * sizeof(V::Statistics),
                                                  uint64_t(command.StatisticsCount) * sizeof(V::Statistics));
                    Transition(statistics, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                }
                Transition(error, D3D12_RESOURCE_STATE_COPY_SOURCE);
                CommandList->CopyBufferRegion(readback.Resource.Get(), counterReadbackBytes,
                                              error.Resource.Get(), 0, sizeof(uint32_t));
                Transition(error, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                CopyInstanceError();
                // This wait also retires resources when cancellation arrives
                // during the dispatch; the planner then stops before the next one.
                if (!ExecuteCommands(result.Error) || !CheckInstanceError(result.Error))
                {
                    result.Error = "Compute VCM stage " + std::to_string(uint32_t(command.Program)) +
                        " pass " + std::to_string(command.Walk.Pass) + " work " +
                        std::to_string(command.Frame.WorkBegin) + "+" +
                        std::to_string(command.Frame.WorkCount) + " camera prefix " +
                        std::to_string(command.Frame.CameraPrefix) + ": " + result.Error;
                    return false;
                }
                void* data = nullptr;
                const D3D12_RANGE read{0, size_t(readbackBytes)}, noWrite{0, 0};
                if (!Check(readback.Resource->Map(0, &read, &data), "Map(VCM diagnostics)", result.Error))
                    return false;
                uint32_t flags;
                std::memcpy(&flags, static_cast<const uint8_t*>(data) + counterReadbackBytes, sizeof(flags));
                const auto* counters = static_cast<const GroupStatistics*>(data);
                for (uint32_t row = 0; row < command.StatisticsCount; ++row)
                {
                    result.Statistics.PrimaryRays += counters[row].PrimaryRays.Value();
                    result.Statistics.ShadowRays += counters[row].ShadowRays.Value();
                    result.Statistics.ReflectionRays += counters[row].ReflectionRays.Value();
                    result.Statistics.PrimitiveTests += counters[row].PrimitiveTests.Value();
                }
                readback.Resource->Unmap(0, &noWrite);
                if (flags)
                    result.Error =
                        "Compute VCM encountered invalid configuration, indexing or nonfinite transport "
                        "in pass " +
                        std::to_string(command.Walk.Pass) + ".";
                return flags == 0;
            };
            return V::RunDispatchPlan(plan, cancellation, execute);
        }

        bool EnsureMetropolisPipelines(std::string& error) const
        {
            if (!MetropolisRoot &&
                !Check(Device->CreateRootSignature(0, Detail::ComputeMetropolisRootSignature,
                                                   sizeof(Detail::ComputeMetropolisRootSignature),
                                                   IID_PPV_ARGS(&MetropolisRoot)),
                       "CreateRootSignature(MLT)", error))
                return false;
            return CreatePipeline(Detail::ComputeMetropolisBootstrapDxil,
                                  sizeof(Detail::ComputeMetropolisBootstrapDxil), MetropolisBootstrap,
                                  "CreateComputePipelineState(MLT bootstrap)", error, MetropolisRoot.Get()) &&
                   CreatePipeline(Detail::ComputeMetropolisDistributionDxil,
                                  sizeof(Detail::ComputeMetropolisDistributionDxil), MetropolisDistribution,
                                  "CreateComputePipelineState(MLT distribution)", error,
                                  MetropolisRoot.Get()) &&
                   CreatePipeline(Detail::ComputeMetropolisInitializeDxil,
                                  sizeof(Detail::ComputeMetropolisInitializeDxil), MetropolisInitialize,
                                  "CreateComputePipelineState(MLT initialization)", error,
                                  MetropolisRoot.Get()) &&
                   CreatePipeline(Detail::ComputeMetropolisMutateDxil,
                                  sizeof(Detail::ComputeMetropolisMutateDxil), MetropolisMutate,
                                  "CreateComputePipelineState(MLT mutations)", error, MetropolisRoot.Get()) &&
                   CreatePipeline(Detail::ComputeMetropolisClearDxil,
                                  sizeof(Detail::ComputeMetropolisClearDxil), MetropolisClear,
                                  "CreateComputePipelineState(MLT clear)", error, MetropolisRoot.Get()) &&
                   CreatePipeline(Detail::ComputeMetropolisCommitDxil,
                                  sizeof(Detail::ComputeMetropolisCommitDxil), MetropolisCommit,
                                  "CreateComputePipelineState(MLT film)", error, MetropolisRoot.Get());
        }

        bool RenderMetropolis(const Scene& scene, const FrameConstants& constants,
                              const RenderSettings& settings, const Detail::CancellationProbe& cancellation,
                              RenderResult& result) const
        {
            if (settings.MaxReflectionBounces > ComputeMlt::MaximumReflections || settings.Adaptive.Enabled ||
                !settings.ShadowsEnabled || settings.ShadowStrength != 1)
            {
                result.Error = "Compute MLT requires bounded depth, fixed mutation budgets and full physical "
                               "visibility.";
                return false;
            }
            const auto emitters = ComputeEmission::PrepareEmitters(scene);
            for (const auto& source : emitters.Sources)
                if (!(source.Flags & EmitterCastsShadows))
                {
                    result.Error = "Compute MLT requires shadow casting on every emitter.";
                    return false;
                }
            const auto& options = settings.Metropolis;
            const uint32_t pixels = constants.Width * constants.Height;
            const uint32_t fullPixels = constants.FullWidth * constants.FullHeight;
            const uint32_t chainCount = std::min(options.ChainCount, fullPixels);
            const uint32_t capacity = std::max(chainCount, ComputeMlt::BootstrapBatch);
            const uint64_t statisticsBytes = uint64_t(capacity) * sizeof(GroupStatistics);
            const uint64_t readbackBytes = statisticsBytes + sizeof(uint32_t);
            GpuBuffer sources, chains, primary, bootstrap, cdf, distribution, pass, statistics, error,
                readback, frame;
            struct Allocation
            {
                GpuBuffer* Buffer;
                uint64_t Bytes;
            };
            const Allocation allocations[] = {
                {&chains, uint64_t(chainCount) * sizeof(ComputeMlt::Chain)},
                {&primary, uint64_t(chainCount) * sizeof(ComputeMlt::Primary)},
                {&bootstrap, uint64_t(options.BootstrapSamples) * sizeof(ComputeMlt::PathValue)},
                {&cdf, uint64_t(options.BootstrapSamples) * sizeof(double)},
                {&distribution, 2 * sizeof(double)},
                {&pass, uint64_t(pixels) * sizeof(Vec4)},
                {&statistics, statisticsBytes},
                {&error, sizeof(uint32_t)}};
            uint64_t memory = emitters.Sources.size() * sizeof(PackedEmitter);
            for (const auto& allocation : allocations)
                memory += allocation.Bytes;
            if (!CheckVideoMemory(memory, "Compute MLT buffers", result.Error))
                return false;
            for (const auto& allocation : allocations)
                if (!CreateBuffer(*allocation.Buffer, allocation.Bytes, D3D12_HEAP_TYPE_DEFAULT,
                                  D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                  D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, result.Error))
                    return false;
            ComPtr<ID3D12Resource> staging;
            if (!PrepareUpload(sources, emitters.Sources.data(),
                               emitters.Sources.size() * sizeof(PackedEmitter), staging, result.Error) ||
                !CreateBuffer(readback, readbackBytes, D3D12_HEAP_TYPE_READBACK,
                              D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE, result.Error) ||
                !CreateBuffer(frame, D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT, D3D12_HEAP_TYPE_UPLOAD,
                              D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_FLAG_NONE, result.Error))
                return false;
            void* mapped = nullptr;
            const D3D12_RANGE noRead{0, 0}, written{0, sizeof(constants)};
            if (!Check(frame.Resource->Map(0, &noRead, &mapped), "Map(MLT frame)", result.Error))
                return false;
            std::memcpy(mapped, &constants, sizeof(constants));
            frame.Resource->Unmap(0, &written);
            if (sources.Resource)
            {
                if (!BeginCommands(result.Error))
                    return false;
                CommandList->CopyBufferRegion(sources.Resource.Get(), 0, staging.Get(), 0, sources.Size);
                Transition(sources, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                if (!ExecuteCommands(result.Error))
                    return false;
                staging.Reset();
            }
            D3D12_DESCRIPTOR_HEAP_DESC description{};
            description.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
            description.NumDescriptors = ComputeMlt::DescriptorCount;
            description.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
            ComPtr<ID3D12DescriptorHeap> heap;
            if (!Check(Device->CreateDescriptorHeap(&description, IID_PPV_ARGS(&heap)),
                       "CreateDescriptorHeap(MLT)", result.Error))
                return false;
            CreateRawSrv(0, Nodes, heap.Get());
            CreateStructuredSrv(1, PrimitiveRefs, PrimitiveRefCount, sizeof(PackedPrimitiveRef), heap.Get());
            CreateStructuredSrv(2, Triangles, TriangleCount, sizeof(PackedTriangle), heap.Get());
            CreateStructuredSrv(3, Spheres, SphereCount, sizeof(PackedSphere), heap.Get());
            CreateStructuredSrv(4, Materials, MaterialCount, sizeof(PackedMaterial), heap.Get());
            CreateStructuredSrv(5, sources, uint32_t(emitters.Sources.size()), sizeof(PackedEmitter),
                                heap.Get());
            CreateStructuredSrv(6, TextureDescriptors, TextureCount,
                                sizeof(Detail::PackedRayTextureDescriptor), heap.Get());
            CreateStructuredSrv(7, TextureTexels, TextureTexelCount, sizeof(uint32_t), heap.Get());
            CreateStructuredSrv(8, Instances, InstanceCount, sizeof(ComputeInstances::Instance), heap.Get());
            CreateStructuredSrv(9, InstanceNodes, InstanceNodeCount, sizeof(ComputeInstances::Node),
                                heap.Get());
            constexpr uint32_t output = ComputeMlt::SceneDescriptors;
            CreateStructuredUav(output, Accumulation, pixels, sizeof(Vec4), heap.Get());
            CreateStructuredUav(output + 1, SampleCounts, pixels, sizeof(uint32_t), heap.Get());
            CreateRawUav(output + 2, pass, uint32_t(pass.Size / sizeof(uint32_t)), heap.Get());
            CreateStructuredUav(output + 3, statistics, capacity, sizeof(GroupStatistics), heap.Get());
            CreateStructuredUav(output + 4, SampleTargets, pixels, sizeof(uint32_t), heap.Get());
            CreateStructuredUav(output + 5, chains, chainCount, sizeof(ComputeMlt::Chain), heap.Get());
            CreateRawUav(output + 6, InstanceError, 1, heap.Get());
            CreateRawUav(output + 7, error, 1, heap.Get());
            CreateStructuredUav(output + 8, primary, chainCount, sizeof(ComputeMlt::Primary), heap.Get());
            CreateStructuredUav(output + 9, bootstrap, options.BootstrapSamples,
                                sizeof(ComputeMlt::PathValue), heap.Get());
            CreateStructuredUav(output + 10, cdf, options.BootstrapSamples, sizeof(double), heap.Get());
            CreateStructuredUav(output + 11, distribution, 1, 2 * sizeof(double), heap.Get());
            ComputeMlt::Controls controls{{emitters.Center.X, emitters.Center.Y, emitters.Center.Z},
                                          emitters.Radius,
                                          uint32_t(emitters.Sources.size()),
                                          options.BootstrapSamples,
                                          chainCount,
                                          options.Seed ^ settings.SamplingSeed,
                                          0,
                                          0,
                                          options.LargeStepProbability,
                                          options.MutationScale};
            const auto dispatch = [&](const char* stage, ID3D12PipelineState* pipeline, uint32_t groupsX, uint32_t groupsY,
                                      uint32_t count) {
                if (cancellation() || !BeginCommands(result.Error))
                    return false;
                ID3D12DescriptorHeap* heaps[] = {heap.Get()};
                CommandList->SetDescriptorHeaps(1, heaps);
                CommandList->SetComputeRootSignature(MetropolisRoot.Get());
                auto handle = heap->GetGPUDescriptorHandleForHeapStart();
                CommandList->SetComputeRootDescriptorTable(0, handle);
                handle.ptr += uint64_t(output) * DescriptorSize;
                CommandList->SetComputeRootDescriptorTable(1, handle);
                CommandList->SetComputeRootConstantBufferView(2, frame.Resource->GetGPUVirtualAddress());
                CommandList->SetComputeRoot32BitConstants(3, sizeof(controls) / sizeof(uint32_t), &controls,
                                                          0);
                CommandList->SetPipelineState(pipeline);
                CommandList->Dispatch(groupsX, groupsY, 1);
                // Publish the CDF, persistent chain and film splats before a later
                // stage consumes them. No cross-wave locks or host path evaluation.
                for (const auto& allocation : allocations)
                    UavBarrier(*allocation.Buffer);
                UavBarrier(Accumulation);
                UavBarrier(SampleCounts);
                UavBarrier(SampleTargets);
                if (count)
                {
                    Transition(statistics, D3D12_RESOURCE_STATE_COPY_SOURCE);
                    CommandList->CopyBufferRegion(readback.Resource.Get(), 0, statistics.Resource.Get(), 0,
                                                  uint64_t(count) * sizeof(GroupStatistics));
                    Transition(statistics, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                }
                Transition(error, D3D12_RESOURCE_STATE_COPY_SOURCE);
                CommandList->CopyBufferRegion(readback.Resource.Get(), statisticsBytes, error.Resource.Get(),
                                              0, sizeof(uint32_t));
                Transition(error, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                CopyInstanceError();
                if (!ExecuteCommands(result.Error) || !CheckInstanceError(result.Error))
                {
                    const std::string interval =
                        pipeline == MetropolisMutate.Get() || pipeline == MetropolisBootstrap.Get()
                            ? " (work " + std::to_string(controls.Begin) + "+" +
                                  std::to_string(controls.Count) + ")"
                            : "";
                    result.Error = std::string("Compute MLT ") + stage + interval + ": " + result.Error;
                    return false;
                }
                void* data = nullptr;
                const D3D12_RANGE range{0, size_t(readbackBytes)}, noWrite{0, 0};
                if (!Check(readback.Resource->Map(0, &range, &data), "Map(MLT readback)", result.Error))
                    return false;
                uint32_t flags;
                std::memcpy(&flags, static_cast<const uint8_t*>(data) + statisticsBytes, sizeof(flags));
                const auto* rows = static_cast<const GroupStatistics*>(data);
                for (uint32_t index = 0; index < count; ++index)
                {
                    result.Statistics.PrimaryRays += rows[index].PrimaryRays.Value();
                    result.Statistics.ShadowRays += rows[index].ShadowRays.Value();
                    result.Statistics.ReflectionRays += rows[index].ReflectionRays.Value();
                    result.Statistics.PrimitiveTests += rows[index].PrimitiveTests.Value();
                }
                readback.Resource->Unmap(0, &noWrite);
                if (flags)
                    result.Error = "Compute MLT encountered invalid or nonfinite path/film arithmetic.";
                return flags == 0;
            };
            const auto groups = [](uint32_t lanes) {
                return (lanes + ComputeMlt::ThreadsPerGroup - 1) / ComputeMlt::ThreadsPerGroup;
            };
            const uint32_t filmX =
                (constants.Width + ComputeMlt::FilmGroupWidth - 1) / ComputeMlt::FilmGroupWidth;
            const uint32_t filmY =
                (constants.Height + ComputeMlt::FilmGroupWidth - 1) / ComputeMlt::FilmGroupWidth;
            if (!dispatch("clear", MetropolisClear.Get(), filmX, filmY, 0))
                return false;
            for (uint32_t begin = 0; begin < options.BootstrapSamples; begin += ComputeMlt::BootstrapBatch)
            {
                controls.Begin = begin;
                controls.Count = std::min(ComputeMlt::BootstrapBatch, options.BootstrapSamples - begin);
                if (!dispatch("bootstrap", MetropolisBootstrap.Get(), groups(controls.Count), 1, controls.Count))
                    return false;
            }
            if (!dispatch("distribution", MetropolisDistribution.Get(), 1, 1, 0) ||
                !dispatch("initialize", MetropolisInitialize.Get(), groups(chainCount), 1, 0))
                return false;
            const uint32_t quota = (fullPixels + chainCount - 1) / chainCount;
            for (uint32_t passIndex = 0; passIndex < constants.TotalSamples; ++passIndex)
            {
                if (passIndex && !dispatch("clear", MetropolisClear.Get(), filmX, filmY, 0))
                    return false;
                for (uint32_t begin = 0; begin < quota; begin += ComputeMlt::MutationBatch)
                {
                    controls.Begin = begin;
                    controls.Count = std::min(ComputeMlt::MutationBatch, quota - begin);
                    if (!dispatch("mutate", MetropolisMutate.Get(), groups(chainCount), 1, chainCount))
                        return false;
                }
                if (!dispatch("commit", MetropolisCommit.Get(), filmX, filmY, 0))
                    return false;
            }
            return true;
        }

        RenderResult RenderNative(const Scene& scene, const std::shared_ptr<const Scene>& owner,
                                  const Camera& camera, const RenderSettings& settings,
                                  const Detail::CancellationProbe& cancellation) const
        {
            CurrentMedium = {PackMedium(settings.Medium), {camera.Motion, HasSceneMotion(scene)}};
            RenderResult result;
            if (!ValidateInput(scene, camera, settings, result.Error))
                return result;
            if (!Detail::ValidateMaterialTransport(scene, BackendType::ComputeShader,
                    settings.Integrator, result.Error)) return result;
            const auto buildBegin = std::chrono::steady_clock::now();
            if (!EnsureRenderPipelines(settings.Integrator, result.Error)) return result;
            if (Selection == ComputeDeviceSelection::WarpDiagnostic)
                std::fprintf(stderr, "WARP scene upload begin\n");
            if (cancellation()) return Detail::MakeCancelledResult();
            if (!EnsureScene(scene, owner, {settings.Acceleration, settings.BvhPolicy}, cancellation,
                             result.Error,
                             settings.Radiometry == RadiometryMode::Spectral ? &settings.Medium : nullptr))
            {
                if (cancellation()) return Detail::MakeCancelledResult();
                return result;
            }
            const auto buildEnd = std::chrono::steady_clock::now();
            if (Selection == ComputeDeviceSelection::WarpDiagnostic)
                std::fprintf(stderr, "WARP scene upload end\n");
            if (cancellation()) return Detail::MakeCancelledResult();
            const std::uint32_t pixelCount = settings.Width * settings.Height;
            if (!EnsureFrameBuffers(pixelCount, result.Error)) return result;
            if (Selection == ComputeDeviceSelection::WarpDiagnostic)
                std::fprintf(stderr, "WARP frame buffers ready\n");

            FrameConstants constants;
            constants.InstanceNodeCount = InstanceNodeCount;
            constants.CameraPosition = ToVec3(camera.Position);
            constants.Width = settings.Width;
            constants.FullWidth = FullImageWidth(settings);
            constants.FullHeight = FullImageHeight(settings);
            constants.TileX = settings.TileX;
            constants.TileY = settings.TileY;
            constants.Integrator = static_cast<uint32_t>(settings.Integrator);
            constants.Acceleration = static_cast<uint32_t>(settings.Acceleration);
            constexpr uint32_t SpectralTransportBit = 1u << 5;
            constants.TransportFlags =
                PackTransportFlags(settings.Transport) |
                (settings.Radiometry == RadiometryMode::Spectral ? SpectralTransportBit : 0);
            constants.RouletteStartDepth = settings.Transport.RouletteStartDepth;
            constants.SamplingSequence = static_cast<uint32_t>(settings.Sampling);
            constants.SamplingSeed = settings.SamplingSeed;
            constants.AdaptiveEnabled = settings.Adaptive.Enabled ? 1u : 0u;
            constants.AdaptivePilotSamples = settings.Adaptive.PilotSamples;
            constants.AdaptiveMinimumSamples = settings.Adaptive.MinimumSamples;
            constants.AdaptiveRelativeError = settings.Adaptive.RelativeError;
            constants.AdaptiveAbsoluteError = settings.Adaptive.AbsoluteError;
            constants.Height = settings.Height;
            constants.TotalSamples = settings.SamplesPerPixel;
            constants.MaxReflectionBounces = settings.MaxReflectionBounces;
            const Vec3 forward = NormalizeVector(
                Subtract(ToVec3(camera.Target), ToVec3(camera.Position)));
            const Vec3 right = NormalizeVector(Cross(forward, ToVec3(camera.Up)));
            constants.CameraForward = forward;
            constants.LensRadius = camera.Lens.Radius;
            constants.FocusDistance = camera.Lens.FocusDistance;
            constants.CameraRight = right;
            constants.CameraUp = NormalizeVector(Cross(right, forward));
            constants.HalfHeight = std::tan(camera.VerticalFieldOfViewDegrees * Pi / 360.0f);
            constants.HalfWidth = constants.HalfHeight
                * static_cast<float>(FullImageWidth(settings)) / FullImageHeight(settings);
            constants.MaterialCount = MaterialCount;
            constants.LightCount = LightCount;
            constants.BackgroundTop = ToVec3(scene.BackgroundTop);
            constants.AmbientIntensity = scene.AmbientIntensity;
            constants.BackgroundBottom = ToVec3(scene.BackgroundBottom);
            constants.NodeCount = NodeCount;
            constants.PrimitiveRefCount = PrimitiveRefCount;
            constants.TriangleCount = TriangleCount;
            constants.SphereCount = SphereCount;
            constants.ShadowStrength = settings.ShadowsEnabled
                ? std::clamp(settings.ShadowStrength, 0.0f, 1.0f)
                : 0.0f;
            constants.GammaCorrection = settings.GammaCorrection ? 1u : 0u;
            constants.TextureCount = TextureCount;
            constants.TextureTexelCount = TextureTexelCount;

            if (!ClearFrame(constants, pixelCount, cancellation, result.Error))
            {
                if (cancellation()) return Detail::MakeCancelledResult();
                return result;
            }
            if (Selection == ComputeDeviceSelection::WarpDiagnostic)
                std::fprintf(stderr, "WARP frame clear end\n");
            const auto renderBegin = std::chrono::steady_clock::now();
            constexpr std::uint32_t PilotPhase = 1;
            const bool variancePilot = settings.Adaptive.Enabled && !settings.ProductionBudgets;
            const std::uint32_t phaseCount = variancePilot ? 2u : 1u;
            std::vector<std::uint32_t> sampleTargets(pixelCount, settings.SamplesPerPixel);
            std::uint32_t productionLimit = settings.SamplesPerPixel;
            if (settings.ProductionBudgets)
            {
                sampleTargets = settings.ProductionBudgets->Counts;
                productionLimit = *std::max_element(sampleTargets.begin(), sampleTargets.end());
                const size_t bytes = sampleTargets.size() * sizeof(uint32_t);
                GpuBuffer upload;
                if (!CreateBuffer(upload, bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ,
                                  D3D12_RESOURCE_FLAG_NONE, result.Error)) return result;
                void* mapped = nullptr;
                const D3D12_RANGE noRead{0, 0};
                if (!Check(upload.Resource->Map(0, &noRead, &mapped), "Map(production budgets)", result.Error)) return result;
                std::memcpy(mapped, sampleTargets.data(), bytes);
                upload.Resource->Unmap(0, nullptr);
                if (!BeginCommands(result.Error)) return result;
                Transition(SampleTargets, D3D12_RESOURCE_STATE_COPY_DEST);
                CommandList->CopyBufferRegion(SampleTargets.Resource.Get(), 0, upload.Resource.Get(), 0, bytes);
                Transition(SampleTargets, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                if (!ExecuteCommands(result.Error)) return result;
            }
            const bool bidirectional = settings.Integrator == Algorithm::BidirectionalPathTracing;
            const bool metropolis = settings.Integrator == Algorithm::MetropolisLightTransport;
            const bool photon = settings.Integrator == Algorithm::PhotonMapping;
            const bool progressive = settings.Integrator == Algorithm::ProgressivePhotonMapping;
            const bool vcm = settings.Integrator == Algorithm::VertexConnectionMerging;
            const bool pointGi = settings.Integrator == Algorithm::PointBasedGlobalIllumination;
            if (pointGi && !RenderPointGi(scene, constants, settings, cancellation, result))
            {
                if (cancellation()) result.Status = RenderStatus::Cancelled;
                return result;
            }
            if (vcm && !RenderVcm(scene, constants, settings, cancellation, result))
            {
                if (cancellation()) result.Status = RenderStatus::Cancelled;
                return result;
            }
            if (progressive && !RenderProgressivePhotons(scene, constants, settings, cancellation, result))
            {
                if (cancellation()) result.Status = RenderStatus::Cancelled;
                return result;
            }
            if (photon && !RenderPhotons(scene, constants, settings, cancellation, result))
            {
                if (cancellation()) result.Status = RenderStatus::Cancelled;
                return result;
            }
            if (bidirectional && !RenderBidirectional(scene, constants, settings, cancellation, result))
            {
                if (cancellation()) result.Status = RenderStatus::Cancelled;
                return result;
            }
            if (metropolis && !RenderMetropolis(scene, constants, settings, cancellation, result))
            {
                if (cancellation()) result.Status = RenderStatus::Cancelled;
                return result;
            }
            for (std::uint32_t phase = 0; !bidirectional && !metropolis && !photon && !progressive && !vcm && !pointGi && phase < phaseCount; ++phase)
            {
                const bool pilot = variancePilot && phase == 0;
                constants.AdaptivePhase = pilot ? PilotPhase : uint32_t(settings.SampleStream);
                const std::uint32_t sampleLimit = pilot ? settings.Adaptive.PilotSamples : productionLimit;
                for (std::uint32_t sample = 0; sample < sampleLimit; ++sample)
                {
                    for (std::uint32_t begin = 0; begin < pixelCount; begin += PixelChunkSize)
                    {
                        if (cancellation())
                        {
                            result.Status = RenderStatus::Cancelled;
                            result.Statistics.BuildMilliseconds =
                                std::chrono::duration<double, std::milli>(
                                    buildEnd - buildBegin).count();
                            result.Statistics.RenderMilliseconds =
                                std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() - renderBegin).count();
                            return result;
                        }
                        constants.PixelBegin = begin;
                        constants.PixelCount = std::min(PixelChunkSize, pixelCount - begin);
                        constants.SampleBegin = sample;
                        constants.SampleCount = 1;
                        constants.GroupCount =
                            (constants.PixelCount + ThreadsPerGroup - 1) / ThreadsPerGroup;
                        GroupStatistics statistics;
                        if (Selection == ComputeDeviceSelection::WarpDiagnostic)
                            std::fprintf(stderr, "WARP trace begin: sample %u\n", sample);
                        if (!DispatchTrace(constants, statistics, result.Error)) return result;
                        if (Selection == ComputeDeviceSelection::WarpDiagnostic)
                            std::fprintf(stderr, "WARP trace end\n");
                        result.Statistics.PrimaryRays += statistics.PrimaryRays.Value();
                        if (pilot) result.Statistics.PilotRays += statistics.PrimaryRays.Value();
                        result.Statistics.ShadowRays += statistics.ShadowRays.Value();
                        result.Statistics.ReflectionRays += statistics.ReflectionRays.Value();
                        result.Statistics.PrimitiveTests += statistics.PrimitiveTests.Value();
                    }
                }
                if (pilot)
                {
                    if (!ReadSampleTargets(pixelCount, sampleTargets, result.Error)) return result;
                    for (const std::uint32_t target : sampleTargets)
                    {
                        if (target < settings.Adaptive.MinimumSamples || target > settings.SamplesPerPixel)
                            return Detail::MakeFailedResult("Compute adaptive target is outside the requested sample limits.");
                    }
                    productionLimit = *std::max_element(sampleTargets.begin(), sampleTargets.end());
                }
            }

            std::vector<std::uint32_t> packedOutput;
            std::vector<std::uint32_t> completedSamples;
            result.Film = LinearFilm(settings.Width, settings.Height);
            if (!ResolveAndReadback(
                    constants, pixelCount, cancellation,
                    packedOutput, completedSamples, result.Film.Pixels, result.Error))
            {
                if (cancellation()) result.Status = RenderStatus::Cancelled;
                return result;
            }
            const auto renderEnd = std::chrono::steady_clock::now();
            result.Output = Image(settings.Width, settings.Height);
            std::memcpy(
                result.Output.GetPixelsRGBA8().data(), packedOutput.data(),
                static_cast<std::size_t>(pixelCount) * sizeof(std::uint32_t));
            for (size_t pixel = 0; pixel < completedSamples.size(); ++pixel)
            {
                if (completedSamples[pixel] != sampleTargets[pixel])
                    return Detail::MakeFailedResult("Compute film did not reach its per-pixel sample targets.");
                ++result.Statistics.CompletedPixels;
            }
            result.Statistics.BuildMilliseconds =
                std::chrono::duration<double, std::milli>(buildEnd - buildBegin).count();
            result.Statistics.RenderMilliseconds =
                std::chrono::duration<double, std::milli>(renderEnd - renderBegin).count();
            result.Status = cancellation() ? RenderStatus::Cancelled : RenderStatus::Completed;
            return result;
        }

        DynamicModule D3D12Module;
        DynamicModule DxgiModule;
        D3D12CreateDeviceFunction CreateDevice = nullptr;
        CreateDxgiFactory2Function CreateFactory = nullptr;
        ComPtr<IDXGIAdapter1> Adapter;
        ComPtr<IDXGIAdapter3> Adapter3;
        ComPtr<ID3D12Device> Device;
        ComPtr<ID3D12CommandQueue> Queue;
        ComPtr<ID3D12CommandAllocator> Allocator;
        ComPtr<ID3D12GraphicsCommandList> CommandList;
        ComPtr<ID3D12Fence> Fence;
        EventHandle FenceEvent;
        ComPtr<ID3D12InfoQueue1> DiagnosticInfo;
        DWORD DiagnosticCookie = 0;
        ComPtr<ID3D12DescriptorHeap> DescriptorHeap;
        ComPtr<ID3D12DescriptorHeap> ErrorClearHeap;
        ComPtr<ID3D12RootSignature> RootSignature;
        mutable ComPtr<ID3D12RootSignature> BidirectionalRoot;
        mutable ComPtr<ID3D12RootSignature> MetropolisRoot;
        mutable ComPtr<ID3D12RootSignature> PhotonRoot;
        mutable ComPtr<ID3D12PipelineState> PhotonClear, PhotonEmit, PhotonIndex, PhotonCamera;
        mutable ComPtr<ID3D12RootSignature> ProgressivePhotonRoot;
        mutable ComPtr<ID3D12RootSignature> VcmRoot;
        mutable ComPtr<ID3D12RootSignature> PointGiRoot;
        mutable ComPtr<ID3D12PipelineState> PointGiInitialize, PointGiBake, PointGiCamera;
        mutable std::array<ComPtr<ID3D12PipelineState>, size_t(ComputeVcm::Stage::Count)> VcmPipelines;
        mutable ComPtr<ID3D12PipelineState> ProgressivePhotonClear, ProgressivePhotonInitialize,
            ProgressivePhotonCamera, ProgressivePhotonEmit, ProgressivePhotonIndex, ProgressivePhotonGather,
            ProgressivePhotonUpdate;
        mutable ComPtr<ID3D12PipelineState> MetropolisBootstrap, MetropolisDistribution, MetropolisInitialize,
                                           MetropolisMutate, MetropolisClear, MetropolisCommit;
        mutable ComPtr<ID3D12PipelineState> BidirectionalPairs, BidirectionalClear, BidirectionalCommit;
        mutable ComPtr<ID3D12PipelineState> ClearPixelsPipeline;
        mutable ComPtr<ID3D12PipelineState> ClearStatisticsPipeline;
        mutable ComPtr<ID3D12PipelineState> WhittedPipeline;
        mutable ComPtr<ID3D12PipelineState> PathTracingPipeline;
        mutable ComPtr<ID3D12PipelineState> DistributedRayPipeline;
        mutable ComPtr<ID3D12PipelineState> ResolvePixelsPipeline;
        mutable ComPtr<ID3D12RootSignature> SpatialDenoiseRoot;
        mutable ComPtr<ID3D12RootSignature> AdaptiveDenoiseRoot;
        mutable ComPtr<ID3D12PipelineState> AdaptiveDenoisePipeline;
        mutable ComPtr<ID3D12RootSignature> GeometryGuidesRoot;
        mutable ComPtr<ID3D12RootSignature> TemporalRoot;
        mutable ComPtr<ID3D12RootSignature> DicingRoot;
        mutable ComPtr<ID3D12PipelineState> DicingPipeline;
        mutable ComPtr<ID3D12PipelineState> TemporalPipeline;
        mutable ComPtr<ID3D12PipelineState> GeometryGuideGenerate, GeometryGuideResolve;
        mutable ComPtr<ID3D12PipelineState> SpatialDenoisePipeline;
        mutable ComPtr<ID3D12PipelineState> QueryRaysPipeline;
        std::uint32_t DescriptorSize = 0;
        std::string AdapterName;

        mutable std::uint64_t FenceValue = 0;
        mutable GpuBuffer Instances;
        mutable GpuBuffer InstanceNodes;
        mutable GpuBuffer InstanceError;
        mutable GpuBuffer InstanceErrorReadback;
        mutable uint32_t InstanceCount = 0, InstanceNodeCount = 0;
        mutable GpuBuffer Nodes;
        mutable GpuBuffer PrimitiveRefs;
        mutable GpuBuffer Triangles;
        mutable GpuBuffer Spheres;
        mutable GpuBuffer Materials;
        mutable GpuBuffer Lights;
        mutable GpuBuffer TextureDescriptors;
        mutable GpuBuffer TextureTexels;
        mutable GpuBuffer SpectralPalette;
        mutable GpuBuffer SpectralReferences;
        mutable uint32_t SpectralSceneSourceCount = 0;
        mutable MediumOptions SpectralMedium;
        mutable ComPtr<ID3D12RootSignature> SpectralRoot;
        mutable ComPtr<ID3D12PipelineState> SpectralResponsePipeline, SpectralPalettePipeline;
        mutable GpuBuffer Accumulation;
        mutable GpuBuffer SampleCounts;
        mutable GpuBuffer SampleTargets;
        mutable GpuBuffer Output;
        mutable GpuBuffer Statistics;
        mutable GpuBuffer OutputReadback;
        mutable GpuBuffer FilmReadback;
        mutable GpuBuffer SamplesReadback;
        mutable GpuBuffer TargetsReadback;
        mutable GpuBuffer StatisticsReadback;
        mutable std::vector<GpuBuffer> FrameConstantPages;
        mutable uint64_t FrameConstantCursor = 0;
        mutable PathFrameData CurrentMedium;
        mutable std::uint32_t NodeCount = 0;
        mutable std::uint32_t PrimitiveRefCount = 0;
        mutable std::uint32_t TriangleCount = 0;
        mutable std::uint32_t SphereCount = 0;
        mutable std::uint32_t MaterialCount = 0;
        mutable std::uint32_t LightCount = 0;
        mutable std::uint32_t TextureCount = 0;
        mutable std::uint32_t TextureTexelCount = 0;
        mutable std::uint32_t FramePixelCapacity = 0;
        mutable std::weak_ptr<const Scene> CachedOwner;
        mutable AccelerationOptions CachedAcceleration = AccelerationType::Bvh;
#endif
        const ComputeDeviceSelection Selection;
        bool Available = false;
        std::string AvailabilityError;
        mutable Detail::PersistentRenderQueue ComputeQueue;
    };

    ComputeShaderRayTracingRHI::ComputeShaderRayTracingRHI()
        : ComputeShaderRayTracingRHI(ComputeDeviceSelection::Hardware)
    {
    }

    ComputeShaderRayTracingRHI::ComputeShaderRayTracingRHI(ComputeDeviceSelection selection)
        : m_Impl(std::make_unique<Impl>(selection))
    {
    }

    ComputeShaderRayTracingRHI::~ComputeShaderRayTracingRHI() = default;

    BackendInfo ComputeShaderRayTracingRHI::GetBackendInfo() const
    {
        BackendInfo info;
        info.Type = BackendType::ComputeShader;
        info.Name = m_Impl->Selection == ComputeDeviceSelection::WarpDiagnostic
            ? "ComputeShader (WARP diagnostic)" : "ComputeShader";
        info.IsNative = m_Impl->Available;
        info.IsHardwareAccelerated = m_Impl->Available && m_Impl->Selection == ComputeDeviceSelection::Hardware;
        info.IsAvailable = m_Impl->Available;
        info.Description = m_Impl->Available
#if defined(_WIN32)
            ? std::string("Native D3D12 compute shader backend on ") + m_Impl->AdapterName
#else
            ? "Native D3D12 compute shader backend"
#endif
            : std::string("Unavailable: ") + m_Impl->AvailabilityError;
        return info;
    }

    MicropolygonResult ComputeShaderRayTracingRHI::PrepareMicropolygons(
        const Scene& scene, const Camera& camera, const RenderSettings& settings,
        const CancellationToken& cancellation) const
    {
        MicropolygonResult result;
        const auto task = m_Impl->ComputeQueue
                              .Submit([&](const CancellationToken& queued) {
                                  const auto stopped = Detail::CombineCancellation(queued, cancellation);
                                  if (stopped())
                                      return Detail::MakeCancelledResult();
                                  if (!m_Impl->Available)
                                      return Detail::MakeFailedResult("Compute is unavailable: " +
                                                                      m_Impl->AvailabilityError);
#if defined(_WIN32)
                                  result = Detail::PrepareMicropolygons(
                                      scene, camera, settings, stopped,
                                      [&](const auto& controls, const auto& patches, const auto& ranges,
                                          auto& words, auto& error) {
                                          return m_Impl->MicropolygonPassNative(controls, patches, ranges,
                                                                                words, stopped, error);
                                      });
                                  RenderResult completion;
                                  completion.Status = result.Status;
                                  completion.Error = result.Error;
                                  return completion;
#else
                                  return Detail::MakeFailedResult("Micropolygons require native Compute");
#endif
                              })
                              .GetResult();
        result.Status = task.Status;
        result.Error = task.Error;
        return result;
    }

    TemporalResult ComputeShaderRayTracingRHI::ReconstructTemporal(
        const TemporalFrame& frame, const TemporalHistory* history, const TemporalOptions& options,
        const CancellationToken& cancellation) const
    {
        TemporalResult result;
        const auto task = m_Impl->ComputeQueue
                              .Submit([&](const CancellationToken& queued) {
                                  const auto stopped = Detail::CombineCancellation(queued, cancellation);
                                  if (stopped())
                                      return Detail::MakeCancelledResult();
                                  if (!m_Impl->Available)
                                      return Detail::MakeFailedResult("Compute is unavailable: " +
                                                                      m_Impl->AvailabilityError);
#if defined(_WIN32)
                                  result = Detail::ReconstructTemporal(
                                      BackendType::ComputeShader, frame, history, options, stopped,
                                      [&](const auto& controls, const auto& current, const auto& guides,
                                          const auto& previous, const auto& oldGuides, auto& output,
                                          auto& display, auto& error) {
                                          return m_Impl->ReconstructTemporalNative(
                                              controls, current, guides, previous, oldGuides, stopped, output,
                                              display, error);
                                      });
                                  RenderResult completion;
                                  completion.Status = result.Status;
                                  completion.Error = result.Error;
                                  return completion;
#else
                                  return Detail::MakeFailedResult(
                                      "Temporal reconstruction requires native D3D12 compute");
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

    GeometryGuideResult ComputeShaderRayTracingRHI::CaptureGeometryGuides(
        const Scene& scene, const Camera& camera, const RenderSettings& settings,
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
            m_Impl->ComputeQueue
                .Submit([&](const CancellationToken& queued) {
                    const auto stopped = Detail::CombineCancellation(queued, cancellation);
                    if (stopped())
                        return Detail::MakeCancelledResult();
                    if (!m_Impl->Available)
                        return Detail::MakeFailedResult("D3D12 compute is unavailable: " +
                                                        m_Impl->AvailabilityError);
#if defined(_WIN32)
                    result = m_Impl->CaptureGuidesNative(scene, camera, settings, stopped);
                    RenderResult completion;
                    completion.Status = result.Status;
                    completion.Error = result.Error;
                    return completion;
#else
                    return Detail::MakeFailedResult("Geometry guides require native D3D12 compute");
#endif
                })
                .GetResult();
        result.Status = task.Status;
        result.Error = task.Error;
        if (result.Status != RenderStatus::Completed)
            result.Image = {};
        return result;
    }

    RenderResult ComputeShaderRayTracingRHI::Render(
        const Scene& scene,
        const Camera& camera,
        const RenderSettings& settings,
        const CancellationToken& cancellation) const
    {
        return m_Impl->Submit(&scene, nullptr, camera, settings, cancellation).GetResult();
    }

    RayQueryResult ComputeShaderRayTracingRHI::QueryRays(
        const Scene& scene, const std::vector<Ray>& rays, AccelerationOptions acceleration,
        const CancellationToken& cancellation) const
    {
        return m_Impl->Query(scene, rays, acceleration, cancellation);
    }

    RenderJob ComputeShaderRayTracingRHI::RenderAsync(
        Scene scene,
        Camera camera,
        RenderSettings settings) const
    {
        return RenderAsync(std::make_shared<Scene>(std::move(scene)), camera, settings);
    }

    RenderJob ComputeShaderRayTracingRHI::RenderAsync(
        std::shared_ptr<const Scene> scene,
        Camera camera,
        RenderSettings settings) const
    {
        const Scene* pointer = scene.get();
        return m_Impl->Submit(pointer, std::move(scene), camera, settings);
    }

    void ComputeShaderRayTracingRHI::WaitIdle() const
    {
        m_Impl->ComputeQueue.WaitIdle();
    }
}
}
