#include "Renderer/SubsurfaceValidation.h"
#include "CudaRayTracingRHI.h"
#include "Renderer/Reconstruction.h"
#include "Renderer/AdaptiveReconstruction.h"
#include "Renderer/ParticipatingMedia.h"
#include "Renderer/SpectralSceneData.h"
#include "Renderer/GeometryGuides.h"
#include "Renderer/TemporalReconstruction.h"
#include "Renderer/MicropolygonPreparation.h"
#include <type_traits>

#include "Renderer/RayTracingBackendCommon.h"
#include "Backends/CUDA/KdTreeBuilder.h"
#include "Backends/CUDA/BihBuilder.h"
#include "Backends/CUDA/SpatialBvhBuilder.h"
#include "Backends/CUDA/BvhBuilder.h"
#include "Backends/CUDA/InstancingBuilder.h"
#include "Backends/CUDA/Limits.h"
#include "Backends/CUDA/BidirectionalLayout.h"
#include "Backends/CUDA/MetropolisLayout.h"
#include "Backends/CUDA/PhotonLayout.h"
#include "Backends/CUDA/ProgressivePhotonLayout.h"
#include "Backends/CUDA/VertexConnectionMergingLayout.h"
#include "Backends/CUDA/PointGiGeometry.h"
#include "Renderer/EmitterData.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32) && defined(ARVO_CUDA_KERNELS)
#  include "CudaShaderBytecode.h"
#  if !defined(NOMINMAX)
#    define NOMINMAX
#  endif
#  include <Windows.h>
#endif

namespace Arvo
{
namespace RayTracing
{
    namespace
    {
#if defined(_WIN32) && defined(ARVO_CUDA_KERNELS)
        constexpr std::uint32_t SphereBit = 0x80000000u;

        constexpr std::uint32_t ThreadsPerBlock = 256;
        constexpr std::uint32_t PixelChunkSize = 16384;
        constexpr std::uint32_t DirectionalLightType = 0;
        constexpr std::uint32_t PointLightType = 1;
        constexpr std::uint32_t RectAreaLightType = 2;
        constexpr std::size_t UploadChunkSize = 32u * 1024u * 1024u;
        constexpr float Pi = 3.14159265358979323846f;

        using CUdevice = int;
        using CUdeviceptr = unsigned long long;
        using CUresult = int;
        struct CUctx_st;
        struct CUmod_st;
        struct CUfunc_st;
        struct CUstream_st;
        using CUcontext = CUctx_st*;
        using CUmodule = CUmod_st*;
        using CUfunction = CUfunc_st*;
        using CUstream = CUstream_st*;
        constexpr CUresult CudaSuccess = 0;

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

        struct QueryRay
        {
            Vec4 OriginMinimum;
            Vec4 DirectionMaximum;
        };
        static_assert(sizeof(QueryRay) == 32 && sizeof(RayQueryHit) == 64
            && offsetof(RayQueryHit, Flags) == 44 && offsetof(RayQueryHit, GeometricNormal) == 48,
            "CUDA query buffer ABI mismatch.");

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

#include "RayTracer/Backends/CUDA/MaterialTypes.cuh"

        using PackedTexture = Detail::PackedRayTextureDescriptor;

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

        struct alignas(16) FrameConstants
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
            CUdeviceptr InstanceRecords = 0;
            CUdeviceptr InstanceNodes = 0;
            CUdeviceptr InstanceError = 0;
            float LensRadius = 0;
            float FocusDistance = 0;
            std::uint32_t SamplingSeed = 0;
            std::uint32_t SamplingPadding[3]{};
        };

        // Lens values occupy the former trailing padding, preserving instance-pointer offsets.
        static_assert(offsetof(FrameConstants, LensRadius) == 248 && offsetof(FrameConstants, FocusDistance) == 252,
            "CUDA lens constant ABI mismatch.");

        struct alignas(16) GroupStatistics
        {
            std::uint64_t PrimaryRays = 0;
            std::uint64_t ShadowRays = 0;
            std::uint64_t ReflectionRays = 0;
            std::uint64_t PrimitiveTests = 0;
        };

        static_assert(sizeof(Vec4) == 16, "CUDA Vec4 ABI mismatch.");
        static_assert(sizeof(PackedAccelerationNode) == 48, "CUDA acceleration node ABI mismatch.");
        static_assert(offsetof(PackedAccelerationNode, SplitPosition) == 32
            && offsetof(PackedAccelerationNode, AboveChild) == 40, "CUDA kd-tree field ABI mismatch.");
        static_assert(offsetof(PackedAccelerationNode, SecondPlane) == 44, "CUDA BIH field ABI mismatch.");
        static_assert(sizeof(PackedPrimitiveRef) == 4, "CUDA primitive ABI mismatch.");
        static_assert(sizeof(PackedTriangle) == 192, "CUDA triangle ABI mismatch.");
        static_assert(
            offsetof(PackedTriangle, TexCoord01) == 144
                && offsetof(PackedTriangle, TexCoord2) == 160
                && offsetof(PackedTriangle, MaterialIndex) == 176
                && offsetof(PackedTriangle, Flags) == 180,
            "CUDA triangle field ABI mismatch.");
        static_assert(sizeof(PackedSphere) == 32, "CUDA sphere ABI mismatch.");
        static_assert(sizeof(PackedMaterial) == 128, "CUDA material ABI mismatch.");
        static_assert(
            offsetof(PackedMaterial, AmbientShininess) == 16
                && offsetof(PackedMaterial, Specular) == 32
                && offsetof(PackedMaterial, Emission) == 48
                && offsetof(PackedMaterial, DiffuseTextureIndex) == 64,
            "CUDA material field ABI mismatch.");
        static_assert(sizeof(PackedTexture) == 16, "CUDA texture ABI mismatch.");
        static_assert(sizeof(PackedLight) == 64, "CUDA light ABI mismatch.");
        static_assert(
            offsetof(PackedLight, Type) == 12
                && offsetof(PackedLight, Direction) == 16
                && offsetof(PackedLight, Intensity) == 28
                && offsetof(PackedLight, Color) == 32
                && offsetof(PackedLight, CastShadows) == 44
                && offsetof(PackedLight, Width) == 48,
            "CUDA light field ABI mismatch.");
        static_assert(
            sizeof(FrameConstants) == 272
                && offsetof(FrameConstants, ShadowStrength) == 124,
            "CUDA constants ABI mismatch.");
        static_assert(offsetof(FrameConstants, FullWidth) == 160, "Tile constant ABI mismatch.");
        static_assert(offsetof(FrameConstants, Integrator) == 176, "Integrator constant ABI mismatch.");
        static_assert(offsetof(FrameConstants, AdaptiveEnabled) == 192, "Adaptive constant ABI mismatch.");
        static_assert(offsetof(FrameConstants, Acceleration) == 216, "Acceleration constant ABI mismatch.");
        static_assert(offsetof(FrameConstants, InstanceRecords) == 224 && offsetof(FrameConstants, InstanceError) == 240,
            "CUDA instance constants ABI mismatch.");
        static_assert(sizeof(GroupStatistics) == 32, "CUDA statistics ABI mismatch.");

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

        bool IsFinite(const Float2& value)
        {
            return std::isfinite(value.X) && std::isfinite(value.Y);
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
            std::vector<CudaInstances::Record> Instances;
            std::vector<CudaInstances::Node> InstanceNodes;
            std::vector<PackedAccelerationNode> Nodes;
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
            if (!ValidateCameraLens(camera.Lens, error) ||
                (validateMotionSchedule && !ValidateSubsurfaceScene(scene, settings, error)) ||
                !ValidateCameraMotionSupport(settings.Integrator, camera.Motion, error) ||
                (validateMotionSchedule && !ValidateSceneMotionSupport(settings.Integrator, scene, error)) ||
                !ValidateCameraSamplePlan(camera, settings, error))
                return false;
            if (!ValidateTileSettings(settings, error)) return false;
            if (!ValidateIntegrator(BackendType::CUDA, settings.Integrator, error)) return false;
            if (settings.Width == 0 || settings.Height == 0
                || settings.Width > 16384 || settings.Height > 16384)
            {
                error = "CUDA image dimensions must be in the range [1, 16384].";
                return false;
            }
            if (settings.SamplesPerPixel == 0 || settings.SamplesPerPixel > 4096
                || settings.MaxReflectionBounces > CudaLimits::MaximumReflectionBounces
                || !std::isfinite(settings.ShadowStrength)
                || settings.ShadowStrength < 0.0f
                || settings.ShadowStrength > 1.0f)
            {
                error = "CUDA sample, reflection, or shadow settings exceed their safety limits.";
                return false;
            }
            if (!IsFinite(camera.Position) || !IsFinite(camera.Target) || !IsFinite(camera.Up)
                || !std::isfinite(camera.VerticalFieldOfViewDegrees)
                || camera.VerticalFieldOfViewDegrees <= 1.0f
                || camera.VerticalFieldOfViewDegrees >= 179.0f)
            {
                error = "CUDA camera values are invalid.";
                return false;
            }
            const Vec3 cameraDirection = Subtract(ToVec3(camera.Target), ToVec3(camera.Position));
            const Vec3 cameraUp = ToVec3(camera.Up);
            const float cameraDirectionLength = Dot(cameraDirection, cameraDirection);
            const float cameraUpLength = Dot(cameraUp, cameraUp);
            if (!std::isfinite(cameraDirectionLength) || !std::isfinite(cameraUpLength)
                || cameraDirectionLength < 1.0e-10f || cameraUpLength < 1.0e-10f
                || Dot(Cross(NormalizeVector(cameraDirection), NormalizeVector(cameraUp)),
                       Cross(NormalizeVector(cameraDirection), NormalizeVector(cameraUp))) < 1.0e-10f)
            {
                error = "CUDA camera vectors do not form a valid frame.";
                return false;
            }
            if ((!scene.Triangles.empty() || !scene.Spheres.empty()) && scene.Materials.empty())
            {
                error = "CUDA geometry requires at least one material.";
                return false;
            }
            if (scene.Triangles.size() >= SphereBit || scene.Spheres.size() >= SphereBit
                || scene.Triangles.size() + scene.Spheres.size() >= SphereBit
                || scene.Materials.size() > std::numeric_limits<std::uint32_t>::max()
                || scene.Textures.size() > std::numeric_limits<std::uint32_t>::max()
                || scene.DirectionalLights.size() > std::numeric_limits<std::uint32_t>::max()
                || scene.PointLights.size() > std::numeric_limits<std::uint32_t>::max()
                || scene.RectAreaLights.size() > std::numeric_limits<std::uint32_t>::max()
                || static_cast<std::uint64_t>(scene.DirectionalLights.size())
                        + scene.PointLights.size()
                        + scene.RectAreaLights.size()
                    > std::numeric_limits<std::uint32_t>::max())
            {
                error = "CUDA scene counts exceed the 31-bit primitive layout.";
                return false;
            }
            std::uint64_t textureTexelCount = 0;
            for (const RayTexture& texture : scene.Textures)
            {
                if (!texture.IsValid()) continue;
                textureTexelCount +=
                    static_cast<std::uint64_t>(texture.Width) * texture.Height;
                if (textureTexelCount > std::numeric_limits<std::uint32_t>::max())
                {
                    error = "CUDA texture atlas exceeds its 32-bit texel layout.";
                    return false;
                }
            }
            if (!IsFinite(scene.BackgroundTop) || !IsFinite(scene.BackgroundBottom)
                || !std::isfinite(scene.AmbientIntensity) || scene.AmbientIntensity < 0.0f)
            {
                error = "CUDA background and ambient values are invalid.";
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
                    error = "CUDA material values must be finite.";
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
                    error = "CUDA material ambient and specular values must be non-negative.";
                    return false;
                }
            }
            if (!CudaInstances::Validate(scene, error)) return false;
            std::vector<const SceneGeometry*> geometries{&scene};
            for (const auto& geometry : scene.Geometries) geometries.push_back(geometry.get());
            for (const SceneGeometry* geometry : geometries)
            {
                for (const Triangle& triangle : geometry->Triangles)
                {
                    if (triangle.MaterialIndex >= scene.Materials.size()
                        || !IsFinite(triangle.Position0) || !IsFinite(triangle.Position1)
                        || !IsFinite(triangle.Position2) || !IsFinite(triangle.Normal0)
                        || !IsFinite(triangle.Normal1) || !IsFinite(triangle.Normal2)
                        || !IsFinite(triangle.Color0) || !IsFinite(triangle.Color1)
                        || !IsFinite(triangle.Color2) || !IsFinite(triangle.TexCoord0)
                        || !IsFinite(triangle.TexCoord1) || !IsFinite(triangle.TexCoord2))
                    {
                        error = "CUDA triangle data is invalid.";
                        return false;
                    }
                    const Vec3 edge1 = Subtract(ToVec3(triangle.Position1), ToVec3(triangle.Position0));
                    const Vec3 edge2 = Subtract(ToVec3(triangle.Position2), ToVec3(triangle.Position0));
                    const float edge1Length = Dot(edge1, edge1);
                    const float edge2Length = Dot(edge2, edge2);
                    const float areaSquared = Dot(Cross(edge1, edge2), Cross(edge1, edge2));
                    if (!std::isfinite(edge1Length) || !std::isfinite(edge2Length)
                        || !std::isfinite(areaSquared) || areaSquared < 1.0e-16f)
                    {
                        error = "CUDA does not accept degenerate or numerically unsafe triangles.";
                        return false;
                    }
                }
                for (const Sphere& sphere : geometry->Spheres)
                {
                    if (sphere.MaterialIndex >= scene.Materials.size() || !IsFinite(sphere.Center)
                        || !std::isfinite(sphere.Radius) || sphere.Radius <= 0.0f)
                    {
                        error = "CUDA sphere data is invalid.";
                        return false;
                    }
                }
            }
            for (const DirectionalLight& light : scene.DirectionalLights)
            {
                const Vec3 direction = ToVec3(light.DirectionToLight);
                if (!IsFinite(light.DirectionToLight) || !IsFinite(light.Color)
                    || !std::isfinite(light.Intensity) || light.Intensity < 0.0f
                    || !std::isfinite(Dot(direction, direction)) || Dot(direction, direction) < 1.0e-12f)
                {
                    error = "CUDA directional light data is invalid.";
                    return false;
                }
            }
            for (const PointLight& light : scene.PointLights)
            {
                if (!IsFinite(light.Position) || !IsFinite(light.Color)
                    || !std::isfinite(light.Intensity) || light.Intensity < 0.0f)
                {
                    error = "CUDA point light data is invalid.";
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
                    error = "CUDA rect area light data is invalid.";
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
                    error = "CUDA rect area light bounds are invalid.";
                    return false;
                }
            }
            return true;
        }

        class ScenePacker
        {
        public:
            explicit ScenePacker(const Detail::CancellationProbe& cancellation)
                : m_Cancellation(cancellation)
            {
            }

            bool Pack(const Scene& scene, AccelerationOptions acceleration, PackedScene& output, std::string& error)
            {
                output = {};
                if (!Detail::BuildRayTextureAtlas(scene, output.TextureAtlas, error)) return false;
                output.Materials.reserve(scene.Materials.size());
                output.Lights.reserve(scene.DirectionalLights.size() + scene.PointLights.size() + scene.RectAreaLights.size());
                for (const Material& source : scene.Materials)
                {
                    PackedMaterial material{};
                    material.AlbedoReflectivity = ToVec4(source.Albedo, source.Reflectivity);
                    material.AmbientShininess = ToVec4(source.AmbientColor, source.Shininess);
                    material.Specular = ToVec4(source.SpecularColor);
                    material.Emission = ToVec4(source.Emission);
                    material.DiffuseTextureIndex = source.DiffuseTextureIndex;
                    material.SampledEmitter = source.SampledEmitter ? 1u : 0u;
                    material.Model = static_cast<unsigned>(source.Model);
                    material.Optical = {source.MicrofacetSlope, source.ExteriorIndex, source.InteriorIndex, source.SubsurfaceMeanFreePath};
                    material.ConductorIndex = ToVec4(source.ConductorIndex, source.SubsurfaceAnisotropy);
                    material.ConductorExtinction = ToVec4(source.ConductorExtinction, source.SubsurfaceAlbedoScale);
                    output.Materials.push_back(material);
                }
                for (const DirectionalLight& source : scene.DirectionalLights)
                {
                    PackedLight light;
                    const Vec3 direction = NormalizeVector(ToVec3(source.DirectionToLight));
                    light.Type = DirectionalLightType;
                    light.Direction = direction;
                    light.Intensity = source.Intensity;
                    light.Color = ToVec3(source.Color);
                    light.CastShadows = source.CastShadows ? 1u : 0u;
                    output.Lights.push_back(light);
                }
                for (const PointLight& source : scene.PointLights)
                {
                    PackedLight light;
                    light.Position = ToVec3(source.Position);
                    light.Type = PointLightType;
                    light.Intensity = source.Intensity;
                    light.Color = ToVec3(source.Color);
                    light.CastShadows = source.CastShadows ? 1u : 0u;
                    output.Lights.push_back(light);
                }
                for (const RectAreaLight& source : scene.RectAreaLights)
                {
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
                CudaInstances::Hierarchy hierarchy;
                if (!CudaInstances::Build(scene, m_Cancellation, hierarchy, error)) return false;
                std::vector<CudaInstances::Record> locations(hierarchy.Objects.size());
                for (size_t object = 0; object < hierarchy.Objects.size(); ++object)
                {
                    if (IsCancelled()) return Cancel(error);
                    PackedScene local;
                    if (!PackGeometry(*hierarchy.Objects[object], acceleration, local, error)) return false;
                    const auto fits = [](size_t size, size_t added) {
                        return uint64_t(size) + added <= std::numeric_limits<uint32_t>::max();
                    };
                    if (!fits(output.Nodes.size(), local.Nodes.size())
                        || !fits(output.PrimitiveRefs.size(), local.PrimitiveRefs.size())
                        || !fits(output.Triangles.size(), local.Triangles.size())
                        || !fits(output.Spheres.size(), local.Spheres.size()))
                    {
                        error = "CUDA concatenated object buffers exceed 32-bit addressing.";
                        return false;
                    }
                    auto& location = locations[object];
                    location.NodeOffset = uint32_t(output.Nodes.size());
                    location.NodeCount = uint32_t(local.Nodes.size());
                    location.ReferenceOffset = uint32_t(output.PrimitiveRefs.size());
                    location.TriangleOffset = uint32_t(output.Triangles.size());
                    location.TriangleCount = uint32_t(local.Triangles.size());
                    location.SphereOffset = uint32_t(output.Spheres.size());
                    output.Nodes.insert(output.Nodes.end(), local.Nodes.begin(), local.Nodes.end());
                    output.PrimitiveRefs.insert(output.PrimitiveRefs.end(),
                        local.PrimitiveRefs.begin(), local.PrimitiveRefs.end());
                    output.Triangles.insert(output.Triangles.end(), local.Triangles.begin(), local.Triangles.end());
                    output.Spheres.insert(output.Spheres.end(), local.Spheres.begin(), local.Spheres.end());
                }
                for (auto& record : hierarchy.Records)
                {
                    if (IsCancelled()) return Cancel(error);
                    const auto& location = locations[record.ObjectIndex];
                    record.NodeOffset = location.NodeOffset;
                    record.NodeCount = location.NodeCount;
                    record.ReferenceOffset = location.ReferenceOffset;
                    record.TriangleOffset = location.TriangleOffset;
                    record.TriangleCount = location.TriangleCount;
                    record.SphereOffset = location.SphereOffset;
                }
                output.Instances = std::move(hierarchy.Records);
                output.InstanceNodes = std::move(hierarchy.Nodes);
                return !IsCancelled() || Cancel(error);
            }

            bool PackGeometry(const SceneGeometry& scene, AccelerationOptions acceleration, PackedScene& output, std::string& error)
            {
                m_Primitives.clear();
                m_Primitives.reserve(scene.Triangles.size() + scene.Spheres.size());
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
                    m_Primitives.push_back(primitive);
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
                    m_Primitives.push_back(primitive);
                }
                if (acceleration.Type == AccelerationType::Bvh
                    && acceleration.BvhPolicy == BvhBuildPolicy::SpatialSplits)
                {
                    constexpr size_t CancellationStride = 1024;
                    std::vector<CudaSpatial::Primitive> input;
                    input.reserve(m_Primitives.size());
                    for (size_t index = 0; index < m_Primitives.size(); ++index)
                    {
                        if (index % CancellationStride == 0 && IsCancelled()) return Cancel(error);
                        const BuildPrimitive& source = m_Primitives[index];
                        const Bounds& box = source.BoundingBox;
                        input.push_back({{box.Minimum.X, box.Minimum.Y, box.Minimum.Z},
                            {box.Maximum.X, box.Maximum.Y, box.Maximum.Z}, source.Reference.TypeAndIndex});
                    }
                    CudaSpatial::Tree tree;
                    if (!CudaSpatial::Build(input, m_Cancellation, tree, error)) return false;
                    output.Nodes.reserve(tree.Nodes.size());
                    for (size_t index = 0; index < tree.Nodes.size(); ++index)
                    {
                        if (index % CancellationStride == 0 && IsCancelled()) return Cancel(error);
                        const auto& node = tree.Nodes[index];
                        PackedAccelerationNode packed;
                        packed.MinX = node.Minimum[0];
                        packed.MinY = node.Minimum[1];
                        packed.MinZ = node.Minimum[2];
                        packed.MaxX = node.Maximum[0];
                        packed.MaxY = node.Maximum[1];
                        packed.MaxZ = node.Maximum[2];
                        packed.FirstOrEscape = node.FirstOrEscape;
                        packed.PrimitiveCount = node.PrimitiveCount;
                        output.Nodes.push_back(packed);
                    }
                    output.PrimitiveRefs.reserve(tree.References.size());
                    for (size_t index = 0; index < tree.References.size(); ++index)
                    {
                        if (index % CancellationStride == 0 && IsCancelled()) return Cancel(error);
                        output.PrimitiveRefs.push_back({tree.References[index]});
                    }
                }
                else if (acceleration.Type == AccelerationType::KdTree)
                {
                    std::vector<CudaKd::Primitive> input;
                    input.reserve(m_Primitives.size());
                    for (std::size_t index = 0; index < m_Primitives.size(); ++index)
                    {
                        if ((index & 4095u) == 0 && IsCancelled()) return Cancel(error);
                        const BuildPrimitive& primitive = m_Primitives[index];
                        const auto& bounds = primitive.BoundingBox;
                        input.push_back({{bounds.Minimum.X, bounds.Minimum.Y, bounds.Minimum.Z},
                            {bounds.Maximum.X, bounds.Maximum.Y, bounds.Maximum.Z},
                            primitive.Reference.TypeAndIndex});
                    }
                    std::vector<CudaKd::Node> nodes;
                    std::vector<std::uint32_t> references;
                    if (!CudaKd::Build(input, nodes, references, m_Cancellation, error)) return false;
                    output.Nodes.reserve(nodes.size());
                    for (std::size_t index = 0; index < nodes.size(); ++index)
                    {
                        if ((index & 4095u) == 0 && IsCancelled()) return Cancel(error);
                        const auto& node = nodes[index];
                        PackedAccelerationNode packed;
                        packed.MinX = node.Minimum[0];
                        packed.MinY = node.Minimum[1];
                        packed.MinZ = node.Minimum[2];
                        packed.MaxX = node.Maximum[0];
                        packed.MaxY = node.Maximum[1];
                        packed.MaxZ = node.Maximum[2];
                        packed.FirstOrEscape = node.FirstReference;
                        packed.PrimitiveCount = node.ReferenceCount;
                        packed.SplitPosition = node.SplitPosition;
                        packed.SplitAxis = node.SplitAxis;
                        packed.AboveChild = node.AboveChild;
                        output.Nodes.push_back(packed);
                    }
                    output.PrimitiveRefs.reserve(references.size());
                    for (std::size_t index = 0; index < references.size(); ++index)
                    {
                        if ((index & 4095u) == 0 && IsCancelled()) return Cancel(error);
                        output.PrimitiveRefs.push_back({references[index]});
                    }
                }
                else if (acceleration.Type == AccelerationType::Bih)
                {
                    std::vector<CudaBih::Primitive> objects;
                    objects.reserve(m_Primitives.size());
                    for (std::size_t index = 0; index < m_Primitives.size(); ++index)
                    {
                        if ((index & 4095u) == 0 && IsCancelled()) return Cancel(error);
                        const BuildPrimitive& primitive = m_Primitives[index];
                        const Bounds& bounds = primitive.BoundingBox;
                        objects.push_back({{bounds.Minimum.X, bounds.Minimum.Y, bounds.Minimum.Z},
                            {bounds.Maximum.X, bounds.Maximum.Y, bounds.Maximum.Z},
                            primitive.Reference.TypeAndIndex});
                    }
                    std::vector<CudaBih::Node> nodes;
                    std::vector<std::uint32_t> references;
                    if (!CudaBih::Build(objects, nodes, references, m_Cancellation, error)) return false;
                    output.Nodes.reserve(nodes.size());
                    for (std::size_t index = 0; index < nodes.size(); ++index)
                    {
                        if ((index & 4095u) == 0 && IsCancelled()) return Cancel(error);
                        const CudaBih::Node& node = nodes[index];
                        PackedAccelerationNode target;
                        target.MinX = node.Minimum[0];
                        target.MinY = node.Minimum[1];
                        target.MinZ = node.Minimum[2];
                        target.MaxX = node.Maximum[0];
                        target.MaxY = node.Maximum[1];
                        target.MaxZ = node.Maximum[2];
                        target.FirstOrEscape = node.FirstReference;
                        target.PrimitiveCount = node.ReferenceCount;
                        target.SplitAxis = node.Axis;
                        target.AboveChild = node.AboveChild;
                        target.SplitPosition = node.LeftMaximum;
                        target.SecondPlane = node.RightMinimum;
                        output.Nodes.push_back(target);
                    }
                    output.PrimitiveRefs.reserve(references.size());
                    for (std::size_t index = 0; index < references.size(); ++index)
                    {
                        if ((index & 4095u) == 0 && IsCancelled()) return Cancel(error);
                        output.PrimitiveRefs.push_back({references[index]});
                    }
                }
                else if (!m_Primitives.empty())
                {
                    constexpr size_t CancellationStride = 1024;
                    std::vector<CudaBvh::Primitive> input;
                    input.reserve(m_Primitives.size());
                    for (size_t index = 0; index < m_Primitives.size(); ++index)
                    {
                        if (index % CancellationStride == 0 && IsCancelled()) return Cancel(error);
                        const auto& primitive = m_Primitives[index];
                        const auto& box = primitive.BoundingBox;
                        const auto& center = primitive.Centroid;
                        input.push_back({{box.Minimum.X, box.Minimum.Y, box.Minimum.Z},
                            {box.Maximum.X, box.Maximum.Y, box.Maximum.Z}, {center.X, center.Y, center.Z},
                            primitive.Reference.TypeAndIndex});
                    }
                    CudaBvh::Tree tree;
                    if (!CudaBvh::Build(input, m_Cancellation, tree, error)) return false;
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
                return m_Cancellation && m_Cancellation();
            }

            bool Cancel(std::string& error) const
            {
                error = "CUDA scene packing was cancelled.";
                return false;
            }

            Detail::CancellationProbe m_Cancellation;
            std::vector<BuildPrimitive> m_Primitives;
        };

        struct DynamicModule
        {
            ~DynamicModule()
            {
                if (Handle) FreeLibrary(Handle);
            }

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

        struct CudaApi
        {
            using InitFunction = CUresult(WINAPI*)(unsigned int);
            using DriverGetVersionFunction = CUresult(WINAPI*)(int*);
            using DeviceGetCountFunction = CUresult(WINAPI*)(int*);
            using DeviceGetFunction = CUresult(WINAPI*)(CUdevice*, int);
            using DeviceGetNameFunction = CUresult(WINAPI*)(char*, int, CUdevice);
            using DeviceComputeCapabilityFunction = CUresult(WINAPI*)(int*, int*, CUdevice);
            using DevicePrimaryCtxRetainFunction = CUresult(WINAPI*)(CUcontext*, CUdevice);
            using DevicePrimaryCtxReleaseFunction = CUresult(WINAPI*)(CUdevice);
            using CtxSetCurrentFunction = CUresult(WINAPI*)(CUcontext);
            using StreamCreateFunction = CUresult(WINAPI*)(CUstream*, unsigned int);
            using StreamDestroyFunction = CUresult(WINAPI*)(CUstream);
            using StreamSynchronizeFunction = CUresult(WINAPI*)(CUstream);
            using ModuleLoadDataExFunction = CUresult(WINAPI*)(CUmodule*, const void*, unsigned int, int*, void**);
            using ModuleGetFunctionFunction = CUresult(WINAPI*)(CUfunction*, CUmodule, const char*);
            using ModuleUnloadFunction = CUresult(WINAPI*)(CUmodule);
            using MemGetInfoFunction = CUresult(WINAPI*)(std::size_t*, std::size_t*);
            using MemAllocFunction = CUresult(WINAPI*)(CUdeviceptr*, std::size_t);
            using MemFreeFunction = CUresult(WINAPI*)(CUdeviceptr);
            using MemcpyHtoDAsyncFunction = CUresult(WINAPI*)(CUdeviceptr, const void*, std::size_t, CUstream);
            using MemcpyDtoHAsyncFunction = CUresult(WINAPI*)(void*, CUdeviceptr, std::size_t, CUstream);
            using MemsetD8AsyncFunction = CUresult(WINAPI*)(CUdeviceptr, unsigned char, std::size_t, CUstream);
            using LaunchKernelFunction = CUresult(WINAPI*)(
                CUfunction, unsigned int, unsigned int, unsigned int,
                unsigned int, unsigned int, unsigned int, unsigned int, CUstream, void**, void**);
            using GetErrorNameFunction = CUresult(WINAPI*)(CUresult, const char**);
            using GetErrorStringFunction = CUresult(WINAPI*)(CUresult, const char**);

            bool Load(DynamicModule& module)
            {
                Init = module.Get<InitFunction>("cuInit");
                DriverGetVersion = module.Get<DriverGetVersionFunction>("cuDriverGetVersion");
                DeviceGetCount = module.Get<DeviceGetCountFunction>("cuDeviceGetCount");
                DeviceGet = module.Get<DeviceGetFunction>("cuDeviceGet");
                DeviceGetName = module.Get<DeviceGetNameFunction>("cuDeviceGetName");
                DeviceComputeCapability = module.Get<DeviceComputeCapabilityFunction>("cuDeviceComputeCapability");
                DevicePrimaryCtxRetain = module.Get<DevicePrimaryCtxRetainFunction>("cuDevicePrimaryCtxRetain");
                DevicePrimaryCtxRelease = module.Get<DevicePrimaryCtxReleaseFunction>("cuDevicePrimaryCtxRelease_v2");
                if (!DevicePrimaryCtxRelease)
                {
                    DevicePrimaryCtxRelease = module.Get<DevicePrimaryCtxReleaseFunction>("cuDevicePrimaryCtxRelease");
                }
                CtxSetCurrent = module.Get<CtxSetCurrentFunction>("cuCtxSetCurrent");
                StreamCreate = module.Get<StreamCreateFunction>("cuStreamCreate");
                StreamDestroy = module.Get<StreamDestroyFunction>("cuStreamDestroy_v2");
                StreamSynchronize = module.Get<StreamSynchronizeFunction>("cuStreamSynchronize");
                ModuleLoadDataEx = module.Get<ModuleLoadDataExFunction>("cuModuleLoadDataEx");
                ModuleGetFunction = module.Get<ModuleGetFunctionFunction>("cuModuleGetFunction");
                ModuleUnload = module.Get<ModuleUnloadFunction>("cuModuleUnload");
                MemGetInfo = module.Get<MemGetInfoFunction>("cuMemGetInfo_v2");
                MemAlloc = module.Get<MemAllocFunction>("cuMemAlloc_v2");
                MemFree = module.Get<MemFreeFunction>("cuMemFree_v2");
                MemcpyHtoDAsync = module.Get<MemcpyHtoDAsyncFunction>("cuMemcpyHtoDAsync_v2");
                MemcpyDtoHAsync = module.Get<MemcpyDtoHAsyncFunction>("cuMemcpyDtoHAsync_v2");
                MemsetD8Async = module.Get<MemsetD8AsyncFunction>("cuMemsetD8Async");
                LaunchKernel = module.Get<LaunchKernelFunction>("cuLaunchKernel");
                GetErrorName = module.Get<GetErrorNameFunction>("cuGetErrorName");
                GetErrorString = module.Get<GetErrorStringFunction>("cuGetErrorString");
                return Init && DriverGetVersion && DeviceGetCount && DeviceGet && DeviceGetName
                    && DeviceComputeCapability && DevicePrimaryCtxRetain && DevicePrimaryCtxRelease
                    && CtxSetCurrent && StreamCreate && StreamDestroy && StreamSynchronize
                    && ModuleLoadDataEx && ModuleGetFunction && ModuleUnload && MemGetInfo
                    && MemAlloc && MemFree && MemcpyHtoDAsync && MemcpyDtoHAsync
                    && MemsetD8Async && LaunchKernel && GetErrorName && GetErrorString;
            }

            std::string Error(const char* operation, CUresult result) const
            {
                const char* name = nullptr;
                const char* message = nullptr;
                if (GetErrorName) GetErrorName(result, &name);
                if (GetErrorString) GetErrorString(result, &message);
                std::string error = operation;
                error += " failed";
                if (name)
                {
                    error += " (";
                    error += name;
                    error += ')';
                }
                if (message)
                {
                    error += ": ";
                    error += message;
                }
                error += '.';
                return error;
            }

            InitFunction Init = nullptr;
            DriverGetVersionFunction DriverGetVersion = nullptr;
            DeviceGetCountFunction DeviceGetCount = nullptr;
            DeviceGetFunction DeviceGet = nullptr;
            DeviceGetNameFunction DeviceGetName = nullptr;
            DeviceComputeCapabilityFunction DeviceComputeCapability = nullptr;
            DevicePrimaryCtxRetainFunction DevicePrimaryCtxRetain = nullptr;
            DevicePrimaryCtxReleaseFunction DevicePrimaryCtxRelease = nullptr;
            CtxSetCurrentFunction CtxSetCurrent = nullptr;
            StreamCreateFunction StreamCreate = nullptr;
            StreamDestroyFunction StreamDestroy = nullptr;
            StreamSynchronizeFunction StreamSynchronize = nullptr;
            ModuleLoadDataExFunction ModuleLoadDataEx = nullptr;
            ModuleGetFunctionFunction ModuleGetFunction = nullptr;
            ModuleUnloadFunction ModuleUnload = nullptr;
            MemGetInfoFunction MemGetInfo = nullptr;
            MemAllocFunction MemAlloc = nullptr;
            MemFreeFunction MemFree = nullptr;
            MemcpyHtoDAsyncFunction MemcpyHtoDAsync = nullptr;
            MemcpyDtoHAsyncFunction MemcpyDtoHAsync = nullptr;
            MemsetD8AsyncFunction MemsetD8Async = nullptr;
            LaunchKernelFunction LaunchKernel = nullptr;
            GetErrorNameFunction GetErrorName = nullptr;
            GetErrorStringFunction GetErrorString = nullptr;
        };

        struct DeviceBuffer
        {
            CUdeviceptr Pointer = 0;
            std::size_t Size = 0;
        };
#endif
    }

    struct CudaRayTracingRHI::Impl
    {
        Impl()
        {
#if defined(_WIN32) && defined(ARVO_CUDA_KERNELS)
            Initialize();
#else
            AvailabilityError = "CUDA requires toolkit-compiled kernels and a supported Windows NVIDIA driver.";
#endif
        }

        ~Impl()
        {
            StreamQueue.WaitIdle();
#if defined(_WIN32) && defined(ARVO_CUDA_KERNELS)
            if (Context) Api.CtxSetCurrent(Context);
            ReleaseFrameBuffers();
            ReleaseSceneBuffers();
            if (WhittedModule) Api.ModuleUnload(WhittedModule);
            if (PathTracingModule) Api.ModuleUnload(PathTracingModule);
            if (DistributedRayModule) Api.ModuleUnload(DistributedRayModule);
            if (FilmModule) Api.ModuleUnload(FilmModule);
            if (BidirectionalModule) Api.ModuleUnload(BidirectionalModule);
            if (MetropolisModule) Api.ModuleUnload(MetropolisModule);
            if (PhotonModule) Api.ModuleUnload(PhotonModule);
            if (ProgressivePhotonModule) Api.ModuleUnload(ProgressivePhotonModule);
            if (VcmModule) Api.ModuleUnload(VcmModule);
            if (PointGiModule) Api.ModuleUnload(PointGiModule);
            if (SpatialDenoiseModule)
                Api.ModuleUnload(SpatialDenoiseModule);
            if (AdaptiveDenoiseModule)
                Api.ModuleUnload(AdaptiveDenoiseModule);
            if (GeometryGuidesModule)
                Api.ModuleUnload(GeometryGuidesModule);
            if (DicingModule)
                Api.ModuleUnload(DicingModule);
            if (TemporalDenoiseModule)
                Api.ModuleUnload(TemporalDenoiseModule);
            if (SpectralModule)
                Api.ModuleUnload(SpectralModule);
            if (Stream) Api.StreamDestroy(Stream);
            if (Context) Api.DevicePrimaryCtxRelease(Device);
#endif
        }

        RenderJob Submit(
            const Scene* scene,
            std::shared_ptr<const Scene> owner,
            Camera camera,
            RenderSettings settings,
            CancellationToken external = CancellationToken()) const
        {
            return StreamQueue.Submit(
                [this, scene, owner = std::move(owner), camera, settings, external]
                (const CancellationToken& queued) mutable
                {
                    if (!scene) return Detail::MakeFailedResult("The CUDA submission has no scene.");
                    const Detail::CancellationProbe cancellation =
                        Detail::CombineCancellation(queued, external);
                    if (cancellation()) return Detail::MakeCancelledResult();
                    if (!Available)
                    {
                        return Detail::MakeFailedResult(
                            std::string("CUDA is unavailable: ") + AvailabilityError);
                    }
#if defined(_WIN32) && defined(ARVO_CUDA_KERNELS)
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
                    return Detail::MakeFailedResult("CUDA is available only on Windows.");
#endif
                });
        }

#if defined(_WIN32) && defined(ARVO_CUDA_KERNELS)
        bool Check(CUresult result, const char* operation, std::string& error) const
        {
            if (result == CudaSuccess) return true;
            error = Api.Error(operation, result);
            return false;
        }

        bool Initialize()
        {
            if (!DriverModule.Load(L"nvcuda.dll"))
            {
                AvailabilityError = "nvcuda.dll could not be loaded.";
                return false;
            }
            if (!Api.Load(DriverModule))
            {
                AvailabilityError = "The NVIDIA driver does not export the required Driver API.";
                return false;
            }
            if (!Check(Api.Init(0), "cuInit", AvailabilityError)) return false;
            Api.DriverGetVersion(&DriverVersion);
            int deviceCount = 0;
            if (!Check(Api.DeviceGetCount(&deviceCount), "cuDeviceGetCount", AvailabilityError)) return false;
            for (int ordinal = 0; ordinal < deviceCount; ++ordinal)
            {
                CUdevice candidate = 0;
                int major = 0;
                int minor = 0;
                if (Api.DeviceGet(&candidate, ordinal) != CudaSuccess
                    || Api.DeviceComputeCapability(&major, &minor, candidate) != CudaSuccess)
                {
                    continue;
                }
                if (major > 7 || (major == 7 && minor >= 5))
                {
                    Device = candidate;
                    ComputeMajor = major;
                    ComputeMinor = minor;
                    break;
                }
            }
            if (ComputeMajor == 0)
            {
                AvailabilityError = "No CUDA device supports the embedded compute_75 PTX target.";
                return false;
            }
            char deviceName[256]{};
            Api.DeviceGetName(deviceName, static_cast<int>(sizeof(deviceName)), Device);
            DeviceName = deviceName;
            if (!Check(Api.DevicePrimaryCtxRetain(&Context, Device),
                    "cuDevicePrimaryCtxRetain", AvailabilityError)
                || !Check(Api.CtxSetCurrent(Context), "cuCtxSetCurrent", AvailabilityError)
                || !Check(Api.StreamCreate(&Stream, 1u), "cuStreamCreate", AvailabilityError)
                || !Check(Api.ModuleLoadDataEx(
                    &WhittedModule, Detail::CudaWhittedPtx, 0, nullptr, nullptr),
                    "cuModuleLoadDataEx(Whitted)", AvailabilityError)
                || !Check(Api.ModuleLoadDataEx(
                    &PathTracingModule, Detail::CudaPathTracingPtx, 0, nullptr, nullptr),
                    "cuModuleLoadDataEx(PathTracing)", AvailabilityError)
                || !Check(Api.ModuleLoadDataEx(
                    &DistributedRayModule, Detail::CudaDistributedRayPtx, 0, nullptr, nullptr),
                    "cuModuleLoadDataEx(DistributedRayTracing)", AvailabilityError)
                || !Check(Api.ModuleLoadDataEx(
                    &FilmModule, Detail::CudaFilmPtx, 0, nullptr, nullptr),
                    "cuModuleLoadDataEx(Film)", AvailabilityError)
                || !Check(Api.ModuleGetFunction(&WhittedFunction, WhittedModule, "TracePixelsKernel"),
                    "cuModuleGetFunction(Whitted)", AvailabilityError)
                || !Check(Api.ModuleGetFunction(&QueryFunction, WhittedModule, "QueryRaysKernel"),
                    "cuModuleGetFunction(QueryRays)", AvailabilityError)
                || !Check(Api.ModuleGetFunction(&PathTracingFunction, PathTracingModule, "TracePixelsKernel"),
                    "cuModuleGetFunction(PathTracing)", AvailabilityError)
                || !Check(Api.ModuleGetFunction(&DistributedRayFunction, DistributedRayModule, "TracePixelsKernel"),
                    "cuModuleGetFunction(DistributedRayTracing)", AvailabilityError)
                || !Check(Api.ModuleGetFunction(&ResolveFunction, FilmModule, "ResolvePixelsKernel"),
                    "cuModuleGetFunction(ResolvePixelsKernel)", AvailabilityError))
            {
                return false;
            }
            Available = true;
            return true;
        }

        void FreeBuffer(DeviceBuffer& buffer) const
        {
            if (buffer.Pointer) Api.MemFree(buffer.Pointer);
            buffer = {};
        }

        bool AllocateBuffer(DeviceBuffer& buffer, std::size_t size, std::string& error) const
        {
            FreeBuffer(buffer);
            if (size == 0) return true;
            if (!Check(Api.MemAlloc(&buffer.Pointer, size), "cuMemAlloc", error)) return false;
            buffer.Size = size;
            return true;
        }

        bool UploadBuffer(
            DeviceBuffer& buffer,
            const void* data,
            std::size_t size,
            const Detail::CancellationProbe& cancellation,
            std::string& error) const
        {
            if (!AllocateBuffer(buffer, size, error)) return false;
            const auto* bytes = static_cast<const std::uint8_t*>(data);
            for (std::size_t offset = 0; offset < size; offset += UploadChunkSize)
            {
                if (cancellation())
                {
                    error = "CUDA scene upload was cancelled.";
                    return false;
                }
                const std::size_t chunk = std::min(UploadChunkSize, size - offset);
                if (!Check(Api.MemcpyHtoDAsync(
                        buffer.Pointer + offset, bytes + offset, chunk, Stream),
                    "cuMemcpyHtoDAsync", error)
                    || !Check(Api.StreamSynchronize(Stream), "cuStreamSynchronize", error))
                {
                    return false;
                }
            }
            return true;
        }

        bool DownloadBuffer(
            void* data,
            CUdeviceptr source,
            std::size_t size,
            const Detail::CancellationProbe& cancellation,
            std::string& error) const
        {
            auto* bytes = static_cast<std::uint8_t*>(data);
            for (std::size_t offset = 0; offset < size; offset += UploadChunkSize)
            {
                if (cancellation())
                {
                    error = "CUDA readback was cancelled.";
                    return false;
                }
                const std::size_t chunk = std::min(UploadChunkSize, size - offset);
                if (!Check(Api.MemcpyDtoHAsync(
                        bytes + offset, source + offset, chunk, Stream),
                    "cuMemcpyDtoHAsync", error)
                    || !Check(Api.StreamSynchronize(Stream), "cuStreamSynchronize", error))
                {
                    return false;
                }
            }
            return true;
        }

        void ReleaseSceneBuffers() const
        {
            FreeBuffer(InstanceRecords);
            FreeBuffer(InstanceNodes);
            FreeBuffer(InstanceError);
            InstanceNodeCount = 0;
            FreeBuffer(Nodes);
            FreeBuffer(PrimitiveRefs);
            FreeBuffer(Triangles);
            FreeBuffer(Spheres);
            FreeBuffer(Materials);
            FreeBuffer(Lights);
            FreeBuffer(TextureDescriptors);
            FreeBuffer(TextureTexels);
            FreeBuffer(SpectralPalette);
            FreeBuffer(SpectralReferences);
            NodeCount = PrimitiveRefCount = TriangleCount = SphereCount = MaterialCount = LightCount = 0;
            TextureCount = TextureTexelCount = 0;
            CachedOwner.reset();
        }

        void ReleaseFrameBuffers() const
        {
            FreeBuffer(Constants);
            FreeBuffer(Accumulation);
            FreeBuffer(SampleCounts);
            FreeBuffer(SampleTargets);
            FreeBuffer(Output);
            FreeBuffer(Statistics);
            FramePixelCapacity = 0;
            StatisticsGroupCapacity = 0;
        }

        bool OwnerMatches(const std::shared_ptr<const Scene>& owner) const
        {
            if (!owner) return false;
            const std::shared_ptr<const Scene> cached = CachedOwner.lock();
            return cached && cached.get() == owner.get();
        }

        bool PrepareSpectralCoefficients(const SpectralSceneData::Request* input, uint32_t count,
                                         const Detail::CancellationProbe& cancelled, DeviceBuffer& output,
                                         std::string& error) const
        {
            namespace S = SpectralSceneData;
            constexpr uint32_t QuadratureCount = 256, Threads = 64, BatchCapacity = 8192;
            constexpr uint32_t ConvergedStatus = 0;
            if (count == 0)
            {
                error = "CUDA spectral preparation received an empty palette.";
                return false;
            }
            if (!SpectralModule &&
                !Check(Api.ModuleLoadDataEx(&SpectralModule, Detail::CudaSpectralPreparePtx, 0, nullptr,
                                            nullptr),
                       "Loading CUDA spectral preparation", error))
                return false;
            if ((!SpectralResponseFunction &&
                 !Check(Api.ModuleGetFunction(&SpectralResponseFunction, SpectralModule,
                                              "SpectralPrepareResponse"),
                        "Finding CUDA spectral quadrature", error)) ||
                (!SpectralPaletteFunction &&
                 !Check(Api.ModuleGetFunction(&SpectralPaletteFunction, SpectralModule,
                                              "SpectralPreparePalette"),
                        "Finding CUDA spectral fitting", error)))
                return false;
            struct Temporary
            {
                const Impl& Owner;
                DeviceBuffer Sources, Response, Coefficients;
                ~Temporary()
                {
                    Owner.FreeBuffer(Sources);
                    Owner.FreeBuffer(Response);
                    Owner.FreeBuffer(Coefficients);
                }
            } storage{*this};
            const size_t sourceBytes = size_t(count) * sizeof(S::Request);
            const size_t coefficientBytes = size_t(count) * sizeof(S::GpuSpectrum);
            const size_t responseBytes = QuadratureCount * sizeof(Vec4);
            size_t available = 0, total = 0;
            if (!Check(Api.MemGetInfo(&available, &total), "Querying CUDA spectral allocation budget", error))
                return false;
            if (sourceBytes + coefficientBytes + responseBytes > available - available / 10)
            {
                error = "CUDA spectral preparation exceeds the available device memory budget.";
                return false;
            }
            if (!UploadBuffer(storage.Sources, input, sourceBytes, cancelled, error) ||
                !AllocateBuffer(storage.Response, responseBytes, error) ||
                !AllocateBuffer(storage.Coefficients, coefficientBytes, error))
                return false;
            if (cancelled())
            {
                error = "CUDA spectral preparation cancelled.";
                return false;
            }
            void* responseArguments[]{&storage.Response.Pointer};
            if (!Check(Api.LaunchKernel(SpectralResponseFunction, (QuadratureCount + Threads - 1) / Threads,
                                        1, 1, Threads, 1, 1, 0, Stream, responseArguments, nullptr),
                       "Preparing CUDA spectral quadrature", error) ||
                !Check(Api.StreamSynchronize(Stream), "Waiting for CUDA spectral quadrature", error))
                return false;
            std::vector<S::GpuSpectrum> results(std::min(count, BatchCapacity));
            for (uint32_t first = 0; first < count;)
            {
                if (cancelled())
                {
                    error = "CUDA spectral fitting cancelled.";
                    return false;
                }
                uint32_t batch = std::min(BatchCapacity, count - first);
                CUdeviceptr sources = storage.Sources.Pointer + size_t(first) * sizeof(S::Request);
                CUdeviceptr destination =
                    storage.Coefficients.Pointer + size_t(first) * sizeof(S::GpuSpectrum);
                void* arguments[]{&sources, &storage.Response.Pointer, &destination, &batch};
                if (!Check(Api.LaunchKernel(SpectralPaletteFunction, (batch + Threads - 1) / Threads, 1, 1,
                                            Threads, 1, 1, 0, Stream, arguments, nullptr),
                           "Fitting CUDA spectral palette", error) ||
                    !Check(Api.StreamSynchronize(Stream), "Waiting for CUDA spectral palette", error))
                    return false;
                if (!DownloadBuffer(results.data(), destination, size_t(batch) * sizeof(S::GpuSpectrum),
                                    cancelled, error))
                    return false;
                for (uint32_t index = 0; index < batch; ++index)
                {
                    const auto& fit = results[index];
                    if (fit.Status != ConvergedStatus || !std::isfinite(fit.Quadratic) ||
                        !std::isfinite(fit.Linear) || !std::isfinite(fit.Offset) ||
                        !std::isfinite(fit.Scale) || !std::isfinite(fit.Constant) ||
                        !std::isfinite(fit.Error) || fit.Scale < 0 || fit.Constant < 0 || fit.Error < 0)
                    {
                        error = "CUDA spectral fit failed at entry " + std::to_string(first + index);
                        return false;
                    }
                }
                first += batch;
            }
            if (cancelled())
            {
                error = "CUDA spectral coefficient publication cancelled.";
                return false;
            }
            FreeBuffer(output);
            output = std::exchange(storage.Coefficients, {});
            return true;
        }

        bool UpdateSpectralMedium(const MediumOptions& medium, const Detail::CancellationProbe& cancelled,
                                  std::string& error) const
        {
            if (SpectralMedium == medium)
                return true;
            namespace S = SpectralSceneData;
            const S::Request sources[]{{medium.Enabled ? medium.Absorption : Float3(), S::Role::Nonnegative},
                                       {medium.Enabled ? medium.Scattering : Float3(), S::Role::Nonnegative}};
            DeviceBuffer candidate;
            if (!PrepareSpectralCoefficients(sources, uint32_t(std::size(sources)), cancelled, candidate,
                                             error))
                return false;
            std::array<S::GpuSpectrum, std::size(sources)> coefficients;
            const bool downloaded = DownloadBuffer(coefficients.data(), candidate.Pointer,
                                                   sizeof(coefficients), cancelled, error);
            FreeBuffer(candidate);
            if (!downloaded)
                return false;
            if (cancelled())
            {
                error = "CUDA spectral medium publication cancelled.";
                return false;
            }
            // These are device-fitted and verified coefficients. This tiny upload
            // updates only the two medium entries, leaving the scene palette intact.
            if (!Check(Api.MemcpyHtoDAsync(SpectralPalette.Pointer +
                                               size_t(SpectralSceneSourceCount) * sizeof(S::GpuSpectrum),
                                           coefficients.data(), sizeof(coefficients), Stream),
                       "Publishing CUDA spectral medium", error) ||
                !Check(Api.StreamSynchronize(Stream), "Waiting for CUDA spectral medium", error))
                return false;
            SpectralMedium = medium;
            return true;
        }

        bool EnsureScene(const Scene& scene, const std::shared_ptr<const Scene>& owner,
                         AccelerationOptions acceleration, const Detail::CancellationProbe& cancellation,
                         std::string& error, const MediumOptions* spectralMedium = nullptr) const
        {
            if (OwnerMatches(owner) && CachedAcceleration == acceleration &&
                (!spectralMedium || SpectralPalette.Pointer != 0))
                return !spectralMedium || UpdateSpectralMedium(*spectralMedium, cancellation, error);
            PackedScene packed;
            ScenePacker packer(cancellation);
            if (!packer.Pack(scene, acceleration, packed, error)) return false;
            struct SpectralCandidate
            {
                const Impl& Owner;
                SpectralSceneData::GpuScene Inputs;
                DeviceBuffer Palette, References;
                ~SpectralCandidate()
                {
                    Owner.FreeBuffer(Palette);
                    Owner.FreeBuffer(References);
                }
            } spectral{*this};
            if (spectralMedium)
            {
                if (CudaSpectralHost::BuildGpuInputs(scene, *spectralMedium, cancellation, spectral.Inputs,
                                                     error) != RenderStatus::Completed)
                    return false;
                for (auto& triangle : packed.Triangles)
                {
                    const auto& a = triangle.Color0;
                    const auto& b = triangle.Color1;
                    const auto& c = triangle.Color2;
                    const auto entry = spectral.Inputs.TriangleReferences.find(
                        {a.X, a.Y, a.Z, b.X, b.Y, b.Z, c.X, c.Y, c.Z});
                    if (entry == spectral.Inputs.TriangleReferences.end())
                    {
                        error = "CUDA packed triangle colors do not match the spectral scene.";
                        return false;
                    }
                    triangle.Padding0 = entry->second;
                }
                if (!PrepareSpectralCoefficients(spectral.Inputs.Sources.data(),
                                                 uint32_t(spectral.Inputs.Sources.size()), cancellation,
                                                 spectral.Palette, error) ||
                    !UploadBuffer(spectral.References, spectral.Inputs.References.data(),
                                  spectral.Inputs.References.size() * sizeof(uint32_t), cancellation, error))
                    return false;
            }
            if (cancellation())
            {
                error = "CUDA scene construction was cancelled.";
                return false;
            }
            const std::size_t requiredBytes = packed.Instances.size() * sizeof(CudaInstances::Record)
                + packed.InstanceNodes.size() * sizeof(CudaInstances::Node) + sizeof(uint32_t)
                + packed.Nodes.size() * sizeof(PackedAccelerationNode)
                + packed.PrimitiveRefs.size() * sizeof(PackedPrimitiveRef)
                + packed.Triangles.size() * sizeof(PackedTriangle)
                + packed.Spheres.size() * sizeof(PackedSphere)
                + packed.Materials.size() * sizeof(PackedMaterial)
                + packed.Lights.size() * sizeof(PackedLight)
                + packed.TextureAtlas.Descriptors.size() * sizeof(PackedTexture)
                + packed.TextureAtlas.Texels.size() * sizeof(std::uint32_t);
            std::size_t freeBytes = 0;
            std::size_t totalBytes = 0;
            if (!Check(Api.MemGetInfo(&freeBytes, &totalBytes), "cuMemGetInfo", error)) return false;
            if (requiredBytes > freeBytes - freeBytes / 10)
            {
                error = "CUDA scene buffers exceed 90 percent of currently free device memory.";
                return false;
            }

            ReleaseSceneBuffers();
            if (!UploadBuffer(InstanceRecords, packed.Instances.data(),
                    packed.Instances.size() * sizeof(CudaInstances::Record), cancellation, error)
                || !UploadBuffer(InstanceNodes, packed.InstanceNodes.data(),
                    packed.InstanceNodes.size() * sizeof(CudaInstances::Node), cancellation, error)
                || (!packed.InstanceNodes.empty() && !AllocateBuffer(InstanceError, sizeof(uint32_t), error))
                || !UploadBuffer(Nodes, packed.Nodes.data(),
                    packed.Nodes.size() * sizeof(PackedAccelerationNode), cancellation, error)
                || !UploadBuffer(PrimitiveRefs, packed.PrimitiveRefs.data(),
                    packed.PrimitiveRefs.size() * sizeof(PackedPrimitiveRef), cancellation, error)
                || !UploadBuffer(Triangles, packed.Triangles.data(),
                    packed.Triangles.size() * sizeof(PackedTriangle), cancellation, error)
                || !UploadBuffer(Spheres, packed.Spheres.data(),
                    packed.Spheres.size() * sizeof(PackedSphere), cancellation, error)
                || !UploadBuffer(Materials, packed.Materials.data(),
                    packed.Materials.size() * sizeof(PackedMaterial), cancellation, error)
                || !UploadBuffer(Lights, packed.Lights.data(),
                    packed.Lights.size() * sizeof(PackedLight), cancellation, error)
                || !UploadBuffer(TextureDescriptors, packed.TextureAtlas.Descriptors.data(),
                    packed.TextureAtlas.Descriptors.size() * sizeof(PackedTexture), cancellation, error)
                || !UploadBuffer(TextureTexels, packed.TextureAtlas.Texels.data(),
                    packed.TextureAtlas.Texels.size() * sizeof(std::uint32_t), cancellation, error))
            {
                ReleaseSceneBuffers();
                return false;
            }
            InstanceNodeCount = static_cast<uint32_t>(packed.InstanceNodes.size());
            NodeCount = static_cast<std::uint32_t>(packed.Nodes.size());
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
            SpectralPalette = std::exchange(spectral.Palette, {});
            SpectralReferences = std::exchange(spectral.References, {});
            SpectralSceneSourceCount = spectral.Inputs.SceneSourceCount;
            if (spectralMedium)
                SpectralMedium = *spectralMedium;
            return true;
        }

        bool EnsureFrameBuffers(std::uint32_t pixelCount, std::string& error) const
        {
            const std::uint32_t maximumGroups =
                (std::min(pixelCount, PixelChunkSize) + ThreadsPerBlock - 1) / ThreadsPerBlock;
            if (FramePixelCapacity >= pixelCount && StatisticsGroupCapacity >= maximumGroups) return true;
            ReleaseFrameBuffers();
            if (!AllocateBuffer(Constants, sizeof(FrameConstants), error)
                || !AllocateBuffer(Accumulation, static_cast<std::size_t>(pixelCount) * sizeof(Vec4), error)
                || !AllocateBuffer(SampleCounts,
                    static_cast<std::size_t>(pixelCount) * sizeof(std::uint32_t), error)
                || !AllocateBuffer(SampleTargets,
                    static_cast<std::size_t>(pixelCount) * sizeof(std::uint32_t), error)
                || !AllocateBuffer(Output,
                    static_cast<std::size_t>(pixelCount) * sizeof(std::uint32_t), error)
                || !AllocateBuffer(Statistics,
                    static_cast<std::size_t>(maximumGroups) * sizeof(GroupStatistics), error))
            {
                ReleaseFrameBuffers();
                return false;
            }
            FramePixelCapacity = pixelCount;
            StatisticsGroupCapacity = maximumGroups;
            return true;
        }

        bool UploadConstants(const FrameConstants& constants, std::string& error) const
        {
            if (InstanceNodeCount && !Check(Api.MemsetD8Async(InstanceError.Pointer, 0, sizeof(uint32_t), Stream),
                "cuMemsetD8Async(instance error)", error)) return false;
            return Check(Api.MemcpyHtoDAsync(
                Constants.Pointer, &constants, sizeof(constants), Stream),
                "cuMemcpyHtoDAsync(constants)", error);
        }

        void BindInstances(FrameConstants& constants) const
        {
            constants.InstanceNodeCount = InstanceNodeCount;
            constants.InstanceRecords = InstanceRecords.Pointer;
            constants.InstanceNodes = InstanceNodes.Pointer;
            constants.InstanceError = InstanceError.Pointer;
        }

        bool CheckInstanceError(std::string& error) const
        {
            if (!InstanceNodeCount) return true;
            uint32_t flags = 0;
            if (!Check(Api.MemcpyDtoHAsync(&flags, InstanceError.Pointer, sizeof(flags), Stream),
                    "cuMemcpyDtoHAsync(instance error)", error)
                || !Check(Api.StreamSynchronize(Stream), "cuStreamSynchronize(instance error)", error)) return false;
            if (flags != 0)
            {
                error = "CUDA instance ray cannot be represented in object-space float coordinates.";
                return false;
            }
            return true;
        }

        bool LaunchTrace(const FrameConstants& constants, MediumData medium,
                         MotionData::FrameExposure cameraExposure, std::vector<GroupStatistics>& hostStatistics,
                         std::string& error) const
        {
            if (!UploadConstants(constants, error)) return false;
            CUdeviceptr nodes = Nodes.Pointer;
            CUdeviceptr primitiveRefs = PrimitiveRefs.Pointer;
            CUdeviceptr triangles = Triangles.Pointer;
            CUdeviceptr spheres = Spheres.Pointer;
            CUdeviceptr materials = Materials.Pointer;
            CUdeviceptr lights = Lights.Pointer;
            CUdeviceptr textureDescriptors = TextureDescriptors.Pointer;
            CUdeviceptr textureTexels = TextureTexels.Pointer;
            CUdeviceptr frameConstants = Constants.Pointer;
            CUdeviceptr accumulation = Accumulation.Pointer;
            CUdeviceptr sampleCounts = SampleCounts.Pointer;
            CUdeviceptr statistics = Statistics.Pointer;
            CUdeviceptr sampleTargets = SampleTargets.Pointer;
            CUdeviceptr spectralPalette = SpectralPalette.Pointer;
            CUdeviceptr spectralReferences = SpectralReferences.Pointer;
            void* parameters[] = {&nodes,          &primitiveRefs, &triangles,          &spheres,
                                  &materials,      &lights,        &textureDescriptors, &textureTexels,
                                  &frameConstants, &accumulation,  &sampleCounts,       &statistics,
                                  &sampleTargets,  &medium,        &spectralPalette,    &spectralReferences,
                                  &cameraExposure};
            const std::uint32_t groups = constants.GroupCount;
            CUfunction traceFunction = nullptr;
            switch (static_cast<Algorithm>(constants.Integrator))
            {
            case Algorithm::RayTracing: traceFunction = WhittedFunction; break;
            case Algorithm::PathTracing: traceFunction = PathTracingFunction; break;
            case Algorithm::DistributedRayTracing:
            {
                traceFunction = DistributedRayFunction;
                // Exposure follows this kernel's ordinary transport arguments.
                constexpr size_t DistributedExposureArgument = 13;
                parameters[DistributedExposureArgument] = &cameraExposure;
                break;
            }
            default: error = "No native CUDA kernel for the integrator"; return false;
            }
            if (!Check(Api.LaunchKernel(
                    traceFunction, groups, 1, 1, ThreadsPerBlock, 1, 1, 0,
                    Stream, parameters, nullptr),
                "cuLaunchKernel(TracePixelsKernel)", error))
            {
                return false;
            }
            hostStatistics.resize(groups);
            if (!Check(Api.MemcpyDtoHAsync(
                    hostStatistics.data(), Statistics.Pointer,
                    static_cast<std::size_t>(groups) * sizeof(GroupStatistics), Stream),
                "cuMemcpyDtoHAsync(statistics)", error)
                || !Check(Api.StreamSynchronize(Stream), "cuStreamSynchronize(trace)", error))
            {
                return false;
            }
            return CheckInstanceError(error);
        }

        bool LaunchResolve(
            FrameConstants constants,
            std::uint32_t pixelCount,
            const Detail::CancellationProbe& cancellation,
            std::string& error) const
        {
            for (std::uint32_t pixelBegin = 0; pixelBegin < pixelCount; pixelBegin += PixelChunkSize)
            {
                if (cancellation())
                {
                    error = "CUDA resolve was cancelled.";
                    return false;
                }
                constants.PixelBegin = pixelBegin;
                constants.PixelCount = std::min(PixelChunkSize, pixelCount - pixelBegin);
                constants.GroupCount =
                    (constants.PixelCount + ThreadsPerBlock - 1) / ThreadsPerBlock;
                if (!UploadConstants(constants, error)) return false;
                CUdeviceptr frameConstants = Constants.Pointer;
                CUdeviceptr accumulation = Accumulation.Pointer;
                CUdeviceptr sampleCounts = SampleCounts.Pointer;
                CUdeviceptr output = Output.Pointer;
                void* parameters[] = {&frameConstants, &accumulation, &sampleCounts, &output};
                if (!Check(Api.LaunchKernel(
                        ResolveFunction, constants.GroupCount, 1, 1, ThreadsPerBlock, 1, 1, 0,
                        Stream, parameters, nullptr),
                    "cuLaunchKernel(ResolvePixelsKernel)", error)
                    || !Check(Api.StreamSynchronize(Stream), "cuStreamSynchronize(resolve)", error))
                {
                    return false;
                }
            }
            return true;
        }

        RayQueryResult QueryNative(const Scene& scene, const std::vector<Ray>& rays, AccelerationOptions acceleration,
            const Detail::CancellationProbe& cancellation) const
        {
            RayQueryResult result;
            if (!ValidateAcceleration(acceleration, result.Error)
                || !ValidateRayQueries(rays, result.Error)
                || !ValidateInput(scene, Camera{}, RenderSettings{}, result.Error, false)) return result;
            if (cancellation()) { result.Status = RenderStatus::Cancelled; return result; }
            if (rays.empty()) { result.Status = RenderStatus::Completed; return result; }
            if (!Check(Api.CtxSetCurrent(Context), "cuCtxSetCurrent(query)", result.Error)) return result;
            if (!EnsureScene(scene, {}, acceleration, cancellation, result.Error))
            {
                if (cancellation()) result.Status = RenderStatus::Cancelled;
                return result;
            }
            struct QueryBuffers
            {
                const Impl& Backend;
                DeviceBuffer Input, Output;
                ~QueryBuffers() { Backend.FreeBuffer(Input); Backend.FreeBuffer(Output); }
            } buffers{*this};
            std::vector<QueryRay> input;
            input.reserve(rays.size());
            for (const Ray& ray : rays)
                input.push_back({{ray.Origin.X, ray.Origin.Y, ray.Origin.Z, ray.MinDistance},
                    {ray.Direction.X, ray.Direction.Y, ray.Direction.Z, ray.MaxDistance}});
            if (!UploadBuffer(buffers.Input, input.data(), input.size() * sizeof(QueryRay), cancellation, result.Error)
                || !AllocateBuffer(buffers.Output, input.size() * sizeof(RayQueryHit), result.Error)
                || (Constants.Size != sizeof(FrameConstants) && !AllocateBuffer(Constants, sizeof(FrameConstants), result.Error)))
            {
                if (cancellation()) result.Status = RenderStatus::Cancelled;
                return result;
            }
            FrameConstants constants;
            BindInstances(constants);
            constants.Acceleration = static_cast<std::uint32_t>(acceleration.Type);
            constants.NodeCount = NodeCount;
            constants.TriangleCount = TriangleCount;
            constants.SphereCount = SphereCount;
            constants.PrimitiveRefCount = PrimitiveRefCount;
            for (std::uint32_t begin = 0; begin < input.size(); begin += PixelChunkSize)
            {
                if (cancellation()) { result.Status = RenderStatus::Cancelled; return result; }
                constants.PixelBegin = begin;
                constants.PixelCount = std::min(PixelChunkSize, static_cast<std::uint32_t>(input.size()) - begin);
                if (!UploadConstants(constants, result.Error)) return result;
                CUdeviceptr nodes = Nodes.Pointer, references = PrimitiveRefs.Pointer;
                CUdeviceptr triangles = Triangles.Pointer, spheres = Spheres.Pointer;
                CUdeviceptr globals = Constants.Pointer, queryInput = buffers.Input.Pointer, queryOutput = buffers.Output.Pointer;
                void* parameters[] = {&nodes, &references, &triangles, &spheres, &globals, &queryInput, &queryOutput};
                const std::uint32_t groups = (constants.PixelCount + ThreadsPerBlock - 1) / ThreadsPerBlock;
                if (!Check(Api.LaunchKernel(QueryFunction, groups, 1, 1, ThreadsPerBlock, 1, 1, 0,
                        Stream, parameters, nullptr), "cuLaunchKernel(QueryRays)", result.Error)
                    || !Check(Api.StreamSynchronize(Stream), "cuStreamSynchronize(query)", result.Error)
                    || !CheckInstanceError(result.Error)) return result;
            }
            result.Hits.resize(rays.size());
            if (!DownloadBuffer(result.Hits.data(), buffers.Output.Pointer, result.Hits.size() * sizeof(RayQueryHit),
                cancellation, result.Error))
            {
                if (cancellation()) result.Status = RenderStatus::Cancelled;
                return result;
            }
            result.Status = cancellation() ? RenderStatus::Cancelled : RenderStatus::Completed;
            return result;
        }

        bool RenderBidirectional(const Scene& scene, FrameConstants constants, const RenderSettings& settings,
                                 const Detail::CancellationProbe& cancellation, RenderResult& result) const
        {
            if (settings.Adaptive.Enabled || settings.Transport.RussianRoulette ||
                !settings.Transport.NextEventEstimation || !settings.ShadowsEnabled ||
                settings.ShadowStrength != 1)
            {
                result.Error = "CUDA BDPT requires fixed samples/depth, NEE and full physical visibility.";
                return false;
            }
            const auto emitters = CudaEmissionHost::PrepareEmitters(scene);
            for (const auto& emitter : emitters.Sources)
                if (!(emitter.Flags & EmitterCastsShadows))
                {
                    result.Error = "CUDA BDPT requires shadow casting on every emitter.";
                    return false;
                }
            // Compile/load this independent program only when BDPT is selected.
            // A module failure does not invalidate the other CUDA integrators.
            if (!BidirectionalModule &&
                !Check(Api.ModuleLoadDataEx(&BidirectionalModule, Detail::CudaBidirectionalPtx, 0, nullptr,
                                            nullptr),
                       "cuModuleLoadDataEx(BDPT)", result.Error))
                return false;
            if (!Check(Api.ModuleGetFunction(&BidirectionalPairs, BidirectionalModule,
                                             "BidirectionalPairsKernel"),
                       "cuModuleGetFunction(BidirectionalPairs)", result.Error) ||
                !Check(Api.ModuleGetFunction(&BidirectionalCommit, BidirectionalModule,
                                             "BidirectionalCommitKernel"),
                       "cuModuleGetFunction(BidirectionalCommit)", result.Error))
                return false;
            struct Buffers
            {
                const Impl& Owner;
                DeviceBuffer Emitters, Controls, Scratch, Pass, Statistics, Error;
                ~Buffers()
                {
                    // Also retire partially queued work on a launch/readback failure.
                    Owner.Api.StreamSynchronize(Owner.Stream);
                    Owner.FreeBuffer(Emitters);
                    Owner.FreeBuffer(Controls);
                    Owner.FreeBuffer(Scratch);
                    Owner.FreeBuffer(Pass);
                    Owner.FreeBuffer(Statistics);
                    Owner.FreeBuffer(Error);
                }
            } buffers{*this};
            const uint32_t fullPixels = constants.FullWidth * constants.FullHeight;
            const uint32_t pixels = constants.Width * constants.Height;
            const uint32_t capacity = std::min(CudaBdpt::BatchCapacity, fullPixels);
            if (!UploadBuffer(buffers.Emitters, emitters.Sources.data(),
                              emitters.Sources.size() * sizeof(PackedEmitter), cancellation, result.Error) ||
                !AllocateBuffer(buffers.Controls, sizeof(CudaBdpt::Controls), result.Error) ||
                !AllocateBuffer(buffers.Scratch, size_t(capacity) * sizeof(CudaBdpt::Pair), result.Error) ||
                !AllocateBuffer(buffers.Pass, size_t(pixels) * CudaBdpt::PassPixelBytes, result.Error) ||
                !AllocateBuffer(buffers.Statistics, size_t(capacity) * sizeof(GroupStatistics),
                                result.Error) ||
                !AllocateBuffer(buffers.Error, sizeof(uint32_t), result.Error))
                return false;
            CudaBdpt::Controls controls{{emitters.Center.X, emitters.Center.Y, emitters.Center.Z},
                                        emitters.Radius,
                                        uint32_t(emitters.Sources.size()),
                                        0,
                                        0,
                                        0};
            std::vector<GroupStatistics> statistics(capacity);
            std::vector<uint32_t> targets(pixels, constants.TotalSamples);
            struct HostTransferLifetime
            {
                const Impl& Owner;
                ~HostTransferLifetime() { Owner.Api.StreamSynchronize(Owner.Stream); }
            } retireTransfers{*this};
            if (!Check(Api.MemcpyHtoDAsync(SampleTargets.Pointer, targets.data(),
                                           size_t(pixels) * sizeof(uint32_t), Stream),
                       "cuMemcpyHtoDAsync(BDPT targets)", result.Error))
                return false;
            void* pairArguments[] = {&Nodes.Pointer,
                                     &PrimitiveRefs.Pointer,
                                     &Triangles.Pointer,
                                     &Spheres.Pointer,
                                     &Materials.Pointer,
                                     &TextureDescriptors.Pointer,
                                     &TextureTexels.Pointer,
                                     &buffers.Emitters.Pointer,
                                     &Constants.Pointer,
                                     &buffers.Controls.Pointer,
                                     &buffers.Scratch.Pointer,
                                     &buffers.Pass.Pointer,
                                     &buffers.Statistics.Pointer,
                                     &buffers.Error.Pointer};
            void* commitArguments[] = {&Constants.Pointer, &buffers.Pass.Pointer, &Accumulation.Pointer,
                                       &SampleCounts.Pointer, &buffers.Error.Pointer};
            for (uint32_t sample = 0; sample < constants.TotalSamples; ++sample)
            {
                if (cancellation())
                    return false;
                constants.SampleBegin = sample;
                if (!UploadConstants(constants, result.Error) ||
                    !Check(Api.MemsetD8Async(buffers.Pass.Pointer, 0, buffers.Pass.Size, Stream),
                           "cuMemsetD8Async(BDPT pass)", result.Error) ||
                    !Check(Api.MemsetD8Async(buffers.Error.Pointer, 0, buffers.Error.Size, Stream),
                           "cuMemsetD8Async(BDPT error)", result.Error))
                    return false;
                for (uint32_t begin = 0; begin < fullPixels; begin += capacity)
                {
                    if (cancellation())
                        return false;
                    controls.PathBegin = begin;
                    controls.PathCount = std::min(capacity, fullPixels - begin);
                    const uint32_t groups =
                        (controls.PathCount + CudaBdpt::ThreadsPerBlock - 1) / CudaBdpt::ThreadsPerBlock;
                    uint32_t flags = 0;
                    if (!Check(Api.MemcpyHtoDAsync(buffers.Controls.Pointer, &controls, sizeof(controls),
                                                   Stream),
                               "cuMemcpyHtoDAsync(BDPT controls)", result.Error) ||
                        !Check(Api.LaunchKernel(BidirectionalPairs, groups, 1, 1, CudaBdpt::ThreadsPerBlock,
                                                1, 1, 0, Stream, pairArguments, nullptr),
                               "cuLaunchKernel(BDPT pairs)", result.Error) ||
                        !Check(Api.MemcpyDtoHAsync(statistics.data(), buffers.Statistics.Pointer,
                                                   size_t(controls.PathCount) * sizeof(GroupStatistics),
                                                   Stream),
                               "cuMemcpyDtoHAsync(BDPT statistics)", result.Error) ||
                        !Check(Api.MemcpyDtoHAsync(&flags, buffers.Error.Pointer, sizeof(flags), Stream),
                               "cuMemcpyDtoHAsync(BDPT error)", result.Error) ||
                        !Check(Api.StreamSynchronize(Stream), "cuStreamSynchronize(BDPT pairs)",
                               result.Error) ||
                        !CheckInstanceError(result.Error))
                        return false;
                    if (flags)
                    {
                        result.Error = "CUDA BDPT encountered nonfinite path arithmetic.";
                        return false;
                    }
                    for (uint32_t i = 0; i < controls.PathCount; ++i)
                    {
                        result.Statistics.PrimaryRays += statistics[i].PrimaryRays;
                        result.Statistics.ShadowRays += statistics[i].ShadowRays;
                        result.Statistics.ReflectionRays += statistics[i].ReflectionRays;
                        result.Statistics.PrimitiveTests += statistics[i].PrimitiveTests;
                    }
                }
                if (cancellation())
                    return false;
                uint32_t flags = 0;
                const uint32_t groups = (pixels + CudaBdpt::ThreadsPerBlock - 1) / CudaBdpt::ThreadsPerBlock;
                if (!Check(Api.LaunchKernel(BidirectionalCommit, groups, 1, 1, CudaBdpt::ThreadsPerBlock, 1,
                                            1, 0, Stream, commitArguments, nullptr),
                           "cuLaunchKernel(BDPT commit)", result.Error) ||
                    !Check(Api.MemcpyDtoHAsync(&flags, buffers.Error.Pointer, sizeof(flags), Stream),
                           "cuMemcpyDtoHAsync(BDPT film error)", result.Error) ||
                    !Check(Api.StreamSynchronize(Stream), "cuStreamSynchronize(BDPT commit)", result.Error))
                    return false;
                if (flags)
                {
                    result.Error = "CUDA BDPT sample pass overflowed the film.";
                    return false;
                }
            }
            return true;
        }

        bool RenderMetropolis(const Scene& scene, const FrameConstants& constants,
                              const RenderSettings& settings, const Detail::CancellationProbe& cancellation,
                              RenderResult& result) const
        {
            if (settings.MaxReflectionBounces > CudaMlt::MaximumReflections || settings.Adaptive.Enabled ||
                !settings.ShadowsEnabled || settings.ShadowStrength != 1)
            {
                result.Error =
                    "CUDA MLT requires bounded depth, fixed mutation budgets and full physical visibility.";
                return false;
            }
            const auto emitters = CudaEmissionHost::PrepareEmitters(scene);
            for (const auto& source : emitters.Sources)
                if (!(source.Flags & EmitterCastsShadows))
                {
                    result.Error = "CUDA MLT requires shadow casting on every emitter.";
                    return false;
                }
            if (!MetropolisModule && !Check(Api.ModuleLoadDataEx(&MetropolisModule, Detail::CudaMetropolisPtx,
                                                                 0, nullptr, nullptr),
                                            "cuModuleLoadDataEx(MLT)", result.Error))
                return false;
            CUfunction bootstrapKernel{}, distributionKernel{}, initializeKernel{}, mutateKernel{},
                commitKernel{};
            if (!Check(Api.ModuleGetFunction(&bootstrapKernel, MetropolisModule, "MetropolisBootstrapKernel"),
                       "cuModuleGetFunction(MLT bootstrap)", result.Error) ||
                !Check(Api.ModuleGetFunction(&distributionKernel, MetropolisModule,
                                             "MetropolisDistributionKernel"),
                       "cuModuleGetFunction(MLT distribution)", result.Error) ||
                !Check(
                    Api.ModuleGetFunction(&initializeKernel, MetropolisModule, "MetropolisInitializeKernel"),
                    "cuModuleGetFunction(MLT initialization)", result.Error) ||
                !Check(Api.ModuleGetFunction(&mutateKernel, MetropolisModule, "MetropolisMutateKernel"),
                       "cuModuleGetFunction(MLT mutation)", result.Error) ||
                !Check(Api.ModuleGetFunction(&commitKernel, MetropolisModule, "MetropolisCommitKernel"),
                       "cuModuleGetFunction(MLT film)", result.Error))
                return false;
            struct Buffers
            {
                const Impl& Owner;
                DeviceBuffer Emitters, Controls, Bootstrap, Cdf, Distribution, Chains, Pass, Statistics,
                    Error;
                ~Buffers()
                {
                    Owner.Api.StreamSynchronize(Owner.Stream);
                    for (auto* buffer : {&Emitters, &Controls, &Bootstrap, &Cdf, &Distribution, &Chains,
                                         &Pass, &Statistics, &Error})
                        Owner.FreeBuffer(*buffer);
                }
            } buffers{*this};
            const auto& options = settings.Metropolis;
            const uint32_t pixels = constants.Width * constants.Height;
            const uint32_t fullPixels = constants.FullWidth * constants.FullHeight;
            const uint32_t chains = std::min(options.ChainCount, fullPixels);
            const uint32_t statisticsCapacity = std::max(chains, CudaMlt::BootstrapBatch);
            if (!UploadBuffer(buffers.Emitters, emitters.Sources.data(),
                              emitters.Sources.size() * sizeof(PackedEmitter), cancellation, result.Error) ||
                !AllocateBuffer(buffers.Controls, sizeof(CudaMlt::Controls), result.Error) ||
                !AllocateBuffer(buffers.Bootstrap,
                                size_t(options.BootstrapSamples) * sizeof(CudaMlt::PathValue),
                                result.Error) ||
                !AllocateBuffer(buffers.Cdf, size_t(options.BootstrapSamples) * sizeof(double),
                                result.Error) ||
                !AllocateBuffer(buffers.Distribution, sizeof(CudaMlt::Distribution), result.Error) ||
                !AllocateBuffer(buffers.Chains, size_t(chains) * sizeof(CudaMlt::Chain), result.Error) ||
                !AllocateBuffer(buffers.Pass, size_t(pixels) * sizeof(CudaMlt::PassPixel), result.Error) ||
                !AllocateBuffer(buffers.Statistics, size_t(statisticsCapacity) * sizeof(GroupStatistics),
                                result.Error) ||
                !AllocateBuffer(buffers.Error, sizeof(uint32_t), result.Error))
                return false;
            CudaMlt::Controls controls{{emitters.Center.X, emitters.Center.Y, emitters.Center.Z},
                                       emitters.Radius,
                                       uint32_t(emitters.Sources.size()),
                                       options.Seed ^ settings.SamplingSeed,
                                       options.BootstrapSamples,
                                       chains,
                                       0,
                                       0,
                                       options.LargeStepProbability,
                                       options.MutationScale};
            std::vector<GroupStatistics> statistics(statisticsCapacity);
            std::vector<uint32_t> targets(pixels, constants.TotalSamples);
            uint32_t flags = 0;
            // Keep every asynchronous host operand alive even when a later API fails.
            struct TransferLifetime
            {
                const Impl& Owner;
                ~TransferLifetime() { Owner.Api.StreamSynchronize(Owner.Stream); }
            } retire{*this};
            if (!UploadConstants(constants, result.Error) ||
                !Check(Api.MemcpyHtoDAsync(SampleTargets.Pointer, targets.data(),
                                           targets.size() * sizeof(uint32_t), Stream),
                       "cuMemcpyHtoDAsync(MLT targets)", result.Error) ||
                !Check(Api.MemsetD8Async(buffers.Error.Pointer, 0, buffers.Error.Size, Stream),
                       "cuMemsetD8Async(MLT errors)", result.Error))
                return false;
            void* bootstrapArguments[] = {&Nodes.Pointer,
                                          &PrimitiveRefs.Pointer,
                                          &Triangles.Pointer,
                                          &Spheres.Pointer,
                                          &Materials.Pointer,
                                          &TextureDescriptors.Pointer,
                                          &TextureTexels.Pointer,
                                          &buffers.Emitters.Pointer,
                                          &Constants.Pointer,
                                          &buffers.Controls.Pointer,
                                          &buffers.Bootstrap.Pointer,
                                          &buffers.Statistics.Pointer,
                                          &buffers.Error.Pointer};
            void* distributionArguments[] = {&buffers.Controls.Pointer, &buffers.Bootstrap.Pointer,
                                             &buffers.Cdf.Pointer, &buffers.Distribution.Pointer,
                                             &buffers.Error.Pointer};
            void* initializeArguments[] = {&buffers.Controls.Pointer, &buffers.Bootstrap.Pointer,
                                           &buffers.Cdf.Pointer, &buffers.Distribution.Pointer,
                                           &buffers.Chains.Pointer};
            void* mutationArguments[] = {&Nodes.Pointer,
                                         &PrimitiveRefs.Pointer,
                                         &Triangles.Pointer,
                                         &Spheres.Pointer,
                                         &Materials.Pointer,
                                         &TextureDescriptors.Pointer,
                                         &TextureTexels.Pointer,
                                         &buffers.Emitters.Pointer,
                                         &Constants.Pointer,
                                         &buffers.Controls.Pointer,
                                         &buffers.Distribution.Pointer,
                                         &buffers.Chains.Pointer,
                                         &buffers.Pass.Pointer,
                                         &buffers.Statistics.Pointer,
                                         &buffers.Error.Pointer};
            void* commitArguments[] = {&Constants.Pointer, &buffers.Pass.Pointer, &Accumulation.Pointer,
                                       &SampleCounts.Pointer, &buffers.Error.Pointer};
            const auto dispatch = [&](CUfunction kernel, uint32_t lanes, void** arguments,
                                      uint32_t statisticsCount) {
                if (cancellation())
                    return false;
                uint32_t groups = (lanes + CudaMlt::ThreadsPerBlock - 1) / CudaMlt::ThreadsPerBlock;
                if (!Check(Api.MemcpyHtoDAsync(buffers.Controls.Pointer, &controls, sizeof(controls), Stream),
                           "cuMemcpyHtoDAsync(MLT controls)", result.Error) ||
                    !Check(Api.LaunchKernel(kernel, groups, 1, 1, CudaMlt::ThreadsPerBlock, 1, 1, 0, Stream,
                                            arguments, nullptr),
                           "cuLaunchKernel(MLT)", result.Error) ||
                    !DownloadBuffer(&flags, buffers.Error.Pointer, sizeof(flags), cancellation,
                                    result.Error) ||
                    !CheckInstanceError(result.Error))
                    return false;
                if (flags)
                {
                    result.Error = "CUDA MLT encountered invalid or nonfinite path/film arithmetic.";
                    return false;
                }
                if (statisticsCount && !DownloadBuffer(statistics.data(), buffers.Statistics.Pointer,
                                                       size_t(statisticsCount) * sizeof(GroupStatistics),
                                                       cancellation, result.Error))
                    return false;
                for (uint32_t index = 0; index < statisticsCount; ++index)
                {
                    result.Statistics.PrimaryRays += statistics[index].PrimaryRays;
                    result.Statistics.ShadowRays += statistics[index].ShadowRays;
                    result.Statistics.ReflectionRays += statistics[index].ReflectionRays;
                    result.Statistics.PrimitiveTests += statistics[index].PrimitiveTests;
                }
                return true;
            };
            for (uint32_t begin = 0; begin < options.BootstrapSamples; begin += CudaMlt::BootstrapBatch)
            {
                controls.Begin = begin;
                controls.Count = std::min(CudaMlt::BootstrapBatch, options.BootstrapSamples - begin);
                if (!dispatch(bootstrapKernel, controls.Count, bootstrapArguments, controls.Count))
                    return false;
            }
            if (!dispatch(distributionKernel, 1, distributionArguments, 0) ||
                !dispatch(initializeKernel, chains, initializeArguments, 0))
                return false;
            const uint32_t longestChain = (fullPixels + chains - 1) / chains;
            for (uint32_t pass = 0; pass < constants.TotalSamples; ++pass)
            {
                if (cancellation())
                    return false;
                if (!Check(Api.MemsetD8Async(buffers.Pass.Pointer, 0, buffers.Pass.Size, Stream),
                           "cuMemsetD8Async(MLT pass)", result.Error))
                    return false;
                for (uint32_t begin = 0; begin < longestChain; begin += CudaMlt::MutationsPerBatch)
                {
                    controls.Begin = begin;
                    controls.Count = std::min(CudaMlt::MutationsPerBatch, longestChain - begin);
                    if (!dispatch(mutateKernel, chains, mutationArguments, chains))
                        return false;
                }
                if (!dispatch(commitKernel, pixels, commitArguments, 0))
                    return false;
            }
            return true;
        }

        bool RenderPhotons(const Scene& scene, const FrameConstants& constants,
                           const RenderSettings& settings, const Detail::CancellationProbe& cancellation,
                           RenderResult& result) const
        {
            namespace Photon = CudaPhoton;
            if (settings.MaxReflectionBounces > Photon::MaximumReflections || settings.Adaptive.Enabled ||
                settings.Transport.Mis != MisHeuristic::None || !settings.ShadowsEnabled ||
                settings.ShadowStrength != 1)
            {
                result.Error = "CUDA photon mapping requires fixed budgets, bounded depth, no connection MIS "
                               "and full visibility.";
                return false;
            }
            const auto sources = CudaEmissionHost::PrepareEmitters(scene);
            for (const auto& source : sources.Sources)
                if (!(source.Flags & EmitterCastsShadows))
                {
                    result.Error = "CUDA photon mapping requires shadows on every emitter.";
                    return false;
                }
            if (!PhotonModule &&
                !Check(Api.ModuleLoadDataEx(&PhotonModule, Detail::CudaPhotonPtx, 0, nullptr, nullptr),
                       "cuModuleLoadDataEx(Photon mapping)", result.Error))
                return false;
            CUfunction emit{}, index{}, camera{};
            if (!Check(Api.ModuleGetFunction(&emit, PhotonModule, "PhotonEmitKernel"),
                       "cuModuleGetFunction(Photon emit)", result.Error) ||
                !Check(Api.ModuleGetFunction(&index, PhotonModule, "PhotonIndexKernel"),
                       "cuModuleGetFunction(Photon index)", result.Error) ||
                !Check(Api.ModuleGetFunction(&camera, PhotonModule, "PhotonCameraKernel"),
                       "cuModuleGetFunction(Photon camera)", result.Error))
                return false;
            struct Buffers
            {
                const Impl& Owner;
                DeviceBuffer Emitters, Controls, Photons, Heads, Statistics, Error;
                ~Buffers()
                {
                    Owner.Api.StreamSynchronize(Owner.Stream);
                    for (auto* buffer : {&Emitters, &Controls, &Photons, &Heads, &Statistics, &Error})
                        Owner.FreeBuffer(*buffer);
                }
            } buffers{*this};
            const auto& options = settings.Photons;
            uint32_t buckets = 1;
            while (buckets < options.PhotonPaths)
                buckets *= 2;
            const uint32_t slotsPerPath = settings.MaxReflectionBounces + 1;
            const uint32_t records = options.PhotonPaths * slotsPerPath;
            const uint32_t pixels = constants.Width * constants.Height;
            constexpr uint32_t StatisticsCapacity = std::max(Photon::EmissionBatch, Photon::CameraBatch);
            if (!UploadBuffer(buffers.Emitters, sources.Sources.data(),
                              sources.Sources.size() * sizeof(PackedEmitter), cancellation, result.Error) ||
                !AllocateBuffer(buffers.Controls, sizeof(Photon::Controls), result.Error) ||
                !AllocateBuffer(buffers.Photons, size_t(records) * sizeof(Photon::Photon), result.Error) ||
                !AllocateBuffer(buffers.Heads, size_t(buckets) * sizeof(uint32_t), result.Error) ||
                !AllocateBuffer(buffers.Statistics, size_t(StatisticsCapacity) * sizeof(GroupStatistics),
                                result.Error) ||
                !AllocateBuffer(buffers.Error, sizeof(uint32_t), result.Error))
                return false;
            Photon::Controls control{{sources.Center.X, sources.Center.Y, sources.Center.Z},
                                     sources.Radius,
                                     double(sources.Radius) * options.RadiusScale,
                                     uint32_t(sources.Sources.size()),
                                     options.PhotonPaths,
                                     slotsPerPath,
                                     buckets,
                                     options.Seed ^ settings.SamplingSeed,
                                     (options.Caustics ? Photon::CausticEnabled : 0u) |
                                         (options.Indirect ? Photon::IndirectEnabled : 0u)};
            std::vector<GroupStatistics> statistics(StatisticsCapacity);
            std::vector<uint32_t> targets(pixels, constants.TotalSamples);
            uint32_t flags = 0;
            struct TransferLifetime
            {
                const Impl& Owner;
                ~TransferLifetime() { Owner.Api.StreamSynchronize(Owner.Stream); }
            } retire{*this};
            constexpr unsigned char EmptyBucketByte = 0xff;
            if (!UploadConstants(constants, result.Error) ||
                !Check(Api.MemcpyHtoDAsync(SampleTargets.Pointer, targets.data(),
                                           targets.size() * sizeof(uint32_t), Stream),
                       "cuMemcpyHtoDAsync(Photon targets)", result.Error) ||
                !Check(Api.MemsetD8Async(buffers.Heads.Pointer, EmptyBucketByte, buffers.Heads.Size, Stream),
                       "cuMemsetD8Async(Photon buckets)", result.Error) ||
                !Check(Api.MemsetD8Async(buffers.Error.Pointer, 0, buffers.Error.Size, Stream),
                       "cuMemsetD8Async(Photon errors)", result.Error))
                return false;
            void* emitArguments[] = {&Nodes.Pointer,           &PrimitiveRefs.Pointer,
                                     &Triangles.Pointer,       &Spheres.Pointer,
                                     &Materials.Pointer,       &TextureDescriptors.Pointer,
                                     &TextureTexels.Pointer,   &buffers.Emitters.Pointer,
                                     &Constants.Pointer,       &buffers.Controls.Pointer,
                                     &buffers.Photons.Pointer, &buffers.Statistics.Pointer,
                                     &buffers.Error.Pointer};
            void* indexArguments[] = {&buffers.Controls.Pointer, &buffers.Photons.Pointer,
                                      &buffers.Heads.Pointer, &buffers.Error.Pointer};
            void* cameraArguments[] = {&Nodes.Pointer,
                                       &PrimitiveRefs.Pointer,
                                       &Triangles.Pointer,
                                       &Spheres.Pointer,
                                       &Materials.Pointer,
                                       &TextureDescriptors.Pointer,
                                       &TextureTexels.Pointer,
                                       &buffers.Emitters.Pointer,
                                       &Constants.Pointer,
                                       &buffers.Controls.Pointer,
                                       &buffers.Photons.Pointer,
                                       &buffers.Heads.Pointer,
                                       &Accumulation.Pointer,
                                       &SampleCounts.Pointer,
                                       &buffers.Statistics.Pointer,
                                       &buffers.Error.Pointer};
            const auto dispatch = [&](CUfunction kernel, void** arguments, bool countStatistics) {
                if (cancellation())
                    return false;
                const uint32_t groups =
                    (control.Count + Photon::ThreadsPerBlock - 1) / Photon::ThreadsPerBlock;
                if (!Check(Api.MemcpyHtoDAsync(buffers.Controls.Pointer, &control, sizeof(control), Stream),
                           "cuMemcpyHtoDAsync(Photon controls)", result.Error) ||
                    !Check(Api.LaunchKernel(kernel, groups, 1, 1, Photon::ThreadsPerBlock, 1, 1, 0, Stream,
                                            arguments, nullptr),
                           "cuLaunchKernel(Photon mapping)", result.Error) ||
                    !DownloadBuffer(&flags, buffers.Error.Pointer, sizeof(flags), cancellation,
                                    result.Error) ||
                    !CheckInstanceError(result.Error))
                    return false;
                if (flags)
                {
                    result.Error = "CUDA photon mapping encountered invalid indexing or nonfinite transport.";
                    return false;
                }
                if (countStatistics)
                {
                    if (!DownloadBuffer(statistics.data(), buffers.Statistics.Pointer,
                                        size_t(control.Count) * sizeof(GroupStatistics), cancellation,
                                        result.Error))
                        return false;
                    for (uint32_t lane = 0; lane < control.Count; ++lane)
                    {
                        result.Statistics.PrimaryRays += statistics[lane].PrimaryRays;
                        result.Statistics.ReflectionRays += statistics[lane].ReflectionRays;
                        result.Statistics.ShadowRays += statistics[lane].ShadowRays;
                        result.Statistics.PrimitiveTests += statistics[lane].PrimitiveTests;
                    }
                }
                return true;
            };
            // Light population, index and camera stages share one stream. Camera
            // sampling starts only after every light record and bucket link exists.
            for (uint32_t begin = 0; begin < options.PhotonPaths; begin += Photon::EmissionBatch)
            {
                control.Begin = begin;
                control.Count = std::min(Photon::EmissionBatch, options.PhotonPaths - begin);
                if (!dispatch(emit, emitArguments, true))
                    return false;
            }
            for (uint32_t begin = 0; begin < records; begin += Photon::IndexBatch)
            {
                control.Begin = begin;
                control.Count = std::min(Photon::IndexBatch, records - begin);
                if (!dispatch(index, indexArguments, false))
                    return false;
            }
            for (uint32_t sample = 0; sample < constants.TotalSamples; ++sample)
                for (uint32_t begin = 0; begin < pixels; begin += Photon::CameraBatch)
                {
                    control.Begin = begin;
                    control.Count = std::min(Photon::CameraBatch, pixels - begin);
                    control.Sample = sample;
                    if (!dispatch(camera, cameraArguments, true))
                        return false;
                }
            return true;
        }

        bool RenderProgressivePhotons(const Scene& scene, const FrameConstants& constants,
                                      const RenderSettings& settings,
                                      const Detail::CancellationProbe& cancellation,
                                      RenderResult& result) const
        {
            namespace Ppm = CudaPpm;
            if (settings.MaxReflectionBounces > Ppm::MaximumReflections || settings.Adaptive.Enabled ||
                settings.Transport.Mis != MisHeuristic::None || !settings.ShadowsEnabled ||
                settings.ShadowStrength != 1)
            {
                result.Error = "CUDA PPM requires fixed pass budgets, bounded depth, no connection MIS and "
                               "full visibility.";
                return false;
            }
            const auto sources = CudaEmissionHost::PrepareEmitters(scene);
            for (const auto& source : sources.Sources)
                if (!(source.Flags & EmitterCastsShadows))
                {
                    result.Error = "CUDA PPM requires shadows on every emitter.";
                    return false;
                }
            if (!ProgressivePhotonModule &&
                !Check(Api.ModuleLoadDataEx(&ProgressivePhotonModule, Detail::CudaProgressivePhotonPtx, 0,
                                            nullptr, nullptr),
                       "cuModuleLoadDataEx(PPM)", result.Error))
                return false;
            CUfunction initialize{}, camera{}, index{}, photon{}, update{};
            const std::pair<CUfunction*, const char*> entries[] = {{&initialize, "PpmInitializeKernel"},
                                                                   {&camera, "PpmCameraKernel"},
                                                                   {&index, "PpmIndexKernel"},
                                                                   {&photon, "PpmPhotonKernel"},
                                                                   {&update, "PpmUpdateKernel"}};
            for (const auto& entry : entries)
                if (!Check(Api.ModuleGetFunction(entry.first, ProgressivePhotonModule, entry.second),
                           "cuModuleGetFunction(PPM)", result.Error))
                    return false;
            struct Buffers
            {
                const Impl& Owner;
                DeviceBuffer Emitters, Controls, Points, Pixels, Heads, Statistics, Error;
                ~Buffers()
                {
                    Owner.Api.StreamSynchronize(Owner.Stream);
                    for (auto* buffer : {&Emitters, &Controls, &Points, &Pixels, &Heads, &Statistics, &Error})
                        Owner.FreeBuffer(*buffer);
                }
            } buffers{*this};
            const uint32_t pixels = constants.Width * constants.Height;
            uint32_t buckets = 1;
            while (buckets < pixels)
                buckets *= 2;
            if (!UploadBuffer(buffers.Emitters, sources.Sources.data(),
                              sources.Sources.size() * sizeof(PackedEmitter), cancellation, result.Error) ||
                !AllocateBuffer(buffers.Controls, sizeof(Ppm::Controls), result.Error) ||
                !AllocateBuffer(buffers.Points, size_t(pixels) * sizeof(Ppm::VisiblePoint), result.Error) ||
                !AllocateBuffer(buffers.Pixels, size_t(pixels) * sizeof(Ppm::Pixel), result.Error) ||
                !AllocateBuffer(buffers.Heads, size_t(buckets) * sizeof(uint32_t), result.Error) ||
                !AllocateBuffer(buffers.Statistics, size_t(Ppm::TransportBatch) * sizeof(GroupStatistics),
                                result.Error) ||
                !AllocateBuffer(buffers.Error, sizeof(uint32_t), result.Error))
                return false;
            const auto& options = settings.ProgressivePhotons;
            Ppm::Controls control{{sources.Center.X, sources.Center.Y, sources.Center.Z},
                                  sources.Radius,
                                  double(sources.Radius) * options.InitialRadiusScale,
                                  options.Alpha,
                                  uint32_t(sources.Sources.size()),
                                  options.PathsPerPass,
                                  buckets,
                                  options.Seed ^ settings.SamplingSeed,
                                  (options.Caustics ? Ppm::Caustics : 0u) |
                                      (options.Indirect ? Ppm::Indirect : 0u)};
            std::vector<GroupStatistics> statistics(Ppm::TransportBatch);
            uint32_t flags = 0;
            // Cancellation can leave asynchronous work in flight. Retire it before
            // host upload sources and device resources leave their scopes.
            struct TransferLifetime
            {
                const Impl& Owner;
                ~TransferLifetime() { Owner.Api.StreamSynchronize(Owner.Stream); }
            } retire{*this};
            if (!UploadConstants(constants, result.Error) ||
                !Check(Api.MemsetD8Async(buffers.Error.Pointer, 0, buffers.Error.Size, Stream),
                       "cuMemsetD8Async(PPM errors)", result.Error))
                return false;
            void* initializeArguments[] = {&Constants.Pointer,      &buffers.Controls.Pointer,
                                           &buffers.Pixels.Pointer, &buffers.Points.Pointer,
                                           &Accumulation.Pointer,   &SampleCounts.Pointer,
                                           &SampleTargets.Pointer,  &buffers.Error.Pointer};
            void* cameraArguments[] = {&Nodes.Pointer,
                                       &PrimitiveRefs.Pointer,
                                       &Triangles.Pointer,
                                       &Spheres.Pointer,
                                       &Materials.Pointer,
                                       &TextureDescriptors.Pointer,
                                       &TextureTexels.Pointer,
                                       &buffers.Emitters.Pointer,
                                       &Constants.Pointer,
                                       &buffers.Controls.Pointer,
                                       &buffers.Points.Pointer,
                                       &buffers.Pixels.Pointer,
                                       &buffers.Statistics.Pointer,
                                       &buffers.Error.Pointer};
            void* indexArguments[] = {&Constants.Pointer,      &buffers.Controls.Pointer,
                                      &buffers.Points.Pointer, &buffers.Pixels.Pointer,
                                      &buffers.Heads.Pointer,  &buffers.Error.Pointer};
            void* photonArguments[] = {&Nodes.Pointer,          &PrimitiveRefs.Pointer,
                                       &Triangles.Pointer,      &Spheres.Pointer,
                                       &Materials.Pointer,      &TextureDescriptors.Pointer,
                                       &TextureTexels.Pointer,  &buffers.Emitters.Pointer,
                                       &Constants.Pointer,      &buffers.Controls.Pointer,
                                       &buffers.Points.Pointer, &buffers.Pixels.Pointer,
                                       &buffers.Heads.Pointer,  &buffers.Statistics.Pointer,
                                       &buffers.Error.Pointer};
            void* updateArguments[] = {&Constants.Pointer,      &buffers.Controls.Pointer,
                                       &buffers.Pixels.Pointer, &Accumulation.Pointer,
                                       &SampleCounts.Pointer,   &buffers.Error.Pointer};
            const auto stage = [&](CUfunction kernel, void** arguments, uint32_t total, bool transport) {
                const uint32_t batch = transport ? Ppm::TransportBatch : Ppm::LinearBatch;
                for (uint32_t begin = 0; begin < total; begin += batch)
                {
                    if (cancellation())
                        return false;
                    control.Begin = begin;
                    control.Count = std::min(batch, total - begin);
                    const uint32_t groups = (control.Count + Ppm::ThreadsPerBlock - 1) / Ppm::ThreadsPerBlock;
                    if (!Check(
                            Api.MemcpyHtoDAsync(buffers.Controls.Pointer, &control, sizeof(control), Stream),
                            "cuMemcpyHtoDAsync(PPM controls)", result.Error) ||
                        !Check(Api.LaunchKernel(kernel, groups, 1, 1, Ppm::ThreadsPerBlock, 1, 1, 0, Stream,
                                                arguments, nullptr),
                               "cuLaunchKernel(PPM)", result.Error) ||
                        !DownloadBuffer(&flags, buffers.Error.Pointer, sizeof(flags), cancellation,
                                        result.Error) ||
                        !CheckInstanceError(result.Error))
                        return false;
                    if (flags)
                    {
                        result.Error =
                            "CUDA PPM encountered invalid configuration, indexing or nonfinite transport.";
                        return false;
                    }
                    if (transport)
                    {
                        if (!DownloadBuffer(statistics.data(), buffers.Statistics.Pointer,
                                            size_t(control.Count) * sizeof(GroupStatistics), cancellation,
                                            result.Error))
                            return false;
                        for (uint32_t lane = 0; lane < control.Count; ++lane)
                        {
                            result.Statistics.PrimaryRays += statistics[lane].PrimaryRays;
                            result.Statistics.ShadowRays += statistics[lane].ShadowRays;
                            result.Statistics.ReflectionRays += statistics[lane].ReflectionRays;
                            result.Statistics.PrimitiveTests += statistics[lane].PrimitiveTests;
                        }
                    }
                }
                return true;
            };
            if (!stage(initialize, initializeArguments, pixels, false))
                return false;
            constexpr unsigned char EmptyBucketByte = 0xff;
            for (uint32_t pass = 0; pass < constants.TotalSamples; ++pass)
            {
                control.Pass = pass;
                // One stream establishes camera -> index -> deposits -> update.
                // Only incoming flux and visible points reset; N, R and tau persist.
                if (!stage(camera, cameraArguments, pixels, true) ||
                    !Check(
                        Api.MemsetD8Async(buffers.Heads.Pointer, EmptyBucketByte, buffers.Heads.Size, Stream),
                        "cuMemsetD8Async(PPM buckets)", result.Error) ||
                    !stage(index, indexArguments, pixels, false) ||
                    !stage(photon, photonArguments, options.PathsPerPass, true) ||
                    !stage(update, updateArguments, pixels, false))
                    return false;
            }
            return true;
        }

        bool RenderVcm(const Scene& scene, const FrameConstants& constants, const RenderSettings& settings,
                       const Detail::CancellationProbe& cancellation, RenderResult& result) const
        {
            namespace Vcm = CudaVcm;
            const auto& options = settings.Vcm;
            if (settings.MaxReflectionBounces > Vcm::MaximumReflections || settings.Adaptive.Enabled ||
                settings.Transport.RussianRoulette || !settings.ShadowsEnabled ||
                settings.ShadowStrength != 1 ||
                !ValidateVertexConnectionMergingOptions(options, result.Error))
            {
                if (result.Error.empty())
                    result.Error =
                        "CUDA VCM requires fixed passes/depth, at most 32 reflections, no adaptive "
                        "stopping or roulette, and full visibility.";
                return false;
            }
            if (uint64_t(options.LightPathsPerPass) * sizeof(Vcm::Path) > Vcm::MaximumPathCacheBytes)
            {
                result.Error =
                    "CUDA VCM light paths exceed the 512 MiB path-cache budget; reduce paths per pass.";
                return false;
            }
            const auto sources = CudaEmissionHost::PrepareEmitters(scene);
            for (const auto& source : sources.Sources)
                if (!(source.Flags & EmitterCastsShadows))
                {
                    result.Error = "CUDA VCM requires shadow casting on every emitter.";
                    return false;
                }
            if (!VcmModule &&
                !Check(Api.ModuleLoadDataEx(&VcmModule, Detail::CudaVcmPtx, 0, nullptr, nullptr),
                       "cuModuleLoadDataEx(VCM)", result.Error))
                return false;
            CUfunction initialize{}, traceLight{}, traceCamera{}, buildIndex{}, sortIndex{}, buildRanges{},
                cameraEstimate{}, splatEstimate{}, commit{};
            const std::pair<CUfunction*, const char*> kernels[] = {
                {&initialize, "VcmInitializeFrameKernel"},
                {&traceLight, "VcmTraceLightPathsKernel"},
                {&traceCamera, "VcmTraceCameraPathsKernel"},
                {&buildIndex, "VcmBuildIndexKernel"},
                {&sortIndex, "VcmSortIndexKernel"},
                {&buildRanges, "VcmBuildRangesKernel"},
                {&cameraEstimate, "VcmCameraEstimateKernel"},
                {&splatEstimate, "VcmSplatEstimateKernel"},
                {&commit, "VcmCommitPassKernel"}};
            for (const auto& kernel : kernels)
                if (!Check(Api.ModuleGetFunction(kernel.first, VcmModule, kernel.second),
                           "cuModuleGetFunction(VCM)", result.Error))
                    return false;
            struct Buffers
            {
                const Impl& Owner;
                DeviceBuffer Emitters, Walks, Controls, Lights, Cameras, LensRoot, Index, Ranges, Splats,
                    Pixels, Statistics, Error;
                ~Buffers()
                {
                    Owner.Api.StreamSynchronize(Owner.Stream);
                    for (auto* buffer : {&Emitters, &Walks, &Controls, &Lights, &Cameras, &LensRoot, &Index,
                                         &Ranges, &Splats, &Pixels, &Statistics, &Error})
                        Owner.FreeBuffer(*buffer);
                }
            } buffers{*this};
            Vcm::WalkControls walk{{sources.Center.X, sources.Center.Y, sources.Center.Z},
                                   sources.Radius,
                                   uint32_t(sources.Sources.size()),
                                   0,
                                   0,
                                   options.Seed ^ settings.SamplingSeed,
                                   0,
                                   settings.MaxReflectionBounces + 3,
                                   {}};
            const uint32_t vertices = options.LightPathsPerPass * (walk.VertexLimit - 2);
            uint32_t capacity = 1;
            while (capacity < vertices)
                capacity *= 2;
            const Vcm::FrameControls control{double(sources.Radius) * options.InitialRadiusScale,
                                             options.Alpha,
                                             options.LightPathsPerPass,
                                             capacity,
                                             capacity,
                                             uint32_t(settings.Transport.Mis),
                                             settings.Transport.NextEventEstimation ? 1u : 0u,
                                             {}};
            const uint32_t pixels = constants.Width * constants.Height;
            if (!UploadBuffer(buffers.Emitters, sources.Sources.data(),
                              sources.Sources.size() * sizeof(PackedEmitter), cancellation, result.Error) ||
                !UploadBuffer(buffers.Controls, &control, sizeof(control), cancellation, result.Error) ||
                !AllocateBuffer(buffers.Walks, sizeof(walk), result.Error) ||
                !AllocateBuffer(buffers.Lights, size_t(options.LightPathsPerPass) * sizeof(Vcm::Path),
                                result.Error) ||
                !AllocateBuffer(buffers.Cameras, size_t(Vcm::TransportBatch) * sizeof(Vcm::Path),
                                result.Error) ||
                !AllocateBuffer(buffers.LensRoot, sizeof(Vcm::Path), result.Error) ||
                !AllocateBuffer(buffers.Index, size_t(capacity) * sizeof(Vcm::IndexEntry), result.Error) ||
                !AllocateBuffer(buffers.Ranges, size_t(control.Buckets) * sizeof(Vcm::BucketRange),
                                result.Error) ||
                !AllocateBuffer(buffers.Splats, size_t(vertices) * sizeof(Vcm::Contribution), result.Error) ||
                !AllocateBuffer(buffers.Pixels, size_t(pixels) * sizeof(Vcm::PixelState), result.Error) ||
                !AllocateBuffer(buffers.Statistics, size_t(Vcm::TransportBatch) * sizeof(GroupStatistics),
                                result.Error) ||
                !AllocateBuffer(buffers.Error, sizeof(uint32_t), result.Error))
                return false;
            uint32_t flags = 0, span = 0, stride = 0;
            CUdeviceptr lightDestination = buffers.Lights.Pointer;
            std::vector<GroupStatistics> statistics(Vcm::TransportBatch);
            // Host transfer storage outlives asynchronous commands, even when a
            // cancellation interrupts the next readback. All pass storage is per job.
            struct TransferLifetime
            {
                const Impl& Owner;
                ~TransferLifetime() { Owner.Api.StreamSynchronize(Owner.Stream); }
            } retire{*this};
            if (!UploadConstants(constants, result.Error) ||
                !Check(Api.MemsetD8Async(buffers.Error.Pointer, 0, buffers.Error.Size, Stream),
                       "cuMemsetD8Async(VCM errors)", result.Error))
                return false;
            void* initializeArgs[] = {
                &Constants.Pointer,      &buffers.Walks.Pointer,    &buffers.Controls.Pointer,
                &buffers.Pixels.Pointer, &buffers.LensRoot.Pointer, &Accumulation.Pointer,
                &SampleCounts.Pointer,   &SampleTargets.Pointer,    &buffers.Error.Pointer};
            void* lightArgs[] = {
                &Nodes.Pointer,         &PrimitiveRefs.Pointer,    &Triangles.Pointer,
                &Spheres.Pointer,       &Materials.Pointer,        &TextureDescriptors.Pointer,
                &TextureTexels.Pointer, &buffers.Emitters.Pointer, &Constants.Pointer,
                &buffers.Walks.Pointer, &lightDestination,         &buffers.Statistics.Pointer,
                &buffers.Error.Pointer};
            void* cameraWalkArgs[] = {
                &Nodes.Pointer,         &PrimitiveRefs.Pointer,    &Triangles.Pointer,
                &Spheres.Pointer,       &Materials.Pointer,        &TextureDescriptors.Pointer,
                &TextureTexels.Pointer, &buffers.Emitters.Pointer, &Constants.Pointer,
                &buffers.Walks.Pointer, &buffers.Cameras.Pointer,  &buffers.Statistics.Pointer,
                &buffers.Error.Pointer};
            void* indexArgs[] = {&buffers.Walks.Pointer, &buffers.Controls.Pointer, &buffers.Lights.Pointer,
                                 &buffers.Index.Pointer, &buffers.Ranges.Pointer,   &buffers.Error.Pointer};
            void* sortArgs[] = {&buffers.Walks.Pointer, &buffers.Controls.Pointer, &buffers.Index.Pointer,
                                &span, &stride};
            void* rangeArgs[] = {&buffers.Walks.Pointer, &buffers.Controls.Pointer, &buffers.Index.Pointer,
                                 &buffers.Ranges.Pointer, &buffers.Error.Pointer};
            void* cameraArgs[] = {
                &Nodes.Pointer,           &PrimitiveRefs.Pointer,      &Triangles.Pointer,
                &Spheres.Pointer,         &Materials.Pointer,          &TextureDescriptors.Pointer,
                &TextureTexels.Pointer,   &buffers.Emitters.Pointer,   &Constants.Pointer,
                &buffers.Walks.Pointer,   &buffers.Controls.Pointer,   &buffers.Lights.Pointer,
                &buffers.Cameras.Pointer, &buffers.Index.Pointer,      &buffers.Ranges.Pointer,
                &buffers.Pixels.Pointer,  &buffers.Statistics.Pointer, &buffers.Error.Pointer};
            void* splatArgs[] = {&Nodes.Pointer,
                                 &PrimitiveRefs.Pointer,
                                 &Triangles.Pointer,
                                 &Spheres.Pointer,
                                 &Materials.Pointer,
                                 &TextureDescriptors.Pointer,
                                 &TextureTexels.Pointer,
                                 &buffers.Emitters.Pointer,
                                 &Constants.Pointer,
                                 &buffers.Walks.Pointer,
                                 &buffers.Controls.Pointer,
                                 &buffers.Lights.Pointer,
                                 &buffers.LensRoot.Pointer,
                                 &buffers.Index.Pointer,
                                 &buffers.Splats.Pointer,
                                 &buffers.Statistics.Pointer,
                                 &buffers.Error.Pointer};
            void* commitArgs[] = {&Constants.Pointer,     &buffers.Walks.Pointer,  &buffers.Controls.Pointer,
                                  &buffers.Index.Pointer, &buffers.Splats.Pointer, &buffers.Pixels.Pointer,
                                  &Accumulation.Pointer,  &SampleCounts.Pointer,   &buffers.Error.Pointer};
            const auto launch = [&](CUfunction kernel, void** args, bool transport) {
                if (cancellation())
                    return false;
                const unsigned groups =
                    (walk.Count + Vcm::WalkThreadsPerBlock - 1) / Vcm::WalkThreadsPerBlock;
                if (!Check(Api.MemcpyHtoDAsync(buffers.Walks.Pointer, &walk, sizeof(walk), Stream),
                           "cuMemcpyHtoDAsync(VCM controls)", result.Error) ||
                    !Check(Api.LaunchKernel(kernel, groups, 1, 1, Vcm::WalkThreadsPerBlock, 1, 1, 0, Stream,
                                            args, nullptr),
                           "cuLaunchKernel(VCM)", result.Error) ||
                    !DownloadBuffer(&flags, buffers.Error.Pointer, sizeof(flags), cancellation,
                                    result.Error) ||
                    !CheckInstanceError(result.Error))
                    return false;
                if (flags)
                {
                    result.Error =
                        "CUDA VCM encountered invalid configuration, indexing or nonfinite transport.";
                    return false;
                }
                if (transport)
                {
                    if (!DownloadBuffer(statistics.data(), buffers.Statistics.Pointer,
                                        size_t(walk.Count) * sizeof(GroupStatistics), cancellation,
                                        result.Error))
                        return false;
                    for (unsigned i = 0; i < walk.Count; ++i)
                    {
                        result.Statistics.PrimaryRays += statistics[i].PrimaryRays;
                        result.Statistics.ReflectionRays += statistics[i].ReflectionRays;
                        result.Statistics.ShadowRays += statistics[i].ShadowRays;
                        result.Statistics.PrimitiveTests += statistics[i].PrimitiveTests;
                    }
                }
                return true;
            };
            const auto stage = [&](CUfunction kernel, void** args, uint32_t total, bool transport) {
                const uint32_t batch = transport ? Vcm::TransportBatch : Vcm::LinearBatch;
                for (uint32_t begin = 0; begin < total; begin += batch)
                {
                    walk.Begin = begin;
                    walk.Count = std::min(batch, total - begin);
                    lightDestination = buffers.Lights.Pointer + size_t(begin) * sizeof(Vcm::Path);
                    if (!launch(kernel, args, transport))
                        return false;
                }
                return true;
            };
            const auto sort = [&]() {
                for (span = 2; span <= capacity; span *= 2)
                    for (stride = span / 2; stride; stride /= 2)
                        if (!stage(sortIndex, sortArgs, capacity, false))
                            return false;
                return true;
            };
            if (!stage(initialize, initializeArgs, pixels, false))
                return false;
            for (walk.Pass = 0; walk.Pass < constants.TotalSamples; ++walk.Pass)
            {
                if (!stage(traceLight, lightArgs, options.LightPathsPerPass, true) ||
                    !stage(buildIndex, indexArgs, std::max(capacity, control.Buckets), false) || !sort() ||
                    !stage(buildRanges, rangeArgs, capacity, false))
                    return false;
                for (uint32_t begin = 0; begin < pixels; begin += Vcm::TransportBatch)
                {
                    walk.Begin = begin;
                    walk.Count = std::min(Vcm::TransportBatch, pixels - begin);
                    if (!launch(traceCamera, cameraWalkArgs, true) ||
                        !launch(cameraEstimate, cameraArgs, true))
                        return false;
                }
                if (!stage(splatEstimate, splatArgs, capacity, true) || !sort() ||
                    !stage(commit, commitArgs, pixels, false))
                    return false;
            }
            return true;
        }

        bool RenderPointGi(const Scene& scene, const FrameConstants& constants,
                           const RenderSettings& settings, const Detail::CancellationProbe& cancellation,
                           RenderResult& result) const
        {
            namespace G = CudaPointGi;
            if (!ValidatePointGiOptions(settings.PointGi, result.Error) ||
                !ValidateTransportOptions(settings.Integrator, settings.Transport, result.Error))
                return false;
            if (settings.MaxReflectionBounces > G::MaximumReflections || settings.Adaptive.Enabled ||
                !settings.ShadowsEnabled || settings.ShadowStrength != 1)
            {
                result.Error =
                    "CUDA Point GI requires bounded diffuse orders, fixed budgets and full visibility.";
                return false;
            }
            const auto sources = CudaEmissionHost::PrepareEmitters(scene);
            for (const auto& source : sources.Sources)
                if (!(source.Flags & EmitterCastsShadows))
                {
                    result.Error = "CUDA Point GI requires shadows on every source.";
                    return false;
                }
            if (sources.Sources.size() > MaximumUniformEmitters)
            {
                result.Error = "CUDA Point GI source catalog exceeds uniform selection precision.";
                return false;
            }
            G::GeometryCache geometry;
            if (settings.MaxReflectionBounces &&
                !G::BuildGeometry(scene, settings.PointGi, geometry, cancellation, result.Error))
                return false;
            if (!PointGiModule &&
                !Check(Api.ModuleLoadDataEx(&PointGiModule, Detail::CudaPointGiPtx, 0, nullptr, nullptr),
                       "cuModuleLoadDataEx(Point GI)", result.Error))
                return false;
            CUfunction bake{}, camera{};
            if (!Check(Api.ModuleGetFunction(&bake, PointGiModule, "PointGiBakeKernel"),
                       "cuModuleGetFunction(Point GI bake)", result.Error) ||
                !Check(Api.ModuleGetFunction(&camera, PointGiModule, "PointGiCameraKernel"),
                       "cuModuleGetFunction(Point GI camera)", result.Error))
                return false;
            G::Controls control{};
            control.Surfels = uint32_t(geometry.Surfels.size());
            control.Nodes = uint32_t(geometry.Nodes.size());
            control.Sources = uint32_t(sources.Sources.size());
            control.Seed = settings.PointGi.Seed ^ settings.SamplingSeed;
            control.DirectSamples = settings.PointGi.DirectSamples;
            control.GatherSamples = settings.PointGi.GatherSamples;
            const uint32_t pixels = constants.Width * constants.Height;
            std::vector<uint32_t> targets(pixels, settings.SamplesPerPixel);
            std::vector<GroupStatistics> work(G::TransportBatch);
            uint32_t flags = 0;
            // Drain outstanding transfers before their host storage or device
            // buffers leave scope, including cancellation and launch failures.
            struct Buffers
            {
                const Impl& Owner;
                DeviceBuffer Sources, Points, Tree, Order, Controls, Previous, Next, Total, Statistics,
                    Errors;
                ~Buffers()
                {
                    Owner.Api.StreamSynchronize(Owner.Stream);
                    for (auto* buffer : {&Sources, &Points, &Tree, &Order, &Controls, &Previous, &Next,
                                         &Total, &Statistics, &Errors})
                        Owner.FreeBuffer(*buffer);
                }
            } buffers{*this};
            const size_t radianceBytes = geometry.Surfels.size() * sizeof(G::Radiance);
            if (!UploadBuffer(buffers.Sources, sources.Sources.data(),
                              sources.Sources.size() * sizeof(PackedEmitter), cancellation, result.Error) ||
                !UploadBuffer(buffers.Points, geometry.Surfels.data(),
                              geometry.Surfels.size() * sizeof(G::Surfel), cancellation, result.Error) ||
                !UploadBuffer(buffers.Tree, geometry.Nodes.data(), geometry.Nodes.size() * sizeof(G::Node),
                              cancellation, result.Error) ||
                !UploadBuffer(buffers.Order, geometry.Order.data(), geometry.Order.size() * sizeof(uint32_t),
                              cancellation, result.Error) ||
                !AllocateBuffer(buffers.Controls, sizeof(control), result.Error) ||
                !AllocateBuffer(buffers.Previous, radianceBytes, result.Error) ||
                !AllocateBuffer(buffers.Next, radianceBytes, result.Error) ||
                !AllocateBuffer(buffers.Total, radianceBytes, result.Error) ||
                !AllocateBuffer(buffers.Statistics, work.size() * sizeof(GroupStatistics), result.Error) ||
                !AllocateBuffer(buffers.Errors, sizeof(flags), result.Error))
                return false;
            if (!UploadConstants(constants, result.Error) ||
                !Check(Api.MemcpyHtoDAsync(SampleTargets.Pointer, targets.data(),
                                           targets.size() * sizeof(uint32_t), Stream),
                       "cuMemcpyHtoDAsync(Point GI sample targets)", result.Error) ||
                !Check(Api.MemsetD8Async(buffers.Errors.Pointer, 0, sizeof(flags), Stream),
                       "cuMemsetD8Async(Point GI errors)", result.Error))
                return false;
            CUdeviceptr previous = buffers.Previous.Pointer, next = buffers.Next.Pointer;
            void* bakeArguments[] = {&Nodes.Pointer,
                                     &PrimitiveRefs.Pointer,
                                     &Triangles.Pointer,
                                     &Spheres.Pointer,
                                     &Materials.Pointer,
                                     &TextureDescriptors.Pointer,
                                     &TextureTexels.Pointer,
                                     &buffers.Sources.Pointer,
                                     &Constants.Pointer,
                                     &buffers.Controls.Pointer,
                                     &buffers.Points.Pointer,
                                     &buffers.Tree.Pointer,
                                     &buffers.Order.Pointer,
                                     &previous,
                                     &next,
                                     &buffers.Total.Pointer,
                                     &buffers.Statistics.Pointer,
                                     &buffers.Errors.Pointer};
            void* cameraArguments[] = {&Nodes.Pointer,
                                       &PrimitiveRefs.Pointer,
                                       &Triangles.Pointer,
                                       &Spheres.Pointer,
                                       &Materials.Pointer,
                                       &TextureDescriptors.Pointer,
                                       &TextureTexels.Pointer,
                                       &buffers.Sources.Pointer,
                                       &Constants.Pointer,
                                       &buffers.Controls.Pointer,
                                       &buffers.Points.Pointer,
                                       &buffers.Tree.Pointer,
                                       &buffers.Order.Pointer,
                                       &buffers.Total.Pointer,
                                       &Accumulation.Pointer,
                                       &SampleCounts.Pointer,
                                       &buffers.Statistics.Pointer,
                                       &buffers.Errors.Pointer};
            const auto stage = [&](CUfunction kernel, void** arguments, uint32_t population,
                                   uint32_t samples) {
                for (control.SampleBegin = 0; control.SampleBegin < samples;
                     control.SampleBegin += G::SamplesPerDispatch)
                {
                    control.SampleCount = std::min(G::SamplesPerDispatch, samples - control.SampleBegin);
                    for (control.Begin = 0; control.Begin < population; control.Begin += G::TransportBatch)
                    {
                        if (cancellation())
                            return false;
                        control.Count = std::min(G::TransportBatch, population - control.Begin);
                        const uint32_t groups = (control.Count + G::ThreadsPerBlock - 1) / G::ThreadsPerBlock;
                        if (!Check(Api.MemcpyHtoDAsync(buffers.Controls.Pointer, &control, sizeof(control),
                                                       Stream),
                                   "cuMemcpyHtoDAsync(Point GI controls)", result.Error) ||
                            !Check(Api.LaunchKernel(kernel, groups, 1, 1, G::ThreadsPerBlock, 1, 1, 0, Stream,
                                                    arguments, nullptr),
                                   "cuLaunchKernel(Point GI)", result.Error) ||
                            !DownloadBuffer(&flags, buffers.Errors.Pointer, sizeof(flags), cancellation,
                                            result.Error) ||
                            !CheckInstanceError(result.Error))
                            return false;
                        if (flags)
                        {
                            result.Error =
                                "CUDA Point GI reported invalid controls, indexing or nonfinite transport.";
                            return false;
                        }
                        if (!DownloadBuffer(work.data(), buffers.Statistics.Pointer,
                                            size_t(control.Count) * sizeof(GroupStatistics), cancellation,
                                            result.Error))
                            return false;
                        for (uint32_t lane = 0; lane < control.Count; ++lane)
                        {
                            result.Statistics.PrimaryRays += work[lane].PrimaryRays;
                            result.Statistics.ShadowRays += work[lane].ShadowRays;
                            result.Statistics.ReflectionRays += work[lane].ReflectionRays;
                            result.Statistics.PrimitiveTests += work[lane].PrimitiveTests;
                        }
                    }
                }
                return true;
            };
            for (control.Order = 0; control.Order < settings.MaxReflectionBounces; ++control.Order)
            {
                if (!stage(bake, bakeArguments, control.Surfels,
                           control.Order ? control.GatherSamples : control.DirectSamples))
                    return false;
                // Every surfel and sample window has finished before the next
                // order can read this buffer. Total receives each order once.
                std::swap(previous, next);
            }
            return stage(camera, cameraArguments, pixels, settings.SamplesPerPixel);
        }

        bool MicropolygonPassNative(MicropolygonData::Controls controls,
                                    const std::vector<MicropolygonData::Patch>& patches,
                                    const std::vector<MicropolygonData::Range>& ranges,
                                    std::vector<uint32_t>& words, const Detail::CancellationProbe& cancelled,
                                    std::string& error) const
        {
            namespace M = MicropolygonData;
            if (!Check(Api.CtxSetCurrent(Context), "Selecting CUDA dicing context", error))
                return false;
            if (!DicingModule &&
                !Check(Api.ModuleLoadDataEx(&DicingModule, Detail::CudaMicropolygonPtx, 0, nullptr, nullptr),
                       "Loading CUDA dicing program", error))
                return false;
            CUfunction kernel{};
            if (!Check(Api.ModuleGetFunction(&kernel, DicingModule, "DiceMicropolygons"),
                       "Resolving CUDA dicing kernel", error))
                return false;
            constexpr uint32_t Patches = 0, Ranges = 1, Output = 2, BufferCount = 3;
            struct Storage
            {
                const Impl& Owner;
                DeviceBuffer Buffers[BufferCount];
                ~Storage()
                {
                    Owner.Api.StreamSynchronize(Owner.Stream);
                    for (auto& buffer : Buffers)
                        Owner.FreeBuffer(buffer);
                }
            } storage{*this};
            auto& buffers = storage.Buffers;
            const M::Range empty{};
            const size_t outputBytes = words.size() * sizeof(uint32_t);
            if (!UploadBuffer(buffers[Patches], patches.data(), patches.size() * sizeof(M::Patch), cancelled,
                              error) ||
                !UploadBuffer(buffers[Ranges], ranges.empty() ? &empty : ranges.data(),
                              std::max(size_t(1), ranges.size()) * sizeof(M::Range), cancelled, error) ||
                !AllocateBuffer(buffers[Output], outputBytes, error) ||
                !Check(Api.MemsetD8Async(buffers[Output].Pointer, 0, outputBytes, Stream),
                       "Clearing CUDA dicing output", error))
                return false;
            constexpr uint32_t BatchItems = 16384, Threads = 64;
            const uint32_t population = controls.Dispatch.Y;
            for (controls.Dispatch.X = 0; controls.Dispatch.X < population; controls.Dispatch.X += BatchItems)
            {
                if (cancelled())
                    return false;
                controls.Dispatch.Y = std::min(BatchItems, population - controls.Dispatch.X);
                void* arguments[]{&controls, &buffers[Patches].Pointer, &buffers[Ranges].Pointer,
                                  &buffers[Output].Pointer};
                if (!Check(Api.LaunchKernel(kernel, (controls.Dispatch.Y + Threads - 1) / Threads, 1, 1,
                                            Threads, 1, 1, 0, Stream, arguments, nullptr),
                           "Dispatching CUDA micropolygons", error) ||
                    !Check(Api.StreamSynchronize(Stream), "Waiting for CUDA dicing batch", error))
                    return false;
            }
            if (cancelled() ||
                !Check(Api.MemcpyDtoHAsync(words.data(), buffers[Output].Pointer, outputBytes, Stream),
                       "Reading CUDA micropolygons", error) ||
                !Check(Api.StreamSynchronize(Stream), "Waiting for CUDA dicing output", error))
                return false;
            return !cancelled();
        }

        bool ReconstructTemporalNative(
            TemporalData::Controls controls, const std::vector<TemporalPixel>& current,
            const std::vector<GeometryGuide>& guides, const std::vector<TemporalPixel>& previous,
            const std::vector<GeometryGuide>& oldGuides, const Detail::CancellationProbe& cancelled,
            std::vector<TemporalPixel>& result, Image& display, std::string& error) const
        {
            if (!Check(Api.CtxSetCurrent(Context), "cuCtxSetCurrent(temporal reconstruction)", error))
                return false;
            if (!TemporalDenoiseModule &&
                !Check(Api.ModuleLoadDataEx(&TemporalDenoiseModule, Detail::CudaTemporalDenoisePtx, 0,
                                            nullptr, nullptr),
                       "cuModuleLoadDataEx(temporal reconstruction)", error))
                return false;
            CUfunction kernel{};
            if (!Check(Api.ModuleGetFunction(&kernel, TemporalDenoiseModule, "ArvoTemporalDenoise"),
                       "cuModuleGetFunction(temporal reconstruction)", error))
                return false;
            const uint32_t count = controls.Width * controls.Height;
            const size_t colorBytes = size_t(count) * sizeof(TemporalPixel),
                         displayBytes = size_t(count) * sizeof(uint32_t);
            std::vector<TemporalPixel> filtered(count);
            Image encoded(controls.Width, controls.Height);
            const TemporalPixel emptyColor{};
            const GeometryGuide emptyGuide{};
            uint32_t failure = 0;
            constexpr size_t Current = 0, Guides = 1, Previous = 2, OldGuides = 3, ColorOutput = 4,
                             Display = 5, Errors = 6, BufferCount = 7;
            struct Storage
            {
                const Impl& Owner;
                DeviceBuffer Buffers[BufferCount];
                ~Storage()
                {
                    Owner.Api.StreamSynchronize(Owner.Stream);
                    for (auto& buffer : Buffers)
                        Owner.FreeBuffer(buffer);
                }
            } storage{*this};
            auto& buffers = storage.Buffers;
            if (!UploadBuffer(buffers[Current], current.data(), current.size() * sizeof(current[0]),
                              cancelled, error) ||
                !UploadBuffer(buffers[Guides], guides.data(), guides.size() * sizeof(guides[0]), cancelled,
                              error) ||
                !UploadBuffer(buffers[Previous], previous.empty() ? &emptyColor : previous.data(),
                              previous.empty() ? sizeof(emptyColor) : previous.size() * sizeof(previous[0]),
                              cancelled, error) ||
                !UploadBuffer(buffers[OldGuides], oldGuides.empty() ? &emptyGuide : oldGuides.data(),
                              oldGuides.empty() ? sizeof(emptyGuide)
                                                : oldGuides.size() * sizeof(oldGuides[0]),
                              cancelled, error) ||
                !AllocateBuffer(buffers[ColorOutput], colorBytes, error) ||
                !AllocateBuffer(buffers[Display], displayBytes, error) ||
                !AllocateBuffer(buffers[Errors], sizeof(failure), error) ||
                !Check(Api.MemsetD8Async(buffers[Errors].Pointer, 0, sizeof(failure), Stream),
                       "Clearing CUDA temporal status", error))
                return false;
            constexpr uint32_t Batch = 16384, Threads = 64;
            for (controls.Begin = 0; controls.Begin < count; controls.Begin += Batch)
            {
                if (cancelled())
                    return false;
                controls.Count = std::min(Batch, count - controls.Begin);
                void* arguments[] = {&buffers[Current].Pointer,     &buffers[Guides].Pointer,
                                     &buffers[Previous].Pointer,    &buffers[OldGuides].Pointer,
                                     &buffers[ColorOutput].Pointer, &buffers[Display].Pointer,
                                     &buffers[Errors].Pointer,      &controls};
                if (!Check(Api.LaunchKernel(kernel, (controls.Count + Threads - 1) / Threads, 1, 1, Threads,
                                            1, 1, 0, Stream, arguments, nullptr),
                           "Launching CUDA temporal reconstruction", error) ||
                    !Check(Api.MemcpyDtoHAsync(&failure, buffers[Errors].Pointer, sizeof(failure), Stream),
                           "Reading CUDA temporal status", error) ||
                    !Check(Api.StreamSynchronize(Stream), "Waiting for CUDA temporal reconstruction", error))
                    return false;
                if (failure)
                {
                    error = "CUDA temporal reconstruction reported invalid configuration or radiance";
                    return false;
                }
            }
            if (cancelled())
                return false;
            if (!Check(Api.MemcpyDtoHAsync(filtered.data(), buffers[ColorOutput].Pointer, colorBytes, Stream),
                       "Reading CUDA temporal radiance", error) ||
                !Check(Api.MemcpyDtoHAsync(encoded.GetPixelsRGBA8().data(), buffers[Display].Pointer,
                                           displayBytes, Stream),
                       "Reading CUDA temporal display", error) ||
                !Check(Api.StreamSynchronize(Stream), "Waiting for CUDA temporal readback", error))
                return false;
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
            const uint32_t count = controls.Width * controls.Height;
            if (!Check(Api.CtxSetCurrent(Context), "cuCtxSetCurrent(geometry guides)", error))
                return false;
            if (!GeometryGuidesModule &&
                !Check(Api.ModuleLoadDataEx(&GeometryGuidesModule, Detail::CudaGeometryGuidesPtx, 0, nullptr,
                                            nullptr),
                       "cuModuleLoadDataEx(geometry guides)", error))
                return false;
            const char* entry = Generate ? "ArvoGenerateGeometryGuideRays" : "ArvoResolveGeometryGuideHits";
            CUfunction kernel{};
            if (!Check(Api.ModuleGetFunction(&kernel, GeometryGuidesModule, entry),
                       "cuModuleGetFunction(geometry guides)", error))
                return false;
            std::vector<Output> output(count);
            uint32_t failure = 0;
            struct Storage
            {
                const Impl& Owner;
                DeviceBuffer Rays, Hits, OutputBuffer, Errors;
                ~Storage()
                {
                    Owner.Api.StreamSynchronize(Owner.Stream);
                    Owner.FreeBuffer(Rays);
                    Owner.FreeBuffer(Hits);
                    Owner.FreeBuffer(OutputBuffer);
                    Owner.FreeBuffer(Errors);
                }
            } storage{*this};
            if constexpr (!Generate)
            {
                if (rays.size() != count || hits.size() != count)
                {
                    error = "CUDA geometry guide input extent mismatch";
                    return false;
                }
                if (!UploadBuffer(storage.Rays, rays.data(), rays.size() * sizeof(rays[0]), cancelled,
                                  error) ||
                    !UploadBuffer(storage.Hits, hits.data(), hits.size() * sizeof(hits[0]), cancelled, error))
                    return false;
            }
            const size_t bytes = output.size() * sizeof(Output);
            if (!AllocateBuffer(storage.OutputBuffer, bytes, error) ||
                !AllocateBuffer(storage.Errors, sizeof(failure), error) ||
                !Check(Api.MemsetD8Async(storage.Errors.Pointer, 0, sizeof(failure), Stream),
                       "Clearing CUDA geometry guide errors", error))
                return false;
            constexpr uint32_t Threads = 64, Batch = 16384;
            for (controls.Begin = 0; controls.Begin < count; controls.Begin += Batch)
            {
                if (cancelled())
                    return false;
                controls.Count = std::min(Batch, count - controls.Begin);
                void* generateArgs[]{&storage.OutputBuffer.Pointer, &storage.Errors.Pointer, &controls};
                void* resolveArgs[]{&storage.Rays.Pointer, &storage.Hits.Pointer,
                                    &storage.OutputBuffer.Pointer, &storage.Errors.Pointer, &controls};
                if (!Check(Api.LaunchKernel(kernel, (controls.Count + Threads - 1) / Threads, 1, 1, Threads,
                                            1, 1, 0, Stream, Generate ? generateArgs : resolveArgs, nullptr),
                           "CUDA geometry guide dispatch", error) ||
                    !Check(Api.MemcpyDtoHAsync(&failure, storage.Errors.Pointer, sizeof(failure), Stream),
                           "Reading CUDA geometry guide status", error) ||
                    !Check(Api.StreamSynchronize(Stream), "Waiting for CUDA geometry guides", error))
                    return false;
                if (failure)
                {
                    error = "CUDA geometry guides encountered invalid dispatch or geometry";
                    return false;
                }
            }
            if (!Check(Api.MemcpyDtoHAsync(output.data(), storage.OutputBuffer.Pointer, bytes, Stream),
                       "Reading CUDA geometry guides", error) ||
                !Check(Api.StreamSynchronize(Stream), "Waiting for CUDA geometry guide readback", error))
                return false;
            if (cancelled())
                return false;
            result = std::move(output);
            return true;
        }

        bool AllocateDenoisedNative(const std::vector<AdaptiveReconstructionData::Pixel>& input,
            AdaptiveReconstructionData::Controls controls, const Detail::CancellationProbe& cancellation,
            std::vector<uint32_t>& result, std::string& error) const
        {
            if (!Check(Api.CtxSetCurrent(Context), "cuCtxSetCurrent(adaptive denoising)", error)) return false;
            if (!AdaptiveDenoiseModule && !Check(Api.ModuleLoadDataEx(&AdaptiveDenoiseModule,
                Detail::CudaAdaptiveDenoisePtx, 0, nullptr, nullptr), "cuModuleLoadDataEx(adaptive denoising)", error)) return false;
            CUfunction allocate{};
            if (!Check(Api.ModuleGetFunction(&allocate, AdaptiveDenoiseModule, "ArvoAdaptiveDenoise"),
                "cuModuleGetFunction(adaptive denoising)", error)) return false;
            struct Storage
            {
                const Impl& Owner;
                DeviceBuffer Input, Output;
                ~Storage()
                {
                    Owner.Api.StreamSynchronize(Owner.Stream);
                    Owner.FreeBuffer(Input); Owner.FreeBuffer(Output);
                }
            } storage{*this};
            const size_t bytes = input.size() * sizeof(uint32_t);
            if (!UploadBuffer(storage.Input, input.data(), input.size() * sizeof(input.front()), cancellation, error) ||
                !AllocateBuffer(storage.Output, bytes, error)) return false;
            constexpr uint32_t Batch = 16384, Threads = 64;
            const auto pixels = uint32_t(input.size());
            for (controls.Begin = 0; controls.Begin < pixels; controls.Begin += Batch)
            {
                if (cancellation()) return false;
                controls.Count = std::min(Batch, pixels - controls.Begin);
                void* arguments[]{&storage.Input.Pointer, &storage.Output.Pointer, &controls};
                if (!Check(Api.LaunchKernel(allocate, (controls.Count + Threads - 1) / Threads, 1, 1,
                    Threads, 1, 1, 0, Stream, arguments, nullptr), "cuLaunchKernel(adaptive denoising)", error) ||
                    !Check(Api.StreamSynchronize(Stream), "cuStreamSynchronize(adaptive denoising)", error)) return false;
            }
            std::vector<uint32_t> candidate(input.size());
            if (!Check(Api.MemcpyDtoHAsync(candidate.data(), storage.Output.Pointer, bytes, Stream),
                "cuMemcpyDtoHAsync(adaptive allocation)", error) ||
                !Check(Api.StreamSynchronize(Stream), "cuStreamSynchronize(allocation readback)", error) || cancellation()) return false;
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
                !ValidateReconstructionGuides(film, settings.Reconstruction, guides, error) ||
                !Check(Api.CtxSetCurrent(Context), "cuCtxSetCurrent(reconstruction)", error))
                return false;
            if (!SpatialDenoiseModule &&
                !Check(Api.ModuleLoadDataEx(&SpatialDenoiseModule, Detail::CudaSpatialDenoisePtx, 0, nullptr,
                                            nullptr),
                       "cuModuleLoadDataEx(spatial denoising)", error))
                return false;
            CUfunction filter{};
            if (!Check(Api.ModuleGetFunction(&filter, SpatialDenoiseModule, "ArvoSpatialDenoise"),
                       "cuModuleGetFunction(spatial denoising)", error))
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
            ReconstructedImage output{film.Width, film.Height, {}, Image(film.Width, film.Height)};
            std::vector<Vec4> colors(input.size());
            uint32_t failure = 0;
            struct Storage
            {
                const Impl& Owner;
                DeviceBuffer Raw, Previous, Next, Display, Error;
                ~Storage()
                {
                    // Pending work retains all allocations until the stream retires.
                    Owner.Api.StreamSynchronize(Owner.Stream);
                    Owner.FreeBuffer(Raw);
                    Owner.FreeBuffer(Previous);
                    Owner.FreeBuffer(Next);
                    Owner.FreeBuffer(Display);
                    Owner.FreeBuffer(Error);
                }
            } storage{*this};
            const size_t colorBytes = input.size() * sizeof(Vec4),
                         displayBytes = input.size() * sizeof(uint32_t);
            if (!UploadBuffer(storage.Raw, input.data(), input.size() * sizeof(Data::Pixel), cancellation,
                              error) ||
                !AllocateBuffer(storage.Previous, colorBytes, error) ||
                !AllocateBuffer(storage.Next, colorBytes, error) ||
                !AllocateBuffer(storage.Display, displayBytes, error) ||
                !AllocateBuffer(storage.Error, sizeof(uint32_t), error) ||
                !Check(Api.MemsetD8Async(storage.Error.Pointer, 0, sizeof(uint32_t), Stream),
                       "cuMemsetD8Async(denoise status)", error))
                return false;
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
            const uint32_t count = uint32_t(input.size());
            for (controls.Pass = 0; controls.Pass < options.Passes; ++controls.Pass)
            {
                for (controls.Begin = 0; controls.Begin < count; controls.Begin += Batch)
                {
                    if (cancellation())
                        return false;
                    controls.Count = std::min(Batch, count - controls.Begin);
                    void* arguments[]{&storage.Raw.Pointer,   &storage.Previous.Pointer,
                                      &storage.Next.Pointer,  &storage.Display.Pointer,
                                      &storage.Error.Pointer, &controls};
                    const uint32_t groups = (controls.Count + Threads - 1) / Threads;
                    failure = 0;
                    if (!Check(Api.LaunchKernel(filter, groups, 1, 1, Threads, 1, 1, 0, Stream, arguments,
                                                nullptr),
                               "cuLaunchKernel(spatial denoising)", error) ||
                        !Check(Api.MemcpyDtoHAsync(&failure, storage.Error.Pointer, sizeof(failure), Stream),
                               "cuMemcpyDtoHAsync(denoise status)", error) ||
                        !Check(Api.StreamSynchronize(Stream), "cuStreamSynchronize(spatial denoising)",
                               error))
                        return false;
                    if (failure)
                    {
                        error = "CUDA spatial denoising reported invalid state";
                        return false;
                    }
                }
                std::swap(storage.Previous, storage.Next);
            }
            if (!Check(Api.MemcpyDtoHAsync(colors.data(), storage.Previous.Pointer, colorBytes, Stream),
                       "cuMemcpyDtoHAsync(reconstructed HDR)", error) ||
                !Check(Api.MemcpyDtoHAsync(output.Display.GetPixelsRGBA8().data(), storage.Display.Pointer,
                                           displayBytes, Stream),
                       "cuMemcpyDtoHAsync(reconstructed display)", error) ||
                !Check(Api.StreamSynchronize(Stream), "cuStreamSynchronize(reconstruction readback)", error))
                return false;
            output.Pixels.reserve(colors.size());
            for (const auto& pixel : colors)
            {
                if (cancellation())
                    return false;
                output.Pixels.push_back({pixel.X, pixel.Y, pixel.Z});
            }
            result = std::move(output);
            return true;
        }

        RenderResult RenderNative(
            const Scene& scene,
            const std::shared_ptr<const Scene>& owner,
            const Camera& camera,
            const RenderSettings& settings,
            const Detail::CancellationProbe& cancellation) const
        {
            RenderResult result;
            if (!ValidateInput(scene, camera, settings, result.Error)) return result;
            if (!Detail::ValidateMaterialTransport(scene, BackendType::CUDA,
                    settings.Integrator, result.Error)) return result;
            if (!Check(Api.CtxSetCurrent(Context), "cuCtxSetCurrent", result.Error)) return result;
            const auto buildBegin = std::chrono::steady_clock::now();
            if (!EnsureScene(scene, owner, {settings.Acceleration, settings.BvhPolicy}, cancellation,
                             result.Error,
                             settings.Radiometry == RadiometryMode::Spectral ? &settings.Medium : nullptr))
            {
                if (cancellation()) return Detail::MakeCancelledResult();
                return result;
            }
            const auto buildEnd = std::chrono::steady_clock::now();
            if (cancellation()) return Detail::MakeCancelledResult();

            const std::uint32_t pixelCount = settings.Width * settings.Height;
            if (!EnsureFrameBuffers(pixelCount, result.Error)) return result;
            if (!Check(Api.MemsetD8Async(
                    Accumulation.Pointer, 0, static_cast<std::size_t>(pixelCount) * sizeof(Vec4), Stream),
                    "cuMemsetD8Async(accumulation)", result.Error)
                || !Check(Api.MemsetD8Async(
                    SampleCounts.Pointer, 0,
                    static_cast<std::size_t>(pixelCount) * sizeof(std::uint32_t), Stream),
                    "cuMemsetD8Async(sample counts)", result.Error)
                || !Check(Api.MemsetD8Async(
                    SampleTargets.Pointer, 0,
                    static_cast<std::size_t>(pixelCount) * sizeof(std::uint32_t), Stream),
                    "cuMemsetD8Async(sample targets)", result.Error)
                || !Check(Api.StreamSynchronize(Stream), "cuStreamSynchronize(clear)", result.Error))
            {
                return result;
            }
            if (cancellation()) return Detail::MakeCancelledResult();

            FrameConstants constants;
            BindInstances(constants);
            constants.CameraPosition = ToVec3(camera.Position);
            constants.Width = settings.Width;
            constants.FullWidth = FullImageWidth(settings);
            constants.FullHeight = FullImageHeight(settings);
            constants.TileX = settings.TileX;
            constants.TileY = settings.TileY;
            constants.Integrator = static_cast<uint32_t>(settings.Integrator);
            constexpr uint32_t SpectralTransportBit = 1u << 5;
            constants.TransportFlags =
                PackTransportFlags(settings.Transport) |
                (settings.Radiometry == RadiometryMode::Spectral ? SpectralTransportBit : 0);
            constants.RouletteStartDepth = settings.Transport.RouletteStartDepth;
            constants.SamplingSequence = static_cast<uint32_t>(settings.Sampling);
            constants.SamplingSeed = settings.SamplingSeed;
            constants.Acceleration = static_cast<std::uint32_t>(settings.Acceleration);
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
            const Vec3 up = NormalizeVector(Cross(right, forward));
            constants.CameraForward = forward;
            constants.LensRadius = camera.Lens.Radius;
            constants.FocusDistance = camera.Lens.FocusDistance;
            constants.CameraRight = right;
            constants.CameraUp = up;
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

            const auto renderBegin = std::chrono::steady_clock::now();
            std::vector<GroupStatistics> groupStatistics;
            bool cancelled = false;
            constexpr std::uint32_t PilotPhase = 1u;
            const bool variancePilot = settings.Adaptive.Enabled && !settings.ProductionBudgets;
            const std::uint32_t phaseCount = variancePilot ? 2u : 1u;
            if (settings.ProductionBudgets &&
                (!Check(Api.MemcpyHtoDAsync(SampleTargets.Pointer, settings.ProductionBudgets->Counts.data(),
                    size_t(pixelCount) * sizeof(uint32_t), Stream), "cuMemcpyHtoDAsync(production budgets)", result.Error) ||
                 !Check(Api.StreamSynchronize(Stream), "cuStreamSynchronize(production budgets)", result.Error)))
                return result;
            const bool bidirectional = settings.Integrator == Algorithm::BidirectionalPathTracing;
            const bool metropolis = settings.Integrator == Algorithm::MetropolisLightTransport;
            const bool photon = settings.Integrator == Algorithm::PhotonMapping;
            const bool progressive = settings.Integrator == Algorithm::ProgressivePhotonMapping;
            const bool vcm = settings.Integrator == Algorithm::VertexConnectionMerging;
            const bool pointGi = settings.Integrator == Algorithm::PointBasedGlobalIllumination;
            if (bidirectional && !RenderBidirectional(scene, constants, settings, cancellation, result))
            {
                if (!cancellation()) return result;
                cancelled = true;
            }
            if (metropolis && !RenderMetropolis(scene, constants, settings, cancellation, result))
            {
                if (!cancellation()) return result;
                cancelled = true;
            }
            if (photon && !RenderPhotons(scene, constants, settings, cancellation, result))
            {
                if (!cancellation()) return result;
                cancelled = true;
            }
            if (progressive && !RenderProgressivePhotons(scene, constants, settings, cancellation, result))
            {
                if (!cancellation()) return result;
                cancelled = true;
            }
            if (vcm && !RenderVcm(scene, constants, settings, cancellation, result))
            {
                if (!cancellation()) return result;
                cancelled = true;
            }
            if (pointGi && !RenderPointGi(scene, constants, settings, cancellation, result))
            {
                if (!cancellation()) return result;
                cancelled = true;
            }
            for (std::uint32_t phase = 0; !bidirectional && !metropolis && !photon && !progressive && !vcm && !pointGi && phase < phaseCount && !cancelled; ++phase)
            {
                const bool pilot = variancePilot && phase == 0u;
                constants.AdaptivePhase = pilot ? PilotPhase : uint32_t(settings.SampleStream);
                const std::uint32_t phaseSamples = pilot ? settings.Adaptive.PilotSamples : settings.SamplesPerPixel;
                for (std::uint32_t sample = 0; sample < phaseSamples && !cancelled; ++sample)
                {
                    std::uint64_t passPrimaryRays = 0;
                    for (std::uint32_t pixelBegin = 0; pixelBegin < pixelCount; pixelBegin += PixelChunkSize)
                    {
                        if (cancellation())
                        {
                            cancelled = true;
                            break;
                        }
                        constants.PixelBegin = pixelBegin;
                        constants.PixelCount = std::min(PixelChunkSize, pixelCount - pixelBegin);
                        constants.SampleBegin = sample;
                        constants.SampleCount = 1;
                        constants.GroupCount =
                            (constants.PixelCount + ThreadsPerBlock - 1) / ThreadsPerBlock;
                        if (!LaunchTrace(constants, PackMedium(settings.Medium), {camera.Motion, HasSceneMotion(scene)},
                                         groupStatistics, result.Error))
                            return result;
                        for (const GroupStatistics& statistics : groupStatistics)
                        {
                            result.Statistics.PrimaryRays += statistics.PrimaryRays;
                            passPrimaryRays += statistics.PrimaryRays;
                            if (pilot) result.Statistics.PilotRays += statistics.PrimaryRays;
                            result.Statistics.ShadowRays += statistics.ShadowRays;
                            result.Statistics.ReflectionRays += statistics.ReflectionRays;
                            result.Statistics.PrimitiveTests += statistics.PrimitiveTests;
                        }
                    }
                    // Pilot-selected targets are fixed, so an empty production
                    // pass proves every later pass is empty as well.
                    if (!pilot && passPrimaryRays == 0) break;
                }
            }

            if (cancelled || cancellation())
            {
                const auto renderEnd = std::chrono::steady_clock::now();
                result.Statistics.BuildMilliseconds =
                    std::chrono::duration<double, std::milli>(buildEnd - buildBegin).count();
                result.Statistics.RenderMilliseconds =
                    std::chrono::duration<double, std::milli>(renderEnd - renderBegin).count();
                result.Status = RenderStatus::Cancelled;
                return result;
            }

            if (!LaunchResolve(constants, pixelCount, cancellation, result.Error))
            {
                if (cancellation())
                {
                    const auto renderEnd = std::chrono::steady_clock::now();
                    result.Statistics.BuildMilliseconds =
                        std::chrono::duration<double, std::milli>(buildEnd - buildBegin).count();
                    result.Statistics.RenderMilliseconds =
                        std::chrono::duration<double, std::milli>(renderEnd - renderBegin).count();
                    result.Status = RenderStatus::Cancelled;
                }
                return result;
            }
            std::vector<std::uint32_t> packedOutput(pixelCount);
            std::vector<std::uint32_t> completedSamples(pixelCount);
            std::vector<std::uint32_t> sampleTargets(pixelCount);
            std::vector<Vec4> filmMoments(pixelCount);
            if (!DownloadBuffer(
                    packedOutput.data(), Output.Pointer,
                    static_cast<std::size_t>(pixelCount) * sizeof(std::uint32_t),
                    cancellation, result.Error)
                || !DownloadBuffer(
                    completedSamples.data(), SampleCounts.Pointer,
                    static_cast<std::size_t>(pixelCount) * sizeof(std::uint32_t),
                    cancellation, result.Error)
                || !DownloadBuffer(
                    sampleTargets.data(), SampleTargets.Pointer,
                    static_cast<std::size_t>(pixelCount) * sizeof(std::uint32_t),
                    cancellation, result.Error)
                || !DownloadBuffer(
                    filmMoments.data(), Accumulation.Pointer,
                    static_cast<std::size_t>(pixelCount) * sizeof(Vec4),
                    cancellation, result.Error))
            {
                if (cancellation()) result.Status = RenderStatus::Cancelled;
                return result;
            }
            const auto renderEnd = std::chrono::steady_clock::now();
            result.Output = Image(settings.Width, settings.Height);
            result.Film = LinearFilm(settings.Width, settings.Height);
            for (size_t pixel = 0; pixel < pixelCount; ++pixel)
            {
                const std::uint32_t minimum = settings.Adaptive.Enabled ? settings.Adaptive.MinimumSamples : settings.SamplesPerPixel;
                if (sampleTargets[pixel] < minimum || sampleTargets[pixel] > settings.SamplesPerPixel
                    || completedSamples[pixel] != sampleTargets[pixel])
                {
                    result.Error = "CUDA did not complete the selected per-pixel sample budget.";
                    return result;
                }
                const Vec4& value = filmMoments[pixel];
                result.Film.Pixels[pixel] = {Float3(value.X, value.Y, value.Z), value.W, completedSamples[pixel]};
                ++result.Statistics.CompletedPixels;
            }
            std::memcpy(
                result.Output.GetPixelsRGBA8().data(), packedOutput.data(),
                static_cast<std::size_t>(pixelCount) * sizeof(std::uint32_t));
            result.Statistics.BuildMilliseconds =
                std::chrono::duration<double, std::milli>(buildEnd - buildBegin).count();
            result.Statistics.RenderMilliseconds =
                std::chrono::duration<double, std::milli>(renderEnd - renderBegin).count();
            result.Status = cancellation() ? RenderStatus::Cancelled : RenderStatus::Completed;
            return result;
        }

        DynamicModule DriverModule;
        CudaApi Api;
        CUdevice Device = 0;
        CUcontext Context = nullptr;
        CUstream Stream = nullptr;
        CUmodule WhittedModule = nullptr;
        CUmodule PathTracingModule = nullptr;
        CUmodule DistributedRayModule = nullptr;
        CUmodule FilmModule = nullptr;
        mutable CUmodule BidirectionalModule = nullptr;
        mutable CUmodule MetropolisModule = nullptr;
        mutable CUmodule PhotonModule = nullptr;
        mutable CUmodule ProgressivePhotonModule = nullptr;
        mutable CUmodule VcmModule = nullptr;
        mutable CUmodule PointGiModule = nullptr;
        mutable CUmodule SpatialDenoiseModule = nullptr;
        mutable CUmodule AdaptiveDenoiseModule = nullptr;
        mutable CUmodule GeometryGuidesModule = nullptr;
        mutable CUmodule TemporalDenoiseModule = nullptr;
        mutable CUmodule DicingModule = nullptr;
        mutable CUfunction BidirectionalPairs = nullptr;
        mutable CUfunction BidirectionalCommit = nullptr;
        CUfunction WhittedFunction = nullptr;
        CUfunction QueryFunction = nullptr;
        CUfunction PathTracingFunction = nullptr;
        CUfunction DistributedRayFunction = nullptr;
        CUfunction ResolveFunction = nullptr;
        int DriverVersion = 0;
        int ComputeMajor = 0;
        int ComputeMinor = 0;
        std::string DeviceName;

        mutable DeviceBuffer InstanceRecords;
        mutable DeviceBuffer InstanceNodes;
        mutable DeviceBuffer InstanceError;
        mutable uint32_t InstanceNodeCount = 0;
        mutable DeviceBuffer Nodes;
        mutable DeviceBuffer PrimitiveRefs;
        mutable DeviceBuffer Triangles;
        mutable DeviceBuffer Spheres;
        mutable DeviceBuffer Materials;
        mutable DeviceBuffer Lights;
        mutable DeviceBuffer TextureDescriptors;
        mutable DeviceBuffer TextureTexels;
        mutable DeviceBuffer SpectralPalette;
        mutable DeviceBuffer SpectralReferences;
        mutable uint32_t SpectralSceneSourceCount = 0;
        mutable MediumOptions SpectralMedium;
        mutable CUmodule SpectralModule = nullptr;
        mutable CUfunction SpectralResponseFunction = nullptr, SpectralPaletteFunction = nullptr;
        mutable DeviceBuffer Constants;
        mutable DeviceBuffer Accumulation;
        mutable DeviceBuffer SampleCounts;
        mutable DeviceBuffer SampleTargets;
        mutable DeviceBuffer Output;
        mutable DeviceBuffer Statistics;
        mutable std::uint32_t NodeCount = 0;
        mutable std::uint32_t PrimitiveRefCount = 0;
        mutable std::uint32_t TriangleCount = 0;
        mutable std::uint32_t SphereCount = 0;
        mutable std::uint32_t MaterialCount = 0;
        mutable std::uint32_t LightCount = 0;
        mutable std::uint32_t TextureCount = 0;
        mutable std::uint32_t TextureTexelCount = 0;
        mutable std::uint32_t FramePixelCapacity = 0;
        mutable std::uint32_t StatisticsGroupCapacity = 0;
        mutable std::weak_ptr<const Scene> CachedOwner;
        mutable AccelerationOptions CachedAcceleration = AccelerationType::Bvh;
#endif
        bool Available = false;
        std::string AvailabilityError;
        mutable Detail::PersistentRenderQueue StreamQueue;
    };

    CudaRayTracingRHI::CudaRayTracingRHI()
        : m_Impl(std::make_unique<Impl>())
    {
    }

    CudaRayTracingRHI::~CudaRayTracingRHI() = default;

    BackendInfo CudaRayTracingRHI::GetBackendInfo() const
    {
        BackendInfo info;
        info.Type = BackendType::CUDA;
        info.Name = "CUDA";
        info.IsNative = m_Impl->Available;
        info.IsHardwareAccelerated = m_Impl->Available;
        info.IsAvailable = m_Impl->Available;
        info.Description = m_Impl->Available
#if defined(_WIN32) && defined(ARVO_CUDA_KERNELS)
            ? std::string("Native CUDA Driver API backend on ") + m_Impl->DeviceName
                + " (compute " + std::to_string(m_Impl->ComputeMajor) + '.'
                + std::to_string(m_Impl->ComputeMinor) + ')'
#else
            ? "Native CUDA Driver API backend"
#endif
            : std::string("Unavailable: ") + m_Impl->AvailabilityError;
        return info;
    }

    RayQueryResult CudaRayTracingRHI::QueryRays(const Scene& scene, const std::vector<Ray>& rays,
        AccelerationOptions acceleration,
        const CancellationToken& cancellation) const
    {
        RayQueryResult result;
        // Queries join the production queue: scene buffers and the CUDA stream
        // must never be mutated concurrently with a queued render.
        auto job = m_Impl->StreamQueue.Submit([&](const CancellationToken& queued)
        {
            const auto probe = Detail::CombineCancellation(queued, cancellation);
            if (probe()) return Detail::MakeCancelledResult();
            if (!m_Impl->Available) return Detail::MakeFailedResult("CUDA is unavailable: " + m_Impl->AvailabilityError);
#if defined(_WIN32) && defined(ARVO_CUDA_KERNELS)
            result = m_Impl->QueryNative(scene, rays, acceleration, probe);
            RenderResult completion;
            completion.Status = result.Status;
            completion.Error = result.Error;
            return completion;
#else
            return Detail::MakeFailedResult("CUDA queries require the native CUDA backend.");
#endif
        });
        const auto completion = job.GetResult();
        result.Status = completion.Status;
        result.Error = completion.Error;
        return result;
    }

    MicropolygonResult CudaRayTracingRHI::PrepareMicropolygons(const Scene& scene, const Camera& camera,
                                                               const RenderSettings& settings,
                                                               const CancellationToken& cancellation) const
    {
        MicropolygonResult result;
        const auto task =
            m_Impl->StreamQueue
                .Submit([&](const CancellationToken& queued) {
                    const auto stopped = Detail::CombineCancellation(queued, cancellation);
                    if (stopped())
                        return Detail::MakeCancelledResult();
                    if (!m_Impl->Available)
                        return Detail::MakeFailedResult("CUDA is unavailable: " + m_Impl->AvailabilityError);
#if defined(_WIN32) && defined(ARVO_CUDA_KERNELS)
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
                    return Detail::MakeFailedResult("Micropolygons require native CUDA");
#endif
                })
                .GetResult();
        result.Status = task.Status;
        result.Error = task.Error;
        return result;
    }

    TemporalResult CudaRayTracingRHI::ReconstructTemporal(const TemporalFrame& frame,
                                                          const TemporalHistory* history,
                                                          const TemporalOptions& options,
                                                          const CancellationToken& cancellation) const
    {
        TemporalResult result;
        const auto task =
            m_Impl->StreamQueue
                .Submit([&](const CancellationToken& queued) {
                    const auto stopped = Detail::CombineCancellation(queued, cancellation);
                    if (stopped())
                        return Detail::MakeCancelledResult();
                    if (!m_Impl->Available)
                        return Detail::MakeFailedResult("CUDA is unavailable: " + m_Impl->AvailabilityError);
#if defined(_WIN32) && defined(ARVO_CUDA_KERNELS)
                    result = Detail::ReconstructTemporal(
                        BackendType::CUDA, frame, history, options, stopped,
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
                    return Detail::MakeFailedResult("Temporal reconstruction requires native CUDA");
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

    GeometryGuideResult CudaRayTracingRHI::CaptureGeometryGuides(const Scene& scene, const Camera& camera,
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
        const auto completion =
            m_Impl->StreamQueue
                .Submit([&](const CancellationToken& queued) {
                    const auto stopped = Detail::CombineCancellation(queued, cancellation);
                    if (stopped())
                        return Detail::MakeCancelledResult();
                    if (!m_Impl->Available)
                        return Detail::MakeFailedResult("CUDA is unavailable: " + m_Impl->AvailabilityError);
#if defined(_WIN32) && defined(ARVO_CUDA_KERNELS)
                    result = m_Impl->CaptureGuidesNative(scene, camera, settings, stopped);
                    RenderResult task;
                    task.Status = result.Status;
                    task.Error = result.Error;
                    return task;
#else
                    return Detail::MakeFailedResult("Geometry guides require the native CUDA backend");
#endif
                })
                .GetResult();
        result.Status = completion.Status;
        result.Error = completion.Error;
        if (result.Status != RenderStatus::Completed)
            result.Image = {};
        return result;
    }

    RenderResult CudaRayTracingRHI::Render(
        const Scene& scene,
        const Camera& camera,
        const RenderSettings& settings,
        const CancellationToken& cancellation) const
    {
        return m_Impl->Submit(&scene, nullptr, camera, settings, cancellation).GetResult();
    }

    RenderJob CudaRayTracingRHI::RenderAsync(
        Scene scene,
        Camera camera,
        RenderSettings settings) const
    {
        return RenderAsync(std::make_shared<Scene>(std::move(scene)), camera, settings);
    }

    RenderJob CudaRayTracingRHI::RenderAsync(
        std::shared_ptr<const Scene> scene,
        Camera camera,
        RenderSettings settings) const
    {
        const Scene* pointer = scene.get();
        return m_Impl->Submit(pointer, std::move(scene), camera, settings);
    }

    void CudaRayTracingRHI::WaitIdle() const
    {
        m_Impl->StreamQueue.WaitIdle();
    }
}
}
