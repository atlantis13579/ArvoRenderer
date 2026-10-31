#include "CudaRayTracingRHI.h"

#include "Renderer/RayTracingBackendCommon.h"
#include "CudaRayTracingPtx.h"

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

#if defined(_WIN32)
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
#if defined(_WIN32)
        constexpr std::uint32_t SphereBit = 0x80000000u;
        constexpr std::uint32_t LeafPrimitiveCount = 8;
        constexpr std::uint32_t MaximumBuildDepth = 48;
        constexpr std::uint32_t SahBinCount = 16;
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
            std::uint32_t Padding1 = 0;
            std::uint32_t Padding2 = 0;
        };

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
            std::uint32_t IntegratorPadding0 = 0;
            std::uint32_t IntegratorPadding1 = 0;
            std::uint32_t IntegratorPadding2 = 0;
        };

        struct alignas(16) GroupStatistics
        {
            std::uint64_t PrimaryRays = 0;
            std::uint64_t ShadowRays = 0;
            std::uint64_t ReflectionRays = 0;
            std::uint64_t PrimitiveTests = 0;
        };

        static_assert(sizeof(Vec4) == 16, "CUDA Vec4 ABI mismatch.");
        static_assert(sizeof(PackedBvhNode) == 32, "CUDA BVH ABI mismatch.");
        static_assert(sizeof(PackedPrimitiveRef) == 4, "CUDA primitive ABI mismatch.");
        static_assert(sizeof(PackedTriangle) == 192, "CUDA triangle ABI mismatch.");
        static_assert(
            offsetof(PackedTriangle, TexCoord01) == 144
                && offsetof(PackedTriangle, TexCoord2) == 160
                && offsetof(PackedTriangle, MaterialIndex) == 176
                && offsetof(PackedTriangle, Flags) == 180,
            "CUDA triangle field ABI mismatch.");
        static_assert(sizeof(PackedSphere) == 32, "CUDA sphere ABI mismatch.");
        static_assert(sizeof(PackedMaterial) == 80, "CUDA material ABI mismatch.");
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
            sizeof(FrameConstants) == 192
                && offsetof(FrameConstants, ShadowStrength) == 124,
            "CUDA constants ABI mismatch.");
        static_assert(offsetof(FrameConstants, FullWidth) == 160, "Tile constant ABI mismatch.");
        static_assert(offsetof(FrameConstants, Integrator) == 176, "Integrator constant ABI mismatch.");
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

            float SurfaceArea() const
            {
                const Vec3 extent = Subtract(Maximum, Minimum);
                return 2.0f * (extent.X * extent.Y + extent.Y * extent.Z + extent.Z * extent.X);
            }
        };

        float Component(const Vec3& value, std::uint32_t axis)
        {
            return axis == 0 ? value.X : (axis == 1 ? value.Y : value.Z);
        }

        struct BuildPrimitive
        {
            Bounds BoundingBox;
            Vec3 Centroid;
            PackedPrimitiveRef Reference;
        };

        struct BuildNode
        {
            Bounds BoundingBox;
            std::uint32_t Left = 0;
            std::uint32_t Right = 0;
            std::uint32_t FirstPrimitive = 0;
            std::uint32_t PrimitiveCount = 0;

            bool IsLeaf() const { return PrimitiveCount != 0; }
        };

        struct PackedScene
        {
            std::vector<PackedBvhNode> Nodes;
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
            std::string& error)
        {
            if (!ValidateTileSettings(settings, error)) return false;
            if (settings.Width == 0 || settings.Height == 0
                || settings.Width > 16384 || settings.Height > 16384)
            {
                error = "CUDA image dimensions must be in the range [1, 16384].";
                return false;
            }
            if (settings.SamplesPerPixel == 0 || settings.SamplesPerPixel > 4096
                || settings.MaxReflectionBounces > 32
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
            for (const Triangle& triangle : scene.Triangles)
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
            for (const Sphere& sphere : scene.Spheres)
            {
                if (sphere.MaterialIndex >= scene.Materials.size() || !IsFinite(sphere.Center)
                    || !std::isfinite(sphere.Radius) || sphere.Radius <= 0.0f)
                {
                    error = "CUDA sphere data is invalid.";
                    return false;
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

            bool Pack(const Scene& scene, PackedScene& output, std::string& error)
            {
                output = {};
                output.Triangles.reserve(scene.Triangles.size());
                output.Spheres.reserve(scene.Spheres.size());
                output.Materials.reserve(scene.Materials.size());
                output.Lights.reserve(
                    scene.DirectionalLights.size()
                    + scene.PointLights.size()
                    + scene.RectAreaLights.size());
                if (!Detail::BuildRayTextureAtlas(
                        scene, output.TextureAtlas, error))
                {
                    return false;
                }
                m_Primitives.reserve(scene.Triangles.size() + scene.Spheres.size());

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
                for (const Material& source : scene.Materials)
                {
                    PackedMaterial material;
                    material.AlbedoReflectivity = ToVec4(source.Albedo, source.Reflectivity);
                    material.AmbientShininess = ToVec4(source.AmbientColor, source.Shininess);
                    material.Specular = ToVec4(source.SpecularColor);
                    material.Emission = ToVec4(source.Emission);
                    material.DiffuseTextureIndex = source.DiffuseTextureIndex;
                    material.SampledEmitter = source.SampledEmitter ? 1u : 0u;
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

                if (!m_Primitives.empty())
                {
                    m_BuildNodes.reserve(m_Primitives.size() * 2);
                    const std::uint32_t root = BuildRecursive(
                        0, static_cast<std::uint32_t>(m_Primitives.size()), 0);
                    if (IsCancelled()) return Cancel(error);
                    output.PrimitiveRefs.reserve(m_Primitives.size());
                    for (const BuildPrimitive& primitive : m_Primitives)
                    {
                        output.PrimitiveRefs.push_back(primitive.Reference);
                    }
                    output.Nodes.reserve(m_BuildNodes.size());
                    Flatten(root, output.Nodes);
                    if (output.Nodes.size() > std::numeric_limits<std::uint32_t>::max())
                    {
                        error = "CUDA BVH exceeds its 32-bit node limit.";
                        return false;
                    }
                }
                return true;
            }

        private:
            struct Bin
            {
                Bounds BoundingBox;
                std::uint32_t Count = 0;
            };

            bool IsCancelled() const
            {
                return m_Cancellation && m_Cancellation();
            }

            bool Cancel(std::string& error) const
            {
                error = "CUDA scene packing was cancelled.";
                return false;
            }

            std::uint32_t BuildRecursive(std::uint32_t begin, std::uint32_t end, std::uint32_t depth)
            {
                const std::uint32_t nodeIndex = static_cast<std::uint32_t>(m_BuildNodes.size());
                m_BuildNodes.emplace_back();
                Bounds primitiveBounds;
                Bounds centroidBounds;
                for (std::uint32_t index = begin; index < end; ++index)
                {
                    primitiveBounds.Expand(m_Primitives[index].BoundingBox);
                    centroidBounds.Expand(m_Primitives[index].Centroid);
                }
                m_BuildNodes[nodeIndex].BoundingBox = primitiveBounds;
                const std::uint32_t count = end - begin;
                if (count <= LeafPrimitiveCount || depth >= MaximumBuildDepth || IsCancelled())
                {
                    m_BuildNodes[nodeIndex].FirstPrimitive = begin;
                    m_BuildNodes[nodeIndex].PrimitiveCount = count;
                    return nodeIndex;
                }

                float bestCost = std::numeric_limits<float>::max();
                std::uint32_t bestAxis = 0;
                std::uint32_t bestSplit = 0;
                bool foundSplit = false;
                for (std::uint32_t axis = 0; axis < 3; ++axis)
                {
                    const float minimum = Component(centroidBounds.Minimum, axis);
                    const float maximum = Component(centroidBounds.Maximum, axis);
                    const float extent = maximum - minimum;
                    if (extent <= 1.0e-6f) continue;
                    std::array<Bin, SahBinCount> bins{};
                    for (std::uint32_t index = begin; index < end; ++index)
                    {
                        const float normalized =
                            (Component(m_Primitives[index].Centroid, axis) - minimum) / extent;
                        const std::uint32_t binIndex = std::min(
                            static_cast<std::uint32_t>(normalized * SahBinCount), SahBinCount - 1);
                        ++bins[binIndex].Count;
                        bins[binIndex].BoundingBox.Expand(m_Primitives[index].BoundingBox);
                    }
                    std::array<Bounds, SahBinCount> leftBounds{};
                    std::array<Bounds, SahBinCount> rightBounds{};
                    std::array<std::uint32_t, SahBinCount> leftCounts{};
                    std::array<std::uint32_t, SahBinCount> rightCounts{};
                    Bounds runningLeft;
                    std::uint32_t runningLeftCount = 0;
                    for (std::uint32_t bin = 0; bin < SahBinCount; ++bin)
                    {
                        if (bins[bin].Count != 0) runningLeft.Expand(bins[bin].BoundingBox);
                        runningLeftCount += bins[bin].Count;
                        leftBounds[bin] = runningLeft;
                        leftCounts[bin] = runningLeftCount;
                    }
                    Bounds runningRight;
                    std::uint32_t runningRightCount = 0;
                    for (std::uint32_t bin = SahBinCount; bin-- > 0;)
                    {
                        if (bins[bin].Count != 0) runningRight.Expand(bins[bin].BoundingBox);
                        runningRightCount += bins[bin].Count;
                        rightBounds[bin] = runningRight;
                        rightCounts[bin] = runningRightCount;
                    }
                    for (std::uint32_t split = 0; split + 1 < SahBinCount; ++split)
                    {
                        if (leftCounts[split] == 0 || rightCounts[split + 1] == 0) continue;
                        const float cost = leftBounds[split].SurfaceArea() * leftCounts[split]
                            + rightBounds[split + 1].SurfaceArea() * rightCounts[split + 1];
                        if (cost < bestCost)
                        {
                            bestCost = cost;
                            bestAxis = axis;
                            bestSplit = split;
                            foundSplit = true;
                        }
                    }
                }

                std::uint32_t middle = begin;
                if (foundSplit)
                {
                    const float minimum = Component(centroidBounds.Minimum, bestAxis);
                    const float extent = Component(centroidBounds.Maximum, bestAxis) - minimum;
                    const auto middleIterator = std::partition(
                        m_Primitives.begin() + begin,
                        m_Primitives.begin() + end,
                        [minimum, extent, bestAxis, bestSplit](const BuildPrimitive& primitive)
                        {
                            const float normalized =
                                (Component(primitive.Centroid, bestAxis) - minimum) / extent;
                            const std::uint32_t binIndex = std::min(
                                static_cast<std::uint32_t>(normalized * SahBinCount), SahBinCount - 1);
                            return binIndex <= bestSplit;
                        });
                    middle = static_cast<std::uint32_t>(middleIterator - m_Primitives.begin());
                }
                if (middle == begin || middle == end)
                {
                    const Vec3 extent = Subtract(centroidBounds.Maximum, centroidBounds.Minimum);
                    bestAxis = extent.Y > extent.X ? 1u : 0u;
                    if (Component(extent, 2) > Component(extent, bestAxis)) bestAxis = 2;
                    middle = begin + count / 2;
                    std::nth_element(
                        m_Primitives.begin() + begin,
                        m_Primitives.begin() + middle,
                        m_Primitives.begin() + end,
                        [bestAxis](const BuildPrimitive& left, const BuildPrimitive& right)
                        {
                            return Component(left.Centroid, bestAxis) < Component(right.Centroid, bestAxis);
                        });
                }
                const std::uint32_t left = BuildRecursive(begin, middle, depth + 1);
                const std::uint32_t right = BuildRecursive(middle, end, depth + 1);
                m_BuildNodes[nodeIndex].Left = left;
                m_BuildNodes[nodeIndex].Right = right;
                return nodeIndex;
            }

            void Flatten(std::uint32_t buildIndex, std::vector<PackedBvhNode>& output)
            {
                const std::uint32_t outputIndex = static_cast<std::uint32_t>(output.size());
                output.emplace_back();
                const BuildNode& source = m_BuildNodes[buildIndex];
                PackedBvhNode& destination = output[outputIndex];
                destination.MinX = source.BoundingBox.Minimum.X;
                destination.MinY = source.BoundingBox.Minimum.Y;
                destination.MinZ = source.BoundingBox.Minimum.Z;
                destination.MaxX = source.BoundingBox.Maximum.X;
                destination.MaxY = source.BoundingBox.Maximum.Y;
                destination.MaxZ = source.BoundingBox.Maximum.Z;
                if (source.IsLeaf())
                {
                    destination.FirstOrEscape = source.FirstPrimitive;
                    destination.PrimitiveCount = source.PrimitiveCount;
                    return;
                }
                Flatten(source.Left, output);
                Flatten(source.Right, output);
                output[outputIndex].FirstOrEscape = static_cast<std::uint32_t>(output.size());
                output[outputIndex].PrimitiveCount = 0;
            }

            Detail::CancellationProbe m_Cancellation;
            std::vector<BuildPrimitive> m_Primitives;
            std::vector<BuildNode> m_BuildNodes;
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
#if defined(_WIN32)
            Initialize();
#else
            AvailabilityError = "CUDA is available only on supported Windows NVIDIA drivers.";
#endif
        }

        ~Impl()
        {
            StreamQueue.WaitIdle();
#if defined(_WIN32)
            if (Context) Api.CtxSetCurrent(Context);
            ReleaseFrameBuffers();
            ReleaseSceneBuffers();
            if (Module) Api.ModuleUnload(Module);
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
#if defined(_WIN32)
                    return RenderNative(*scene, owner, camera, settings, cancellation);
#else
                    return Detail::MakeFailedResult("CUDA is available only on Windows.");
#endif
                });
        }

#if defined(_WIN32)
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
            std::string ptx;
            for (const char* chunk : Detail::CudaRayTracingPtxChunks)
            {
                ptx += chunk;
            }
            if (!Check(Api.DevicePrimaryCtxRetain(&Context, Device),
                    "cuDevicePrimaryCtxRetain", AvailabilityError)
                || !Check(Api.CtxSetCurrent(Context), "cuCtxSetCurrent", AvailabilityError)
                || !Check(Api.StreamCreate(&Stream, 1u), "cuStreamCreate", AvailabilityError)
                || !Check(Api.ModuleLoadDataEx(
                    &Module, ptx.c_str(), 0, nullptr, nullptr),
                    "cuModuleLoadDataEx", AvailabilityError)
                || !Check(Api.ModuleGetFunction(&TraceFunction, Module, "TracePixelsKernel"),
                    "cuModuleGetFunction(TracePixelsKernel)", AvailabilityError)
                || !Check(Api.ModuleGetFunction(&ResolveFunction, Module, "ResolvePixelsKernel"),
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
            FreeBuffer(Nodes);
            FreeBuffer(PrimitiveRefs);
            FreeBuffer(Triangles);
            FreeBuffer(Spheres);
            FreeBuffer(Materials);
            FreeBuffer(Lights);
            FreeBuffer(TextureDescriptors);
            FreeBuffer(TextureTexels);
            NodeCount = PrimitiveRefCount = TriangleCount = SphereCount = MaterialCount = LightCount = 0;
            TextureCount = TextureTexelCount = 0;
            CachedOwner.reset();
        }

        void ReleaseFrameBuffers() const
        {
            FreeBuffer(Constants);
            FreeBuffer(Accumulation);
            FreeBuffer(SampleCounts);
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

        bool EnsureScene(
            const Scene& scene,
            const std::shared_ptr<const Scene>& owner,
            const Detail::CancellationProbe& cancellation,
            std::string& error) const
        {
            if (OwnerMatches(owner)) return true;
            PackedScene packed;
            ScenePacker packer(cancellation);
            if (!packer.Pack(scene, packed, error)) return false;
            if (cancellation())
            {
                error = "CUDA scene construction was cancelled.";
                return false;
            }
            const std::size_t requiredBytes = packed.Nodes.size() * sizeof(PackedBvhNode)
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
            if (!UploadBuffer(Nodes, packed.Nodes.data(),
                    packed.Nodes.size() * sizeof(PackedBvhNode), cancellation, error)
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
            return Check(Api.MemcpyHtoDAsync(
                Constants.Pointer, &constants, sizeof(constants), Stream),
                "cuMemcpyHtoDAsync(constants)", error);
        }

        bool LaunchTrace(
            const FrameConstants& constants,
            std::vector<GroupStatistics>& hostStatistics,
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
            void* parameters[] = {
                &nodes, &primitiveRefs, &triangles, &spheres, &materials, &lights,
                &textureDescriptors, &textureTexels,
                &frameConstants, &accumulation, &sampleCounts, &statistics};
            const std::uint32_t groups = constants.GroupCount;
            if (!Check(Api.LaunchKernel(
                    TraceFunction, groups, 1, 1, ThreadsPerBlock, 1, 1, 0,
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
            return true;
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

        RenderResult RenderNative(
            const Scene& scene,
            const std::shared_ptr<const Scene>& owner,
            const Camera& camera,
            const RenderSettings& settings,
            const Detail::CancellationProbe& cancellation) const
        {
            RenderResult result;
            if (!ValidateInput(scene, camera, settings, result.Error)) return result;
            if (!Check(Api.CtxSetCurrent(Context), "cuCtxSetCurrent", result.Error)) return result;
            const auto buildBegin = std::chrono::steady_clock::now();
            if (!EnsureScene(scene, owner, cancellation, result.Error))
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
                || !Check(Api.StreamSynchronize(Stream), "cuStreamSynchronize(clear)", result.Error))
            {
                return result;
            }
            if (cancellation()) return Detail::MakeCancelledResult();

            FrameConstants constants;
            constants.CameraPosition = ToVec3(camera.Position);
            constants.Width = settings.Width;
            constants.FullWidth = FullImageWidth(settings);
            constants.FullHeight = FullImageHeight(settings);
            constants.TileX = settings.TileX;
            constants.TileY = settings.TileY;
            constants.Integrator = static_cast<uint32_t>(settings.Integrator);
            constants.Height = settings.Height;
            constants.TotalSamples = settings.SamplesPerPixel;
            constants.MaxReflectionBounces = settings.MaxReflectionBounces;
            const Vec3 forward = NormalizeVector(
                Subtract(ToVec3(camera.Target), ToVec3(camera.Position)));
            const Vec3 right = NormalizeVector(Cross(forward, ToVec3(camera.Up)));
            const Vec3 up = NormalizeVector(Cross(right, forward));
            constants.CameraForward = forward;
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
            for (std::uint32_t sample = 0; sample < settings.SamplesPerPixel && !cancelled; ++sample)
            {
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
                    if (!LaunchTrace(constants, groupStatistics, result.Error)) return result;
                    for (const GroupStatistics& statistics : groupStatistics)
                    {
                        result.Statistics.PrimaryRays += statistics.PrimaryRays;
                        result.Statistics.ShadowRays += statistics.ShadowRays;
                        result.Statistics.ReflectionRays += statistics.ReflectionRays;
                        result.Statistics.PrimitiveTests += statistics.PrimitiveTests;
                    }
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
            if (!DownloadBuffer(
                    packedOutput.data(), Output.Pointer,
                    static_cast<std::size_t>(pixelCount) * sizeof(std::uint32_t),
                    cancellation, result.Error)
                || !DownloadBuffer(
                    completedSamples.data(), SampleCounts.Pointer,
                    static_cast<std::size_t>(pixelCount) * sizeof(std::uint32_t),
                    cancellation, result.Error))
            {
                if (cancellation()) result.Status = RenderStatus::Cancelled;
                return result;
            }
            const auto renderEnd = std::chrono::steady_clock::now();
            result.Output = Image(settings.Width, settings.Height);
            std::memcpy(
                result.Output.GetPixelsRGBA8().data(), packedOutput.data(),
                static_cast<std::size_t>(pixelCount) * sizeof(std::uint32_t));
            result.Statistics.CompletedPixels = static_cast<std::uint64_t>(std::count_if(
                completedSamples.begin(), completedSamples.end(),
                [&settings](std::uint32_t samples)
                {
                    return samples == settings.SamplesPerPixel;
                }));
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
        CUmodule Module = nullptr;
        CUfunction TraceFunction = nullptr;
        CUfunction ResolveFunction = nullptr;
        int DriverVersion = 0;
        int ComputeMajor = 0;
        int ComputeMinor = 0;
        std::string DeviceName;

        mutable DeviceBuffer Nodes;
        mutable DeviceBuffer PrimitiveRefs;
        mutable DeviceBuffer Triangles;
        mutable DeviceBuffer Spheres;
        mutable DeviceBuffer Materials;
        mutable DeviceBuffer Lights;
        mutable DeviceBuffer TextureDescriptors;
        mutable DeviceBuffer TextureTexels;
        mutable DeviceBuffer Constants;
        mutable DeviceBuffer Accumulation;
        mutable DeviceBuffer SampleCounts;
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
#if defined(_WIN32)
            ? std::string("Native CUDA Driver API backend on ") + m_Impl->DeviceName
                + " (compute " + std::to_string(m_Impl->ComputeMajor) + '.'
                + std::to_string(m_Impl->ComputeMinor) + ')'
#else
            ? "Native CUDA Driver API backend"
#endif
            : std::string("Unavailable: ") + m_Impl->AvailabilityError;
        return info;
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
