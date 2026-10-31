#include "ComputeShaderRayTracingRHI.h"

#include "Renderer/RayTracingBackendCommon.h"
#include "ComputeShaderBytecode.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32)
#  if !defined(NOMINMAX)
#    define NOMINMAX
#  endif
#  include <Windows.h>
#  include <d3d12.h>
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
        constexpr std::uint32_t LeafPrimitiveCount = 8;
        constexpr std::uint32_t MaximumBuildDepth = 48;
        constexpr std::uint32_t SahBinCount = 16;
        constexpr std::uint32_t ThreadsPerGroup = 256;
        constexpr std::uint32_t PixelChunkSize = 16384;
        constexpr std::uint32_t DirectionalLightType = 0;
        constexpr std::uint32_t PointLightType = 1;
        constexpr std::uint32_t RectAreaLightType = 2;
        constexpr std::size_t ReadbackChunkSize = 32u * 1024u * 1024u;
        constexpr float Pi = 3.14159265358979323846f;
        constexpr DWORD GpuWaitTimeoutMilliseconds = 5000;

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
        static_assert(sizeof(PackedMaterial) == 80, "Compute material ABI mismatch.");
        static_assert(
            offsetof(PackedMaterial, AmbientShininess) == 16
                && offsetof(PackedMaterial, Specular) == 32
                && offsetof(PackedMaterial, Emission) == 48
                && offsetof(PackedMaterial, DiffuseTextureIndex) == 64,
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
            sizeof(FrameConstants) == 192
                && offsetof(FrameConstants, ShadowStrength) == 124,
            "Compute constants ABI mismatch.");
        static_assert(sizeof(Counter64Parts) == 8, "Compute counter ABI mismatch.");
        static_assert(offsetof(FrameConstants, FullWidth) == 160, "Tile constant ABI mismatch.");
        static_assert(offsetof(FrameConstants, Integrator) == 176, "Integrator constant ABI mismatch.");
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
                error = "Compute image dimensions must be in the range [1, 16384].";
                return false;
            }
            if (settings.SamplesPerPixel == 0 || settings.SamplesPerPixel > 4096
                || settings.MaxReflectionBounces > 32
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
            for (const Triangle& triangle : scene.Triangles)
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
            for (const Sphere& sphere : scene.Spheres)
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

            bool Pack(const Scene& scene, PackedScene& output, std::string& error)
            {
                output = {};
                if (!Detail::BuildRayTextureAtlas(
                        scene,
                        output.TextureAtlas,
                        error))
                {
                    return false;
                }
                output.Triangles.reserve(scene.Triangles.size());
                output.Spheres.reserve(scene.Spheres.size());
                output.Materials.reserve(scene.Materials.size());
                output.Lights.reserve(
                    scene.DirectionalLights.size()
                    + scene.PointLights.size()
                    + scene.RectAreaLights.size());
                Primitives.reserve(scene.Triangles.size() + scene.Spheres.size());

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

                if (!Primitives.empty())
                {
                    BuildNodes.reserve(Primitives.size() * 2);
                    const std::uint32_t root = BuildRecursive(
                        0, static_cast<std::uint32_t>(Primitives.size()), 0);
                    if (IsCancelled()) return Cancel(error);
                    output.PrimitiveRefs.reserve(Primitives.size());
                    for (const BuildPrimitive& primitive : Primitives)
                    {
                        output.PrimitiveRefs.push_back(primitive.Reference);
                    }
                    output.Nodes.reserve(BuildNodes.size());
                    Flatten(root, output.Nodes);
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
                return Cancellation && Cancellation();
            }

            bool Cancel(std::string& error) const
            {
                error = "Compute scene packing was cancelled.";
                return false;
            }

            std::uint32_t BuildRecursive(
                std::uint32_t begin,
                std::uint32_t end,
                std::uint32_t depth)
            {
                const std::uint32_t nodeIndex = static_cast<std::uint32_t>(BuildNodes.size());
                BuildNodes.emplace_back();
                Bounds primitiveBounds;
                Bounds centroidBounds;
                for (std::uint32_t index = begin; index < end; ++index)
                {
                    primitiveBounds.Expand(Primitives[index].BoundingBox);
                    centroidBounds.Expand(Primitives[index].Centroid);
                }
                BuildNodes[nodeIndex].BoundingBox = primitiveBounds;
                const std::uint32_t count = end - begin;
                if (count <= LeafPrimitiveCount || depth >= MaximumBuildDepth || IsCancelled())
                {
                    BuildNodes[nodeIndex].FirstPrimitive = begin;
                    BuildNodes[nodeIndex].PrimitiveCount = count;
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
                            (Component(Primitives[index].Centroid, axis) - minimum) / extent;
                        const std::uint32_t binIndex = std::min(
                            static_cast<std::uint32_t>(normalized * SahBinCount), SahBinCount - 1);
                        ++bins[binIndex].Count;
                        bins[binIndex].BoundingBox.Expand(Primitives[index].BoundingBox);
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
                        Primitives.begin() + begin,
                        Primitives.begin() + end,
                        [minimum, extent, bestAxis, bestSplit](const BuildPrimitive& primitive)
                        {
                            const float normalized =
                                (Component(primitive.Centroid, bestAxis) - minimum) / extent;
                            const std::uint32_t binIndex = std::min(
                                static_cast<std::uint32_t>(normalized * SahBinCount),
                                SahBinCount - 1);
                            return binIndex <= bestSplit;
                        });
                    middle = static_cast<std::uint32_t>(middleIterator - Primitives.begin());
                }
                if (middle == begin || middle == end)
                {
                    const Vec3 extent = Subtract(centroidBounds.Maximum, centroidBounds.Minimum);
                    bestAxis = extent.Y > extent.X ? 1u : 0u;
                    if (Component(extent, 2) > Component(extent, bestAxis)) bestAxis = 2;
                    middle = begin + count / 2;
                    std::nth_element(
                        Primitives.begin() + begin,
                        Primitives.begin() + middle,
                        Primitives.begin() + end,
                        [bestAxis](const BuildPrimitive& left, const BuildPrimitive& right)
                        {
                            return Component(left.Centroid, bestAxis)
                                < Component(right.Centroid, bestAxis);
                        });
                }
                const std::uint32_t left = BuildRecursive(begin, middle, depth + 1);
                const std::uint32_t right = BuildRecursive(middle, end, depth + 1);
                BuildNodes[nodeIndex].Left = left;
                BuildNodes[nodeIndex].Right = right;
                return nodeIndex;
            }

            void Flatten(std::uint32_t buildIndex, std::vector<PackedBvhNode>& output)
            {
                const std::uint32_t outputIndex = static_cast<std::uint32_t>(output.size());
                output.emplace_back();
                const BuildNode& source = BuildNodes[buildIndex];
                output[outputIndex].MinX = source.BoundingBox.Minimum.X;
                output[outputIndex].MinY = source.BoundingBox.Minimum.Y;
                output[outputIndex].MinZ = source.BoundingBox.Minimum.Z;
                output[outputIndex].MaxX = source.BoundingBox.Maximum.X;
                output[outputIndex].MaxY = source.BoundingBox.Maximum.Y;
                output[outputIndex].MaxZ = source.BoundingBox.Maximum.Z;
                if (source.IsLeaf())
                {
                    output[outputIndex].FirstOrEscape = source.FirstPrimitive;
                    output[outputIndex].PrimitiveCount = source.PrimitiveCount;
                    return;
                }
                Flatten(source.Left, output);
                Flatten(source.Right, output);
                output[outputIndex].FirstOrEscape = static_cast<std::uint32_t>(output.size());
                output[outputIndex].PrimitiveCount = 0;
            }

            Detail::CancellationProbe Cancellation;
            std::vector<BuildPrimitive> Primitives;
            std::vector<BuildNode> BuildNodes;
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
        Impl()
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
                    return RenderNative(*scene, owner, camera, settings, cancellation);
#else
                    return Detail::MakeFailedResult("D3D12 compute is available only on Windows.");
#endif
                });
        }

#if defined(_WIN32)
        using D3D12CreateDeviceFunction = HRESULT(WINAPI*)(
            IUnknown*, D3D_FEATURE_LEVEL, REFIID, void**);
        using D3D12SerializeRootSignatureFunction = HRESULT(WINAPI*)(
            const D3D12_ROOT_SIGNATURE_DESC*, D3D_ROOT_SIGNATURE_VERSION,
            ID3DBlob**, ID3DBlob**);
        using CreateDxgiFactory2Function = HRESULT(WINAPI*)(UINT, REFIID, void**);

        bool Check(HRESULT result, const char* operation, std::string& error) const
        {
            if (SUCCEEDED(result)) return true;
            error = HResultError(operation, result);
            return false;
        }

        bool Initialize()
        {
            if (!D3D12Module.Load(L"d3d12.dll") || !DxgiModule.Load(L"dxgi.dll"))
            {
                AvailabilityError = "The D3D12 or DXGI runtime could not be loaded.";
                return false;
            }
            CreateDevice = D3D12Module.Get<D3D12CreateDeviceFunction>("D3D12CreateDevice");
            SerializeRootSignature =
                D3D12Module.Get<D3D12SerializeRootSignatureFunction>(
                    "D3D12SerializeRootSignature");
            CreateFactory = DxgiModule.Get<CreateDxgiFactory2Function>("CreateDXGIFactory2");
            if (!CreateDevice || !SerializeRootSignature || !CreateFactory)
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
                if (factory->EnumAdapters1(ordinal, &adapter) == DXGI_ERROR_NOT_FOUND) break;
                DXGI_ADAPTER_DESC1 description{};
                if (FAILED(adapter->GetDesc1(&description))
                    || (description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0)
                {
                    continue;
                }
                ComPtr<ID3D12Device> candidate;
                if (FAILED(CreateDevice(
                        adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&candidate))))
                {
                    continue;
                }
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
                AvailabilityError = "No hardware D3D12 adapter is available.";
                return false;
            }
            Adapter.As(&Adapter3);
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
            queueDescription.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
            if (!Check(Device->CreateCommandQueue(
                    &queueDescription, IID_PPV_ARGS(&Queue)),
                    "ID3D12Device::CreateCommandQueue", AvailabilityError)
                || !Check(Device->CreateCommandAllocator(
                    D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&Allocator)),
                    "ID3D12Device::CreateCommandAllocator", AvailabilityError)
                || !Check(Device->CreateCommandList(
                    0, D3D12_COMMAND_LIST_TYPE_COMPUTE, Allocator.Get(), nullptr,
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
            heapDescription.NumDescriptors = 12;
            heapDescription.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
            if (!Check(Device->CreateDescriptorHeap(
                    &heapDescription, IID_PPV_ARGS(&DescriptorHeap)),
                    "ID3D12Device::CreateDescriptorHeap", AvailabilityError))
            {
                return false;
            }
            DescriptorSize = Device->GetDescriptorHandleIncrementSize(
                D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            if (!CreateRootSignature(AvailabilityError)
                || !CreatePipeline(
                    Detail::ComputeClearPixelsDxil,
                    sizeof(Detail::ComputeClearPixelsDxil), ClearPixelsPipeline,
                    "CreateComputePipelineState(ClearPixels)", AvailabilityError)
                || !CreatePipeline(
                    Detail::ComputeClearStatisticsDxil,
                    sizeof(Detail::ComputeClearStatisticsDxil), ClearStatisticsPipeline,
                    "CreateComputePipelineState(ClearStatistics)", AvailabilityError)
                || !CreatePipeline(
                    Detail::ComputeTracePixelsDxil,
                    sizeof(Detail::ComputeTracePixelsDxil), TracePixelsPipeline,
                    "CreateComputePipelineState(TracePixels)", AvailabilityError)
                || !CreatePipeline(
                    Detail::ComputeResolvePixelsDxil,
                    sizeof(Detail::ComputeResolvePixelsDxil), ResolvePixelsPipeline,
                    "CreateComputePipelineState(ResolvePixels)", AvailabilityError))
            {
                return false;
            }
            Available = true;
            return true;
        }

        bool CreateRootSignature(std::string& error)
        {
            D3D12_DESCRIPTOR_RANGE ranges[2]{};
            ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
            ranges[0].NumDescriptors = 8;
            ranges[0].BaseShaderRegister = 0;
            ranges[0].OffsetInDescriptorsFromTableStart = 0;
            ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
            ranges[1].NumDescriptors = 4;
            ranges[1].BaseShaderRegister = 0;
            ranges[1].OffsetInDescriptorsFromTableStart = 0;

            D3D12_ROOT_PARAMETER parameters[3]{};
            parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            parameters[0].DescriptorTable.NumDescriptorRanges = 1;
            parameters[0].DescriptorTable.pDescriptorRanges = &ranges[0];
            parameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
            parameters[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            parameters[1].DescriptorTable.NumDescriptorRanges = 1;
            parameters[1].DescriptorTable.pDescriptorRanges = &ranges[1];
            parameters[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
            parameters[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
            parameters[2].Constants.ShaderRegister = 0;
            parameters[2].Constants.RegisterSpace = 0;
            parameters[2].Constants.Num32BitValues = sizeof(FrameConstants) / sizeof(std::uint32_t);
            parameters[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

            D3D12_ROOT_SIGNATURE_DESC description{};
            description.NumParameters = 3;
            description.pParameters = parameters;
            ComPtr<ID3DBlob> serialized;
            ComPtr<ID3DBlob> diagnostics;
            const HRESULT serializeResult = SerializeRootSignature(
                &description, D3D_ROOT_SIGNATURE_VERSION_1, &serialized, &diagnostics);
            if (FAILED(serializeResult))
            {
                error = HResultError("D3D12SerializeRootSignature", serializeResult);
                if (diagnostics && diagnostics->GetBufferPointer())
                {
                    error += " ";
                    error.append(
                        static_cast<const char*>(diagnostics->GetBufferPointer()),
                        diagnostics->GetBufferSize());
                }
                return false;
            }
            return Check(Device->CreateRootSignature(
                    0, serialized->GetBufferPointer(), serialized->GetBufferSize(),
                    IID_PPV_ARGS(&RootSignature)),
                "ID3D12Device::CreateRootSignature", error);
        }

        bool CreatePipeline(
            const void* bytecode,
            std::size_t bytecodeSize,
            ComPtr<ID3D12PipelineState>& pipeline,
            const char* operation,
            std::string& error)
        {
            D3D12_COMPUTE_PIPELINE_STATE_DESC description{};
            description.pRootSignature = RootSignature.Get();
            description.CS = {bytecode, bytecodeSize};
            return Check(Device->CreateComputePipelineState(
                    &description, IID_PPV_ARGS(&pipeline)), operation, error);
        }

        bool BeginCommands(std::string& error) const
        {
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
            handle.ptr += static_cast<std::uint64_t>(8) * DescriptorSize;
            CommandList->SetComputeRootDescriptorTable(1, handle);
            return true;
        }

        bool SignalAndWait(std::string& error) const
        {
            const std::uint64_t value = ++FenceValue;
            if (!Check(Queue->Signal(Fence.Get(), value), "ID3D12CommandQueue::Signal", error))
            {
                return false;
            }
            if (Fence->GetCompletedValue() < value)
            {
                if (!Check(Fence->SetEventOnCompletion(value, FenceEvent.Handle),
                        "ID3D12Fence::SetEventOnCompletion", error))
                {
                    return false;
                }
                const DWORD waitResult =
                    WaitForSingleObject(FenceEvent.Handle, GpuWaitTimeoutMilliseconds);
                if (waitResult != WAIT_OBJECT_0)
                {
                    const HRESULT removedReason = Device->GetDeviceRemovedReason();
                    error = FAILED(removedReason)
                        ? HResultError("The D3D12 compute device was removed", removedReason)
                        : waitResult == WAIT_TIMEOUT
                        ? "Waiting for the D3D12 compute fence timed out."
                        : "Waiting for the D3D12 compute fence failed.";
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

        void BindDispatch(
            ID3D12PipelineState* pipeline,
            const FrameConstants& constants) const
        {
            CommandList->SetPipelineState(pipeline);
            CommandList->SetComputeRoot32BitConstants(
                2, sizeof(constants) / sizeof(std::uint32_t), &constants, 0);
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
            Nodes.Reset();
            PrimitiveRefs.Reset();
            Triangles.Reset();
            Spheres.Reset();
            Materials.Reset();
            Lights.Reset();
            TextureDescriptors.Reset();
            TextureTexels.Reset();
            NodeCount = PrimitiveRefCount = TriangleCount = SphereCount = 0;
            MaterialCount = LightCount = 0;
            TextureCount = TextureTexelCount = 0;
            CachedOwner.reset();
        }

        void ReleaseFrameBuffers() const
        {
            Accumulation.Reset();
            SampleCounts.Reset();
            Output.Reset();
            Statistics.Reset();
            OutputReadback.Reset();
            SamplesReadback.Reset();
            StatisticsReadback.Reset();
            FramePixelCapacity = 0;
        }

        bool OwnerMatches(const std::shared_ptr<const Scene>& owner) const
        {
            if (!owner) return false;
            const std::shared_ptr<const Scene> cached = CachedOwner.lock();
            return cached && cached.get() == owner.get();
        }

        void CreateStructuredSrv(
            std::uint32_t descriptorIndex,
            const GpuBuffer& buffer,
            std::uint32_t count,
            std::uint32_t stride) const
        {
            D3D12_SHADER_RESOURCE_VIEW_DESC description{};
            description.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            description.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            description.Format = DXGI_FORMAT_UNKNOWN;
            description.Buffer.NumElements = std::max(count, 1u);
            description.Buffer.StructureByteStride = stride;
            D3D12_CPU_DESCRIPTOR_HANDLE handle =
                DescriptorHeap->GetCPUDescriptorHandleForHeapStart();
            handle.ptr += static_cast<std::uint64_t>(descriptorIndex) * DescriptorSize;
            Device->CreateShaderResourceView(buffer.Resource.Get(), &description, handle);
        }

        void CreateStructuredUav(
            std::uint32_t descriptorIndex,
            const GpuBuffer& buffer,
            std::uint32_t count,
            std::uint32_t stride) const
        {
            D3D12_UNORDERED_ACCESS_VIEW_DESC description{};
            description.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
            description.Format = DXGI_FORMAT_UNKNOWN;
            description.Buffer.NumElements = std::max(count, 1u);
            description.Buffer.StructureByteStride = stride;
            D3D12_CPU_DESCRIPTOR_HANDLE handle =
                DescriptorHeap->GetCPUDescriptorHandleForHeapStart();
            handle.ptr += static_cast<std::uint64_t>(descriptorIndex) * DescriptorSize;
            Device->CreateUnorderedAccessView(
                buffer.Resource.Get(), nullptr, &description, handle);
        }

        void CreateRawUav(
            std::uint32_t descriptorIndex,
            const GpuBuffer& buffer,
            std::uint32_t dwordCount) const
        {
            D3D12_UNORDERED_ACCESS_VIEW_DESC description{};
            description.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
            description.Format = DXGI_FORMAT_R32_TYPELESS;
            description.Buffer.NumElements = dwordCount;
            description.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
            D3D12_CPU_DESCRIPTOR_HANDLE handle =
                DescriptorHeap->GetCPUDescriptorHandleForHeapStart();
            handle.ptr += static_cast<std::uint64_t>(descriptorIndex) * DescriptorSize;
            Device->CreateUnorderedAccessView(
                buffer.Resource.Get(), nullptr, &description, handle);
        }

        void RefreshSceneDescriptors() const
        {
            CreateStructuredSrv(0, Nodes, NodeCount, sizeof(PackedBvhNode));
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
            CreateStructuredUav(8, Accumulation, pixelCount, sizeof(Vec4));
            CreateStructuredUav(9, SampleCounts, pixelCount, sizeof(std::uint32_t));
            CreateStructuredUav(10, Output, pixelCount, sizeof(std::uint32_t));
            CreateRawUav(11, Statistics, sizeof(GroupStatistics) / sizeof(std::uint32_t));
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
                error = "Compute scene construction was cancelled.";
                return false;
            }
            const std::uint64_t requiredBytes =
                packed.Nodes.size() * sizeof(PackedBvhNode)
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
            std::array<ComPtr<ID3D12Resource>, 8> staging;
            if (!PrepareUpload(Nodes, packed.Nodes.data(),
                    packed.Nodes.size() * sizeof(PackedBvhNode), staging[0], error)
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
                    error))
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
            std::array<GpuBuffer*, 8> destinations = {
                &Nodes,
                &PrimitiveRefs,
                &Triangles,
                &Spheres,
                &Materials,
                &Lights,
                &TextureDescriptors,
                &TextureTexels};
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
            RefreshSceneDescriptors();
            return true;
        }

        bool EnsureFrameBuffers(std::uint32_t pixelCount, std::string& error) const
        {
            if (FramePixelCapacity >= pixelCount) return true;
            const std::uint64_t defaultBytes =
                static_cast<std::uint64_t>(pixelCount)
                    * (sizeof(Vec4) + sizeof(std::uint32_t) + sizeof(std::uint32_t))
                + sizeof(GroupStatistics);
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
                BindDispatch(ClearPixelsPipeline.Get(), constants);
                CommandList->Dispatch(
                    (constants.PixelCount + ThreadsPerGroup - 1) / ThreadsPerGroup, 1, 1);
            }
            UavBarrier(Accumulation);
            UavBarrier(SampleCounts);
            UavBarrier(Output);
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

        bool DispatchTrace(
            const FrameConstants& constants,
            GroupStatistics& statistics,
            std::string& error) const
        {
            if (!BeginCommands(error)) return false;
            BindDispatch(ClearStatisticsPipeline.Get(), constants);
            CommandList->Dispatch(1, 1, 1);
            UavBarrier(Statistics);
            BindDispatch(TracePixelsPipeline.Get(), constants);
            CommandList->Dispatch(constants.GroupCount, 1, 1);
            UavBarrier(Accumulation);
            UavBarrier(SampleCounts);
            UavBarrier(Statistics);
            Transition(Statistics, D3D12_RESOURCE_STATE_COPY_SOURCE);
            CommandList->CopyBufferRegion(
                StatisticsReadback.Resource.Get(), 0, Statistics.Resource.Get(), 0,
                sizeof(GroupStatistics));
            Transition(Statistics, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            if (!ExecuteCommands(error)) return false;
            return ReadStatistics(statistics, error);
        }

        bool ResolveAndReadback(
            FrameConstants constants,
            std::uint32_t pixelCount,
            const Detail::CancellationProbe& cancellation,
            std::vector<std::uint32_t>& packedOutput,
            std::vector<std::uint32_t>& completedSamples,
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
                BindDispatch(ResolvePixelsPipeline.Get(), constants);
                CommandList->Dispatch(
                    (constants.PixelCount + ThreadsPerGroup - 1) / ThreadsPerGroup, 1, 1);
            }
            UavBarrier(Output);
            Transition(Output, D3D12_RESOURCE_STATE_COPY_SOURCE);
            Transition(SampleCounts, D3D12_RESOURCE_STATE_COPY_SOURCE);
            CommandList->CopyBufferRegion(
                OutputReadback.Resource.Get(), 0, Output.Resource.Get(), 0,
                static_cast<std::uint64_t>(pixelCount) * sizeof(std::uint32_t));
            CommandList->CopyBufferRegion(
                SamplesReadback.Resource.Get(), 0, SampleCounts.Resource.Get(), 0,
                static_cast<std::uint64_t>(pixelCount) * sizeof(std::uint32_t));
            Transition(Output, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            Transition(SampleCounts, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
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
                if (!Check(OutputReadback.Resource->Map(0, &readRange, &outputMapped),
                        "ID3D12Resource::Map(output readback)", error)
                    || !Check(SamplesReadback.Resource->Map(0, &readRange, &samplesMapped),
                        "ID3D12Resource::Map(sample readback)", error))
                {
                    if (outputMapped)
                    {
                        D3D12_RANGE noWrite{0, 0};
                        OutputReadback.Resource->Unmap(0, &noWrite);
                    }
                    return false;
                }
                std::memcpy(
                    reinterpret_cast<std::uint8_t*>(packedOutput.data()) + offset,
                    static_cast<const std::uint8_t*>(outputMapped) + offset, chunk);
                std::memcpy(
                    reinterpret_cast<std::uint8_t*>(completedSamples.data()) + offset,
                    static_cast<const std::uint8_t*>(samplesMapped) + offset, chunk);
                D3D12_RANGE noWrite{0, 0};
                OutputReadback.Resource->Unmap(0, &noWrite);
                SamplesReadback.Resource->Unmap(0, &noWrite);
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
            constants.CameraForward = forward;
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
            const auto renderBegin = std::chrono::steady_clock::now();
            for (std::uint32_t sample = 0; sample < settings.SamplesPerPixel; ++sample)
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
                    if (!DispatchTrace(constants, statistics, result.Error)) return result;
                    result.Statistics.PrimaryRays += statistics.PrimaryRays.Value();
                    result.Statistics.ShadowRays += statistics.ShadowRays.Value();
                    result.Statistics.ReflectionRays += statistics.ReflectionRays.Value();
                    result.Statistics.PrimitiveTests += statistics.PrimitiveTests.Value();
                }
            }

            std::vector<std::uint32_t> packedOutput;
            std::vector<std::uint32_t> completedSamples;
            if (!ResolveAndReadback(
                    constants, pixelCount, cancellation,
                    packedOutput, completedSamples, result.Error))
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

        DynamicModule D3D12Module;
        DynamicModule DxgiModule;
        D3D12CreateDeviceFunction CreateDevice = nullptr;
        D3D12SerializeRootSignatureFunction SerializeRootSignature = nullptr;
        CreateDxgiFactory2Function CreateFactory = nullptr;
        ComPtr<IDXGIAdapter1> Adapter;
        ComPtr<IDXGIAdapter3> Adapter3;
        ComPtr<ID3D12Device> Device;
        ComPtr<ID3D12CommandQueue> Queue;
        ComPtr<ID3D12CommandAllocator> Allocator;
        ComPtr<ID3D12GraphicsCommandList> CommandList;
        ComPtr<ID3D12Fence> Fence;
        EventHandle FenceEvent;
        ComPtr<ID3D12DescriptorHeap> DescriptorHeap;
        ComPtr<ID3D12RootSignature> RootSignature;
        ComPtr<ID3D12PipelineState> ClearPixelsPipeline;
        ComPtr<ID3D12PipelineState> ClearStatisticsPipeline;
        ComPtr<ID3D12PipelineState> TracePixelsPipeline;
        ComPtr<ID3D12PipelineState> ResolvePixelsPipeline;
        std::uint32_t DescriptorSize = 0;
        std::string AdapterName;

        mutable std::uint64_t FenceValue = 0;
        mutable GpuBuffer Nodes;
        mutable GpuBuffer PrimitiveRefs;
        mutable GpuBuffer Triangles;
        mutable GpuBuffer Spheres;
        mutable GpuBuffer Materials;
        mutable GpuBuffer Lights;
        mutable GpuBuffer TextureDescriptors;
        mutable GpuBuffer TextureTexels;
        mutable GpuBuffer Accumulation;
        mutable GpuBuffer SampleCounts;
        mutable GpuBuffer Output;
        mutable GpuBuffer Statistics;
        mutable GpuBuffer OutputReadback;
        mutable GpuBuffer SamplesReadback;
        mutable GpuBuffer StatisticsReadback;
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
#endif
        bool Available = false;
        std::string AvailabilityError;
        mutable Detail::PersistentRenderQueue ComputeQueue;
    };

    ComputeShaderRayTracingRHI::ComputeShaderRayTracingRHI()
        : m_Impl(std::make_unique<Impl>())
    {
    }

    ComputeShaderRayTracingRHI::~ComputeShaderRayTracingRHI() = default;

    BackendInfo ComputeShaderRayTracingRHI::GetBackendInfo() const
    {
        BackendInfo info;
        info.Type = BackendType::ComputeShader;
        info.Name = "ComputeShader";
        info.IsNative = m_Impl->Available;
        info.IsHardwareAccelerated = m_Impl->Available;
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

    RenderResult ComputeShaderRayTracingRHI::Render(
        const Scene& scene,
        const Camera& camera,
        const RenderSettings& settings,
        const CancellationToken& cancellation) const
    {
        return m_Impl->Submit(&scene, nullptr, camera, settings, cancellation).GetResult();
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
