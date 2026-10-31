#include "MetalRayTracingRHI.h"

#if defined(__APPLE__)

#include "Renderer/RayTracingBackendCommon.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace Arvo
{
namespace RayTracing
{
namespace
{

constexpr std::uint32_t SphereBit = 0x80000000u;
constexpr std::uint32_t LeafPrimitiveCount = 8u;
constexpr std::uint32_t MaximumBuildDepth = 48u;
constexpr std::uint32_t ThreadsPerGroup = 256u;
constexpr std::uint32_t PixelChunkSize = 16384u;
constexpr std::uint32_t LightTypeDirectional = 0u;
constexpr std::uint32_t LightTypePoint = 1u;
constexpr std::uint32_t LightTypeRectArea = 2u;
constexpr float Pi = 3.14159265358979323846f;

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

    [[nodiscard]] std::uint64_t Value() const noexcept
    {
        return static_cast<std::uint64_t>(Low)
            | (static_cast<std::uint64_t>(High) << 32u);
    }
};

struct GroupStatistics
{
    Counter64Parts PrimaryRays;
    Counter64Parts ShadowRays;
    Counter64Parts ReflectionRays;
    Counter64Parts PrimitiveTests;
};

static_assert(sizeof(Vec3) == 12u, "Metal packed float3 host ABI mismatch.");
static_assert(sizeof(Vec4) == 16u, "Metal float4 host ABI mismatch.");
static_assert(sizeof(PackedBvhNode) == 32u, "Metal BVH ABI mismatch.");
static_assert(offsetof(PackedBvhNode, FirstOrEscape) == 12u, "Metal BVH field ABI mismatch.");
static_assert(offsetof(PackedBvhNode, PrimitiveCount) == 28u, "Metal BVH field ABI mismatch.");
static_assert(sizeof(PackedPrimitiveRef) == 4u, "Metal primitive reference ABI mismatch.");
static_assert(sizeof(PackedTriangle) == 192u, "Metal triangle ABI mismatch.");
static_assert(offsetof(PackedTriangle, TexCoord01) == 144u, "Metal triangle field ABI mismatch.");
static_assert(offsetof(PackedTriangle, TexCoord2) == 160u, "Metal triangle field ABI mismatch.");
static_assert(offsetof(PackedTriangle, MaterialIndex) == 176u, "Metal triangle field ABI mismatch.");
static_assert(sizeof(PackedSphere) == 32u, "Metal sphere ABI mismatch.");
static_assert(sizeof(PackedMaterial) == 80u, "Metal material ABI mismatch.");
static_assert(
    offsetof(PackedMaterial, AmbientShininess) == 16u
        && offsetof(PackedMaterial, Specular) == 32u
        && offsetof(PackedMaterial, Emission) == 48u
        && offsetof(PackedMaterial, DiffuseTextureIndex) == 64u,
    "Metal material field ABI mismatch.");
static_assert(
    sizeof(Detail::PackedRayTextureDescriptor) == 16u,
    "Metal texture descriptor ABI mismatch.");
static_assert(sizeof(PackedLight) == 64u, "Metal light ABI mismatch.");
static_assert(offsetof(PackedLight, Position) == 0u, "Metal light position ABI mismatch.");
static_assert(offsetof(PackedLight, Type) == 12u, "Metal light type ABI mismatch.");
static_assert(offsetof(PackedLight, Direction) == 16u, "Metal light direction ABI mismatch.");
static_assert(offsetof(PackedLight, Intensity) == 28u, "Metal light intensity ABI mismatch.");
static_assert(offsetof(PackedLight, Color) == 32u, "Metal light color ABI mismatch.");
static_assert(offsetof(PackedLight, CastShadows) == 44u, "Metal light shadow ABI mismatch.");
static_assert(offsetof(PackedLight, Width) == 48u, "Metal light width ABI mismatch.");
static_assert(offsetof(PackedLight, Height) == 52u, "Metal light height ABI mismatch.");
static_assert(offsetof(PackedLight, Padding0) == 56u, "Metal light padding ABI mismatch.");
static_assert(offsetof(PackedLight, Padding1) == 60u, "Metal light padding ABI mismatch.");
static_assert(sizeof(FrameConstants) == 192u, "Metal frame constant ABI mismatch.");
static_assert(offsetof(FrameConstants, Width) == 12u, "Metal camera constant ABI mismatch.");
static_assert(offsetof(FrameConstants, BackgroundBottom) == 96u, "Metal environment constant ABI mismatch.");
static_assert(offsetof(FrameConstants, ShadowStrength) == 124u, "Metal shadow constant ABI mismatch.");
static_assert(offsetof(FrameConstants, PixelBegin) == 128u, "Metal dispatch constant ABI mismatch.");
static_assert(offsetof(FrameConstants, TextureCount) == 152u, "Metal texture constant ABI mismatch.");
static_assert(offsetof(FrameConstants, FullWidth) == 160u, "Metal tile constant ABI mismatch.");
static_assert(offsetof(FrameConstants, Integrator) == 176u, "Metal integrator constant ABI mismatch.");
static_assert(sizeof(GroupStatistics) == 32u, "Metal statistics ABI mismatch.");

[[nodiscard]] Vec3 ToVec3(const Float3& value)
{
    return {value.X, value.Y, value.Z};
}

[[nodiscard]] Vec4 ToVec4(const Float3& value, float w = 0.0f)
{
    return {value.X, value.Y, value.Z, w};
}

[[nodiscard]] Vec3 Add(const Vec3& left, const Vec3& right)
{
    return {left.X + right.X, left.Y + right.Y, left.Z + right.Z};
}

[[nodiscard]] Vec3 Subtract(const Vec3& left, const Vec3& right)
{
    return {left.X - right.X, left.Y - right.Y, left.Z - right.Z};
}

[[nodiscard]] Vec3 Multiply(const Vec3& value, float scalar)
{
    return {value.X * scalar, value.Y * scalar, value.Z * scalar};
}

[[nodiscard]] float DotVector(const Vec3& left, const Vec3& right)
{
    return left.X * right.X + left.Y * right.Y + left.Z * right.Z;
}

[[nodiscard]] Vec3 CrossVector(const Vec3& left, const Vec3& right)
{
    return {
        left.Y * right.Z - left.Z * right.Y,
        left.Z * right.X - left.X * right.Z,
        left.X * right.Y - left.Y * right.X};
}

[[nodiscard]] Vec3 NormalizeVector(const Vec3& value)
{
    const float lengthSquared = DotVector(value, value);
    if (lengthSquared <= 1.0e-20f)
    {
        return {};
    }
    return Multiply(value, 1.0f / std::sqrt(lengthSquared));
}

[[nodiscard]] bool IsFinite(const Float3& value)
{
    return std::isfinite(value.X) && std::isfinite(value.Y) && std::isfinite(value.Z);
}

[[nodiscard]] bool IsNonNegative(const Float3& value)
{
    return value.X >= 0.0f && value.Y >= 0.0f && value.Z >= 0.0f;
}

[[nodiscard]] bool IsFinite(const Float2& value)
{
    return std::isfinite(value.X) && std::isfinite(value.Y);
}

[[nodiscard]] bool IsFinite(const Vec3& value)
{
    return std::isfinite(value.X) && std::isfinite(value.Y) && std::isfinite(value.Z);
}

struct Bounds
{
    Vec3 Minimum{
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max()};
    Vec3 Maximum{
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

    [[nodiscard]] Vec3 Center() const
    {
        return Multiply(Add(Minimum, Maximum), 0.5f);
    }
};

[[nodiscard]] float Component(const Vec3& value, std::uint32_t axis)
{
    return axis == 0u ? value.X : (axis == 1u ? value.Y : value.Z);
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

    [[nodiscard]] bool IsLeaf() const noexcept
    {
        return PrimitiveCount != 0u;
    }
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

[[nodiscard]] bool ValidateInput(
    const Scene& scene,
    const Camera& camera,
    const RenderSettings& settings,
    std::string& error)
{
    if (!ValidateTileSettings(settings, error)) return false;
    if (settings.Width == 0u || settings.Height == 0u
        || settings.Width > 16384u || settings.Height > 16384u)
    {
        error = "Metal image dimensions must be in the range [1, 16384].";
        return false;
    }
    if (settings.SamplesPerPixel == 0u || settings.SamplesPerPixel > 4096u
        || settings.MaxReflectionBounces > 32u)
    {
        error = "Metal sample or reflection settings exceed their safety limits.";
        return false;
    }
    if (!std::isfinite(settings.ShadowStrength)
        || settings.ShadowStrength < 0.0f || settings.ShadowStrength > 1.0f)
    {
        error = "Metal shadow strength must be finite and in the range [0, 1].";
        return false;
    }
    if (!IsFinite(camera.Position) || !IsFinite(camera.Target) || !IsFinite(camera.Up)
        || !std::isfinite(camera.VerticalFieldOfViewDegrees)
        || camera.VerticalFieldOfViewDegrees <= 1.0f
        || camera.VerticalFieldOfViewDegrees >= 179.0f)
    {
        error = "Metal camera values are invalid.";
        return false;
    }
    const Vec3 cameraDirection = Subtract(ToVec3(camera.Target), ToVec3(camera.Position));
    const Vec3 cameraUp = ToVec3(camera.Up);
    const Vec3 frameCross = CrossVector(
        NormalizeVector(cameraDirection),
        NormalizeVector(cameraUp));
    const float directionLengthSquared = DotVector(cameraDirection, cameraDirection);
    const float upLengthSquared = DotVector(cameraUp, cameraUp);
    const float frameAreaSquared = DotVector(frameCross, frameCross);
    if (!std::isfinite(directionLengthSquared) || !std::isfinite(upLengthSquared)
        || !std::isfinite(frameAreaSquared)
        || directionLengthSquared < 1.0e-10f
        || upLengthSquared < 1.0e-10f
        || frameAreaSquared < 1.0e-10f)
    {
        error = "Metal camera vectors do not form a valid frame.";
        return false;
    }
    if ((!scene.Triangles.empty() || !scene.Spheres.empty()) && scene.Materials.empty())
    {
        error = "Metal geometry requires at least one material.";
        return false;
    }
    if (scene.Triangles.size() >= SphereBit || scene.Spheres.size() >= SphereBit
        || scene.Triangles.size() + scene.Spheres.size() >= SphereBit
        || scene.Materials.size() > std::numeric_limits<std::uint32_t>::max()
        || scene.Textures.size() > std::numeric_limits<std::uint32_t>::max())
    {
        error = "Metal scene counts exceed the 31-bit primitive layout.";
        return false;
    }
    std::uint64_t textureTexelCount = 0u;
    for (const RayTexture& texture : scene.Textures)
    {
        if (!texture.IsValid())
        {
            continue;
        }
        textureTexelCount +=
            static_cast<std::uint64_t>(texture.Width) * texture.Height;
        if (textureTexelCount > std::numeric_limits<std::uint32_t>::max())
        {
            error = "Metal texture atlas exceeds its 32-bit texel layout.";
            return false;
        }
    }
    const std::size_t maximumLightCount =
        std::numeric_limits<std::uint32_t>::max();
    if (scene.DirectionalLights.size() > maximumLightCount
        || scene.PointLights.size() > maximumLightCount
        || scene.RectAreaLights.size() > maximumLightCount
        || scene.PointLights.size()
            > maximumLightCount - scene.DirectionalLights.size()
        || scene.RectAreaLights.size()
            > maximumLightCount
                - scene.DirectionalLights.size()
                - scene.PointLights.size())
    {
        error = "Metal light counts exceed the 32-bit light layout.";
        return false;
    }
    if (!IsFinite(scene.BackgroundTop) || !IsFinite(scene.BackgroundBottom)
        || !std::isfinite(scene.AmbientIntensity) || scene.AmbientIntensity < 0.0f)
    {
        error = "Metal background and ambient values are invalid.";
        return false;
    }
    for (const Material& material : scene.Materials)
    {
        if (!IsFinite(material.Albedo) || !IsFinite(material.AmbientColor)
            || !IsFinite(material.SpecularColor) || !IsFinite(material.Emission)
            || !std::isfinite(material.Shininess)
            || !std::isfinite(material.Reflectivity)
            || !IsNonNegative(material.AmbientColor)
            || !IsNonNegative(material.SpecularColor)
            || material.Shininess < 0.0f)
        {
            error = "Metal material values must be finite; ambient, specular, and shininess must be non-negative.";
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
            error = "Metal triangle data is invalid.";
            return false;
        }
        const Vec3 edge1 = Subtract(ToVec3(triangle.Position1), ToVec3(triangle.Position0));
        const Vec3 edge2 = Subtract(ToVec3(triangle.Position2), ToVec3(triangle.Position0));
        const Vec3 area = CrossVector(edge1, edge2);
        const float edge1LengthSquared = DotVector(edge1, edge1);
        const float edge2LengthSquared = DotVector(edge2, edge2);
        const float areaSquared = DotVector(area, area);
        if (!std::isfinite(edge1LengthSquared) || !std::isfinite(edge2LengthSquared)
            || !std::isfinite(areaSquared) || areaSquared < 1.0e-16f)
        {
            error = "Metal does not accept degenerate or numerically unsafe triangles.";
            return false;
        }
    }
    for (const Sphere& sphere : scene.Spheres)
    {
        const float radiusSquared = sphere.Radius * sphere.Radius;
        const Float3 minimum{
            sphere.Center.X - sphere.Radius,
            sphere.Center.Y - sphere.Radius,
            sphere.Center.Z - sphere.Radius};
        const Float3 maximum{
            sphere.Center.X + sphere.Radius,
            sphere.Center.Y + sphere.Radius,
            sphere.Center.Z + sphere.Radius};
        if (sphere.MaterialIndex >= scene.Materials.size() || !IsFinite(sphere.Center)
            || !std::isfinite(sphere.Radius) || sphere.Radius <= 0.0f
            || !std::isfinite(radiusSquared) || !IsFinite(minimum) || !IsFinite(maximum))
        {
            error = "Metal sphere data is invalid.";
            return false;
        }
    }
    for (const DirectionalLight& light : scene.DirectionalLights)
    {
        const Vec3 direction = ToVec3(light.DirectionToLight);
        const float directionLengthSquared = DotVector(direction, direction);
        if (!IsFinite(light.DirectionToLight) || !IsFinite(light.Color)
            || !std::isfinite(light.Intensity) || light.Intensity < 0.0f
            || !std::isfinite(directionLengthSquared)
            || directionLengthSquared < 1.0e-12f)
        {
            error = "Metal directional light data is invalid.";
            return false;
        }
    }
    for (const PointLight& light : scene.PointLights)
    {
        if (!IsFinite(light.Position) || !IsFinite(light.Color)
            || !std::isfinite(light.Intensity) || light.Intensity < 0.0f)
        {
            error =
                "Metal point light values must be finite with non-negative intensity.";
            return false;
        }
    }
    for (const RectAreaLight& light : scene.RectAreaLights)
    {
        const Vec3 direction = ToVec3(light.Direction);
        const float directionLengthSquared = DotVector(direction, direction);
        if (!IsFinite(light.Position) || !IsFinite(light.Direction)
            || !IsFinite(light.Color) || !IsFinite(light.Size)
            || !std::isfinite(light.Intensity) || light.Intensity < 0.0f
            || !std::isfinite(directionLengthSquared)
            || directionLengthSquared < 1.0e-12f
            || light.Size.X <= 0.0f || light.Size.Y <= 0.0f)
        {
            error =
                "Metal rect area light values must be finite with a non-zero "
                "direction, positive size, and non-negative intensity.";
            return false;
        }
        const Vec3 normalizedDirection = NormalizeVector(direction);
        const Vec3 helper = std::fabs(normalizedDirection.Y) < 0.999f
            ? Vec3{0.0f, 1.0f, 0.0f}
            : Vec3{1.0f, 0.0f, 0.0f};
        const Vec3 tangent =
            NormalizeVector(CrossVector(helper, normalizedDirection));
        const Vec3 bitangent =
            NormalizeVector(CrossVector(normalizedDirection, tangent));
        const Vec3 halfTangent = Multiply(tangent, light.Size.X * 0.5f);
        const Vec3 halfBitangent = Multiply(bitangent, light.Size.Y * 0.5f);
        const Vec3 position = ToVec3(light.Position);
        if (!IsFinite(Add(Add(position, halfTangent), halfBitangent))
            || !IsFinite(Subtract(Add(position, halfTangent), halfBitangent))
            || !IsFinite(Add(Subtract(position, halfTangent), halfBitangent))
            || !IsFinite(Subtract(Subtract(position, halfTangent), halfBitangent)))
        {
            error = "Metal rect area light bounds must remain finite.";
            return false;
        }
    }
    return true;
}

class ScenePacker
{
public:
    explicit ScenePacker(Detail::CancellationProbe cancellation)
        : Cancellation_(std::move(cancellation))
    {
    }

    bool Pack(const Scene& scene, PackedScene& output, std::string& error)
    {
        output = {};
        if (!Detail::BuildRayTextureAtlas(scene, output.TextureAtlas, error))
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
        Primitives_.reserve(scene.Triangles.size() + scene.Spheres.size());

        for (std::uint32_t index = 0; index < scene.Triangles.size(); ++index)
        {
            if ((index & 4095u) == 0u && IsCancelled())
            {
                return Cancel(error);
            }
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
            Primitives_.push_back(primitive);
        }
        for (std::uint32_t index = 0; index < scene.Spheres.size(); ++index)
        {
            if ((index & 4095u) == 0u && IsCancelled())
            {
                return Cancel(error);
            }
            const Sphere& source = scene.Spheres[index];
            PackedSphere sphere;
            sphere.CenterRadius = ToVec4(source.Center, source.Radius);
            sphere.MaterialIndex = source.MaterialIndex;
            output.Spheres.push_back(sphere);

            const Vec3 center = ToVec3(source.Center);
            const Vec3 radius{source.Radius, source.Radius, source.Radius};
            BuildPrimitive primitive;
            primitive.BoundingBox.Expand(Subtract(center, radius));
            primitive.BoundingBox.Expand(Add(center, radius));
            primitive.Centroid = center;
            primitive.Reference.TypeAndIndex = SphereBit | index;
            Primitives_.push_back(primitive);
        }
        for (std::uint32_t index = 0; index < scene.Materials.size(); ++index)
        {
            if ((index & 4095u) == 0u && IsCancelled())
            {
                return Cancel(error);
            }
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
            if ((index & 4095u) == 0u && IsCancelled())
            {
                return Cancel(error);
            }
            const DirectionalLight& source = scene.DirectionalLights[index];
            const Vec3 direction = NormalizeVector(ToVec3(source.DirectionToLight));
            PackedLight light;
            light.Type = LightTypeDirectional;
            light.Direction = direction;
            light.Intensity = source.Intensity;
            light.Color = ToVec3(source.Color);
            light.CastShadows = source.CastShadows ? 1u : 0u;
            output.Lights.push_back(light);
        }
        for (std::uint32_t index = 0; index < scene.PointLights.size(); ++index)
        {
            if ((index & 4095u) == 0u && IsCancelled())
            {
                return Cancel(error);
            }
            const PointLight& source = scene.PointLights[index];
            PackedLight light;
            light.Position = ToVec3(source.Position);
            light.Type = LightTypePoint;
            light.Intensity = source.Intensity;
            light.Color = ToVec3(source.Color);
            light.CastShadows = source.CastShadows ? 1u : 0u;
            output.Lights.push_back(light);
        }
        for (std::uint32_t index = 0; index < scene.RectAreaLights.size(); ++index)
        {
            if ((index & 4095u) == 0u && IsCancelled())
            {
                return Cancel(error);
            }
            const RectAreaLight& source = scene.RectAreaLights[index];
            PackedLight light;
            light.Position = ToVec3(source.Position);
            light.Type = LightTypeRectArea;
            light.Direction = NormalizeVector(ToVec3(source.Direction));
            light.Intensity = source.Intensity;
            light.Color = ToVec3(source.Color);
            light.CastShadows = source.CastShadows ? 1u : 0u;
            light.Width = source.Size.X;
            light.Height = source.Size.Y;
            output.Lights.push_back(light);
        }

        if (!Primitives_.empty())
        {
            BuildNodes_.reserve(Primitives_.size() * 2u);
            const std::uint32_t root = BuildRecursive(
                0u,
                static_cast<std::uint32_t>(Primitives_.size()),
                0u);
            if (IsCancelled())
            {
                return Cancel(error);
            }
            output.PrimitiveRefs.reserve(Primitives_.size());
            for (const BuildPrimitive& primitive : Primitives_)
            {
                output.PrimitiveRefs.push_back(primitive.Reference);
            }
            output.Nodes.reserve(BuildNodes_.size());
            Flatten(root, output.Nodes);
        }
        return true;
    }

private:
    [[nodiscard]] bool IsCancelled() const
    {
        return Cancellation_ && Cancellation_();
    }

    bool Cancel(std::string& error) const
    {
        error = "Metal scene packing was cancelled.";
        return false;
    }

    std::uint32_t BuildRecursive(
        std::uint32_t begin,
        std::uint32_t end,
        std::uint32_t depth)
    {
        const std::uint32_t nodeIndex = static_cast<std::uint32_t>(BuildNodes_.size());
        BuildNodes_.emplace_back();
        Bounds primitiveBounds;
        Bounds centroidBounds;
        for (std::uint32_t index = begin; index < end; ++index)
        {
            primitiveBounds.Expand(Primitives_[index].BoundingBox);
            centroidBounds.Expand(Primitives_[index].Centroid);
        }
        BuildNodes_[nodeIndex].BoundingBox = primitiveBounds;
        const std::uint32_t count = end - begin;
        const Vec3 centroidExtent = Subtract(centroidBounds.Maximum, centroidBounds.Minimum);
        if (count <= LeafPrimitiveCount || depth >= MaximumBuildDepth || IsCancelled()
            || std::max({centroidExtent.X, centroidExtent.Y, centroidExtent.Z}) <= 1.0e-6f)
        {
            BuildNodes_[nodeIndex].FirstPrimitive = begin;
            BuildNodes_[nodeIndex].PrimitiveCount = count;
            return nodeIndex;
        }

        std::uint32_t axis = centroidExtent.Y > centroidExtent.X ? 1u : 0u;
        if (Component(centroidExtent, 2u) > Component(centroidExtent, axis))
        {
            axis = 2u;
        }
        const std::uint32_t middle = begin + count / 2u;
        std::nth_element(
            Primitives_.begin() + begin,
            Primitives_.begin() + middle,
            Primitives_.begin() + end,
            [axis](const BuildPrimitive& left, const BuildPrimitive& right)
            {
                return Component(left.Centroid, axis) < Component(right.Centroid, axis);
            });
        const std::uint32_t left = BuildRecursive(begin, middle, depth + 1u);
        const std::uint32_t right = BuildRecursive(middle, end, depth + 1u);
        BuildNodes_[nodeIndex].Left = left;
        BuildNodes_[nodeIndex].Right = right;
        return nodeIndex;
    }

    void Flatten(std::uint32_t buildIndex, std::vector<PackedBvhNode>& output)
    {
        const std::uint32_t outputIndex = static_cast<std::uint32_t>(output.size());
        output.emplace_back();
        const BuildNode& source = BuildNodes_[buildIndex];
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
    }

    Detail::CancellationProbe Cancellation_;
    std::vector<BuildPrimitive> Primitives_;
    std::vector<BuildNode> BuildNodes_;
};

constexpr const char* MetalShaderSource = R"MSL(
#include <metal_stdlib>
using namespace metal;

constant uint SphereBit = 0x80000000u;
constant uint LightTypeDirectional = 0u;
constant uint LightTypePoint = 1u;
constant uint LightTypeRectArea = 2u;
constant float RayEpsilon = 1.0e-4f;

struct PackedBvhNode
{
    packed_float3 Minimum;
    uint FirstOrEscape;
    packed_float3 Maximum;
    uint PrimitiveCount;
};

struct PackedTriangle
{
    float4 Position0;
    float4 Edge1;
    float4 Edge2;
    float4 Normal0;
    float4 Normal1;
    float4 Normal2;
    float4 Color0;
    float4 Color1;
    float4 Color2;
    float4 TexCoord01;
    float4 TexCoord2;
    uint MaterialIndex;
    uint Flags;
    uint Padding0;
    uint Padding1;
};

struct PackedSphere
{
    float4 CenterRadius;
    uint MaterialIndex;
    uint Padding0;
    uint Padding1;
    uint Padding2;
};

struct PackedMaterial
{
    float4 AlbedoReflectivity;
    float4 AmbientShininess;
    float4 Specular;
    float4 Emission;
    uint DiffuseTextureIndex;
    uint SampledEmitter;
    uint Padding1;
    uint Padding2;
};

struct PackedTextureDescriptor
{
    uint OffsetTexels;
    uint Width;
    uint Height;
    uint Padding;
};

struct PackedLight
{
    packed_float3 Position;
    uint Type;
    packed_float3 Direction;
    float Intensity;
    packed_float3 Color;
    uint CastShadows;
    float Width;
    float Height;
    uint Padding0;
    uint Padding1;
};

struct FrameConstants
{
    packed_float3 CameraPosition;
    uint Width;
    packed_float3 CameraForward;
    uint Height;
    packed_float3 CameraRight;
    uint TotalSamples;
    packed_float3 CameraUp;
    uint MaxReflectionBounces;
    float HalfWidth;
    float HalfHeight;
    uint MaterialCount;
    uint LightCount;
    packed_float3 BackgroundTop;
    float AmbientIntensity;
    packed_float3 BackgroundBottom;
    uint NodeCount;
    uint PrimitiveRefCount;
    uint TriangleCount;
    uint SphereCount;
    float ShadowStrength;
    uint PixelBegin;
    uint PixelCount;
    uint SampleBegin;
    uint SampleCount;
    uint GammaCorrection;
    uint GroupCount;
    uint TextureCount;
    uint TextureTexelCount;
    uint FullWidth;
    uint FullHeight;
    uint TileX;
    uint TileY;
    uint Integrator;
    uint IntegratorPadding0;
    uint IntegratorPadding1;
    uint IntegratorPadding2;
};

struct Counter64Parts
{
    uint Low;
    uint High;
};

struct GroupStatistics
{
    Counter64Parts PrimaryRays;
    Counter64Parts ShadowRays;
    Counter64Parts ReflectionRays;
    Counter64Parts PrimitiveTests;
};

struct RayData
{
    float3 Origin;
    float3 Direction;
    float MinimumDistance;
    float MaximumDistance;
};

struct HitData
{
    float Distance;
    float3 Normal;
    float3 Color;
    uint MaterialIndex;
    float2 TexCoord;
    uint HasTexCoord;
};

uint Hash(uint value)
{
    value ^= value >> 16u;
    value *= 0x7feb352du;
    value ^= value >> 15u;
    value *= 0x846ca68bu;
    value ^= value >> 16u;
    return value;
}

float Random01(uint seed)
{
    return float(Hash(seed) & 0x00ffffffu) / 16777216.0f;
}

float SRGBToLinear(float value)
{
    return value <= 0.04045f
        ? value / 12.92f
        : pow((value + 0.055f) / 1.055f, 2.4f);
}

int WrapTextureCoordinate(int value, uint size)
{
    const int signedSize = int(size);
    const int wrapped = value % signedSize;
    return wrapped < 0 ? wrapped + signedSize : wrapped;
}

float3 LoadDiffuseTexel(
    device const uint* textureTexels,
    PackedTextureDescriptor descriptor,
    int x,
    int y)
{
    const uint offset = descriptor.OffsetTexels
        + uint(WrapTextureCoordinate(y, descriptor.Height)) * descriptor.Width
        + uint(WrapTextureCoordinate(x, descriptor.Width));
    const uint packed = textureTexels[offset];
    return float3(
        SRGBToLinear(float(packed & 0xffu) / 255.0f),
        SRGBToLinear(float((packed >> 8u) & 0xffu) / 255.0f),
        SRGBToLinear(float((packed >> 16u) & 0xffu) / 255.0f));
}

float3 SampleDiffuse(
    device const PackedTextureDescriptor* textures,
    device const uint* textureTexels,
    constant FrameConstants& constants,
    uint textureIndex,
    float2 texCoord,
    uint hasTexCoord)
{
    if (hasTexCoord == 0u || textureIndex >= constants.TextureCount)
    {
        return float3(1.0f);
    }
    const PackedTextureDescriptor descriptor = textures[textureIndex];
    if (descriptor.Width == 0u || descriptor.Height == 0u
        || descriptor.OffsetTexels >= constants.TextureTexelCount)
    {
        return float3(1.0f);
    }
    const uint availableTexels =
        constants.TextureTexelCount - descriptor.OffsetTexels;
    if (descriptor.Height > availableTexels / descriptor.Width)
    {
        return float3(1.0f);
    }

    const float x = fract(texCoord.x) * float(descriptor.Width) - 0.5f;
    const float y = fract(1.0f - texCoord.y) * float(descriptor.Height) - 0.5f;
    const float floorX = floor(x);
    const float floorY = floor(y);
    const int x0 = int(floorX);
    const int y0 = int(floorY);
    const float amountX = x - floorX;
    const float amountY = y - floorY;
    const float3 top = mix(
        LoadDiffuseTexel(textureTexels, descriptor, x0, y0),
        LoadDiffuseTexel(textureTexels, descriptor, x0 + 1, y0),
        amountX);
    const float3 bottom = mix(
        LoadDiffuseTexel(textureTexels, descriptor, x0, y0 + 1),
        LoadDiffuseTexel(textureTexels, descriptor, x0 + 1, y0 + 1),
        amountX);
    return mix(top, bottom, amountY);
}

void IncrementCounter(thread uint2& counter)
{
    const uint previousLow = counter.x;
    counter.x += 1u;
    if (counter.x < previousLow)
    {
        counter.y += 1u;
    }
}

uint2 AddCounter(uint2 left, uint2 right)
{
    const uint low = left.x + right.x;
    return uint2(low, left.y + right.y + (low < left.x ? 1u : 0u));
}

bool IntersectBounds(PackedBvhNode node, RayData ray, float closestDistance)
{
    const float3 minimum = float3(node.Minimum);
    const float3 maximum = float3(node.Maximum);
    float nearDistance = ray.MinimumDistance;
    float farDistance = min(ray.MaximumDistance, closestDistance);
    for (uint axis = 0u; axis < 3u; ++axis)
    {
        const float origin = ray.Origin[axis];
        const float direction = ray.Direction[axis];
        if (fabs(direction) < 1.0e-12f)
        {
            if (origin < minimum[axis] || origin > maximum[axis])
            {
                return false;
            }
            continue;
        }
        const float inverseDirection = 1.0f / direction;
        float distance0 = (minimum[axis] - origin) * inverseDirection;
        float distance1 = (maximum[axis] - origin) * inverseDirection;
        if (distance0 > distance1)
        {
            const float temporary = distance0;
            distance0 = distance1;
            distance1 = temporary;
        }
        nearDistance = max(nearDistance, distance0);
        farDistance = min(farDistance, distance1);
        if (nearDistance > farDistance)
        {
            return false;
        }
    }
    return true;
}

bool IntersectTriangle(
    PackedTriangle primitive,
    RayData ray,
    thread HitData& hit)
{
    const float3 position0 = primitive.Position0.xyz;
    const float3 edge1 = primitive.Edge1.xyz;
    const float3 edge2 = primitive.Edge2.xyz;
    const float3 determinantVector = cross(ray.Direction, edge2);
    const float determinant = dot(edge1, determinantVector);
    if (fabs(determinant) < 1.0e-8f)
    {
        return false;
    }
    const float inverseDeterminant = 1.0f / determinant;
    const float3 originOffset = ray.Origin - position0;
    const float barycentric1 = dot(originOffset, determinantVector) * inverseDeterminant;
    if (barycentric1 < 0.0f || barycentric1 > 1.0f)
    {
        return false;
    }
    const float3 crossValue = cross(originOffset, edge1);
    const float barycentric2 = dot(ray.Direction, crossValue) * inverseDeterminant;
    if (barycentric2 < 0.0f || barycentric1 + barycentric2 > 1.0f)
    {
        return false;
    }
    const float distance = dot(edge2, crossValue) * inverseDeterminant;
    if (distance < ray.MinimumDistance || distance > ray.MaximumDistance
        || distance >= hit.Distance)
    {
        return false;
    }

    const float barycentric0 = 1.0f - barycentric1 - barycentric2;
    float3 normal = normalize(cross(edge1, edge2));
    if ((primitive.Flags & 1u) != 0u)
    {
        const float3 smoothNormal = primitive.Normal0.xyz * barycentric0
            + primitive.Normal1.xyz * barycentric1
            + primitive.Normal2.xyz * barycentric2;
        if (dot(smoothNormal, smoothNormal) > 1.0e-12f)
        {
            normal = normalize(smoothNormal);
        }
    }
    if (dot(normal, ray.Direction) > 0.0f)
    {
        normal = -normal;
    }
    hit.Distance = distance;
    hit.Normal = normal;
    hit.Color = primitive.Color0.xyz * barycentric0
        + primitive.Color1.xyz * barycentric1
        + primitive.Color2.xyz * barycentric2;
    hit.MaterialIndex = primitive.MaterialIndex;
    hit.TexCoord = primitive.TexCoord01.xy * barycentric0
        + primitive.TexCoord01.zw * barycentric1
        + primitive.TexCoord2.xy * barycentric2;
    hit.HasTexCoord = 1u;
    return true;
}

bool IntersectSphere(
    PackedSphere sphere,
    RayData ray,
    thread HitData& hit)
{
    const float3 offset = ray.Origin - sphere.CenterRadius.xyz;
    const float halfB = dot(offset, ray.Direction);
    const float c = dot(offset, offset)
        - sphere.CenterRadius.w * sphere.CenterRadius.w;
    const float discriminant = halfB * halfB - c;
    if (discriminant < 0.0f)
    {
        return false;
    }
    const float root = sqrt(discriminant);
    float distance = -halfB - root;
    if (distance < ray.MinimumDistance)
    {
        distance = -halfB + root;
    }
    if (distance < ray.MinimumDistance || distance > ray.MaximumDistance
        || distance >= hit.Distance)
    {
        return false;
    }
    const float3 position = ray.Origin + ray.Direction * distance;
    float3 normal = normalize(position - sphere.CenterRadius.xyz);
    if (dot(normal, ray.Direction) > 0.0f)
    {
        normal = -normal;
    }
    hit.Distance = distance;
    hit.Normal = normal;
    hit.Color = float3(1.0f);
    hit.MaterialIndex = sphere.MaterialIndex;
    hit.TexCoord = float2(0.0f);
    hit.HasTexCoord = 0u;
    return true;
}

bool IntersectScene(
    device const PackedBvhNode* nodes,
    device const uint* primitiveRefs,
    device const PackedTriangle* triangles,
    device const PackedSphere* spheres,
    constant FrameConstants& constants,
    RayData ray,
    bool anyHit,
    thread HitData& hit,
    thread uint2& primitiveTests)
{
    if (constants.NodeCount == 0u)
    {
        return false;
    }
    bool found = false;
    uint nodeIndex = 0u;
    while (nodeIndex < constants.NodeCount)
    {
        const PackedBvhNode node = nodes[nodeIndex];
        if (!IntersectBounds(node, ray, hit.Distance))
        {
            nodeIndex = node.PrimitiveCount == 0u
                ? node.FirstOrEscape
                : nodeIndex + 1u;
            continue;
        }
        if (node.PrimitiveCount == 0u)
        {
            ++nodeIndex;
            continue;
        }
        for (uint offset = 0u; offset < node.PrimitiveCount; ++offset)
        {
            const uint encoded = primitiveRefs[node.FirstOrEscape + offset];
            IncrementCounter(primitiveTests);
            bool primitiveHit = false;
            if ((encoded & SphereBit) != 0u)
            {
                primitiveHit = IntersectSphere(
                    spheres[encoded & ~SphereBit],
                    ray,
                    hit);
            }
            else
            {
                primitiveHit = IntersectTriangle(
                    triangles[encoded],
                    ray,
                    hit);
            }
            if (primitiveHit)
            {
                found = true;
                if (anyHit)
                {
                    return true;
                }
            }
        }
        ++nodeIndex;
    }
    return found;
}

bool IsLightVisible(
    device const PackedBvhNode* nodes,
    device const uint* primitiveRefs,
    device const PackedTriangle* triangles,
    device const PackedSphere* spheres,
    constant FrameConstants& constants,
    float3 origin,
    float3 direction,
    float maximumDistance,
    thread uint2& primitiveTests)
{
    RayData shadowRay;
    shadowRay.Origin = origin;
    shadowRay.Direction = direction;
    shadowRay.MinimumDistance = RayEpsilon;
    shadowRay.MaximumDistance = maximumDistance;
    HitData shadowHit;
    shadowHit.Distance = maximumDistance;
    shadowHit.Normal = float3(0.0f);
    shadowHit.Color = float3(1.0f);
    shadowHit.MaterialIndex = 0u;
    shadowHit.TexCoord = float2(0.0f);
    shadowHit.HasTexCoord = 0u;
    return !IntersectScene(
        nodes,
        primitiveRefs,
        triangles,
        spheres,
        constants,
        shadowRay,
        true,
        shadowHit,
        primitiveTests);
}

float3 Background(constant FrameConstants& constants, float3 direction)
{
    const float blend = clamp(direction.y * 0.5f + 0.5f, 0.0f, 1.0f);
    return mix(
        float3(constants.BackgroundBottom),
        float3(constants.BackgroundTop),
        blend);
}

float3 NormalizeSafe(float3 value, float3 fallback)
{
    const float lengthSquared = dot(value, value);
    return lengthSquared > 1.0e-12f
        ? value * rsqrt(lengthSquared)
        : fallback;
}

void AccumulateLightSample(
    float3 lightDirection,
    float3 incidentRadiance,
    float visibility,
    float3 surfaceNormal,
    float3 viewDirection,
    float shininess,
    thread float3& diffuseLight,
    thread float3& specularLight)
{
    const float directionLengthSquared = dot(lightDirection, lightDirection);
    if (directionLengthSquared <= 1.0e-12f)
    {
        return;
    }
    const float3 direction = lightDirection * rsqrt(directionLengthSquared);
    const float normalToLight = max(dot(surfaceNormal, direction), 0.0f);
    if (normalToLight <= 0.0f)
    {
        return;
    }
    const float3 visibleRadiance = incidentRadiance * visibility;
    diffuseLight += visibleRadiance * normalToLight;
    const float3 halfVector = NormalizeSafe(
        direction + viewDirection,
        surfaceNormal);
    const float specularAmount = pow(
        max(dot(surfaceNormal, halfVector), 0.0f),
        max(shininess, 1.0f));
    specularLight += visibleRadiance * specularAmount;
}

float3 LinearToSRGB(float3 color)
{
    color = clamp(color, float3(0.0f), float3(1.0f));
    const float3 low = color * 12.92f;
    const float3 high = 1.055f * pow(color, float3(1.0f / 2.4f)) - 0.055f;
    return select(high, low, color <= float3(0.0031308f));
}

kernel void ArvoTracePixels(
    device const PackedBvhNode* nodes [[buffer(0)]],
    device const uint* primitiveRefs [[buffer(1)]],
    device const PackedTriangle* triangles [[buffer(2)]],
    device const PackedSphere* spheres [[buffer(3)]],
    device const PackedMaterial* materials [[buffer(4)]],
    device const PackedLight* lights [[buffer(5)]],
    constant FrameConstants& constants [[buffer(6)]],
    device float4* accumulation [[buffer(7)]],
    device uint* sampleCounts [[buffer(8)]],
    device GroupStatistics* groupStatistics [[buffer(9)]],
    device const PackedTextureDescriptor* textures [[buffer(10)]],
    device const uint* textureTexels [[buffer(11)]],
    uint gridIndex [[thread_position_in_grid]],
    uint localIndex [[thread_index_in_threadgroup]],
    uint3 groupPosition [[threadgroup_position_in_grid]])
{
    uint2 primaryRays = uint2(0u);
    uint2 shadowRays = uint2(0u);
    uint2 reflectionRays = uint2(0u);
    uint2 primitiveTests = uint2(0u);
    if (gridIndex < constants.PixelCount)
    {
        const uint pixelIndex = constants.PixelBegin + gridIndex;
        const uint x = pixelIndex % constants.Width;
        const uint y = pixelIndex / constants.Width;
        float3 accumulated = float3(0.0f);
        for (uint sampleOffset = 0u;
             sampleOffset < constants.SampleCount;
             ++sampleOffset)
        {
            const uint sample = constants.SampleBegin + sampleOffset;
            IncrementCounter(primaryRays);
            const uint pathSeed = ((y + constants.TileY) * constants.FullWidth + x + constants.TileX) * 9781u + sample * 6271u;
            float jitterX = 0.5f;
            float jitterY = 0.5f;
            if (constants.TotalSamples > 1u)
            {
                const uint seed = pathSeed;
                jitterX = Random01(seed);
                jitterY = Random01(seed ^ 0x68bc21ebu);
            }
            const float u = (float(x + constants.TileX) + jitterX) / float(constants.FullWidth);
            const float v = (float(y + constants.TileY) + jitterY) / float(constants.FullHeight);
            const float screenX = (2.0f * u - 1.0f) * constants.HalfWidth;
            const float screenY = (1.0f - 2.0f * v) * constants.HalfHeight;
            RayData ray;
            ray.Origin = float3(constants.CameraPosition);
            ray.Direction = normalize(
                float3(constants.CameraForward)
                + float3(constants.CameraRight) * screenX
                + float3(constants.CameraUp) * screenY);
            ray.MinimumDistance = RayEpsilon;
            ray.MaximumDistance = 1.0e30f;
            float3 radiance = float3(0.0f);
            float3 throughput = float3(1.0f);
            bool previousDelta = true;
            for (uint bounce = 0u;
                 bounce <= constants.MaxReflectionBounces;
                 ++bounce)
            {
                HitData hit;
                hit.Distance = ray.MaximumDistance;
                hit.Normal = float3(0.0f);
                hit.Color = float3(1.0f);
                hit.MaterialIndex = 0u;
                hit.TexCoord = float2(0.0f);
                hit.HasTexCoord = 0u;
                if (!IntersectScene(
                        nodes,
                        primitiveRefs,
                        triangles,
                        spheres,
                        constants,
                        ray,
                        false,
                        hit,
                        primitiveTests))
                {
                    radiance += throughput * Background(constants, ray.Direction);
                    break;
                }
                const PackedMaterial material = materials[
                    min(hit.MaterialIndex, constants.MaterialCount - 1u)];
                const float3 surface = max(hit.Color, float3(0.0f))
                    * max(
                        SampleDiffuse(
                            textures,
                            textureTexels,
                            constants,
                            material.DiffuseTextureIndex,
                            hit.TexCoord,
                            hit.HasTexCoord),
                        float3(0.0f));
                const float3 position = ray.Origin + ray.Direction * hit.Distance;
                const float3 viewDirection = NormalizeSafe(-ray.Direction, hit.Normal);
                float3 diffuseLight = float3(0.0f);
                float3 specularLight = float3(0.0f);
                for (uint lightIndex = 0u;
                     lightIndex < constants.LightCount;
                     ++lightIndex)
                {
                    const PackedLight light = lights[lightIndex];
                    if (light.Intensity <= 0.0f)
                    {
                        continue;
                    }
                    const float3 lightColor = float3(light.Color);
                    if (light.Type == LightTypeDirectional)
                    {
                        const float3 directionToLight =
                            normalize(float3(light.Direction));
                        const float cosine =
                            max(0.0f, dot(hit.Normal, directionToLight));
                        if (cosine <= 0.0f)
                        {
                            continue;
                        }
                        float visibility = 1.0f;
                        if (constants.ShadowStrength > 0.0f
                            && light.CastShadows != 0u)
                        {
                            IncrementCounter(shadowRays);
                            const float3 shadowNormal =
                                dot(hit.Normal, directionToLight) >= 0.0f
                                    ? hit.Normal
                                    : -hit.Normal;
                            const bool visible = IsLightVisible(
                                nodes,
                                primitiveRefs,
                                triangles,
                                spheres,
                                constants,
                                position + shadowNormal * RayEpsilon,
                                directionToLight,
                                1.0e30f,
                                primitiveTests);
                            if (!visible)
                            {
                                visibility = 1.0f - constants.ShadowStrength;
                            }
                        }
                        AccumulateLightSample(
                            directionToLight,
                            max(lightColor, float3(0.0f))
                                * max(light.Intensity, 0.0f),
                            visibility,
                            hit.Normal,
                            viewDirection,
                            material.AmbientShininess.w,
                            diffuseLight,
                            specularLight);
                        continue;
                    }
                    if (light.Type != LightTypePoint
                        && light.Type != LightTypeRectArea)
                    {
                        continue;
                    }

                    const bool isArea = light.Type == LightTypeRectArea;
                    float3 emissionDirection = float3(0.0f);
                    float3 tangent = float3(0.0f);
                    float3 bitangent = float3(0.0f);
                    uint finiteSampleCount = 1u;
                    if (isArea)
                    {
                        emissionDirection = normalize(float3(light.Direction));
                        const float3 helper = fabs(emissionDirection.y) < 0.999f
                            ? float3(0.0f, 1.0f, 0.0f)
                            : float3(1.0f, 0.0f, 0.0f);
                        tangent = normalize(cross(helper, emissionDirection));
                        bitangent =
                            normalize(cross(emissionDirection, tangent));
                        finiteSampleCount = 4u;
                    }
                    for (uint finiteSample = 0u;
                         finiteSample < finiteSampleCount;
                         ++finiteSample)
                    {
                        float3 samplePosition = float3(light.Position);
                        float sampleWeight = 1.0f;
                        if (isArea)
                        {
                            float offsetX = (finiteSample & 1u) != 0u ? 0.25f : -0.25f;
                            float offsetY = (finiteSample & 2u) != 0u ? 0.25f : -0.25f;
                            if (constants.Integrator != 0u)
                            {
                                const uint lightSeed = pathSeed ^ ((bounce + 1u) * 0x9e3779b9u)
                                    ^ ((lightIndex + 1u) * 0x85ebca6bu) ^ ((finiteSample + 1u) * 0xc2b2ae35u);
                                offsetX = (float(finiteSample & 1u) + Random01(lightSeed)) * 0.5f - 0.5f;
                                offsetY = (float(finiteSample >> 1u) + Random01(lightSeed ^ 0x68bc21ebu)) * 0.5f - 0.5f;
                            }
                            samplePosition += tangent * (offsetX * light.Width)
                                + bitangent * (offsetY * light.Height);
                            sampleWeight = 0.25f;
                        }
                        const float3 toLight = samplePosition - position;
                        const float distanceSquared = dot(toLight, toLight);
                        if (!isfinite(distanceSquared)
                            || distanceSquared <= RayEpsilon * RayEpsilon)
                        {
                            continue;
                        }
                        const float distance = sqrt(distanceSquared);
                        const float3 directionToLight = toLight / distance;
                        const float surfaceCosine =
                            max(0.0f, dot(hit.Normal, directionToLight));
                        if (surfaceCosine <= 0.0f)
                        {
                            continue;
                        }
                        float emitterCosine = 1.0f;
                        if (isArea)
                        {
                            emitterCosine = max(
                                0.0f,
                                dot(emissionDirection, -directionToLight));
                            if (emitterCosine <= 0.0f)
                            {
                                continue;
                            }
                        }

                        float visibility = 1.0f;
                        const float shadowMaximumDistance =
                            distance - RayEpsilon;
                        if (constants.ShadowStrength > 0.0f
                            && light.CastShadows != 0u
                            && shadowMaximumDistance > RayEpsilon)
                        {
                            IncrementCounter(shadowRays);
                            const bool visible = IsLightVisible(
                                nodes,
                                primitiveRefs,
                                triangles,
                                spheres,
                                constants,
                                position + hit.Normal * RayEpsilon,
                                directionToLight,
                                shadowMaximumDistance,
                                primitiveTests);
                            if (!visible)
                            {
                                visibility = 1.0f - constants.ShadowStrength;
                            }
                        }
                        const float3 incidentRadiance =
                            max(lightColor, float3(0.0f))
                            * (max(light.Intensity, 0.0f)
                                * emitterCosine
                                * sampleWeight
                                / distanceSquared);
                        AccumulateLightSample(
                            directionToLight,
                            incidentRadiance,
                            visibility,
                            hit.Normal,
                            viewDirection,
                            material.AmbientShininess.w,
                            diffuseLight,
                            specularLight);
                    }
                }
                float3 direct = constants.Integrator != 0u
                    ? max(material.Emission.xyz, float3(0.0f))
                        + surface * max(material.AlbedoReflectivity.xyz, float3(0.0f)) * diffuseLight * 0.31830988618f
                    : surface
                        * max(material.AmbientShininess.xyz, float3(0.0f))
                        * max(constants.AmbientIntensity, 0.0f)
                    + surface
                        * max(material.AlbedoReflectivity.xyz, float3(0.0f))
                        * diffuseLight
                    + max(material.Specular.xyz, float3(0.0f)) * specularLight
                    + max(material.Emission.xyz, float3(0.0f));
                const float reflectivity =
                    clamp(material.AlbedoReflectivity.w, 0.0f, 1.0f);
                if (constants.Integrator != 0u)
                {
                    radiance += throughput * ((previousDelta || material.SampledEmitter == 0u
                        ? material.Emission.xyz : float3(0.0f, 0.0f, 0.0f))
                        + (direct - material.Emission.xyz) * (1.0f - reflectivity));
                    if (bounce == constants.MaxReflectionBounces) break;
                    const uint seed = pathSeed ^ ((bounce + 1u) * 0x9e3779b9u);
                    float3 outgoing;
                    previousDelta = Random01(seed ^ 0xa511e9b3u) < reflectivity;
                    if (previousDelta)
                    {
                        outgoing = normalize(reflect(ray.Direction, hit.Normal));
                    }
                    else
                    {
                        const float radius = sqrt(Random01(seed));
                        const float phi = 6.28318530718f * Random01(seed ^ 0x68bc21ebu);
                        const float3 helper = abs(hit.Normal.y) < 0.999f
                            ? float3(0.0f, 1.0f, 0.0f) : float3(1.0f, 0.0f, 0.0f);
                        const float3 tangent = normalize(cross(helper, hit.Normal));
                        const float3 bitangent = cross(hit.Normal, tangent);
                        outgoing = normalize(tangent * (radius * cos(phi))
                            + bitangent * (radius * sin(phi))
                            + hit.Normal * sqrt(max(0.0f, 1.0f - radius * radius)));
                        throughput *= clamp(surface * material.AlbedoReflectivity.xyz, 0.0f, 1.0f);
                    }
                    ray.Origin = position + hit.Normal * RayEpsilon;
                    ray.Direction = outgoing;
                    IncrementCounter(reflectionRays);
                    continue;
                }
                radiance += throughput * direct * (1.0f - reflectivity);
                if (reflectivity <= 0.0f
                    || bounce == constants.MaxReflectionBounces)
                {
                    break;
                }
                throughput *= reflectivity;
                const float3 reflectedDirection =
                    normalize(reflect(ray.Direction, hit.Normal));
                const float3 reflectionNormal =
                    dot(hit.Normal, reflectedDirection) >= 0.0f
                        ? hit.Normal
                        : -hit.Normal;
                ray.Origin = position + reflectionNormal * RayEpsilon;
                ray.Direction = reflectedDirection;
                IncrementCounter(reflectionRays);
            }
            accumulated += radiance;
        }
        float4 current = accumulation[pixelIndex];
        current.xyz += accumulated;
        accumulation[pixelIndex] = current;
        sampleCounts[pixelIndex] += constants.SampleCount;
    }

    threadgroup uint2 primaryShared[256];
    threadgroup uint2 shadowShared[256];
    threadgroup uint2 reflectionShared[256];
    threadgroup uint2 primitiveShared[256];
    primaryShared[localIndex] = primaryRays;
    shadowShared[localIndex] = shadowRays;
    reflectionShared[localIndex] = reflectionRays;
    primitiveShared[localIndex] = primitiveTests;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128u; stride != 0u; stride >>= 1u)
    {
        if (localIndex < stride)
        {
            primaryShared[localIndex] = AddCounter(
                primaryShared[localIndex],
                primaryShared[localIndex + stride]);
            shadowShared[localIndex] = AddCounter(
                shadowShared[localIndex],
                shadowShared[localIndex + stride]);
            reflectionShared[localIndex] = AddCounter(
                reflectionShared[localIndex],
                reflectionShared[localIndex + stride]);
            primitiveShared[localIndex] = AddCounter(
                primitiveShared[localIndex],
                primitiveShared[localIndex + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (localIndex == 0u)
    {
        GroupStatistics value;
        value.PrimaryRays.Low = primaryShared[0].x;
        value.PrimaryRays.High = primaryShared[0].y;
        value.ShadowRays.Low = shadowShared[0].x;
        value.ShadowRays.High = shadowShared[0].y;
        value.ReflectionRays.Low = reflectionShared[0].x;
        value.ReflectionRays.High = reflectionShared[0].y;
        value.PrimitiveTests.Low = primitiveShared[0].x;
        value.PrimitiveTests.High = primitiveShared[0].y;
        groupStatistics[groupPosition.x] = value;
    }
}

kernel void ArvoResolvePixels(
    device const float4* accumulation [[buffer(0)]],
    device const uint* sampleCounts [[buffer(1)]],
    device uint* outputPixels [[buffer(2)]],
    constant FrameConstants& constants [[buffer(3)]],
    uint gridIndex [[thread_position_in_grid]])
{
    if (gridIndex >= constants.PixelCount)
    {
        return;
    }
    const uint pixelIndex = constants.PixelBegin + gridIndex;
    const uint sampleCount = sampleCounts[pixelIndex];
    float3 color = float3(0.0f);
    if (sampleCount != 0u)
    {
        color = max(
            accumulation[pixelIndex].xyz / float(sampleCount),
            float3(0.0f));
        color /= max(1.0f, max(color.r, max(color.g, color.b)));
        if (constants.GammaCorrection != 0u)
        {
            color = LinearToSRGB(color);
        }
    }
    const uint3 packed = uint3(color * 255.0f + 0.5f);
    outputPixels[pixelIndex] =
        packed.x | (packed.y << 8u) | (packed.z << 16u) | 0xff000000u;
}
)MSL";

[[nodiscard]] std::string ErrorDescription(NSError* error, const char* fallback)
{
    if (error != nil)
    {
        const char* description = [[error localizedDescription] UTF8String];
        if (description != nullptr)
        {
            return description;
        }
    }
    return fallback;
}

void ReleaseObject(id object)
{
    if (object != nil)
    {
        [object release];
    }
}

struct CommandCompletionState
{
    std::mutex Mutex;
    std::condition_variable Condition;
    bool Completed = false;
};

} // namespace

struct MetalRayTracingRHI::Impl
{
    struct AvailabilitySnapshot
    {
        bool Available = false;
        bool Terminal = false;
        std::string Error;
        std::string DeviceName;
    };

    struct Buffer
    {
        Buffer() = default;

        Buffer(const Buffer&) = delete;
        Buffer& operator=(const Buffer&) = delete;

        Buffer(Buffer&& other) noexcept
            : Handle(other.Handle)
            , LogicalSize(other.LogicalSize)
        {
            other.Handle = nil;
            other.LogicalSize = 0u;
        }

        Buffer& operator=(Buffer&& other) noexcept
        {
            if (this != &other)
            {
                Reset();
                Handle = other.Handle;
                LogicalSize = other.LogicalSize;
                other.Handle = nil;
                other.LogicalSize = 0u;
            }
            return *this;
        }

        ~Buffer()
        {
            Reset();
        }

        void Reset()
        {
            ReleaseObject(Handle);
            Handle = nil;
            LogicalSize = 0u;
        }

        id<MTLBuffer> Handle = nil;
        std::size_t LogicalSize = 0u;
    };

    Impl()
    {
        Initialize();
    }

    ~Impl()
    {
        MetalQueue.WaitIdle();
        Shutdown();
    }

    [[nodiscard]] AvailabilitySnapshot GetAvailability() const
    {
        std::lock_guard<std::mutex> lock(AvailabilityMutex);
        return {Available, Terminal, AvailabilityError, DeviceName};
    }

    void SetUnavailable(std::string error)
    {
        std::lock_guard<std::mutex> lock(AvailabilityMutex);
        Available = false;
        if (!Terminal)
        {
            AvailabilityError = std::move(error);
        }
    }

    void SetAvailable(std::string deviceName)
    {
        std::lock_guard<std::mutex> lock(AvailabilityMutex);
        Available = true;
        Terminal = false;
        AvailabilityError.clear();
        DeviceName = std::move(deviceName);
    }

    void MarkTerminal(std::string error) const
    {
        std::lock_guard<std::mutex> lock(AvailabilityMutex);
        Available = false;
        Terminal = true;
        AvailabilityError = std::move(error);
    }

    RenderJob Submit(
        const Scene* scene,
        std::shared_ptr<const Scene> owner,
        Camera camera,
        RenderSettings settings,
        CancellationToken external = CancellationToken()) const
    {
        return MetalQueue.Submit(
            [this,
             scene,
             owner = std::move(owner),
             camera,
             settings,
             external](const CancellationToken& queued) mutable
            {
                if (scene == nullptr)
                {
                    return Detail::MakeFailedResult(
                        "The Metal ray tracing submission has no scene.");
                }
                const Detail::CancellationProbe cancellation =
                    Detail::CombineCancellation(queued, external);
                if (cancellation())
                {
                    return Detail::MakeCancelledResult();
                }
                const AvailabilitySnapshot availability = GetAvailability();
                if (!availability.Available)
                {
                    return Detail::MakeFailedResult(
                        std::string("Metal compute ray tracing is unavailable: ")
                        + availability.Error);
                }
                @autoreleasepool
                {
                    return RenderNative(
                        *scene,
                        owner,
                        camera,
                        settings,
                        cancellation);
                }
            });
    }

    bool Initialize()
    {
        @autoreleasepool
        {
            Device = MTLCreateSystemDefaultDevice();
            if (Device == nil)
            {
                SetUnavailable("No Metal device is available.");
                return false;
            }
            CommandQueue = [Device newCommandQueue];
            if (CommandQueue == nil)
            {
                SetUnavailable(
                    "The Metal device could not create a command queue.");
                return false;
            }

            NSError* error = nil;
            MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
            options.fastMathEnabled = NO;
            NSString* source = [NSString stringWithUTF8String:MetalShaderSource];
            id<MTLLibrary> library = [Device
                newLibraryWithSource:source
                options:options
                error:&error];
            ReleaseObject(options);
            if (library == nil)
            {
                SetUnavailable(ErrorDescription(
                    error,
                    "Metal ray tracing shader compilation failed."));
                return false;
            }

            id<MTLFunction> traceFunction =
                [library newFunctionWithName:@"ArvoTracePixels"];
            id<MTLFunction> resolveFunction =
                [library newFunctionWithName:@"ArvoResolvePixels"];
            if (traceFunction == nil || resolveFunction == nil)
            {
                SetUnavailable(
                    "The Metal ray tracing library is missing a compute entry point.");
                ReleaseObject(traceFunction);
                ReleaseObject(resolveFunction);
                ReleaseObject(library);
                return false;
            }

            TracePipeline = [Device
                newComputePipelineStateWithFunction:traceFunction
                error:&error];
            if (TracePipeline == nil)
            {
                SetUnavailable(ErrorDescription(
                    error,
                    "Metal trace pipeline creation failed."));
            }
            ResolvePipeline = [Device
                newComputePipelineStateWithFunction:resolveFunction
                error:&error];
            if (ResolvePipeline == nil)
            {
                SetUnavailable(ErrorDescription(
                    error,
                    "Metal resolve pipeline creation failed."));
            }
            ReleaseObject(traceFunction);
            ReleaseObject(resolveFunction);
            ReleaseObject(library);
            if (TracePipeline == nil || ResolvePipeline == nil)
            {
                return false;
            }
            if (TracePipeline.maxTotalThreadsPerThreadgroup < ThreadsPerGroup
                || ResolvePipeline.maxTotalThreadsPerThreadgroup < ThreadsPerGroup)
            {
                SetUnavailable(
                    "The Metal device does not support 256-thread compute groups.");
                return false;
            }

            const char* name = [[Device name] UTF8String];
            SetAvailable(name != nullptr ? name : "Unknown Metal Device");
            return true;
        }
    }

    void Shutdown()
    {
        SetUnavailable("The Metal compute backend has been shut down.");
        CachedOwner.reset();
        ReleaseFrameBuffers();
        ReleaseSceneBuffers();
        ReleaseObject(TracePipeline);
        ReleaseObject(ResolvePipeline);
        ReleaseObject(CommandQueue);
        ReleaseObject(Device);
        TracePipeline = nil;
        ResolvePipeline = nil;
        CommandQueue = nil;
        Device = nil;
    }

    [[nodiscard]] bool CommandSucceeded(
        id<MTLCommandBuffer> commandBuffer,
        const char* operation,
        std::string& error) const
    {
        if (commandBuffer == nil)
        {
            error = std::string(operation) + " did not create a command buffer.";
            return false;
        }
        if (commandBuffer.status == MTLCommandBufferStatusCompleted)
        {
            return true;
        }
        error = std::string(operation) + " failed: "
            + ErrorDescription(
                commandBuffer.error,
                "the Metal command buffer did not complete.");
        return false;
    }

    [[nodiscard]] bool CommitAndWait(
        id<MTLCommandBuffer> commandBuffer,
        const char* operation,
        const Detail::CancellationProbe& cancellation,
        std::string& error) const
    {
        if (commandBuffer == nil)
        {
            error = std::string(operation) + " did not create a command buffer.";
            return false;
        }
        const std::shared_ptr<CommandCompletionState> completion =
            std::make_shared<CommandCompletionState>();
        [commandBuffer
            addCompletedHandler:^(id<MTLCommandBuffer>)
            {
                {
                    std::lock_guard<std::mutex> lock(completion->Mutex);
                    completion->Completed = true;
                }
                completion->Condition.notify_all();
            }];
        [commandBuffer commit];

        constexpr auto CommandTimeout = std::chrono::seconds(30);
        constexpr auto CancellationPollInterval = std::chrono::milliseconds(10);
        bool cancellationObserved = false;
        const auto deadline = std::chrono::steady_clock::now() + CommandTimeout;
        std::unique_lock<std::mutex> lock(completion->Mutex);
        while (!completion->Completed)
        {
            completion->Condition.wait_for(
                lock,
                CancellationPollInterval,
                [&completion]() { return completion->Completed; });
            if (cancellation && cancellation())
            {
                cancellationObserved = true;
            }
            if (!completion->Completed
                && std::chrono::steady_clock::now() >= deadline)
            {
                lock.unlock();
                error = std::string("Terminal Metal device-lost state: ")
                    + operation
                    + " timed out after 30 seconds; the backend was disabled "
                      "to protect resources retained by the in-flight command buffer.";
                MarkTerminal(error);
                return false;
            }
        }
        lock.unlock();
        if (!CommandSucceeded(commandBuffer, operation, error))
        {
            MarkTerminal(
                std::string("Terminal Metal device-lost state: ") + error);
            return false;
        }
        if (cancellationObserved)
        {
            error = std::string(operation)
                + " completed after cancellation was requested.";
            return false;
        }
        return true;
    }

    [[nodiscard]] bool CreateBuffer(
        Buffer& output,
        std::size_t logicalSize,
        const void* data,
        const char* label,
        std::string& error) const
    {
        const std::size_t allocationSize = std::max<std::size_t>(logicalSize, 16u);
        if (allocationSize > std::numeric_limits<NSUInteger>::max())
        {
            error = std::string(label) + " exceeds Metal's addressable buffer size.";
            return false;
        }
        id<MTLBuffer> buffer = [Device
            newBufferWithLength:static_cast<NSUInteger>(allocationSize)
            options:MTLResourceStorageModeShared];
        if (buffer == nil)
        {
            error = std::string("Metal could not allocate ") + label + ".";
            return false;
        }
        buffer.label = [NSString stringWithUTF8String:label];
        if (data != nullptr && logicalSize != 0u)
        {
            void* destination = buffer.contents;
            if (destination == nullptr)
            {
                ReleaseObject(buffer);
                error = std::string("Metal could not map ") + label + ".";
                return false;
            }
            std::memcpy(destination, data, logicalSize);
        }
        output.Reset();
        output.Handle = buffer;
        output.LogicalSize = logicalSize;
        return true;
    }

    template <typename Element>
    [[nodiscard]] bool CreateVectorBuffer(
        Buffer& output,
        const std::vector<Element>& source,
        const char* label,
        std::string& error) const
    {
        if (source.size() > std::numeric_limits<std::size_t>::max() / sizeof(Element))
        {
            error = std::string(label) + " size overflows host memory.";
            return false;
        }
        return CreateBuffer(
            output,
            source.size() * sizeof(Element),
            source.empty() ? nullptr : source.data(),
            label,
            error);
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
        NodeCount = 0u;
        PrimitiveRefCount = 0u;
        TriangleCount = 0u;
        SphereCount = 0u;
        MaterialCount = 0u;
        LightCount = 0u;
        TextureCount = 0u;
        TextureTexelCount = 0u;
        CachedOwner.reset();
    }

    void ReleaseFrameBuffers() const
    {
        Accumulation.Reset();
        SampleCounts.Reset();
        Output.Reset();
        GroupStats.Reset();
        FramePixelCapacity = 0u;
    }

    [[nodiscard]] bool OwnerMatches(
        const std::shared_ptr<const Scene>& owner) const
    {
        if (!owner)
        {
            return false;
        }
        const std::shared_ptr<const Scene> cached = CachedOwner.lock();
        return cached && cached.get() == owner.get();
    }

    [[nodiscard]] bool EnsureScene(
        const Scene& scene,
        const std::shared_ptr<const Scene>& owner,
        const Detail::CancellationProbe& cancellation,
        std::string& error) const
    {
        if (OwnerMatches(owner))
        {
            return true;
        }
        PackedScene packed;
        ScenePacker packer(cancellation);
        if (!packer.Pack(scene, packed, error))
        {
            return false;
        }
        if (cancellation())
        {
            error = "Metal scene construction was cancelled.";
            return false;
        }

        Buffer nodes;
        Buffer primitiveRefs;
        Buffer triangles;
        Buffer spheres;
        Buffer materials;
        Buffer lights;
        Buffer textureDescriptors;
        Buffer textureTexels;
        if (!CreateVectorBuffer(nodes, packed.Nodes, "Arvo Metal BVH nodes", error)
            || !CreateVectorBuffer(
                primitiveRefs,
                packed.PrimitiveRefs,
                "Arvo Metal primitive references",
                error)
            || !CreateVectorBuffer(
                triangles,
                packed.Triangles,
                "Arvo Metal triangles",
                error)
            || !CreateVectorBuffer(
                spheres,
                packed.Spheres,
                "Arvo Metal spheres",
                error)
            || !CreateVectorBuffer(
                materials,
                packed.Materials,
                "Arvo Metal materials",
                error)
            || !CreateVectorBuffer(
                lights,
                packed.Lights,
                "Arvo Metal lights",
                error)
            || !CreateVectorBuffer(
                textureDescriptors,
                packed.TextureAtlas.Descriptors,
                "Arvo Metal texture descriptors",
                error)
            || !CreateVectorBuffer(
                textureTexels,
                packed.TextureAtlas.Texels,
                "Arvo Metal texture texels",
                error))
        {
            return false;
        }
        if (cancellation())
        {
            error = "Metal scene upload was cancelled.";
            return false;
        }

        ReleaseSceneBuffers();
        Nodes = std::move(nodes);
        PrimitiveRefs = std::move(primitiveRefs);
        Triangles = std::move(triangles);
        Spheres = std::move(spheres);
        Materials = std::move(materials);
        Lights = std::move(lights);
        TextureDescriptors = std::move(textureDescriptors);
        TextureTexels = std::move(textureTexels);
        NodeCount = static_cast<std::uint32_t>(packed.Nodes.size());
        PrimitiveRefCount =
            static_cast<std::uint32_t>(packed.PrimitiveRefs.size());
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

    [[nodiscard]] bool EnsureFrameBuffers(
        std::uint32_t pixelCount,
        std::string& error) const
    {
        if (FramePixelCapacity >= pixelCount)
        {
            return true;
        }
        if (pixelCount > std::numeric_limits<std::size_t>::max() / sizeof(Vec4))
        {
            error = "Metal frame buffer dimensions overflow host memory.";
            return false;
        }

        Buffer accumulation;
        Buffer sampleCounts;
        Buffer output;
        Buffer groupStats;
        const std::size_t groupCapacity =
            (PixelChunkSize + ThreadsPerGroup - 1u) / ThreadsPerGroup;
        if (!CreateBuffer(
                accumulation,
                static_cast<std::size_t>(pixelCount) * sizeof(Vec4),
                nullptr,
                "Arvo Metal accumulation",
                error)
            || !CreateBuffer(
                sampleCounts,
                static_cast<std::size_t>(pixelCount) * sizeof(std::uint32_t),
                nullptr,
                "Arvo Metal sample counts",
                error)
            || !CreateBuffer(
                output,
                static_cast<std::size_t>(pixelCount) * sizeof(std::uint32_t),
                nullptr,
                "Arvo Metal output",
                error)
            || !CreateBuffer(
                groupStats,
                groupCapacity * sizeof(GroupStatistics),
                nullptr,
                "Arvo Metal group statistics",
                error))
        {
            return false;
        }
        ReleaseFrameBuffers();
        Accumulation = std::move(accumulation);
        SampleCounts = std::move(sampleCounts);
        Output = std::move(output);
        GroupStats = std::move(groupStats);
        FramePixelCapacity = pixelCount;
        return true;
    }

    [[nodiscard]] bool ClearFrame(
        std::uint32_t pixelCount,
        const Detail::CancellationProbe& cancellation,
        std::string& error) const
    {
        void* accumulation = Accumulation.Handle.contents;
        void* sampleCounts = SampleCounts.Handle.contents;
        if (accumulation == nullptr || sampleCounts == nullptr)
        {
            error = "Metal frame buffers are not CPU-visible.";
            return false;
        }
        constexpr std::size_t ClearChunkSize = 32u * 1024u * 1024u;
        const std::size_t accumulationBytes =
            static_cast<std::size_t>(pixelCount) * sizeof(Vec4);
        const std::size_t sampleBytes =
            static_cast<std::size_t>(pixelCount) * sizeof(std::uint32_t);
        for (std::size_t offset = 0u;
             offset < accumulationBytes;
             offset += ClearChunkSize)
        {
            if (cancellation())
            {
                error = "Metal frame clear was cancelled.";
                return false;
            }
            std::memset(
                static_cast<std::uint8_t*>(accumulation) + offset,
                0,
                std::min(ClearChunkSize, accumulationBytes - offset));
        }
        for (std::size_t offset = 0u; offset < sampleBytes; offset += ClearChunkSize)
        {
            if (cancellation())
            {
                error = "Metal frame clear was cancelled.";
                return false;
            }
            std::memset(
                static_cast<std::uint8_t*>(sampleCounts) + offset,
                0,
                std::min(ClearChunkSize, sampleBytes - offset));
        }
        return true;
    }

    [[nodiscard]] bool DispatchTrace(
        const FrameConstants& constants,
        GroupStatistics& totals,
        const Detail::CancellationProbe& cancellation,
        std::string& error) const
    {
        @autoreleasepool
        {
            id<MTLCommandBuffer> commandBuffer = [CommandQueue commandBuffer];
            if (commandBuffer == nil)
            {
                error = "Metal trace dispatch did not create a command buffer.";
                return false;
            }
            id<MTLComputeCommandEncoder> encoder =
                [commandBuffer computeCommandEncoder];
            if (encoder == nil)
            {
                error = "Metal trace dispatch did not create a compute encoder.";
                return false;
            }
            [encoder setComputePipelineState:TracePipeline];
            [encoder setBuffer:Nodes.Handle offset:0u atIndex:0u];
            [encoder setBuffer:PrimitiveRefs.Handle offset:0u atIndex:1u];
            [encoder setBuffer:Triangles.Handle offset:0u atIndex:2u];
            [encoder setBuffer:Spheres.Handle offset:0u atIndex:3u];
            [encoder setBuffer:Materials.Handle offset:0u atIndex:4u];
            [encoder setBuffer:Lights.Handle offset:0u atIndex:5u];
            [encoder
                setBytes:&constants
                length:sizeof(constants)
                atIndex:6u];
            [encoder setBuffer:Accumulation.Handle offset:0u atIndex:7u];
            [encoder setBuffer:SampleCounts.Handle offset:0u atIndex:8u];
            [encoder setBuffer:GroupStats.Handle offset:0u atIndex:9u];
            [encoder setBuffer:TextureDescriptors.Handle offset:0u atIndex:10u];
            [encoder setBuffer:TextureTexels.Handle offset:0u atIndex:11u];
            [encoder
                dispatchThreadgroups:MTLSizeMake(constants.GroupCount, 1u, 1u)
                threadsPerThreadgroup:MTLSizeMake(ThreadsPerGroup, 1u, 1u)];
            [encoder endEncoding];
            if (!CommitAndWait(
                    commandBuffer,
                    "Metal trace dispatch",
                    cancellation,
                    error))
            {
                return false;
            }
            if (cancellation())
            {
                error = "Metal trace dispatch was cancelled.";
                return false;
            }
        }

        const auto* groups =
            static_cast<const GroupStatistics*>(GroupStats.Handle.contents);
        if (groups == nullptr)
        {
            error = "Metal group statistics are not CPU-visible.";
            return false;
        }
        for (std::uint32_t group = 0u; group < constants.GroupCount; ++group)
        {
            const GroupStatistics& value = groups[group];
            auto add = [](Counter64Parts& destination, std::uint64_t amount)
            {
                const std::uint64_t total = destination.Value() + amount;
                destination.Low = static_cast<std::uint32_t>(total);
                destination.High = static_cast<std::uint32_t>(total >> 32u);
            };
            add(totals.PrimaryRays, value.PrimaryRays.Value());
            add(totals.ShadowRays, value.ShadowRays.Value());
            add(totals.ReflectionRays, value.ReflectionRays.Value());
            add(totals.PrimitiveTests, value.PrimitiveTests.Value());
        }
        return true;
    }

    [[nodiscard]] bool ResolvePixels(
        FrameConstants constants,
        std::uint32_t pixelCount,
        const Detail::CancellationProbe& cancellation,
        std::string& error) const
    {
        for (std::uint32_t begin = 0u; begin < pixelCount; begin += PixelChunkSize)
        {
            if (cancellation())
            {
                error = "Metal resolve was cancelled.";
                return false;
            }
            constants.PixelBegin = begin;
            constants.PixelCount = std::min(PixelChunkSize, pixelCount - begin);
            const std::uint32_t groupCount =
                (constants.PixelCount + ThreadsPerGroup - 1u) / ThreadsPerGroup;
            @autoreleasepool
            {
                id<MTLCommandBuffer> commandBuffer = [CommandQueue commandBuffer];
                if (commandBuffer == nil)
                {
                    error = "Metal resolve did not create a command buffer.";
                    return false;
                }
                id<MTLComputeCommandEncoder> encoder =
                    [commandBuffer computeCommandEncoder];
                if (encoder == nil)
                {
                    error = "Metal resolve did not create a compute encoder.";
                    return false;
                }
                [encoder setComputePipelineState:ResolvePipeline];
                [encoder setBuffer:Accumulation.Handle offset:0u atIndex:0u];
                [encoder setBuffer:SampleCounts.Handle offset:0u atIndex:1u];
                [encoder setBuffer:Output.Handle offset:0u atIndex:2u];
                [encoder
                    setBytes:&constants
                    length:sizeof(constants)
                    atIndex:3u];
                [encoder
                    dispatchThreadgroups:MTLSizeMake(groupCount, 1u, 1u)
                    threadsPerThreadgroup:MTLSizeMake(ThreadsPerGroup, 1u, 1u)];
                [encoder endEncoding];
                if (!CommitAndWait(
                        commandBuffer,
                        "Metal resolve",
                        cancellation,
                        error))
                {
                    return false;
                }
            }
        }
        return true;
    }

    [[nodiscard]] RenderResult RenderNative(
        const Scene& scene,
        const std::shared_ptr<const Scene>& owner,
        const Camera& camera,
        const RenderSettings& settings,
        const Detail::CancellationProbe& cancellation) const
    {
        RenderResult result;
        if (!ValidateInput(scene, camera, settings, result.Error))
        {
            return result;
        }
        const auto buildBegin = std::chrono::steady_clock::now();
        if (!EnsureScene(scene, owner, cancellation, result.Error))
        {
            if (cancellation())
            {
                return Detail::MakeCancelledResult();
            }
            return result;
        }
        const auto buildEnd = std::chrono::steady_clock::now();
        if (cancellation())
        {
            return Detail::MakeCancelledResult();
        }

        const std::uint32_t pixelCount = settings.Width * settings.Height;
        if (!EnsureFrameBuffers(pixelCount, result.Error)
            || !ClearFrame(pixelCount, cancellation, result.Error))
        {
            if (cancellation())
            {
                return Detail::MakeCancelledResult();
            }
            return result;
        }

        FrameConstants constants;
        constants.CameraPosition = ToVec3(camera.Position);
        constants.Width = settings.Width;
        constants.FullWidth = FullImageWidth(settings);
        constants.FullHeight = FullImageHeight(settings);
        constants.TileX = settings.TileX;
        constants.TileY = settings.TileY;
        constants.Integrator = static_cast<std::uint32_t>(settings.Integrator);
        constants.Height = settings.Height;
        constants.TotalSamples = settings.SamplesPerPixel;
        constants.MaxReflectionBounces = settings.MaxReflectionBounces;
        const Vec3 forward = NormalizeVector(
            Subtract(ToVec3(camera.Target), ToVec3(camera.Position)));
        const Vec3 right = NormalizeVector(CrossVector(forward, ToVec3(camera.Up)));
        constants.CameraForward = forward;
        constants.CameraRight = right;
        constants.CameraUp = NormalizeVector(CrossVector(right, forward));
        constants.HalfHeight =
            std::tan(camera.VerticalFieldOfViewDegrees * Pi / 360.0f);
        constants.HalfWidth = constants.HalfHeight
            * static_cast<float>(FullImageWidth(settings))
            / static_cast<float>(FullImageHeight(settings));
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
        for (std::uint32_t sample = 0u;
             sample < settings.SamplesPerPixel;
             ++sample)
        {
            for (std::uint32_t begin = 0u;
                 begin < pixelCount;
                 begin += PixelChunkSize)
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
                constants.SampleCount = 1u;
                constants.GroupCount =
                    (constants.PixelCount + ThreadsPerGroup - 1u)
                    / ThreadsPerGroup;
                GroupStatistics statistics;
                if (!DispatchTrace(
                        constants,
                        statistics,
                        cancellation,
                        result.Error))
                {
                    if (cancellation())
                    {
                        result.Status = RenderStatus::Cancelled;
                    }
                    return result;
                }
                result.Statistics.PrimaryRays += statistics.PrimaryRays.Value();
                result.Statistics.ShadowRays += statistics.ShadowRays.Value();
                result.Statistics.ReflectionRays +=
                    statistics.ReflectionRays.Value();
                result.Statistics.PrimitiveTests +=
                    statistics.PrimitiveTests.Value();
            }
        }

        if (!ResolvePixels(constants, pixelCount, cancellation, result.Error))
        {
            if (cancellation())
            {
                result.Status = RenderStatus::Cancelled;
            }
            return result;
        }
        if (cancellation())
        {
            return Detail::MakeCancelledResult();
        }

        const void* output = Output.Handle.contents;
        const auto* completedSamples =
            static_cast<const std::uint32_t*>(SampleCounts.Handle.contents);
        if (output == nullptr || completedSamples == nullptr)
        {
            result.Error = "Metal frame readback buffers are not CPU-visible.";
            return result;
        }
        result.Output = Image(settings.Width, settings.Height);
        std::vector<std::uint8_t>& pixels = result.Output.GetPixelsRGBA8();
        constexpr std::size_t ReadbackChunkSize = 32u * 1024u * 1024u;
        const std::size_t outputBytes =
            static_cast<std::size_t>(pixelCount) * sizeof(std::uint32_t);
        for (std::size_t offset = 0u;
             offset < outputBytes;
             offset += ReadbackChunkSize)
        {
            if (cancellation())
            {
                return Detail::MakeCancelledResult();
            }
            const std::size_t chunk =
                std::min(ReadbackChunkSize, outputBytes - offset);
            std::memcpy(
                pixels.data() + offset,
                static_cast<const std::uint8_t*>(output) + offset,
                chunk);
        }
        for (std::uint32_t pixel = 0u; pixel < pixelCount; ++pixel)
        {
            if (completedSamples[pixel] == settings.SamplesPerPixel)
            {
                ++result.Statistics.CompletedPixels;
            }
        }
        const auto renderEnd = std::chrono::steady_clock::now();
        result.Statistics.BuildMilliseconds =
            std::chrono::duration<double, std::milli>(
                buildEnd - buildBegin).count();
        result.Statistics.RenderMilliseconds =
            std::chrono::duration<double, std::milli>(
                renderEnd - renderBegin).count();
        result.Status = cancellation()
            ? RenderStatus::Cancelled
            : RenderStatus::Completed;
        return result;
    }

    id<MTLDevice> Device = nil;
    id<MTLCommandQueue> CommandQueue = nil;
    id<MTLComputePipelineState> TracePipeline = nil;
    id<MTLComputePipelineState> ResolvePipeline = nil;
    mutable std::mutex AvailabilityMutex;
    mutable bool Available = false;
    mutable bool Terminal = false;
    mutable std::string AvailabilityError;
    mutable std::string DeviceName;

    mutable Buffer Nodes;
    mutable Buffer PrimitiveRefs;
    mutable Buffer Triangles;
    mutable Buffer Spheres;
    mutable Buffer Materials;
    mutable Buffer Lights;
    mutable Buffer TextureDescriptors;
    mutable Buffer TextureTexels;
    mutable Buffer Accumulation;
    mutable Buffer SampleCounts;
    mutable Buffer Output;
    mutable Buffer GroupStats;
    mutable std::uint32_t NodeCount = 0u;
    mutable std::uint32_t PrimitiveRefCount = 0u;
    mutable std::uint32_t TriangleCount = 0u;
    mutable std::uint32_t SphereCount = 0u;
    mutable std::uint32_t MaterialCount = 0u;
    mutable std::uint32_t LightCount = 0u;
    mutable std::uint32_t TextureCount = 0u;
    mutable std::uint32_t TextureTexelCount = 0u;
    mutable std::uint32_t FramePixelCapacity = 0u;
    mutable std::weak_ptr<const Scene> CachedOwner;
    mutable Detail::PersistentRenderQueue MetalQueue;
};

MetalRayTracingRHI::MetalRayTracingRHI()
    : m_Impl(std::make_unique<Impl>())
{
}

MetalRayTracingRHI::~MetalRayTracingRHI() = default;

BackendInfo MetalRayTracingRHI::GetBackendInfo() const
{
    const Impl::AvailabilitySnapshot availability = m_Impl->GetAvailability();
    BackendInfo info;
    info.Type = BackendType::Metal;
    info.Name = "Metal Compute";
    info.IsNative = availability.Available;
    info.IsHardwareAccelerated = availability.Available;
    info.IsAvailable = availability.Available;
    info.Description = availability.Available
        ? std::string("Native Metal compute BVH backend on ")
            + availability.DeviceName
        : std::string("Unavailable")
            + (availability.Terminal ? " (terminal)" : "")
            + ": "
            + availability.Error;
    return info;
}

RenderResult MetalRayTracingRHI::Render(
    const Scene& scene,
    const Camera& camera,
    const RenderSettings& settings,
    const CancellationToken& cancellation) const
{
    return m_Impl->Submit(
        &scene,
        nullptr,
        camera,
        settings,
        cancellation).GetResult();
}

RenderJob MetalRayTracingRHI::RenderAsync(
    Scene scene,
    Camera camera,
    RenderSettings settings) const
{
    return RenderAsync(
        std::make_shared<Scene>(std::move(scene)),
        camera,
        settings);
}

RenderJob MetalRayTracingRHI::RenderAsync(
    std::shared_ptr<const Scene> scene,
    Camera camera,
    RenderSettings settings) const
{
    const Scene* pointer = scene.get();
    return m_Impl->Submit(pointer, std::move(scene), camera, settings);
}

void MetalRayTracingRHI::WaitIdle() const
{
    m_Impl->MetalQueue.WaitIdle();
}

} // namespace RayTracing
} // namespace Arvo

#endif
