#pragma once

#include "Distributed/Http.h"
#include "Renderer/RayTracing.h"

#include <cmath>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace Arvo::Distributed::Protocol
{
inline std::string Number(double value)
{
    std::ostringstream text;
    text << std::setprecision(17) << value;
    return text.str();
}

inline std::uint32_t UInt(const std::map<std::string, std::string>& fields,
                          const std::string& name, std::uint32_t minimum, std::uint32_t maximum)
{
    const auto& text = fields.at(name);
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error("Invalid numeric field: " + name);
    const auto value = std::stoull(text);
    if (value < minimum || value > maximum) throw std::runtime_error("Protocol field out of range: " + name);
    return static_cast<std::uint32_t>(value);
}

inline float Float(const std::map<std::string, std::string>& fields, const std::string& name)
{
    std::size_t used = 0;
    const float value = std::stof(fields.at(name), &used);
    if (used != fields.at(name).size() || !std::isfinite(value) || std::fabs(value) > 1000000.0f)
        throw std::runtime_error("Invalid camera field: " + name);
    return value;
}

inline void Append32(Http::Bytes& bytes, std::uint32_t value)
{
    for (int shift = 0; shift != 32; shift += 8) bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

inline void Append64(Http::Bytes& bytes, std::uint64_t value)
{
    for (int shift = 0; shift != 64; shift += 8) bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

inline std::uint32_t Read32(const Http::Bytes& bytes, std::size_t at)
{
    if (at + 4 > bytes.size()) throw std::runtime_error("Truncated tile response");
    std::uint32_t value = 0;
    for (int index = 0; index != 4; ++index) value |= std::uint32_t(bytes[at + index]) << (index * 8);
    return value;
}

inline std::uint64_t Read64(const Http::Bytes& bytes, std::size_t at)
{
    if (at + 8 > bytes.size()) throw std::runtime_error("Truncated tile response");
    std::uint64_t value = 0;
    for (int index = 0; index != 8; ++index) value |= std::uint64_t(bytes[at + index]) << (index * 8);
    return value;
}

inline std::uint64_t SceneHash(const RayTracing::Scene& scene)
{
    // Hash the rendering contract, not native struct storage or platform padding.
    // Counts delimit every collection; integers and IEEE float bits use little endian.
    std::uint64_t hash = 14695981039346656037ULL;
    const auto bytes = [&](const void* data, std::size_t size) {
        const auto* source = static_cast<const std::uint8_t*>(data);
        for (std::size_t index = 0; index < size; ++index)
        {
            hash ^= source[index];
            hash *= 1099511628211ULL;
        }
    };
    const auto integer = [&](std::uint64_t value, unsigned count) {
        for (unsigned index = 0; index < count; ++index)
        {
            const auto byte = static_cast<std::uint8_t>(value >> (index * 8));
            bytes(&byte, 1);
        }
    };
    const auto scalar = [&](float value) {
        static_assert(sizeof(float) == sizeof(std::uint32_t), "Scene hashes require 32-bit floats");
        std::uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        integer(bits, 4);
    };
    const auto vector = [&](const RayTracing::Float3& value) {
        scalar(value.X); scalar(value.Y); scalar(value.Z);
    };
    const auto uv = [&](const RayTracing::Float2& value) {
        scalar(value.X); scalar(value.Y);
    };
    constexpr char Schema[] = "arvo-render-scene-v3";
    bytes(Schema, sizeof(Schema));
    integer(scene.Materials.size(), 8);
    for (const auto& material : scene.Materials)
    {
        vector(material.Albedo); vector(material.AmbientColor);
        vector(material.SpecularColor); vector(material.Emission);
        scalar(material.Shininess); scalar(material.Reflectivity);
        integer(material.DiffuseTextureIndex, 4);
        integer(material.SampledEmitter ? 1 : 0, 1);
    }
    integer(scene.Textures.size(), 8);
    for (const auto& texture : scene.Textures)
    {
        integer(texture.Width, 4); integer(texture.Height, 4);
        integer(texture.PixelsRGBA8 ? 1 : 0, 1);
        if (texture.PixelsRGBA8)
        {
            integer(texture.PixelsRGBA8->size(), 8);
            bytes(texture.PixelsRGBA8->data(), texture.PixelsRGBA8->size());
        }
    }
    integer(scene.Triangles.size(), 8);
    for (const auto& triangle : scene.Triangles)
    {
        vector(triangle.Position0); vector(triangle.Position1); vector(triangle.Position2);
        vector(triangle.Normal0); vector(triangle.Normal1); vector(triangle.Normal2);
        vector(triangle.Color0); vector(triangle.Color1); vector(triangle.Color2);
        uv(triangle.TexCoord0); uv(triangle.TexCoord1); uv(triangle.TexCoord2);
        integer(triangle.MaterialIndex, 4);
        integer(triangle.SmoothNormals ? 1 : 0, 1);
    }
    integer(scene.Spheres.size(), 8);
    for (const auto& sphere : scene.Spheres)
    {
        vector(sphere.Center); scalar(sphere.Radius);
        integer(sphere.MaterialIndex, 4);
    }
    integer(scene.DirectionalLights.size(), 8);
    for (const auto& light : scene.DirectionalLights)
    {
        vector(light.DirectionToLight); vector(light.Color); scalar(light.Intensity);
        integer(light.CastShadows ? 1 : 0, 1);
    }
    integer(scene.PointLights.size(), 8);
    for (const auto& light : scene.PointLights)
    {
        vector(light.Position); vector(light.Color); scalar(light.Intensity);
        integer(light.CastShadows ? 1 : 0, 1);
    }
    integer(scene.RectAreaLights.size(), 8);
    for (const auto& light : scene.RectAreaLights)
    {
        vector(light.Position); vector(light.Direction); vector(light.Color);
        scalar(light.Intensity); uv(light.Size);
        integer(light.CastShadows ? 1 : 0, 1);
    }
    vector(scene.BackgroundTop); vector(scene.BackgroundBottom);
    scalar(scene.AmbientIntensity);
    return hash;
}

} // namespace Arvo::Distributed::Protocol
