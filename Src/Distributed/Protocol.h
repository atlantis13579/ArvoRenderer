#pragma once

#include "Distributed/Http.h"
#include "Distributed/Distributed.h"
#include "Renderer/RayTracing.h"

#include <cmath>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <locale>

namespace Arvo::Distributed::Protocol
{
// Version 10 requires complete Markov-chain settings. Crop retries must replay
// the same full-image bootstrap and chains instead of resetting local seeds.
// Version 12 restores DXR offline jobs using the same HLSL as realtime preview.
// Version 11 workers excluded DXR and must not serve this capability contract.
// Version 13 requires the global photon budget, radius, seed and path classes.
// Crops must replay the same light population, including missed emission paths.
// Version 14 requires progressive per-pass budgets, seeds, initial radius, alpha
// and path classes. Tile retries replay all passes, retaining local pixel state.
// Version 15 requires the global VCM light population, seed and radius schedule.
// Crops replay every light path/pass and retain full-image light-tracing density.
// Version 16 requires Point GI area, surface budget, bake samples and seed.
// Every crop rebuilds the same global cache; pixel samples use absolute coordinates.
// Version 17 requires spatial-reconstruction controls. Workers render the halo
// in full-film coordinates and return only the requested core RGBA rectangle.
// Version 18 carries the observation seed. Every tile and retry must replay the
// same camera, light-population and blue-noise frame domains.
// Version 19 negotiates display versus lossless film/geometry tile payloads.
// Raw tiles echo the complete request identity before any assembly is accepted.
// Version 20 requires denoiser-aware pilot allocation and variance-credit controls.
// Version 21 carries the complete homogeneous medium for transport and retry identity.
// Version 22 carries radiometry in requests and raw-film identity. Spectral
// camera observations retain signed linear RGB until display reconstruction.
// Version 23 carries both rigid camera keyframes and their shutter interval.
// Version 24 carries per-instance rigid tracks, including scene/cache identity.
// Version 25 carries optional normal/plane/material guidance for spatial filters.
// Guides use the expanded full-film raster, including both adaptive pilot halos.
inline constexpr const char *Version = "25";
std::string EncodeInstanceMotions(std::vector<InstanceMotionOverride>);
std::vector<InstanceMotionOverride> ParseInstanceMotions(const std::string&);
void ApplyInstanceMotions(RayTracing::Scene&, const std::vector<InstanceMotionOverride>&);
inline std::string Number(double value)
{
    std::ostringstream text;
    text.imbue(std::locale::classic());
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
        throw std::runtime_error("Invalid finite floating-point field: " + name);
    return value;
}

inline constexpr size_t MaximumMaterialOverrides = 4096;
inline std::string VectorText(RayTracing::Float3 value)
{
    return Number(value.X) + ',' + Number(value.Y) + ',' + Number(value.Z);
}
inline RayTracing::Float3 Vector(const std::string& text)
{
    const auto first = text.find(',');
    const auto second = first == std::string::npos ? std::string::npos : text.find(',', first + 1);
    if (second == std::string::npos || text.find(',', second + 1) != std::string::npos)
        throw std::runtime_error("Expected three comma-separated medium components");
    const std::map<std::string, std::string> parts{{"x", text.substr(0, first)},
        {"y", text.substr(first + 1, second - first - 1)}, {"z", text.substr(second + 1)}};
    return {Float(parts, "x"), Float(parts, "y"), Float(parts, "z")};
}
inline std::string EncodeCameraExposure(const RayTracing::MotionData::Exposure& exposure)
{
    std::string error;
    if (!RayTracing::ValidateCameraMotion(exposure, error))
        throw std::runtime_error(error);
    std::string text = std::to_string(exposure.Enabled) + ',' + Number(exposure.Interval.Open) + ',' +
                       Number(exposure.Interval.Close);
    for (const auto& pose : {exposure.Motion.Start, exposure.Motion.End})
        text += ',' + Number(pose.Rotation.X) + ',' + Number(pose.Rotation.Y) + ',' +
                Number(pose.Rotation.Z) + ',' + Number(pose.Rotation.W) + ',' + VectorText(pose.Translation);
    return text;
}
inline RayTracing::MotionData::Exposure ParseCameraExposure(const std::string& text)
{
    constexpr size_t FieldCount = 17;
    std::map<std::string, std::string> fields;
    size_t begin = 0;
    for (size_t i = 0; i < FieldCount; ++i)
    {
        const size_t end = text.find(',', begin);
        if ((i + 1 < FieldCount) == (end == std::string::npos))
            throw std::runtime_error("Incomplete camera exposure record");
        fields[std::to_string(i)] = text.substr(begin, end == std::string::npos ? end : end - begin);
        begin = end == std::string::npos ? text.size() : end + 1;
    }
    RayTracing::MotionData::Exposure result;
    result.Enabled = UInt(fields, "0", 0, 1);
    result.Interval = {Float(fields, "1"), Float(fields, "2")};
    size_t cursor = 3;
    const auto next = [&]() { return Float(fields, std::to_string(cursor++)); };
    for (auto* pose : {&result.Motion.Start, &result.Motion.End})
    {
        pose->Rotation = {next(), next(), next(), next()};
        pose->Translation = {next(), next(), next()};
    }
    std::string error;
    if (!RayTracing::ValidateCameraMotion(result, error))
        throw std::runtime_error(error);
    return result;
}
inline std::string EncodeMaterials(std::vector<MaterialOverride> materials)
{
    if (materials.size() > MaximumMaterialOverrides) throw std::runtime_error("Too many material overrides");
    std::sort(materials.begin(), materials.end(), [](const auto& a, const auto& b) { return a.Index < b.Index; });
    std::string result;
    uint32_t previous = 0;
    bool first = true;
    for (const auto& entry : materials)
    {
        if (!first && entry.Index == previous) throw std::runtime_error("Duplicate material override");
        std::string error;
        if (!RayTracing::ValidateMaterialScattering(entry.Scattering, error)) throw std::runtime_error(error);
        if (!first) result += ';';
        first = false;
        previous = entry.Index;
        const auto& value = entry.Scattering;
        result += std::to_string(entry.Index) + ',' + RayTracing::MaterialModelKey(value.Model);
        for (float scalar : {value.MicrofacetSlope, value.ExteriorIndex, value.InteriorIndex,
            value.ConductorIndex.X, value.ConductorIndex.Y, value.ConductorIndex.Z,
            value.ConductorExtinction.X, value.ConductorExtinction.Y, value.ConductorExtinction.Z})
            result += ',' + Number(scalar);
        // Optional trailing fields preserve the legacy material form. Older
        // workers reject these extra columns rather than silently dropping SSS.
        if (value.SubsurfaceMeanFreePath != 0 || value.SubsurfaceAlbedoScale != 1 || value.SubsurfaceAnisotropy != 0)
            for (float scalar : {value.SubsurfaceMeanFreePath, value.SubsurfaceAlbedoScale, value.SubsurfaceAnisotropy})
                result += ',' + Number(scalar);
    }
    return result;
}
inline std::vector<MaterialOverride> ParseMaterials(const std::string& text)
{
    // Rows are semicolon-separated. Decimal floats round-trip exactly; no scene
    // assets, resource names or native struct bytes travel in this field.
    constexpr const char* columns[] = {"index", "model", "slope", "exterior", "interior",
        "eta_r", "eta_g", "eta_b", "k_r", "k_g", "k_b"};
    std::vector<MaterialOverride> result;
    if (text.empty()) return result;
    if (text.back() == ';') throw std::runtime_error("Empty material override row");
    std::istringstream rows(text);
    std::string row;
    while (std::getline(rows, row, ';'))
    {
        if (result.size() == MaximumMaterialOverrides) throw std::runtime_error("Too many material overrides");
        if (row.empty() || row.back() == ',') throw std::runtime_error("Malformed material override row");
        std::istringstream values(row);
        std::map<std::string, std::string> fields;
        std::string value;
        for (const auto* column : columns)
        {
            if (!std::getline(values, value, ',') || value.empty()) throw std::runtime_error("Missing material field");
            fields.emplace(column, value);
        }
        const bool subsurface = bool(std::getline(values, value, ','));
        if (subsurface)
        {
            if (value.empty()) throw std::runtime_error("Missing subsurface mean free path");
            fields.emplace("sss_distance", value);
            for (const auto* column : {"sss_albedo", "sss_g"})
            {
                if (!std::getline(values, value, ',') || value.empty()) throw std::runtime_error("Missing subsurface material field");
                fields.emplace(column, value);
            }
            if (std::getline(values, value, ',')) throw std::runtime_error("Unexpected material field");
        }
        MaterialOverride entry;
        entry.Index = UInt(fields, "index", 0, UINT32_MAX);
        auto& scattering = entry.Scattering;
        if (!RayTracing::ParseMaterialModel(fields.at("model"), scattering.Model))
            throw std::runtime_error("Unknown material model");
        scattering.MicrofacetSlope = Float(fields, "slope");
        scattering.ExteriorIndex = Float(fields, "exterior");
        scattering.InteriorIndex = Float(fields, "interior");
        scattering.ConductorIndex = {Float(fields, "eta_r"), Float(fields, "eta_g"), Float(fields, "eta_b")};
        scattering.ConductorExtinction = {Float(fields, "k_r"), Float(fields, "k_g"), Float(fields, "k_b")};
        if (subsurface)
        {
            scattering.SubsurfaceMeanFreePath = Float(fields, "sss_distance");
            scattering.SubsurfaceAlbedoScale = Float(fields, "sss_albedo");
            scattering.SubsurfaceAnisotropy = Float(fields, "sss_g");
        }
        result.push_back(entry);
    }
    (void)EncodeMaterials(result); // Validate ranges and duplicate indices before accepting any record.
    return result;
}
inline void ApplyMaterials(RayTracing::Scene& scene, const std::vector<MaterialOverride>& materials)
{
    (void)EncodeMaterials(materials);
    for (const auto& entry : materials)
        if (entry.Index >= scene.Materials.size()) throw std::runtime_error("Material override index exceeds scene");
    for (const auto& entry : materials)
        static_cast<RayTracing::MaterialScattering&>(scene.Materials[entry.Index]) = entry.Scattering;
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
    constexpr char Schema[] = "arvo-render-scene-v4";
    bytes(Schema, sizeof(Schema));
    integer(scene.Materials.size(), 8);
    for (const auto& material : scene.Materials)
    {
        vector(material.Albedo); vector(material.AmbientColor);
        vector(material.SpecularColor); vector(material.Emission);
        scalar(material.Shininess); scalar(material.Reflectivity);
        integer(material.DiffuseTextureIndex, 4);
        integer(material.SampledEmitter ? 1 : 0, 1);
        integer(static_cast<uint32_t>(material.Model), 4);
        scalar(material.MicrofacetSlope);
        scalar(material.ExteriorIndex); scalar(material.InteriorIndex);
        vector(material.ConductorIndex); vector(material.ConductorExtinction);
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
    if (!scene.Geometries.empty() || !scene.Instances.empty())
    {
        // Preserve existing flat-scene hashes; extend the contract explicitly
        // for immutable object tables and instance transforms. Pointer addresses
        // never enter a portable scene identity.
        constexpr char InstanceSchema[] = "arvo-instance-graph-v1";
        bytes(InstanceSchema, sizeof(InstanceSchema));
        integer(scene.Geometries.size(), 8);
        for (const auto& geometry : scene.Geometries)
        {
            integer(geometry ? 1 : 0, 1);
            if (!geometry) continue;
            integer(geometry->Triangles.size(), 8);
            for (const auto& triangle : geometry->Triangles)
            {
                vector(triangle.Position0); vector(triangle.Position1); vector(triangle.Position2);
                vector(triangle.Normal0); vector(triangle.Normal1); vector(triangle.Normal2);
                vector(triangle.Color0); vector(triangle.Color1); vector(triangle.Color2);
                uv(triangle.TexCoord0); uv(triangle.TexCoord1); uv(triangle.TexCoord2);
                integer(triangle.MaterialIndex, 4);
                integer(triangle.SmoothNormals ? 1 : 0, 1);
            }
            integer(geometry->Spheres.size(), 8);
            for (const auto& sphere : geometry->Spheres)
            {
                vector(sphere.Center); scalar(sphere.Radius);
                integer(sphere.MaterialIndex, 4);
            }
        }
        integer(scene.Instances.size(), 8);
        for (const auto& instance : scene.Instances)
        {
            integer(instance.GeometryIndex, 4);
            for (float value : instance.ObjectToWorld) scalar(value);
        }
    }
    if (RayTracing::HasSceneMotion(scene))
    {
        constexpr char MotionSchema[] = "arvo-rigid-instance-motion-v1";
        for (char byte : MotionSchema) integer(static_cast<unsigned char>(byte), 1);
        for (const auto& instance : scene.Instances)
        {
            integer(instance.Motion.Enabled, 4);
            if (!instance.Motion.Enabled) continue;
            for (const auto& pose : {instance.Motion.Transform.Start, instance.Motion.Transform.End})
            {
                scalar(pose.Rotation.X); scalar(pose.Rotation.Y);
                scalar(pose.Rotation.Z); scalar(pose.Rotation.W);
                vector(pose.Translation);
            }
        }
    }
    const bool subsurface = std::any_of(scene.Materials.begin(), scene.Materials.end(),
        [](const auto& material) { return material.SubsurfaceMeanFreePath > 0; });
    if (subsurface)
    {
        constexpr char SubsurfaceSchema[] = "arvo-subsurface-materials-v1";
        for (char byte : SubsurfaceSchema) integer(static_cast<unsigned char>(byte), 1);
        for (const auto& material : scene.Materials)
        {
            scalar(material.SubsurfaceMeanFreePath);
            scalar(material.SubsurfaceAlbedoScale);
            scalar(material.SubsurfaceAnisotropy);
        }
    }
    return hash;
}

} // namespace Arvo::Distributed::Protocol
