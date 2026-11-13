#include "Renderer/SpectralSceneData.h"
#include <array>
#include <cmath>
#include <map>
#include <stdexcept>
#include <unordered_map>

namespace Arvo::RayTracing::ComputeSpectralHost
{
namespace
{
namespace S = SpectralSceneData;
constexpr uint32_t ChannelsPerPixel = 4, BitsPerChannel = 8, ChannelMaximum = 255, CancellationStride = 1024;
struct Cancelled
{
};
struct Collector
{
    S::Scene Scene;
    const S::Cancellation& Cancellation;
    std::map<std::array<float, 4>, uint32_t> Palette;
    std::unordered_map<uint32_t, uint32_t> EncodedPalette;
    void Poll() const
    {
        if (Cancellation && Cancellation())
            throw Cancelled{};
    }
    uint32_t Append(Float3 rgb, S::Role kind)
    {
        const bool bounded = kind == S::Role::Reflectance;
        for (float channel : {rgb.X, rgb.Y, rgb.Z})
            if (!std::isfinite(channel) || channel < 0 || (bounded && channel > 1))
                throw std::runtime_error("Compute spectral input has an invalid color component.");
        const std::array<float, 4> key{float(kind), rgb.X, rgb.Y, rgb.Z};
        auto found = Palette.find(key);
        if (found != Palette.end())
            return found->second;
        if (Scene.Requests.size() >= S::InvalidIndex)
            throw std::runtime_error("Compute spectral palette is too large.");
        const auto index = uint32_t(Scene.Requests.size());
        Scene.Requests.push_back({rgb, kind});
        Palette.emplace(key, index);
        return index;
    }
    void Geometry(const SceneGeometry& geometry)
    {
        const size_t count = geometry.Triangles.size();
        if (count > S::InvalidIndex - Scene.Triangles.size())
            throw std::runtime_error("Compute spectral triangle table is too large.");
        Scene.Geometries.push_back({uint32_t(Scene.Triangles.size()), uint32_t(count)});
        for (const auto& triangle : geometry.Triangles)
        {
            Poll();
            S::Triangle colors;
            colors.Color[0] = Append(triangle.Color0, S::Role::Reflectance);
            colors.Color[1] = Append(triangle.Color1, S::Role::Reflectance);
            colors.Color[2] = Append(triangle.Color2, S::Role::Reflectance);
            Scene.Triangles.push_back(colors);
        }
    }
    uint32_t Texel(const uint8_t* rgba)
    {
        const uint32_t key = uint32_t(rgba[0]) + (uint32_t(rgba[1]) << BitsPerChannel) +
                             (uint32_t(rgba[2]) << (BitsPerChannel * 2));
        const auto found = EncodedPalette.find(key);
        if (found != EncodedPalette.end())
            return found->second;
        constexpr float LinearLimit = 0.04045f, InverseSlope = 12.92f, TransferOffset = 0.055f,
                        TransferScale = 1.055f, TransferPower = 2.4f;
        float values[3];
        for (unsigned channel = 0; channel < 3; ++channel)
        {
            const float encoded = float(rgba[channel]) / ChannelMaximum;
            values[channel] = encoded <= LinearLimit
                                  ? encoded / InverseSlope
                                  : std::pow((encoded + TransferOffset) / TransferScale, TransferPower);
        }
        const uint32_t index = Append({values[0], values[1], values[2]}, S::Role::Reflectance);
        EncodedPalette.emplace(key, index);
        return index;
    }
};
} // namespace
RenderStatus BuildSceneInputs(const RayTracing::Scene& scene, const S::Cancellation& cancel, S::Scene& output,
                              std::string& error)
{
    // This backend constructs its own upload palette. Texture alpha remains in
    // the original atlas; repeat/filter operations consume these spectral IDs.
    // Light intensity stays a scalar, separate from its illuminant color.
    try
    {
        Collector collector{{}, cancel, {}, {}};
        collector.Poll();
        auto& data = collector.Scene;
        data.WhiteReflectance = collector.Append({1, 1, 1}, S::Role::Reflectance);
        for (const auto& material : scene.Materials)
        {
            collector.Poll();
            S::Material fields;
            fields.Albedo = collector.Append(material.Albedo, S::Role::Reflectance);
            fields.Ambient = collector.Append(material.AmbientColor, S::Role::Reflectance);
            fields.Specular = collector.Append(material.SpecularColor, S::Role::Reflectance);
            fields.Emission = collector.Append(material.Emission, S::Role::Illuminant);
            fields.ConductorIndex = collector.Append(material.ConductorIndex, S::Role::Nonnegative);
            fields.ConductorExtinction = collector.Append(material.ConductorExtinction, S::Role::Nonnegative);
            data.Materials.push_back(fields);
        }
        collector.Geometry(scene);
        for (const auto& geometry : scene.Geometries)
        {
            if (!geometry)
                throw std::runtime_error("Compute spectral geometry is null.");
            collector.Geometry(*geometry);
        }
        for (const auto& texture : scene.Textures)
        {
            collector.Poll();
            S::Texture descriptor;
            if (texture.IsValid())
            {
                const size_t count = size_t(texture.Width) * texture.Height;
                if (count > S::InvalidIndex - data.Texels.size())
                    throw std::runtime_error("Compute spectral texel table is too large.");
                descriptor = {uint32_t(data.Texels.size()), texture.Width, texture.Height, 0};
                const auto& bytes = *texture.PixelsRGBA8;
                for (size_t index = 0; index < count; ++index)
                {
                    if (index % CancellationStride == 0)
                        collector.Poll();
                    data.Texels.push_back(collector.Texel(bytes.data() + index * ChannelsPerPixel));
                }
            }
            data.Textures.push_back(descriptor);
        }
        for (const auto& light : scene.DirectionalLights)
        {
            collector.Poll();
            data.DirectionalLights.push_back(collector.Append(light.Color, S::Role::Illuminant));
        }
        for (const auto& light : scene.PointLights)
        {
            collector.Poll();
            data.PointLights.push_back(collector.Append(light.Color, S::Role::Illuminant));
        }
        for (const auto& light : scene.RectAreaLights)
        {
            collector.Poll();
            data.RectAreaLights.push_back(collector.Append(light.Color, S::Role::Illuminant));
        }
        data.BackgroundTop = collector.Append(scene.BackgroundTop, S::Role::Illuminant);
        data.BackgroundBottom = collector.Append(scene.BackgroundBottom, S::Role::Illuminant);
        collector.Poll();
        output = std::move(data);
        error.clear();
        return RenderStatus::Completed;
    }
    catch (const Cancelled&)
    {
        error = "Compute spectral preparation cancelled.";
        return RenderStatus::Cancelled;
    }
    catch (const std::exception& failure)
    {
        error = failure.what();
        return RenderStatus::Failed;
    }
}
} // namespace Arvo::RayTracing::ComputeSpectralHost
