#include "Renderer/SpectralSceneData.h"
#include <array>
#include <cmath>
#include <cstring>
#include <map>
#include <stdexcept>
#include <unordered_map>

namespace Arvo::RayTracing::CudaSpectralHost
{
namespace
{
namespace S = SpectralSceneData;
constexpr uint32_t ByteMask = 255, ByteShift = 8, TexelStride = 4, CancellationBlock = 1024;
struct Cancelled
{
};
struct SceneCollector
{
    const S::Cancellation& Cancel;
    S::Scene Data;
    std::map<std::array<uint32_t, 4>, uint32_t> Requests;
    std::unordered_map<uint32_t, uint32_t> ByteRequests;
    void Poll() const
    {
        if (Cancel && Cancel())
            throw Cancelled{};
    }
    uint32_t Intern(Float3 color, S::Role role)
    {
        std::array<uint32_t, 4> key{uint32_t(role), 0, 0, 0};
        const float values[]{color.X, color.Y, color.Z};
        for (unsigned axis = 0; axis < 3; ++axis)
        {
            if (!std::isfinite(values[axis]) || values[axis] < 0 ||
                (role == S::Role::Reflectance && values[axis] > 1))
                throw std::runtime_error("CUDA spectral input has an invalid physical color.");
            const float canonical = values[axis] == 0 ? 0 : values[axis];
            std::memcpy(&key[axis + 1], &canonical, sizeof(canonical));
        }
        const auto entry = Requests.find(key);
        if (entry != Requests.end())
            return entry->second;
        if (Data.Requests.size() >= S::InvalidIndex)
            throw std::runtime_error("CUDA spectral palette is too large.");
        const uint32_t address = uint32_t(Data.Requests.size());
        Data.Requests.push_back({color, role});
        Requests.emplace(key, address);
        return address;
    }
    void GeometryColors(const SceneGeometry& geometry)
    {
        if (geometry.Triangles.size() > S::InvalidIndex - Data.Triangles.size())
            throw std::runtime_error("CUDA spectral triangle indices overflow.");
        Data.Geometries.push_back({uint32_t(Data.Triangles.size()), uint32_t(geometry.Triangles.size())});
        for (const auto& triangle : geometry.Triangles)
        {
            Poll();
            S::Triangle record;
            record.Color[0] = Intern(triangle.Color0, S::Role::Reflectance);
            record.Color[1] = Intern(triangle.Color1, S::Role::Reflectance);
            record.Color[2] = Intern(triangle.Color2, S::Role::Reflectance);
            Data.Triangles.push_back(record);
        }
    }
    uint32_t EncodedColor(const uint8_t* bytes)
    {
        uint32_t key = 0;
        for (unsigned channel = 0; channel < 3; ++channel)
            key |= uint32_t(bytes[channel]) << (ByteShift * channel);
        const auto previous = ByteRequests.find(key);
        if (previous != ByteRequests.end())
            return previous->second;
        constexpr float LinearThreshold = 0.04045f, LinearDenominator = 12.92f, CurveOffset = 0.055f,
                        CurveDenominator = 1.055f, CurvePower = 2.4f;
        float linear[3];
        for (unsigned channel = 0; channel < 3; ++channel)
        {
            const float value = float((key >> (channel * ByteShift)) & ByteMask) / ByteMask;
            linear[channel] = value <= LinearThreshold
                                  ? value / LinearDenominator
                                  : std::pow((value + CurveOffset) / CurveDenominator, CurvePower);
        }
        const uint32_t index = Intern({linear[0], linear[1], linear[2]}, S::Role::Reflectance);
        ByteRequests.emplace(key, index);
        return index;
    }
};
} // namespace
RenderStatus BuildSceneInputs(const RayTracing::Scene& scene, const S::Cancellation& cancel, S::Scene& output,
                              std::string& error)
{
    // CUDA owns the host index construction and the later device fit. Only
    // records are shared with other backends. RGB8 deduplication avoids fitting
    // repeated texels; conductor parameters remain nonnegative spectra, not lights.
    try
    {
        SceneCollector collector{cancel, {}, {}, {}};
        collector.Poll();
        auto& data = collector.Data;
        data.WhiteReflectance = collector.Intern({1, 1, 1}, S::Role::Reflectance);
        for (const auto& material : scene.Materials)
        {
            collector.Poll();
            data.Materials.push_back({collector.Intern(material.Albedo, S::Role::Reflectance),
                                      collector.Intern(material.AmbientColor, S::Role::Reflectance),
                                      collector.Intern(material.SpecularColor, S::Role::Reflectance),
                                      collector.Intern(material.Emission, S::Role::Illuminant),
                                      collector.Intern(material.ConductorIndex, S::Role::Nonnegative),
                                      collector.Intern(material.ConductorExtinction, S::Role::Nonnegative)});
        }
        collector.GeometryColors(scene);
        for (const auto& geometry : scene.Geometries)
        {
            if (!geometry)
                throw std::runtime_error("CUDA spectral geometry reference is null.");
            collector.GeometryColors(*geometry);
        }
        for (const auto& texture : scene.Textures)
        {
            collector.Poll();
            if (!texture.IsValid())
            {
                data.Textures.push_back({});
                continue;
            }
            const size_t count = size_t(texture.Width) * texture.Height;
            if (count > S::InvalidIndex - data.Texels.size())
                throw std::runtime_error("CUDA spectral texel indices overflow.");
            data.Textures.push_back({uint32_t(data.Texels.size()), texture.Width, texture.Height, 0});
            for (size_t pixel = 0; pixel < count; ++pixel)
            {
                if (pixel % CancellationBlock == 0)
                    collector.Poll();
                data.Texels.push_back(
                    collector.EncodedColor(texture.PixelsRGBA8->data() + pixel * TexelStride));
            }
        }
        for (const auto& light : scene.DirectionalLights)
        {
            collector.Poll();
            data.DirectionalLights.push_back(collector.Intern(light.Color, S::Role::Illuminant));
        }
        for (const auto& light : scene.PointLights)
        {
            collector.Poll();
            data.PointLights.push_back(collector.Intern(light.Color, S::Role::Illuminant));
        }
        for (const auto& light : scene.RectAreaLights)
        {
            collector.Poll();
            data.RectAreaLights.push_back(collector.Intern(light.Color, S::Role::Illuminant));
        }
        data.BackgroundTop = collector.Intern(scene.BackgroundTop, S::Role::Illuminant);
        data.BackgroundBottom = collector.Intern(scene.BackgroundBottom, S::Role::Illuminant);
        collector.Poll();
        output = std::move(data);
        error.clear();
        return RenderStatus::Completed;
    }
    catch (const Cancelled&)
    {
        error = "CUDA spectral scene preparation cancelled.";
        return RenderStatus::Cancelled;
    }
    catch (const std::exception& failure)
    {
        error = failure.what();
        return RenderStatus::Failed;
    }
}
} // namespace Arvo::RayTracing::CudaSpectralHost
