#include "Renderer/SpectralSceneData.h"
#include <cmath>
#include <map>
#include <stdexcept>
#include <tuple>
#include <unordered_map>

namespace Arvo::RayTracing::DxrSpectralHost
{
namespace
{
namespace S = SpectralSceneData;
constexpr uint32_t ByteMaximum = 255, ChannelBits = 8, PixelStride = 4, CancelInterval = 1024;
struct Cancelled
{
};
class PaletteBuilder
{
public:
    PaletteBuilder(const S::Cancellation& cancel) : Cancel_(cancel) {}
    S::Scene Data;
    void Poll() const
    {
        if (Cancel_ && Cancel_())
            throw Cancelled{};
    }
    uint32_t Add(S::Role role, Float3 value)
    {
        if (!std::isfinite(value.X) || !std::isfinite(value.Y) || !std::isfinite(value.Z) || value.X < 0 ||
            value.Y < 0 || value.Z < 0 ||
            (role == S::Role::Reflectance && (value.X > 1 || value.Y > 1 || value.Z > 1)))
            throw std::runtime_error("DXR spectral color is outside the supported domain.");
        const auto key = std::make_tuple(value.X, value.Y, value.Z, role);
        const auto previous = Entries_.find(key);
        if (previous != Entries_.end())
            return previous->second;
        if (Data.Requests.size() >= S::InvalidIndex)
            throw std::runtime_error("DXR spectral palette overflows its index.");
        const uint32_t slot = uint32_t(Data.Requests.size());
        Entries_.emplace(key, slot);
        Data.Requests.push_back({value, role});
        return slot;
    }
    void AddGeometry(const SceneGeometry& source)
    {
        if (source.Triangles.size() > S::InvalidIndex - Data.Triangles.size())
            throw std::runtime_error("DXR spectral geometry table overflows its index.");
        S::Geometry range{uint32_t(Data.Triangles.size()), uint32_t(source.Triangles.size())};
        for (const auto& primitive : source.Triangles)
        {
            Poll();
            Data.Triangles.push_back(
                {{Add(S::Role::Reflectance, primitive.Color0), Add(S::Role::Reflectance, primitive.Color1),
                  Add(S::Role::Reflectance, primitive.Color2)}});
        }
        Data.Geometries.push_back(range);
    }
    void AddTexture(const RayTexture& source)
    {
        Poll();
        if (!source.IsValid())
        {
            Data.Textures.push_back({});
            return;
        }
        const size_t count = size_t(source.Width) * source.Height;
        if (count > S::InvalidIndex - Data.Texels.size())
            throw std::runtime_error("DXR spectral texture table overflows its index.");
        Data.Textures.push_back({uint32_t(Data.Texels.size()), source.Width, source.Height, 0});
        const auto& bytes = *source.PixelsRGBA8;
        for (size_t pixel = 0; pixel < count; pixel++)
        {
            if (pixel % CancelInterval == 0)
                Poll();
            const size_t byte = pixel * PixelStride;
            const uint32_t rgb = uint32_t(bytes[byte]) | uint32_t(bytes[byte + 1]) << ChannelBits |
                                 uint32_t(bytes[byte + 2]) << (2 * ChannelBits);
            auto entry = Encoded_.find(rgb);
            if (entry == Encoded_.end())
            {
                Float3 linear{Decode(bytes[byte]), Decode(bytes[byte + 1]), Decode(bytes[byte + 2])};
                entry = Encoded_.emplace(rgb, Add(S::Role::Reflectance, linear)).first;
            }
            Data.Texels.push_back(entry->second);
        }
    }
    template <class Light> void Lights(const std::vector<Light>& source, std::vector<uint32_t>& destination)
    {
        for (const auto& light : source)
        {
            Poll();
            destination.push_back(Add(S::Role::Illuminant, light.Color));
        }
    }

private:
    static float Decode(uint8_t encoded)
    {
        constexpr float Threshold = 0.04045f, Slope = 12.92f, Offset = 0.055f, Gain = 1.055f, Power = 2.4f;
        const float normalized = float(encoded) / ByteMaximum;
        return normalized <= Threshold ? normalized / Slope : std::pow((normalized + Offset) / Gain, Power);
    }
    const S::Cancellation& Cancel_;
    std::map<std::tuple<float, float, float, S::Role>, uint32_t> Entries_;
    std::unordered_map<uint32_t, uint32_t> Encoded_;
};
} // namespace
RenderStatus BuildSceneInputs(const RayTracing::Scene& scene, const S::Cancellation& cancel, S::Scene& output,
                              std::string& error)
{
    // Spectral offline DXR consumes this palette contract. BLAS/TLAS
    // transforms never duplicate source color spectra. Vertex interpolation and
    // texture filtering must evaluate spectra first, not blend fit coefficients.
    try
    {
        PaletteBuilder builder(cancel);
        builder.Poll();
        auto& data = builder.Data;
        data.WhiteReflectance = builder.Add(S::Role::Reflectance, {1, 1, 1});
        for (const auto& source : scene.Materials)
        {
            builder.Poll();
            S::Material fields{builder.Add(S::Role::Reflectance, source.Albedo),
                               builder.Add(S::Role::Reflectance, source.AmbientColor),
                               builder.Add(S::Role::Reflectance, source.SpecularColor),
                               builder.Add(S::Role::Illuminant, source.Emission),
                               builder.Add(S::Role::Nonnegative, source.ConductorIndex),
                               builder.Add(S::Role::Nonnegative, source.ConductorExtinction)};
            data.Materials.push_back(fields);
        }
        builder.AddGeometry(scene);
        for (const auto& source : scene.Geometries)
        {
            if (!source)
                throw std::runtime_error("DXR spectral preparation received null geometry.");
            builder.AddGeometry(*source);
        }
        for (const auto& texture : scene.Textures)
            builder.AddTexture(texture);
        builder.Lights(scene.DirectionalLights, data.DirectionalLights);
        builder.Lights(scene.PointLights, data.PointLights);
        builder.Lights(scene.RectAreaLights, data.RectAreaLights);
        data.BackgroundTop = builder.Add(S::Role::Illuminant, scene.BackgroundTop);
        data.BackgroundBottom = builder.Add(S::Role::Illuminant, scene.BackgroundBottom);
        builder.Poll();
        output = std::move(data);
        error.clear();
        return RenderStatus::Completed;
    }
    catch (const Cancelled&)
    {
        error = "DXR spectral scene preparation cancelled.";
        return RenderStatus::Cancelled;
    }
    catch (const std::exception& failure)
    {
        error = failure.what();
        return RenderStatus::Failed;
    }
}
} // namespace Arvo::RayTracing::DxrSpectralHost
