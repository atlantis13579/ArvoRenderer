#include "Renderer/SpectralSceneData.h"
#include <cmath>
#include <map>
#include <stdexcept>
#include <tuple>
#include <unordered_map>

namespace Arvo::RayTracing::MetalSpectralHost
{
namespace
{
namespace S = SpectralSceneData;
constexpr unsigned EncodedMaximum = 255, ChannelShift = 8, PixelComponents = 4, PollingBlock = 1024;
struct Cancelled
{
};
class Palette
{
public:
    Palette(const S::Cancellation& cancellation) : Cancellation_(cancellation) {}
    S::Scene Result;
    void Poll() const
    {
        if (Cancellation_ && Cancellation_())
            throw Cancelled{};
    }
    uint32_t Add(Float3 rgb, S::Role role)
    {
        for (float component : {rgb.X, rgb.Y, rgb.Z})
            if (!std::isfinite(component) || component < 0 || (role == S::Role::Reflectance && component > 1))
                throw std::runtime_error("Metal spectral color is outside its physical domain.");
        const auto key = std::make_tuple(uint32_t(role), rgb.X, rgb.Y, rgb.Z);
        auto found = ColorIndex_.find(key);
        if (found != ColorIndex_.end())
            return found->second;
        if (Result.Requests.size() >= S::InvalidIndex)
            throw std::runtime_error("Metal spectral palette index overflow.");
        const uint32_t index = uint32_t(Result.Requests.size());
        Result.Requests.push_back({rgb, role});
        ColorIndex_.emplace(key, index);
        return index;
    }
    void Geometry(const SceneGeometry& source)
    {
        if (source.Triangles.size() > S::InvalidIndex - Result.Triangles.size())
            throw std::runtime_error("Metal spectral triangle index overflow.");
        Result.Geometries.push_back({uint32_t(Result.Triangles.size()), uint32_t(source.Triangles.size())});
        for (const auto& triangle : source.Triangles)
        {
            Poll();
            Result.Triangles.push_back(
                {{Add(triangle.Color0, S::Role::Reflectance), Add(triangle.Color1, S::Role::Reflectance),
                  Add(triangle.Color2, S::Role::Reflectance)}});
        }
    }
    void Texture(const RayTexture& texture)
    {
        Poll();
        S::Texture descriptor;
        if (texture.IsValid())
        {
            const size_t count = size_t(texture.Width) * texture.Height;
            if (count > S::InvalidIndex - Result.Texels.size())
                throw std::runtime_error("Metal spectral texture index overflow.");
            descriptor = {uint32_t(Result.Texels.size()), texture.Width, texture.Height, 0};
            for (size_t pixel = 0; pixel < count; pixel++)
            {
                if (pixel % PollingBlock == 0)
                    Poll();
                const auto* texel = texture.PixelsRGBA8->data() + pixel * PixelComponents;
                const uint32_t rgb = uint32_t(texel[0]) | uint32_t(texel[1]) << ChannelShift |
                                     uint32_t(texel[2]) << (ChannelShift * 2);
                auto previous = ByteIndex_.find(rgb);
                if (previous == ByteIndex_.end())
                {
                    const Float3 decoded{Decode(texel[0]), Decode(texel[1]), Decode(texel[2])};
                    previous = ByteIndex_.emplace(rgb, Add(decoded, S::Role::Reflectance)).first;
                }
                Result.Texels.push_back(previous->second);
            }
        }
        Result.Textures.push_back(descriptor);
    }
    template <class T> void Lights(const std::vector<T>& source, std::vector<uint32_t>& indices)
    {
        for (const auto& light : source)
        {
            Poll();
            indices.push_back(Add(light.Color, S::Role::Illuminant));
        }
    }

private:
    static float Decode(uint8_t channel)
    {
        constexpr float Cutoff = 0.04045f, LowDivisor = 12.92f, Bias = 0.055f, HighDivisor = 1.055f,
                        Exponent = 2.4f;
        const float value = float(channel) / EncodedMaximum;
        if (value <= Cutoff)
            return value / LowDivisor;
        return std::pow((value + Bias) / HighDivisor, Exponent);
    }
    const S::Cancellation& Cancellation_;
    std::map<std::tuple<uint32_t, float, float, float>, uint32_t> ColorIndex_;
    std::unordered_map<uint32_t, uint32_t> ByteIndex_;
};
} // namespace
RenderStatus BuildSceneInputs(const RayTracing::Scene& scene, const S::Cancellation& cancel, S::Scene& output,
                              std::string& error)
{
    // Backend-owned immutable input packing. Metal fits these records on-device;
    // the host neither reuses CPU coefficients nor interprets optical constants
    // as illuminants. Instanced geometry keeps one source-color range per asset.
    try
    {
        Palette palette(cancel);
        palette.Poll();
        auto& data = palette.Result;
        data.WhiteReflectance = palette.Add({1, 1, 1}, S::Role::Reflectance);
        for (const auto& material : scene.Materials)
        {
            palette.Poll();
            S::Material fields;
            fields.Albedo = palette.Add(material.Albedo, S::Role::Reflectance);
            fields.Ambient = palette.Add(material.AmbientColor, S::Role::Reflectance);
            fields.Specular = palette.Add(material.SpecularColor, S::Role::Reflectance);
            fields.Emission = palette.Add(material.Emission, S::Role::Illuminant);
            fields.ConductorIndex = palette.Add(material.ConductorIndex, S::Role::Nonnegative);
            fields.ConductorExtinction = palette.Add(material.ConductorExtinction, S::Role::Nonnegative);
            data.Materials.push_back(fields);
        }
        palette.Geometry(scene);
        for (const auto& geometry : scene.Geometries)
        {
            if (!geometry)
                throw std::runtime_error("Metal spectral geometry asset is null.");
            palette.Geometry(*geometry);
        }
        for (const auto& texture : scene.Textures)
            palette.Texture(texture);
        palette.Lights(scene.DirectionalLights, data.DirectionalLights);
        palette.Lights(scene.PointLights, data.PointLights);
        palette.Lights(scene.RectAreaLights, data.RectAreaLights);
        data.BackgroundTop = palette.Add(scene.BackgroundTop, S::Role::Illuminant);
        data.BackgroundBottom = palette.Add(scene.BackgroundBottom, S::Role::Illuminant);
        palette.Poll();
        output = std::move(data);
        error.clear();
        return RenderStatus::Completed;
    }
    catch (const Cancelled&)
    {
        error = "Metal spectral scene preparation cancelled.";
        return RenderStatus::Cancelled;
    }
    catch (const std::exception& failure)
    {
        error = failure.what();
        return RenderStatus::Failed;
    }
}
} // namespace Arvo::RayTracing::MetalSpectralHost
