#include "Renderer/SpectralSceneData.h"
#include <cmath>
#include <map>
#include <stdexcept>
#include <tuple>
#include <unordered_map>

namespace Arvo::RayTracing::Cpu
{
namespace
{
namespace S = SpectralSceneData;
constexpr uint32_t ByteRange = 255, ByteBits = 8, PixelChannels = 4, CancellationBatch = 1024;
struct Cancelled
{
};
class InputBuilder
{
public:
    explicit InputBuilder(const S::Cancellation& cancellation) : Cancellation_(cancellation) {}
    S::Scene Data;
    void CheckCancellation() const
    {
        if (Cancellation_ && Cancellation_())
            throw Cancelled{};
    }
    uint32_t Color(Float3 color, S::Role role)
    {
        if (!std::isfinite(color.X) || !std::isfinite(color.Y) || !std::isfinite(color.Z) || color.X < 0 ||
            color.Y < 0 || color.Z < 0 ||
            (role == S::Role::Reflectance && (color.X > 1 || color.Y > 1 || color.Z > 1)))
            throw std::runtime_error("Spectral scene color is outside its physical input domain.");
        const auto key = std::make_tuple(role, color.X, color.Y, color.Z);
        const auto found = Colors_.find(key);
        if (found != Colors_.end())
            return found->second;
        if (Data.Requests.size() >= S::InvalidIndex)
            throw std::runtime_error("Spectral palette exceeds 32-bit indexing.");
        const uint32_t index = uint32_t(Data.Requests.size());
        Data.Requests.push_back({color, role});
        Colors_.emplace(key, index);
        return index;
    }
    void AddGeometry(const SceneGeometry& geometry)
    {
        if (geometry.Triangles.size() > S::InvalidIndex - Data.Triangles.size())
            throw std::runtime_error("Spectral geometry exceeds 32-bit indexing.");
        Data.Geometries.push_back({uint32_t(Data.Triangles.size()), uint32_t(geometry.Triangles.size())});
        for (const auto& triangle : geometry.Triangles)
        {
            CheckCancellation();
            Data.Triangles.push_back(
                {{Color(triangle.Color0, S::Role::Reflectance), Color(triangle.Color1, S::Role::Reflectance),
                  Color(triangle.Color2, S::Role::Reflectance)}});
        }
    }
    void AddTexture(const RayTexture& texture)
    {
        CheckCancellation();
        S::Texture descriptor;
        if (texture.IsValid())
        {
            const size_t texelCount = size_t(texture.Width) * texture.Height;
            if (texelCount > S::InvalidIndex - Data.Texels.size())
                throw std::runtime_error("Spectral texture atlas exceeds 32-bit indexing.");
            descriptor = {uint32_t(Data.Texels.size()), texture.Width, texture.Height, 0};
            const auto& bytes = *texture.PixelsRGBA8;
            for (size_t texel = 0; texel < texelCount; ++texel)
            {
                if (texel % CancellationBatch == 0)
                    CheckCancellation();
                const size_t first = texel * PixelChannels;
                const uint32_t key = bytes[first] | uint32_t(bytes[first + 1]) << ByteBits |
                                     uint32_t(bytes[first + 2]) << (ByteBits * 2);
                auto found = ByteColors_.find(key);
                if (found == ByteColors_.end())
                {
                    const Float3 linear{Decode(bytes[first]), Decode(bytes[first + 1]),
                                        Decode(bytes[first + 2])};
                    found = ByteColors_.emplace(key, Color(linear, S::Role::Reflectance)).first;
                }
                Data.Texels.push_back(found->second);
            }
        }
        Data.Textures.push_back(descriptor);
    }

private:
    static float Decode(uint8_t byte)
    {
        // IEC sRGB inverse transfer, before wavelength-space interpolation.
        constexpr float Threshold = 0.04045f, LinearSlope = 12.92f, Offset = 0.055f, Scale = 1.055f,
                        Exponent = 2.4f;
        const float value = float(byte) / ByteRange;
        return value <= Threshold ? value / LinearSlope : std::pow((value + Offset) / Scale, Exponent);
    }
    const S::Cancellation& Cancellation_;
    std::map<std::tuple<S::Role, float, float, float>, uint32_t> Colors_;
    std::unordered_map<uint32_t, uint32_t> ByteColors_;
};
} // namespace
RenderStatus BuildSpectralSceneInputs(const Scene& scene, const S::Cancellation& cancel, S::Scene& output,
                                      std::string& error)
{
    // The immutable palette distinguishes reflectance, emitted radiance and
    // nonnegative optical parameters. No spectrum is fitted at a ray hit.
    // Exact RGB8 deduplication ignores alpha, whose existing coverage stays RGB8.
    try
    {
        InputBuilder builder(cancel);
        builder.CheckCancellation();
        auto& data = builder.Data;
        data.WhiteReflectance = builder.Color({1, 1, 1}, S::Role::Reflectance);
        for (const auto& material : scene.Materials)
        {
            builder.CheckCancellation();
            data.Materials.push_back({builder.Color(material.Albedo, S::Role::Reflectance),
                                      builder.Color(material.AmbientColor, S::Role::Reflectance),
                                      builder.Color(material.SpecularColor, S::Role::Reflectance),
                                      builder.Color(material.Emission, S::Role::Illuminant),
                                      builder.Color(material.ConductorIndex, S::Role::Nonnegative),
                                      builder.Color(material.ConductorExtinction, S::Role::Nonnegative)});
        }
        builder.AddGeometry(scene);
        for (const auto& geometry : scene.Geometries)
        {
            if (!geometry)
                throw std::runtime_error("Spectral scene contains null shared geometry.");
            builder.AddGeometry(*geometry);
        }
        for (const auto& texture : scene.Textures)
            builder.AddTexture(texture);
        for (const auto& light : scene.DirectionalLights)
        {
            builder.CheckCancellation();
            data.DirectionalLights.push_back(builder.Color(light.Color, S::Role::Illuminant));
        }
        for (const auto& light : scene.PointLights)
        {
            builder.CheckCancellation();
            data.PointLights.push_back(builder.Color(light.Color, S::Role::Illuminant));
        }
        for (const auto& light : scene.RectAreaLights)
        {
            builder.CheckCancellation();
            data.RectAreaLights.push_back(builder.Color(light.Color, S::Role::Illuminant));
        }
        data.BackgroundTop = builder.Color(scene.BackgroundTop, S::Role::Illuminant);
        data.BackgroundBottom = builder.Color(scene.BackgroundBottom, S::Role::Illuminant);
        builder.CheckCancellation();
        output = std::move(data);
        error.clear();
        return RenderStatus::Completed;
    }
    catch (const Cancelled&)
    {
        error = "CPU spectral scene preparation cancelled.";
        return RenderStatus::Cancelled;
    }
    catch (const std::exception& failure)
    {
        error = failure.what();
        return RenderStatus::Failed;
    }
}
} // namespace Arvo::RayTracing::Cpu
