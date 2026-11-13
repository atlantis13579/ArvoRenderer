#include "RayTracer/Backends/CPU/CpuTransport.h"
#include "Maths/ColorSpace.h"
#include <algorithm>
#include <cmath>

namespace Arvo::RayTracing::Cpu
{
// Texture filtering is performed in linear light; environment colors are linear.
float RepeatCoordinate(float value) noexcept
{
    if (!std::isfinite(value))
    {
        return 0.0f;
    }
    return value - std::floor(value);
}

uint32_t WrapTexel(int64_t coordinate, uint32_t size) noexcept
{
    const int64_t signedSize = static_cast<int64_t>(size);
    int64_t wrapped = coordinate % signedSize;
    if (wrapped < 0)
    {
        wrapped += signedSize;
    }
    return static_cast<uint32_t>(wrapped);
}

Float3 SampleDiffuseTexture(
    const Scene& scene,
    uint32_t textureIndex,
    const Float2& texCoord) noexcept
{
    if (textureIndex >= scene.Textures.size() ||
        !scene.Textures[textureIndex].IsValid())
    {
        return Float3(1.0f, 1.0f, 1.0f);
    }

    const RayTexture& texture = scene.Textures[textureIndex];
    const float x = RepeatCoordinate(texCoord.X) *
                        static_cast<float>(texture.Width) -
                    0.5f;
    const float y = RepeatCoordinate(1.0f - texCoord.Y) *
                        static_cast<float>(texture.Height) -
                    0.5f;
    const float floorX = std::floor(x);
    const float floorY = std::floor(y);
    const int64_t x0 = static_cast<int64_t>(floorX);
    const int64_t y0 = static_cast<int64_t>(floorY);
    const uint32_t sampleX[2] = {
        WrapTexel(x0, texture.Width),
        WrapTexel(x0 + 1, texture.Width)};
    const uint32_t sampleY[2] = {
        WrapTexel(y0, texture.Height),
        WrapTexel(y0 + 1, texture.Height)};
    const float blendX = x - floorX;
    const float blendY = y - floorY;
    const std::vector<uint8_t>& pixels = *texture.PixelsRGBA8;
    const auto load = [&](uint32_t pixelX, uint32_t pixelY)
    {
        const size_t byte =
            (static_cast<size_t>(pixelY) * texture.Width + pixelX) * 4U;
        return Float3(
            Maths::SRGBToLinear(
                static_cast<float>(pixels[byte]) / 255.0f),
            Maths::SRGBToLinear(
                static_cast<float>(pixels[byte + 1U]) / 255.0f),
            Maths::SRGBToLinear(
                static_cast<float>(pixels[byte + 2U]) / 255.0f));
    };
    const auto lerp = [](
        const Float3& left,
        const Float3& right,
        float amount)
    {
        return left * (1.0f - amount) + right * amount;
    };
    const Float3 top = lerp(
        load(sampleX[0], sampleY[0]),
        load(sampleX[1], sampleY[0]),
        blendX);
    const Float3 bottom = lerp(
        load(sampleX[0], sampleY[1]),
        load(sampleX[1], sampleY[1]),
        blendX);
    return lerp(top, bottom, blendY);
}

Float3 Background(const Scene& scene, const Ray& ray)
{
    const float blend = std::max(0.0f, std::min(1.0f, ray.Direction.Y * 0.5f + 0.5f));
    return scene.BackgroundBottom * (1.0f - blend) + scene.BackgroundTop * blend;
}
}
