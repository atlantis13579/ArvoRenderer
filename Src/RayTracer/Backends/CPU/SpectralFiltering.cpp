#include "RayTracer/Backends/CPU/SpectralScene.h"
#include <cmath>

namespace Arvo::RayTracing::Cpu
{
namespace
{
bool Visible(float wavelength)
{
    return std::isfinite(wavelength) && wavelength >= VisibleMinimumNm && wavelength <= VisibleMaximumNm;
}
float RepeatSpectralCoordinate(float value)
{
    return std::isfinite(value) ? value - std::floor(value) : 0;
}
uint32_t WrapSpectralTexel(int64_t value, uint32_t extent)
{
    const auto remainder = value % int64_t(extent);
    return uint32_t(remainder < 0 ? remainder + extent : remainder);
}
} // namespace
float EvaluateSceneSpectrum(RgbSpectrum spectrum, SpectralSceneData::Role role, float wavelength)
{
    switch (role)
    {
    case SpectralSceneData::Role::Reflectance:
    case SpectralSceneData::Role::Nonnegative:
        return EvaluateRgbReflectance(spectrum, wavelength);
    case SpectralSceneData::Role::Illuminant:
        return EvaluateRgbIlluminant(spectrum, wavelength);
    }
    return 0;
}
float SampleSpectralTexture(const PreparedSpectralScene& scene, uint32_t index, Float2 uv, float wavelength)
{
    // Evaluate the nonlinear reconstruction before bilinear filtering. Blending
    // coefficients first would change the integrated color of texture gradients.
    // Repeat, half-texel centers and V flip match the existing RGB sampler.
    if (!Visible(wavelength))
        return 0;
    if (index >= scene.Inputs.Textures.size())
        return 1;
    const auto& texture = scene.Inputs.Textures[index];
    if (!texture.Width || !texture.Height)
        return 1;
    const float x = RepeatSpectralCoordinate(uv.X) * texture.Width - 0.5f;
    const float y = RepeatSpectralCoordinate(1 - uv.Y) * texture.Height - 0.5f;
    const auto ix = int64_t(std::floor(x)), iy = int64_t(std::floor(y));
    const float u = x - std::floor(x), v = y - std::floor(y);
    float result = 0;
    for (unsigned row = 0; row < 2; ++row)
        for (unsigned column = 0; column < 2; ++column)
        {
            const uint32_t tx = WrapSpectralTexel(ix + column, texture.Width);
            const uint32_t ty = WrapSpectralTexel(iy + row, texture.Height);
            const auto spectrum = scene.Inputs.Texels[texture.FirstTexel + size_t(ty) * texture.Width + tx];
            const float weight = (column ? u : 1 - u) * (row ? v : 1 - v);
            result += weight * EvaluateRgbReflectance(scene.Spectra[spectrum], wavelength);
        }
    return result;
}
float InterpolateSpectralVertex(RgbSpectrum a, RgbSpectrum b, RgbSpectrum c, Float3 weights, float wavelength)
{
    return weights.X * EvaluateRgbReflectance(a, wavelength) +
           weights.Y * EvaluateRgbReflectance(b, wavelength) +
           weights.Z * EvaluateRgbReflectance(c, wavelength);
}
} // namespace Arvo::RayTracing::Cpu
