#include "SpectralData.h"
// MSL independently evaluates the four reconstructed texels before filtering.
// This preserves the color integral and the RGB sampler's repeat/V-flip rules;
// nonlinear polynomial coefficients are never interpolated as if they were RGB.
float EvaluateSceneSpectrum(RgbSpectrum value, uint role, float wavelength)
{
    switch (role)
    {
    case SpectrumRoleIlluminant:
        return EvaluateRgbIlluminant(value, wavelength);
    case SpectrumRoleReflectance:
    case SpectrumRoleNonnegative:
        return EvaluateRgbReflectance(value, wavelength);
    default:
        return 0;
    }
}
float MetalSpectralRepeat(float value)
{
    if (!isfinite(value))
        return 0;
    return value - floor(value);
}
uint MetalSpectralLower(float coordinate, uint extent)
{
    if (coordinate < 0)
        return extent - 1;
    const uint integer = uint(coordinate);
    return integer < extent ? integer : 0;
}
float SampleSpectralTexture(SpectralTextureDescriptor texture, float2 uv, float wavelength,
                            device const uint* indices, device const RgbSpectrum* coefficients)
{
    if (!isfinite(wavelength) || wavelength < VisibleMinimumNm || wavelength > VisibleMaximumNm)
        return 0;
    if (texture.Width == 0 || texture.Height == 0)
        return 1;
    const float x = MetalSpectralRepeat(uv.x) * texture.Width - 0.5f;
    const float y = MetalSpectralRepeat(1 - uv.y) * texture.Height - 0.5f;
    const float fx = floor(x), fy = floor(y);
    const uint x0 = MetalSpectralLower(fx, texture.Width), y0 = MetalSpectralLower(fy, texture.Height);
    const uint x1 = x0 + 1 < texture.Width ? x0 + 1 : 0;
    const uint y1 = y0 + 1 < texture.Height ? y0 + 1 : 0;
    const uint top = texture.FirstTexel + y0 * texture.Width;
    const uint bottom = texture.FirstTexel + y1 * texture.Width;
    const float a = EvaluateRgbReflectance(coefficients[indices[top + x0]], wavelength);
    const float b = EvaluateRgbReflectance(coefficients[indices[top + x1]], wavelength);
    const float c = EvaluateRgbReflectance(coefficients[indices[bottom + x0]], wavelength);
    const float d = EvaluateRgbReflectance(coefficients[indices[bottom + x1]], wavelength);
    const float u = x - fx, v = y - fy;
    return (a * (1 - u) + b * u) * (1 - v) + (c * (1 - u) + d * u) * v;
}
float InterpolateSpectralVertex(RgbSpectrum a, RgbSpectrum b, RgbSpectrum c, float3 coordinates,
                                float wavelength)
{
    return dot(coordinates,
               float3(EvaluateRgbReflectance(a, wavelength), EvaluateRgbReflectance(b, wavelength),
                      EvaluateRgbReflectance(c, wavelength)));
}
