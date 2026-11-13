#include "SpectralData.hlsli"
// Evaluate before filtering: the sigmoid reconstruction is nonlinear in its
// coefficients. Wavelength-space bilinear interpolation preserves RGB integrals.
// Repeat addressing, half-texel centers and V flip match the RGB compute sampler.
float EvaluateSceneSpectrum(RgbSpectrum spectrum, uint kind, float wavelength)
{
    if (kind == SpectrumRoleIlluminant)
        return EvaluateRgbIlluminant(spectrum, wavelength);
    if (kind == SpectrumRoleReflectance || kind == SpectrumRoleNonnegative)
        return EvaluateRgbReflectance(spectrum, wavelength);
    return 0;
}
float ComputeSpectralRepeat(float coordinate)
{
    return isfinite(coordinate) ? coordinate - floor(coordinate) : 0;
}
uint ComputeSpectralFirst(float lower, uint extent)
{
    if (lower < 0)
        return extent - 1;
    uint index = uint(lower);
    return index < extent ? index : 0;
}
float SampleSpectralTexture(SpectralTextureDescriptor descriptor, float2 uv, float wavelength,
                            StructuredBuffer<uint> texels, StructuredBuffer<RgbSpectrum> spectra)
{
    if (!isfinite(wavelength) || wavelength < VisibleMinimumNm || wavelength > VisibleMaximumNm)
        return 0;
    if (descriptor.Width == 0 || descriptor.Height == 0)
        return 1;
    float x = ComputeSpectralRepeat(uv.x) * descriptor.Width - 0.5f;
    float y = ComputeSpectralRepeat(1 - uv.y) * descriptor.Height - 0.5f;
    float fx = floor(x), fy = floor(y);
    uint left = ComputeSpectralFirst(fx, descriptor.Width), top = ComputeSpectralFirst(fy, descriptor.Height);
    uint right = left + 1 < descriptor.Width ? left + 1 : 0;
    uint bottom = top + 1 < descriptor.Height ? top + 1 : 0;
    float sum = 0;
    for (uint row = 0; row < 2; ++row)
        for (uint column = 0; column < 2; ++column)
        {
            uint tx = column == 0 ? left : right, ty = row == 0 ? top : bottom;
            float weight = (column == 0 ? 1 - (x - fx) : x - fx) * (row == 0 ? 1 - (y - fy) : y - fy);
            uint index = texels[descriptor.FirstTexel + ty * descriptor.Width + tx];
            sum += weight * EvaluateRgbReflectance(spectra[index], wavelength);
        }
    return sum;
}
float InterpolateSpectralVertex(RgbSpectrum a, RgbSpectrum b, RgbSpectrum c, float3 weights, float wavelength)
{
    const float3 values = float3(EvaluateRgbReflectance(a, wavelength), EvaluateRgbReflectance(b, wavelength),
                                 EvaluateRgbReflectance(c, wavelength));
    return dot(weights, values);
}
