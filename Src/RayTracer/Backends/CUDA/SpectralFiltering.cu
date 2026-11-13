#include "SpectralData.cuh"
// CUDA spectral interpolation operates on evaluated wavelengths. Polynomial
// interpolation would distort the RGB integral. Alpha continues to use the
// original coverage texture; this palette contains reflectance colors only.
__device__ float EvaluateSceneSpectrum(RgbSpectrum value, unsigned kind, float wavelength)
{
    if (kind == SpectrumRoleIlluminant)
        return EvaluateRgbIlluminant(value, wavelength);
    if (kind == SpectrumRoleReflectance || kind == SpectrumRoleNonnegative)
        return EvaluateRgbReflectance(value, wavelength);
    return 0;
}
__device__ float CudaSpectralRepeat(float value)
{
    return isfinite(value) ? value - floorf(value) : 0;
}
__device__ unsigned CudaSpectralWrap(long long value, unsigned extent)
{
    const long long remainder = value % extent;
    return unsigned(remainder < 0 ? remainder + extent : remainder);
}
__device__ float SampleSpectralTexture(SpectralTextureDescriptor texture, float2 uv, float wavelength,
                                       const unsigned* texels, const RgbSpectrum* spectra)
{
    if (!isfinite(wavelength) || wavelength < VisibleMinimumNm || wavelength > VisibleMaximumNm)
        return 0;
    if (!texture.Width || !texture.Height)
        return 1;
    const float x = CudaSpectralRepeat(uv.x) * texture.Width - 0.5f;
    const float y = CudaSpectralRepeat(1 - uv.y) * texture.Height - 0.5f;
    const float lowerX = floorf(x), lowerY = floorf(y);
    const float blendX = x - lowerX, blendY = y - lowerY;
    float filtered = 0;
    for (unsigned row = 0; row != 2; ++row)
    {
        const unsigned ty = CudaSpectralWrap(static_cast<long long>(lowerY) + row, texture.Height);
        const float verticalWeight = row == 0 ? 1 - blendY : blendY;
        for (unsigned column = 0; column != 2; ++column)
        {
            const unsigned tx = CudaSpectralWrap(static_cast<long long>(lowerX) + column, texture.Width);
            const float horizontalWeight = column == 0 ? 1 - blendX : blendX;
            const unsigned index = texels[texture.FirstTexel + ty * texture.Width + tx];
            filtered +=
                horizontalWeight * verticalWeight * EvaluateRgbReflectance(spectra[index], wavelength);
        }
    }
    return filtered;
}
__device__ float InterpolateSpectralVertex(RgbSpectrum a, RgbSpectrum b, RgbSpectrum c, float3 coordinates,
                                           float wavelength)
{
    return coordinates.x * EvaluateRgbReflectance(a, wavelength) +
           coordinates.y * EvaluateRgbReflectance(b, wavelength) +
           coordinates.z * EvaluateRgbReflectance(c, wavelength);
}
