#include "SpectralData.hlsli"
// Independent spectral DXR filtering for offline jobs. Reconstruct each
// texel at the sampled wavelength before interpolation, never blend nonlinear
// coefficients. Invalid descriptors use the established white texture fallback.
float EvaluateSceneSpectrum(RgbSpectrum fit, uint role, float wavelength)
{
    if (role == SpectrumRoleReflectance || role == SpectrumRoleNonnegative)
        return EvaluateRgbReflectance(fit, wavelength);
    return role == SpectrumRoleIlluminant ? EvaluateRgbIlluminant(fit, wavelength) : 0;
}
float DxrSpectralRepeat(float value)
{
    if (!isfinite(value))
        return 0;
    return value - floor(value);
}
uint DxrSpectralWrapLower(float value, uint size)
{
    if (value < 0)
        return size - 1;
    const uint index = uint(value);
    return index >= size ? 0 : index;
}
float SampleSpectralTexture(SpectralTextureDescriptor descriptor, float2 uv, float wavelength,
                            StructuredBuffer<uint> indices, StructuredBuffer<RgbSpectrum> palette)
{
    if (!isfinite(wavelength) || wavelength < VisibleMinimumNm || wavelength > VisibleMaximumNm)
        return 0;
    if (!descriptor.Width || !descriptor.Height)
        return 1;
    const float horizontal = DxrSpectralRepeat(uv.x) * descriptor.Width - 0.5f;
    const float vertical = DxrSpectralRepeat(1 - uv.y) * descriptor.Height - 0.5f;
    const float u = horizontal - floor(horizontal), v = vertical - floor(vertical);
    const uint x0 = DxrSpectralWrapLower(floor(horizontal), descriptor.Width);
    const uint y0 = DxrSpectralWrapLower(floor(vertical), descriptor.Height);
    const uint x1 = x0 + 1 == descriptor.Width ? 0 : x0 + 1;
    const uint y1 = y0 + 1 == descriptor.Height ? 0 : y0 + 1;
    const uint upper = descriptor.FirstTexel + y0 * descriptor.Width;
    const uint lower = descriptor.FirstTexel + y1 * descriptor.Width;
    const float a = EvaluateRgbReflectance(palette[indices[upper + x0]], wavelength);
    const float b = EvaluateRgbReflectance(palette[indices[upper + x1]], wavelength);
    const float c = EvaluateRgbReflectance(palette[indices[lower + x0]], wavelength);
    const float d = EvaluateRgbReflectance(palette[indices[lower + x1]], wavelength);
    return ((1 - u) * a + u * b) * (1 - v) + ((1 - u) * c + u * d) * v;
}
float InterpolateSpectralVertex(RgbSpectrum a, RgbSpectrum b, RgbSpectrum c, float3 barycentric,
                                float wavelength)
{
    return EvaluateRgbReflectance(a, wavelength) * barycentric.x +
           EvaluateRgbReflectance(b, wavelength) * barycentric.y +
           EvaluateRgbReflectance(c, wavelength) * barycentric.z;
}
