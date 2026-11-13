#include "SpectralColor.metal"
#include "RgbSpectrum.metal"
#include "SpectralFiltering.metal"
struct SpectralFilterQuery
{
    SpectralTextureDescriptor Texture;
    float4 Coordinates;
    uint4 Indices;
    float4 Barycentric;
};
kernel void SpectralPrepareResponse(device float4* response [[buffer(0)]],
                                    uint index [[thread_position_in_grid]])
{
    if (index < RgbFitSampleCount)
        response[index] = MakeRgbFitResponse(index);
}
kernel void SpectralPreparePalette(device const SpectralSource* source [[buffer(0)]],
                                   device const float4* response [[buffer(1)]],
                                   device RgbSpectrum* output [[buffer(2)]],
                                   constant uint& count [[buffer(3)]], uint index [[thread_position_in_grid]])
{
    if (index < count)
        output[index] = FitSceneSpectrum(source[index], response);
}
kernel void SpectralFilterProbe(device const SpectralFilterQuery* queries [[buffer(0)]],
                                device const uint* texels [[buffer(1)]],
                                device const RgbSpectrum* spectra [[buffer(2)]],
                                device const SpectralSource* source [[buffer(3)]],
                                device float4* output [[buffer(4)]], constant uint& count [[buffer(5)]],
                                uint index [[thread_position_in_grid]])
{
    if (index >= count)
        return;
    const SpectralFilterQuery query = queries[index];
    output[index] = float4(
        SampleSpectralTexture(query.Texture, query.Coordinates.xy, query.Coordinates.z, texels, spectra),
        InterpolateSpectralVertex(spectra[query.Indices.x], spectra[query.Indices.y],
                                  spectra[query.Indices.z], query.Barycentric.xyz, query.Coordinates.z),
        EvaluateSceneSpectrum(spectra[query.Indices.w], source[query.Indices.w].Kind, query.Coordinates.z),
        0);
}
