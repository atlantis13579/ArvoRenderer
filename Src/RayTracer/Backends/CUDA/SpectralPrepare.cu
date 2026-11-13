#include "SpectralColor.cu"
#include "RgbSpectrum.cu"
#include "SpectralFiltering.cu"
struct SpectralFilterQuery
{
    SpectralTextureDescriptor Texture;
    float4 Coordinates;
    uint4 Indices;
    float4 Barycentric;
};
extern "C" __global__ void SpectralPrepareResponse(float4* response)
{
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < RgbFitSampleCount)
        response[index] = MakeRgbFitResponse(index);
}
extern "C" __global__ void SpectralPreparePalette(const SpectralSource* input, const float4* response,
                                                  RgbSpectrum* output, unsigned count)
{
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count)
        output[index] = FitSceneSpectrum(input[index], response);
}
extern "C" __global__ void SpectralFilterProbe(const SpectralFilterQuery* queries, const unsigned* texels,
                                               const RgbSpectrum* spectra, const SpectralSource* sources,
                                               float4* output, unsigned count)
{
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count)
        return;
    const auto query = queries[index];
    const float wavelength = query.Coordinates.z;
    output[index] = make_float4(
        SampleSpectralTexture(query.Texture, make_float2(query.Coordinates.x, query.Coordinates.y),
                              wavelength, texels, spectra),
        InterpolateSpectralVertex(
            spectra[query.Indices.x], spectra[query.Indices.y], spectra[query.Indices.z],
            make_float3(query.Barycentric.x, query.Barycentric.y, query.Barycentric.z), wavelength),
        EvaluateSceneSpectrum(spectra[query.Indices.w], sources[query.Indices.w].Kind, wavelength), 0);
}
