#include "SpectralColor.hlsl"
#include "RgbSpectrum.hlsl"
#include "SpectralFiltering.hlsl"
struct SpectralFilterQuery
{
    SpectralTextureDescriptor Texture;
    float4 Coordinates;
    uint4 Indices;
    float4 Barycentric;
};
StructuredBuffer<SpectralSource> Sources : register(t0);
StructuredBuffer<float4> Response : register(t1);
StructuredBuffer<uint> Texels : register(t2);
StructuredBuffer<RgbSpectrum> Spectra : register(t3);
StructuredBuffer<SpectralFilterQuery> FilterQueries : register(t4);
RWStructuredBuffer<float4> PreparedResponse : register(u0);
RWStructuredBuffer<RgbSpectrum> PreparedSpectra : register(u1);
RWStructuredBuffer<float4> FilterResults : register(u2);
[shader("raygeneration")] void SpectralPrepareResponse() {
    uint index = DispatchRaysIndex().x;
    if (index < RgbFitSampleCount)
        PreparedResponse[index] = MakeRgbFitResponse(index);
}[shader("raygeneration")] void SpectralPreparePalette()
{
    const uint index = DispatchRaysIndex().x;
    PreparedSpectra[index] = FitSceneSpectrum(Sources[index], Response);
}
[shader("raygeneration")] void SpectralFilterProbe() {
    const uint index = DispatchRaysIndex().x;
    const SpectralFilterQuery query = FilterQueries[index];
    const float wavelength = query.Coordinates.z;
    float4 result;
    result.x = SampleSpectralTexture(query.Texture, query.Coordinates.xy, wavelength, Texels, Spectra);
    result.y = InterpolateSpectralVertex(Spectra[query.Indices.x], Spectra[query.Indices.y],
                                         Spectra[query.Indices.z], query.Barycentric.xyz, wavelength);
    result.z = EvaluateSceneSpectrum(Spectra[query.Indices.w], Sources[query.Indices.w].Kind, wavelength);
    result.w = 0;
    FilterResults[index] = result;
}
