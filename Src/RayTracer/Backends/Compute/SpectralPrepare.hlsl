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
cbuffer SpectralPreparationConstants : register(b0)
{
    uint SourceCount, FilterCount;
}
static const uint SpectralPreparationGroupSize = 64;
[numthreads(SpectralPreparationGroupSize, 1, 1)] void SpectralPrepareResponse(
    uint3 id : SV_DispatchThreadID) {
    if (id.x < RgbFitSampleCount)
        PreparedResponse[id.x] = MakeRgbFitResponse(id.x);
}[numthreads(SpectralPreparationGroupSize, 1, 1)] void SpectralPreparePalette(uint3 id : SV_DispatchThreadID)
{
    if (id.x < SourceCount)
        PreparedSpectra[id.x] = FitSceneSpectrum(Sources[id.x], Response);
}
[numthreads(SpectralPreparationGroupSize, 1, 1)] void SpectralFilterProbe(uint3 id : SV_DispatchThreadID) {
    if (id.x >= FilterCount)
        return;
    const SpectralFilterQuery query = FilterQueries[id.x];
    const float wavelength = query.Coordinates.z;
    FilterResults[id.x] =
        float4(SampleSpectralTexture(query.Texture, query.Coordinates.xy, wavelength, Texels, Spectra),
               InterpolateSpectralVertex(Spectra[query.Indices.x], Spectra[query.Indices.y],
                                         Spectra[query.Indices.z], query.Barycentric.xyz, wavelength),
               EvaluateSceneSpectrum(Spectra[query.Indices.w], Sources[query.Indices.w].Kind, wavelength), 0);
}
