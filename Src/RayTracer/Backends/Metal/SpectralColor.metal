#include "SpectralData.h"
// Native MSL evaluation of the 1931 two-degree analytic observer.
// Wyman/Sloan/Shirley (2013), Table 1: amplitude, center nm, left/right scales.
// https://jcgt.org/published/0002/02/01/paper.pdf
constant uint MatchingXLobes = 3, MatchingYLobes = 2, MatchingLobeCount = 7;
constant float4 MatchingLobes[MatchingLobeCount] = {
    float4(0.362f, 442, 0.0624f, 0.0374f),     float4(1.056f, 599.8f, 0.0264f, 0.0323f),
    float4(-0.065f, 501.1f, 0.0490f, 0.0382f), float4(0.821f, 568.8f, 0.0213f, 0.0247f),
    float4(0.286f, 530.9f, 0.0613f, 0.0322f),  float4(1.217f, 437, 0.0845f, 0.0278f),
    float4(0.681f, 459, 0.0385f, 0.0725f)};
float3 CieMatching(float wavelength)
{
    float3 matching = float3(0, 0, 0);
    if (wavelength < VisibleMinimumNm || wavelength > VisibleMaximumNm)
        return matching;
    for (uint index = 0; index < MatchingLobeCount; index++)
    {
        const float4 lobe = MatchingLobes[index];
        const float offset = wavelength - lobe.y;
        const float scaled = offset * (offset < 0 ? lobe.z : lobe.w);
        const float contribution = lobe.x * exp(-scaled * scaled / 2);
        if (index < MatchingXLobes)
            matching.x += contribution;
        else if (index < MatchingXLobes + MatchingYLobes)
            matching.y += contribution;
        else
            matching.z += contribution;
    }
    return matching;
}
float3 SpectrumToXYZ(WavelengthPacket wavelengths, SampledSpectrum spectrum)
{
    // Gaussian integral of this fit's y curve, with equal-energy Y normalized to one.
    const float YCurveIntegral = 106.94617150055822f;
    float3 value = float3(0, 0, 0);
    for (uint index = 0; index < SpectralPacketSize; index++)
        if (wavelengths.Pdf[index] > 0)
        {
            const float scale =
                spectrum.Value[index] / (wavelengths.Pdf[index] * YCurveIntegral * SpectralPacketSize);
            value += scale * CieMatching(wavelengths.Lambda[index]);
        }
    return value;
}
float3 XYZToLinearSRGB(float3 xyz)
{
    const float3 Red = float3(3.2406f, -1.5372f, -0.4986f), Green = float3(-0.9689f, 1.8758f, 0.0415f),
                 Blue = float3(0.0557f, -0.2040f, 1.0570f);
    // Four-decimal IEC sRGB/D65 transform. Negative channels are retained.
    return float3(dot(Red, xyz), dot(Green, xyz), dot(Blue, xyz));
}
