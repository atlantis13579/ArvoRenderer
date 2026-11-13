#include "SpectralData.hlsli"
// Wyman/Sloan/Shirley (2013), Eq. 4 / Table 1, independently evaluated in HLSL.
// https://jcgt.org/published/0002/02/01/paper.pdf
// Lobe records hold amplitude, center in nm, left scale and right scale.
float ComputeMatchingLobe(float wavelength, float4 fit)
{
    const float offset = wavelength - fit.y;
    const float argument = offset * (offset < 0 ? fit.z : fit.w);
    return fit.x * exp(-argument * argument / 2);
}
float3 CieMatching(float wavelength)
{
    if (wavelength < VisibleMinimumNm || wavelength > VisibleMaximumNm)
        return float3(0, 0, 0);
    const float4 XBlue = float4(0.362f, 442, 0.0624f, 0.0374f);
    const float4 XRed = float4(1.056f, 599.8f, 0.0264f, 0.0323f);
    const float4 XNotch = float4(-0.065f, 501.1f, 0.0490f, 0.0382f);
    const float4 YBroad = float4(0.821f, 568.8f, 0.0213f, 0.0247f);
    const float4 YNarrow = float4(0.286f, 530.9f, 0.0613f, 0.0322f);
    const float4 ZBlue = float4(1.217f, 437, 0.0845f, 0.0278f);
    const float4 ZViolet = float4(0.681f, 459, 0.0385f, 0.0725f);
    return float3(ComputeMatchingLobe(wavelength, XBlue) + ComputeMatchingLobe(wavelength, XRed) +
                      ComputeMatchingLobe(wavelength, XNotch),
                  ComputeMatchingLobe(wavelength, YBroad) + ComputeMatchingLobe(wavelength, YNarrow),
                  ComputeMatchingLobe(wavelength, ZBlue) + ComputeMatchingLobe(wavelength, ZViolet));
}
float3 SpectrumToXYZ(WavelengthPacket packet, SampledSpectrum spectrum)
{
    // Exact Gaussian integral of the fitted y curve on the visible domain.
    const float MatchingYIntegral = 106.94617150055822f;
    float3 xyz = float3(0, 0, 0);
    for (uint lane = 0; lane < SpectralPacketSize; ++lane)
        if (packet.Pdf[lane] > 0)
            xyz += CieMatching(packet.Lambda[lane]) *
                   (spectrum.Value[lane] / packet.Pdf[lane] / (MatchingYIntegral * SpectralPacketSize));
    return xyz;
}
float3 XYZToLinearSRGB(float3 xyz)
{
    // IEC sRGB/D65 matrix. Signed film values are intentional, without clipping.
    const float3 RedRow = float3(3.2406f, -1.5372f, -0.4986f);
    const float3 GreenRow = float3(-0.9689f, 1.8758f, 0.0415f);
    const float3 BlueRow = float3(0.0557f, -0.2040f, 1.0570f);
    return float3(dot(RedRow, xyz), dot(GreenRow, xyz), dot(BlueRow, xyz));
}
