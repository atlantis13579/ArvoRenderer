#include "SpectralData.hlsli"
// Independent DXR colorimetry: Wyman/Sloan/Shirley (2013), Eq. 4 / Table 1.
// https://jcgt.org/published/0002/02/01/paper.pdf
// The approximation is explicitly the CIE 1931 two-degree observer.
float DxrMatchingLobe(float wavelength, float amplitude, float center, float leftScale, float rightScale)
{
    const float distance = wavelength - center;
    const float width = wavelength < center ? leftScale : rightScale;
    return amplitude * exp(-0.5f * (distance * width) * (distance * width));
}
float3 CieMatching(float wavelength)
{
    float3 result = float3(0, 0, 0);
    if (wavelength >= VisibleMinimumNm && wavelength <= VisibleMaximumNm)
    {
        // Each named row is (amplitude, center in nm, left scale, right scale).
        const float4 XBlue = float4(0.362f, 442, 0.0624f, 0.0374f),
                     XRed = float4(1.056f, 599.8f, 0.0264f, 0.0323f);
        const float4 XNotch = float4(-0.065f, 501.1f, 0.0490f, 0.0382f);
        const float4 YBroad = float4(0.821f, 568.8f, 0.0213f, 0.0247f),
                     YNarrow = float4(0.286f, 530.9f, 0.0613f, 0.0322f);
        const float4 ZBlue = float4(1.217f, 437, 0.0845f, 0.0278f),
                     ZViolet = float4(0.681f, 459, 0.0385f, 0.0725f);
        result.x = DxrMatchingLobe(wavelength, XBlue.x, XBlue.y, XBlue.z, XBlue.w) +
                   DxrMatchingLobe(wavelength, XRed.x, XRed.y, XRed.z, XRed.w) +
                   DxrMatchingLobe(wavelength, XNotch.x, XNotch.y, XNotch.z, XNotch.w);
        result.y = DxrMatchingLobe(wavelength, YBroad.x, YBroad.y, YBroad.z, YBroad.w) +
                   DxrMatchingLobe(wavelength, YNarrow.x, YNarrow.y, YNarrow.z, YNarrow.w);
        result.z = DxrMatchingLobe(wavelength, ZBlue.x, ZBlue.y, ZBlue.z, ZBlue.w) +
                   DxrMatchingLobe(wavelength, ZViolet.x, ZViolet.y, ZViolet.z, ZViolet.w);
    }
    return result;
}
float3 SpectrumToXYZ(WavelengthPacket packet, SampledSpectrum spectrum)
{
    const float FittedYIntegral = 106.94617150055822f; // Gaussian erf integral over [360,830].
    float3 integral = float3(0, 0, 0);
    for (uint lane = 0; lane != SpectralPacketSize; ++lane)
    {
        if (packet.Pdf[lane] <= 0)
            continue;
        const float importanceWeight = spectrum.Value[lane] / packet.Pdf[lane];
        integral += CieMatching(packet.Lambda[lane]) * importanceWeight;
    }
    return integral / (SpectralPacketSize * FittedYIntegral);
}
float3 XYZToLinearSRGB(float3 xyz)
{
    const float3 RedPrimary = float3(3.2406f, -1.5372f, -0.4986f);
    const float3 GreenPrimary = float3(-0.9689f, 1.8758f, 0.0415f);
    const float3 BluePrimary = float3(0.0557f, -0.2040f, 1.0570f);
    // Four-decimal IEC sRGB/D65 matrix; spectral colors may be outside its gamut.
    return float3(dot(xyz, RedPrimary), dot(xyz, GreenPrimary), dot(xyz, BluePrimary));
}
