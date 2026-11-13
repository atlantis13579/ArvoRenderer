#include "SpectralData.hlsli"
// Independent DXR wavelength sampling for spectral offline jobs.
// PBRT v4, "Representing Spectral Distributions": each rotated lane spans the
// full visible interval; retaining lane zero after dispersion needs a 1/N PDF.
// https://pbr-book.org/4ed/Radiometry,_Spectra,_and_Color/Representing_Spectral_Distributions
static const float DxrResponseCenterNm = 538, DxrResponseSlope = 0.0072f;
float DxrWavelengthCdf(float wavelength)
{
    return 1 / (1 + exp(-2 * DxrResponseSlope * (wavelength - DxrResponseCenterNm)));
}
float WavelengthPdf(float wavelength, bool importance)
{
    if (wavelength < VisibleMinimumNm || wavelength > VisibleMaximumNm)
        return 0;
    float density = 1 / (VisibleMaximumNm - VisibleMinimumNm);
    if (importance)
    {
        const float cdf = DxrWavelengthCdf(wavelength);
        const float domainMass = DxrWavelengthCdf(VisibleMaximumNm) - DxrWavelengthCdf(VisibleMinimumNm);
        density = 2 * DxrResponseSlope * cdf * (1 - cdf) / domainMass;
    }
    return density;
}
WavelengthPacket SampleWavelengths(float variate, bool importance)
{
    const float OpenUnitMaximum = 0.9999999403953552f;
    const float lowerMass = DxrWavelengthCdf(VisibleMinimumNm);
    const float upperMass = DxrWavelengthCdf(VisibleMaximumNm);
    WavelengthPacket result;
    for (uint lane = 0; lane != SpectralPacketSize; ++lane)
    {
        float rotated = clamp(variate, 0.0f, OpenUnitMaximum) + float(lane) / SpectralPacketSize;
        rotated -= floor(rotated);
        const float probability = lowerMass + (upperMass - lowerMass) * rotated;
        const float wavelength =
            importance ? DxrResponseCenterNm + log(probability / (1 - probability)) / (2 * DxrResponseSlope)
                       : VisibleMinimumNm + (VisibleMaximumNm - VisibleMinimumNm) * rotated;
        result.Lambda[lane] = clamp(wavelength, VisibleMinimumNm, VisibleMaximumNm);
        result.Pdf[lane] = WavelengthPdf(result.Lambda[lane], importance);
    }
    return result;
}
WavelengthPacket TerminateSecondaryWavelengths(WavelengthPacket packet)
{
    float secondaryMass = 0;
    for (uint lane = 1; lane != SpectralPacketSize; ++lane)
        secondaryMass += packet.Pdf[lane];
    if (secondaryMass > 0)
    {
        for (uint lane = 1; lane != SpectralPacketSize; ++lane)
            packet.Pdf[lane] = 0;
        packet.Pdf[0] *= 1.0f / SpectralPacketSize;
    }
    return packet;
}
