#include "SpectralData.hlsli"
// Independent compute wavelength proposals. A finite-domain sech^2 density
// approximates visual response; its truncated integral is normalized exactly.
// PBRT v4, "Representing Spectral Distributions": randomly rotated packets.
// https://pbr-book.org/4ed/Radiometry,_Spectra,_and_Color/Representing_Spectral_Distributions
static const float WavelengthCenter = 538, WavelengthInverseWidth = 0.0072f;
static const float WavelengthOpenUnit = 0.9999999403953552f;
float WavelengthResponse(float wavelength)
{
    return tanh((wavelength - WavelengthCenter) * WavelengthInverseWidth);
}
float WavelengthPdf(float wavelength, bool importance)
{
    if (wavelength < VisibleMinimumNm || wavelength > VisibleMaximumNm)
        return 0;
    if (!importance)
        return 1 / (VisibleMaximumNm - VisibleMinimumNm);
    const float span = WavelengthResponse(VisibleMaximumNm) - WavelengthResponse(VisibleMinimumNm);
    const float response = WavelengthResponse(wavelength);
    return (1 - response) * (1 + response) * WavelengthInverseWidth / span;
}
WavelengthPacket SampleWavelengths(float variate, bool importance)
{
    WavelengthPacket packet;
    const float low = WavelengthResponse(VisibleMinimumNm);
    const float high = WavelengthResponse(VisibleMaximumNm);
    const float start = clamp(variate, 0.0f, WavelengthOpenUnit);
    for (uint lane = 0; lane < SpectralPacketSize; ++lane)
    {
        float u = start + float(lane) / SpectralPacketSize;
        if (u >= 1)
            u -= 1;
        float wavelength = VisibleMinimumNm + u * (VisibleMaximumNm - VisibleMinimumNm);
        if (importance)
        {
            const float cdf = low + u * (high - low);
            wavelength = WavelengthCenter + log((1 + cdf) / (1 - cdf)) / (2 * WavelengthInverseWidth);
        }
        packet.Lambda[lane] = clamp(wavelength, VisibleMinimumNm, VisibleMaximumNm);
        packet.Pdf[lane] = WavelengthPdf(packet.Lambda[lane], importance);
    }
    return packet;
}
WavelengthPacket TerminateSecondaryWavelengths(WavelengthPacket packet)
{
    bool live = false;
    for (uint lane = 1; lane < SpectralPacketSize; ++lane)
        live = live || packet.Pdf[lane] > 0;
    if (live)
    {
        packet.Pdf[0] /= SpectralPacketSize;
        for (uint lane = 1; lane < SpectralPacketSize; ++lane)
            packet.Pdf[lane] = 0;
    }
    return packet;
}
