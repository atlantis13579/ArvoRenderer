#include "SpectralData.h"
// Native MSL wavelength sampling, independently authored. Invert the truncated
// logistic CDF; its derivative is the visible-response sech^2 proposal.
// Rotated packets and idempotent hero reduction follow PBRT v4:
// https://pbr-book.org/4ed/Radiometry,_Spectra,_and_Color/Representing_Spectral_Distributions
constant float MetalResponseCenterNm = 538, MetalResponseScaleNm = 1 / 0.0072f;
float MetalSpectralMass(float wavelength)
{
    const float exponent = 2 * (wavelength - MetalResponseCenterNm) / MetalResponseScaleNm;
    const float exponential = exp(exponent);
    return exponential / (1 + exponential);
}
float WavelengthPdf(float wavelength, bool importance)
{
    if (wavelength < VisibleMinimumNm || wavelength > VisibleMaximumNm)
        return 0;
    if (!importance)
        return 1 / (VisibleMaximumNm - VisibleMinimumNm);
    const float mass = MetalSpectralMass(wavelength);
    const float normalization = MetalSpectralMass(VisibleMaximumNm) - MetalSpectralMass(VisibleMinimumNm);
    return (2 / MetalResponseScaleNm) * mass * (1 - mass) / normalization;
}
WavelengthPacket SampleWavelengths(float variate, bool importance)
{
    const float MaximumUnit = 0.9999999403953552f;
    const float lower = MetalSpectralMass(VisibleMinimumNm), upper = MetalSpectralMass(VisibleMaximumNm);
    const float rotation = clamp(variate, 0.0f, MaximumUnit);
    WavelengthPacket sampled;
    for (uint lane = 0; lane < SpectralPacketSize; lane++)
    {
        float unit = rotation + float(lane) / float(SpectralPacketSize);
        unit = unit >= 1 ? unit - 1 : unit;
        float wavelength;
        if (importance)
        {
            const float mass = lower * (1 - unit) + upper * unit;
            wavelength = MetalResponseCenterNm + (MetalResponseScaleNm / 2) * (log(mass) - log(1 - mass));
        }
        else
            wavelength = VisibleMinimumNm * (1 - unit) + VisibleMaximumNm * unit;
        sampled.Lambda[lane] = clamp(wavelength, VisibleMinimumNm, VisibleMaximumNm);
        sampled.Pdf[lane] = WavelengthPdf(sampled.Lambda[lane], importance);
    }
    return sampled;
}
WavelengthPacket TerminateSecondaryWavelengths(WavelengthPacket packet)
{
    uint secondaryCount = 0;
    for (uint lane = 1; lane < SpectralPacketSize; lane++)
        secondaryCount += packet.Pdf[lane] > 0 ? 1u : 0u;
    if (secondaryCount != 0)
    {
        packet.Pdf[0] = packet.Pdf[0] / float(SpectralPacketSize);
        for (uint lane = 1; lane < SpectralPacketSize; lane++)
            packet.Pdf[lane] = 0;
    }
    return packet;
}
