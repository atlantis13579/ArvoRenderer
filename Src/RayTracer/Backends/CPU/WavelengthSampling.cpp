#include "RayTracer/Backends/CPU/Spectral.h"
#include <algorithm>
#include <cmath>

namespace Arvo::RayTracing::Cpu
{
namespace
{
// PBRT v4, "Representing Spectral Distributions" and "Film and Imaging".
// https://pbr-book.org/4ed/Radiometry,_Spectra,_and_Color/Representing_Spectral_Distributions
// Four correlated wavelengths share a randomly rotated unit interval. Every
// lane has the full marginal distribution, including the surviving hero lane.
constexpr double ResponseCenterNm = 538, ResponseInverseWidth = 0.0072;
constexpr float OpenUnitMaximum = 0.9999999403953552f;
const double LowerResponse = std::tanh((VisibleMinimumNm - ResponseCenterNm) * ResponseInverseWidth);
const double UpperResponse = std::tanh((VisibleMaximumNm - ResponseCenterNm) * ResponseInverseWidth);
} // namespace
float WavelengthPdf(float wavelengthNm, bool importance)
{
    if (wavelengthNm < VisibleMinimumNm || wavelengthNm > VisibleMaximumNm)
        return 0;
    if (!importance)
        return 1 / (VisibleMaximumNm - VisibleMinimumNm);
    const double response = std::tanh((wavelengthNm - ResponseCenterNm) * ResponseInverseWidth);
    return float((1 - response) * (1 + response) * ResponseInverseWidth / (UpperResponse - LowerResponse));
}
WavelengthPacket SampleWavelengths(float variate, bool importance)
{
    WavelengthPacket packet;
    const double start = std::clamp(variate, 0.0f, OpenUnitMaximum);
    for (unsigned lane = 0; lane < SpectralPacketSize; ++lane)
    {
        double u = start + double(lane) / SpectralPacketSize;
        if (u >= 1)
            u -= 1;
        const double wavelength =
            importance ? ResponseCenterNm + std::atanh(LowerResponse + u * (UpperResponse - LowerResponse)) /
                                                ResponseInverseWidth
                       : VisibleMinimumNm + u * (VisibleMaximumNm - VisibleMinimumNm);
        packet.Lambda[lane] = std::clamp(float(wavelength), VisibleMinimumNm, VisibleMaximumNm);
        packet.Pdf[lane] = WavelengthPdf(packet.Lambda[lane], importance);
    }
    return packet;
}
WavelengthPacket TerminateSecondaryWavelengths(WavelengthPacket packet)
{
    bool hasSecondary = false;
    for (unsigned lane = 1; lane < SpectralPacketSize; ++lane)
        hasSecondary = hasSecondary || packet.Pdf[lane] > 0;
    if (hasSecondary)
    {
        // Preserve the packet estimator's 1/N normalization after dispersion.
        packet.Pdf[0] /= SpectralPacketSize;
        for (unsigned lane = 1; lane < SpectralPacketSize; ++lane)
            packet.Pdf[lane] = 0;
    }
    return packet;
}
} // namespace Arvo::RayTracing::Cpu
