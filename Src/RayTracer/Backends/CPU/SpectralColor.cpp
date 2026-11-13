#include "RayTracer/Backends/CPU/Spectral.h"
#include <cmath>

namespace Arvo::RayTracing::Cpu
{
namespace
{
// Wyman, Sloan and Shirley, "Simple Analytic Approximations to the CIE XYZ
// Color Matching Functions" (2013), Eq. 4 / Table 1: a 1931 two-degree fit.
// https://jcgt.org/published/0002/02/01/paper.pdf
// These coefficients are mathematical fit data, not measured scene resources.
struct MatchingLobe
{
    double Amplitude, Center, LeftScale, RightScale;
};
constexpr MatchingLobe XBlue{0.362, 442, 0.0624, 0.0374}, XRed{1.056, 599.8, 0.0264, 0.0323},
    XNotch{-0.065, 501.1, 0.0490, 0.0382}, YBroad{0.821, 568.8, 0.0213, 0.0247},
    YNarrow{0.286, 530.9, 0.0613, 0.0322}, ZBlue{1.217, 437, 0.0845, 0.0278},
    ZViolet{0.681, 459, 0.0385, 0.0725};
double Lobe(double wavelength, MatchingLobe fit)
{
    const double delta = wavelength - fit.Center;
    const double scaled = delta * (delta < 0 ? fit.LeftScale : fit.RightScale);
    return fit.Amplitude * std::exp(-scaled * scaled / 2);
}
// Integral of this fit's y curve over [360,830], using the Gaussian erf integral.
// Equal-energy unit spectra have Y=1; this is not the tabulated CIE normalization.
constexpr double MatchingYIntegral = 106.94617150055822;
} // namespace
Float3 CieMatching(float wavelengthNm)
{
    if (wavelengthNm < VisibleMinimumNm || wavelengthNm > VisibleMaximumNm)
        return {};
    return {float(Lobe(wavelengthNm, XBlue) + Lobe(wavelengthNm, XRed) + Lobe(wavelengthNm, XNotch)),
            float(Lobe(wavelengthNm, YBroad) + Lobe(wavelengthNm, YNarrow)),
            float(Lobe(wavelengthNm, ZBlue) + Lobe(wavelengthNm, ZViolet))};
}
Float3 SpectrumToXYZ(const WavelengthPacket& packet, const SampledSpectrum& spectrum)
{
    double x = 0, y = 0, z = 0;
    for (unsigned lane = 0; lane < SpectralPacketSize; ++lane)
    {
        if (packet.Pdf[lane] <= 0)
            continue;
        const auto matching = CieMatching(packet.Lambda[lane]);
        const double weight =
            double(spectrum.Value[lane]) / packet.Pdf[lane] / SpectralPacketSize / MatchingYIntegral;
        x += matching.X * weight;
        y += matching.Y * weight;
        z += matching.Z * weight;
    }
    return {float(x), float(y), float(z)};
}
Float3 XYZToLinearSRGB(Float3 xyz)
{
    // IEC sRGB / D65 inverse primary matrix, rounded to four decimal places.
    // https://registry.color.org/rgb-registry/files/sRGB.pdf
    // Retain signed out-of-gamut components in the linear film; do not clip here.
    const Float3 RedRow{3.2406f, -1.5372f, -0.4986f}, GreenRow{-0.9689f, 1.8758f, 0.0415f},
        BlueRow{0.0557f, -0.2040f, 1.0570f};
    return {Dot(RedRow, xyz), Dot(GreenRow, xyz), Dot(BlueRow, xyz)};
}
} // namespace Arvo::RayTracing::Cpu
