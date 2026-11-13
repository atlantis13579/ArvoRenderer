#include "RayTracer/Backends/CPU/Spectral.h"
#include <cmath>

namespace Arvo::RayTracing::Cpu
{
// Analytic spectra need no material assets. Planck emission is normalized at
// its wavelength-domain Wien peak, not at an arbitrary visible wavelength.
// Cauchy A/B/C use wavelength in micrometers; B and C carry um^2 and um^4.
// They model transparent dispersion away from resonances, not absorption.
float BlackbodySpectrum(float wavelengthNm, float temperatureKelvin)
{
    if (!(wavelengthNm > 0) || !(temperatureKelvin > 0) || !std::isfinite(wavelengthNm) ||
        !std::isfinite(temperatureKelvin))
        return 0;
    constexpr double SecondRadiationConstantNmK = 14387768.775039337;
    constexpr double WienRoot = 4.965114231744276; // x = 5*(1-exp(-x)).
    const double exponent = SecondRadiationConstantNmK / wavelengthNm / temperatureKelvin;
    const double logDenominator =
        exponent > 1 ? exponent + std::log1p(-std::exp(-exponent)) : std::log(std::expm1(exponent));
    return float(
        std::exp(5 * std::log(exponent / WienRoot) + std::log(std::expm1(WienRoot)) - logDenominator));
}
float GaussianSpectrum(float wavelengthNm, float centerNm, float widthNm)
{
    if (!(widthNm > 0))
        return 0;
    const double distance = (double(wavelengthNm) - centerNm) / widthNm;
    return float(std::exp(-distance * distance / 2));
}
float CauchyIndex(float wavelengthNm, Float3 coefficients)
{
    constexpr double NanometersPerMicrometer = 1000;
    if (!(wavelengthNm > 0))
        return 0;
    const double inverseMicrometers = NanometersPerMicrometer / wavelengthNm;
    const double inverseSquared = inverseMicrometers * inverseMicrometers;
    return float(coefficients.X + inverseSquared * (coefficients.Y + coefficients.Z * inverseSquared));
}
} // namespace Arvo::RayTracing::Cpu
