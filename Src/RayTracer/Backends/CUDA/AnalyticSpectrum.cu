#include "SpectralData.cuh"
// Independent analytic spectral inputs, with no RGB-to-spectrum table.
// Planck's exponent is evaluated in log space to cover cold and hot limits.
// Cauchy B/C coefficients use um^2/um^4, whereas public wavelengths use nm.
__device__ float BlackbodySpectrum(float wavelength, float temperature)
{
    if (!(wavelength > 0) || !(temperature > 0) || !isfinite(wavelength) || !isfinite(temperature))
        return 0;
    constexpr double RadiationNmK = 14387768.775039337, WienRoot = 4.965114231744276;
    const double x = RadiationNmK / wavelength / temperature;
    const double logThermal = x < 1 ? log(expm1(x)) : x + log1p(-exp(-x));
    const double logPeak = log(expm1(WienRoot));
    return float(exp(5 * (log(x) - log(WienRoot)) + logPeak - logThermal));
}
__device__ float GaussianSpectrum(float wavelength, float center, float width)
{
    if (!(width > 0))
        return 0;
    const double standardized = (double(wavelength) - center) / width;
    return float(exp(-0.5 * standardized * standardized));
}
__device__ float CauchyIndex(float wavelength, float3 coefficients)
{
    if (!(wavelength > 0))
        return 0;
    constexpr double NanometersPerMicrometer = 1000;
    const double ratio = NanometersPerMicrometer / wavelength;
    const double inverseSquare = ratio * ratio;
    return float(double(coefficients.x) + coefficients.y * inverseSquare +
                 coefficients.z * inverseSquare * inverseSquare);
}
