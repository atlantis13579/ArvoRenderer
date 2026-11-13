#include "SpectralData.h"
// Independent MSL spectral inputs: peak-normalized Planck radiance, a smooth
// wavelength band, and Cauchy's transparent-index model in micrometers.
float BlackbodySpectrum(float wavelength, float temperature)
{
    if (!(wavelength > 0) || !(temperature > 0) || !isfinite(wavelength) || !isfinite(temperature))
        return 0;
    const float RadiationConstant = 14387768.775039337f; // h*c/k, in nm*K.
    const float WienPeakExponent = 4.965114231744276f;
    const float SmallThermalExponent = 0.01f;
    const float thermal = (RadiationConstant / wavelength) / temperature;
    if (!(thermal > 0) || !isfinite(thermal))
        return 0;
    float logarithm;
    if (thermal >= SmallThermalExponent)
        logarithm = thermal + log(1 - exp(-thermal));
    else
        logarithm = log(thermal) + log(1 + thermal * (0.5f + thermal / 6));
    const float peakLogarithm = WienPeakExponent + log(1 - exp(-WienPeakExponent));
    return exp(5 * (log(thermal) - log(WienPeakExponent)) + peakLogarithm - logarithm);
}
float GaussianSpectrum(float wavelength, float center, float width)
{
    if (width <= 0)
        return 0;
    float distance = (wavelength - center) / width;
    return exp(-distance * distance * 0.5f);
}
float CauchyIndex(float wavelength, float3 coefficients)
{
    if (wavelength <= 0)
        return 0;
    const float NmPerMicrometer = 1000;
    float frequency = NmPerMicrometer / wavelength;
    float squared = frequency * frequency;
    return coefficients.x + squared * coefficients.y + squared * squared * coefficients.z;
}
