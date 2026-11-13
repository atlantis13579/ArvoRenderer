// Independent compute Planck emission, Gaussian band and Cauchy dispersion.
// Planck radiance is divided by its wavelength-domain Wien peak. No RGB lifting
// or measured optical data is implied by these analytic building blocks.
float BlackbodySpectrum(float wavelength, float temperature)
{
    if (!(wavelength > 0) || !(temperature > 0) || !isfinite(wavelength) || !isfinite(temperature))
        return 0;
    const float SecondRadiationConstantNmK = 14387768.775039337f;
    const float WienRoot = 4.965114231744276f;
    const float SmallExponent = 0.01f;
    const float exponent = (SecondRadiationConstantNmK / wavelength) / temperature;
    if (!(exponent > 0) || !isfinite(exponent))
        return 0;
    float logDenominator;
    if (exponent < SmallExponent)
        logDenominator = log(exponent) + log(1 + exponent * (0.5f + exponent / 6));
    else
        logDenominator = exponent + log(1 - exp(-exponent));
    const float peakDenominator = WienRoot + log(1 - exp(-WienRoot));
    return exp(5 * (log(exponent) - log(WienRoot)) + peakDenominator - logDenominator);
}
float GaussianSpectrum(float wavelength, float center, float width)
{
    if (!(width > 0))
        return 0;
    const float coordinate = (wavelength - center) / width;
    return exp(-coordinate * coordinate / 2);
}
float CauchyIndex(float wavelength, float3 coefficients)
{
    if (!(wavelength > 0))
        return 0;
    const float NanometersPerMicrometer = 1000;
    const float reciprocal = NanometersPerMicrometer / wavelength;
    const float inverseSquared = reciprocal * reciprocal;
    return coefficients.x + inverseSquared * (coefficients.y + inverseSquared * coefficients.z);
}
