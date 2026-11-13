// Independent DXR analytic optical inputs. Wien-normalized Planck emission
// retains radiance shape, while Cauchy dispersion expects micrometer units.
float BlackbodySpectrum(float wavelength, float temperature)
{
    if (!(wavelength > 0) || !(temperature > 0) || !isfinite(wavelength) || !isfinite(temperature))
        return 0;
    const float RadiationConstantNmK = 14387768.775039337f;
    const float WienMaximum = 4.965114231744276f;
    const float SeriesThreshold = 0.01f;
    float x = RadiationConstantNmK / wavelength / temperature;
    if (!(x > 0) || !isfinite(x))
        return 0;
    float thermalLog = x + log(1 - exp(-x));
    if (x < SeriesThreshold)
        thermalLog = log(x) + log(1 + x / 2 + x * x / 6);
    const float peakLog = WienMaximum + log(1 - exp(-WienMaximum));
    return exp(5 * log(x) - 5 * log(WienMaximum) + peakLog - thermalLog);
}
float GaussianSpectrum(float wavelength, float center, float width)
{
    float value = 0;
    if (width > 0)
    {
        const float offset = (wavelength - center) / width;
        value = exp(-0.5f * offset * offset);
    }
    return value;
}
float CauchyIndex(float wavelength, float3 coefficients)
{
    if (!(wavelength > 0))
        return 0;
    const float NanometersToMicrometers = 0.001f;
    const float micrometers = wavelength * NanometersToMicrometers;
    const float wavelengthSquared = micrometers * micrometers;
    return coefficients.x + coefficients.y / wavelengthSquared +
           coefficients.z / (wavelengthSquared * wavelengthSquared);
}
