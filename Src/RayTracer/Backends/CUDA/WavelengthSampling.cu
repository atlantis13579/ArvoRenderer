#include "SpectralData.cuh"
// CUDA owns this wavelength law and the packet survival correction.
// The sech^2 response is normalized on a finite wavelength interval; density
// is per nanometer. A rotated packet gives each lane the same full marginal.
// PBRT v4: https://pbr-book.org/4ed/Radiometry,_Spectra,_and_Color/Representing_Spectral_Distributions
constexpr double CudaVisibleCenterNm = 538, CudaVisibleSlope = 0.0072;
__device__ float WavelengthPdf(float wavelength, bool importance)
{
    if (wavelength < VisibleMinimumNm || wavelength > VisibleMaximumNm)
        return 0;
    if (!importance)
        return 1 / (VisibleMaximumNm - VisibleMinimumNm);
    const double low = tanh((VisibleMinimumNm - CudaVisibleCenterNm) * CudaVisibleSlope);
    const double high = tanh((VisibleMaximumNm - CudaVisibleCenterNm) * CudaVisibleSlope);
    const double argument = (wavelength - CudaVisibleCenterNm) * CudaVisibleSlope;
    const double inverseCosh = 1 / cosh(argument);
    return float(CudaVisibleSlope * inverseCosh * inverseCosh / (high - low));
}
__device__ WavelengthPacket SampleWavelengths(float variate, bool importance)
{
    constexpr float OpenUnitMaximum = 0.9999999403953552f;
    const double low = tanh((VisibleMinimumNm - CudaVisibleCenterNm) * CudaVisibleSlope);
    const double mass = tanh((VisibleMaximumNm - CudaVisibleCenterNm) * CudaVisibleSlope) - low;
    const double rotation = fminf(OpenUnitMaximum, fmaxf(0, variate));
    WavelengthPacket result;
    for (unsigned lane = 0; lane < SpectralPacketSize; ++lane)
    {
        const double shifted = rotation + double(lane) / SpectralPacketSize;
        const double u = shifted - floor(shifted);
        const double wavelength = importance ? CudaVisibleCenterNm + atanh(low + mass * u) / CudaVisibleSlope
                                             : VisibleMinimumNm + (VisibleMaximumNm - VisibleMinimumNm) * u;
        result.Lambda[lane] = fminf(VisibleMaximumNm, fmaxf(VisibleMinimumNm, float(wavelength)));
        result.Pdf[lane] = WavelengthPdf(result.Lambda[lane], importance);
    }
    return result;
}
__device__ WavelengthPacket TerminateSecondaryWavelengths(WavelengthPacket result)
{
    bool alreadyTerminated = true;
    for (unsigned lane = 1; lane < SpectralPacketSize; ++lane)
        alreadyTerminated = alreadyTerminated && result.Pdf[lane] == 0;
    if (!alreadyTerminated)
    {
        result.Pdf[0] /= SpectralPacketSize;
        for (unsigned lane = 1; lane < SpectralPacketSize; ++lane)
            result.Pdf[lane] = 0;
    }
    return result;
}
