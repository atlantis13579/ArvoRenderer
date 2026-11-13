#include "SpectralData.cuh"
// Independent CUDA CIE 1931 approximation, Wyman/Sloan/Shirley (2013), Table 1.
// https://jcgt.org/published/0002/02/01/paper.pdf
struct CudaColorLobe
{
    float Amplitude, Center, LeftInverseWidth, RightInverseWidth;
};
__device__ float CudaLobeValue(float wavelength, CudaColorLobe fit)
{
    const float scale = wavelength < fit.Center ? fit.LeftInverseWidth : fit.RightInverseWidth;
    const float standardized = (wavelength - fit.Center) * scale;
    return fit.Amplitude * expf(-standardized * standardized * 0.5f);
}
__device__ float3 CieMatching(float wavelength)
{
    if (wavelength < VisibleMinimumNm || wavelength > VisibleMaximumNm)
        return make_float3(0, 0, 0);
    const CudaColorLobe XBlue{0.362f, 442, 0.0624f, 0.0374f}, XRed{1.056f, 599.8f, 0.0264f, 0.0323f},
        XNotch{-0.065f, 501.1f, 0.0490f, 0.0382f}, YBroad{0.821f, 568.8f, 0.0213f, 0.0247f},
        YNarrow{0.286f, 530.9f, 0.0613f, 0.0322f}, ZBlue{1.217f, 437, 0.0845f, 0.0278f},
        ZViolet{0.681f, 459, 0.0385f, 0.0725f};
    return make_float3(CudaLobeValue(wavelength, XBlue) + CudaLobeValue(wavelength, XRed) +
                           CudaLobeValue(wavelength, XNotch),
                       CudaLobeValue(wavelength, YBroad) + CudaLobeValue(wavelength, YNarrow),
                       CudaLobeValue(wavelength, ZBlue) + CudaLobeValue(wavelength, ZViolet));
}
__device__ float3 SpectrumToXYZ(WavelengthPacket packet, SampledSpectrum spectrum)
{
    // Normalize by the fitted y integral, not a different tabulated observer.
    constexpr double FittedYIntegral = 106.94617150055822;
    double result[3]{};
    for (unsigned lane = 0; lane < SpectralPacketSize; ++lane)
    {
        if (packet.Pdf[lane] <= 0)
            continue;
        const float3 observer = CieMatching(packet.Lambda[lane]);
        const double scale =
            double(spectrum.Value[lane]) / (double(packet.Pdf[lane]) * SpectralPacketSize * FittedYIntegral);
        result[0] += observer.x * scale;
        result[1] += observer.y * scale;
        result[2] += observer.z * scale;
    }
    return make_float3(float(result[0]), float(result[1]), float(result[2]));
}
__device__ float3 XYZToLinearSRGB(float3 xyz)
{
    // IEC sRGB/D65 inverse primary matrix; preserve signed/HDR film components.
    const float3 Red{3.2406f, -1.5372f, -0.4986f}, Green{-0.9689f, 1.8758f, 0.0415f},
        Blue{0.0557f, -0.2040f, 1.0570f};
    return make_float3(Red.x * xyz.x + Red.y * xyz.y + Red.z * xyz.z,
                       Green.x * xyz.x + Green.y * xyz.y + Green.z * xyz.z,
                       Blue.x * xyz.x + Blue.y * xyz.y + Blue.z * xyz.z);
}
