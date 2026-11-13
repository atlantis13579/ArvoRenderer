#pragma once
#include "Renderer/RayTracing.h"
#include "Renderer/SpectralSceneData.h"
#include <array>

namespace Arvo::RayTracing::Cpu
{
inline constexpr unsigned SpectralPacketSize = 4;
inline constexpr float VisibleMinimumNm = 360, VisibleMaximumNm = 830;
struct WavelengthPacket
{
    float Lambda[SpectralPacketSize]{};
    float Pdf[SpectralPacketSize]{};
};
struct SampledSpectrum
{
    float Value[SpectralPacketSize]{};
};
WavelengthPacket SampleWavelengths(float variate, bool importance);
float WavelengthPdf(float wavelengthNm, bool importance);
WavelengthPacket TerminateSecondaryWavelengths(WavelengthPacket packet);
Float3 CieMatching(float wavelengthNm);
Float3 SpectrumToXYZ(const WavelengthPacket&, const SampledSpectrum&);
Float3 XYZToLinearSRGB(Float3 xyz);
float BlackbodySpectrum(float wavelengthNm, float temperatureKelvin);
float GaussianSpectrum(float wavelengthNm, float centerNm, float widthNm);
float CauchyIndex(float wavelengthNm, Float3 coefficients);

inline constexpr unsigned RgbFitSampleCount = 256;
inline constexpr unsigned SpectrumFitConverged = 0, SpectrumFitInvalid = 1, SpectrumFitUnconverged = 2;
struct RgbSpectrum
{
    // Constant + Scale * sigmoid(quadratic); Error is normalized chromatic residual.
    float Quadratic, Linear, Offset, Scale;
    float Constant, Error;
    unsigned Status, Iterations;
};
struct RgbFitResponse
{
    Float3 Weight;
    float Coordinate;
};
using RgbFitTable = std::array<RgbFitResponse, RgbFitSampleCount>;
float RgbReferenceIlluminant(float wavelengthNm);
RgbFitResponse MakeRgbFitResponse(unsigned index);
RgbSpectrum FitRgbReflectance(Float3 linearRgb, const RgbFitTable& response);
RgbSpectrum FitRgbIlluminant(Float3 linearRgb, const RgbFitTable& response);
float EvaluateRgbReflectance(RgbSpectrum spectrum, float wavelengthNm);
float EvaluateRgbIlluminant(RgbSpectrum spectrum, float wavelengthNm);
RgbSpectrum FitSceneSpectrum(const SpectralSceneData::Request&, const RgbFitTable&);
} // namespace Arvo::RayTracing::Cpu
