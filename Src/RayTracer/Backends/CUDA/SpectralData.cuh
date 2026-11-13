#pragma once
#include <cuda_runtime.h>
#include <cmath>
constexpr unsigned SpectralPacketSize = 4;
constexpr float VisibleMinimumNm = 360, VisibleMaximumNm = 830;
struct WavelengthPacket
{
    float Lambda[SpectralPacketSize];
    float Pdf[SpectralPacketSize];
};
struct SampledSpectrum
{
    float Value[SpectralPacketSize];
};
constexpr unsigned RgbFitSampleCount = 256;
constexpr unsigned SpectrumFitConverged = 0, SpectrumFitInvalid = 1, SpectrumFitUnconverged = 2;
struct RgbSpectrum
{
    // Constant + Scale * sigmoid(quadratic); Error is normalized chromatic residual.
    float Quadratic, Linear, Offset, Scale;
    float Constant, Error;
    unsigned Status, Iterations;
};
constexpr unsigned SpectrumRoleReflectance = 0, SpectrumRoleIlluminant = 1, SpectrumRoleNonnegative = 2;
struct SpectralSource
{
    float Red, Green, Blue;
    unsigned Kind;
};
struct SpectralTextureDescriptor
{
    unsigned FirstTexel, Width, Height, Reserved;
};
