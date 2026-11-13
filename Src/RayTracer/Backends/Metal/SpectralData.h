#ifndef ARVO_METAL_SPECTRAL_DATA
#define ARVO_METAL_SPECTRAL_DATA
#include <metal_stdlib>
using namespace metal;
constant uint SpectralPacketSize = 4;
constant float VisibleMinimumNm = 360, VisibleMaximumNm = 830;
struct WavelengthPacket
{
    float Lambda[SpectralPacketSize];
    float Pdf[SpectralPacketSize];
};
struct SampledSpectrum
{
    float Value[SpectralPacketSize];
};
constant uint RgbFitSampleCount = 256;
constant uint SpectrumFitConverged = 0, SpectrumFitInvalid = 1, SpectrumFitUnconverged = 2;
struct RgbSpectrum
{
    // Constant + Scale * sigmoid(quadratic); Error is normalized chromatic residual.
    float Quadratic, Linear, Offset, Scale;
    float Constant, Error;
    uint Status, Iterations;
};
constant uint SpectrumRoleReflectance = 0, SpectrumRoleIlluminant = 1, SpectrumRoleNonnegative = 2;
struct SpectralSource
{
    float Red, Green, Blue;
    uint Kind;
};
struct SpectralTextureDescriptor
{
    uint FirstTexel, Width, Height, Reserved;
};

#endif // ARVO_METAL_SPECTRAL_DATA
