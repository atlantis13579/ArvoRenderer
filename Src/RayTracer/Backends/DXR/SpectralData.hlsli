#ifndef ARVO_DXR_SPECTRAL_DATA
#define ARVO_DXR_SPECTRAL_DATA
static const uint SpectralPacketSize = 4;
static const float VisibleMinimumNm = 360, VisibleMaximumNm = 830;
struct WavelengthPacket
{
    float Lambda[SpectralPacketSize];
    float Pdf[SpectralPacketSize];
};
struct SampledSpectrum
{
    float Value[SpectralPacketSize];
};
static const uint RgbFitSampleCount = 256;
static const uint SpectrumFitConverged = 0, SpectrumFitInvalid = 1, SpectrumFitUnconverged = 2;
struct RgbSpectrum
{
    // Constant + Scale * sigmoid(quadratic); Error is normalized chromatic residual.
    float Quadratic, Linear, Offset, Scale;
    float Constant, Error;
    uint Status, Iterations;
};
static const uint SpectrumRoleReflectance = 0, SpectrumRoleIlluminant = 1, SpectrumRoleNonnegative = 2;
struct SpectralSource
{
    float Red, Green, Blue;
    uint Kind;
};
struct SpectralTextureDescriptor
{
    uint FirstTexel, Width, Height, Reserved;
};
#endif
