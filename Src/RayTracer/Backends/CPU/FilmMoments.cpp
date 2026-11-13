#include "RayTracer/Backends/CPU/CpuTransport.h"

namespace Arvo::RayTracing::Cpu
{
// Welford (1962), "Note on a Method for Calculating Corrected Sums of Squares
// and Products", https://doi.org/10.1080/00401706.1962.10490022.
// Keep a running RGB mean and the luminance central moment, never E[L^2]-E[L]^2.
// The rank-one M2 update avoids subtracting nearly equal, bright luminances.
void AccumulateFilmSample(FilmPixel& pixel, const Float3& radiance)
{
    const Float3 LinearSrgbLuminance{0.2126f, 0.7152f, 0.0722f};
    if (pixel.SampleCount == 0)
    {
        pixel.MeanRadiance = radiance;
        pixel.LuminanceM2 = 0.0f;
        pixel.SampleCount = 1;
        return;
    }
    const Float3 delta = radiance - pixel.MeanRadiance;
    const float luminanceDelta = Dot(delta, LinearSrgbLuminance);
    const float oldCount = static_cast<float>(pixel.SampleCount);
    const float newCount = static_cast<float>(++pixel.SampleCount);
    pixel.MeanRadiance = pixel.MeanRadiance + delta / newCount;
    pixel.LuminanceM2 += luminanceDelta * luminanceDelta * (oldCount / newCount);
}
}
