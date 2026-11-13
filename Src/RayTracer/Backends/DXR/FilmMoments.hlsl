// DXR owns its linear film separately from the display UAV. Welford (1962)
// tracks central luminance deviations without subtracting large squared sums.
struct LinearFilmPixel
{
    float3 MeanRadiance;
    float LuminanceM2;
    uint SampleCount;
};

void AccumulateFilmSample(inout LinearFilmPixel pixel, float3 radiance)
{
    static const float3 LinearSrgbLuminance = float3(0.2126f, 0.7152f, 0.0722f);
    if (pixel.SampleCount == 0u)
    {
        pixel.MeanRadiance = radiance;
        pixel.LuminanceM2 = 0.0f;
        pixel.SampleCount = 1u;
        return;
    }
    const float3 delta = radiance - pixel.MeanRadiance;
    const float deviation = dot(delta, LinearSrgbLuminance);
    const float oldCount = float(pixel.SampleCount);
    ++pixel.SampleCount;
    pixel.MeanRadiance += delta / float(pixel.SampleCount);
    pixel.LuminanceM2 += deviation * deviation * (oldCount / float(pixel.SampleCount));
}
