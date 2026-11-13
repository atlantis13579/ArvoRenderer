#include "RayTracer/Backends/CPU/CpuTransport.h"
#include <algorithm>
#include <cmath>

namespace Arvo::RayTracing::Cpu
{
// Kirk and Arvo (1991), "Unbiased Sampling Techniques for Image Synthesis".
// https://authors.library.caltech.edu/records/8sntk-73q87
// A fixed independent pilot estimates cost before production starts. Reusing
// those pilot samples in the final image would couple the estimate to its
// stopping decision. Production starts at sample zero in a separate stream.
SampleContext MakePilotSampleContext(const RenderSettings& settings, uint32_t x, uint32_t y, uint32_t sample)
{
    constexpr uint32_t AdaptivePilotScramble = 0x7f4a7c15u;
    SampleContext context = MakeSampleContext(settings, x, y, sample);
    context.PixelSeed = Hash(context.PixelSeed ^ AdaptivePilotScramble);
    context.SampleCount = settings.Adaptive.PilotSamples;
    context.Sequence = SampleSequence::Independent;
    return context;
}

// This is an RMS-error allocation heuristic, not a confidence guarantee. The
// independent pilot variance conservatively estimates MC cost even when the
// production sequence is QMC. Spatially varying blue-noise counts no longer
// guarantee the fixed-count neighborhood nets of the screen-space sampler.
uint32_t AdaptiveSampleCount(const FilmPixel& pilot, const RenderSettings& settings)
{
    const uint32_t maximum = settings.SamplesPerPixel;
    if (!settings.Adaptive.Enabled) return maximum;
    const Float3 LinearSrgbLuminance{0.2126f, 0.7152f, 0.0722f};
    if (pilot.SampleCount < 2 || !std::isfinite(pilot.LuminanceM2)
        || !std::isfinite(pilot.MeanRadiance.X) || !std::isfinite(pilot.MeanRadiance.Y)
        || !std::isfinite(pilot.MeanRadiance.Z)) return maximum;
    const float mean = Dot(pilot.MeanRadiance, LinearSrgbLuminance);
    const float threshold = settings.Adaptive.AbsoluteError + settings.Adaptive.RelativeError * std::fabs(mean);
    const float variance = std::max(0.0f, pilot.LuminanceM2) / static_cast<float>(pilot.SampleCount - 1);
    const float estimate = variance / (threshold * threshold);
    if (!std::isfinite(estimate)) return maximum;
    uint32_t count = static_cast<uint32_t>(std::ceil(std::clamp(estimate,
        static_cast<float>(settings.Adaptive.MinimumSamples), static_cast<float>(maximum))));
    if (settings.Sampling == SampleSequence::Sobol || settings.Sampling == SampleSequence::BlueNoise)
    {
        uint32_t netSize = 1;
        while (netSize < count) netSize *= 2;
        count = std::min(netSize, maximum);
    }
    return count;
}
}
