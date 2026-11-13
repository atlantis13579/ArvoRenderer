#include "RayTracer/Backends/CPU/CpuTransport.h"
#include <stdexcept>

namespace Arvo::RayTracing::Cpu
{
// A dimension identifies a random variable, not the order in which a branch
// happens to request it. Absolute coordinates preserve tile/worker identity.
SampleContext MakeSampleContext(const RenderSettings& settings, uint32_t x, uint32_t y, uint32_t sample)
{
    const uint32_t pixelX = x + settings.TileX, pixelY = y + settings.TileY;
    uint32_t pixelSeed = (pixelY * FullImageWidth(settings) + pixelX) * PixelSeedMultiplier;
    if (settings.SamplingSeed != 0)
        pixelSeed = Hash(pixelSeed ^ Hash(settings.SamplingSeed));
    constexpr uint32_t DenoisePilotAScramble = 0x68bc21ebu, DenoisePilotBScramble = 0x02e5be93u;
    if (settings.SampleStream == CameraSampleStream::DenoisePilotA)
        pixelSeed = Hash(pixelSeed ^ DenoisePilotAScramble);
    else if (settings.SampleStream == CameraSampleStream::DenoisePilotB)
        pixelSeed = Hash(pixelSeed ^ DenoisePilotBScramble);
    // Blue noise needs a film-wide scramble; a per-pixel seed would destroy its
    // spatial ordering. Other sequences use the independently scrambled pixel.
    return {pixelX, pixelY, pixelSeed, sample, settings.SamplesPerPixel,
            settings.Sampling, settings.SamplingSeed};
}

Float2 Sample2D(const SampleContext& context, uint32_t dimension)
{
    switch (context.Sequence)
    {
    case SampleSequence::Independent: return IndependentSample(context, dimension);
    case SampleSequence::Stratified: return StratifiedSample(context, dimension);
    case SampleSequence::Sobol: return SobolSample(context, dimension);
    case SampleSequence::BlueNoise: return BlueNoiseSample(context, dimension);
    default: throw std::invalid_argument("Unknown CPU sampling sequence");
    }
}
}
