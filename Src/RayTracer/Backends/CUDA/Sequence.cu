// Native CUDA dispatch over independent sampler implementations.
struct SampleContext
{
    unsigned int PixelX, PixelY, PixelSeed;
    unsigned int SampleIndex, SampleCount, Sequence;
    unsigned int SamplingSeed;
};

#include "Independent.cu"
#include "Stratified.cu"
#include "Sobol.cu"
#include "BlueNoise.cu"

constexpr unsigned int IndependentSequence = 0u;
constexpr unsigned int StratifiedSequence = 1u;
constexpr unsigned int SobolSequence = 2u;
constexpr unsigned int BlueNoiseSequence = 3u;

__device__ SampleContext MakeSampleContext(unsigned int x, unsigned int y, unsigned int imageWidth,
    unsigned int sample, unsigned int count, unsigned int sequence, unsigned int samplingSeed)
{
    unsigned int pixelSeed = (y * imageWidth + x) * PixelSeedMultiplier;
    if (samplingSeed != 0u)
        pixelSeed = Hash(pixelSeed ^ Hash(samplingSeed));
    // Preserve absolute raster identity and a common blue-noise frame scramble.
    return {x, y, pixelSeed, sample, count, sequence, samplingSeed};
}

__device__ float2 Sample2D(SampleContext context, unsigned int dimension)
{
    switch (context.Sequence)
    {
    case IndependentSequence: return IndependentSample(context, dimension);
    case StratifiedSequence: return StratifiedSample(context, dimension);
    case SobolSequence: return SobolSample(context, dimension);
    case BlueNoiseSequence: return BlueNoiseSample(context, dimension);
    default: return make_float2(0.0f, 0.0f); // Invalid settings are rejected on the host.
    }
}
