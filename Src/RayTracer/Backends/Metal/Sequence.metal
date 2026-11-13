// Each native Metal integrator receives the same sample-domain interface.
struct SampleContext
{
    uint PixelX, PixelY, PixelSeed;
    uint SampleIndex, SampleCount, Sequence;
    uint SamplingSeed;
};

#include "Independent.metal"
#include "Stratified.metal"
#include "Sobol.metal"
#include "BlueNoise.metal"

constant uint IndependentSequence = 0u, StratifiedSequence = 1u;
constant uint SobolSequence = 2u, BlueNoiseSequence = 3u;

SampleContext MakeSampleContext(uint x, uint y, uint imageWidth, uint sample, uint count, uint sequence,
                              uint samplingSeed)
{
    uint pixelSeed = (y * imageWidth + x) * PixelSeedMultiplier;
    if (samplingSeed)
        pixelSeed = Hash(pixelSeed ^ Hash(samplingSeed));
    // A separate film-wide seed preserves the blue-noise hierarchy between pixels.
    return SampleContext{x, y, pixelSeed, sample, count, sequence, samplingSeed};
}

float2 Sample2D(SampleContext context, uint dimension)
{
    switch (context.Sequence)
    {
    case IndependentSequence: return IndependentSample(context, dimension);
    case StratifiedSequence: return StratifiedSample(context, dimension);
    case SobolSequence: return SobolSample(context, dimension);
    case BlueNoiseSequence: return BlueNoiseSample(context, dimension);
    default: return float2(0.0f); // Invalid selections are rejected before dispatch.
    }
}
