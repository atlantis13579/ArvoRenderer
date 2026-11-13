// Native compute sampler interface. Integrators select a sequence without
// including one another; algorithm files contain only their own estimator.
struct SampleContext
{
    uint PixelX, PixelY, PixelSeed;
    uint SampleIndex, SampleCount, Sequence;
    uint SamplingSeed;
};

#include "Independent.hlsl"
#include "Stratified.hlsl"
#include "Sobol.hlsl"
#include "BlueNoise.hlsl"

static const uint IndependentSequence = 0u;
static const uint StratifiedSequence = 1u;
static const uint SobolSequence = 2u;
static const uint BlueNoiseSequence = 3u;

SampleContext MakeSampleContext(uint x, uint y, uint imageWidth, uint sample, uint count, uint sequence,
                              uint samplingSeed)
{
    SampleContext context;
    context.PixelX = x; context.PixelY = y;
    context.PixelSeed = (y * imageWidth + x) * PixelSeedMultiplier;
    if (samplingSeed != 0u)
        context.PixelSeed = Hash(context.PixelSeed ^ Hash(samplingSeed));
    context.SampleIndex = sample; context.SampleCount = count; context.Sequence = sequence;
    // Shared across pixels so that blue-noise hierarchy coverage survives.
    context.SamplingSeed = samplingSeed;
    return context;
}

float2 Sample2D(SampleContext context, uint dimension)
{
    switch (context.Sequence)
    {
    case IndependentSequence: return IndependentSample(context, dimension);
    case StratifiedSequence: return StratifiedSample(context, dimension);
    case SobolSequence: return SobolSample(context, dimension);
    case BlueNoiseSequence: return BlueNoiseSample(context, dimension);
    default: return float2(0.0f, 0.0f); // The host rejects invalid identifiers before dispatch.
    }
}
