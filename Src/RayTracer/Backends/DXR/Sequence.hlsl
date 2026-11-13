// DXR ray generation owns its sample state; hit shaders never advance a stream.
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

static const uint IndependentSequence = 0u, StratifiedSequence = 1u;
static const uint SobolSequence = 2u, BlueNoiseSequence = 3u;

SampleContext MakeSampleContext(uint x, uint y, uint imageWidth, uint sample, uint count, uint sequence,
                              uint samplingSeed)
{
    SampleContext context;
    context.PixelX = x; context.PixelY = y;
    context.PixelSeed = (y * imageWidth + x) * PixelSeedMultiplier;
    if (samplingSeed != 0)
        context.PixelSeed = Hash(context.PixelSeed ^ Hash(samplingSeed));
    context.SampleIndex = sample; context.SampleCount = count; context.Sequence = sequence;
    // The same offline/preview film key is used in every ray-generation dispatch.
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
    default: return float2(0, 0); // Host validation rejects this case.
    }
}
