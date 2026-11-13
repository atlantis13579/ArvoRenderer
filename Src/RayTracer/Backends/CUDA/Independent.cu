// The counter identifies an absolute pixel, sample and random-variable domain.
__device__ float2 IndependentSample(SampleContext context, unsigned int dimension)
{
    const unsigned int seed = (context.PixelSeed + context.SampleIndex * SampleSeedMultiplier) ^ dimension;
    return make_float2(Random01(seed), Random01(seed ^ SecondDimensionScramble));
}
