// Counter-based independent samples retain the original native compute stream.
float2 IndependentSample(SampleContext context, uint dimension)
{
    const uint seed = (context.PixelSeed + context.SampleIndex * SampleSeedMultiplier) ^ dimension;
    return float2(Random01(seed), Random01(seed ^ SecondDimensionScramble));
}
