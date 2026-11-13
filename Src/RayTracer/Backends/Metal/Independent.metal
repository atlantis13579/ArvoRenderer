// A fixed counter stream preserves sample identity under Metal command scheduling.
float2 IndependentSample(SampleContext context, uint dimension)
{
    const uint counter = (context.PixelSeed + context.SampleIndex * SampleSeedMultiplier) ^ dimension;
    return float2(Random01(counter), Random01(counter ^ SecondDimensionScramble));
}
