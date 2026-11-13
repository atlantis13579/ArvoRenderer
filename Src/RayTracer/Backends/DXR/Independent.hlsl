// A counter stream is independent of DXR ray scheduling and shader invocation order.
float2 IndependentSample(SampleContext context, uint dimension)
{
    const uint counter = (context.PixelSeed + context.SampleIndex * SampleSeedMultiplier) ^ dimension;
    return float2(Random01(counter), Random01(counter ^ SecondDimensionScramble));
}
