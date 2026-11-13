__device__ unsigned int StratifiedGcd(unsigned int a, unsigned int b)
{
    while (b != 0u) { const unsigned int remainder = a % b; a = b; b = remainder; }
    return a;
}

// Native CUDA equal-area stratification. Wider rows are taller by the same
// factor, preserving one equal-probability cell per sample for arbitrary N.
__device__ float2 StratifiedSample(SampleContext context, unsigned int dimension)
{
    constexpr float OneMinusEpsilon = 1.0f - 1.0f / 16777216.0f;
    const unsigned int count = context.SampleCount;
    const unsigned int key = Hash(context.PixelSeed ^ dimension);
    unsigned int multiplier = key % count;
    while (StratifiedGcd(multiplier, count) != 1u) multiplier = (multiplier + 1u) % count;
    const unsigned int stratum = (context.SampleIndex * multiplier + Hash(key) % count) % count;
    const unsigned int rows = static_cast<unsigned int>(sqrtf(static_cast<float>(count)));
    const unsigned int columns = count / rows;
    const unsigned int wideCells = (count % rows) * (columns + 1u);
    const bool wide = stratum < wideCells;
    const unsigned int rowColumns = columns + (wide ? 1u : 0u);
    const unsigned int offset = wide ? stratum : stratum - wideCells;
    const unsigned int row = offset / rowColumns, column = offset % rowColumns;
    const unsigned int preceding = (wide ? 0u : wideCells) + row * rowColumns;
    const unsigned int seed = (context.PixelSeed + context.SampleIndex * SampleSeedMultiplier) ^ dimension;
    return make_float2(fminf(OneMinusEpsilon, (column + Random01(seed)) / rowColumns),
        fminf(OneMinusEpsilon, (preceding + Random01(seed ^ SecondDimensionScramble) * rowColumns) / count));
}
