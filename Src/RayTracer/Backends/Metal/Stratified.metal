uint StratifiedGcd(uint a, uint b)
{
    while (b != 0u)
    {
        const uint remainder = a % b;
        a = b; b = remainder;
    }
    return a;
}

// Native MSL equal-area stratification with independently shuffled domains.
// Ragged row heights compensate their different column counts for arbitrary N.
float2 StratifiedSample(SampleContext context, uint dimension)
{
    const float OneMinusEpsilon = 1.0f - 1.0f / 16777216.0f;
    const uint count = context.SampleCount;
    const uint key = Hash(context.PixelSeed ^ dimension);
    uint multiplier = key % count;
    while (StratifiedGcd(multiplier, count) != 1u) multiplier = (multiplier + 1u) % count;
    const uint stratum = (context.SampleIndex * multiplier + Hash(key) % count) % count;
    const uint rows = uint(sqrt(float(count))), columns = count / rows;
    const uint wideCells = (count % rows) * (columns + 1u);
    const bool wide = stratum < wideCells;
    const uint rowColumns = columns + (wide ? 1u : 0u);
    const uint offset = wide ? stratum : stratum - wideCells;
    const uint preceding = (wide ? 0u : wideCells) + (offset / rowColumns) * rowColumns;
    const uint counter = (context.PixelSeed + context.SampleIndex * SampleSeedMultiplier) ^ dimension;
    return min(float2(OneMinusEpsilon), float2((float(offset % rowColumns) + Random01(counter)) / rowColumns,
        (float(preceding) + Random01(counter ^ SecondDimensionScramble) * rowColumns) / count));
}
