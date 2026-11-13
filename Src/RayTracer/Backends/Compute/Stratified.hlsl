uint StratifiedGcd(uint a, uint b)
{
    [loop] while (b != 0u) { const uint remainder = a % b; a = b; b = remainder; }
    return a;
}

// Equal-area ragged rows support non-square and prime sample counts. Each
// domain permutes the cells, then jitters within its assigned cell (PBRT 4e).
float2 StratifiedSample(SampleContext context, uint dimension)
{
    const float OneMinusEpsilon = 1.0f - 1.0f / 16777216.0f;
    const uint count = context.SampleCount;
    const uint key = Hash(context.PixelSeed ^ dimension);
    uint multiplier = key % count;
    [loop] while (StratifiedGcd(multiplier, count) != 1u) multiplier = (multiplier + 1u) % count;
    const uint stratum = (context.SampleIndex * multiplier + Hash(key) % count) % count;
    const uint rows = uint(sqrt(float(count)));
    const uint columns = count / rows, wideRows = count % rows;
    const uint wideCells = wideRows * (columns + 1u);
    const bool wide = stratum < wideCells;
    const uint rowColumns = columns + (wide ? 1u : 0u);
    const uint offset = wide ? stratum : stratum - wideCells;
    const uint row = offset / rowColumns, column = offset % rowColumns;
    const uint precedingCells = (wide ? 0u : wideCells) + row * rowColumns;
    const uint seed = (context.PixelSeed + context.SampleIndex * SampleSeedMultiplier) ^ dimension;
    return min(float2(OneMinusEpsilon, OneMinusEpsilon), float2((float(column) + Random01(seed)) / rowColumns,
        (float(precedingCells) + Random01(seed ^ SecondDimensionScramble) * rowColumns) / count));
}
