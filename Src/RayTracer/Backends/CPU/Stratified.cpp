#include "RayTracer/Backends/CPU/CpuTransport.h"
#include <cmath>
#include <algorithm>

namespace Arvo::RayTracing::Cpu
{
namespace
{
uint32_t GreatestCommonDivisor(uint32_t a, uint32_t b)
{
    while (b) { const uint32_t remainder = a % b; a = b; b = remainder; }
    return a;
}
}

// Jitter one point in each equal-area cell. Ragged rows have height columns/N,
// so prime sample counts also cover the entire square without unequal weights.
// Independent affine permutations associate the 2D strata across path domains.
// See PBRT 4e, "Stratified Sampler" (padding low-dimensional sample domains).
Float2 StratifiedSample(const SampleContext& context, uint32_t dimension)
{
    constexpr float OneMinusEpsilon = 1.0f - 1.0f / 16777216.0f;
    const uint32_t count = context.SampleCount;
    const uint32_t key = Hash(context.PixelSeed ^ dimension);
    uint32_t multiplier = key % count;
    while (GreatestCommonDivisor(multiplier, count) != 1u) multiplier = (multiplier + 1u) % count;
    const uint32_t stratum = (context.SampleIndex * multiplier + Hash(key) % count) % count;
    const uint32_t rows = static_cast<uint32_t>(std::sqrt(static_cast<float>(count)));
    const uint32_t columns = count / rows, wideRows = count % rows;
    const uint32_t wideCells = wideRows * (columns + 1u);
    const bool wide = stratum < wideCells;
    const uint32_t rowColumns = columns + (wide ? 1u : 0u);
    const uint32_t offset = wide ? stratum : stratum - wideCells;
    const uint32_t row = offset / rowColumns, column = offset % rowColumns;
    const uint32_t precedingCells = (wide ? 0u : wideCells) + row * rowColumns;
    const uint32_t seed = (context.PixelSeed + context.SampleIndex * SampleSeedMultiplier) ^ dimension;
    return {std::min(OneMinusEpsilon, (static_cast<float>(column) + Random01(seed)) / rowColumns),
        std::min(OneMinusEpsilon, (static_cast<float>(precedingCells) + Random01(seed ^ SecondDimensionScramble) * rowColumns) / count)};
}
}
