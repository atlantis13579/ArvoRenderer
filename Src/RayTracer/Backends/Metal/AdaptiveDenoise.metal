// Independent Metal allocation with variance-normalized arithmetic. No double
// support is assumed. Rousselle et al. (2012) motivate independent pilot pairs;
// our bounded a-trous error heuristic preserves production film and its moments.
#include <metal_stdlib>
using namespace metal;
struct AllocationPilot
{
    packed_float3 MeanA; float M2A;
    packed_float3 MeanB; float M2B;
    packed_float3 FilteredA; uint SamplesA;
    packed_float3 FilteredB; uint SamplesB;
};
struct AllocationSettings
{
    uint Begin, Count, MinimumSamples, MaximumSamples;
    float RelativeError, AbsoluteError, MinimumVarianceFraction;
    uint Sequence;
};
static_assert(sizeof(AllocationPilot) == 64 && sizeof(AllocationSettings) == 32, "Metal allocation ABI");
constant uint AllocationSobol = 2, AllocationBlueNoise = 3;
float AllocationLuma(float3 value)
{
    const float Red = 0.2126f, Green = 0.7152f, Blue = 0.0722f;
    return value.x * Red + value.y * Green + value.z * Blue;
}
bool AllocationValid(float3 value) { return all(isfinite(value)) && all(value >= 0); }
uint FilteredAllocation(AllocationPilot p, AllocationSettings c)
{
    if (p.SamplesA < 2 || p.SamplesA != p.SamplesB || !AllocationValid(float3(p.MeanA)) ||
        !AllocationValid(float3(p.MeanB)) || !AllocationValid(float3(p.FilteredA)) ||
        !AllocationValid(float3(p.FilteredB)) || !isfinite(p.M2A) || !isfinite(p.M2B) || p.M2A < 0 || p.M2B < 0)
        return c.MaximumSamples;
    float required = 0;
    const float largestMoment = max(p.M2A, p.M2B);
    if (largestMoment > 0)
    {
        // Scale before averaging, and take roots before dividing. This retains
        // subnormal M2 without overflowing an HDR sum or squaring huge colors.
        const float deviation = sqrt(largestMoment) *
            sqrt((p.M2A / largestMoment + p.M2B / largestMoment) / (2 * float(p.SamplesA - 1)));
        const float a = AllocationLuma(float3(p.MeanA)), b = AllocationLuma(float3(p.MeanB));
        const float fa = AllocationLuma(float3(p.FilteredA)), fb = AllocationLuma(float3(p.FilteredB));
        if (!isfinite(a) || !isfinite(b) || !isfinite(fa) || !isfinite(fb) || deviation <= 0)
            return c.MaximumSamples;
        const float difference = (fa - fb) / deviation;
        const float residualA = fa - a, residualB = fb - b;
        float biasFraction = 0;
        if (residualA != 0 && residualB != 0 && (residualA > 0) == (residualB > 0))
            biasFraction = (abs(residualA) / deviation) * (abs(residualB) / deviation) * float(p.SamplesA);
        const float noiseFraction = difference * difference * (float(p.SamplesA) / 2);
        const float credit = clamp(noiseFraction + biasFraction, c.MinimumVarianceFraction, 1.0f);
        const float tolerance = c.AbsoluteError + c.RelativeError * (a / 2 + b / 2);
        const float errorRatio = deviation * sqrt(credit) / tolerance;
        required = errorRatio * errorRatio;
        if (!isfinite(required)) return c.MaximumSamples;
    }
    uint count = uint(ceil(clamp(required, float(c.MinimumSamples), float(c.MaximumSamples))));
    if (c.Sequence == AllocationSobol || c.Sequence == AllocationBlueNoise)
    {
        uint net = 1;
        while (net < count) net *= 2;
        count = min(net, c.MaximumSamples);
    }
    return count;
}
kernel void ArvoAdaptiveDenoise(const device AllocationPilot* pilots [[buffer(0)]],
                              device uint* counts [[buffer(1)]],
                              constant AllocationSettings& settings [[buffer(2)]],
                              uint local [[thread_position_in_grid]])
{
    if (local < settings.Count)
    {
        uint index = settings.Begin + local;
        counts[index] = FilteredAllocation(pilots[index], settings);
    }
}
