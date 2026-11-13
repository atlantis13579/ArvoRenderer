// Independent Compute allocation from two filtered, independent pilots.
// Rousselle et al. (2012), "Adaptive Rendering with Non-Local Means Filtering".
// This a-trous risk heuristic grants bounded variance credit, with a positive
// cross-residual bias guard. Raw film moments and production RNG are untouched.
struct AdaptivePilotPair
{
    float3 MeanA; float M2A;
    float3 MeanB; float M2B;
    float3 FilteredA; uint SamplesA;
    float3 FilteredB; uint SamplesB;
};
struct AdaptiveFilterControls
{
    uint Begin, Count, MinimumSamples, MaximumSamples;
    float RelativeError, AbsoluteError, MinimumVarianceFraction;
    uint Sequence;
};
static const uint AdaptiveSobol = 2, AdaptiveBlueNoise = 3, AdaptiveThreads = 64;
double AdaptiveLuminance(float3 color)
{
    static const double Red = 0.2126, Green = 0.7152, Blue = 0.0722;
    return double(color.x) * Red + double(color.y) * Green + double(color.z) * Blue;
}
bool AdaptiveValidColor(float3 color) { return all(isfinite(color)) && all(color >= 0); }
uint AdaptiveFilteredBudget(AdaptivePilotPair p, AdaptiveFilterControls c)
{
    if (p.SamplesA < 2 || p.SamplesA != p.SamplesB || !AdaptiveValidColor(p.MeanA) ||
        !AdaptiveValidColor(p.MeanB) || !AdaptiveValidColor(p.FilteredA) ||
        !AdaptiveValidColor(p.FilteredB) || !isfinite(p.M2A) || !isfinite(p.M2B) || p.M2A < 0 || p.M2B < 0)
        return c.MaximumSamples;
    const double variance = (double(p.M2A) + double(p.M2B)) / (2 * double(p.SamplesA - 1));
    const double a = AdaptiveLuminance(p.MeanA), b = AdaptiveLuminance(p.MeanB);
    const double fa = AdaptiveLuminance(p.FilteredA), fb = AdaptiveLuminance(p.FilteredB);
    const double noise = (fa - fb) * (fa - fb) / 2;
    const double bias = max(0.0, (fa - a) * (fb - b));
    const double residual = clamp(double(p.SamplesA) * (noise + bias),
                                  variance * double(c.MinimumVarianceFraction), variance);
    const double threshold = double(c.AbsoluteError) + double(c.RelativeError) * (a + b) / 2;
    const double estimate = residual / (threshold * threshold);
    if (estimate != estimate) return c.MaximumSamples;
    const double bounded = clamp(estimate, double(c.MinimumSamples), double(c.MaximumSamples));
    // HLSL ceil/isfinite narrow doubles to float. Round only after bounding,
    // using an integer comparison to preserve near-integer budget decisions.
    uint result = uint(bounded);
    if (double(result) < bounded) ++result;
    if (c.Sequence == AdaptiveSobol || c.Sequence == AdaptiveBlueNoise)
    {
        uint net = 1;
        while (net < result) net <<= 1;
        result = min(net, c.MaximumSamples);
    }
    return result;
}
StructuredBuffer<AdaptivePilotPair> Pilots : register(t0);
RWStructuredBuffer<uint> Budgets : register(u0);
cbuffer Parameters : register(b0) { AdaptiveFilterControls Controls; }
[numthreads(AdaptiveThreads, 1, 1)]
void AdaptiveDenoiseMain(uint3 thread : SV_DispatchThreadID)
{
    if (thread.x < Controls.Count)
    {
        uint index = Controls.Begin + thread.x;
        Budgets[index] = AdaptiveFilteredBudget(Pilots[index], Controls);
    }
}
