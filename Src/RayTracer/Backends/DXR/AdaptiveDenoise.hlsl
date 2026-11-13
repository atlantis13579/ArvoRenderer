// Independent DXR offline ray-generation allocation; preview uses a fixed budget.
// Split-pilot risk estimation is inspired by Rousselle/Knaus/Zwicker (2012).
// Positive cross residuals guard against filter bias; bounded variance credit
// changes future sample counts, never the pilots or an accepted linear film.
struct DenoisingPilot
{
    float3 FirstMean; float FirstM2;
    float3 SecondMean; float SecondM2;
    float3 FirstFiltered; uint FirstCount;
    float3 SecondFiltered; uint SecondCount;
};
struct DenoisingAllocation
{
    uint Begin, Count, MinimumSamples, MaximumSamples;
    float RelativeError, AbsoluteError, MinimumVarianceFraction;
    uint Sequence;
};
static const uint DenoisingSobol = 2, DenoisingBlueNoise = 3;
double DenoisingBrightness(float3 rgb)
{
    static const double Red = 0.2126, Green = 0.7152, Blue = 0.0722;
    return double(rgb.x) * Red + double(rgb.y) * Green + double(rgb.z) * Blue;
}
bool DenoisingFinite(float3 rgb) { return all(isfinite(rgb)) && all(rgb >= 0); }
uint DenoisingProductionCount(DenoisingPilot p, DenoisingAllocation c)
{
    if (p.FirstCount < 2 || p.FirstCount != p.SecondCount ||
        !DenoisingFinite(p.FirstMean) || !DenoisingFinite(p.SecondMean) ||
        !DenoisingFinite(p.FirstFiltered) || !DenoisingFinite(p.SecondFiltered) ||
        !isfinite(p.FirstM2) || !isfinite(p.SecondM2) || p.FirstM2 < 0 || p.SecondM2 < 0)
        return c.MaximumSamples;
    double rawVariance = (double(p.FirstM2) + double(p.SecondM2)) / (2 * double(p.FirstCount - 1));
    double first = DenoisingBrightness(p.FirstMean), second = DenoisingBrightness(p.SecondMean);
    double firstFiltered = DenoisingBrightness(p.FirstFiltered);
    double secondFiltered = DenoisingBrightness(p.SecondFiltered);
    double disagreement = firstFiltered - secondFiltered;
    double systematic = max(0.0, (firstFiltered - first) * (secondFiltered - second));
    double prediction = double(p.FirstCount) * (disagreement * disagreement / 2 + systematic);
    double credited = max(rawVariance * double(c.MinimumVarianceFraction), min(rawVariance, prediction));
    double allowed = double(c.AbsoluteError) + double(c.RelativeError) * (first + second) / 2;
    double needed = credited / (allowed * allowed);
    if (needed != needed) return c.MaximumSamples;
    const double bounded = clamp(needed, double(c.MinimumSamples), double(c.MaximumSamples));
    // Keep the bounded decision in double; HLSL ceil would narrow it first.
    uint count = uint(bounded);
    if (double(count) < bounded) ++count;
    if (c.Sequence == DenoisingSobol || c.Sequence == DenoisingBlueNoise)
    {
        uint block = 1;
        while (block < count) block *= 2;
        count = min(block, c.MaximumSamples);
    }
    return count;
}
StructuredBuffer<DenoisingPilot> PilotPairs : register(t0);
RWStructuredBuffer<uint> ProductionCounts : register(u0);
cbuffer Parameters : register(b0) { DenoisingAllocation Allocation; }
[shader("raygeneration")]
void AdaptiveDenoiseRayGen()
{
    uint local = DispatchRaysIndex().x;
    if (local < Allocation.Count)
    {
        uint address = Allocation.Begin + local;
        ProductionCounts[address] = DenoisingProductionCount(PilotPairs[address], Allocation);
    }
}
