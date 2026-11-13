// Independent CUDA allocation from paired a-trous pilot reconstructions.
// Rousselle et al. (2012), "Adaptive Rendering with Non-Local Means Filtering".
// A positive cross-residual bias term withholds credit at lost image detail.
// This bounded pilot heuristic is not a confidence bound or NLM implementation.
#include <cuda_runtime.h>
#include <cmath>
namespace
{
struct alignas(16) PilotPair
{
    float3 MeanA; float M2A;
    float3 MeanB; float M2B;
    float3 FilteredA; unsigned SamplesA;
    float3 FilteredB; unsigned SamplesB;
};
struct alignas(16) AllocationControls
{
    unsigned Begin, Count, MinimumSamples, MaximumSamples;
    float RelativeError, AbsoluteError, MinimumVarianceFraction;
    unsigned Sequence;
};
static_assert(sizeof(PilotPair) == 64 && sizeof(AllocationControls) == 32);
constexpr unsigned AllocationSobol = 2, AllocationBlueNoise = 3;
__device__ double PilotLuma(float3 value)
{
    constexpr double Red = 0.2126, Green = 0.7152, Blue = 0.0722;
    return double(value.x) * Red + double(value.y) * Green + double(value.z) * Blue;
}
__device__ bool ValidPilotColor(float3 value)
{
    return isfinite(value.x) && isfinite(value.y) && isfinite(value.z) && value.x >= 0 && value.y >= 0 && value.z >= 0;
}
__device__ unsigned FilteredSampleCount(PilotPair p, AllocationControls c)
{
    if (p.SamplesA < 2 || p.SamplesA != p.SamplesB || !ValidPilotColor(p.MeanA) || !ValidPilotColor(p.MeanB) ||
        !ValidPilotColor(p.FilteredA) || !ValidPilotColor(p.FilteredB) ||
        !isfinite(p.M2A) || !isfinite(p.M2B) || p.M2A < 0 || p.M2B < 0)
        return c.MaximumSamples;
    const double variance = (double(p.M2A) + double(p.M2B)) / (2 * double(p.SamplesA - 1));
    const double a = PilotLuma(p.MeanA), b = PilotLuma(p.MeanB);
    const double filteredA = PilotLuma(p.FilteredA), filteredB = PilotLuma(p.FilteredB);
    const double difference = filteredA - filteredB;
    const double bias = fmax(0.0, (filteredA - a) * (filteredB - b));
    const double predicted = double(p.SamplesA) * (difference * difference / 2 + bias);
    const double effective = fmin(variance, fmax(variance * double(c.MinimumVarianceFraction), predicted));
    const double target = double(c.AbsoluteError) + double(c.RelativeError) * (a + b) / 2;
    const double required = effective / (target * target);
    if (!isfinite(required)) return c.MaximumSamples;
    unsigned result = unsigned(ceil(fmin(double(c.MaximumSamples), fmax(double(c.MinimumSamples), required))));
    if (c.Sequence == AllocationSobol || c.Sequence == AllocationBlueNoise)
    {
        unsigned completeNet = 1;
        while (completeNet < result) completeNet <<= 1;
        result = completeNet < c.MaximumSamples ? completeNet : c.MaximumSamples;
    }
    return result;
}
} // namespace
extern "C" __global__ void ArvoAdaptiveDenoise(const PilotPair* pilots, unsigned* counts,
                                             AllocationControls controls)
{
    const unsigned local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local < controls.Count)
    {
        const unsigned index = controls.Begin + local;
        counts[index] = FilteredSampleCount(pilots[index], controls);
    }
}
