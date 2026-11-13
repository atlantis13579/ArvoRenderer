#include "RayTracer/Backends/CPU/AdaptiveDenoise.h"
#include <algorithm>
#include <cmath>

namespace Arvo::RayTracing::Cpu
{
// Rousselle, Knaus and Zwicker (2012), "Adaptive Rendering with Non-Local Means
// Filtering", https://www.cs.umd.edu/~zwicker/publications/AdaptiveRenderingNLM-SIGA12.pdf.
// Inspired by independent buffer error estimation; this is an allocation rule
// for our a-trous filter, not a reproduction of that paper's NLM algorithm.
// Var(F) is estimated by (F(A)-F(B))^2/2. Cross residuals estimate squared bias.
// Their positive sum earns bounded variance credit; detail loss restores raw
// allocation. The floor limits optimistic credit from only two observations.
// Neither the 1/N prediction nor its bias estimate is a confidence guarantee.
uint32_t DenoisedSampleBudget(const AdaptiveReconstructionData::Pixel& p,
                              const AdaptiveReconstructionData::Controls& c)
{
    const auto luma = [](Float3 value) {
        constexpr double Red = 0.2126, Green = 0.7152, Blue = 0.0722;
        return double(value.X) * Red + double(value.Y) * Green + double(value.Z) * Blue;
    };
    const auto valid = [](Float3 value) {
        return std::isfinite(value.X) && std::isfinite(value.Y) && std::isfinite(value.Z) && value.X >= 0 &&
               value.Y >= 0 && value.Z >= 0;
    };
    if (p.SamplesA < 2 || p.SamplesA != p.SamplesB || !valid(p.MeanA) || !valid(p.MeanB) ||
        !valid(p.FilteredA) || !valid(p.FilteredB) || !std::isfinite(p.M2A) || !std::isfinite(p.M2B) ||
        p.M2A < 0 || p.M2B < 0)
        return c.MaximumSamples;
    const double rawVariance = (double(p.M2A) + double(p.M2B)) / (2 * double(p.SamplesA - 1));
    const double meanA = luma(p.MeanA), meanB = luma(p.MeanB);
    const double filteredA = luma(p.FilteredA), filteredB = luma(p.FilteredB);
    const double difference = filteredA - filteredB;
    const double filteredVariance = difference * difference / 2;
    const double biasSquared = std::max(0.0, (filteredA - meanA) * (filteredB - meanB));
    const double predictedVariance = double(p.SamplesA) * (filteredVariance + biasSquared);
    const double allocationVariance =
        std::clamp(predictedVariance, rawVariance * double(c.MinimumVarianceFraction), rawVariance);
    const double tolerance = double(c.AbsoluteError) + double(c.RelativeError) * (meanA + meanB) / 2;
    const double required = allocationVariance / (tolerance * tolerance);
    if (!std::isfinite(required))
        return c.MaximumSamples;
    uint32_t budget =
        uint32_t(std::ceil(std::clamp(required, double(c.MinimumSamples), double(c.MaximumSamples))));
    if (c.Sequence == uint32_t(SampleSequence::Sobol) || c.Sequence == uint32_t(SampleSequence::BlueNoise))
    {
        uint32_t completeNet = 1;
        while (completeNet < budget)
            completeNet *= 2;
        budget = std::min(completeNet, c.MaximumSamples);
    }
    return budget;
}
} // namespace Arvo::RayTracing::Cpu
