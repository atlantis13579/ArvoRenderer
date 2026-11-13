#pragma once
#include "Renderer/RayTracing.h"
#include "Renderer/Reconstruction.h"

namespace Arvo::RayTracing::AdaptiveReconstructionData
{
// Upload schema only. Each backend independently estimates filtered risk and
// selects production counts. The two equal-size pilots are never production.
struct alignas(16) Pixel
{
    Float3 MeanA;
    float M2A;
    Float3 MeanB;
    float M2B;
    Float3 FilteredA;
    uint32_t SamplesA;
    Float3 FilteredB;
    uint32_t SamplesB;
};
struct alignas(16) Controls
{
    uint32_t Begin = 0, Count = 0, MinimumSamples = 16, MaximumSamples = 64;
    float RelativeError = 0.05f, AbsoluteError = 0.001f, MinimumVarianceFraction = 0.1f;
    uint32_t Sequence = uint32_t(SampleSequence::Independent);
};
static_assert(sizeof(Pixel) == 64 && offsetof(Pixel, SamplesB) == 60);
static_assert(sizeof(Controls) == 32 && offsetof(Controls, Sequence) == 28);
} // namespace Arvo::RayTracing::AdaptiveReconstructionData

namespace Arvo::RayTracing
{
bool ValidateCameraSamplePlan(const Camera&, const RenderSettings&, std::string&);
namespace Detail
{
// Only packing, crop scheduling and publication are shared. Allocation arithmetic
// and pilot/production transport execute in the selected backend.
using AdaptiveAllocationFunction =
    std::function<bool(const std::vector<AdaptiveReconstructionData::Pixel>&,
                       AdaptiveReconstructionData::Controls, std::vector<uint32_t>&, std::string&)>;
RenderResult RenderWithDenoisedSampling(const Scene &, const Camera &, const RenderSettings &,
                                        const CancellationProbe &, const RawRenderFunction &,
                                        const ReconstructionFunction &, const AdaptiveAllocationFunction &);
} // namespace Detail
} // namespace Arvo::RayTracing
