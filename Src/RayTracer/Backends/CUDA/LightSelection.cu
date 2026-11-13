#include <cuda_runtime.h>
namespace ArvoCudaSelection
{
// Independent CUDA uniform source PMF. PBRT 4e, "Light Sampling".
// Endpoint flux / PMF and endpoint density * PMF are the two uses; this
// discrete source probability must not be mistaken for an area/angle PDF.
constexpr unsigned MaximumSources=1u<<24; // Binary32 significand bits.
constexpr unsigned InvalidSource=0xffffffffu;
struct Selection { unsigned Index; float Probability; };
__device__ Selection Select(unsigned count,float u)
{
    Selection selection{InvalidSource,0};
    if (!count || count > MaximumSources || !(u >= 0 && u < 1)) return selection;
    unsigned index=unsigned(u*float(count));
    selection.Index=index < count ? index : count-1;
    selection.Probability=1.0f/float(count);
    return selection;
}
__device__ float Probability(unsigned count,unsigned index)
{
    if (count > MaximumSources || index >= count) return 0;
    return 1.0f/float(count);
}
} // namespace ArvoCudaSelection
