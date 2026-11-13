#include "Renderer/EmissionCatalog.h"
#include <algorithm>
#include <cmath>

namespace Arvo::RayTracing::Cpu
{
// Uniform discrete source selection. PBRT 4e, "Light Sampling":
// https://pbr-book.org/4ed/Light_Sources/Light_Sampling
// The chosen endpoint's conditional flux is divided by Probability. This PMF
// also multiplies its emission densities in path-space MIS. It is not a PDF
// over emitter area or directions. Black sources are removed by the catalog.
EmitterSelection SelectEmitter(uint32_t count, float variate)
{
    if (!count || count > MaximumUniformEmitters || !(variate >= 0 && variate < 1))
        return {};
    return {std::min(uint32_t(variate * float(count)), count - 1), 1.0f / float(count)};
}
float EmitterProbability(uint32_t count, uint32_t index)
{
    return count <= MaximumUniformEmitters && index < count ? 1.0f / float(count) : 0;
}
} // namespace Arvo::RayTracing::Cpu
