#pragma once
#include "Renderer/RayTracing.h"

namespace Arvo::RayTracing
{
// Upload data only. Each API independently clips segments, samples free paths,
// evaluates Beer transmittance and samples the Henyey-Greenstein phase function.
struct alignas(16) MediumData
{
    Float3 Minimum;
    uint32_t Enabled = 0;
    Float3 Maximum;
    float Anisotropy = 0;
    Float3 Absorption;
    uint32_t Padding0 = 0;
    Float3 Scattering;
    uint32_t Padding1 = 0;
};
static_assert(sizeof(MediumData) == 64 && offsetof(MediumData, Scattering) == 48);
MediumData PackMedium(const MediumOptions&);
} // namespace Arvo::RayTracing
