#pragma once
#include "Renderer/AdaptiveReconstruction.h"

namespace Arvo::RayTracing::Cpu
{
uint32_t DenoisedSampleBudget(const AdaptiveReconstructionData::Pixel&,
                              const AdaptiveReconstructionData::Controls&);
}
