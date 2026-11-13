#pragma once
#include "Renderer/ParticipatingMedia.h"

namespace Arvo::RayTracing::Cpu
{
struct MediumSegmentSample
{
    Float3 Weight = {1, 1, 1};
    float Distance = 0;
    bool Scattered = false;
};
Float3 MediumTransmittance(const MediumOptions&, const Ray&, float endpointDistance);
MediumSegmentSample SampleMedium(const MediumOptions&, const Ray&, float endpointDistance, Float2 variates);
float MediumPhase(float propagationCosine, float anisotropy);
Float3 SampleMediumDirection(Float3 incidentPropagation, float anisotropy, Float2 variates,
                             bool importanceSampling, float& pdf);
} // namespace Arvo::RayTracing::Cpu
