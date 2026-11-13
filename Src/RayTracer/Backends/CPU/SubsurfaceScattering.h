#pragma once
#include "Renderer/RayTracing.h"

namespace Arvo::RayTracing::Cpu
{
struct SubsurfaceFlight
{
    Float3 Weight = {1, 1, 1};
    float Distance = 0;
    bool Scattered = false;
};
struct SubsurfaceDirection
{
    Float3 Direction;
    float Pdf = 0, Weight = 0;
};
SubsurfaceFlight SampleSubsurfaceFlight(float meanFreePath, Float3 albedo, Float3 direction, float endpoint,
                                        float variate);
float SubsurfacePhase(float propagationCosine, float anisotropy);
SubsurfaceDirection SampleSubsurfaceDirection(Float3 incoming, float anisotropy, Float2 variates,
                                              bool importance);
} // namespace Arvo::RayTracing::Cpu
