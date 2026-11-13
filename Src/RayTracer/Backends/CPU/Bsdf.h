#pragma once
#include "Renderer/RayTracing.h"

namespace Arvo::RayTracing::Cpu
{
// Scalar (one color channel/wavelength) surface scattering in a local +Z frame.
// Unit directions point away from the surface; view.Z > 0. Geometry/medium state
// is assigned by the caller. Alpha is either (0,0) or two positive GGX slopes.
// Indices are positive, absorption nonnegative, reflectance in [0,1], variates in [0,1).
enum ScatteringEvent : uint32_t
{
    ScatteringNone = 0,
    ScatteringReflection = 1,
    ScatteringTransmission = 2,
    ScatteringDiffuse = 4,
    ScatteringGlossy = 8,
    ScatteringDelta = 16
};
enum class TransportMode : uint32_t
{
    Radiance,
    Importance
};
struct BsdfEvaluation
{
    float Value = 0, Pdf = 0, ReversePdf = 0;
};
struct BsdfSample
{
    Float3 Direction;
    float Value = 0, Pdf = 0, ReversePdf = 0, RelativeIndex = 1;
    uint32_t Event = ScatteringNone;
};
// Continuous evaluation returns solid-angle densities. Delta samples instead
// carry discrete masses; Event distinguishes the measures, including in MIS.
// Sample Value * abs(Direction.Z) / Pdf is the continuation weight in both cases.
// ReversePdf reverses the directions AND the incident/transmitted indices.
BsdfEvaluation EvaluateDiffuse(Float3 view, Float3 light, float reflectance, bool importance);
BsdfSample SampleDiffuse(Float3 view, float reflectance, Float2 variates, bool importance);
BsdfEvaluation EvaluateConductor(Float3 view, Float3 light, Float2 alpha, float eta, float absorption);
BsdfSample SampleConductor(Float3 view, Float2 alpha, float eta, float absorption, Float2 variates);
BsdfEvaluation EvaluateDielectric(Float3 view, Float3 light, Float2 alpha, float incidentIndex,
                                  float transmittedIndex, TransportMode mode);
BsdfSample SampleDielectric(Float3 view, Float2 alpha, float incidentIndex, float transmittedIndex,
                            TransportMode mode, Float2 variates, float lobeVariate);
} // namespace Arvo::RayTracing::Cpu
