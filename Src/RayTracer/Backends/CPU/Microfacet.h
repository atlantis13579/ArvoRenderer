#pragma once
#include "Renderer/RayTracing.h"

namespace Arvo::RayTracing::Cpu
{
// Directions point away from the surface in an orthonormal local +Z frame.
// GGX slopes alpha.x/y are positive (not perceptual roughness). The continuous
// functions do not represent the alpha=0 delta distribution.
// Fresnel indices are positive, absorption is nonnegative. Refraction requires
// a unit normal facing the unit view and indexRatio=incident/transmitted.
float DielectricFresnel(float cosine, float incidentIndex, float transmittedIndex);
float ConductorFresnel(float cosine, float relativeIndex, float relativeAbsorption);
bool RefractDirection(Float3 view, Float3 normal, float indexRatio, Float3 &direction);
float GgxDistribution(Float3 normal, Float2 alpha);
float GgxMasking(Float3 direction, Float2 alpha);
float GgxMaskingShadowing(Float3 view, Float3 light, Float2 alpha);
float GgxVisibleNormalPdf(Float3 view, Float3 normal, Float2 alpha);
Float3 SampleGgxVisibleNormal(Float3 view, Float2 alpha, Float2 variates);
float GgxReflectionPdf(Float3 view, Float3 light, Float2 alpha);
float GgxReflectionFactor(Float3 view, Float3 light, Float2 alpha);
} // namespace Arvo::RayTracing::Cpu
