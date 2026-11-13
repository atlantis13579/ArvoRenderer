#include "RayTracer/Backends/CPU/Microfacet.h"
#include <algorithm>
#include <cmath>

namespace Arvo::RayTracing::Cpu
{
// Unpolarized Fresnel equations: average the squared s/p amplitudes. Indices
// describe the actual incident/transmitted media; cosine is nonnegative.
// Equal indices transmit even at grazing incidence. At the critical angle the
// transmitted ray is tangent and contributes no transmitted flux.
// PBRT 4e, "Specular Reflection and Transmission"; no Schlick approximation.
// https://pbr-book.org/4ed/Reflection_Models/Specular_Reflection_and_Transmission
float DielectricFresnel(float cosine, float incidentIndex, float transmittedIndex)
{
    if (incidentIndex == transmittedIndex)
        return 0.0f;
    const float incidentCosine = std::clamp(cosine, 0.0f, 1.0f);
    const float ratio = incidentIndex / transmittedIndex;
    // Factoring the difference of squared indices retains nearly matched media
    // at grazing angles, where subtracting the transmitted sine from one fails.
    const float indexDifference = (transmittedIndex - incidentIndex) / transmittedIndex;
    const float indexSum = (transmittedIndex + incidentIndex) / transmittedIndex;
    const float transmittedCosineSquared =
        indexDifference * indexSum + ratio * ratio * incidentCosine * incidentCosine;
    if (transmittedCosineSquared <= 0.0f)
        return 1.0f;
    const float transmittedCosine = std::sqrt(transmittedCosineSquared);
    const float incidentNormal = incidentIndex * incidentCosine;
    const float transmittedNormal = transmittedIndex * transmittedCosine;
    const float perpendicular = (incidentNormal - transmittedNormal) / (incidentNormal + transmittedNormal);
    const float incidentParallel = transmittedIndex * incidentCosine;
    const float transmittedParallel = incidentIndex * transmittedCosine;
    const float parallel =
        (incidentParallel - transmittedParallel) / (incidentParallel + transmittedParallel);
    return (perpendicular * perpendicular + parallel * parallel) * 0.5f;
}

float ConductorFresnel(float cosine, float relativeIndex, float relativeAbsorption)
{
    if (relativeAbsorption == 0.0f)
        return DielectricFresnel(cosine, 1.0f, relativeIndex);
    const float c = std::clamp(cosine, 0.0f, 1.0f);
    const float sineSquared = 1.0f - c * c;
    const float indexSquared = relativeIndex * relativeIndex;
    const float absorptionSquared = relativeAbsorption * relativeAbsorption;
    const float realPart = (relativeIndex - 1.0f) * (relativeIndex + 1.0f) + c * c - absorptionSquared;
    const float magnitude = std::sqrt(realPart * realPart + 4.0f * indexSquared * absorptionSquared);
    const float rootSquared = realPart < 0.0f
                                  ? (2.0f * indexSquared * absorptionSquared) / (magnitude - realPart)
                                  : (magnitude + realPart) * 0.5f;
    const float realRoot = std::sqrt(std::max(0.0f, rootSquared));
    const float projectedRoot = 2.0f * c * realRoot;
    const float perpendicular = (magnitude + c * c - projectedRoot) / (magnitude + c * c + projectedRoot);
    const float parallelBase = c * c * magnitude + sineSquared * sineSquared;
    const float parallelCross = projectedRoot * sineSquared;
    const float parallel = perpendicular * (parallelBase - parallelCross) / (parallelBase + parallelCross);
    return std::clamp((perpendicular + parallel) * 0.5f, 0.0f, 1.0f);
}

bool RefractDirection(Float3 view, Float3 normal, float indexRatio, Float3 &direction)
{
    direction = {};
    const float cosine = Dot(view, normal);
    if (cosine < 0.0f)
        return false;
    if (indexRatio == 1.0f)
    {
        direction = -view;
        return true;
    }
    const float projected = indexRatio * cosine;
    const float transmittedCosineSquared = (1.0f - indexRatio) * (1.0f + indexRatio) + projected * projected;
    if (transmittedCosineSquared <= 0.0f)
        return false;
    const float transmittedCosine = std::sqrt(transmittedCosineSquared);
    direction = -view * indexRatio + normal * (indexRatio * cosine - transmittedCosine);
    return true;
}
} // namespace Arvo::RayTracing::Cpu
