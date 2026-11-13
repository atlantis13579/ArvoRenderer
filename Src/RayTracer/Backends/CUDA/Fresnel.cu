#include <cuda_runtime.h>

namespace ArvoCuda
{
// Exact unpolarized Fresnel amplitudes, with indices assigned to the incident
// and transmitted media. The conductor uses the real form of complex Snell.
// PBRT 4e, "Specular Reflection and Transmission".
__device__ float DielectricFresnel(float cosine, float incidentIndex, float transmittedIndex)
{
    if (incidentIndex == transmittedIndex)
        return 0.0f;
    const float c = fminf(1.0f, fmaxf(0.0f, cosine));
    const float ratio = incidentIndex / transmittedIndex;
    // Preserve the small index difference before squaring near matched media.
    const float difference = (transmittedIndex - incidentIndex) / transmittedIndex;
    const float sum = (transmittedIndex + incidentIndex) / transmittedIndex;
    const float transmittedCosineSquared = difference * sum + ratio * ratio * c * c;
    if (transmittedCosineSquared <= 0.0f)
        return 1.0f;
    const float transmittedCosine = sqrtf(transmittedCosineSquared);
    const float perpendicularNumerator = incidentIndex * c - transmittedIndex * transmittedCosine;
    const float perpendicularDenominator = incidentIndex * c + transmittedIndex * transmittedCosine;
    const float parallelNumerator = transmittedIndex * c - incidentIndex * transmittedCosine;
    const float parallelDenominator = transmittedIndex * c + incidentIndex * transmittedCosine;
    const float perpendicular = perpendicularNumerator / perpendicularDenominator;
    const float parallel = parallelNumerator / parallelDenominator;
    return 0.5f * (perpendicular * perpendicular + parallel * parallel);
}

__device__ float ConductorFresnel(float cosine, float relativeIndex, float relativeAbsorption)
{
    if (relativeAbsorption == 0.0f)
        return DielectricFresnel(cosine, 1.0f, relativeIndex);
    const float c = fminf(1.0f, fmaxf(0.0f, cosine));
    const float sinSquared = 1.0f - c * c;
    const float etaSquared = relativeIndex * relativeIndex;
    const float kSquared = relativeAbsorption * relativeAbsorption;
    const float realPart = (relativeIndex - 1.0f) * (relativeIndex + 1.0f) + c * c - kSquared;
    const float magnitude = sqrtf(realPart * realPart + 4.0f * etaSquared * kSquared);
    // Rationalize the complex square root when its real part is negative.
    const float rootSquared = realPart < 0.0f ? 2.0f * etaSquared * kSquared / (magnitude - realPart)
                                              : 0.5f * (magnitude + realPart);
    const float crossTerm = 2.0f * c * sqrtf(fmaxf(0.0f, rootSquared));
    const float perpendicular = (magnitude + c * c - crossTerm) / (magnitude + c * c + crossTerm);
    const float parallelBase = c * c * magnitude + sinSquared * sinSquared;
    const float parallelTerm = crossTerm * sinSquared;
    const float parallel = perpendicular * (parallelBase - parallelTerm) / (parallelBase + parallelTerm);
    return fminf(1.0f, fmaxf(0.0f, 0.5f * (perpendicular + parallel)));
}

__device__ bool RefractDirection(float3 view, float3 normal, float indexRatio, float3 &direction)
{
    direction = make_float3(0.0f, 0.0f, 0.0f);
    const float cosine = view.x * normal.x + view.y * normal.y + view.z * normal.z;
    if (cosine < 0.0f)
        return false;
    if (indexRatio == 1.0f)
    {
        direction = make_float3(-view.x, -view.y, -view.z);
        return true;
    }
    const float projected = indexRatio * cosine;
    const float transmittedCosineSquared = (1.0f - indexRatio) * (1.0f + indexRatio) + projected * projected;
    if (transmittedCosineSquared <= 0.0f)
        return false;
    const float normalWeight = projected - sqrtf(transmittedCosineSquared);
    direction = make_float3(normal.x * normalWeight - view.x * indexRatio,
                            normal.y * normalWeight - view.y * indexRatio,
                            normal.z * normalWeight - view.z * indexRatio);
    return true;
}
} // namespace ArvoCuda
