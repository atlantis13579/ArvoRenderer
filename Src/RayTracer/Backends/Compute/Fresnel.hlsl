// Native compute Fresnel equations, averaged over s and p polarization.
// Indices name both sides explicitly; a shading normal alone cannot choose them.
// PBRT 4e, "Specular Reflection and Transmission".
float DielectricFresnel(float cosine, float incidentIndex, float transmittedIndex)
{
    if (incidentIndex == transmittedIndex) return 0.0f;
    float c = saturate(cosine);
    float ratio = incidentIndex / transmittedIndex;
    // Factored index differences preserve grazing, nearly matched interfaces.
    float difference = (transmittedIndex - incidentIndex) / transmittedIndex;
    float sum = (transmittedIndex + incidentIndex) / transmittedIndex;
    precise float transmittedCosineSquared = difference * sum + ratio * ratio * c * c;
    if (transmittedCosineSquared <= 0.0f) return 1.0f;
    float ct = sqrt(transmittedCosineSquared);
    float perpendicular = (incidentIndex * c - transmittedIndex * ct) / (incidentIndex * c + transmittedIndex * ct);
    float parallel = (transmittedIndex * c - incidentIndex * ct) / (transmittedIndex * c + incidentIndex * ct);
    return 0.5f * (perpendicular * perpendicular + parallel * parallel);
}
float ConductorFresnel(float cosine, float relativeIndex, float relativeAbsorption)
{
    if (relativeAbsorption == 0.0f) return DielectricFresnel(cosine, 1.0f, relativeIndex);
    float c = saturate(cosine), sinSquared = 1.0f - c * c;
    float etaSquared = relativeIndex * relativeIndex, kSquared = relativeAbsorption * relativeAbsorption;
    precise float realPart = (relativeIndex - 1.0f) * (relativeIndex + 1.0f) + c * c - kSquared;
    float magnitude = sqrt(realPart * realPart + 4.0f * etaSquared * kSquared);
    float rootSquared = realPart < 0.0f ? 2.0f * etaSquared * kSquared / (magnitude - realPart)
        : (magnitude + realPart) * 0.5f;
    float crossTerm = 2.0f * c * sqrt(max(0.0f, rootSquared));
    float perpendicular = (magnitude + c * c - crossTerm) / (magnitude + c * c + crossTerm);
    float parallelBase = c * c * magnitude + sinSquared * sinSquared;
    float parallelCross = crossTerm * sinSquared;
    float parallel = perpendicular * (parallelBase - parallelCross) / (parallelBase + parallelCross);
    return saturate(0.5f * (perpendicular + parallel));
}
bool RefractDirection(float3 view, float3 normal, float indexRatio, out float3 direction)
{
    direction = 0.0f;
    float cosine = dot(view, normal);
    if (cosine < 0.0f) return false;
    if (indexRatio == 1.0f) { direction = -view; return true; }
    float projected = indexRatio * cosine;
    precise float transmittedCosineSquared = (1.0f - indexRatio) * (1.0f + indexRatio) + projected * projected;
    if (transmittedCosineSquared <= 0.0f) return false;
    float transmittedCosine = sqrt(transmittedCosineSquared);
    direction = normal * (indexRatio * cosine - transmittedCosine) - view * indexRatio;
    return true;
}
