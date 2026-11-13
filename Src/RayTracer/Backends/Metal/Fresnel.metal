// Metal Fresnel interface equations. Caller assigns medium indices from geometry;
// face-forward shading normals alone cannot distinguish entry from exit.
// PBRT 4e, "Specular Reflection and Transmission". The conductor calculation is
// the exact complex-index result expressed in real arithmetic, not Schlick.
float DielectricFresnel(float cosine, float incidentIndex, float transmittedIndex)
{
    if (incidentIndex == transmittedIndex) return 0.0f;
    float c = clamp(cosine, 0.0f, 1.0f);
    float ratio = incidentIndex / transmittedIndex;
    // Keep the small index difference before squaring at grazing incidence.
    float difference = (transmittedIndex - incidentIndex) / transmittedIndex;
    float sum = (transmittedIndex + incidentIndex) / transmittedIndex;
    float transmittedCosineSquared = difference * sum + ratio * ratio * c * c;
    if (transmittedCosineSquared <= 0.0f) return 1.0f;
    float ct = sqrt(transmittedCosineSquared);
    float incidentNormal = incidentIndex * c, transmittedNormal = transmittedIndex * ct;
    float s = (incidentNormal - transmittedNormal) / (incidentNormal + transmittedNormal);
    float incidentParallel = transmittedIndex * c, transmittedParallel = incidentIndex * ct;
    float p = (incidentParallel - transmittedParallel) / (incidentParallel + transmittedParallel);
    return 0.5f * (s * s + p * p);
}
float ConductorFresnel(float cosine, float relativeIndex, float relativeAbsorption)
{
    if (relativeAbsorption == 0.0f) return DielectricFresnel(cosine, 1.0f, relativeIndex);
    float c = clamp(cosine, 0.0f, 1.0f), sineSquared = 1.0f - c * c;
    float etaSquared = relativeIndex * relativeIndex, kSquared = relativeAbsorption * relativeAbsorption;
    float realPart = (relativeIndex - 1.0f) * (relativeIndex + 1.0f) + c * c - kSquared;
    float magnitude = sqrt(realPart * realPart + 4.0f * etaSquared * kSquared);
    float rootSquared = realPart < 0.0f ? (2.0f * etaSquared * kSquared) / (magnitude - realPart)
        : (magnitude + realPart) * 0.5f;
    float projectedRoot = 2.0f * c * sqrt(max(0.0f, rootSquared));
    float s = (magnitude + c * c - projectedRoot) / (magnitude + c * c + projectedRoot);
    float parallelBase = c * c * magnitude + sineSquared * sineSquared;
    float parallelTerm = projectedRoot * sineSquared;
    float p = s * (parallelBase - parallelTerm) / (parallelBase + parallelTerm);
    return clamp(0.5f * (s + p), 0.0f, 1.0f);
}
bool RefractDirection(float3 view, float3 normal, float indexRatio, thread float3& direction)
{
    direction = float3(0.0f);
    float cosine = dot(view, normal);
    if (cosine < 0.0f) return false;
    if (indexRatio == 1.0f) { direction = -view; return true; }
    float projected = indexRatio * cosine;
    float transmittedCosineSquared = (1.0f - indexRatio) * (1.0f + indexRatio) + projected * projected;
    if (transmittedCosineSquared <= 0.0f) return false;
    float transmittedCosine = sqrt(transmittedCosineSquared);
    direction = normal * (indexRatio * cosine - transmittedCosine) - view * indexRatio;
    return true;
}
