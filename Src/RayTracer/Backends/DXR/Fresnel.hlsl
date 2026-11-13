// DXR-owned unpolarized Fresnel evaluation and Snell refraction. These are
// interface events; medium indices must be chosen using geometric orientation.
// PBRT 4e, "Specular Reflection and Transmission".
float DielectricFresnel(float cosine, float incidentIndex, float transmittedIndex)
{
    if (incidentIndex == transmittedIndex) return 0.0f;
    float incidentCosine = clamp(cosine, 0.0f, 1.0f);
    float eta = incidentIndex / transmittedIndex;
    // Avoid cancellation between squared indices near an index-matched interface.
    float difference = (transmittedIndex - incidentIndex) / transmittedIndex;
    float sum = (transmittedIndex + incidentIndex) / transmittedIndex;
    precise float transmittedCosineSquared = difference * sum + eta * eta * incidentCosine * incidentCosine;
    if (transmittedCosineSquared <= 0.0f) return 1.0f;
    float transmittedCosine = sqrt(transmittedCosineSquared);
    float normalIncident = incidentIndex * incidentCosine;
    float normalTransmitted = transmittedIndex * transmittedCosine;
    float s = (normalIncident - normalTransmitted) / (normalIncident + normalTransmitted);
    float parallelIncident = transmittedIndex * incidentCosine;
    float parallelTransmitted = incidentIndex * transmittedCosine;
    float p = (parallelIncident - parallelTransmitted) / (parallelIncident + parallelTransmitted);
    return (s * s + p * p) * 0.5f;
}
float ConductorFresnel(float cosine, float relativeIndex, float relativeAbsorption)
{
    if (relativeAbsorption == 0.0f) return DielectricFresnel(cosine, 1.0f, relativeIndex);
    float c = clamp(cosine, 0.0f, 1.0f);
    float sineSquared = 1.0f - c * c;
    float etaSquared = relativeIndex * relativeIndex;
    float absorptionSquared = relativeAbsorption * relativeAbsorption;
    precise float realPart = (relativeIndex - 1.0f) * (relativeIndex + 1.0f) + c * c - absorptionSquared;
    float magnitude = sqrt(realPart * realPart + 4.0f * etaSquared * absorptionSquared);
    float rootSquared;
    if (realPart < 0.0f)
        rootSquared = 2.0f * etaSquared * absorptionSquared / (magnitude - realPart);
    else
        rootSquared = (magnitude + realPart) * 0.5f;
    float twiceProjectedRoot = 2.0f * c * sqrt(max(0.0f, rootSquared));
    float s = (magnitude + c * c - twiceProjectedRoot) / (magnitude + c * c + twiceProjectedRoot);
    float parallelBase = c * c * magnitude + sineSquared * sineSquared;
    float parallelCross = twiceProjectedRoot * sineSquared;
    float p = s * (parallelBase - parallelCross) / (parallelBase + parallelCross);
    return clamp((s + p) * 0.5f, 0.0f, 1.0f);
}
bool RefractDirection(float3 view, float3 normal, float indexRatio, out float3 direction)
{
    direction = float3(0.0f, 0.0f, 0.0f);
    float cosine = dot(view, normal);
    if (cosine < 0.0f) return false;
    if (indexRatio == 1.0f) { direction = -view; return true; }
    float projected = indexRatio * cosine;
    precise float transmittedCosineSquared = (1.0f - indexRatio) * (1.0f + indexRatio) + projected * projected;
    if (transmittedCosineSquared <= 0.0f) return false;
    float transmittedCosine = sqrt(transmittedCosineSquared);
    direction = -indexRatio * view + (indexRatio * cosine - transmittedCosine) * normal;
    return true;
}
