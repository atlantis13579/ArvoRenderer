// Independent DX11 preview scattering. No offline backend kernel is included.
// Lambert reflection, exact dielectric/conductor Fresnel and isotropic GGX use
// the same optical parameters as the transport pipelines, without remapping alpha.
// Walter et al., "Microfacet Models for Refraction through Rough Surfaces" (2007).
// Heitz, "Sampling the GGX Distribution of Visible Normals" (2018).
// https://pbr-book.org/4ed/Reflection_Models/Dielectric_BSDF
// https://pbr-book.org/4ed/Reflection_Models/Conductor_BRDF
// Environment integration has no scene visibility or interreflection. Refraction
// samples the distant environment, not geometry behind the surface; use DXR for
// those paths. Delta surfaces retain discrete reflection/refraction directions.

static const uint RasterLegacy = 0;
static const uint RasterDiffuse = 1;
static const uint RasterConductor = 2;
static const uint RasterDielectric = 3;
static const float RasterPi = 3.14159265358979323846f;
static const uint RasterEnvironmentSamples = 32;
static const float RasterGoldenRotation = 0.6180339887498948482f;

struct RasterMaterial
{
    uint Model;
    float Slope;
    float IncidentIndex;
    float TransmittedIndex;
    float3 ConductorIndex;
    float3 ConductorExtinction;
    float3 Reflectance;
};

float RasterDielectricFresnel(float cosine, float incidentIndex, float transmittedIndex)
{
    if (incidentIndex == transmittedIndex) return 0;
    float c = clamp(abs(cosine), 0.0f, 1.0f);
    float ratio = incidentIndex / transmittedIndex;
    float sineSquared = ratio * ratio * max(0.0f, 1 - c * c);
    if (sineSquared >= 1) return 1;
    float transmittedCosine = sqrt(1 - sineSquared);
    float perpendicular = (incidentIndex * c - transmittedIndex * transmittedCosine) /
                          (incidentIndex * c + transmittedIndex * transmittedCosine);
    float parallel = (transmittedIndex * c - incidentIndex * transmittedCosine) /
                     (transmittedIndex * c + incidentIndex * transmittedCosine);
    return (perpendicular * perpendicular + parallel * parallel) / 2;
}

float RasterConductorFresnel(float cosine, float eta, float extinction)
{
    float c = clamp(abs(cosine), 0.0f, 1.0f);
    if (eta == 1 && extinction == 0) return 0;
    if (c == 0) return 1;
    float cSquared = c * c;
    float sineSquared = 1 - cSquared;
    float opticalDifference = eta * eta - extinction * extinction - sineSquared;
    float magnitude = sqrt(opticalDifference * opticalDifference +
                           4 * eta * eta * extinction * extinction);
    float realRoot = sqrt(max(0.0f, (magnitude + opticalDifference) / 2));
    float twiceCosineRoot = 2 * c * realRoot;
    float perpendicular = (magnitude + cSquared - twiceCosineRoot) /
                          (magnitude + cSquared + twiceCosineRoot);
    float sum = cSquared * magnitude + sineSquared * sineSquared;
    float parallel = perpendicular * (sum - twiceCosineRoot * sineSquared) /
                                     (sum + twiceCosineRoot * sineSquared);
    return clamp((perpendicular + parallel) / 2, 0.0f, 1.0f);
}

float3 RasterFresnel(RasterMaterial material, float cosine)
{
    float3 result = float3(0, 0, 0);
    if (material.Model == RasterConductor)
    {
        result.x = RasterConductorFresnel(cosine, material.ConductorIndex.x, material.ConductorExtinction.x);
        result.y = RasterConductorFresnel(cosine, material.ConductorIndex.y, material.ConductorExtinction.y);
        result.z = RasterConductorFresnel(cosine, material.ConductorIndex.z, material.ConductorExtinction.z);
    }
    else
    {
        float fresnel = RasterDielectricFresnel(cosine, material.IncidentIndex, material.TransmittedIndex);
        result = float3(fresnel, fresnel, fresnel);
    }
    return result;
}

float RasterGgxDistribution(float3 normal, float3 facet, float slope)
{
    float cosine = dot(normal, facet);
    float3 tangentProjection = facet - normal * cosine;
    float alphaSquared = slope * slope;
    // Project explicitly instead of subtracting two nearly equal squared cosines.
    // This preserves narrow highlights at the minimum supported slope.
    float denominator = dot(tangentProjection, tangentProjection) + alphaSquared * cosine * cosine;
    return alphaSquared / (RasterPi * denominator * denominator);
}

float RasterGgxLambda(float normalCosine, float slope)
{
    float cosineSquared = normalCosine * normalCosine;
    return (sqrt(1 + slope * slope * max(0.0f, 1 - cosineSquared) / cosineSquared) - 1) / 2;
}

float RasterGgxGeometry(float viewCosine, float lightCosine, float slope)
{
    if (viewCosine == 0 || lightCosine == 0) return 0;
    return 1 / (1 + RasterGgxLambda(viewCosine, slope) + RasterGgxLambda(lightCosine, slope));
}

float3 RasterEvaluateBsdf(RasterMaterial material, float3 normal, float3 view, float3 light,
                          bool opaqueInterior = false)
{
    float viewCosine = dot(normal, view), lightCosine = dot(normal, light);
    if (viewCosine <= 0 || lightCosine == 0) return float3(0, 0, 0);
    if (material.Model == RasterDiffuse)
        return lightCosine > 0 ? material.Reflectance / RasterPi : float3(0, 0, 0);
    bool matched = material.Model == RasterDielectric && material.IncidentIndex == material.TransmittedIndex;
    if (material.Model == RasterLegacy || material.Slope == 0 || matched) return float3(0, 0, 0);
    bool reflection = lightCosine > 0;
    if (opaqueInterior && !reflection) return float3(0, 0, 0);
    if (!reflection && material.Model != RasterDielectric) return float3(0, 0, 0);
    float ratio = material.IncidentIndex / material.TransmittedIndex;
    float3 halfVector = reflection ? view + light : view * ratio + light;
    float halfLengthSquared = dot(halfVector, halfVector);
    if (halfLengthSquared <= 0) return float3(0, 0, 0);
    float3 facet = halfVector / sqrt(halfLengthSquared);
    if (dot(normal, facet) < 0) facet = -facet;
    float viewFacet = dot(view, facet), lightFacet = dot(light, facet);
    if (viewFacet <= 0 || (!reflection && lightFacet >= 0)) return float3(0, 0, 0);
    float distributionGeometry = RasterGgxDistribution(normal, facet, material.Slope) *
                                 RasterGgxGeometry(viewCosine, lightCosine, material.Slope);
    if (reflection)
        return RasterFresnel(material, viewFacet) *
               (distributionGeometry / (4 * viewCosine * lightCosine));
    float fresnel = material.IncidentIndex <= material.TransmittedIndex
        ? RasterDielectricFresnel(viewFacet, material.IncidentIndex, material.TransmittedIndex)
        : RasterDielectricFresnel(-lightFacet, material.TransmittedIndex, material.IncidentIndex);
    float transmission = (1 - fresnel) * distributionGeometry * abs(viewFacet * lightFacet) * ratio * ratio /
                         (viewCosine * abs(lightCosine) * halfLengthSquared);
    return float3(transmission, transmission, transmission);
}

float3 RasterEnvironment(float3 direction, float3 top, float3 bottom)
{
    float upperWeight = (clamp(direction.y, -1.0f, 1.0f) + 1) / 2;
    return top * upperWeight + bottom * (1 - upperWeight);
}

float3 RasterRefract(float3 view, float3 facet, float indexRatio)
{
    float cosine = dot(view, facet);
    float sineSquared = indexRatio * indexRatio * max(0.0f, 1 - cosine * cosine);
    if (sineSquared >= 1) return float3(0, 0, 0);
    return -view * indexRatio + facet * (indexRatio * cosine - sqrt(1 - sineSquared));
}

float3 RasterVisibleNormal(float3 localView, float slope, float u, float v)
{
    float3 stretchedView = normalize(float3(slope * localView.x, slope * localView.y, localView.z));
    float planarLength = stretchedView.x * stretchedView.x + stretchedView.y * stretchedView.y;
    float3 tangent = planarLength > 0
        ? float3(-stretchedView.y, stretchedView.x, 0) / sqrt(planarLength) : float3(1, 0, 0);
    float3 bitangent = cross(stretchedView, tangent);
    float radius = sqrt(u), azimuth = 2 * RasterPi * v;
    float x = radius * cos(azimuth), y = radius * sin(azimuth);
    float blend = (1 + stretchedView.z) / 2;
    y = (1 - blend) * sqrt(max(0.0f, 1 - x * x)) + blend * y;
    float3 projected = tangent * x + bitangent * y +
                       stretchedView * sqrt(max(0.0f, 1 - x * x - y * y));
    return normalize(float3(slope * projected.x, slope * projected.y, max(0.0f, projected.z)));
}

float3 RasterEnvironmentPath(RasterMedium medium, float3 position, float3 direction, float3 top,
                             float3 bottom)
{
    return RasterEnvironment(direction, top, bottom) *
           RasterMediumTransmission(medium, position, direction, RasterMediumRayLimit);
}

float3 RasterDiffuseEnvironment(float3 normal, float3 top, float3 bottom, RasterMedium medium,
                                float3 position)
{
    float3 incident = (top + bottom) / 2 + (top - bottom) * (normal.y / 3);
    const float3 extinction = medium.Absorption + medium.Scattering;
    if (medium.Enabled != 0 && (extinction.x > 0 || extinction.y > 0 || extinction.z > 0))
    {
        const float ParallelThreshold = 0.999f;
        const float3 helper = abs(normal.y) < ParallelThreshold ? float3(0, 1, 0) : float3(1, 0, 0);
        const float3 tangent = normalize(cross(helper, normal)), bitangent = cross(normal, tangent);
        incident = float3(0, 0, 0);
        for (uint index = 0; index < RasterEnvironmentSamples; ++index)
        {
            const float u = (index + 0.5f) / RasterEnvironmentSamples;
            const float rotation = index * RasterGoldenRotation;
            const float azimuth = 2 * RasterPi * (rotation - floor(rotation));
            const float radial = sqrt(u);
            const float3 direction = tangent * (radial * cos(azimuth)) + bitangent * (radial * sin(azimuth)) +
                                     normal * sqrt(1 - u);
            incident += RasterEnvironmentPath(medium, position, direction, top, bottom);
        }
        incident /= RasterEnvironmentSamples;
    }
    return incident;
}

float3 RasterEnvironmentLighting(RasterMaterial material, float3 normal, float3 view, float3 top,
                                 float3 bottom, RasterMedium medium, float3 position, bool opaqueInterior = false)
{
    float3 result = float3(0, 0, 0);
    float viewCosine = dot(normal, view);
    float ratio = material.IncidentIndex / material.TransmittedIndex;
    if (material.Model == RasterDiffuse)
    {
        result = material.Reflectance * RasterDiffuseEnvironment(normal, top, bottom, medium, position);
    }
    else if (viewCosine <= 0)
    {
        result = float3(0, 0, 0);
    }
    else if (material.Model == RasterDielectric && material.IncidentIndex == material.TransmittedIndex)
    {
        if (!opaqueInterior) result = RasterEnvironmentPath(medium, position, -view, top, bottom);
    }
    else if (material.Slope == 0)
    {
        result = RasterFresnel(material, viewCosine) *
                 RasterEnvironmentPath(medium, position, normal * (2 * viewCosine) - view, top, bottom);
        if (material.Model == RasterDielectric && !opaqueInterior)
        {
            float3 transmitted = RasterRefract(view, normal, ratio);
            float fresnel =
                RasterDielectricFresnel(viewCosine, material.IncidentIndex, material.TransmittedIndex);
            result += RasterEnvironmentPath(medium, position, transmitted, top, bottom) *
                      ((1 - fresnel) * ratio * ratio);
        }
    }
    else
    {
        float3 helper = abs(normal.z) < 1 ? float3(0, 0, 1) : float3(0, 1, 0);
        float3 tangent = normalize(cross(helper, normal)), bitangent = cross(normal, tangent);
        float3 localView = float3(dot(view, tangent), dot(view, bitangent), viewCosine);
        float viewLambda = RasterGgxLambda(viewCosine, material.Slope);
        for (uint sampleIndex = 0; sampleIndex < RasterEnvironmentSamples; ++sampleIndex)
        {
            float u = (sampleIndex + 0.5f) / RasterEnvironmentSamples;
            float rotation = sampleIndex * RasterGoldenRotation;
            float v = rotation - floor(rotation);
            float3 localFacet = RasterVisibleNormal(localView, material.Slope, u, v);
            float3 facet = tangent * localFacet.x + bitangent * localFacet.y + normal * localFacet.z;
            float viewFacet = dot(view, facet);
            float3 reflected = facet * (2 * viewFacet) - view;
            float reflectedCosine = dot(normal, reflected);
            if (reflectedCosine > 0)
                result += RasterFresnel(material, viewFacet) *
                          RasterEnvironmentPath(medium, position, reflected, top, bottom) *
                          ((1 + viewLambda) * RasterGgxGeometry(viewCosine, reflectedCosine, material.Slope));
            if (material.Model == RasterDielectric && !opaqueInterior)
            {
                float3 transmitted = RasterRefract(view, facet, ratio);
                float transmittedCosine = dot(normal, transmitted);
                if (transmittedCosine < 0)
                {
                    float transmission = 1 - RasterDielectricFresnel(viewFacet, material.IncidentIndex,
                                                                     material.TransmittedIndex);
                    result += RasterEnvironmentPath(medium, position, transmitted, top, bottom) *
                              (transmission * ratio * ratio * (1 + viewLambda) *
                               RasterGgxGeometry(viewCosine, transmittedCosine, material.Slope));
                }
            }
        }
        result /= RasterEnvironmentSamples;
    }
    return result;
}

float3 RasterEnvironmentLighting(RasterMaterial material, float3 normal, float3 view, float3 top,
                                 float3 bottom)
{
    RasterMedium vacuum;
    vacuum.Minimum = vacuum.Maximum = vacuum.Absorption = vacuum.Scattering = float3(0, 0, 0);
    vacuum.Enabled = 0;
    vacuum.Anisotropy = 0;
    return RasterEnvironmentLighting(material, normal, view, top, bottom, vacuum, float3(0, 0, 0));
}

// Unit-radiance visibility of an ideal reflected/refracted ray against a finite
// one-sided rectangle. The return value converts integrated light intensity to
// emitted radiance; it deliberately has no inverse-square falloff.
float RasterRectangleRadianceWeight(float3 origin, float3 direction, float3 position,
                                    float3 emissionDirection, float3 tangent, float3 bitangent,
                                    float2 size)
{
    static const float ParallelTolerance = 1.0e-7f;
    static const float MinimumEmitterDistance = 1.0e-4f;
    float denominator = dot(emissionDirection, direction);
    if (denominator >= -ParallelTolerance) return 0;
    float distance = dot(position - origin, emissionDirection) / denominator;
    if (distance <= MinimumEmitterDistance) return 0;
    float3 offset = origin + direction * distance - position;
    if (abs(dot(offset, tangent)) > size.x / 2 || abs(dot(offset, bitangent)) > size.y / 2) return 0;
    float area = size.x * size.y;
    return area > 0 ? 1 / area : 0;
}
