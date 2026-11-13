#include "RayTracer/Backends/CPU/CpuTransport.h"
#include "RayTracer/Backends/CPU/SurfaceScattering.h"
#include "RayTracer/Backends/CPU/HomogeneousMedium.h"
#include "RayTracer/Backends/CPU/SpectralTransport.h"
#include <algorithm>
#include <cmath>

namespace Arvo::RayTracing::Cpu
{
// Direct lighting estimates each emitter independently. Rectangular lights
// use a two-by-two quadrature for the legacy shader and stratified samples for
// Lambertian transport. Intensity follows the existing demo convention: area
// radiance is Color * Intensity / Area, so its area PDF cancels that area.
Float3 DirectLighting(const Scene& scene, const SceneAccelerator& accelerator, const Ray& ray,
    const SurfaceHit& hit, const Float3& surface, const RenderSettings& settings,
    AtomicStatistics& statistics, const CancellationProbe& cancellation,
    const SampleContext& samples, uint32_t bounce, DirectLightingModel model, bool stochasticLights,
    const SpectralPath* spectral)
{
    const MediumOptions medium = spectral && spectral->Enabled() ? spectral->Medium() : settings.Medium;
    const bool lambertian = model == DirectLightingModel::Lambertian;
    const Material material = spectral ? spectral->EvaluateMaterial(scene.Materials[hit.MaterialIndex], hit.MaterialIndex)
                                       : scene.Materials[hit.MaterialIndex];
    const Float3 diffuse = Multiply(surface, material.Albedo);
    const Float3 position = ray.Origin + ray.Direction * hit.Distance;
    const float occludedVisibility = 1.0f - std::clamp(settings.ShadowStrength, 0.0f, 1.0f);
    const Float3 viewDirection = Normalize(-ray.Direction);
    Float3 direct = material.Emission
        + Multiply(surface, material.AmbientColor)
            * (lambertian ? 0.0f : std::max(0.0f, scene.AmbientIntensity));
    const auto accumulateLightSample = [&](
        const Float3& lightDirection,
        const Float3& lightRadiance,
        float visibility)
    {
        const float directionLengthSquared = LengthSquared(lightDirection);
        if (!std::isfinite(directionLengthSquared)
            || directionLengthSquared <= MinimumDirectionLengthSquared)
        {
            return;
        }
        const Float3 direction =
            lightDirection / std::sqrt(directionLengthSquared);
        const float normalToLight =
            std::max(0.0f, Dot(hit.Normal, direction));
        if (normalToLight <= 0.0f)
        {
            return;
        }
        const Float3 visibleRadiance =
            lightRadiance * std::clamp(visibility, 0.0f, 1.0f);
        direct = direct
            + Multiply(diffuse, visibleRadiance) * normalToLight
                * (lambertian ? 1.0f / Pi : 1.0f);
        const Float3 halfVector =
            Normalize(direction + viewDirection);
        const float specularAmount = std::pow(
            std::max(0.0f, Dot(hit.Normal, halfVector)),
            std::max(material.Shininess, 1.0f));
        direct = direct
            + Multiply(material.SpecularColor, visibleRadiance)
                * (lambertian ? 0.0f : specularAmount);
    };
    const auto accumulateFiniteLight = [&](
        const Float3& lightPosition,
        const Float3& lightColor,
        float intensity,
        bool castShadows,
        const Float3* emissionDirection,
        float sampleWeight)
    {
        if (intensity <= 0.0f || sampleWeight <= 0.0f)
        {
            return;
        }
        const Float3 toLight = lightPosition - position;
        const float distanceSquared = LengthSquared(toLight);
        if (!std::isfinite(distanceSquared)
            || distanceSquared <= RayEpsilon * RayEpsilon)
        {
            return;
        }
        const float distance = std::sqrt(distanceSquared);
        const Float3 directionToLight = toLight / distance;
        const float surfaceCosine =
            std::max(0.0f, Dot(hit.Normal, directionToLight));
        if (surfaceCosine <= 0.0f)
        {
            return;
        }
        float emitterCosine = 1.0f;
        if (emissionDirection != nullptr)
        {
            // A rectangular emitter radiates only through its front face.
            emitterCosine = std::max(
                0.0f,
                Dot(*emissionDirection, -directionToLight));
            if (emitterCosine <= 0.0f)
            {
                return;
            }
        }

        float visibility = 1.0f;
        if (settings.ShadowsEnabled &&
            settings.ShadowStrength > 0.0f && castShadows)
        {
            Ray shadowRay;
            shadowRay.Origin = position + hit.Normal * RayEpsilon;
            const Float3 shadowOffset = lightPosition - shadowRay.Origin;
            const float shadowDistance = std::sqrt(LengthSquared(shadowOffset));
            shadowRay.MinDistance = RayEpsilon;
            // Recompute the segment from the offset origin. Keeping the old
            // direction/distance can intersect the emitter itself at grazing angles.
            shadowRay.MaxDistance = shadowDistance - RayEpsilon;
            if (shadowRay.MaxDistance > shadowRay.MinDistance)
            {
                shadowRay.Direction = shadowOffset / shadowDistance;
                uint64_t shadowTests = 0;
                statistics.ShadowRays.fetch_add(
                    1,
                    std::memory_order_relaxed);
                if (accelerator.OccludedAtTime(shadowRay, shadowTests, samples.SceneTime))
                {
                    visibility = occludedVisibility;
                }
                statistics.PrimitiveTests.fetch_add(
                    shadowTests,
                    std::memory_order_relaxed);
            }
        }
        const float attenuation =
            intensity * emitterCosine * sampleWeight / distanceSquared;
        accumulateLightSample(
            directionToLight,
            Multiply(lightColor * attenuation, MediumTransmittance(medium,
                Ray{position, directionToLight, 0, distance}, distance)),
            visibility);
    };
    for (const DirectionalLight& light : scene.DirectionalLights)
    {
        const Float3 lightColor = spectral ? spectral->DirectionalColor(scene, light) : light.Color;
        const Float3 directionToLight = Normalize(light.DirectionToLight);
        const float cosine = std::max(0.0f, Dot(hit.Normal, directionToLight));
        if (cosine <= 0.0f || light.Intensity <= 0.0f) continue;
        float visibility = 1.0f;
        if (settings.ShadowsEnabled &&
            settings.ShadowStrength > 0.0f && light.CastShadows)
        {
            Ray shadowRay;
            const Float3 shadowNormal =
                Dot(hit.Normal, directionToLight) >= 0.0f
                    ? hit.Normal
                    : -hit.Normal;
            shadowRay.Origin =
                position + shadowNormal * RayEpsilon;
            shadowRay.Direction = directionToLight;
            shadowRay.MinDistance = RayEpsilon;
            uint64_t shadowTests = 0;
            statistics.ShadowRays.fetch_add(1, std::memory_order_relaxed);
            if (accelerator.OccludedAtTime(shadowRay, shadowTests, samples.SceneTime))
            {
                visibility = occludedVisibility;
            }
            statistics.PrimitiveTests.fetch_add(shadowTests, std::memory_order_relaxed);
        }
        accumulateLightSample(
            directionToLight,
            Multiply(lightColor * light.Intensity, MediumTransmittance(medium,
                Ray{position, directionToLight, 0, MaximumRayDistance}, MaximumRayDistance)),
            visibility);
    }
    for (const PointLight& light : scene.PointLights)
    {
        const Float3 lightColor = spectral ? spectral->PointColor(scene, light) : light.Color;
        accumulateFiniteLight(
            light.Position,
            lightColor,
            light.Intensity,
            light.CastShadows,
            nullptr,
            1.0f);
    }
    uint32_t areaLightIndex = static_cast<uint32_t>(scene.DirectionalLights.size() + scene.PointLights.size());
    for (const RectAreaLight& light : scene.RectAreaLights)
    {
        const Float3 lightColor = spectral ? spectral->RectColor(scene, light) : light.Color;
        const Float3 emissionDirection = Normalize(light.Direction);
        const Float3 helper = std::fabs(emissionDirection.Y) < BasisParallelThreshold
            ? Float3(0.0f, 1.0f, 0.0f)
            : Float3(1.0f, 0.0f, 0.0f);
        const Float3 tangent =
            Normalize(Cross(helper, emissionDirection));
        const Float3 bitangent =
            Normalize(Cross(emissionDirection, tangent));
        const float area = light.Size.X * light.Size.Y;
        const bool nee = !lambertian || settings.Transport.NextEventEstimation;
        for (uint32_t sampleIndex = 0; nee && sampleIndex < AreaLightSamples; ++sampleIndex)
        {
            if (cancellation && cancellation()) break;
            float offsetX = (sampleIndex & 1u) != 0 ? (1.0f / AreaLightSamples) : -(1.0f / AreaLightSamples);
            float offsetY = (sampleIndex & 2u) != 0 ? (1.0f / AreaLightSamples) : -(1.0f / AreaLightSamples);
            if (lambertian || stochasticLights)
            {
                const uint32_t dimension = ((bounce + 1u) * BounceSeedMultiplier)
                    ^ ((areaLightIndex + 1u) * LightSeedMultiplier) ^ ((sampleIndex + 1u) * LightSampleSeedMultiplier);
                const Float2 variates = Sample2D(samples, dimension);
                offsetX = (static_cast<float>(sampleIndex & 1u) + variates.X) * 0.5f - 0.5f;
                offsetY = (static_cast<float>(sampleIndex >> 1u) + variates.Y) * 0.5f - 0.5f;
            }
            const Float3 samplePosition = light.Position
                + tangent * (offsetX * light.Size.X)
                + bitangent * (offsetY * light.Size.Y);
            float weight = 1.0f / AreaLightSamples;
            if (lambertian && settings.Transport.Mis != MisHeuristic::None && area > 0.0f)
            {
                const Float3 offset = samplePosition - position;
                const float distanceSquared = LengthSquared(offset);
                const Float3 direction = Normalize(offset);
                const float emitterCosine = Dot(emissionDirection, -direction);
                if (emitterCosine > 0.0f)
                {
                    const float lightPdf = distanceSquared / (area * emitterCosine);
                    const float bsdfPdf = DiffusePdf(Dot(hit.Normal, direction), settings.Transport.ImportanceSampling);
                    weight *= MultipleImportance(AreaLightSamples * lightPdf, bsdfPdf, settings.Transport.Mis);
                }
            }
            accumulateFiniteLight(samplePosition, lightColor, light.Intensity,
                light.CastShadows, &emissionDirection, weight);
        }
        // Veach (1997), ch. 9: weights use sample-count times solid-angle PDF.
        // This independent BSDF sample also handles analytic lights without an
        // emissive mesh. Converting its weight to area measure reuses visibility.
        if (lambertian && (!nee || settings.Transport.Mis != MisHeuristic::None) && area > 0.0f)
        {
            const uint32_t dimension = ((bounce + 1u) * BounceSeedMultiplier)
                ^ ((areaLightIndex + 1u) * LightSeedMultiplier) ^ DirectBsdfScramble;
            const Float2 variates = Sample2D(samples, dimension);
            const Float3 direction = SampleDiffuseDirection(hit.Normal, variates.X, variates.Y, settings.Transport.ImportanceSampling);
            const float emitterCosine = Dot(emissionDirection, -direction);
            if (emitterCosine > 0.0f)
            {
                const float distance = Dot(position - light.Position, emissionDirection) / emitterCosine;
                const Float3 samplePosition = position + direction * distance;
                const Float3 offset = samplePosition - light.Position;
                const float bsdfPdf = DiffusePdf(Dot(hit.Normal, direction), settings.Transport.ImportanceSampling);
                if (distance > RayEpsilon && bsdfPdf > 0.0f
                    && std::fabs(Dot(offset, tangent)) <= light.Size.X * 0.5f
                    && std::fabs(Dot(offset, bitangent)) <= light.Size.Y * 0.5f)
                {
                    const float lightPdf = distance * distance / (area * emitterCosine);
                    const float weight = lightPdf / bsdfPdf
                        * MultipleImportance(bsdfPdf, AreaLightSamples * lightPdf, settings.Transport.Mis);
                    accumulateFiniteLight(samplePosition, lightColor, light.Intensity,
                        light.CastShadows, &emissionDirection, weight);
                }
            }
        }
        ++areaLightIndex;
    }

    return direct;
}

// Each analytic emitter has an independent light/BSDF estimate. Delta samples
// use discrete masses and never compete with a solid-angle light PDF. This also
// lights ideal glass/mirrors from analytic rectangles without an emissive mesh.
// Consequently this estimator owns registered emission after ALL physical
// continuations, including delta events; stochastic integrators suppress the duplicate hit.
Float3 PhysicalDirectLighting(const Scene& scene, const SceneAccelerator& accelerator,
    Float3 position, const SurfaceBsdf& bsdf, const RenderSettings& settings,
    AtomicStatistics& statistics, const CancellationProbe& cancellation,
    const SampleContext& samples, uint32_t bounce, const SpectralPath* spectral, float interiorMeanFreePath)
{
    const MediumOptions medium = spectral && spectral->Enabled() ? spectral->Medium() : settings.Medium;
    Float3 result;
    const bool importance = settings.Transport.ImportanceSampling;
    const auto beam = [&](Float3 direction, const Float3* endpoint) {
        const float distance = endpoint ? std::sqrt(Dot(*endpoint - position, *endpoint - position)) : MaximumRayDistance;
        const float interior = bsdf.ConnectionMeanFreePath(direction, interiorMeanFreePath);
        if (interior > 0)
        {
            const float transmission = std::exp(-distance / interior);
            return Float3(transmission, transmission, transmission);
        }
        return MediumTransmittance(medium, Ray{position, direction, 0, distance}, distance);
    };
    const auto visibility = [&](Float3 direction, const Float3* endpoint, bool shadows)
    {
        if (!shadows || !settings.ShadowsEnabled || settings.ShadowStrength <= 0) return 1.0f;
        Ray shadow;
        shadow.Origin = bsdf.Offset(position, direction);
        shadow.Direction = direction;
        shadow.MinDistance = RayEpsilon;
        shadow.MaxDistance = MaximumRayDistance;
        if (endpoint)
        {
            const Float3 segment = *endpoint - shadow.Origin;
            const float distance = std::sqrt(Dot(segment, segment));
            shadow.MaxDistance = distance - RayEpsilon;
            if (shadow.MaxDistance <= shadow.MinDistance) return 1.0f;
            shadow.Direction = segment / distance;
        }
        uint64_t tests = 0;
        statistics.ShadowRays.fetch_add(1, std::memory_order_relaxed);
        const bool blocked = accelerator.OccludedAtTime(shadow, tests, samples.SceneTime);
        statistics.PrimitiveTests.fetch_add(tests, std::memory_order_relaxed);
        return blocked ? 1 - std::clamp(settings.ShadowStrength, 0.0f, 1.0f) : 1.0f;
    };
    const auto accumulate = [&](Float3 direction, Float3 radiance, const Float3* endpoint,
                                bool shadows, float weight)
    {
        const SurfaceEvaluation evaluation = bsdf.Evaluate(direction, importance);
        if (evaluation.Value.X == 0 && evaluation.Value.Y == 0 && evaluation.Value.Z == 0) return;
        result = result + Multiply(Multiply(evaluation.Value, radiance), beam(direction, endpoint))
            * (bsdf.Cosine(direction) * weight * visibility(direction, endpoint, shadows));
    };
    for (const DirectionalLight& light : scene.DirectionalLights)
    {
        const Float3 lightColor = spectral ? spectral->DirectionalColor(scene, light) : light.Color;
        if (cancellation && cancellation()) return result;
        accumulate(Normalize(light.DirectionToLight), lightColor * light.Intensity,
                   nullptr, light.CastShadows, 1);
    }
    for (const PointLight& light : scene.PointLights)
    {
        const Float3 lightColor = spectral ? spectral->PointColor(scene, light) : light.Color;
        if (cancellation && cancellation()) return result;
        const Float3 offset = light.Position - position;
        const float distanceSquared = Dot(offset, offset);
        if (distanceSquared <= RayEpsilon * RayEpsilon) continue;
        accumulate(offset / std::sqrt(distanceSquared), lightColor * (light.Intensity / distanceSquared),
                   &light.Position, light.CastShadows, 1);
    }
    uint32_t lightIndex = static_cast<uint32_t>(scene.DirectionalLights.size() + scene.PointLights.size());
    for (const RectAreaLight& light : scene.RectAreaLights)
    {
        const Float3 lightColor = spectral ? spectral->RectColor(scene, light) : light.Color;
        if (cancellation && cancellation()) return result;
        const uint32_t dimension = (bounce + 1u) * BounceSeedMultiplier
            ^ (++lightIndex * LightSeedMultiplier);
        const float area = light.Size.X * light.Size.Y;
        if (area <= 0 || light.Intensity <= 0) continue;
        const Float3 normal = Normalize(light.Direction);
        const Float3 helper = std::abs(normal.Y) < BasisParallelThreshold ? Float3(0, 1, 0) : Float3(1, 0, 0);
        const Float3 tangent = Normalize(Cross(helper, normal)), bitangent = Cross(normal, tangent);
        const Float3 radiance = lightColor * (light.Intensity / area);
        const bool nee = settings.Transport.NextEventEstimation;
        for (uint32_t index = 0; nee && !bsdf.IsDelta() && index < AreaLightSamples; ++index)
        {
            const Float2 random = Sample2D(samples, dimension ^ ((index + 1u) * LightSampleSeedMultiplier));
            const Float2 uv{(static_cast<float>(index & 1u) + random.X) / AreaLightStrataPerAxis - 0.5f,
                            (static_cast<float>(index >> 1u) + random.Y) / AreaLightStrataPerAxis - 0.5f};
            const Float3 endpoint = light.Position + tangent * (uv.X * light.Size.X)
                + bitangent * (uv.Y * light.Size.Y);
            const Float3 offset = endpoint - position;
            const float squaredDistance = Dot(offset, offset);
            if (squaredDistance <= RayEpsilon * RayEpsilon) continue;
            const Float3 direction = offset / std::sqrt(squaredDistance);
            const float emitterCosine = Dot(normal, -direction);
            if (emitterCosine <= 0) continue;
            const float lightPdf = squaredDistance / (area * emitterCosine);
            const float bsdfPdf = bsdf.Evaluate(direction, importance).Pdf;
            const float weight = MultipleImportance(AreaLightSamples * lightPdf, bsdfPdf,
                settings.Transport.Mis) / (AreaLightSamples * lightPdf);
            accumulate(direction, radiance, &endpoint, light.CastShadows, weight);
        }
        if (bsdf.IsDelta() || !nee || settings.Transport.Mis != MisHeuristic::None)
        {
            const uint32_t bsdfDimension = dimension ^ DirectBsdfScramble;
            const SurfaceSample sample = bsdf.Sample(Sample2D(samples, bsdfDimension),
                Sample2D(samples, bsdfDimension ^ LobeSelectionScramble).X, importance);
            if (sample.Event == ScatteringNone || sample.Pdf <= 0) continue;
            const float emitterCosine = Dot(normal, -sample.Direction);
            if (emitterCosine <= 0) continue;
            const float distance = Dot(position - light.Position, normal) / emitterCosine;
            if (distance <= RayEpsilon) continue;
            const Float3 endpoint = position + sample.Direction * distance;
            const Float3 offset = endpoint - light.Position;
            if (std::abs(Dot(offset, tangent)) > light.Size.X * 0.5f
                || std::abs(Dot(offset, bitangent)) > light.Size.Y * 0.5f) continue;
            const bool delta = (sample.Event & ScatteringDelta) != 0;
            const float lightPdf = distance * distance / (area * emitterCosine);
            const float weight = delta || !nee ? 1.0f
                : MultipleImportance(sample.Pdf, AreaLightSamples * lightPdf, settings.Transport.Mis);
            result = result + Multiply(Multiply(sample.Weight, radiance), beam(sample.Direction, &endpoint))
                * (weight * visibility(sample.Direction, &endpoint, light.CastShadows));
        }
    }
    return result;
}
}
