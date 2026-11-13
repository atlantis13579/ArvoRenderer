#include "RayTracer/Backends/CPU/VolumeLighting.h"
#include "RayTracer/Backends/CPU/HomogeneousMedium.h"
#include "RayTracer/Backends/CPU/SpectralTransport.h"
#include <algorithm>
#include <cmath>

namespace Arvo::RayTracing::Cpu
{
// Volume NEE uses phase density rather than f*cos. Each analytic rectangle has
// independent light and phase proposals with sample-count MIS (Veach, 1997).
// Registered emitter hits after continuation belong to this direct estimator.
Float3 VolumeDirectLighting(const Scene& scene, const SceneAccelerator& accelerator, Float3 position,
                            Float3 incoming, const RenderSettings& settings, AtomicStatistics& statistics,
                            const CancellationProbe& cancellation, const SampleContext& samples,
                            uint32_t bounce, const SpectralPath* spectral, const MediumOptions* activeMedium)
{
    const MediumOptions medium = activeMedium ? *activeMedium :
        spectral && spectral->Enabled() ? spectral->Medium() : settings.Medium;
    Float3 result;
    const auto incident = [&](Float3 direction, float distance, Float3 emitted, bool shadows) {
        Ray connection{position, direction, 0, distance};
        const Float3 beam = MediumTransmittance(medium, connection, distance);
        float visible = 1;
        if (shadows && settings.ShadowsEnabled && settings.ShadowStrength > 0)
        {
            connection.MinDistance = RayEpsilon;
            connection.MaxDistance = distance - RayEpsilon;
            uint64_t tests = 0;
            statistics.ShadowRays.fetch_add(1, std::memory_order_relaxed);
            if (accelerator.OccludedAtTime(connection, tests, samples.SceneTime))
                visible -= std::clamp(settings.ShadowStrength, 0.0f, 1.0f);
            statistics.PrimitiveTests.fetch_add(tests, std::memory_order_relaxed);
        }
        return Multiply(beam, emitted) * visible;
    };
    for (const auto& light : scene.DirectionalLights)
    {
        const Float3 lightColor = spectral ? spectral->DirectionalColor(scene, light) : light.Color;
        if (light.Intensity <= 0)
            continue;
        const Float3 direction = Normalize(light.DirectionToLight);
        result = result +
                 incident(direction, MaximumRayDistance, lightColor * light.Intensity, light.CastShadows) *
                     MediumPhase(Dot(incoming, direction), medium.Anisotropy);
    }
    for (const auto& light : scene.PointLights)
    {
        const Float3 lightColor = spectral ? spectral->PointColor(scene, light) : light.Color;
        if (light.Intensity <= 0)
            continue;
        const Float3 delta = light.Position - position;
        const float squared = Dot(delta, delta);
        if (squared <= RayEpsilon * RayEpsilon)
            continue;
        const float distance = std::sqrt(squared);
        const Float3 direction = delta / distance;
        result = result +
                 incident(direction, distance, lightColor * (light.Intensity / squared), light.CastShadows) *
                     MediumPhase(Dot(incoming, direction), medium.Anisotropy);
    }
    const bool importance = settings.Transport.ImportanceSampling;
    const bool nee = settings.Transport.NextEventEstimation;
    constexpr uint32_t PhaseDirectScramble = 0x6c8e9cf5u;
    uint32_t lightIndex = uint32_t(scene.DirectionalLights.size() + scene.PointLights.size());
    for (const auto& light : scene.RectAreaLights)
    {
        const Float3 lightColor = spectral ? spectral->RectColor(scene, light) : light.Color;
        const uint32_t domain = (bounce + 1) * BounceSeedMultiplier ^ (++lightIndex * LightSeedMultiplier);
        if (cancellation && cancellation())
            return result;
        const float area = light.Size.X * light.Size.Y;
        if (area <= 0 || light.Intensity <= 0)
            continue;
        const Float3 normal = Normalize(light.Direction);
        const Float3 helper = std::abs(normal.Y) < BasisParallelThreshold ? Float3(0, 1, 0) : Float3(1, 0, 0);
        const Float3 tangent = Normalize(Cross(helper, normal)), bitangent = Cross(normal, tangent);
        const Float3 emitted = lightColor * (light.Intensity / area);
        for (uint32_t index = 0; nee && index < AreaLightSamples; ++index)
        {
            const auto u = Sample2D(samples, domain ^ ((index + 1) * LightSampleSeedMultiplier));
            const float x = (float(index % AreaLightStrataPerAxis) + u.X) / AreaLightStrataPerAxis - 0.5f;
            const float y = (float(index / AreaLightStrataPerAxis) + u.Y) / AreaLightStrataPerAxis - 0.5f;
            const Float3 endpoint =
                light.Position + tangent * (x * light.Size.X) + bitangent * (y * light.Size.Y);
            const Float3 offset = endpoint - position;
            const float squared = Dot(offset, offset);
            if (squared <= RayEpsilon * RayEpsilon)
                continue;
            const float distance = std::sqrt(squared);
            const Float3 direction = offset / distance;
            const float facing = Dot(normal, -direction);
            if (facing <= 0)
                continue;
            const float density = squared / (area * facing);
            const float phase = MediumPhase(Dot(incoming, direction), medium.Anisotropy);
            const float competitor = importance ? phase : 1 / (4 * Pi);
            const float weight =
                MultipleImportance(AreaLightSamples * density, competitor, settings.Transport.Mis) * phase /
                (AreaLightSamples * density);
            result = result + incident(direction, distance, emitted, light.CastShadows) * weight;
        }
        if (!nee || settings.Transport.Mis != MisHeuristic::None)
        {
            float density;
            const Float3 direction =
                SampleMediumDirection(incoming, medium.Anisotropy,
                                      Sample2D(samples, domain ^ PhaseDirectScramble), importance, density);
            const float facing = Dot(normal, -direction);
            if (facing <= 0 || density <= 0)
                continue;
            const float distance = Dot(position - light.Position, normal) / facing;
            if (distance <= RayEpsilon)
                continue;
            const Float3 offset = position + direction * distance - light.Position;
            if (std::abs(Dot(offset, tangent)) > light.Size.X / 2 ||
                std::abs(Dot(offset, bitangent)) > light.Size.Y / 2)
                continue;
            const float competitor = distance * distance / (area * facing);
            const float weight =
                (nee ? MultipleImportance(density, AreaLightSamples * competitor, settings.Transport.Mis)
                     : 1) *
                MediumPhase(Dot(incoming, direction), medium.Anisotropy) / density;
            result = result + incident(direction, distance, emitted, light.CastShadows) * weight;
        }
    }
    return result;
}
} // namespace Arvo::RayTracing::Cpu
