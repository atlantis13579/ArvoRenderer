#include "RayTracer/Backends/CPU/CpuTransport.h"
#include "RayTracer/Backends/CPU/SurfaceScattering.h"
#include "RayTracer/Backends/CPU/HomogeneousMedium.h"
#include "RayTracer/Backends/CPU/VolumeLighting.h"
#include "RayTracer/Backends/CPU/SpectralTransport.h"
#include "RayTracer/Backends/CPU/SubsurfaceScattering.h"

#include <algorithm>
#include <cmath>

namespace Arvo::RayTracing::Cpu
{
namespace
{
MediumOptions InteriorMedium(const Material& material)
{
    MediumOptions medium;
    medium.Enabled = true;
    medium.Minimum = {-MaximumRayDistance, -MaximumRayDistance, -MaximumRayDistance};
    medium.Maximum = {MaximumRayDistance, MaximumRayDistance, MaximumRayDistance};
    medium.Anisotropy = material.SubsurfaceAnisotropy;
    const Float3 albedo = material.Albedo * material.SubsurfaceAlbedoScale;
    medium.Scattering = albedo / material.SubsurfaceMeanFreePath;
    medium.Absorption = (Float3(1, 1, 1) - albedo) / material.SubsurfaceMeanFreePath;
    return medium;
}
}
// Kajiya (1986), "The Rendering Equation": a camera random walk estimates
// indirect illumination. Continuation uses f*abs(cos(theta))/pdf and explicit
// event measures. The authored legacy diffuse/mirror mixture remains available.
// DirectLighting owns a separate light/BSDF MIS estimate. Its BSDF sample is
// independent of continuation, so registered emitter hits remain excluded
// after its estimated path classes. Legacy delta hits retain emission; physical
// delta direct lighting also owns a BSDF estimate and suppresses those duplicates.
Float3 PathTracing(const Scene& scene, const SceneAccelerator& accelerator, Ray ray,
    const RenderSettings& settings, AtomicStatistics& statistics,
    const CancellationProbe& cancellation, const SampleContext& samples)
{
    const SpectralPath spectral(samples, settings.Transport.ImportanceSampling);
    const MediumOptions exteriorMedium = spectral.Enabled() ? spectral.Medium() : settings.Medium;
    uint32_t interiorMaterial = InvalidRayPrimitiveIndex;
    Float3 radiance;
    Float3 throughput(1.0f, 1.0f, 1.0f);
    bool includeRegisteredEmission = true;
    float etaScale = 1.0f;
    for (uint32_t bounce = 0; bounce <= settings.MaxReflectionBounces; ++bounce)
    {
        if (cancellation && cancellation()) break;
        SurfaceHit hit;
        uint64_t primitiveTests = 0;
        const bool found = accelerator.IntersectAtTime(ray, hit, primitiveTests, samples.SceneTime);
        statistics.PrimitiveTests.fetch_add(primitiveTests, std::memory_order_relaxed);
        // A primary back face also identifies a camera starting inside a solid.
        // Scene validation guarantees closed, outward, nonoverlapping interiors.
        if (bounce == 0 && found && !hit.FrontFace && scene.Materials[hit.MaterialIndex].SubsurfaceMeanFreePath > 0)
            interiorMaterial = hit.MaterialIndex;
        Material interior;
        MediumOptions medium = exteriorMedium;
        if (interiorMaterial != InvalidRayPrimitiveIndex)
        {
            interior = spectral.EvaluateMaterial(scene.Materials[interiorMaterial], interiorMaterial);
            medium = InteriorMedium(interior);
        }
        if (medium.Enabled)
        {
            constexpr uint32_t FreeFlightScramble = 0x7a4d2e91u, PhaseScramble = 0x3c6ef372u;
            const uint32_t domain = (bounce + 1) * BounceSeedMultiplier;
            Ray segment = ray;
            segment.MinDistance = 0; // Geometry self-intersection offsets do not remove medium length.
            MediumSegmentSample flight;
            const auto flightVariates = Sample2D(samples, domain ^ FreeFlightScramble);
            if (interior.SubsurfaceMeanFreePath > 0)
            {
                const auto event = SampleSubsurfaceFlight(interior.SubsurfaceMeanFreePath,
                    interior.Albedo * interior.SubsurfaceAlbedoScale, ray.Direction,
                    found ? hit.Distance : ray.MaxDistance, flightVariates.X);
                flight = {event.Weight, event.Distance, event.Scattered};
            }
            else flight = SampleMedium(medium, segment, found ? hit.Distance : ray.MaxDistance, flightVariates);
            throughput = Multiply(throughput, flight.Weight);
            if (throughput.X == 0 && throughput.Y == 0 && throughput.Z == 0) break;
            if (flight.Scattered)
            {
                const Float3 position = ray.Origin + ray.Direction * flight.Distance;
                radiance = radiance + Multiply(throughput, VolumeDirectLighting(scene, accelerator, position,
                    ray.Direction, settings, statistics, cancellation, samples, bounce, &spectral, &medium));
                if (bounce == settings.MaxReflectionBounces) break;
                Float3 direction;
                if (interior.SubsurfaceMeanFreePath > 0)
                {
                    const auto phase = SampleSubsurfaceDirection(ray.Direction, medium.Anisotropy,
                        Sample2D(samples, domain ^ PhaseScramble), settings.Transport.ImportanceSampling);
                    direction = phase.Direction; throughput = throughput * phase.Weight;
                }
                else
                {
                    float pdf;
                    direction = SampleMediumDirection(ray.Direction, medium.Anisotropy,
                        Sample2D(samples, domain ^ PhaseScramble), settings.Transport.ImportanceSampling, pdf);
                    throughput = throughput * (MediumPhase(Dot(ray.Direction, direction), medium.Anisotropy) / pdf);
                }
                if (settings.Transport.RussianRoulette && bounce + 1 >= settings.Transport.RouletteStartDepth)
                {
                    const float survival = RouletteProbability(throughput * etaScale);
                    if (Sample2D(samples, domain ^ RouletteScramble).X >= survival) break;
                    throughput = throughput / survival;
                }
                ray = {position, direction, RayEpsilon, MaximumRayDistance};
                includeRegisteredEmission = false;
                statistics.ReflectionRays.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
        }
        if (!found)
        {
            radiance = radiance + Multiply(throughput, spectral.Environment(scene, ray));
            break;
        }

        const Material material = spectral.EvaluateMaterial(scene.Materials[hit.MaterialIndex], hit.MaterialIndex);
        const Float3 surface = spectral.Surface(scene, hit, material.DiffuseTextureIndex);
        const Float3 diffuse = Multiply(surface, material.Albedo);
        const Float3 position = ray.Origin + ray.Direction * hit.Distance;
        if (material.Model != MaterialModel::Legacy)
        {
            const SurfaceBsdf bsdf(material, hit, -ray.Direction, surface);
            const Float3 emission = includeRegisteredEmission || !material.SampledEmitter ? material.Emission : Float3();
            radiance = radiance + Multiply(throughput, emission
                + PhysicalDirectLighting(scene, accelerator, position, bsdf, settings,
                    statistics, cancellation, samples, bounce, &spectral, interior.SubsurfaceMeanFreePath));
            if (bounce == settings.MaxReflectionBounces) break;
            const uint32_t dimension = (bounce + 1u) * BounceSeedMultiplier;
            const SurfaceSample sample = bsdf.Sample(Sample2D(samples, dimension),
                Sample2D(samples, dimension ^ LobeSelectionScramble).X, settings.Transport.ImportanceSampling);
            if (sample.Event == ScatteringNone || sample.Pdf <= 0) break;
            throughput = Multiply(throughput, sample.Weight);
            if (throughput.X == 0 && throughput.Y == 0 && throughput.Z == 0) break;
            etaScale *= sample.RelativeIndex * sample.RelativeIndex;
            if (settings.Transport.RussianRoulette && bounce + 1u >= settings.Transport.RouletteStartDepth)
            {
                // PBRT v4, "A Better Path Tracer": remove the radiance eta^2
                // factor only for roulette decisions.
                // The actual throughput retains it, and opposite boundaries cancel.
                const float survival = RouletteProbability(throughput * etaScale);
                if (Sample2D(samples, dimension ^ RouletteScramble).X >= survival) break;
                throughput = throughput / survival;
            }
            ray.Origin = bsdf.Offset(position, sample.Direction);
            ray.Direction = sample.Direction;
            ray.MinDistance = RayEpsilon;
            ray.MaxDistance = MaximumRayDistance;
            if (material.SubsurfaceMeanFreePath > 0 && (sample.Event & ScatteringTransmission) != 0)
                interiorMaterial = hit.FrontFace ? hit.MaterialIndex : InvalidRayPrimitiveIndex;
            // PhysicalDirectLighting includes discrete BSDF-to-emitter samples.
            includeRegisteredEmission = false;
            statistics.ReflectionRays.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        const Float3 direct = DirectLighting(scene, accelerator, ray, hit, surface, settings,
            statistics, cancellation, samples, bounce, DirectLightingModel::Lambertian, false, &spectral);
        const float reflectivity = std::clamp(material.Reflectivity, 0.0f, 1.0f);
        const Float3 emission = includeRegisteredEmission || !material.SampledEmitter ? material.Emission : Float3();
        radiance = radiance + Multiply(throughput,
            emission + (direct - material.Emission) * (1.0f - reflectivity));
        if (bounce == settings.MaxReflectionBounces) break;

        const uint32_t dimension = (bounce + 1u) * BounceSeedMultiplier;
        const bool mirrorSelected = Sample2D(samples, dimension ^ LobeSelectionScramble).X < reflectivity;
        includeRegisteredEmission = mirrorSelected;
        Float3 outgoing;
        if (mirrorSelected)
        {
            outgoing = Normalize(Reflect(ray.Direction, hit.Normal));
        }
        else
        {
            const Float2 variates = Sample2D(samples, dimension);
            outgoing = SampleDiffuseDirection(hit.Normal, variates.X, variates.Y, settings.Transport.ImportanceSampling);
            throughput = Multiply(throughput, Float3(std::clamp(diffuse.X, 0.0f, 1.0f),
                std::clamp(diffuse.Y, 0.0f, 1.0f), std::clamp(diffuse.Z, 0.0f, 1.0f)));
            if (!settings.Transport.ImportanceSampling)
                throughput = throughput * (2.0f * std::max(0.0f, Dot(hit.Normal, outgoing)));
        }
        if (settings.Transport.RussianRoulette && bounce + 1u >= settings.Transport.RouletteStartDepth)
        {
            const float survival = RouletteProbability(throughput * etaScale);
            if (Sample2D(samples, dimension ^ RouletteScramble).X >= survival) break;
            throughput = throughput / survival;
        }
        ray.Origin = position + hit.Normal * RayEpsilon;
        ray.Direction = outgoing;
        ray.MinDistance = RayEpsilon;
        ray.MaxDistance = MaximumRayDistance;
        statistics.ReflectionRays.fetch_add(1, std::memory_order_relaxed);
    }
    return spectral.Film(radiance);
}
}
