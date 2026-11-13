#include "RayTracer/Backends/CPU/CpuTransport.h"
#include "RayTracer/Backends/CPU/SurfaceScattering.h"
#include "RayTracer/Backends/CPU/SubsurfaceScattering.h"
#include "RayTracer/Backends/CPU/VolumeLighting.h"

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
// Cook, Porter and Carpenter (1984), "Distributed Ray Tracing".
// This camera walk distributes samples over area lights and a glossy reflection
// lobe, without diffuse indirect continuation. It is unrelated to network jobs.
// The normalized Phong lobe has f=rho*(n+2)/(2*pi)*cos(alpha)^n; sampling with
// p=(n+1)/(2*pi)*cos(alpha)^n gives f*cos(theta)/p below. Samples below the
// surface contribute zero, rather than being resampled with an incorrect PDF.
// Legacy direct shading is retained; lens and shutter sampling belong to camera.
// Physical materials use GGX reflection/transmission and Lambertian direct light.
// Diffuse surfaces terminate the walk; this deliberately omits diffuse GI.
// A dielectric solid can continue through sampled interior free-flight and HG
// phase events. This is a volume extension of the distributed camera walk;
// diffuse surface GI remains absent. Jensen et al. (2001), "A Practical Model
// for Subsurface Light Transport": https://graphics.stanford.edu/~srm/publications/SG01-subsurf-abstract.html
// https://graphics.pixar.com/library/DistributedRayTracing/paper.pdf
Float3 DistributedRayTracing(const Scene &scene, const SceneAccelerator &accelerator, Ray ray,
                             const RenderSettings &settings, AtomicStatistics &statistics,
                             const CancellationProbe &cancellation, const SampleContext &samples)
{
    Float3 radiance;
    Float3 throughput(1.0f, 1.0f, 1.0f);
    bool includeRegisteredEmission = true;
    uint32_t interiorMaterial = InvalidRayPrimitiveIndex;
    for (uint32_t bounce = 0; bounce <= settings.MaxReflectionBounces; ++bounce)
    {
        if (cancellation && cancellation())
            break;
        SurfaceHit hit;
        uint64_t tests = 0;
        // One shutter observation is retained by every continuation and light
        // connection; resampling time at a bounce would join different scenes.
        const bool found = accelerator.IntersectAtTime(ray, hit, tests, samples.SceneTime);
        statistics.PrimitiveTests.fetch_add(tests, std::memory_order_relaxed);
        if (bounce == 0 && found && !hit.FrontFace &&
            scene.Materials[hit.MaterialIndex].SubsurfaceMeanFreePath > 0)
            interiorMaterial = hit.MaterialIndex;
        if (interiorMaterial != InvalidRayPrimitiveIndex)
        {
            constexpr uint32_t FreeFlightScramble = 0x7a4d2e91u, PhaseScramble = 0x3c6ef372u;
            const uint32_t domain = (bounce + 1u) * BounceSeedMultiplier;
            const auto& interior = scene.Materials[interiorMaterial];
            const MediumOptions medium = InteriorMedium(interior);
            const auto flight = SampleSubsurfaceFlight(interior.SubsurfaceMeanFreePath,
                interior.Albedo * interior.SubsurfaceAlbedoScale, ray.Direction,
                found ? hit.Distance : ray.MaxDistance, Sample2D(samples, domain ^ FreeFlightScramble).X);
            throughput = Multiply(throughput, flight.Weight);
            if (throughput.X == 0 && throughput.Y == 0 && throughput.Z == 0) break;
            if (flight.Scattered)
            {
                const Float3 position = ray.Origin + ray.Direction * flight.Distance;
                radiance = radiance + Multiply(throughput, VolumeDirectLighting(scene, accelerator, position,
                    ray.Direction, settings, statistics, cancellation, samples, bounce, nullptr, &medium));
                if (bounce == settings.MaxReflectionBounces) break;
                const auto phase = SampleSubsurfaceDirection(ray.Direction, medium.Anisotropy,
                    Sample2D(samples, domain ^ PhaseScramble), settings.Transport.ImportanceSampling);
                throughput = throughput * phase.Weight;
                ray = {position, phase.Direction, RayEpsilon, MaximumRayDistance};
                includeRegisteredEmission = false;
                statistics.ReflectionRays.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
        }
        if (!found)
        {
            radiance = radiance + Multiply(throughput, Background(scene, ray));
            break;
        }
        const Material &material = scene.Materials[hit.MaterialIndex];
        const Float3 surface =
            hit.HasTexCoord
                ? Multiply(hit.Color, SampleDiffuseTexture(scene, material.DiffuseTextureIndex, hit.TexCoord))
                : hit.Color;
        const Float3 position = ray.Origin + ray.Direction * hit.Distance;
        if (material.Model != MaterialModel::Legacy)
        {
            const SurfaceBsdf bsdf(material, hit, -ray.Direction, surface);
            const Float3 emission =
                includeRegisteredEmission || !material.SampledEmitter ? material.Emission : Float3();
            radiance = radiance +
                       Multiply(throughput, emission + PhysicalDirectLighting(scene, accelerator, position,
                                                                              bsdf, settings, statistics,
                                                                              cancellation, samples, bounce,
                                                                              nullptr, interiorMaterial == InvalidRayPrimitiveIndex
                                                                                  ? 0 : scene.Materials[interiorMaterial].SubsurfaceMeanFreePath));
            if (material.Model == MaterialModel::Diffuse || bounce == settings.MaxReflectionBounces)
                break;
            // One visible-normal/Fresnel event continues each glossy or ideal ray.
            // Null samples retain their probability. No diffuse bounce or roulette
            // is hidden in this estimator; its available transport policy is fixed.
            const uint32_t dimension = (bounce + 1u) * BounceSeedMultiplier;
            const SurfaceSample continuation = bsdf.Sample(
                Sample2D(samples, dimension), Sample2D(samples, dimension ^ LobeSelectionScramble).X, true);
            if (continuation.Event == ScatteringNone || continuation.Pdf <= 0)
                break;
            throughput = Multiply(throughput, continuation.Weight);
            if (throughput.X == 0 && throughput.Y == 0 && throughput.Z == 0)
                break;
            ray.Origin = bsdf.Offset(position, continuation.Direction);
            ray.Direction = continuation.Direction;
            ray.MinDistance = RayEpsilon;
            ray.MaxDistance = MaximumRayDistance;
            if (material.SubsurfaceMeanFreePath > 0 && (continuation.Event & ScatteringTransmission) != 0)
                interiorMaterial = hit.FrontFace ? hit.MaterialIndex : InvalidRayPrimitiveIndex;
            // The independent physical direct estimator owns registered emitter
            // connections, including discrete events. Keep only unregistered hits.
            includeRegisteredEmission = false;
            statistics.ReflectionRays.fetch_add(1, std::memory_order_relaxed);
            continue;
        }
        Float3 direct =
            DirectLighting(scene, accelerator, ray, hit, surface, settings, statistics, cancellation, samples,
                           bounce, DirectLightingModel::LegacyBlinnPhong, true);
        if (!includeRegisteredEmission && material.SampledEmitter)
            direct = direct - material.Emission;
        const float reflectivity = std::clamp(material.Reflectivity, 0.0f, 1.0f);
        radiance = radiance + Multiply(throughput, direct) * (1.0f - reflectivity);
        if (reflectivity <= 0.0f || bounce == settings.MaxReflectionBounces)
            break;

        const float exponent = std::max(material.Shininess, 1.0f);
        const Float2 variates = Sample2D(samples, (bounce + 1u) * BounceSeedMultiplier);
        const float lobeCosine = std::pow(variates.X, 1.0f / (exponent + 1.0f));
        const float lobeSine = std::sqrt(std::max(0.0f, 1.0f - lobeCosine * lobeCosine));
        const float azimuth = 2.0f * Pi * variates.Y;
        const Float3 reflection = Normalize(Reflect(ray.Direction, hit.Normal));
        const Float3 helper = std::fabs(reflection.Y) < BasisParallelThreshold ? Float3(0.0f, 1.0f, 0.0f)
                                                                               : Float3(1.0f, 0.0f, 0.0f);
        const Float3 tangent = Normalize(Cross(helper, reflection));
        const Float3 bitangent = Cross(reflection, tangent);
        const Float3 outgoing =
            Normalize(tangent * (lobeSine * std::cos(azimuth)) + bitangent * (lobeSine * std::sin(azimuth)) +
                      reflection * lobeCosine);
        const float surfaceCosine = std::max(0.0f, Dot(hit.Normal, outgoing));
        if (surfaceCosine <= 0.0f)
            break;
        throughput = throughput * (reflectivity * (exponent + 2.0f) / (exponent + 1.0f) * surfaceCosine);
        ray.Origin = position + hit.Normal * RayEpsilon;
        ray.Direction = outgoing;
        ray.MinDistance = RayEpsilon;
        ray.MaxDistance = MaximumRayDistance;
        includeRegisteredEmission = true;
        statistics.ReflectionRays.fetch_add(1, std::memory_order_relaxed);
    }
    return radiance;
}
} // namespace Arvo::RayTracing::Cpu
