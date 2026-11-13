#include "RayTracer/Backends/CPU/CpuTransport.h"
#include "RayTracer/Backends/CPU/SurfaceScattering.h"
#include <algorithm>
#include <cmath>
#include <vector>

namespace Arvo::RayTracing::Cpu
{
namespace
{
bool IsBlack(Float3 value)
{
    return value.X == 0 && value.Y == 0 && value.Z == 0;
}
// Whitted's direct term uses deterministic area quadrature. Ideal events instead
// connect each specular direction to a rectangle with its discrete branch weight.
// These connections own registered emitter hits after physical continuations.
Float3 PhysicalDirect(const Scene &scene, const SceneAccelerator &accelerator, Float3 position,
                      const SurfaceBsdf &bsdf, const std::array<SurfaceSample, 2> &branches,
                      const RenderSettings &settings, AtomicStatistics &statistics,
                      const CancellationProbe &cancellation)
{
    Float3 result;
    const auto visible = [&](Float3 direction, const Float3 *endpoint, bool shadows) {
        if (!shadows || !settings.ShadowsEnabled || settings.ShadowStrength <= 0)
            return 1.0f;
        Ray shadow;
        shadow.Origin = bsdf.Offset(position, direction);
        shadow.Direction = direction;
        shadow.MinDistance = RayEpsilon;
        shadow.MaxDistance = MaximumRayDistance;
        if (endpoint)
        {
            const Float3 offset = *endpoint - shadow.Origin;
            const float distance = std::sqrt(Dot(offset, offset));
            shadow.MaxDistance = distance - RayEpsilon;
            if (shadow.MaxDistance <= shadow.MinDistance)
                return 1.0f;
            shadow.Direction = offset / distance;
        }
        uint64_t tests = 0;
        const bool blocked = accelerator.Occluded(shadow, tests);
        statistics.ShadowRays.fetch_add(1, std::memory_order_relaxed);
        statistics.PrimitiveTests.fetch_add(tests, std::memory_order_relaxed);
        return blocked ? 1 - std::clamp(settings.ShadowStrength, 0.0f, 1.0f) : 1;
    };
    const auto accumulate = [&](Float3 direction, Float3 radiance, const Float3 *endpoint, bool shadows,
                                float weight) {
        const auto evaluation = bsdf.Evaluate(direction, true);
        if (IsBlack(evaluation.Value))
            return;
        result = result + Multiply(evaluation.Value, radiance) *
                              (bsdf.Cosine(direction) * weight * visible(direction, endpoint, shadows));
    };
    for (const auto &light : scene.DirectionalLights)
    {
        if (cancellation && cancellation())
            return result;
        accumulate(Normalize(light.DirectionToLight), light.Color * light.Intensity, nullptr,
                   light.CastShadows, 1);
    }
    for (const auto &light : scene.PointLights)
    {
        if (cancellation && cancellation())
            return result;
        const Float3 offset = light.Position - position;
        const float squaredDistance = Dot(offset, offset);
        if (squaredDistance <= RayEpsilon * RayEpsilon)
            continue;
        accumulate(offset / std::sqrt(squaredDistance), light.Color * (light.Intensity / squaredDistance),
                   &light.Position, light.CastShadows, 1);
    }
    for (const auto &light : scene.RectAreaLights)
    {
        if (cancellation && cancellation())
            return result;
        const float area = light.Size.X * light.Size.Y;
        if (area <= 0 || light.Intensity <= 0)
            continue;
        const Float3 normal = Normalize(light.Direction);
        const Float3 helper = std::abs(normal.Y) < BasisParallelThreshold ? Float3(0, 1, 0) : Float3(1, 0, 0);
        const Float3 tangent = Normalize(Cross(helper, normal)), bitangent = Cross(normal, tangent);
        const Float3 radiance = light.Color * (light.Intensity / area);
        if (bsdf.IsDelta())
        {
            for (const auto &branch : branches)
            {
                if (branch.Event == ScatteringNone || branch.Pdf <= 0)
                    continue;
                const float emitterCosine = Dot(normal, -branch.Direction);
                if (emitterCosine <= 0)
                    continue;
                const float distance = Dot(position - light.Position, normal) / emitterCosine;
                if (distance <= RayEpsilon)
                    continue;
                const Float3 endpoint = position + branch.Direction * distance;
                const Float3 offset = endpoint - light.Position;
                if (std::abs(Dot(offset, tangent)) > light.Size.X * 0.5f ||
                    std::abs(Dot(offset, bitangent)) > light.Size.Y * 0.5f)
                    continue;
                result = result + Multiply(branch.Weight, radiance) *
                                      (branch.Pdf * visible(branch.Direction, &endpoint, light.CastShadows));
            }
            continue;
        }
        for (uint32_t index = 0; index < AreaLightSamples; ++index)
        {
            const float u = (float(index % AreaLightStrataPerAxis) + 0.5f) / AreaLightStrataPerAxis - 0.5f;
            const float v = (float(index / AreaLightStrataPerAxis) + 0.5f) / AreaLightStrataPerAxis - 0.5f;
            const Float3 endpoint =
                light.Position + tangent * (u * light.Size.X) + bitangent * (v * light.Size.Y);
            const Float3 offset = endpoint - position;
            const float squaredDistance = Dot(offset, offset);
            if (squaredDistance <= RayEpsilon * RayEpsilon)
                continue;
            const Float3 direction = offset / std::sqrt(squaredDistance);
            const float emitterCosine = Dot(normal, -direction);
            if (emitterCosine > 0)
                accumulate(direction, radiance, &endpoint, light.CastShadows,
                           area * emitterCosine / (AreaLightSamples * squaredDistance));
        }
    }
    return result;
}
struct Branch
{
    Ray RayValue;
    Float3 Throughput;
    uint32_t Depth;
    bool IncludeRegisteredEmission;
};
} // namespace

// Whitted (1980), "An Improved Illumination Model for Shaded Display".
// A depth-first stack splits ideal dielectric reflection AND transmission.
// Each branch keeps its Fresnel mass: there is no stochastic lobe selection,
// diffuse indirect continuation or roulette. Stack storage grows with depth;
// the number of traced specular branches can still grow exponentially.
Float3 Whitted(const Scene &scene, const SceneAccelerator &accelerator, Ray ray,
               const RenderSettings &settings, AtomicStatistics &statistics,
               const CancellationProbe &cancellation, const SampleContext &samples)
{
    Float3 radiance;
    std::vector<Branch> pending;
    pending.reserve(static_cast<size_t>(settings.MaxReflectionBounces) + 1);
    pending.push_back({ray, {1, 1, 1}, 0, true});
    while (!pending.empty())
    {
        if (cancellation && cancellation())
            break;
        const Branch branch = pending.back();
        pending.pop_back();
        ray = branch.RayValue;
        SurfaceHit hit;
        uint64_t tests = 0;
        const bool found = accelerator.Intersect(ray, hit, tests);
        statistics.PrimitiveTests.fetch_add(tests, std::memory_order_relaxed);
        if (!found)
        {
            radiance = radiance + Multiply(branch.Throughput, Background(scene, ray));
            continue;
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
            const auto events = bsdf.DeltaBranches();
            const Float3 emission =
                branch.IncludeRegisteredEmission || !material.SampledEmitter ? material.Emission : Float3();
            radiance =
                radiance + Multiply(branch.Throughput,
                                    emission + PhysicalDirect(scene, accelerator, position, bsdf, events,
                                                              settings, statistics, cancellation));
            if (branch.Depth == settings.MaxReflectionBounces)
                continue;
            for (const auto &event : events)
            {
                if (event.Event == ScatteringNone || event.Pdf <= 0)
                    continue;
                const Float3 throughput = Multiply(branch.Throughput, event.Weight) * event.Pdf;
                if (IsBlack(throughput))
                    continue;
                Ray continuation;
                continuation.Origin = bsdf.Offset(position, event.Direction);
                continuation.Direction = event.Direction;
                continuation.MinDistance = RayEpsilon;
                continuation.MaxDistance = MaximumRayDistance;
                pending.push_back({continuation, throughput, branch.Depth + 1, false});
                statistics.ReflectionRays.fetch_add(1, std::memory_order_relaxed);
            }
            continue;
        }
        // Preserve the existing demo's Blinn-Phong/mirror convention.
        Float3 direct =
            DirectLighting(scene, accelerator, ray, hit, surface, settings, statistics, cancellation, samples,
                           branch.Depth, DirectLightingModel::LegacyBlinnPhong);
        if (!branch.IncludeRegisteredEmission && material.SampledEmitter)
            direct = direct - material.Emission;
        const float reflectivity = std::clamp(material.Reflectivity, 0.0f, 1.0f);
        radiance = radiance + Multiply(branch.Throughput, direct) * (1 - reflectivity);
        if (reflectivity <= 0 || branch.Depth == settings.MaxReflectionBounces)
            continue;
        ray.Direction = Normalize(Reflect(ray.Direction, hit.Normal));
        const Float3 offsetNormal = Dot(hit.Normal, ray.Direction) >= 0 ? hit.Normal : -hit.Normal;
        ray.Origin = position + offsetNormal * RayEpsilon;
        ray.MinDistance = RayEpsilon;
        ray.MaxDistance = MaximumRayDistance;
        pending.push_back({ray, branch.Throughput * reflectivity, branch.Depth + 1, true});
        statistics.ReflectionRays.fetch_add(1, std::memory_order_relaxed);
    }
    return radiance;
}
} // namespace Arvo::RayTracing::Cpu
