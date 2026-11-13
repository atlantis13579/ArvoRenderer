#include "RayTracer/Backends/CPU/PhotonMapping.h"
#include "RayTracer/Backends/CPU/EmitterEndpoint.h"
#include "RayTracer/Backends/CPU/PointKdTree.h"
#include "RayTracer/Backends/CPU/SubsurfaceScattering.h"
#include "RayTracer/Backends/CPU/SurfaceScattering.h"
#include <algorithm>
#include <array>
#include <cmath>

namespace Arvo::RayTracing::Cpu
{
/* Jensen (1996), "Global Illumination using Photon Maps".
   PBRT 3e, section 16.2, "Stochastic Progressive Photon Mapping", derives the
   flux-density estimator used here; this implementation is the fixed-map,
   two-pass method, with no progressive radius/count update.
   https://pbr-book.org/3ed-2018/Light_Transport_III_Bidirectional_Methods/Stochastic_Progressive_Photon_Mapping

   Emit N paths, store incoming flux at continuous surfaces, and estimate
   radiance with sum(Phi*f)*K/N. K(r)=2(1-r^2/R^2)/(pi R^2) integrates to one
   on a disk. N counts emitted paths, including misses and null emissions.
   Direct photons, delta-prefix caustics and other indirect photons partition
   path space. NEE replaces direct photons only. The light walk uses importance
   scattering; the receiver converts geometric projected area to the shading
   normal measure. Fixed radius and geometric compatibility filters introduce
   bias; more camera samples alone cannot remove it. No other integrator is used.
   Interior collisions use a separate three-dimensional kernel in the same
   photon index. The camera samples a free-flight collision, then estimates its
   phase-weighted source from interior photons of the same closed solid.
*/
namespace
{
constexpr uint32_t PhotonBatch = 256, MaximumStoredPerPath = PhotonMaximumReflections + 1;
constexpr uint32_t EmissionDimensions = 5, ScatteringDimensions = 4;
constexpr uint32_t EmissionSourceOffset = 0, EmissionPositionOffset = 1, EmissionDirectionOffset = 3;
constexpr uint32_t ScatteringLobeOffset = 2, ScatteringRouletteOffset = 3;
constexpr uint32_t InteriorFlightOffset = ScatteringLobeOffset;
constexpr uint32_t PhotonDomain = 0x50484f54u, CameraDomain = 0x50484f43u;
constexpr uint32_t SourceDimension = 0x53524345u, PointDimension = 0x504f5349u,
                   DirectionDimension = 0x44495245u;
constexpr uint32_t SurfaceDimension = 0x53434154u, EventDimension = 0x45564e54u;
constexpr double KernelPi = 3.1415926535897932384626433832795;
// Suppress nearby disconnected sheets. These filters add finite-radius bias at
// corners/high curvature; they are not visibility or topology reconstruction.
constexpr double MinimumNormalAlignment = 0.9, MaximumNormalSeparation = 0.1;
constexpr double VolumeKernelIntegral = 8.0 / 15.0;
enum class PhotonClass : uint32_t
{
    Direct,
    Caustic,
    Indirect
};
struct Photon
{
    Float3 Position, Incoming, Flux, Normal;
    uint32_t LightDepth;
    PhotonClass Kind;
    uint32_t MaterialIndex = 0;
    bool Volume = false;
};
struct PhotonBundle
{
    std::array<Photon, MaximumStoredPerPath> Values;
    uint32_t Count = 0;
    bool Invalid = false;
};
struct Transport
{
    const Scene& SceneData;
    const SceneAccelerator& Traversal;
    const PreparedEmissionScene& Emitters;
    const RenderSettings& Settings;
    AtomicStatistics& Statistics;
    const CancellationProbe& Cancellation;
    bool Stopped() const { return Cancellation && Cancellation(); }
};
bool Finite(Float3 value)
{
    return std::isfinite(value.X) && std::isfinite(value.Y) && std::isfinite(value.Z);
}
bool Valid(Float3 value)
{
    return Finite(value) && value.X >= 0 && value.Y >= 0 && value.Z >= 0;
}
bool Positive(Float3 value)
{
    return value.X > 0 || value.Y > 0 || value.Z > 0;
}
Float3 Offset(Float3 position, Float3 normal, Float3 direction)
{
    return position + normal * (Dot(normal, direction) >= 0 ? RayEpsilon : -RayEpsilon);
}
float PhotonUniform(const PhotonMappingOptions& options, uint32_t path, uint32_t dimension)
{
    return Random01(Hash(path ^ Hash(options.Seed ^ PhotonDomain)) ^ Hash(dimension));
}
Float2 PhotonPair(const PhotonMappingOptions& options, uint32_t path, uint32_t dimension)
{
    return {PhotonUniform(options, path, dimension), PhotonUniform(options, path, dimension + 1)};
}
bool RectangleHit(const PackedEmitter& source, const Ray& ray, float& closest)
{
    if (source.Kind != EmitterKind::Rectangle)
        return false;
    const float projected = Dot(source.Direction, ray.Direction);
    if (projected == 0)
        return false;
    const float distance = Dot(source.Direction, source.Origin - ray.Origin) / projected;
    if (!(distance > ray.MinDistance && distance < ray.MaxDistance && distance <= closest))
        return false;
    const Float3 axis =
        std::abs(source.Direction.Y) < BasisParallelThreshold ? Float3(0, 1, 0) : Float3(1, 0, 0);
    const Float3 tangent = Normalize(Cross(axis, source.Direction)),
                 bitangent = Cross(source.Direction, tangent);
    const Float3 relative = ray.Origin + ray.Direction * distance - source.Origin;
    if (std::abs(Dot(relative, tangent)) > source.Width / 2 ||
        std::abs(Dot(relative, bitangent)) > source.Height / 2)
        return false;
    closest = distance;
    return true;
}
struct Endpoint
{
    SurfaceHit Hit;
    Float3 Position;
    uint32_t Source = InvalidEmitterIndex;
    bool Surface = false;
};
Endpoint Trace(const Transport& transport, const Ray& ray)
{
    Endpoint endpoint;
    uint64_t tests = 0;
    endpoint.Surface = transport.Traversal.Intersect(ray, endpoint.Hit, tests);
    transport.Statistics.PrimitiveTests.fetch_add(tests, std::memory_order_relaxed);
    const bool proxy = endpoint.Surface &&
                       transport.SceneData.Materials[endpoint.Hit.MaterialIndex].SampledEmitter;
    float closest = endpoint.Surface ? endpoint.Hit.Distance + (proxy ? RayEpsilon : 0) : ray.MaxDistance;
    for (uint32_t source = 0; source < transport.Emitters.Sources.size(); ++source)
        if (RectangleHit(transport.Emitters.Sources[source], ray, closest))
            endpoint.Source = source;
    if (endpoint.Source != InvalidEmitterIndex)
    {
        endpoint.Surface = false;
        endpoint.Position = ray.Origin + ray.Direction * closest;
        return endpoint;
    }
    if (endpoint.Surface)
        endpoint.Position = ray.Origin + ray.Direction * endpoint.Hit.Distance;
    for (uint32_t source = 0; source < transport.Emitters.Sources.size(); ++source)
    {
        const auto& emitter = transport.Emitters.Sources[source];
        if (endpoint.Surface ? emitter.Primitive == endpoint.Hit.PrimitiveIndex
                             : emitter.Kind == EmitterKind::Environment)
        {
            endpoint.Source = source;
            break;
        }
    }
    return endpoint;
}
Float3 Emission(const PackedEmitter& source, Float3 direction)
{
    if (source.Kind == EmitterKind::Environment)
    {
        const float vertical = (1 - direction.Y) / 2;
        return source.BackgroundBottom * (1 - vertical) + source.BackgroundTop * vertical;
    }
    if (source.Kind == EmitterKind::Rectangle)
        return Dot(source.Direction, direction) > 0 ? source.Strength / (source.Width * source.Height)
                                                    : Float3();
    return source.Strength;
}
Float3 SurfaceColor(const Scene& scene, const SurfaceHit& hit)
{
    Float3 color = hit.Color;
    if (hit.HasTexCoord)
        color =
            Multiply(color, SampleDiffuseTexture(
                                scene, scene.Materials[hit.MaterialIndex].DiffuseTextureIndex, hit.TexCoord));
    return color;
}
SurfaceSample LegacyMirror(const SurfaceHit& hit, Float3 view, TransportMode mode)
{
    const Float3 outgoing = Normalize(Reflect(-view, hit.Normal));
    const double viewGeometry = Dot(view, hit.GeometricNormal),
                 lightGeometry = Dot(outgoing, hit.GeometricNormal);
    const double viewShading = Dot(view, hit.Normal), lightShading = Dot(outgoing, hit.Normal);
    if (!(viewGeometry * lightGeometry > 0 && viewShading > 0 && lightShading > 0))
        return {};
    const float correction =
        mode == TransportMode::Importance
            ? float(std::abs(lightGeometry * viewShading / (viewGeometry * lightShading)))
            : 1;
    return {outgoing, Float3(correction, correction, correction), 1, 1,
            1,        ScatteringDelta | ScatteringReflection};
}
PhotonClass ClassifyPhoton(uint32_t depth, bool deltaPrefix)
{
    return depth == 0 ? PhotonClass::Direct : deltaPrefix ? PhotonClass::Caustic : PhotonClass::Indirect;
}
void EmitPath(const Transport& transport, uint32_t path, PhotonBundle& stored)
{
    const auto& settings = transport.Settings;
    auto options = settings.Photons;
    options.Seed ^= settings.SamplingSeed;
    const auto light =
        SampleEmitterEndpoint(transport.Emitters, PhotonUniform(options, path, EmissionSourceOffset),
                              PhotonPair(options, path, EmissionPositionOffset),
                              PhotonPair(options, path, EmissionDirectionOffset));
    if (!(light.Emission.Flags & EmissionValid))
        return;
    Float3 flux = light.Emission.Weight;
    Ray ray{Offset(light.Emission.Origin, light.Emission.Normal, light.Emission.Direction),
            light.Emission.Direction, RayEpsilon, MaximumRayDistance};
    bool deltaPrefix = true;
    uint32_t interiorMaterial = InvalidRayPrimitiveIndex;
    for (uint32_t depth = 0; depth <= settings.MaxReflectionBounces; ++depth)
    {
        if (transport.Stopped())
            return;
        if (!Valid(flux))
        {
            stored.Invalid = true;
            return;
        }
        if (!Positive(flux))
            return;
        transport.Statistics.ReflectionRays.fetch_add(1, std::memory_order_relaxed);
        const auto endpoint = Trace(transport, ray);
        const uint32_t dimension = EmissionDimensions + depth * ScatteringDimensions;
        if (endpoint.Surface && endpoint.Hit.MaterialIndex >= transport.SceneData.Materials.size())
        {
            stored.Invalid = true;
            return;
        }
        if (depth == 0 && endpoint.Surface && !endpoint.Hit.FrontFace &&
            transport.SceneData.Materials[endpoint.Hit.MaterialIndex].SubsurfaceMeanFreePath > 0)
            interiorMaterial = endpoint.Hit.MaterialIndex;
        if (interiorMaterial != InvalidRayPrimitiveIndex)
        {
            const auto& interior = transport.SceneData.Materials[interiorMaterial];
            const float boundary = endpoint.Surface || endpoint.Source != InvalidEmitterIndex
                ? std::sqrt(LengthSquared(endpoint.Position - ray.Origin)) : ray.MaxDistance;
            const auto flight = SampleSubsurfaceFlight(interior.SubsurfaceMeanFreePath,
                interior.Albedo * interior.SubsurfaceAlbedoScale, ray.Direction, boundary,
                PhotonUniform(options, path, dimension + InteriorFlightOffset));
            if (flight.Scattered)
            {
                const Float3 position = ray.Origin + ray.Direction * flight.Distance;
                stored.Values[stored.Count++] = {position, -ray.Direction, flux, {}, depth,
                                                 ClassifyPhoton(depth, deltaPrefix), interiorMaterial, true};
            }
            flux = Multiply(flux, flight.Weight);
            if (!Valid(flux))
            {
                stored.Invalid = true;
                return;
            }
            if (!Positive(flux))
                return;
            if (flight.Scattered)
            {
                if (depth == settings.MaxReflectionBounces) return;
                const auto phase = SampleSubsurfaceDirection(ray.Direction, interior.SubsurfaceAnisotropy,
                    PhotonPair(options, path, dimension), settings.Transport.ImportanceSampling);
                if (!(phase.Pdf > 0) || !std::isfinite(phase.Weight))
                {
                    stored.Invalid = true;
                    return;
                }
                flux = flux * phase.Weight;
                if (settings.Transport.RussianRoulette && depth + 1 >= settings.Transport.RouletteStartDepth)
                {
                    const float survival = RouletteProbability(flux);
                    if (PhotonUniform(options, path, dimension + ScatteringRouletteOffset) >= survival)
                        return;
                    flux = flux / survival;
                }
                deltaPrefix = false;
                ray = {ray.Origin + ray.Direction * flight.Distance, phase.Direction,
                       RayEpsilon, MaximumRayDistance};
                continue;
            }
        }
        if (!endpoint.Surface)
            return;
        const auto& hit = endpoint.Hit;
        if (!Finite(endpoint.Position))
        {
            stored.Invalid = true;
            return;
        }
        Material material = transport.SceneData.Materials[hit.MaterialIndex];
        const float mirror =
            material.Model == MaterialModel::Legacy ? std::clamp(material.Reflectivity, 0.0f, 1.0f) : 0;
        if (material.Model == MaterialModel::Legacy)
            material.Model = MaterialModel::Diffuse;
        const SurfaceBsdf bsdf(material, hit, -ray.Direction, SurfaceColor(transport.SceneData, hit),
                               TransportMode::Importance);
        if (!bsdf.IsDelta() && mirror < 1)
        {
            const PhotonClass kind = ClassifyPhoton(depth, deltaPrefix);
            stored.Values[stored.Count++] = {endpoint.Position,   -ray.Direction, flux,
                                             hit.GeometricNormal, depth, kind, hit.MaterialIndex, false};
        }
        if (depth == settings.MaxReflectionBounces)
            return;
        const float lobe = PhotonUniform(options, path, dimension + ScatteringLobeOffset);
        const SurfaceSample scatter = lobe < mirror
                                          ? LegacyMirror(hit, -ray.Direction, TransportMode::Importance)
                                          : bsdf.Sample(PhotonPair(options, path, dimension),
                                                        mirror > 0 ? (lobe - mirror) / (1 - mirror) : lobe,
                                                        settings.Transport.ImportanceSampling);
        if (!scatter.Event || !(scatter.Pdf > 0))
            return;
        flux = Multiply(flux, scatter.Weight);
        if (!Valid(flux))
        {
            stored.Invalid = true;
            return;
        }
        deltaPrefix = deltaPrefix && (scatter.Event & ScatteringDelta);
        if (settings.Transport.RussianRoulette && depth + 1 >= settings.Transport.RouletteStartDepth)
        {
            const float survival = RouletteProbability(flux);
            if (PhotonUniform(options, path, dimension + ScatteringRouletteOffset) >= survival)
                return;
            flux = flux / survival;
        }
        ray.Origin = bsdf.Offset(endpoint.Position, scatter.Direction);
        ray.Direction = scatter.Direction;
        if (material.SubsurfaceMeanFreePath > 0 && (scatter.Event & ScatteringTransmission) != 0)
            interiorMaterial = hit.FrontFace ? hit.MaterialIndex : InvalidRayPrimitiveIndex;
    }
}
bool Visible(const Transport& transport, Ray ray)
{
    transport.Statistics.ShadowRays.fetch_add(1, std::memory_order_relaxed);
    uint64_t tests = 0;
    bool blocked = transport.Traversal.Occluded(ray, tests);
    transport.Statistics.PrimitiveTests.fetch_add(tests, std::memory_order_relaxed);
    float closest = ray.MaxDistance;
    if (!blocked)
        for (const auto& emitter : transport.Emitters.Sources)
            if (RectangleHit(emitter, ray, closest))
            {
                blocked = true;
                break;
            }
    return !blocked;
}
Float3 Direct(const Transport& transport, const SampleContext& samples, uint32_t dimension, Float3 point,
              const SurfaceBsdf& bsdf, const SurfaceHit* subsurfaceBoundary = nullptr)
{
    const auto light = SampleEmitterEndpoint(
        transport.Emitters, Sample2D(samples, dimension ^ SourceDimension).X,
        Sample2D(samples, dimension ^ PointDimension), Sample2D(samples, dimension ^ DirectionDimension));
    const auto& emitted = light.Emission;
    if (!(emitted.Flags & EmissionValid))
        return {};
    const bool infinite = (emitted.Flags & EmissionInfinite) != 0,
               deltaPosition = (emitted.Flags & EmissionDeltaPosition) != 0;
    Float3 incoming = -emitted.Direction;
    double geometry = 1;
    if (!infinite)
    {
        const Float3 connection = emitted.Origin - point;
        const double distance =
            std::sqrt(double(connection.X) * connection.X + double(connection.Y) * connection.Y +
                      double(connection.Z) * connection.Z);
        if (!(distance > 0))
            return {};
        incoming = connection / float(distance);
        geometry = (deltaPosition ? 1 : std::abs(Dot(emitted.Normal, incoming))) / (distance * distance);
    }
    const bool transmitted = subsurfaceBoundary &&
        (subsurfaceBoundary->FrontFace ? Dot(incoming, subsurfaceBoundary->GeometricNormal) < 0
                                       : Dot(incoming, subsurfaceBoundary->GeometricNormal) > 0);
    // A sampled transmitted camera branch already handles non-delta source
    // hits. Delta point/directional lights have no intersectable endpoint.
    if (transmitted && !deltaPosition && !(emitted.Flags & EmissionDeltaDirection))
        return {};
    const double density =
        double(light.SelectionProbability) * (infinite ? emitted.DirectionPdf : emitted.PositionPdf);
    if (!(density > 0))
        return {};
    const auto scattering = bsdf.Evaluate(incoming, transport.Settings.Transport.ImportanceSampling);
    Float3 value =
        Multiply(scattering.Value, Emission(transport.Emitters.Sources[light.EmitterIndex], -incoming)) *
        float(double(bsdf.Cosine(incoming)) * geometry / density);
    if (transmitted && subsurfaceBoundary->FrontFace && deltaPosition)
        value = value * std::exp(-std::sqrt(LengthSquared(emitted.Origin - point)) /
            transport.SceneData.Materials[subsurfaceBoundary->MaterialIndex].SubsurfaceMeanFreePath);
    if (!Valid(value) || !Positive(value))
        return value;
    Ray shadow{bsdf.Offset(point, incoming), incoming, RayEpsilon, MaximumRayDistance};
    if (!infinite)
    {
        const Float3 target =
            deltaPosition ? emitted.Origin : Offset(emitted.Origin, emitted.Normal, -incoming);
        const Float3 connection = target - shadow.Origin;
        const float distance = std::sqrt(LengthSquared(connection));
        if (!(distance > 2 * RayEpsilon))
            return {};
        shadow.Direction = connection / distance;
        shadow.MaxDistance = distance - RayEpsilon;
    }
    return Visible(transport, shadow) ? value : Float3();
}
Float3 Gather(const Transport& transport, const PointKdTree& index, const std::vector<Photon>& photons,
              double radius, const Endpoint& endpoint, const SurfaceBsdf& bsdf, uint32_t cameraDepth)
{
    if (photons.empty())
        return {};
    const auto& settings = transport.Settings;
    const double radiusSquared = radius * radius,
                 normalization = 2 / (KernelPi * radiusSquared * settings.Photons.PhotonPaths);
    double red = 0, green = 0, blue = 0;
    index.Search(endpoint.Position, radiusSquared, [&](uint32_t id, double distanceSquared) {
        const Photon& photon = photons[id];
        if (photon.Volume)
            return;
        if (photon.LightDepth > settings.MaxReflectionBounces - cameraDepth)
            return;
        if (photon.Kind == PhotonClass::Direct && settings.Transport.NextEventEstimation)
            return;
        if (photon.Kind == PhotonClass::Caustic && !settings.Photons.Caustics)
            return;
        if (photon.Kind == PhotonClass::Indirect && !settings.Photons.Indirect)
            return;
        const auto& hit = endpoint.Hit;
        if (transport.SceneData.Materials[hit.MaterialIndex].SubsurfaceMeanFreePath > 0 &&
            (hit.FrontFace ? Dot(photon.Incoming, hit.GeometricNormal) <= 0
                           : Dot(photon.Incoming, hit.GeometricNormal) >= 0))
            return;
        if (Dot(hit.GeometricNormal, photon.Normal) < MinimumNormalAlignment)
            return;
        const Float3 separation = photon.Position - endpoint.Position;
        const double normalDistance = std::abs(double(separation.X) * hit.GeometricNormal.X +
                                               double(separation.Y) * hit.GeometricNormal.Y +
                                               double(separation.Z) * hit.GeometricNormal.Z);
        if (normalDistance > MaximumNormalSeparation * radius)
            return;
        const double projected = std::abs(Dot(hit.GeometricNormal, photon.Incoming));
        if (!(projected > 0))
            return;
        const auto scattering = bsdf.Evaluate(photon.Incoming, settings.Transport.ImportanceSampling);
        const double weight =
            normalization * (1 - distanceSquared / radiusSquared) * bsdf.Cosine(photon.Incoming) / projected;
        red += double(photon.Flux.X) * scattering.Value.X * weight;
        green += double(photon.Flux.Y) * scattering.Value.Y * weight;
        blue += double(photon.Flux.Z) * scattering.Value.Z * weight;
    });
    return {float(red), float(green), float(blue)};
}
Float3 GatherVolume(const Transport& transport, const PointKdTree& index,
                    const std::vector<Photon>& photons, double radius, Float3 position,
                    Float3 cameraDirection, uint32_t materialIndex, uint32_t cameraDepth)
{
    if (photons.empty() || !(radius > 0))
        return {};
    const auto& settings = transport.Settings;
    const auto& material = transport.SceneData.Materials[materialIndex];
    const double radiusSquared = radius * radius;
    // Collision photons have density sigma_t * incident radiance. The camera
    // free-flight sample already applies scattering albedo, so divide the
    // photon estimate by sigma_t = 1 / mean free path. The radial kernel
    // integrates to pi*R^3*(8/15) over a sphere.
    const double normalization = material.SubsurfaceMeanFreePath /
        (KernelPi * radius * radiusSquared * VolumeKernelIntegral * settings.Photons.PhotonPaths);
    double red = 0, green = 0, blue = 0;
    index.Search(position, radiusSquared, [&](uint32_t id, double distanceSquared) {
        const Photon& photon = photons[id];
        if (!photon.Volume || photon.MaterialIndex != materialIndex ||
            photon.LightDepth > settings.MaxReflectionBounces - cameraDepth)
            return;
        if (photon.Kind == PhotonClass::Caustic && !settings.Photons.Caustics)
            return;
        if (photon.Kind == PhotonClass::Indirect && !settings.Photons.Indirect)
            return;
        const Float3 separation = photon.Position - position;
        const float distance = std::sqrt(LengthSquared(separation));
        if (distance > 2 * RayEpsilon)
        {
            uint64_t tests = 0;
            const Ray segment{position, separation / distance, RayEpsilon, distance - RayEpsilon};
            if (transport.Traversal.Occluded(segment, tests))
            {
                transport.Statistics.PrimitiveTests.fetch_add(tests, std::memory_order_relaxed);
                return;
            }
            transport.Statistics.PrimitiveTests.fetch_add(tests, std::memory_order_relaxed);
        }
        const double phase = SubsurfacePhase(Dot(photon.Incoming, cameraDirection),
                                             material.SubsurfaceAnisotropy);
        const double weight = normalization * (1 - distanceSquared / radiusSquared) * phase;
        red += photon.Flux.X * weight;
        green += photon.Flux.Y * weight;
        blue += photon.Flux.Z * weight;
    });
    return {float(red), float(green), float(blue)};
}
Float3 CameraPath(const Transport& transport, const PointKdTree& index, const std::vector<Photon>& photons,
                  double radius, Ray ray, const SampleContext& samples)
{
    const auto& settings = transport.Settings;
    Float3 result, throughput(1, 1, 1);
    float etaScale = 1;
    uint32_t interiorMaterial = InvalidRayPrimitiveIndex;
    for (uint32_t depth = 0; depth <= settings.MaxReflectionBounces + 1; ++depth)
    {
        if (transport.Stopped())
            return {};
        if (!Valid(throughput) || !std::isfinite(etaScale))
            return {std::numeric_limits<float>::quiet_NaN(), 0, 0};
        if (!Positive(throughput))
            break;
        (depth == 0 ? transport.Statistics.PrimaryRays : transport.Statistics.ReflectionRays)
            .fetch_add(1, std::memory_order_relaxed);
        const auto endpoint = Trace(transport, ray);
        if (endpoint.Surface && endpoint.Hit.MaterialIndex >= transport.SceneData.Materials.size())
            return {std::numeric_limits<float>::quiet_NaN(), 0, 0};
        if (depth == 0 && endpoint.Surface && !endpoint.Hit.FrontFace &&
            transport.SceneData.Materials[endpoint.Hit.MaterialIndex].SubsurfaceMeanFreePath > 0)
            interiorMaterial = endpoint.Hit.MaterialIndex;
        if (interiorMaterial != InvalidRayPrimitiveIndex)
        {
            const auto& interior = transport.SceneData.Materials[interiorMaterial];
            const float boundary = endpoint.Surface || endpoint.Source != InvalidEmitterIndex
                ? std::sqrt(LengthSquared(endpoint.Position - ray.Origin)) : ray.MaxDistance;
            const uint32_t dimension = CameraDomain ^ (depth * BounceSeedMultiplier);
            const auto flight = SampleSubsurfaceFlight(interior.SubsurfaceMeanFreePath,
                interior.Albedo * interior.SubsurfaceAlbedoScale, ray.Direction, boundary,
                Sample2D(samples, dimension ^ SurfaceDimension).X);
            throughput = Multiply(throughput, flight.Weight);
            if (!Valid(throughput)) return {std::numeric_limits<float>::quiet_NaN(), 0, 0};
            if (!Positive(throughput)) break;
            if (flight.Scattered)
            {
                if (depth > settings.MaxReflectionBounces)
                    break;
                const Float3 position = ray.Origin + ray.Direction * flight.Distance;
                result = result + Multiply(throughput, GatherVolume(transport, index, photons,
                    radius, position, ray.Direction, interiorMaterial, depth));
                break;
            }
        }
        if (endpoint.Source != InvalidEmitterIndex)
            result = result + Multiply(throughput,
                                       Emission(transport.Emitters.Sources[endpoint.Source], -ray.Direction));
        if (!endpoint.Surface || depth > settings.MaxReflectionBounces)
            break;
        const auto& hit = endpoint.Hit;
        Material material = transport.SceneData.Materials[hit.MaterialIndex];
        const bool legacy = material.Model == MaterialModel::Legacy;
        const float mirror = legacy ? std::clamp(material.Reflectivity, 0.0f, 1.0f) : 0;
        if (legacy)
            material.Model = MaterialModel::Diffuse;
        const SurfaceBsdf bsdf(material, hit, -ray.Direction, SurfaceColor(transport.SceneData, hit));
        const bool subsurfaceTransmission = material.SubsurfaceMeanFreePath > 0 && !bsdf.IsDelta();
        const uint32_t dimension = CameraDomain ^ (depth * BounceSeedMultiplier);
        if (!bsdf.IsDelta() && mirror < 1)
        {
            Float3 illumination = Gather(transport, index, photons, radius, endpoint, bsdf, depth);
            if (settings.Transport.NextEventEstimation)
                illumination = illumination + Direct(transport, samples, dimension, endpoint.Position, bsdf,
                    subsurfaceTransmission ? &hit : nullptr);
            result = result + Multiply(throughput, illumination) * (1 - mirror);
        }
        SurfaceSample scatter;
        if (legacy && mirror > 0)
        {
            scatter = LegacyMirror(hit, -ray.Direction, TransportMode::Radiance);
            scatter.Weight = scatter.Weight * mirror;
        }
        else if (bsdf.IsDelta() || subsurfaceTransmission)
            scatter = bsdf.Sample(Sample2D(samples, dimension ^ SurfaceDimension),
                                  Sample2D(samples, dimension ^ EventDimension).X,
                                  settings.Transport.ImportanceSampling);
        else
            break;
        if (!scatter.Event || !(scatter.Pdf > 0))
            break;
        if (subsurfaceTransmission && !(scatter.Event & ScatteringTransmission))
            break;
        throughput = Multiply(throughput, scatter.Weight);
        etaScale *= scatter.RelativeIndex * scatter.RelativeIndex;
        if (!Valid(throughput) || !std::isfinite(etaScale))
            return {std::numeric_limits<float>::quiet_NaN(), 0, 0};
        if (settings.Transport.RussianRoulette && depth + 1 >= settings.Transport.RouletteStartDepth)
        {
            const float survival = RouletteProbability(throughput * etaScale);
            if (Sample2D(samples, dimension ^ RouletteScramble).X >= survival)
                break;
            throughput = throughput / survival;
        }
        ray.Origin = bsdf.Offset(endpoint.Position, scatter.Direction);
        ray.Direction = scatter.Direction;
        if (material.SubsurfaceMeanFreePath > 0 && (scatter.Event & ScatteringTransmission) != 0)
            interiorMaterial = hit.FrontFace ? hit.MaterialIndex : InvalidRayPrimitiveIndex;
    }
    return result;
}
} // namespace

bool RenderPhotonFrame(const Scene& scene, const SceneAccelerator& accelerator, const CameraFrame& camera,
                       const RenderSettings& settings, LinearFilm& film, AtomicStatistics& statistics,
                       const CancellationProbe& cancellation, const PhotonDispatcher& dispatch,
                       std::string& error, PhotonMappingDiagnostics* diagnostics)
{
    if (diagnostics)
        *diagnostics = {};
    if (!ValidateTileSettings(settings, error))
        return false;
    if (settings.MaxReflectionBounces > PhotonMaximumReflections || settings.Adaptive.Enabled ||
        settings.Transport.Mis != MisHeuristic::None || !settings.ShadowsEnabled ||
        settings.ShadowStrength != 1)
    {
        error = "Photon mapping requires bounded depth, fixed budgets, no connection MIS and full physical "
                "visibility.";
        return false;
    }
    const auto emitters = PrepareEmitters(scene);
    for (const auto& emitter : emitters.Sources)
        if (!(emitter.Flags & EmitterCastsShadows))
        {
            error = "Photon mapping requires shadows on every emitter.";
            return false;
        }
    const Transport transport{scene, accelerator, emitters, settings, statistics, cancellation};
    const double radius = double(emitters.Radius) * settings.Photons.RadiusScale;
    // An empty geometry scene has no density estimates, but retains its camera background.
    if (radius < 0 || !std::isfinite(radius))
    {
        error = "Photon domain radius must be finite and nonnegative.";
        return false;
    }
    if (diagnostics)
        diagnostics->Radius = radius;
    std::vector<Photon> photons;
    std::vector<PhotonBundle> batch(PhotonBatch);
    for (uint32_t first = 0; first < settings.Photons.PhotonPaths; first += PhotonBatch)
    {
        if (transport.Stopped())
            return false;
        const uint32_t count = std::min(PhotonBatch, settings.Photons.PhotonPaths - first);
        dispatch(count, [&](uint32_t lane) {
            batch[lane].Count = 0;
            batch[lane].Invalid = false;
            EmitPath(transport, first + lane, batch[lane]);
        });
        if (transport.Stopped())
            return false;
        for (uint32_t lane = 0; lane < count; ++lane)
        {
            if (batch[lane].Invalid)
            {
                error = "Photon emission produced nonfinite or negative flux.";
                return false;
            }
            for (uint32_t entry = 0; entry < batch[lane].Count; ++entry)
            {
                const auto& photon = batch[lane].Values[entry];
                photons.push_back(photon);
                if (diagnostics)
                {
                    if (photon.Kind == PhotonClass::Direct)
                        ++diagnostics->DirectPhotons;
                    if (photon.Kind == PhotonClass::Caustic)
                        ++diagnostics->CausticPhotons;
                    if (photon.Kind == PhotonClass::Indirect)
                        ++diagnostics->IndirectPhotons;
                }
            }
        }
        if (diagnostics)
            diagnostics->EmittedPaths += count;
    }
    std::vector<Float3> positions;
    positions.reserve(photons.size());
    for (const auto& photon : photons)
        positions.push_back(photon.Position);
    const PointKdTree index(std::move(positions), cancellation);
    if (index.WasCancelled() || transport.Stopped())
        return false;
    film = LinearFilm(settings.Width, settings.Height);
    std::atomic<bool> invalid{false};
    dispatch(settings.Width * settings.Height, [&](uint32_t pixel) {
        const uint32_t x = pixel % settings.Width, y = pixel / settings.Width;
        auto& target = film.Pixels[pixel];
        for (uint32_t sample = 0; sample < settings.SamplesPerPixel; ++sample)
        {
            if (transport.Stopped() || invalid.load(std::memory_order_relaxed))
                return;
            const auto sequence = MakeSampleContext(settings, x, y, sample);
            Ray ray = MakeCameraRay(camera, settings, sequence);
            if (camera.Lens.Radius > 0)
                ApplyThinLens(ray, camera, Sample2D(sequence, LensSampleDimension));
            const Float3 color = CameraPath(transport, index, photons, radius, ray, sequence);
            if (!Valid(color))
            {
                invalid.store(true, std::memory_order_relaxed);
                return;
            }
            AccumulateFilmSample(target, color);
            if (!Valid(target.MeanRadiance) || !std::isfinite(target.LuminanceM2))
            {
                invalid.store(true, std::memory_order_relaxed);
                return;
            }
        }
        statistics.CompletedPixels.fetch_add(1, std::memory_order_relaxed);
    });
    if (invalid.load(std::memory_order_relaxed))
    {
        error = "Photon density or camera transport produced invalid radiance.";
        return false;
    }
    return !transport.Stopped();
}
} // namespace Arvo::RayTracing::Cpu
