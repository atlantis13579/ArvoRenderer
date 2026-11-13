#include "RayTracer/Backends/CPU/ProgressivePhotonMapping.h"
#include "RayTracer/Backends/CPU/EmitterEndpoint.h"
#include "RayTracer/Backends/CPU/PointKdTree.h"
#include "RayTracer/Backends/CPU/SubsurfaceScattering.h"
#include "RayTracer/Backends/CPU/SurfaceScattering.h"
#include <algorithm>
#include <cmath>
#include <optional>

namespace Arvo::RayTracing::Cpu
{
/*
 Hachisuka and Jensen (2009), "Stochastic Progressive Photon Mapping".
 https://pbr-book.org/3ed-2018/Light_Transport_III_Bidirectional_Methods/Stochastic_Progressive_Photon_Mapping
 Every pass resamples a camera visible point, indexes visible points, then streams
 independent light paths into their disks. Pixel state survives visible-point
 replacement: N'=N+alpha*M, q=N'/(N+M), R'^2=q*R^2, tau'=q*(tau+beta*Phi).
 The final estimate is mean(direct)+tau/(pi*R^2*total emitted paths).
 A uniform disk is required by this recurrence, unlike the fixed-map integrator's
 Epanechnikov kernel. Averaging successive density estimates would be incorrect.
 Interior visible points use a uniform sphere and an independent cubic-radius
 recurrence. Reservoir selection over boundary/volume candidates keeps one
 indexed point per pixel while preserving each candidate's expected weight.

 Memory is O(pixels + a bounded path batch), independent of pass count and total
 emitted paths. Light work is parallel; arrival reduction follows path order for
 deterministic thread/crop replay. Crops replay every global emission pass and
 normalize by all attempted paths. No other transport integrator is called.
*/
namespace
{
constexpr double DensityPi = 3.1415926535897932384626433832795;
constexpr double VolumeMeasure = 4.0 / 3.0;
constexpr uint32_t PathBatch = 256, ArrivalsPerPath = ProgressivePhotonMaximumReflections + 1;
constexpr uint32_t LightDomain = 0x50504d4cu, PassDomain = 0x50415353u, CameraDomain = 0x50504d43u;
constexpr uint32_t SourceDomain = 0x534f5552u, PositionDomain = 0x504f534eu, DirectionDomain = 0x44495245u;
constexpr uint32_t ScatterDomain = 0x53434154u, EventDomain = 0x45564e54u, RouletteDomain = 0x524f554cu;
constexpr uint32_t VisibleDomain = 0x56495349u, InteriorDomain = 0x494e5445u;
constexpr uint32_t SourceDimension = 0, PositionDimension = 1, DirectionDimension = 3;
constexpr uint32_t EndpointDimensions = 5, BounceDimensions = 4, EventOffset = 2, RouletteOffset = 3;
// These geometric filters suppress disconnected-sheet leaks. Their finite-radius
// bias vanishes locally on smooth surfaces as the radius shrinks.
constexpr double MinimumNormalAgreement = 0.9, RelativePlaneThickness = 0.1;
using SpectrumSum = std::array<double, 3>;
struct Context
{
    const Scene& World;
    const SceneAccelerator& Accelerator;
    const PreparedEmissionScene& Sources;
    const RenderSettings& Settings;
    AtomicStatistics& Statistics;
    const CancellationProbe& Cancel;
    bool Stopped() const { return Cancel && Cancel(); }
};
struct Endpoint
{
    SurfaceHit Hit;
    Float3 Position;
    uint32_t Source = InvalidEmitterIndex;
    bool Surface = false;
};
struct VisiblePoint
{
    enum class Kind { None, Surface, Volume } Type = Kind::None;
    Float3 Position, Normal, Throughput;
    Float3 Direction;
    uint32_t MaterialIndex = InvalidRayPrimitiveIndex;
    bool FrontFace = true;
    // SurfaceBsdf borrows its material; fixed-size visible-point storage owns it
    // across the entire photon pass, including legacy-to-diffuse conversion.
    Material MaterialData;
    std::optional<SurfaceBsdf> Scattering;
    uint32_t Depth = 0;
};
struct Arrival
{
    Float3 Position, Normal, Incoming, Flux;
    uint32_t Depth = 0;
    uint32_t MaterialIndex = InvalidRayPrimitiveIndex;
    bool DeltaPrefix = false;
    bool Volume = false;
};
struct LightBatchEntry
{
    std::array<Arrival, ArrivalsPerPath> Arrivals;
    uint32_t Count = 0;
    bool Invalid = false;
};
struct PassDeposit
{
    SpectrumSum Flux{};
    uint64_t Count = 0;
};
bool Valid(Float3 value)
{
    return std::isfinite(value.X) && std::isfinite(value.Y) && std::isfinite(value.Z) && value.X >= 0 &&
           value.Y >= 0 && value.Z >= 0;
}
bool Positive(Float3 value)
{
    return value.X > 0 || value.Y > 0 || value.Z > 0;
}
bool Valid(const SpectrumSum& value)
{
    return std::all_of(value.begin(), value.end(),
                       [](double channel) { return channel >= 0 && std::isfinite(channel); });
}
SpectrumSum Product(Float3 a, Float3 b, double factor = 1)
{
    return {double(a.X) * b.X * factor, double(a.Y) * b.Y * factor, double(a.Z) * b.Z * factor};
}
void Add(SpectrumSum& target, const SpectrumSum& value)
{
    for (size_t channel = 0; channel < target.size(); ++channel)
        target[channel] += value[channel];
}
Float3 Offset(Float3 point, Float3 normal, Float3 direction)
{
    return point + normal * (Dot(normal, direction) >= 0 ? RayEpsilon : -RayEpsilon);
}
float LightRandom(const ProgressivePhotonOptions& options, uint32_t pass, uint32_t path, uint32_t dimension)
{
    const uint32_t seed = Hash(options.Seed ^ LightDomain) ^ Hash(pass ^ PassDomain);
    return Random01(seed ^ Hash(path) ^ Hash(dimension ^ SourceDomain));
}
Float2 LightPair(const ProgressivePhotonOptions& options, uint32_t pass, uint32_t path, uint32_t dimension)
{
    return {LightRandom(options, pass, path, dimension), LightRandom(options, pass, path, dimension + 1)};
}
bool Rectangle(const PackedEmitter& source, const Ray& ray, float& closest)
{
    if (source.Kind != EmitterKind::Rectangle)
        return false;
    const float cosine = Dot(source.Direction, ray.Direction);
    if (cosine == 0)
        return false;
    const float distance = Dot(source.Origin - ray.Origin, source.Direction) / cosine;
    if (!(distance > ray.MinDistance && distance < ray.MaxDistance && distance <= closest))
        return false;
    const Float3 axis =
        std::abs(source.Direction.Y) < BasisParallelThreshold ? Float3(0, 1, 0) : Float3(1, 0, 0);
    const Float3 right = Normalize(Cross(axis, source.Direction)), up = Cross(source.Direction, right);
    const Float3 relative = ray.Origin + ray.Direction * distance - source.Origin;
    if (std::abs(Dot(relative, right)) > source.Width / 2 || std::abs(Dot(relative, up)) > source.Height / 2)
        return false;
    closest = distance;
    return true;
}
Endpoint Intersect(const Context& context, const Ray& ray)
{
    Endpoint endpoint;
    uint64_t tests = 0;
    endpoint.Surface = context.Accelerator.Intersect(ray, endpoint.Hit, tests);
    context.Statistics.PrimitiveTests.fetch_add(tests, std::memory_order_relaxed);
    const bool proxy = endpoint.Surface && context.World.Materials[endpoint.Hit.MaterialIndex].SampledEmitter;
    float distance = endpoint.Surface ? endpoint.Hit.Distance + (proxy ? RayEpsilon : 0) : ray.MaxDistance;
    for (uint32_t source = 0; source < context.Sources.Sources.size(); ++source)
        if (Rectangle(context.Sources.Sources[source], ray, distance))
            endpoint.Source = source;
    if (endpoint.Source != InvalidEmitterIndex)
    {
        endpoint.Surface = false;
        endpoint.Position = ray.Origin + ray.Direction * distance;
        return endpoint;
    }
    if (endpoint.Surface)
        endpoint.Position = ray.Origin + ray.Direction * endpoint.Hit.Distance;
    for (uint32_t source = 0; source < context.Sources.Sources.size(); ++source)
    {
        const auto& emitter = context.Sources.Sources[source];
        if (endpoint.Surface ? emitter.Primitive != InvalidRayPrimitiveIndex &&
                                   emitter.Primitive == endpoint.Hit.PrimitiveIndex
                             : emitter.Kind == EmitterKind::Environment)
        {
            endpoint.Source = source;
            break;
        }
    }
    return endpoint;
}
Float3 Radiance(const PackedEmitter& source, Float3 outgoing)
{
    if (source.Kind == EmitterKind::Environment)
    {
        const float blend = (1 - outgoing.Y) / 2;
        return source.BackgroundBottom * (1 - blend) + source.BackgroundTop * blend;
    }
    if (source.Kind == EmitterKind::Rectangle)
        return Dot(source.Direction, outgoing) > 0 ? source.Strength / (source.Width * source.Height)
                                                   : Float3{};
    return source.Strength;
}
Float3 Color(const Context& context, const SurfaceHit& hit)
{
    return hit.HasTexCoord
               ? Multiply(hit.Color,
                          SampleDiffuseTexture(context.World,
                                               context.World.Materials[hit.MaterialIndex].DiffuseTextureIndex,
                                               hit.TexCoord))
               : hit.Color;
}
SurfaceSample Mirror(const SurfaceHit& hit, Float3 view, TransportMode mode)
{
    const Float3 reflected = Normalize(Reflect(-view, hit.Normal));
    const double incomingGeometry = Dot(view, hit.GeometricNormal),
                 outgoingGeometry = Dot(reflected, hit.GeometricNormal);
    const double incomingShading = Dot(view, hit.Normal), outgoingShading = Dot(reflected, hit.Normal);
    if (!(incomingGeometry * outgoingGeometry > 0 && incomingShading > 0 && outgoingShading > 0))
        return {};
    const float correction =
        mode == TransportMode::Importance
            ? float(std::abs(outgoingGeometry * incomingShading / (incomingGeometry * outgoingShading)))
            : 1;
    return {reflected, {correction, correction, correction}, 1, 1, 1, ScatteringDelta | ScatteringReflection};
}
SpectrumSum Direct(const Context& context, const Endpoint& receiver, const SurfaceBsdf& bsdf,
                   const SampleContext& random, uint32_t domain, const SurfaceHit* subsurfaceBoundary)
{
    const auto chosen = SampleEmitterEndpoint(context.Sources, Sample2D(random, domain ^ SourceDomain).X,
                                              Sample2D(random, domain ^ PositionDomain),
                                              Sample2D(random, domain ^ DirectionDomain));
    const auto& sample = chosen.Emission;
    if (!(sample.Flags & EmissionValid))
        return {};
    const bool infinite = (sample.Flags & EmissionInfinite) != 0,
               point = (sample.Flags & EmissionDeltaPosition) != 0;
    Float3 direction = sample.Direction * -1;
    double geometry = 1;
    if (!infinite)
    {
        const Float3 segment = sample.Origin - receiver.Position;
        const double distanceSquared =
            double(segment.X) * segment.X + double(segment.Y) * segment.Y + double(segment.Z) * segment.Z;
        if (!(distanceSquared > 0) || !std::isfinite(distanceSquared))
            return {};
        direction = segment / float(std::sqrt(distanceSquared));
        geometry = (point ? 1 : std::abs(Dot(sample.Normal, direction))) / distanceSquared;
    }
    const bool transmitted = subsurfaceBoundary &&
        (subsurfaceBoundary->FrontFace ? Dot(direction, subsurfaceBoundary->GeometricNormal) < 0
                                       : Dot(direction, subsurfaceBoundary->GeometricNormal) > 0);
    if (transmitted && !point && !(sample.Flags & EmissionDeltaDirection))
        return {};
    const double density =
        double(chosen.SelectionProbability) * (infinite ? sample.DirectionPdf : sample.PositionPdf);
    if (!(density > 0))
        return {};
    const auto scattering = bsdf.Evaluate(direction, context.Settings.Transport.ImportanceSampling);
    auto value =
        Product(scattering.Value, Radiance(context.Sources.Sources[chosen.EmitterIndex], direction * -1),
                geometry * bsdf.Cosine(direction) / density);
    if (transmitted && subsurfaceBoundary->FrontFace && point)
    {
        const double distance = std::sqrt(LengthSquared(sample.Origin - receiver.Position));
        const double attenuation = std::exp(-distance /
            context.World.Materials[subsurfaceBoundary->MaterialIndex].SubsurfaceMeanFreePath);
        for (double& channel : value) channel *= attenuation;
    }
    Ray shadow{bsdf.Offset(receiver.Position, direction), direction, RayEpsilon, MaximumRayDistance};
    if (!infinite)
    {
        const Float3 segment = sample.Origin - shadow.Origin;
        const float distance = std::sqrt(LengthSquared(segment));
        if (!(distance > 2 * RayEpsilon))
            return {};
        shadow.Direction = segment / distance;
        shadow.MaxDistance = distance - RayEpsilon;
    }
    context.Statistics.ShadowRays.fetch_add(1, std::memory_order_relaxed);
    uint64_t tests = 0;
    const bool blocked = context.Accelerator.Occluded(shadow, tests);
    context.Statistics.PrimitiveTests.fetch_add(tests, std::memory_order_relaxed);
    if (blocked)
        return {};
    float closest = shadow.MaxDistance;
    for (const auto& source : context.Sources.Sources)
        if (Rectangle(source, shadow, closest))
            return {};
    return value;
}
bool CameraPass(const Context& context, const CameraFrame& camera, uint32_t pixel, uint32_t pass,
                VisiblePoint& visible, SpectrumSum& direct)
{
    visible.Type = VisiblePoint::Kind::None;
    visible.Scattering.reset();
    const auto& settings = context.Settings;
    const auto random = MakeSampleContext(settings, pixel % settings.Width, pixel / settings.Width, pass);
    Ray ray = MakeCameraRay(camera, settings, random);
    if (camera.Lens.Radius > 0)
        ApplyThinLens(ray, camera, Sample2D(random, LensSampleDimension));
    Float3 beta(1, 1, 1);
    float etaScale = 1;
    uint32_t interiorMaterial = InvalidRayPrimitiveIndex, candidates = 0;
    const auto finish = [&]() {
        if (candidates) visible.Throughput = visible.Throughput * float(candidates);
        return Valid(direct);
    };
    for (uint32_t depth = 0; depth <= settings.MaxReflectionBounces + 1; ++depth)
    {
        if (context.Stopped())
            return false;
        if (!Valid(beta) || !std::isfinite(etaScale))
            return false;
        if (!Positive(beta))
            return finish();
        (depth == 0 ? context.Statistics.PrimaryRays : context.Statistics.ReflectionRays)
            .fetch_add(1, std::memory_order_relaxed);
        const auto endpoint = Intersect(context, ray);
        const uint32_t domain = CameraDomain ^ (depth * BounceSeedMultiplier);
        if (endpoint.Surface && endpoint.Hit.MaterialIndex >= context.World.Materials.size())
            return false;
        if (depth == 0 && endpoint.Surface && !endpoint.Hit.FrontFace &&
            context.World.Materials[endpoint.Hit.MaterialIndex].SubsurfaceMeanFreePath > 0)
            interiorMaterial = endpoint.Hit.MaterialIndex;
        if (interiorMaterial != InvalidRayPrimitiveIndex)
        {
            const auto& interior = context.World.Materials[interiorMaterial];
            const float boundary = endpoint.Surface || endpoint.Source != InvalidEmitterIndex
                ? std::sqrt(LengthSquared(endpoint.Position - ray.Origin)) : ray.MaxDistance;
            const auto flight = SampleSubsurfaceFlight(interior.SubsurfaceMeanFreePath,
                interior.Albedo * interior.SubsurfaceAlbedoScale, ray.Direction, boundary,
                Sample2D(random, domain ^ InteriorDomain).X);
            beta = Multiply(beta, flight.Weight);
            if (!Valid(beta)) return false;
            if (!Positive(beta)) return finish();
            if (flight.Scattered)
            {
                if (depth > settings.MaxReflectionBounces) return finish();
                ++candidates;
                if (candidates == 1 || Sample2D(random, domain ^ VisibleDomain).X < 1.0f / candidates)
                {
                    visible.Type = VisiblePoint::Kind::Volume;
                    visible.Position = ray.Origin + ray.Direction * flight.Distance;
                    visible.Direction = ray.Direction;
                    visible.MaterialIndex = interiorMaterial;
                    visible.Throughput = beta;
                    visible.Depth = depth;
                    visible.Scattering.reset();
                }
                return finish();
            }
        }
        if (endpoint.Source != InvalidEmitterIndex)
            Add(direct,
                Product(beta, Radiance(context.Sources.Sources[endpoint.Source], ray.Direction * -1)));
        if (!endpoint.Surface || depth > settings.MaxReflectionBounces)
            return finish();
        Material material = context.World.Materials[endpoint.Hit.MaterialIndex];
        const bool legacy = material.Model == MaterialModel::Legacy;
        const float mirror = legacy ? std::clamp(material.Reflectivity, 0.0f, 1.0f) : 0;
        if (legacy)
            material.Model = MaterialModel::Diffuse;
        const Float3 view = ray.Direction * -1;
        const SurfaceBsdf bsdf(material, endpoint.Hit, view, Color(context, endpoint.Hit));
        const bool subsurfaceTransmission = material.SubsurfaceMeanFreePath > 0 && !bsdf.IsDelta();
        if (!bsdf.IsDelta() && mirror < 1 && settings.Transport.NextEventEstimation)
        {
            auto estimate = Direct(context, endpoint, bsdf, random, domain,
                                   subsurfaceTransmission ? &endpoint.Hit : nullptr);
            const auto weight = Product(beta, Float3(1, 1, 1), 1 - mirror);
            for (size_t channel = 0; channel < estimate.size(); ++channel)
                estimate[channel] *= weight[channel];
            Add(direct, estimate);
        }
        // Select the legacy mirror/diffuse mixture once. For a diffuse visible
        // point its (1-r) coefficient cancels its selection probability.
        const bool chooseMirror = legacy && Sample2D(random, domain ^ EventDomain).X < mirror;
        if (!chooseMirror && !bsdf.IsDelta())
        {
            ++candidates;
            if (candidates == 1 || Sample2D(random, domain ^ VisibleDomain).X < 1.0f / candidates)
            {
                visible.Type = VisiblePoint::Kind::Surface;
                visible.Position = endpoint.Position;
                visible.Normal = endpoint.Hit.GeometricNormal;
                visible.Throughput = beta;
                visible.Depth = depth;
                visible.MaterialIndex = endpoint.Hit.MaterialIndex;
                visible.FrontFace = endpoint.Hit.FrontFace;
                visible.MaterialData = material;
                visible.Scattering.emplace(visible.MaterialData, endpoint.Hit, view,
                                           Color(context, endpoint.Hit));
            }
            if (!subsurfaceTransmission) return finish();
        }
        const auto scatter = chooseMirror ? Mirror(endpoint.Hit, view, TransportMode::Radiance)
                                          : bsdf.Sample(Sample2D(random, domain ^ ScatterDomain),
                                                        Sample2D(random, domain ^ EventDomain).X,
                                                        settings.Transport.ImportanceSampling);
        if (!scatter.Event || !(scatter.Pdf > 0))
            return finish();
        if (subsurfaceTransmission && !(scatter.Event & ScatteringTransmission))
            return finish();
        beta = Multiply(beta, scatter.Weight);
        etaScale *= scatter.RelativeIndex * scatter.RelativeIndex;
        if (!Valid(beta) || !std::isfinite(etaScale))
            return false;
        if (settings.Transport.RussianRoulette && depth + 1 >= settings.Transport.RouletteStartDepth)
        {
            const float survival = RouletteProbability(beta * etaScale);
            if (Sample2D(random, domain ^ RouletteDomain).X >= survival)
                return finish();
            beta = beta / survival;
        }
        ray = {bsdf.Offset(endpoint.Position, scatter.Direction), scatter.Direction, RayEpsilon,
               MaximumRayDistance};
        if (material.SubsurfaceMeanFreePath > 0 && (scatter.Event & ScatteringTransmission) != 0)
            interiorMaterial = endpoint.Hit.FrontFace ? endpoint.Hit.MaterialIndex : InvalidRayPrimitiveIndex;
    }
    return finish();
}
void LightPass(const Context& context, uint32_t pass, uint32_t path, LightBatchEntry& output)
{
    output.Count = 0;
    output.Invalid = false;
    const auto& settings = context.Settings;
    auto options = settings.ProgressivePhotons;
    options.Seed ^= settings.SamplingSeed;
    const auto endpoint =
        SampleEmitterEndpoint(context.Sources, LightRandom(options, pass, path, SourceDimension),
                              LightPair(options, pass, path, PositionDimension),
                              LightPair(options, pass, path, DirectionDimension));
    const auto& emission = endpoint.Emission;
    if (!(emission.Flags & EmissionValid))
        return;
    Float3 flux = emission.Weight;
    Ray ray{Offset(emission.Origin, emission.Normal, emission.Direction), emission.Direction, RayEpsilon,
            MaximumRayDistance};
    bool deltaPrefix = true;
    uint32_t interiorMaterial = InvalidRayPrimitiveIndex;
    for (uint32_t depth = 0; depth <= settings.MaxReflectionBounces; ++depth)
    {
        if (context.Stopped())
            return;
        if (!Valid(flux))
        {
            output.Invalid = true;
            return;
        }
        if (!Positive(flux))
            return;
        context.Statistics.ReflectionRays.fetch_add(1, std::memory_order_relaxed);
        const auto vertex = Intersect(context, ray);
        const uint32_t dimension = EndpointDimensions + depth * BounceDimensions;
        if (vertex.Surface && vertex.Hit.MaterialIndex >= context.World.Materials.size())
        {
            output.Invalid = true;
            return;
        }
        if (depth == 0 && vertex.Surface && !vertex.Hit.FrontFace &&
            context.World.Materials[vertex.Hit.MaterialIndex].SubsurfaceMeanFreePath > 0)
            interiorMaterial = vertex.Hit.MaterialIndex;
        if (interiorMaterial != InvalidRayPrimitiveIndex)
        {
            const auto& interior = context.World.Materials[interiorMaterial];
            const float boundary = vertex.Surface || vertex.Source != InvalidEmitterIndex
                ? std::sqrt(LengthSquared(vertex.Position - ray.Origin)) : ray.MaxDistance;
            const auto flight = SampleSubsurfaceFlight(interior.SubsurfaceMeanFreePath,
                interior.Albedo * interior.SubsurfaceAlbedoScale, ray.Direction, boundary,
                LightRandom(options, pass, path, dimension + EventOffset));
            if (flight.Scattered)
                output.Arrivals[output.Count++] = {ray.Origin + ray.Direction * flight.Distance, {},
                    -ray.Direction, flux, depth, interiorMaterial, deltaPrefix, true};
            flux = Multiply(flux, flight.Weight);
            if (!Valid(flux))
            {
                output.Invalid = true;
                return;
            }
            if (!Positive(flux)) return;
            if (flight.Scattered)
            {
                if (depth == settings.MaxReflectionBounces) return;
                const auto phase = SampleSubsurfaceDirection(ray.Direction, interior.SubsurfaceAnisotropy,
                    LightPair(options, pass, path, dimension), settings.Transport.ImportanceSampling);
                if (!(phase.Pdf > 0) || !std::isfinite(phase.Weight))
                {
                    output.Invalid = true;
                    return;
                }
                flux = flux * phase.Weight;
                if (settings.Transport.RussianRoulette && depth + 1 >= settings.Transport.RouletteStartDepth)
                {
                    const float survival = RouletteProbability(flux);
                    if (LightRandom(options, pass, path, dimension + RouletteOffset) >= survival) return;
                    flux = flux / survival;
                }
                deltaPrefix = false;
                ray = {ray.Origin + ray.Direction * flight.Distance, phase.Direction,
                       RayEpsilon, MaximumRayDistance};
                continue;
            }
        }
        if (!vertex.Surface)
            return;
        Material material = context.World.Materials[vertex.Hit.MaterialIndex];
        const bool legacy = material.Model == MaterialModel::Legacy;
        const float mirror = legacy ? std::clamp(material.Reflectivity, 0.0f, 1.0f) : 0;
        if (legacy)
            material.Model = MaterialModel::Diffuse;
        const Float3 view = ray.Direction * -1;
        const SurfaceBsdf bsdf(material, vertex.Hit, view, Color(context, vertex.Hit),
                               TransportMode::Importance);
        if (!bsdf.IsDelta() && mirror < 1)
            output.Arrivals[output.Count++] = {vertex.Position, vertex.Hit.GeometricNormal, view, flux, depth,
                                               vertex.Hit.MaterialIndex, deltaPrefix, false};
        if (depth == settings.MaxReflectionBounces)
            return;
        const float event = LightRandom(options, pass, path, dimension + EventOffset);
        const auto scatter = legacy && event < mirror
                                 ? Mirror(vertex.Hit, view, TransportMode::Importance)
                                 : bsdf.Sample(LightPair(options, pass, path, dimension), event,
                                               settings.Transport.ImportanceSampling);
        if (!scatter.Event || !(scatter.Pdf > 0))
            return;
        flux = Multiply(flux, scatter.Weight);
        if (!Valid(flux))
        {
            output.Invalid = true;
            return;
        }
        deltaPrefix = deltaPrefix && (scatter.Event & ScatteringDelta) != 0;
        if (settings.Transport.RussianRoulette && depth + 1 >= settings.Transport.RouletteStartDepth)
        {
            const float survival = RouletteProbability(flux);
            if (LightRandom(options, pass, path, dimension + RouletteOffset) >= survival)
                return;
            flux = flux / survival;
        }
        ray = {bsdf.Offset(vertex.Position, scatter.Direction), scatter.Direction, RayEpsilon,
               MaximumRayDistance};
        if (material.SubsurfaceMeanFreePath > 0 && (scatter.Event & ScatteringTransmission) != 0)
            interiorMaterial = vertex.Hit.FrontFace ? vertex.Hit.MaterialIndex : InvalidRayPrimitiveIndex;
    }
}
bool Accept(const Arrival& photon, const VisiblePoint& visible, const RenderSettings& settings)
{
    if (visible.Type == VisiblePoint::Kind::None ||
        photon.Volume != (visible.Type == VisiblePoint::Kind::Volume) ||
        photon.Depth + visible.Depth > settings.MaxReflectionBounces)
        return false;
    if (!photon.Depth)
        return photon.Volume || !settings.Transport.NextEventEstimation;
    return photon.DeltaPrefix ? settings.ProgressivePhotons.Caustics : settings.ProgressivePhotons.Indirect;
}
void Deposit(const Context& context, const Arrival& photon, const PointKdTree& index,
             double maximumRadiusSquared, const std::vector<uint32_t>& pixelIds,
             const std::vector<VisiblePoint>& visible,
             const std::vector<ProgressivePhotonPixel>& surfaceEstimates,
             const std::vector<ProgressivePhotonPixel>& volumeEstimates,
             std::vector<PassDeposit>& deposits)
{
    index.Search(photon.Position, maximumRadiusSquared, [&](uint32_t entry, double squaredDistance) {
        const uint32_t pixel = pixelIds[entry];
        const auto& point = visible[pixel];
        const double radiusSquared = point.Type == VisiblePoint::Kind::Volume
            ? volumeEstimates[pixel].RadiusSquared : surfaceEstimates[pixel].RadiusSquared;
        if (squaredDistance > radiusSquared || !Accept(photon, point, context.Settings))
            return;
        if (photon.Volume)
        {
            if (photon.MaterialIndex != point.MaterialIndex) return;
            const Float3 separation = photon.Position - point.Position;
            const float distance = std::sqrt(LengthSquared(separation));
            if (distance > 2 * RayEpsilon)
            {
                context.Statistics.ShadowRays.fetch_add(1, std::memory_order_relaxed);
                uint64_t tests = 0;
                const Ray connection{point.Position, separation / distance, RayEpsilon,
                                     distance - RayEpsilon};
                const bool blocked = context.Accelerator.Occluded(connection, tests);
                context.Statistics.PrimitiveTests.fetch_add(tests, std::memory_order_relaxed);
                if (blocked) return;
            }
            const auto& material = context.World.Materials[point.MaterialIndex];
            const double phase = SubsurfacePhase(Dot(photon.Incoming, point.Direction),
                                                 material.SubsurfaceAnisotropy);
            Add(deposits[pixel].Flux,
                Product(photon.Flux, point.Throughput, phase * material.SubsurfaceMeanFreePath));
            ++deposits[pixel].Count;
            return;
        }
        if (Dot(point.Normal, photon.Normal) < MinimumNormalAgreement)
            return;
        const auto& boundary = context.World.Materials[point.MaterialIndex];
        if (boundary.SubsurfaceMeanFreePath > 0)
        {
            const auto normalDirection = Dot(photon.Incoming, point.Normal);
            if (point.FrontFace ? normalDirection <= 0 : normalDirection >= 0)
                return;
        }
        const double separation = (double(photon.Position.X) - point.Position.X) * point.Normal.X +
                                  (double(photon.Position.Y) - point.Position.Y) * point.Normal.Y +
                                  (double(photon.Position.Z) - point.Position.Z) * point.Normal.Z;
        if (separation * separation > RelativePlaneThickness * RelativePlaneThickness * radiusSquared)
            return;
        const double projected = std::abs(Dot(point.Normal, photon.Incoming));
        if (!(projected > 0))
            return;
        const auto scattering =
            point.Scattering->Evaluate(photon.Incoming, context.Settings.Transport.ImportanceSampling);
        auto flux =
            Product(photon.Flux, scattering.Value, point.Scattering->Cosine(photon.Incoming) / projected);
        const auto camera = Product(point.Throughput, Float3(1, 1, 1));
        for (size_t channel = 0; channel < flux.size(); ++channel)
            flux[channel] *= camera[channel];
        Add(deposits[pixel].Flux, flux);
        ++deposits[pixel].Count;
    });
}
} // namespace
void AdvanceProgressivePhotonPixel(ProgressivePhotonPixel& pixel, uint64_t arrivals,
                                   const std::array<double, 3>& weightedFlux, double alpha)
{
    if (!arrivals)
        return;
    const double nextCount = pixel.EffectiveCount + alpha * double(arrivals);
    const double areaRatio = nextCount / (pixel.EffectiveCount + double(arrivals));
    pixel.RadiusSquared *= areaRatio;
    pixel.EffectiveCount = nextCount;
    for (size_t channel = 0; channel < pixel.Flux.size(); ++channel)
        pixel.Flux[channel] = (pixel.Flux[channel] + weightedFlux[channel]) * areaRatio;
}
void AdvanceProgressiveVolumePixel(ProgressivePhotonPixel& pixel, uint64_t arrivals,
                                   const std::array<double, 3>& weightedFlux, double alpha)
{
    if (!arrivals) return;
    const double nextCount = pixel.EffectiveCount + alpha * double(arrivals);
    const double volumeRatio = nextCount / (pixel.EffectiveCount + double(arrivals));
    pixel.RadiusSquared *= std::pow(volumeRatio, 2.0 / 3.0);
    pixel.EffectiveCount = nextCount;
    for (size_t channel = 0; channel < pixel.Flux.size(); ++channel)
        pixel.Flux[channel] = (pixel.Flux[channel] + weightedFlux[channel]) * volumeRatio;
}
bool RenderProgressivePhotonFrame(const Scene& scene, const SceneAccelerator& accelerator,
                                  const CameraFrame& camera, const RenderSettings& settings, LinearFilm& film,
                                  AtomicStatistics& statistics, const CancellationProbe& cancellation,
                                  const ProgressivePhotonDispatcher& dispatch, std::string& error,
                                  ProgressivePhotonDiagnostics* diagnostics)
{
    if (diagnostics)
        *diagnostics = {};
    if (!ValidateTileSettings(settings, error))
        return false;
    if (settings.MaxReflectionBounces > ProgressivePhotonMaximumReflections || settings.Adaptive.Enabled ||
        settings.Transport.Mis != MisHeuristic::None || !settings.ShadowsEnabled ||
        settings.ShadowStrength != 1)
    {
        error = "PPM requires bounded depth, fixed passes, no connection MIS and full physical visibility.";
        return false;
    }
    const auto sources = PrepareEmitters(scene);
    for (const auto& source : sources.Sources)
        if (!(source.Flags & EmitterCastsShadows))
        {
            error = "PPM requires shadow casting on every emitter.";
            return false;
        }
    const Context context{scene, accelerator, sources, settings, statistics, cancellation};
    auto options = settings.ProgressivePhotons;
    options.Seed ^= settings.SamplingSeed;
    const double radius = double(sources.Radius) * options.InitialRadiusScale;
    if (!(radius >= 0) || !std::isfinite(radius))
    {
        error = "Invalid PPM emission domain.";
        return false;
    }
    const uint32_t pixels = settings.Width * settings.Height;
    std::vector<ProgressivePhotonPixel> estimates(pixels);
    for (auto& estimate : estimates)
        estimate.RadiusSquared = radius * radius;
    std::vector<ProgressivePhotonPixel> volumeEstimates(pixels);
    for (auto& estimate : volumeEstimates)
        estimate.RadiusSquared = radius * radius;
    std::vector<VisiblePoint> visible(pixels);
    std::vector<PassDeposit> deposits(pixels);
    std::vector<SpectrumSum> direct(pixels);
    std::vector<LightBatchEntry> batch(PathBatch);
    std::atomic<bool> invalid{false};
    uint64_t emitted = 0, accepted = 0;
    film = LinearFilm(settings.Width, settings.Height);
    for (uint32_t pass = 0; pass < settings.SamplesPerPixel; ++pass)
    {
        if (context.Stopped())
            return false;
        dispatch(pixels, [&](uint32_t pixel) {
            deposits[pixel] = {};
            if (!CameraPass(context, camera, pixel, pass, visible[pixel], direct[pixel]))
                invalid.store(true, std::memory_order_relaxed);
        });
        if (context.Stopped())
            return false;
        if (invalid.load(std::memory_order_relaxed))
        {
            error = "Invalid PPM camera transport.";
            return false;
        }
        std::vector<Float3> positions;
        std::vector<uint32_t> pixelIds;
        double maximumRadiusSquared = 0;
        for (uint32_t pixel = 0; pixel < pixels; ++pixel)
            if (visible[pixel].Type != VisiblePoint::Kind::None)
            {
                const double radiusSquared = visible[pixel].Type == VisiblePoint::Kind::Volume
                    ? volumeEstimates[pixel].RadiusSquared : estimates[pixel].RadiusSquared;
                if (!(radiusSquared > 0)) continue;
                positions.push_back(visible[pixel].Position);
                pixelIds.push_back(pixel);
                maximumRadiusSquared = std::max(maximumRadiusSquared, radiusSquared);
            }
        const PointKdTree index(std::move(positions), cancellation);
        if (index.WasCancelled() || context.Stopped())
            return false;
        for (uint32_t first = 0; first < options.PathsPerPass; first += PathBatch)
        {
            if (context.Stopped())
                return false;
            const uint32_t count = std::min(PathBatch, options.PathsPerPass - first);
            dispatch(count, [&](uint32_t lane) { LightPass(context, pass, first + lane, batch[lane]); });
            if (context.Stopped())
                return false;
            // Stable reduction owns each pixel's flux; workers never race on it.
            for (uint32_t lane = 0; lane < count; ++lane)
            {
                if (context.Stopped())
                    return false;
                if (batch[lane].Invalid)
                {
                    error = "Invalid PPM light transport.";
                    return false;
                }
                for (uint32_t arrival = 0; arrival < batch[lane].Count; ++arrival)
                    Deposit(context, batch[lane].Arrivals[arrival], index, maximumRadiusSquared, pixelIds,
                            visible, estimates, volumeEstimates, deposits);
            }
            emitted += count;
        }
        for (uint32_t pixel = 0; pixel < pixels; ++pixel)
        {
            auto& estimate = estimates[pixel];
            auto& volume = volumeEstimates[pixel];
            const auto& deposit = deposits[pixel];
            if (!Valid(deposit.Flux))
            {
                error = "Invalid PPM incident flux.";
                return false;
            }
            if (visible[pixel].Type == VisiblePoint::Kind::Volume)
                AdvanceProgressiveVolumePixel(volume, deposit.Count, deposit.Flux, options.Alpha);
            else if (visible[pixel].Type == VisiblePoint::Kind::Surface)
                AdvanceProgressivePhotonPixel(estimate, deposit.Count, deposit.Flux, options.Alpha);
            accepted += deposit.Count;
            SpectrumSum value{};
            for (size_t channel = 0; channel < value.size(); ++channel)
                value[channel] =
                    direct[pixel][channel] / double(pass + 1) +
                    (estimate.RadiusSquared > 0
                         ? estimate.Flux[channel] / (DensityPi * estimate.RadiusSquared * double(emitted))
                         : 0) +
                    (volume.RadiusSquared > 0
                         ? volume.Flux[channel] /
                            (VolumeMeasure * DensityPi * volume.RadiusSquared * std::sqrt(volume.RadiusSquared) *
                             double(emitted))
                         : 0);
            const Float3 radiance{float(value[0]), float(value[1]), float(value[2])};
            if (!Valid(radiance) || !Valid(estimate.Flux) || !Valid(volume.Flux) ||
                !std::isfinite(estimate.RadiusSquared) || !std::isfinite(volume.RadiusSquared))
            {
                error = "Invalid progressive photon density.";
                return false;
            }
            film.Pixels[pixel] = {radiance, 0, pass + 1};
        }
        if (diagnostics)
        {
            diagnostics->EmittedPaths = emitted;
            diagnostics->AcceptedArrivals = accepted;
            diagnostics->CompletedPasses = pass + 1;
        }
    }
    if (context.Stopped())
        return false;
    if (diagnostics)
    {
        diagnostics->Pixels = std::move(estimates);
        diagnostics->VolumePixels = std::move(volumeEstimates);
    }
    statistics.CompletedPixels.fetch_add(pixels, std::memory_order_relaxed);
    return true;
}
} // namespace Arvo::RayTracing::Cpu
