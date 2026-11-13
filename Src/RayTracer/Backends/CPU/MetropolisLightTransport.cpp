#include "RayTracer/Backends/CPU/MetropolisLightTransport.h"
#include "RayTracer/Backends/CPU/CameraEndpoint.h"
#include "RayTracer/Backends/CPU/EmitterEndpoint.h"
#include "RayTracer/Backends/CPU/SurfaceScattering.h"
#include "RayTracer/Backends/CPU/SubsurfaceScattering.h"
#include <algorithm>
#include <cmath>

namespace Arvo::RayTracing::Cpu
{
/* Primary-sample-space Metropolis light transport (Kelemen et al., 2002,
   "A Simple and Robust Mutation Strategy for the Metropolis Light Transport Algorithm").
   https://pbr-book.org/3ed-2018/Light_Transport_III_Bidirectional_Methods/Metropolis_Light_Transport

   This translation unit owns both the deterministic camera-path estimator and
   the Markov chain. It never calls the path-tracing or BDPT integrators. CPU
   traversal, BSDF and endpoint leaves supply their own mathematical operations.
   Fixed primary dimensions keep every branch a function on the same unit cube.
   Symmetric wrapped Gaussian/independent proposals give acceptance min(1,I'/I).
   Bootstrap selection is proportional to I, with normalization b = mean(I).
   Both the rejected current state and proposed state contribute: (1-a)L/I and
   aL'/I'. Multiplication by b preserves the original integral, including black
   bootstrap realizations. No burn-in samples are discarded or divided out.

   Subsurface free flight and HG phase use fixed dimensions at every potential
   event. A medium collision therefore remains a deterministic function of the
   Markov primary state, including paths whose camera starts inside a solid.

   A pass contains N full-image mutations. Raster-bin area 1/N cancels this N,
   so pass splats scale by b, not b/N. Chains persist across passes; their pass
   moments are correlated and must not drive independent-sample adaptive stopping.
*/
namespace
{
constexpr uint64_t GoldenIncrement = 0x9e3779b97f4a7c15ull;
constexpr uint64_t MixMultiplierA = 0xbf58476d1ce4e5b9ull, MixMultiplierB = 0x94d049bb133111ebull;
constexpr uint64_t BootstrapDomain = 0x42535452504d4c54ull, ChainDomain = 0x434841494e4d4c54ull;
constexpr uint32_t DoubleMantissaBits = 53, BootstrapBatch = 64, MutationBatch = 64;
constexpr uint32_t MixShiftA = 30, MixShiftB = 27, MixShiftC = 31, GeneratorWordBits = 64;
constexpr double UnitDouble = 1.0 / double(uint64_t{1} << DoubleMantissaBits);
constexpr double TwoPi = 6.283185307179586476925286766559;
constexpr uint32_t LightChoice = 0, LightPosition = 1, LightDirection = 3;
constexpr uint32_t BsdfDirection = 5, BsdfLobe = 7, RouletteDimension = 8;
constexpr uint32_t InteriorFreeFlight = 9, InteriorPhaseDirection = 10;
constexpr double LuminanceRed = 0.2126, LuminanceGreen = 0.7152, LuminanceBlue = 0.0722;
bool Positive(Float3 v)
{
    return v.X > 0 || v.Y > 0 || v.Z > 0;
}
bool Finite(Float3 v)
{
    return std::isfinite(v.X) && std::isfinite(v.Y) && std::isfinite(v.Z);
}
float Variate(const MetropolisPrimaryState& state, uint32_t dimension)
{
    return std::min(float(state.Values[dimension]), std::nextafter(1.0f, 0.0f));
}
Float2 Pair(const MetropolisPrimaryState& state, uint32_t dimension)
{
    return {Variate(state, dimension), Variate(state, dimension + 1)};
}
Float3 Offset(Float3 position, Float3 geometry, Float3 direction)
{
    return position + geometry * (Dot(geometry, direction) >= 0 ? RayEpsilon : -RayEpsilon);
}
Float3 Emitted(const PackedEmitter& source, Float3 direction)
{
    if (source.Kind == EmitterKind::Environment)
    {
        const float weight = (1 - direction.Y) / 2;
        return source.BackgroundBottom * (1 - weight) + source.BackgroundTop * weight;
    }
    if (source.Kind == EmitterKind::Rectangle)
        return Dot(source.Direction, direction) > 0 ? source.Strength / (source.Width * source.Height)
                                                    : Float3();
    return source.Strength;
}
uint32_t FindSource(const PreparedEmissionScene& scene, uint32_t primitive, bool environment)
{
    for (uint32_t i = 0; i < scene.Sources.size(); ++i)
        if (environment ? scene.Sources[i].Kind == EmitterKind::Environment
                        : (primitive != InvalidRayPrimitiveIndex && scene.Sources[i].Primitive == primitive))
            return i;
    return InvalidEmitterIndex;
}
bool RectangleHit(const PackedEmitter& source, const Ray& ray, float& distance)
{
    if (source.Kind != EmitterKind::Rectangle)
        return false;
    const float denominator = Dot(source.Direction, ray.Direction);
    if (denominator == 0)
        return false;
    const float candidate = Dot(source.Direction, source.Origin - ray.Origin) / denominator;
    if (!(candidate > ray.MinDistance && candidate < ray.MaxDistance && candidate <= distance))
        return false;
    const Float3 helper =
        std::abs(source.Direction.Y) < BasisParallelThreshold ? Float3(0, 1, 0) : Float3(1, 0, 0);
    const Float3 tangent = Normalize(Cross(helper, source.Direction)),
                 bitangent = Cross(source.Direction, tangent);
    const Float3 delta = ray.Origin + ray.Direction * candidate - source.Origin;
    if (std::abs(Dot(delta, tangent)) > source.Width / 2 ||
        std::abs(Dot(delta, bitangent)) > source.Height / 2)
        return false;
    distance = candidate;
    return true;
}
struct Transport
{
    const Scene& SceneData;
    const SceneAccelerator& Accelerator;
    const PreparedEmissionScene& Emitters;
    const CameraFrame& Camera;
    const RenderSettings& Settings;
    AtomicStatistics& Statistics;
    const CancellationProbe& Cancelled;
};
bool Visible(const Transport& transport, Float3 origin, Float3 direction, float distance)
{
    transport.Statistics.ShadowRays.fetch_add(1, std::memory_order_relaxed);
    Ray ray{origin, direction, RayEpsilon, distance};
    uint64_t tests = 0;
    bool blocked = transport.Accelerator.Occluded(ray, tests);
    if (!blocked)
        for (const auto& source : transport.Emitters.Sources)
            if (source.Kind == EmitterKind::Rectangle)
            {
                ++tests;
                if (RectangleHit(source, ray, distance))
                {
                    blocked = true;
                    break;
                }
            }
    transport.Statistics.PrimitiveTests.fetch_add(tests, std::memory_order_relaxed);
    return !blocked;
}
float LightPdf(const PreparedEmissionScene& scene, uint32_t source, Float3 normal, Float3 launch,
               Float3 previous, Float3 position, bool infinite)
{
    const auto pdf = EmitterEndpointPdf(scene, source, normal, launch);
    if (pdf.Flags & (EmissionDeltaPosition | EmissionDeltaDirection))
        return 0;
    if (infinite)
        return pdf.SelectionProbability * pdf.Direction;
    const float cosine = std::abs(Dot(normal, launch));
    return cosine > 0 ? pdf.SelectionProbability * pdf.Position * LengthSquared(position - previous) / cosine
                      : 0;
}
float EmitterHitWeight(const TransportOptions& options, bool previousDelta, float previousPdf, float lightPdf)
{
    if (previousDelta || !options.NextEventEstimation)
        return 1;
    // Without MIS, NEE owns continuous-source direct lighting. Delta chains
    // still see emitters, since continuous BSDF evaluation cannot sample them.
    return options.Mis == MisHeuristic::None ? 0 : MultipleImportance(previousPdf, lightPdf, options.Mis);
}
Float3 Direct(const Transport& transport, const MetropolisPrimaryState& primary, uint32_t base,
              Float3 position, const SurfaceBsdf& bsdf, float diffuseProbability,
              float interiorMeanFreePath)
{
    const auto endpoint =
        SampleEmitterEndpoint(transport.Emitters, Variate(primary, base + LightChoice),
                              Pair(primary, base + LightPosition), Pair(primary, base + LightDirection));
    const auto& sample = endpoint.Emission;
    if (!(sample.Flags & EmissionValid))
        return {};
    const bool infinite = (sample.Flags & EmissionInfinite) != 0;
    const bool deltaPosition = (sample.Flags & EmissionDeltaPosition) != 0;
    const bool delta = (sample.Flags & (EmissionDeltaPosition | EmissionDeltaDirection)) != 0;
    const auto& options = transport.Settings.Transport;
    if (!delta && !options.NextEventEstimation)
        return {};
    Float3 direction = -sample.Direction;
    float geometry = 1, distance = MaximumRayDistance;
    if (!infinite)
    {
        const Float3 segment = sample.Origin - position;
        const float squared = LengthSquared(segment);
        if (!(squared > 0) || !std::isfinite(squared))
            return {};
        distance = std::sqrt(squared);
        direction = segment / distance;
        geometry = (deltaPosition ? 1 : std::abs(Dot(sample.Normal, direction))) / squared;
    }
    auto evaluated = bsdf.Evaluate(direction, options.ImportanceSampling);
    evaluated.Value = evaluated.Value * diffuseProbability;
    evaluated.Pdf *= diffuseProbability;
    const float density =
        endpoint.SelectionProbability * (infinite ? sample.DirectionPdf : sample.PositionPdf);
    const float sampledPdf = delta ? density : infinite ? density : geometry > 0 ? density / geometry : 0;
    const float mis = delta ? 1 : MultipleImportance(sampledPdf, evaluated.Pdf, options.Mis);
    const Float3 radiance = Emitted(transport.Emitters.Sources[endpoint.EmitterIndex], -direction);
    Float3 value =
        Multiply(evaluated.Value, radiance) * (bsdf.Cosine(direction) * geometry * mis / density);
    const float connectionMeanFreePath = bsdf.ConnectionMeanFreePath(direction, interiorMeanFreePath);
    if (connectionMeanFreePath > 0)
        value = value * std::exp(-distance / connectionMeanFreePath);
    if (!Finite(value))
        return value; // Propagate numerical failure to the chain before acceptance.
    if (!Positive(value))
        return {};
    const Float3 origin = bsdf.Offset(position, direction);
    if (!infinite)
    {
        const Float3 target =
            deltaPosition ? sample.Origin : Offset(sample.Origin, sample.Normal, -direction);
        const Float3 segment = target - origin;
        distance = std::sqrt(LengthSquared(segment));
        if (!(distance > 2 * RayEpsilon))
            return {};
        direction = segment / distance;
        distance -= RayEpsilon;
    }
    return Visible(transport, origin, direction, distance) ? value : Float3();
}
Float3 InteriorDirect(const Transport& transport, const MetropolisPrimaryState& primary, uint32_t base,
                      Float3 position, Float3 incoming, const Material& interior)
{
    const auto endpoint = SampleEmitterEndpoint(transport.Emitters, Variate(primary, base + LightChoice),
        Pair(primary, base + LightPosition), Pair(primary, base + LightDirection));
    const auto& sample = endpoint.Emission;
    if (!(sample.Flags & EmissionValid)) return {};
    const bool infinite = (sample.Flags & EmissionInfinite) != 0;
    const bool deltaPosition = (sample.Flags & EmissionDeltaPosition) != 0;
    const bool delta = (sample.Flags & (EmissionDeltaPosition | EmissionDeltaDirection)) != 0;
    const auto& options = transport.Settings.Transport;
    if (!delta && !options.NextEventEstimation) return {};
    Float3 direction = -sample.Direction;
    float distance = MaximumRayDistance, geometry = 1;
    if (!infinite)
    {
        const Float3 segment = sample.Origin - position;
        const float squared = LengthSquared(segment);
        if (!(squared > 0) || !std::isfinite(squared)) return {};
        distance = std::sqrt(squared);
        direction = segment / distance;
        geometry = (deltaPosition ? 1 : std::abs(Dot(sample.Normal, direction))) / squared;
    }
    const float density = endpoint.SelectionProbability * (infinite ? sample.DirectionPdf : sample.PositionPdf);
    if (!(density > 0) || !(geometry > 0)) return {};
    const float sampledPdf = delta ? density : infinite ? density : density / geometry;
    const float phase = SubsurfacePhase(Dot(incoming, direction), interior.SubsurfaceAnisotropy);
    const float phasePdf = options.ImportanceSampling ? phase : 1 / (4 * Pi);
    const float mis = delta ? 1 : MultipleImportance(sampledPdf, phasePdf, options.Mis);
    const Float3 emitted = Emitted(transport.Emitters.Sources[endpoint.EmitterIndex], -direction);
    const Float3 value = emitted * (phase * geometry * mis / density)
        * std::exp(-distance / interior.SubsurfaceMeanFreePath);
    if (!Finite(value)) return value;
    if (!Positive(value)) return {};
    if (!infinite)
    {
        const Float3 target = deltaPosition ? sample.Origin : Offset(sample.Origin, sample.Normal, -direction);
        const Float3 segment = target - position;
        distance = std::sqrt(LengthSquared(segment));
        if (!(distance > RayEpsilon)) return {};
        direction = segment / distance;
        distance -= RayEpsilon;
    }
    return Visible(transport, position, direction, distance) ? value : Float3();
}
MetropolisPath Evaluate(const Transport& transport, const MetropolisPrimaryState& primary)
{
    MetropolisPath path;
    path.Raster = Pair(primary, 0);
    const auto camera = SampleCameraEndpoint(transport.Camera, path.Raster, Pair(primary, 2));
    if (!(camera.Flags & CameraEndpointValid))
        return path;
    const auto& options = transport.Settings.Transport;
    Ray ray{camera.Origin, camera.Direction, RayEpsilon, MaximumRayDistance};
    Float3 beta(1, 1, 1), previous;
    float previousPdf = 0, etaScale = 1;
    bool previousDelta = true;
    uint32_t interiorMaterial = InvalidRayPrimitiveIndex;
    for (uint32_t depth = 0; depth <= transport.Settings.MaxReflectionBounces + 1; ++depth)
    {
        const uint32_t base = MetropolisCameraDimensions + depth * MetropolisVertexDimensions;
        if (transport.Cancelled && transport.Cancelled())
            return {};
        (depth == 0 ? transport.Statistics.PrimaryRays : transport.Statistics.ReflectionRays)
            .fetch_add(1, std::memory_order_relaxed);
        uint64_t tests = 0;
        SurfaceHit hit;
        hit.Distance = ray.MaxDistance;
        const bool found = transport.Accelerator.Intersect(ray, hit, tests);
        float nearest = found ? hit.Distance + RayEpsilon : ray.MaxDistance;
        uint32_t source = InvalidEmitterIndex;
        for (uint32_t i = 0; i < transport.Emitters.Sources.size(); ++i)
            if (transport.Emitters.Sources[i].Kind == EmitterKind::Rectangle)
            {
                ++tests;
                if (RectangleHit(transport.Emitters.Sources[i], ray, nearest))
                    source = i;
            }
        transport.Statistics.PrimitiveTests.fetch_add(tests, std::memory_order_relaxed);
        const bool rectangle = source != InvalidEmitterIndex;
        const bool infinite = !found && !rectangle;
        if (depth == 0 && found && !hit.FrontFace &&
            transport.SceneData.Materials[hit.MaterialIndex].SubsurfaceMeanFreePath > 0)
            interiorMaterial = hit.MaterialIndex;
        if (interiorMaterial != InvalidRayPrimitiveIndex)
        {
            const Material& interior = transport.SceneData.Materials[interiorMaterial];
            const float endpoint = rectangle ? nearest : found ? hit.Distance : ray.MaxDistance;
            const auto flight = SampleSubsurfaceFlight(interior.SubsurfaceMeanFreePath,
                interior.Albedo * interior.SubsurfaceAlbedoScale, ray.Direction, endpoint,
                Variate(primary, base + InteriorFreeFlight));
            beta = Multiply(beta, flight.Weight);
            if (!Finite(beta)) { path.Radiance.X = std::numeric_limits<float>::quiet_NaN(); break; }
            if (!Positive(beta)) break;
            if (flight.Scattered)
            {
                if (depth > transport.Settings.MaxReflectionBounces) break;
                const Float3 position = ray.Origin + ray.Direction * flight.Distance;
                path.Radiance = path.Radiance + Multiply(beta,
                    InteriorDirect(transport, primary, base, position, ray.Direction, interior));
                const auto phase = SampleSubsurfaceDirection(ray.Direction, interior.SubsurfaceAnisotropy,
                    Pair(primary, base + InteriorPhaseDirection), options.ImportanceSampling);
                if (!(phase.Pdf > 0)) break;
                beta = beta * phase.Weight;
                if (!Finite(beta)) { path.Radiance.X = std::numeric_limits<float>::quiet_NaN(); break; }
                if (!Positive(beta)) break;
                if (options.RussianRoulette && depth + 1 >= options.RouletteStartDepth)
                {
                    const float survival = RouletteProbability(beta * etaScale);
                    if (Variate(primary, base + RouletteDimension) >= survival) break;
                    beta = beta / survival;
                }
                previous = position;
                previousPdf = phase.Pdf;
                previousDelta = false;
                ray = {position, phase.Direction, RayEpsilon, MaximumRayDistance};
                continue;
            }
        }
        if (!rectangle)
            source = FindSource(transport.Emitters, hit.PrimitiveIndex, infinite);
        const Float3 position = ray.Origin + ray.Direction * (rectangle ? nearest : hit.Distance);
        if (source != InvalidEmitterIndex)
        {
            const auto& emitter = transport.Emitters.Sources[source];
            const Float3 normal = rectangle ? emitter.Direction : hit.GeometricNormal;
            const float pdf = depth == 0 ? 0
                                         : LightPdf(transport.Emitters, source, normal, -ray.Direction,
                                                    previous, position, infinite);
            const float weight = EmitterHitWeight(options, previousDelta, previousPdf, pdf);
            path.Radiance = path.Radiance + Multiply(beta, Emitted(emitter, -ray.Direction)) * weight;
        }
        if (infinite || rectangle || depth > transport.Settings.MaxReflectionBounces)
            break;
        Material material = transport.SceneData.Materials[hit.MaterialIndex];
        const float mirror =
            material.Model == MaterialModel::Legacy ? std::clamp(material.Reflectivity, 0.0f, 1.0f) : 0;
        if (material.Model == MaterialModel::Legacy)
            material.Model = MaterialModel::Diffuse;
        const Float3 color =
            hit.HasTexCoord
                ? Multiply(hit.Color, SampleDiffuseTexture(transport.SceneData, material.DiffuseTextureIndex,
                                                           hit.TexCoord))
                : hit.Color;
        const SurfaceBsdf bsdf(material, hit, -ray.Direction, color);
        path.Radiance =
            path.Radiance + Multiply(beta, Direct(transport, primary, base, position, bsdf, 1 - mirror,
                interiorMaterial == InvalidRayPrimitiveIndex ? 0 :
                    transport.SceneData.Materials[interiorMaterial].SubsurfaceMeanFreePath));
        SurfaceSample scattering;
        if (Variate(primary, base + BsdfLobe) < mirror)
        {
            scattering.Direction = Normalize(Reflect(ray.Direction, hit.Normal));
            if (Dot(scattering.Direction, hit.GeometricNormal) * Dot(-ray.Direction, hit.GeometricNormal) <=
                0)
                break;
            scattering.Weight = {1, 1, 1};
            scattering.Pdf = mirror;
            scattering.Event = ScatteringDelta | ScatteringReflection;
        }
        else
        {
            scattering = bsdf.Sample(Pair(primary, base + BsdfDirection), Variate(primary, base + BsdfLobe),
                                     options.ImportanceSampling);
            scattering.Pdf *= 1 - mirror;
        }
        if (!scattering.Event || !(scattering.Pdf > 0))
            break;
        beta = Multiply(beta, scattering.Weight);
        if (!Finite(beta))
        {
            path.Radiance.X = std::numeric_limits<float>::quiet_NaN();
            break;
        }
        if (!Positive(beta))
            break;
        etaScale *= scattering.RelativeIndex * scattering.RelativeIndex;
        if (options.RussianRoulette && depth + 1 >= options.RouletteStartDepth)
        {
            const float survival = RouletteProbability(beta * etaScale);
            if (Variate(primary, base + RouletteDimension) >= survival)
                break;
            beta = beta / survival;
        }
        previous = position;
        previousPdf = scattering.Pdf;
        previousDelta = (scattering.Event & ScatteringDelta) != 0;
        ray = {bsdf.Offset(position, scattering.Direction), scattering.Direction, RayEpsilon,
               MaximumRayDistance};
        if (material.SubsurfaceMeanFreePath > 0 && (scattering.Event & ScatteringTransmission) != 0)
            interiorMaterial = hit.FrontFace ? hit.MaterialIndex : InvalidRayPrimitiveIndex;
    }
    path.Importance =
        LuminanceRed * path.Radiance.X + LuminanceGreen * path.Radiance.Y + LuminanceBlue * path.Radiance.Z;
    return path;
}
bool Valid(const MetropolisPath& path)
{
    return Finite(path.Radiance) && path.Radiance.X >= 0 && path.Radiance.Y >= 0 && path.Radiance.Z >= 0 &&
           std::isfinite(path.Importance) && path.Importance >= 0;
}
struct BootstrapEntry
{
    uint64_t Seed;
    MetropolisPath Path;
    double Cumulative = 0;
};
struct Splat
{
    MetropolisPath Path;
    double Weight = 0;
};
struct Chain
{
    MetropolisPrimaryState Primary;
    MetropolisPath Path;
    MetropolisRandom Random{0};
    std::array<Splat, 2 * MutationBatch> Splats;
    uint32_t Count = 0;
    MetropolisDiagnostics Statistics;
};
struct PixelSum
{
    double Red = 0, Green = 0, Blue = 0;
};
} // namespace

double MetropolisRandom::Uniform()
{
    // SplitMix64 (Steele, Lea and Flood, 2014), with a 53-bit unit variate.
    uint64_t value = (State += GoldenIncrement);
    value = (value ^ (value >> MixShiftA)) * MixMultiplierA;
    value = (value ^ (value >> MixShiftB)) * MixMultiplierB;
    value ^= value >> MixShiftC;
    return double(value >> (GeneratorWordBits - DoubleMantissaBits)) * UnitDouble;
}
MetropolisPrimaryState ProposeMetropolis(const MetropolisPrimaryState& current, MetropolisRandom& random,
                                         bool largeStep, double mutationScale)
{
    MetropolisPrimaryState proposed;
    for (uint32_t dimension = 0; dimension < MetropolisDimensions; ++dimension)
    {
        if (largeStep)
            proposed.Values[dimension] = random.Uniform();
        else
        {
            const double radius = std::sqrt(-2 * std::log(1 - random.Uniform()));
            const double perturbation = mutationScale * radius * std::cos(TwoPi * random.Uniform());
            const double value = current.Values[dimension] + perturbation;
            proposed.Values[dimension] = value - std::floor(value);
        }
    }
    return proposed;
}
bool RenderMetropolisFrame(const Scene& scene, const SceneAccelerator& accelerator, const CameraFrame& camera,
                           const RenderSettings& settings, LinearFilm& film, AtomicStatistics& statistics,
                           const CancellationProbe& cancellation, const MetropolisDispatcher& dispatch,
                           std::string& error, MetropolisDiagnostics* diagnostics)
{
    if (diagnostics)
        *diagnostics = {};
    if (!ValidateTileSettings(settings, error))
        return false;
    if (settings.MaxReflectionBounces > MetropolisMaximumReflections || settings.Adaptive.Enabled ||
        !settings.ShadowsEnabled || settings.ShadowStrength != 1)
    {
        error = "MLT requires bounded depth, fixed mutation budgets and full physical visibility.";
        return false;
    }
    const auto emitters = PrepareEmitters(scene);
    for (const auto& emitter : emitters.Sources)
        if (!(emitter.Flags & EmitterCastsShadows))
        {
            error = "MLT requires shadow casting on every emitter.";
            return false;
        }
    const auto stopped = [&] { return cancellation && cancellation(); };
    const Transport transport{scene, accelerator, emitters, camera, settings, statistics, cancellation};
    auto options = settings.Metropolis;
    options.Seed ^= settings.SamplingSeed;
    std::vector<BootstrapEntry> bootstrap(options.BootstrapSamples);
    for (uint32_t begin = 0; begin < options.BootstrapSamples; begin += BootstrapBatch)
    {
        if (stopped())
            return false;
        dispatch(std::min(BootstrapBatch, options.BootstrapSamples - begin), [&](uint32_t lane) {
            auto& entry = bootstrap[begin + lane];
            entry.Seed = BootstrapDomain ^ (uint64_t(options.Seed) * GoldenIncrement + begin + lane);
            MetropolisRandom random{entry.Seed};
            const auto primary = ProposeMetropolis({}, random, true, options.MutationScale);
            entry.Path = Evaluate(transport, primary);
        });
    }
    double total = 0;
    for (auto& entry : bootstrap)
    {
        if (!Valid(entry.Path))
        {
            error = "MLT bootstrap produced nonfinite or negative transport.";
            return false;
        }
        total += entry.Path.Importance;
        entry.Cumulative = total;
    }
    const double normalization = total / options.BootstrapSamples;
    if (diagnostics)
        diagnostics->Normalization = normalization;
    const uint32_t fullPixels = FullImageWidth(settings) * FullImageHeight(settings);
    const uint32_t chainCount = std::min(options.ChainCount, fullPixels);
    std::vector<Chain> chains(chainCount);
    if (total > 0)
        for (uint32_t index = 0; index < chainCount; ++index)
        {
            auto& chain = chains[index];
            chain.Random.State = ChainDomain ^ (uint64_t(options.Seed) * GoldenIncrement + index);
            const double choice = std::min(chain.Random.Uniform() * total, std::nextafter(total, 0.0));
            const auto selected = std::upper_bound(
                bootstrap.begin(), bootstrap.end(), choice,
                [](double value, const BootstrapEntry& item) { return value < item.Cumulative; });
            MetropolisRandom replay{selected->Seed};
            chain.Primary = ProposeMetropolis({}, replay, true, options.MutationScale);
            chain.Path = selected->Path;
        }
    std::vector<PixelSum> pass(size_t(settings.Width) * settings.Height);
    const uint32_t maximumPerChain = (fullPixels + chainCount - 1) / chainCount;
    std::atomic<bool> invalid{false};
    for (uint32_t sample = 0; sample < settings.SamplesPerPixel; ++sample)
    {
        if (stopped())
            return false;
        std::fill(pass.begin(), pass.end(), PixelSum{});
        for (uint32_t begin = 0; total > 0 && begin < maximumPerChain; begin += MutationBatch)
        {
            if (stopped())
                return false;
            dispatch(chainCount, [&](uint32_t index) {
                auto& chain = chains[index];
                chain.Count = 0;
                const uint32_t budget = fullPixels / chainCount + (index < fullPixels % chainCount ? 1 : 0);
                const uint32_t count = begin < budget ? std::min(MutationBatch, budget - begin) : 0;
                for (uint32_t mutation = 0; mutation < count; ++mutation)
                {
                    if (stopped() || invalid.load(std::memory_order_relaxed))
                        return;
                    const bool large = chain.Random.Uniform() < options.LargeStepProbability;
                    auto proposed =
                        ProposeMetropolis(chain.Primary, chain.Random, large, options.MutationScale);
                    const auto path = Evaluate(transport, proposed);
                    if (!Valid(path))
                    {
                        invalid.store(true, std::memory_order_relaxed);
                        return;
                    }
                    const double acceptance = std::min(1.0, path.Importance / chain.Path.Importance);
                    chain.Splats[chain.Count++] = {chain.Path,
                                                   normalization * (1 - acceptance) / chain.Path.Importance};
                    if (path.Importance > 0)
                        chain.Splats[chain.Count++] = {path, normalization * acceptance / path.Importance};
                    ++chain.Statistics.Proposals;
                    chain.Statistics.LargeSteps += large;
                    if (chain.Random.Uniform() < acceptance)
                    {
                        chain.Primary = proposed;
                        chain.Path = path;
                        ++chain.Statistics.Accepted;
                    }
                    else if (path.Importance > 0)
                        ++chain.Statistics.RejectedPositive;
                }
            });
            if (invalid.load())
            {
                error = "MLT mutation produced nonfinite or negative transport.";
                return false;
            }
            if (stopped())
                return false;
            // Fixed reduction order makes thread-count changes reproducible.
            for (const auto& chain : chains)
                for (uint32_t i = 0; i < chain.Count; ++i)
                {
                    const auto& splat = chain.Splats[i];
                    const uint32_t x = uint32_t(splat.Path.Raster.X * FullImageWidth(settings));
                    const uint32_t y = uint32_t(splat.Path.Raster.Y * FullImageHeight(settings));
                    if (x < settings.TileX || y < settings.TileY || x - settings.TileX >= settings.Width ||
                        y - settings.TileY >= settings.Height)
                        continue;
                    auto& pixel = pass[size_t(y - settings.TileY) * settings.Width + x - settings.TileX];
                    pixel.Red += splat.Weight * splat.Path.Radiance.X;
                    pixel.Green += splat.Weight * splat.Path.Radiance.Y;
                    pixel.Blue += splat.Weight * splat.Path.Radiance.Z;
                }
        }
        for (size_t pixel = 0; pixel < pass.size(); ++pixel)
        {
            const auto& value = pass[pixel];
            const Float3 radiance(float(value.Red), float(value.Green), float(value.Blue));
            if (!Finite(radiance))
            {
                error = "MLT pass exceeded the linear film range.";
                return false;
            }
            AccumulateFilmSample(film.Pixels[pixel], radiance);
            if (!Finite(film.Pixels[pixel].MeanRadiance) || !std::isfinite(film.Pixels[pixel].LuminanceM2))
            {
                error = "MLT moments exceeded the linear film range.";
                return false;
            }
        }
    }
    if (diagnostics)
        for (const auto& chain : chains)
        {
            diagnostics->Proposals += chain.Statistics.Proposals;
            diagnostics->Accepted += chain.Statistics.Accepted;
            diagnostics->LargeSteps += chain.Statistics.LargeSteps;
            diagnostics->RejectedPositive += chain.Statistics.RejectedPositive;
        }
    statistics.CompletedPixels.fetch_add(pass.size(), std::memory_order_relaxed);
    return !stopped();
}
} // namespace Arvo::RayTracing::Cpu
