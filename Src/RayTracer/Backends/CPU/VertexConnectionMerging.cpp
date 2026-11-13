#include "RayTracer/Backends/CPU/VertexConnectionMerging.h"
#include "RayTracer/Backends/CPU/VertexConnectionMergingMis.h"
#include "RayTracer/Backends/CPU/PointKdTree.h"
#include "RayTracer/Backends/CPU/SubsurfaceScattering.h"
#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <new>

namespace Arvo::RayTracing::Cpu
{
// Georgiev et al. (2012), "Light Transport Simulation with Vertex Connection
// and Merging"; Georgiev, "Implementing Vertex Connection and Merging", rev. 2:
// https://www.iliyan.com/publications/ImplementingVCM/ImplementingVCM_TechRep2012_rev2.pdf
// Camera walks carry radiance, light walks carry importance. A vertex stores
// incident throughput BEFORE its local scattering. VCM's generalized density
// converts even a sampled delta component's PMF by the destination Jacobian;
// its event flag separately prevents a connection/merge at that delta event.
// Infinite emitters project a launch disk onto their first finite surface and
// retain angular measure on the reverse endpoint. No distant sphere is invented.
// Interior collisions add a volume path measure: free-flight extinction enters
// each density, while the sampled attenuation cancels from throughput. Volume
// merging uses a shrinking 3D sphere kernel rather than a surface disk.
namespace
{
constexpr uint32_t CameraDomain = 0x56434d43u, LightDomain = 0x56434d4cu; // "VCMC", "VCML"
constexpr uint32_t SourceDomain = 0x56434d53u, PositionDomain = 0x56434d50u, DirectionDomain = 0x56434d44u;
constexpr uint32_t InteriorFlightDomain = 0x5643464cu, InteriorPhaseDomain = 0x56435048u;
constexpr double VcmPi = 3.14159265358979323846;

bool Finite(Float3 value)
{
    return std::isfinite(value.X) && std::isfinite(value.Y) && std::isfinite(value.Z);
}
bool ValidPower(Float3 value)
{
    return Finite(value) && value.X >= 0 && value.Y >= 0 && value.Z >= 0;
}
bool Nonzero(Float3 value)
{
    return value.X > 0 || value.Y > 0 || value.Z > 0;
}
SurfaceHit Orient(const VcmVertex& vertex, Float3 view)
{
    auto hit = vertex.Hit;
    hit.Normal = Dot(hit.Normal, view) < 0 ? -hit.Normal : hit.Normal;
    hit.FrontFace = Dot(hit.GeometricNormal, view) > 0;
    return hit;
}
Float3 Depart(const VcmVertex& vertex, Float3 outgoing)
{
    if (vertex.Kind == VcmVertexKind::Volume)
        return vertex.Position;
    return vertex.Position + vertex.Hit.GeometricNormal *
                                 (Dot(vertex.Hit.GeometricNormal, outgoing) >= 0 ? RayEpsilon : -RayEpsilon);
}
uint32_t SegmentInterior(const Scene& scene, const VcmVertex& source, const VcmVertex& target)
{
    if (source.Kind == VcmVertexKind::Volume) return source.Hit.MaterialIndex;
    if (target.Kind == VcmVertexKind::Volume) return target.Hit.MaterialIndex;
    if (source.Kind == VcmVertexKind::Light && source.EnclosingMaterial != InvalidRayPrimitiveIndex)
        return source.EnclosingMaterial;
    if (target.Kind == VcmVertexKind::Light && target.EnclosingMaterial != InvalidRayPrimitiveIndex)
        return target.EnclosingMaterial;
    const Float3 displacement = target.Position - source.Position;
    if (source.Kind == VcmVertexKind::Surface &&
        scene.Materials[source.Hit.MaterialIndex].SubsurfaceMeanFreePath > 0 &&
        Dot(source.Hit.GeometricNormal, displacement) < 0)
        return source.Hit.MaterialIndex;
    if (target.Kind == VcmVertexKind::Surface &&
        scene.Materials[target.Hit.MaterialIndex].SubsurfaceMeanFreePath > 0 &&
        Dot(target.Hit.GeometricNormal, -displacement) < 0)
        return target.Hit.MaterialIndex;
    return InvalidRayPrimitiveIndex;
}
double SegmentTransmittance(const Scene& scene, const VcmVertex& source, const VcmVertex& target)
{
    const uint32_t material = SegmentInterior(scene, source, target);
    if (material == InvalidRayPrimitiveIndex) return 1;
    const Float3 delta = target.Position - source.Position;
    return std::exp(-std::sqrt(double(Dot(delta, delta))) /
                    scene.Materials[material].SubsurfaceMeanFreePath);
}
VcmDensity EdgeDensity(const Scene& scene, VcmDensity directional,
                       const VcmVertex& source, const VcmVertex& target)
{
    if (!directional.Positive) return {};
    const Float3 delta = target.Position - source.Position;
    const double distance = std::sqrt(double(Dot(delta, delta)));
    if (!(distance > 0) || !std::isfinite(distance)) return {};
    VcmDensity converted;
    if (target.Kind == VcmVertexKind::Volume)
        converted = {directional.Log2Value - 2 * std::log2(distance), 1};
    else
        converted = VcmSurfaceDensity(directional, delta, target.Hit.GeometricNormal);
    if (!converted.Positive) return {};
    const uint32_t material = SegmentInterior(scene, source, target);
    if (material == InvalidRayPrimitiveIndex)
        return target.Kind == VcmVertexKind::Volume ? VcmDensity{} : converted;
    const double meanFreePath = scene.Materials[material].SubsurfaceMeanFreePath;
    converted.Log2Value -= distance / meanFreePath * std::log2(std::exp(1.0));
    if (target.Kind == VcmVertexKind::Volume)
        converted.Log2Value -= std::log2(meanFreePath);
    return std::isfinite(converted.Log2Value) ? converted : VcmDensity{};
}
bool RectangleHit(const PackedEmitter& source, const Ray& ray, float& nearest)
{
    const float denominator = Dot(source.Direction, ray.Direction);
    if (!denominator)
        return false;
    const float distance = Dot(source.Origin - ray.Origin, source.Direction) / denominator;
    if (!(distance > ray.MinDistance && distance < ray.MaxDistance && distance <= nearest))
        return false;
    const Float3 basis =
        std::abs(source.Direction.Y) < BasisParallelThreshold ? Float3(0, 1, 0) : Float3(1, 0, 0);
    const Float3 right = Normalize(Cross(basis, source.Direction)), up = Cross(source.Direction, right);
    const Float3 relative = ray.Origin + ray.Direction * distance - source.Origin;
    if (std::abs(Dot(relative, right)) > source.Width / 2 || std::abs(Dot(relative, up)) > source.Height / 2)
        return false;
    nearest = distance;
    return true;
}
struct WalkContext
{
    const Scene& World;
    const SceneAccelerator& Accelerator;
    const PreparedEmissionScene& Emitters;
    AtomicStatistics& Statistics;
    const CancellationProbe& Cancel;

    bool Stopped() const { return Cancel && Cancel(); }
    VcmVertex Intersect(const Ray& ray) const
    {
        VcmVertex vertex;
        vertex.ToPrevious = -ray.Direction;
        uint64_t tests = 0;
        const bool surface = Accelerator.Intersect(ray, vertex.Hit, tests);
        if (surface && !vertex.Hit.FrontFace &&
            World.Materials[vertex.Hit.MaterialIndex].SubsurfaceMeanFreePath > 0)
            vertex.EnclosingMaterial = vertex.Hit.MaterialIndex;
        // Only an explicitly registered emitter proxy may defer to an analytic
        // light just behind it. Ordinary geometry keeps its exact closest hit.
        const bool proxy = surface && World.Materials[vertex.Hit.MaterialIndex].SampledEmitter;
        float nearest = surface ? vertex.Hit.Distance + (proxy ? RayEpsilon : 0) : ray.MaxDistance;
        for (uint32_t i = 0; i < Emitters.Sources.size(); ++i)
            if (Emitters.Sources[i].Kind == EmitterKind::Rectangle)
            {
                ++tests;
                if (RectangleHit(Emitters.Sources[i], ray, nearest))
                    vertex.Emitter = i;
            }
        Statistics.PrimitiveTests.fetch_add(tests, std::memory_order_relaxed);
        if (vertex.Emitter != InvalidEmitterIndex)
        {
            vertex.Kind = VcmVertexKind::Light;
            vertex.Position = ray.Origin + ray.Direction * nearest;
            vertex.Hit = {};
            vertex.Hit.Distance = nearest;
            vertex.Hit.GeometricNormal = vertex.Hit.Normal = Emitters.Sources[vertex.Emitter].Direction;
            return vertex;
        }
        vertex.Kind = surface ? VcmVertexKind::Surface : VcmVertexKind::Environment;
        for (uint32_t i = 0; i < Emitters.Sources.size(); ++i)
        {
            const auto& source = Emitters.Sources[i];
            const bool matches = surface ? source.Primitive != InvalidRayPrimitiveIndex &&
                                               source.Primitive == vertex.Hit.PrimitiveIndex
                                         : source.Kind == EmitterKind::Environment;
            if (matches)
            {
                vertex.Emitter = i;
                break;
            }
        }
        if (surface)
        {
            vertex.Position = ray.Origin + ray.Direction * vertex.Hit.Distance;
            vertex.Color = vertex.Hit.Color;
            if (vertex.Hit.HasTexCoord)
                vertex.Color = Multiply(
                    vertex.Color,
                    SampleDiffuseTexture(World, World.Materials[vertex.Hit.MaterialIndex].DiffuseTextureIndex,
                                         vertex.Hit.TexCoord));
        }
        return vertex;
    }
};
bool Begin(VcmSubpath& path, VcmWalkOptions options, const WalkContext& context)
{
    if (!options.VertexLimit || options.VertexLimit > MaximumVcmVertices)
        return false;
    path.VertexLimit = options.VertexLimit;
    if (!context.Stopped())
        return true;
    path.End = VcmWalkEnd::Cancelled;
    return false;
}
void Trace(const WalkContext& context, VcmSubpath& path, Ray ray, Float3 throughput,
           VcmDensity directionDensity, const SampleContext& samples, VcmWalkOptions options)
{
    const uint32_t domain = path.Mode == TransportMode::Radiance ? CameraDomain : LightDomain;
    uint32_t interiorMaterial = InvalidRayPrimitiveIndex;
    while (path.Count < path.VertexLimit)
    {
        if (context.Stopped())
        {
            path.End = VcmWalkEnd::Cancelled;
            return;
        }
        if (!ValidPower(throughput) || !directionDensity.Positive ||
            !std::isfinite(directionDensity.Log2Value))
        {
            path.End = VcmWalkEnd::NumericalFailure;
            return;
        }
        if (!Nonzero(throughput))
        {
            path.End = VcmWalkEnd::Absorbed;
            return;
        }
        auto& rays = path.Mode == TransportMode::Radiance && path.Count == 1
                         ? context.Statistics.PrimaryRays
                         : context.Statistics.ReflectionRays;
        rays.fetch_add(1, std::memory_order_relaxed);
        VcmVertex vertex = context.Intersect(ray);
        const auto& previous = path.Vertices[path.Count - 1];
        const bool firstInfinite = path.Count == 1 && (path.Light.Emission.Flags & EmissionInfinite);
        if (path.Count == 1 && vertex.EnclosingMaterial != InvalidRayPrimitiveIndex)
            interiorMaterial = vertex.EnclosingMaterial;
        const uint32_t dimension = domain ^ (path.Count * BounceSeedMultiplier);
        if (interiorMaterial != InvalidRayPrimitiveIndex)
        {
            const auto& material = context.World.Materials[interiorMaterial];
            const float endpoint = vertex.Kind == VcmVertexKind::Environment
                ? ray.MaxDistance : vertex.Hit.Distance;
            const auto flight = SampleSubsurfaceFlight(material.SubsurfaceMeanFreePath,
                material.Albedo * material.SubsurfaceAlbedoScale, ray.Direction, endpoint,
                Sample2D(samples, dimension ^ InteriorFlightDomain).X);
            if (flight.Scattered)
            {
                // Store incident throughput before the local albedo/phase factor.
                // A merged photon and camera collision then apply that factor once.
                vertex.Kind = VcmVertexKind::Volume;
                vertex.Position = ray.Origin + ray.Direction * flight.Distance;
                vertex.Hit = {};
                vertex.Hit.MaterialIndex = interiorMaterial;
                vertex.Emitter = InvalidEmitterIndex;
            }
        }
        vertex.Throughput = throughput;
        if (vertex.Kind == VcmVertexKind::Environment)
            vertex.Arrival = directionDensity;
        else if (firstInfinite)
            vertex.Arrival = VcmProjectedDiskDensity(MakeVcmDensity(path.Light.Emission.PositionPdf),
                                                     vertex.Hit.GeometricNormal, ray.Direction);
        else
            vertex.Arrival = EdgeDensity(context.World, directionDensity, previous, vertex);
        if (!vertex.Arrival.Positive || !Finite(vertex.Position))
        {
            path.End = VcmWalkEnd::NumericalFailure;
            return;
        }
        path.Vertices[path.Count++] = vertex;
        auto& stored = path.Vertices[path.Count - 1];
        if (stored.Kind != VcmVertexKind::Surface && stored.Kind != VcmVertexKind::Volume)
        {
            path.End = stored.Kind == VcmVertexKind::Environment ? VcmWalkEnd::Escaped : VcmWalkEnd::Absorbed;
            return;
        }
        if (path.Count == path.VertexLimit)
            break;
        const uint32_t scatterDimension = domain ^ (path.Count * BounceSeedMultiplier);
        stored.Continuation = SampleVcmScattering(context.World, stored,
                                                  Sample2D(samples, scatterDimension ^
                                                                    (stored.Kind == VcmVertexKind::Volume
                                                                        ? InteriorPhaseDomain : 0)),
                                                  Sample2D(samples, scatterDimension ^ LobeSelectionScramble).X,
                                                  path.Mode, options.ImportanceSampling);
        const auto& event = stored.Continuation;
        if (event.Event == ScatteringNone)
        {
            path.End = VcmWalkEnd::Absorbed;
            return;
        }
        if (!(event.Pdf > 0) || !(event.ReversePdf >= 0) || !std::isfinite(event.ReversePdf) ||
            !Finite(event.Direction) || !ValidPower(event.Weight))
        {
            path.End = VcmWalkEnd::NumericalFailure;
            return;
        }
        const auto reverse = MakeVcmDensity(event.ReversePdf);
        stored.ReturnToPrevious = firstInfinite
                                      ? reverse
                                      : EdgeDensity(context.World, reverse, stored, previous);
        throughput = Multiply(throughput, event.Weight);
        directionDensity = MakeVcmDensity(event.Pdf);
        ray = {Depart(stored, event.Direction), event.Direction, RayEpsilon, MaximumRayDistance};
        if (stored.Kind == VcmVertexKind::Surface &&
            context.World.Materials[stored.Hit.MaterialIndex].SubsurfaceMeanFreePath > 0 &&
            (event.Event & ScatteringTransmission))
            interiorMaterial = stored.Hit.FrontFace ? stored.Hit.MaterialIndex : InvalidRayPrimitiveIndex;
    }
    path.End = VcmWalkEnd::VertexLimit;
}
} // namespace
SurfaceEvaluation EvaluateVcmScattering(const Scene& scene, const VcmVertex& vertex, Float3 view,
                                        Float3 outgoing, TransportMode mode, bool importance)
{
    if (vertex.Kind == VcmVertexKind::Volume)
    {
        const auto& material = scene.Materials[vertex.Hit.MaterialIndex];
        const float phase = SubsurfacePhase(Dot(-view, outgoing), material.SubsurfaceAnisotropy);
        const float pdf = importance ? phase : float(1 / (4 * VcmPi));
        return {material.Albedo * (material.SubsurfaceAlbedoScale * phase), pdf, pdf};
    }
    if (vertex.Kind != VcmVertexKind::Surface)
        return {};
    Material material = scene.Materials[vertex.Hit.MaterialIndex];
    float continuousFraction = 1;
    if (material.Model == MaterialModel::Legacy)
    {
        continuousFraction -= std::clamp(material.Reflectivity, 0.0f, 1.0f);
        material.Model = MaterialModel::Diffuse;
    }
    auto result =
        SurfaceBsdf(material, Orient(vertex, view), view, vertex.Color, mode).Evaluate(outgoing, importance);
    result.Value = result.Value * continuousFraction;
    result.Pdf *= continuousFraction;
    result.ReversePdf *= continuousFraction;
    return result;
}
SurfaceSample SampleVcmScattering(const Scene& scene, const VcmVertex& vertex, Float2 variates,
                                  float lobeVariate, TransportMode mode, bool importance)
{
    if (vertex.Kind == VcmVertexKind::Volume)
    {
        const auto& material = scene.Materials[vertex.Hit.MaterialIndex];
        const auto phase = SampleSubsurfaceDirection(-vertex.ToPrevious, material.SubsurfaceAnisotropy,
                                                     variates, importance);
        return {phase.Direction, material.Albedo * (material.SubsurfaceAlbedoScale * phase.Weight),
                phase.Pdf, phase.Pdf, 1, ScatteringGlossy};
    }
    if (vertex.Kind != VcmVertexKind::Surface)
        return {};
    Material material = scene.Materials[vertex.Hit.MaterialIndex];
    const auto hit = Orient(vertex, vertex.ToPrevious);
    const float mirrorProbability =
        material.Model == MaterialModel::Legacy ? std::clamp(material.Reflectivity, 0.0f, 1.0f) : 0;
    if (lobeVariate < mirrorProbability)
    {
        const Float3 outgoing = Normalize(Reflect(-vertex.ToPrevious, hit.Normal));
        const double viewGeometry = Dot(vertex.ToPrevious, hit.GeometricNormal);
        const double lightGeometry = Dot(outgoing, hit.GeometricNormal);
        const double viewShading = Dot(vertex.ToPrevious, hit.Normal),
                     lightShading = Dot(outgoing, hit.Normal);
        if (!(viewGeometry * lightGeometry > 0 && viewShading > 0 && lightShading > 0))
            return {};
        const float adjoint =
            mode == TransportMode::Importance
                ? float(std::abs(lightGeometry * viewShading / (viewGeometry * lightShading)))
                : 1;
        return {outgoing, Float3(adjoint, adjoint, adjoint),     mirrorProbability, mirrorProbability,
                1,        ScatteringDelta | ScatteringReflection};
    }
    if (material.Model == MaterialModel::Legacy)
        material.Model = MaterialModel::Diffuse;
    auto result = SurfaceBsdf(material, hit, vertex.ToPrevious, vertex.Color, mode)
                      .Sample(variates, lobeVariate, importance);
    // The selected diffuse branch cancels its mixture factor from throughput,
    // but its forward/reverse sampling densities still include that probability.
    result.Pdf *= 1 - mirrorProbability;
    result.ReversePdf *= 1 - mirrorProbability;
    return result;
}
VcmSubpath TraceVcmCameraPath(const Scene& scene, const SceneAccelerator& accelerator,
                              const PreparedEmissionScene& emitters, const CameraFrame& frame, Float2 raster,
                              const SampleContext& samples, VcmWalkOptions options,
                              AtomicStatistics& statistics, const CancellationProbe& cancellation)
{
    VcmSubpath path;
    const WalkContext context{scene, accelerator, emitters, statistics, cancellation};
    if (!Begin(path, options, context))
        return path;
    path.Camera = SampleCameraEndpoint(frame, raster, Sample2D(samples, LensSampleDimension));
    if (!(path.Camera.Flags & CameraEndpointValid))
    {
        path.End = VcmWalkEnd::NullEndpoint;
        return path;
    }
    auto& root = path.Vertices[path.Count++];
    root.Kind = VcmVertexKind::Camera;
    root.Position = path.Camera.Origin;
    root.Hit.GeometricNormal = frame.Forward;
    root.Throughput = Float3(1, 1, 1) * path.Camera.Weight;
    root.Arrival = MakeVcmDensity(path.Camera.PositionPdf);
    if (!root.Arrival.Positive || !ValidPower(root.Throughput))
    {
        path.End = VcmWalkEnd::NumericalFailure;
        return path;
    }
    Trace(context, path, {root.Position, path.Camera.Direction}, root.Throughput,
          MakeVcmDensity(path.Camera.DirectionPdf), samples, options);
    return path;
}
VcmSubpath TraceVcmLightPath(const Scene& scene, const SceneAccelerator& accelerator,
                             const PreparedEmissionScene& emitters, const SampleContext& samples,
                             VcmWalkOptions options, AtomicStatistics& statistics,
                             const CancellationProbe& cancellation)
{
    VcmSubpath path;
    path.Mode = TransportMode::Importance;
    const WalkContext context{scene, accelerator, emitters, statistics, cancellation};
    if (!Begin(path, options, context))
        return path;
    path.Light = SampleEmitterEndpoint(emitters, Sample2D(samples, SourceDomain).X,
                                       Sample2D(samples, PositionDomain), Sample2D(samples, DirectionDomain));
    const auto& emitted = path.Light.Emission;
    if (!(emitted.Flags & EmissionValid))
    {
        path.End = VcmWalkEnd::NullEndpoint;
        return path;
    }
    auto& root = path.Vertices[path.Count++];
    root.Kind = VcmVertexKind::Light;
    root.Position = emitted.Origin;
    root.Emitter = path.Light.EmitterIndex;
    root.Hit.GeometricNormal = root.Hit.Normal = emitted.Normal;
    root.Hit.PrimitiveIndex = path.Light.Primitive;
    const float endpointDensity =
        emitted.Flags & EmissionInfinite ? emitted.DirectionPdf : emitted.PositionPdf;
    root.Arrival = MakeVcmDensity(endpointDensity);
    root.Throughput = emitted.Value / (path.Light.SelectionProbability * endpointDensity);
    if (!root.Arrival.Positive || !ValidPower(root.Throughput) || !Finite(root.Position))
    {
        path.End = VcmWalkEnd::NumericalFailure;
        return path;
    }
    const Float3 origin =
        emitted.Flags & EmissionDeltaPosition ? emitted.Origin : Depart(root, emitted.Direction);
    Trace(context, path, {origin, emitted.Direction}, emitted.Weight, MakeVcmDensity(emitted.DirectionPdf),
          samples, options);
    return path;
}
namespace
{
constexpr double MinimumMergeNormalAgreement = 0.9, MergePlaneThickness = 0.1;
using Spectrum = std::array<double, 3>;
Spectrum Product(std::initializer_list<Float3> factors, double scale)
{
    Spectrum value{scale, scale, scale};
    for (Float3 factor : factors)
    {
        value[0] *= factor.X;
        value[1] *= factor.Y;
        value[2] *= factor.Z;
    }
    return value;
}
bool Nonzero(const Spectrum& value)
{
    return value[0] > 0 || value[1] > 0 || value[2] > 0;
}
void Resolve(VcmContribution& result, const Spectrum& value, double weight)
{
    result.MisWeight = weight;
    if (!(weight >= 0) || !std::isfinite(weight))
    {
        result.NumericalFailure = true;
        return;
    }
    Float3 radiance(float(value[0] * weight), float(value[1] * weight), float(value[2] * weight));
    result.NumericalFailure = !ValidPower(radiance);
    if (!result.NumericalFailure)
        result.Radiance = radiance;
}
bool Valid(const VcmCouplingOptions& options)
{
    return options.MaximumVertices >= 2 && options.MaximumVertices <= MaximumVcmVertices &&
           options.MergeRadius >= 0 && std::isfinite(options.MergeRadius) && options.MergingSamples >= 0 &&
           options.VolumeMergeRadius >= 0 && std::isfinite(options.VolumeMergeRadius) &&
           std::isfinite(options.MergingSamples) && options.LightTracingSamples >= 0 &&
           std::isfinite(options.LightTracingSamples) &&
           (options.Mis == MisHeuristic::None || options.Mis == MisHeuristic::Balance ||
            options.Mis == MisHeuristic::Power);
}
bool Usable(const VcmSubpath& path)
{
    return path.Count > 0 && path.Count <= path.VertexLimit && path.VertexLimit <= MaximumVcmVertices &&
           path.End != VcmWalkEnd::InvalidInput && path.End != VcmWalkEnd::NumericalFailure &&
           path.End != VcmWalkEnd::Cancelled;
}
CameraEndpointEvaluation CameraPdf(const VcmSubpath& path)
{
    const auto& camera = path.Camera;
    return {camera.Raster, camera.Importance, camera.PositionPdf, camera.DirectionPdf, camera.Flags};
}
Float3 SourceRadiance(const PackedEmitter& source, Float3 outgoing)
{
    if (source.Kind == EmitterKind::Environment)
    {
        const float blend = (1 - outgoing.Y) / 2;
        return source.BackgroundBottom * (1 - blend) + source.BackgroundTop * blend;
    }
    if (source.Kind == EmitterKind::Rectangle)
        return Dot(source.Direction, outgoing) > 0 ? source.Strength / (source.Width * source.Height)
                                                   : Float3{};
    // Geometric material emission is two-sided in the current scene contract.
    return source.Strength;
}
bool VisibleRay(const WalkContext& context, const Ray& ray)
{
    if (context.Stopped())
        return false;
    context.Statistics.ShadowRays.fetch_add(1, std::memory_order_relaxed);
    uint64_t tests = 0;
    bool blocked = context.Accelerator.Occluded(ray, tests);
    if (!blocked)
        for (const auto& source : context.Emitters.Sources)
            if (source.Kind == EmitterKind::Rectangle)
            {
                ++tests;
                float nearest = ray.MaxDistance;
                if (RectangleHit(source, ray, nearest))
                {
                    blocked = true;
                    break;
                }
            }
    context.Statistics.PrimitiveTests.fetch_add(tests, std::memory_order_relaxed);
    return !blocked;
}
bool VisibleSegment(const WalkContext& context, Float3 origin, Float3 target)
{
    const Float3 displacement = target - origin;
    const float distance = std::sqrt(Dot(displacement, displacement));
    return distance > 2 * RayEpsilon && std::isfinite(distance) &&
           VisibleRay(context, {origin, displacement / distance, RayEpsilon, distance - RayEpsilon});
}

// A merged vertex has two nearby positions. Keep the original incoming light
// edge on its left and original incoming camera edge on its right (report,
// section 4.3). Its BSDF is evaluated at the camera surface with those two
// directions. Replacing both positions with one point would perturb cached
// specular chains and compare a different set of path sampling probabilities.
VcmWeights CompletePathWeights(const Scene& scene, const PreparedEmissionScene& sources,
                               const VcmSubpath& light, uint32_t s, const VcmSubpath& camera, uint32_t t,
                               const VcmVertex& cameraRoot, const CameraEndpointEvaluation& cameraPdf,
                               bool merged, const VcmCouplingOptions& options)
{
    const uint32_t overlap = merged ? 1 : 0, count = s + t - overlap;
    std::array<const VcmVertex*, MaximumVcmVertices> left{}, right{};
    for (uint32_t i = 0; i < s; ++i)
        left[i] = right[i] = &light.Vertices[i];
    for (uint32_t i = 0; i < t; ++i)
    {
        const uint32_t index = s + i - overlap;
        right[index] = &camera.Vertices[t - 1 - i];
        if (!merged || i)
            left[index] = right[index];
    }
    left[count - 1] = right[count - 1] = &cameraRoot;
    const bool infinite = (light.Light.Emission.Flags & EmissionInfinite) != 0;
    auto displacement = [&](uint32_t from, uint32_t to) {
        return from < to ? left[to]->Position - right[from]->Position
                         : right[to]->Position - left[from]->Position;
    };
    auto direction = [&](uint32_t from, uint32_t to) {
        if (infinite && !from)
            return light.Light.Emission.Direction;
        if (infinite && !to)
            return -light.Light.Emission.Direction;
        return Normalize(displacement(from, to));
    };
    auto convert = [&](double probability, uint32_t from, uint32_t to) {
        const auto density = MakeVcmDensity(probability);
        const auto* origin = from < to ? right[from] : left[from];
        const auto* destination = from < to ? left[to] : right[to];
        return infinite && !to
                   ? density
                   : EdgeDensity(scene, density, *origin, *destination);
    };
    std::array<VcmVertexPdf, MaximumVcmVertices> densities{};
    for (uint32_t i = 1; i + 1 < count; ++i)
    {
        if (right[i]->Kind == VcmVertexKind::Volume)
            densities[i].Flags |= VcmVolumeScattering;
        float forward = 0, reverse = 0;
        const bool changed = merged ? i == s - 1 : (i == s || i + 1 == s);
        if (changed)
        {
            // Keep the directions actually used by the original random walks.
            // Reconstructing an unchanged edge from offset intersection points
            // can noticeably perturb a short specular segment's BSDF PDF.
            const Float3 toLight = merged || i + 1 == s ? left[i]->ToPrevious : direction(i, i - 1);
            const Float3 toCamera = merged || i == s ? right[i]->ToPrevious : direction(i, i + 1);
            const auto scattering = EvaluateVcmScattering(
                scene, *right[i], toLight, toCamera, TransportMode::Importance, options.ImportanceSampling);
            forward = scattering.Pdf;
            reverse = scattering.ReversePdf;
        }
        else
        {
            const auto& event = right[i]->Continuation;
            forward = i < s ? event.Pdf : event.ReversePdf;
            reverse = i < s ? event.ReversePdf : event.Pdf;
            if (event.Event & ScatteringDelta)
                densities[i].Flags |= VcmDeltaScattering;
        }
        densities[i + 1].Forward = convert(forward, i, i + 1);
        densities[i - 1].Reverse = convert(reverse, i, i - 1);
    }
    const Float3 emittedDirection = s == 1 && !infinite ? direction(0, 1) : light.Light.Emission.Direction;
    const auto emission =
        EmitterEndpointPdf(sources, light.Light.EmitterIndex, left[0]->Hit.GeometricNormal, emittedDirection);
    const auto positionDensity =
        MakeVcmDensity(emission.Flags & EmissionDeltaPosition ? 1 : emission.Position);
    const auto angularDensity =
        MakeVcmDensity(emission.Flags & EmissionDeltaDirection ? 1 : emission.Direction);
    const auto lightEndpoint = infinite ? angularDensity : positionDensity;
    densities[1].Forward =
        infinite ? VcmProjectedDiskDensity(positionDensity, left[1]->Hit.GeometricNormal, emittedDirection)
                 : convert(emission.Flags & EmissionDeltaDirection ? 1 : emission.Direction, 0, 1);
    if (emission.Flags & (EmissionDeltaPosition | EmissionDeltaDirection))
        densities[0].Flags |= VcmEndpointNotIntersectable;
    densities[count - 2].Reverse = convert(cameraPdf.DirectionPdf, count - 1, count - 2);
    densities[count - 1].Flags |= VcmEndpointNotIntersectable;
    const auto cameraEndpoint =
        MakeVcmDensity(cameraPdf.Flags & CameraDeltaPosition ? 1 : cameraPdf.PositionPdf);
    const auto unity = MakeVcmDensity(1);
    std::array<VcmStrategy, MaximumVcmStrategies> connections{}, mergings{};
    for (uint32_t cut = 0; cut < count; ++cut)
    {
        auto& connection = connections[cut];
        connection.Enabled = cut <= light.VertexLimit && count - cut <= camera.VertexLimit &&
                             (cut != 1 || options.NextEventEstimation) && (count > 2 || cut == 0);
        connection.LightEndpoint = cut ? lightEndpoint : unity;
        connection.LightSelection = cut ? emission.SelectionProbability : 1;
        connection.CameraEndpoint = cameraEndpoint;
        connection.SampleCount = cut + 1 == count ? options.LightTracingSamples : 1;
        if (cut < 2)
            continue;
        auto& merge = mergings[cut];
        merge.Enabled = cut <= light.VertexLimit && count + 1 - cut <= camera.VertexLimit;
        merge.LightEndpoint = lightEndpoint;
        merge.LightSelection = emission.SelectionProbability;
        merge.CameraEndpoint = cameraEndpoint;
        merge.SampleCount = options.MergingSamples;
    }
    return VertexConnectionMergingMisWeights(densities.data(), count, connections.data(), mergings.data(),
                                             options.MergeRadius, options.Mis, options.VolumeMergeRadius);
}
} // namespace
VcmContribution ConnectVcmSubpaths(const Scene& scene, const SceneAccelerator& accelerator,
                                   const PreparedEmissionScene& sources, const CameraFrame& frame,
                                   const VcmSubpath& light, uint32_t s, const VcmSubpath& camera, uint32_t t,
                                   Float2 lensVariates, const VcmCouplingOptions& options,
                                   AtomicStatistics& statistics, const CancellationProbe& cancel)
{
    VcmContribution result;
    if (!Valid(options) || !Usable(camera) || camera.Mode != TransportMode::Radiance || t < 1 ||
        t > camera.Count || s > MaximumVcmVertices || s + t > options.MaximumVertices || s + t < 2 ||
        (cancel && cancel()))
        return result;
    result.Raster = camera.Camera.Raster;
    const auto& eye = camera.Vertices[t - 1];
    const auto cameraPdf = CameraPdf(camera);
    if (!s)
    {
        if (t < 2 || eye.Emitter >= sources.Sources.size())
            return result;
        VcmSubpath endpoint;
        endpoint.VertexLimit = options.MaximumVertices;
        endpoint.Light.EmitterIndex = eye.Emitter;
        const auto emission =
            EmitterEndpointPdf(sources, eye.Emitter, eye.Hit.GeometricNormal, eye.ToPrevious);
        endpoint.Light.Emission.Flags = emission.Flags;
        endpoint.Light.Emission.Direction = eye.ToPrevious;
        const auto weights = CompletePathWeights(scene, sources, endpoint, 0, camera, t, camera.Vertices[0],
                                                 cameraPdf, false, options);
        Resolve(result,
                Product({eye.Throughput, SourceRadiance(sources.Sources[eye.Emitter], eye.ToPrevious)}, 1),
                weights.Connection[0]);
        return result;
    }
    if (!Usable(light) || light.Mode != TransportMode::Importance || s > light.Count ||
        light.Light.EmitterIndex >= sources.Sources.size() ||
        (s == 1 && (t == 1 || !options.NextEventEstimation)))
        return result;
    const WalkContext context{scene, accelerator, sources, statistics, cancel};
    if (s == 1)
    {
        if (eye.Kind != VcmVertexKind::Surface && eye.Kind != VcmVertexKind::Volume)
            return result;
        const auto& emitted = light.Light.Emission;
        const bool infinite = (emitted.Flags & EmissionInfinite) != 0;
        Float3 toLight;
        double geometry = 1;
        if (infinite)
            toLight = -emitted.Direction;
        else
        {
            const Float3 delta = light.Vertices[0].Position - eye.Position;
            const double distance = std::hypot(double(delta.X), double(delta.Y), double(delta.Z));
            if (!(distance > 0))
                return result;
            toLight = delta / float(distance);
            geometry = (emitted.Flags & EmissionDeltaPosition
                            ? 1
                            : std::abs(Dot(light.Vertices[0].Hit.GeometricNormal, toLight))) /
                       (distance * distance);
        }
        const auto scattering = EvaluateVcmScattering(scene, eye, eye.ToPrevious, toLight,
                                                      TransportMode::Radiance, options.ImportanceSampling);
        const double endpointDensity = infinite ? emitted.DirectionPdf : emitted.PositionPdf;
        const auto value = Product({eye.Throughput, scattering.Value,
                                    SourceRadiance(sources.Sources[light.Light.EmitterIndex], -toLight)},
                                   (eye.Kind == VcmVertexKind::Volume ? 1 :
                                       std::abs(Dot(eye.Hit.Normal, toLight))) * geometry *
                                       (infinite ? 1 : SegmentTransmittance(scene, eye, light.Vertices[0])) /
                                       (double(light.Light.SelectionProbability) * endpointDensity));
        if (!Nonzero(value))
            return result;
        const bool visible =
            infinite
                ? VisibleRay(context, {Depart(eye, toLight), toLight, RayEpsilon, MaximumRayDistance})
                : VisibleSegment(context, Depart(eye, toLight),
                                 emitted.Flags & EmissionDeltaPosition ? light.Vertices[0].Position
                                                                       : Depart(light.Vertices[0], -toLight));
        if (!visible)
            return result;
        const auto weights = CompletePathWeights(scene, sources, light, s, camera, t, camera.Vertices[0],
                                                 cameraPdf, false, options);
        Resolve(result, value, weights.Connection[s]);
        return result;
    }
    const auto& photon = light.Vertices[s - 1];
    if ((photon.Kind != VcmVertexKind::Surface && photon.Kind != VcmVertexKind::Volume) ||
        (t > 1 && eye.Kind != VcmVertexKind::Surface && eye.Kind != VcmVertexKind::Volume))
        return result;
    VcmVertex lensRoot = camera.Vertices[0];
    auto endpointPdf = cameraPdf;
    Float3 toCamera, target;
    Spectrum value{};
    if (t == 1)
    {
        const auto projection = SampleCameraConnection(frame, photon.Position, lensVariates);
        if (!(projection.Flags & CameraEndpointValid) || !(options.LightTracingSamples > 0))
            return result;
        toCamera = projection.Direction;
        lensRoot.Position = target = projection.Position;
        lensRoot.Hit.GeometricNormal = frame.Forward;
        const auto scattering = EvaluateVcmScattering(scene, photon, photon.ToPrevious, toCamera,
                                                      TransportMode::Importance, options.ImportanceSampling);
        value = Product({photon.Throughput, scattering.Value},
                        (photon.Kind == VcmVertexKind::Volume ? 1 :
                            std::abs(Dot(photon.Hit.Normal, toCamera))) *
                            double(projection.Weight) * SegmentTransmittance(scene, photon, lensRoot));
        endpointPdf = EvaluateCameraEndpoint(frame, projection.LensPoint, -toCamera);
        result.Raster = projection.Raster;
        result.Splat = true;
    }
    else
    {
        const Float3 delta = eye.Position - photon.Position;
        const double distance = std::hypot(double(delta.X), double(delta.Y), double(delta.Z));
        if (!(distance > 0))
            return result;
        toCamera = delta / float(distance);
        const auto importance = EvaluateVcmScattering(scene, photon, photon.ToPrevious, toCamera,
                                                      TransportMode::Importance, options.ImportanceSampling);
        const auto radiance = EvaluateVcmScattering(scene, eye, eye.ToPrevious, -toCamera,
                                                    TransportMode::Radiance, options.ImportanceSampling);
        const double lightCosine = photon.Kind == VcmVertexKind::Volume ? 1 :
            std::abs(double(Dot(photon.Hit.Normal, toCamera)));
        const double eyeCosine = eye.Kind == VcmVertexKind::Volume ? 1 :
            std::abs(double(Dot(eye.Hit.Normal, toCamera)));
        const double geometry = lightCosine * eyeCosine /
            (distance * distance) * SegmentTransmittance(scene, photon, eye);
        value = Product({photon.Throughput, importance.Value, eye.Throughput, radiance.Value}, geometry);
        target = Depart(eye, -toCamera);
    }
    if (!Nonzero(value) || !(endpointPdf.Flags & CameraEndpointValid) ||
        !VisibleSegment(context, Depart(photon, toCamera), target))
        return result;
    const auto weights =
        CompletePathWeights(scene, sources, light, s, camera, t, lensRoot, endpointPdf, false, options);
    Resolve(result, value, weights.Connection[s]);
    return result;
}
VcmContribution MergeVcmSubpaths(const Scene& scene, const PreparedEmissionScene& sources,
                                 const VcmSubpath& light, uint32_t s, const VcmSubpath& camera, uint32_t t,
                                 const VcmCouplingOptions& options, const SceneAccelerator* accelerator)
{
    VcmContribution result;
    if (!Valid(options) || !Usable(light) || !Usable(camera) || light.Mode != TransportMode::Importance ||
        camera.Mode != TransportMode::Radiance || s < 2 || t < 2 || s > light.Count || t > camera.Count ||
        s + t - 1 > options.MaximumVertices || light.Light.EmitterIndex >= sources.Sources.size() ||
        !(options.MergeRadius > 0 && options.MergingSamples > 0))
        return result;
    const auto& photon = light.Vertices[s - 1];
    const auto& eye = camera.Vertices[t - 1];
    if (photon.Kind != eye.Kind ||
        (eye.Kind != VcmVertexKind::Surface && eye.Kind != VcmVertexKind::Volume))
        return result;
    const bool volume = eye.Kind == VcmVertexKind::Volume;
    if (volume && photon.Hit.MaterialIndex != eye.Hit.MaterialIndex)
        return result;
    const double radius = volume && options.VolumeMergeRadius > 0
        ? options.VolumeMergeRadius : options.MergeRadius;
    const Float3 offset = photon.Position - eye.Position;
    const double distance = std::hypot(double(offset.X), double(offset.Y), double(offset.Z));
    const double planeDistance =
        std::abs(double(offset.X) * eye.Hit.GeometricNormal.X + double(offset.Y) * eye.Hit.GeometricNormal.Y +
                 double(offset.Z) * eye.Hit.GeometricNormal.Z);
    // Finite-radius sheet filters suppress leakage. Their bias vanishes locally
    // on a smooth surface as the progressive disk contracts.
    if (distance > radius || (!volume &&
        (planeDistance > MergePlaneThickness * radius ||
         Dot(photon.Hit.GeometricNormal, eye.Hit.GeometricNormal) < MinimumMergeNormalAgreement)))
        return result;
    if (volume && accelerator && distance > 2 * RayEpsilon)
    {
        uint64_t tests = 0;
        const Ray gap{eye.Position, offset / float(distance), RayEpsilon, float(distance) - RayEpsilon};
        if (accelerator->Occluded(gap, tests))
            return result;
    }
    const auto scattering = EvaluateVcmScattering(scene, eye, eye.ToPrevious, photon.ToPrevious,
                                                  TransportMode::Radiance, options.ImportanceSampling);
    const double projectedArea = volume ? 1 : std::abs(Dot(eye.Hit.GeometricNormal, photon.ToPrevious));
    if (!(projectedArea > 0) || !Nonzero(scattering.Value))
        return result;
    const auto weights = CompletePathWeights(scene, sources, light, s, camera, t, camera.Vertices[0],
                                             CameraPdf(camera), true, options);
    const VcmDensity measure = volume ? VcmMergeVolume(radius) : VcmMergeArea(radius);
    const double kernelDensity = std::exp2(-measure.Log2Value) / options.MergingSamples;
    const double areaCorrection = volume ? 1 :
        std::abs(Dot(eye.Hit.Normal, photon.ToPrevious)) / projectedArea;
    result.Raster = camera.Camera.Raster;
    Resolve(result,
            Product({photon.Throughput, eye.Throughput, scattering.Value}, kernelDensity * areaCorrection),
            weights.Merging[s]);
    return result;
}
namespace
{
constexpr uint32_t SplatBatchSize = 128;
constexpr uint32_t PopulationDomain = 0x56434d4eu, PassDomain = 0x56435041u;
constexpr uint32_t PairingDomain = 0x56435052u, SplatLensDomain = 0x56434c4eu;
constexpr uint64_t MaximumPathCacheBytes = 512ull * 1024 * 1024;
struct IndexedVertex
{
    uint32_t Path, Prefix;
};
struct PathSplats
{
    std::array<VcmContribution, MaximumVcmVertices> Values;
    uint32_t Count = 0;
};
SampleContext LightPopulationSample(const VertexConnectionMergingOptions& options, uint32_t pass,
                                    uint32_t path)
{
    // Emission populations are global and independently scrambled each pass.
    // Camera sequence selection remains a separate, pixel-addressed policy.
    const uint32_t seed = Hash(options.Seed ^ PopulationDomain) ^ Hash(pass ^ PassDomain) ^ Hash(path);
    return {path, pass, seed, 0, 1, SampleSequence::Independent};
}
bool FailedWalk(const VcmSubpath& path)
{
    return path.End == VcmWalkEnd::InvalidInput || path.End == VcmWalkEnd::NumericalFailure ||
           (path.Mode == TransportMode::Radiance && path.End == VcmWalkEnd::NullEndpoint);
}
void Add(Spectrum& target, Float3 value, double scale = 1)
{
    target[0] += double(value.X) * scale;
    target[1] += double(value.Y) * scale;
    target[2] += double(value.Z) * scale;
}
} // namespace
double VcmPassRadius(double initialRadius, double alpha, uint32_t pass)
{
    if (!(initialRadius >= 0) || !std::isfinite(initialRadius) || !(alpha > 0 && alpha < 1))
        return std::numeric_limits<double>::quiet_NaN();
    // Progressive VCM bandwidth: r_i = r_0 / (i+1)^((1-alpha)/2).
    // Each pass estimates radiance anew; averaging these estimates is valid.
    // This is distinct from PPM's persistent per-pixel tau/N recurrence.
    // See Georgiev et al. (2012), and SmallVCM's progressive radius schedule:
    // https://github.com/SmallVCM/SmallVCM/blob/master/src/vertexcm.hxx
    return initialRadius / std::pow(double(pass) + 1, (1 - alpha) / 2);
}
double VcmPassVolumeRadius(double initialRadius, double alpha, uint32_t pass)
{
    if (!(initialRadius >= 0) || !std::isfinite(initialRadius) || !(alpha > 0 && alpha < 1))
        return std::numeric_limits<double>::quiet_NaN();
    // A 3D collision cache uses a sphere kernel and a cube-root bandwidth.
    return initialRadius / std::pow(double(pass) + 1, (1 - alpha) / 3);
}
bool RenderVertexConnectionMergingFrame(const Scene& scene, const SceneAccelerator& accelerator,
                                        const CameraFrame& frame, const RenderSettings& settings,
                                        LinearFilm& film, AtomicStatistics& statistics,
                                        const CancellationProbe& cancel, const VcmDispatcher& dispatch,
                                        std::string& error, VcmDiagnostics* diagnostics)
try
{
    error.clear();
    if (diagnostics)
        *diagnostics = {};
    if (!ValidateTileSettings(settings, error))
        return false;
    if (settings.Integrator != Algorithm::VertexConnectionMerging || !settings.SamplesPerPixel || !dispatch ||
        settings.MaxReflectionBounces > MaximumVcmReflections || settings.Adaptive.Enabled ||
        settings.Transport.RussianRoulette || !settings.ShadowsEnabled || settings.ShadowStrength != 1)
    {
        error = "VCM requires fixed passes/depth, at most 32 reflections, no adaptive stopping or roulette, "
                "and full visibility.";
        return false;
    }
    auto options = settings.Vcm;
    options.Seed ^= settings.SamplingSeed;
    if (uint64_t(options.LightPathsPerPass) * sizeof(VcmSubpath) > MaximumPathCacheBytes)
    {
        error = "CPU VCM light paths exceed the 512 MiB path-cache budget; reduce light paths per pass.";
        return false;
    }
    const auto sources = PrepareEmitters(scene);
    for (const auto& source : sources.Sources)
        if (!(source.Flags & EmitterCastsShadows))
        {
            error = "VCM requires shadow casting on every emitter.";
            return false;
        }
    const WalkContext context{scene, accelerator, sources, statistics, cancel};
    const uint32_t maximumVertices = settings.MaxReflectionBounces + 1 + 2;
    const uint32_t pixels = settings.Width * settings.Height;
    const uint32_t fullWidth = FullImageWidth(settings), fullHeight = FullImageHeight(settings);
    const uint64_t fullPixels = uint64_t(fullWidth) * fullHeight;
    const double initialRadius = double(sources.Radius) * options.InitialRadiusScale;
    if (!(initialRadius >= 0) || !std::isfinite(initialRadius))
    {
        error = "Invalid VCM emission domain.";
        return false;
    }
    VcmCouplingOptions coupling;
    coupling.MaximumVertices = maximumVertices;
    coupling.MergingSamples = options.LightPathsPerPass;
    coupling.LightTracingSamples = double(options.LightPathsPerPass) / double(fullPixels);
    coupling.ImportanceSampling = settings.Transport.ImportanceSampling;
    coupling.NextEventEstimation = settings.Transport.NextEventEstimation;
    coupling.Mis = settings.Transport.Mis;
    std::vector<VcmSubpath> lightPaths(options.LightPathsPerPass);
    std::vector<Spectrum> accumulated(pixels), passRadiance(pixels);
    std::vector<PathSplats> splats(SplatBatchSize);
    std::atomic<bool> invalid{false};
    uint64_t mergedPairs = 0, cameraSplats = 0;
    film = LinearFilm(settings.Width, settings.Height);
    if (diagnostics)
        diagnostics->InitialRadius = initialRadius;
    for (uint32_t pass = 0; pass < settings.SamplesPerPixel; ++pass)
    {
        if (context.Stopped())
            return false;
        coupling.MergeRadius = VcmPassRadius(initialRadius, options.Alpha, pass);
        coupling.VolumeMergeRadius = VcmPassVolumeRadius(initialRadius, options.Alpha, pass);
        std::fill(passRadiance.begin(), passRadiance.end(), Spectrum{});
        dispatch(options.LightPathsPerPass, [&](uint32_t path) {
            lightPaths[path] =
                TraceVcmLightPath(scene, accelerator, sources, LightPopulationSample(options, pass, path),
                                  {maximumVertices, coupling.ImportanceSampling}, statistics, cancel);
            if (FailedWalk(lightPaths[path]))
                invalid.store(true, std::memory_order_relaxed);
        });
        if (context.Stopped())
            return false;
        if (invalid.load(std::memory_order_relaxed))
        {
            error = "Invalid VCM light-path transport.";
            return false;
        }
        std::vector<Float3> positions;
        std::vector<IndexedVertex> references;
        const size_t maximumPoints = size_t(options.LightPathsPerPass) * (maximumVertices - 2);
        positions.reserve(maximumPoints);
        references.reserve(maximumPoints);
        for (uint32_t path = 0; path < options.LightPathsPerPass; ++path)
        {
            if (context.Stopped())
                return false;
            const auto& light = lightPaths[path];
            for (uint32_t prefix = 2; prefix <= light.Count && prefix < maximumVertices; ++prefix)
                if (light.Vertices[prefix - 1].Kind == VcmVertexKind::Surface ||
                    light.Vertices[prefix - 1].Kind == VcmVertexKind::Volume)
                {
                    positions.push_back(light.Vertices[prefix - 1].Position);
                    references.push_back({path, prefix});
                }
        }
        const PointKdTree index(std::move(positions), cancel);
        if (index.WasCancelled() || context.Stopped())
            return false;
        std::atomic<uint64_t> passMerges{0};
        dispatch(pixels, [&](uint32_t pixel) {
            if (context.Stopped())
                return;
            const uint32_t x = pixel % settings.Width, y = pixel / settings.Width;
            auto samples = MakeSampleContext(settings, x, y, pass);
            samples.PixelSeed = Hash(samples.PixelSeed ^ Hash(options.Seed ^ CameraDomain));
            const auto raster = CameraPixelRaster(samples.PixelX, samples.PixelY, fullWidth, fullHeight,
                                                  Sample2D(samples, 0));
            const auto camera =
                TraceVcmCameraPath(scene, accelerator, sources, frame, raster, samples,
                                   {maximumVertices, coupling.ImportanceSampling}, statistics, cancel);
            if (FailedWalk(camera))
            {
                invalid.store(true, std::memory_order_relaxed);
                return;
            }
            const uint32_t paired =
                std::min(uint32_t(double(Sample2D(samples, PairingDomain).X) * options.LightPathsPerPass),
                         options.LightPathsPerPass - 1);
            const auto& light = lightPaths[paired];
            uint64_t localMerges = 0;
            auto collect = [&](const VcmContribution& contribution) {
                if (contribution.NumericalFailure)
                    invalid.store(true, std::memory_order_relaxed);
                else
                    Add(passRadiance[pixel], contribution.Radiance);
            };
            for (uint32_t t = 2; t <= camera.Count; ++t)
            {
                if (context.Stopped())
                    return;
                collect(ConnectVcmSubpaths(scene, accelerator, sources, frame, light, 0, camera, t, {},
                                           coupling, statistics, cancel));
                if (t + 1 <= maximumVertices)
                    collect(ConnectVcmSubpaths(scene, accelerator, sources, frame, light, 1, camera, t, {},
                                               coupling, statistics, cancel));
                for (uint32_t s = 2; s <= light.Count && s + t <= maximumVertices; ++s)
                    collect(ConnectVcmSubpaths(scene, accelerator, sources, frame, light, s, camera, t, {},
                                               coupling, statistics, cancel));
                const auto& eye = camera.Vertices[t - 1];
                if ((eye.Kind != VcmVertexKind::Surface && eye.Kind != VcmVertexKind::Volume) ||
                    !(coupling.MergeRadius > 0))
                    continue;
                const double queryRadius = eye.Kind == VcmVertexKind::Volume
                    ? coupling.VolumeMergeRadius : coupling.MergeRadius;
                index.Search(
                    eye.Position, queryRadius * queryRadius, [&](uint32_t found, double) {
                        if (context.Stopped())
                            return;
                        const auto& reference = references[found];
                        if (reference.Prefix + t - 1 > maximumVertices)
                            return;
                        const auto contribution = MergeVcmSubpaths(scene, sources, lightPaths[reference.Path],
                                                                   reference.Prefix, camera, t, coupling,
                                                                   &accelerator);
                        collect(contribution);
                        localMerges += contribution.MisWeight > 0 && Nonzero(contribution.Radiance);
                    });
            }
            passMerges.fetch_add(localMerges, std::memory_order_relaxed);
        });
        if (context.Stopped())
            return false;
        if (invalid.load(std::memory_order_relaxed))
        {
            error = "Invalid VCM camera connection or merging transport.";
            return false;
        }
        // Lens contributions originate over the full emission population, even
        // for a crop. Bounded batches reduce in source-path order, without float
        // atomics or thread-dependent accumulation into destination pixels.
        for (uint32_t first = 0; first < options.LightPathsPerPass; first += SplatBatchSize)
        {
            if (context.Stopped())
                return false;
            const uint32_t count = std::min(SplatBatchSize, options.LightPathsPerPass - first);
            dispatch(count, [&](uint32_t lane) {
                auto& output = splats[lane];
                output.Count = 0;
                const auto samples = LightPopulationSample(options, pass, first + lane);
                auto camera = TraceVcmCameraPath(scene, accelerator, sources, frame, {0.5f, 0.5f}, samples,
                                                 {1, coupling.ImportanceSampling}, statistics, cancel);
                camera.VertexLimit = maximumVertices;
                if (FailedWalk(camera))
                    invalid.store(true, std::memory_order_relaxed);
                const auto& light = lightPaths[first + lane];
                for (uint32_t s = 2; s <= light.Count && s + 1 <= maximumVertices; ++s)
                {
                    if (context.Stopped())
                        return;
                    const auto lens = Sample2D(samples, SplatLensDomain ^ (s * BounceSeedMultiplier));
                    const auto contribution =
                        ConnectVcmSubpaths(scene, accelerator, sources, frame, light, s, camera, 1, lens,
                                           coupling, statistics, cancel);
                    if (contribution.NumericalFailure)
                        invalid.store(true, std::memory_order_relaxed);
                    if (Nonzero(contribution.Radiance))
                        output.Values[output.Count++] = contribution;
                }
            });
            if (context.Stopped())
                return false;
            if (invalid.load(std::memory_order_relaxed))
            {
                error = "Invalid VCM light-to-camera transport.";
                return false;
            }
            for (uint32_t lane = 0; lane < count; ++lane)
                for (uint32_t i = 0; i < splats[lane].Count; ++i)
                {
                    ++cameraSplats;
                    const auto& splat = splats[lane].Values[i];
                    const uint32_t x = std::min(uint32_t(splat.Raster.X * fullWidth), fullWidth - 1);
                    const uint32_t y = std::min(uint32_t(splat.Raster.Y * fullHeight), fullHeight - 1);
                    if (x >= settings.TileX && y >= settings.TileY && x - settings.TileX < settings.Width &&
                        y - settings.TileY < settings.Height)
                        Add(passRadiance[size_t(y - settings.TileY) * settings.Width + x - settings.TileX],
                            splat.Radiance, 1 / coupling.LightTracingSamples);
                }
        }
        for (uint32_t pixel = 0; pixel < pixels; ++pixel)
        {
            for (size_t channel = 0; channel < accumulated[pixel].size(); ++channel)
                accumulated[pixel][channel] += passRadiance[pixel][channel];
            const double passes = double(pass) + 1;
            const Float3 mean(float(accumulated[pixel][0] / passes), float(accumulated[pixel][1] / passes),
                              float(accumulated[pixel][2] / passes));
            if (!ValidPower(mean))
            {
                error = "VCM film accumulation overflow.";
                return false;
            }
            // Changing bandwidths and shared photon populations do not provide
            // the independent, identically distributed variance used by pilots.
            film.Pixels[pixel] = {mean, 0, pass + 1};
        }
        mergedPairs += passMerges.load(std::memory_order_relaxed);
        if (diagnostics)
        {
            diagnostics->CompletedPasses = pass + 1;
            diagnostics->EmittedPaths = uint64_t(pass + 1) * options.LightPathsPerPass;
            diagnostics->MergedPairs = mergedPairs;
            diagnostics->CameraSplats = cameraSplats;
            diagnostics->LastRadius = coupling.MergeRadius;
        }
    }
    if (context.Stopped())
        return false;
    statistics.CompletedPixels.fetch_add(pixels, std::memory_order_relaxed);
    return true;
}
catch (const std::bad_alloc&)
{
    error = "CPU VCM could not allocate its path cache, spatial index or film.";
    return false;
}
} // namespace Arvo::RayTracing::Cpu
