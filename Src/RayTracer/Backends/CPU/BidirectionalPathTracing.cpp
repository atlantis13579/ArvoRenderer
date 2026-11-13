#include "RayTracer/Backends/CPU/BidirectionalPathTracing.h"
#include "RayTracer/Backends/CPU/BidirectionalMis.h"
#include "RayTracer/Backends/CPU/SubsurfaceScattering.h"
#include <algorithm>
#include <cmath>

namespace Arvo::RayTracing::Cpu
{
// Veach (1997), "Robust Monte Carlo Methods for Light Transport Simulation",
// chapter 10; PBRT 3e, "Bidirectional Path Tracing", Generate*Subpath/RandomWalk:
// https://pbr-book.org/3ed-2018/Light_Transport_III_Bidirectional_Methods/Bidirectional_Path_Tracing
// Camera and light walks use radiance and importance transport respectively.
// Throughput arrives BEFORE scattering; direction PDFs become destination-area
// PDFs only after intersection. A sampled delta stays a probability mass.
// Infinite lights instead project their launch disk onto the first surface.
// Interior collisions extend the path measure to volume: free-flight survival
// multiplies each interior edge, and the extinction density appears when its
// destination is a collision. Connections use the same Beer transmittance.
namespace
{
constexpr uint32_t CameraWalkDimension = 0x43414d50u;      // "CAMP"
constexpr uint32_t LightWalkDimension = 0x4c495450u;       // "LITP"
constexpr uint32_t SourceDimension = 0x53524345u;          // "SRCE"
constexpr uint32_t SourcePositionDimension = 0x53504f53u;  // "SPOS"
constexpr uint32_t SourceDirectionDimension = 0x53444952u; // "SDIR"
constexpr uint32_t ConnectionLensDimension = 0x434c454eu;  // "CLEN"
constexpr uint32_t InteriorFlightDimension = 0x49464c54u; // "IFLT"
constexpr uint32_t InteriorPhaseDimension = 0x49504853u;  // "IPHS"
constexpr uint32_t PathBatchSize = 128;

bool Finite(Float3 value)
{
    return std::isfinite(value.X) && std::isfinite(value.Y) && std::isfinite(value.Z);
}
bool Positive(Float3 value)
{
    return value.X > 0 || value.Y > 0 || value.Z > 0;
}
bool Valid(PathDensity value)
{
    return value.Measure != PathMeasure::Zero && std::isfinite(value.Log2Value);
}
bool Valid(BdptWalkOptions options)
{
    return options.MaximumVertices >= 1 && options.MaximumVertices <= MaximumBidirectionalVertices;
}
Float3 Offset(const BdptVertex& vertex, Float3 direction)
{
    if (vertex.Type == BdptVertexType::Volume) return vertex.Position;
    const float side = Dot(vertex.Hit.GeometricNormal, direction) >= 0 ? 1.0f : -1.0f;
    return vertex.Position + vertex.Hit.GeometricNormal * (side * RayEpsilon);
}
SurfaceHit Facing(const BdptVertex& vertex, Float3 view)
{
    SurfaceHit hit = vertex.Hit;
    if (Dot(hit.Normal, view) < 0)
        hit.Normal = -hit.Normal;
    hit.FrontFace = Dot(hit.GeometricNormal, view) > 0;
    return hit;
}
PathDensity DirectionDensity(float value, uint32_t event)
{
    return MakePathDensity(value, event & ScatteringDelta ? PathMeasure::Dirac : PathMeasure::SolidAngle);
}
uint32_t SegmentInterior(const Scene& scene, const BdptVertex& source, const BdptVertex& target)
{
    // Validated SSS solids are closed, outward facing and separated from other
    // geometry. The boundary side and adjacent volume vertices identify the
    // unique material traversed by a straight path-space edge.
    if (source.Type == BdptVertexType::Volume) return source.Hit.MaterialIndex;
    if (target.Type == BdptVertexType::Volume) return target.Hit.MaterialIndex;
    const Float3 displacement = target.Position - source.Position;
    if (source.Type == BdptVertexType::Surface &&
        scene.Materials[source.Hit.MaterialIndex].SubsurfaceMeanFreePath > 0 &&
        Dot(source.Hit.GeometricNormal, displacement) < 0)
        return source.Hit.MaterialIndex;
    if (target.Type == BdptVertexType::Surface &&
        scene.Materials[target.Hit.MaterialIndex].SubsurfaceMeanFreePath > 0 &&
        Dot(target.Hit.GeometricNormal, -displacement) < 0)
        return target.Hit.MaterialIndex;
    return InvalidRayPrimitiveIndex;
}
float SegmentTransmittance(const Scene& scene, const BdptVertex& source, const BdptVertex& target)
{
    const uint32_t interior = SegmentInterior(scene, source, target);
    if (interior == InvalidRayPrimitiveIndex) return 1;
    const float distance = std::sqrt(Dot(target.Position - source.Position,
                                         target.Position - source.Position));
    return std::exp(-distance / scene.Materials[interior].SubsurfaceMeanFreePath);
}
PathDensity ConvertToVertex(const Scene& scene, PathDensity pdf, const BdptVertex& source,
                            const BdptVertex& target)
{
    const PathDestination destination = target.Type == BdptVertexType::Infinite
        ? PathDestination::Infinite : (target.Type == BdptVertexType::Volume
            ? PathDestination::Volume : PathDestination::Surface);
    PathDensity converted = ConvertPathDensity(pdf, target.Position - source.Position,
                                               target.Hit.GeometricNormal, destination);
    if (converted.Measure == PathMeasure::Zero || destination == PathDestination::Infinite)
        return converted;
    const uint32_t interior = SegmentInterior(scene, source, target);
    if (interior == InvalidRayPrimitiveIndex)
        return destination == PathDestination::Volume ? PathDensity{} : converted;
    const float meanFreePath = scene.Materials[interior].SubsurfaceMeanFreePath;
    const float distance = std::sqrt(Dot(target.Position - source.Position,
                                         target.Position - source.Position));
    converted.Log2Value -= distance / meanFreePath * std::log2(std::exp(1.0f));
    if (destination == PathDestination::Volume)
        converted.Log2Value -= std::log2(meanFreePath);
    return std::isfinite(converted.Log2Value) ? converted : PathDensity{};
}

uint32_t FindEmitter(const PreparedEmissionScene& emitters, uint32_t primitive, bool environment = false)
{
    for (uint32_t i = 0; i < emitters.Sources.size(); ++i)
    {
        const auto& source = emitters.Sources[i];
        if (environment ? source.Kind == EmitterKind::Environment
                        : (primitive != InvalidRayPrimitiveIndex && source.Primitive == primitive))
            return i;
    }
    return InvalidEmitterIndex;
}
bool IntersectRectangle(const PackedEmitter& source, const Ray& ray, float& distance)
{
    if (source.Kind != EmitterKind::Rectangle)
        return false;
    const float denominator = Dot(source.Direction, ray.Direction);
    if (denominator == 0)
        return false;
    const float t = Dot(source.Direction, source.Origin - ray.Origin) / denominator;
    if (!(t > ray.MinDistance && t < ray.MaxDistance && t <= distance))
        return false;
    const Float3 helper =
        std::abs(source.Direction.Y) < BasisParallelThreshold ? Float3(0, 1, 0) : Float3(1, 0, 0);
    const Float3 tangent = Normalize(Cross(helper, source.Direction)),
                 bitangent = Cross(source.Direction, tangent);
    const Float3 relative = ray.Origin + ray.Direction * t - source.Origin;
    if (std::abs(Dot(relative, tangent)) > source.Width / 2 ||
        std::abs(Dot(relative, bitangent)) > source.Height / 2)
        return false;
    distance = t;
    return true;
}
Float3 EmitterValue(const PackedEmitter& source, Float3 outgoing)
{
    if (source.Kind == EmitterKind::Environment)
    {
        const float fraction = (1 - outgoing.Y) / 2;
        return source.BackgroundBottom * (1 - fraction) + source.BackgroundTop * fraction;
    }
    if (source.Kind == EmitterKind::Rectangle)
        return Dot(source.Direction, outgoing) > 0 ? source.Strength / (source.Width * source.Height)
                                                   : Float3();
    return source.Strength;
}

void Walk(const Scene& scene, const SceneAccelerator& accelerator, Ray ray, Float3 throughput,
          PathDensity directionPdf, const SampleContext& samples, BdptWalkOptions options,
          AtomicStatistics& statistics, const CancellationProbe& cancellation, BdptSubpath& path,
          const PreparedEmissionScene* emitters)
{
    const uint32_t stream = path.Mode == TransportMode::Radiance ? CameraWalkDimension : LightWalkDimension;
    uint32_t interiorMaterial = InvalidRayPrimitiveIndex;
    while (path.Count < options.MaximumVertices)
    {
        if (cancellation && cancellation())
        {
            path.Termination = BdptTermination::Cancelled;
            return;
        }
        if (!Finite(throughput) || !Valid(directionPdf))
        {
            path.Termination = BdptTermination::NumericalFailure;
            return;
        }
        if (!Positive(throughput))
        {
            path.Termination = BdptTermination::Absorbed;
            return;
        }
        if (path.Mode == TransportMode::Radiance && path.Count == 1)
            statistics.PrimaryRays.fetch_add(1, std::memory_order_relaxed);
        else
            statistics.ReflectionRays.fetch_add(1, std::memory_order_relaxed);
        SurfaceHit hit;
        uint64_t tests = 0;
        const bool found = accelerator.Intersect(ray, hit, tests);
        statistics.PrimitiveTests.fetch_add(tests, std::memory_order_relaxed);
        BdptVertex& vertex = path.Vertices[path.Count];
        vertex.ToPrevious = -ray.Direction;
        vertex.Throughput = throughput;
        uint32_t rectangle = InvalidEmitterIndex;
        float rectangleDistance = found ? hit.Distance + RayEpsilon : ray.MaxDistance;
        if (emitters)
            for (uint32_t i = 0; i < emitters->Sources.size(); ++i)
                if (IntersectRectangle(emitters->Sources[i], ray, rectangleDistance))
                    rectangle = i;
        if (path.Count == 1 && found && !hit.FrontFace &&
            scene.Materials[hit.MaterialIndex].SubsurfaceMeanFreePath > 0)
            interiorMaterial = hit.MaterialIndex;
        if (interiorMaterial != InvalidRayPrimitiveIndex)
        {
            const Material& interior = scene.Materials[interiorMaterial];
            const float endpoint = rectangle != InvalidEmitterIndex ? rectangleDistance :
                (found ? hit.Distance : ray.MaxDistance);
            const uint32_t dimension = stream ^ (path.Count * BounceSeedMultiplier);
            const auto flight = SampleSubsurfaceFlight(interior.SubsurfaceMeanFreePath,
                interior.Albedo * interior.SubsurfaceAlbedoScale, ray.Direction, endpoint,
                Sample2D(samples, dimension ^ InteriorFlightDimension).X);
            if (flight.Scattered)
            {
                vertex.Type = BdptVertexType::Volume;
                vertex.Position = ray.Origin + ray.Direction * flight.Distance;
                vertex.Hit.MaterialIndex = interiorMaterial;
                vertex.Throughput = Multiply(throughput, flight.Weight);
                const BdptVertex& previous = path.Vertices[path.Count - 1];
                vertex.ArrivingDensity = ConvertToVertex(scene, directionPdf, previous, vertex);
                if (!Finite(vertex.Position) || !Finite(vertex.Throughput) ||
                    !Valid(vertex.ArrivingDensity))
                {
                    path.Termination = BdptTermination::NumericalFailure;
                    return;
                }
                ++path.Count;
                if (path.Count == options.MaximumVertices) break;
                const auto phase = SampleSubsurfaceDirection(ray.Direction, interior.SubsurfaceAnisotropy,
                    Sample2D(samples, dimension ^ InteriorPhaseDimension), options.ImportanceSampling);
                if (!(phase.Pdf > 0))
                {
                    path.Termination = BdptTermination::Absorbed;
                    return;
                }
                vertex.Continuation = {phase.Direction, Float3(phase.Weight, phase.Weight, phase.Weight),
                                       phase.Pdf, phase.Pdf, 1, ScatteringGlossy};
                vertex.ReturningDensity = ConvertToVertex(scene,
                    MakePathDensity(phase.Pdf, PathMeasure::SolidAngle), vertex, previous);
                throughput = vertex.Throughput * phase.Weight;
                directionPdf = MakePathDensity(phase.Pdf, PathMeasure::SolidAngle);
                ray = {vertex.Position, phase.Direction, RayEpsilon, MaximumRayDistance};
                continue;
            }
        }
        if (rectangle != InvalidEmitterIndex)
        {
            // Analytic area lights are also opaque intersectable endpoints.
            // Prefer this identity over a coincident SampledEmitter proxy mesh.
            vertex.Type = BdptVertexType::Emitter;
            vertex.Position = ray.Origin + ray.Direction * rectangleDistance;
            vertex.Hit.GeometricNormal = emitters->Sources[rectangle].Direction;
            vertex.EmitterIndex = rectangle;
            vertex.ArrivingDensity =
                path.Count == 1 && (path.Light.Emission.Flags & EmissionInfinite)
                    ? ProjectedDiskDensity(
                          MakePathDensity(path.Light.Emission.PositionPdf, PathMeasure::Area),
                          vertex.Hit.GeometricNormal, ray.Direction)
                    : ConvertToVertex(scene, directionPdf, path.Vertices[path.Count - 1], vertex);
            ++path.Count;
            path.Termination = BdptTermination::Absorbed;
            return;
        }
        if (!found)
        {
            // No invented finite position at infinity. ToPrevious stores the
            // direction toward the last finite vertex; density stays angular.
            vertex.Type = BdptVertexType::Infinite;
            vertex.ArrivingDensity = directionPdf;
            if (emitters)
                vertex.EmitterIndex = FindEmitter(*emitters, InvalidRayPrimitiveIndex, true);
            ++path.Count;
            path.Termination = BdptTermination::Escaped;
            return;
        }
        vertex.Type = BdptVertexType::Surface;
        vertex.Position = ray.Origin + ray.Direction * hit.Distance;
        vertex.Hit = hit;
        if (emitters)
            vertex.EmitterIndex = FindEmitter(*emitters, hit.PrimitiveIndex);
        const Material& material = scene.Materials[hit.MaterialIndex];
        vertex.SurfaceColor =
            hit.HasTexCoord
                ? Multiply(hit.Color, SampleDiffuseTexture(scene, material.DiffuseTextureIndex, hit.TexCoord))
                : hit.Color;
        const BdptVertex& previous = path.Vertices[path.Count - 1];
        if (path.Count == 1 && (path.Light.Emission.Flags & EmissionInfinite))
            vertex.ArrivingDensity =
                ProjectedDiskDensity(MakePathDensity(path.Light.Emission.PositionPdf, PathMeasure::Area),
                                     hit.GeometricNormal, ray.Direction);
        else
            vertex.ArrivingDensity = ConvertToVertex(scene, directionPdf, previous, vertex);
        if (!Finite(vertex.Position) || !Valid(vertex.ArrivingDensity))
        {
            path.Termination = BdptTermination::NumericalFailure;
            return;
        }
        ++path.Count;
        if (path.Count == options.MaximumVertices)
            break;
        const uint32_t dimension = stream ^ (path.Count * BounceSeedMultiplier);
        vertex.Continuation = SampleBidirectionalScattering(
            scene, vertex, Sample2D(samples, dimension),
            Sample2D(samples, dimension ^ LobeSelectionScramble).X, path.Mode, options.ImportanceSampling);
        const SurfaceSample& sample = vertex.Continuation;
        if (sample.Event == ScatteringNone || !(sample.Pdf > 0))
        {
            path.Termination = BdptTermination::Absorbed;
            return;
        }
        const PathDensity reverse = DirectionDensity(sample.ReversePdf, sample.Event);
        // A distant source's preceding endpoint is angular, even though its
        // launch record has a finite auxiliary disk position.
        vertex.ReturningDensity = path.Count == 2 && (path.Light.Emission.Flags & EmissionInfinite)
                                      ? reverse
                                      : ConvertToVertex(scene, reverse, vertex, previous);
        throughput = Multiply(throughput, sample.Weight);
        directionPdf = DirectionDensity(sample.Pdf, sample.Event);
        ray = {Offset(vertex, sample.Direction), sample.Direction, RayEpsilon, MaximumRayDistance};
        if (material.SubsurfaceMeanFreePath > 0 && (sample.Event & ScatteringTransmission))
            interiorMaterial = hit.FrontFace ? hit.MaterialIndex : InvalidRayPrimitiveIndex;
    }
    path.Termination = BdptTermination::VertexLimit;
}
} // namespace

SurfaceEvaluation EvaluateBidirectionalScattering(const Scene& scene, const BdptVertex& vertex, Float3 view,
                                                  Float3 outgoing, TransportMode mode, bool importance)
{
    if (vertex.Type == BdptVertexType::Volume)
    {
        const float anisotropy = scene.Materials[vertex.Hit.MaterialIndex].SubsurfaceAnisotropy;
        const float phase = SubsurfacePhase(Dot(-view, outgoing), anisotropy);
        const float pdf = importance ? phase : 1 / (4 * Pi);
        return {Float3(phase, phase, phase), pdf, pdf};
    }
    if (vertex.Type != BdptVertexType::Surface)
        return {};
    Material material = scene.Materials[vertex.Hit.MaterialIndex];
    const bool legacy = material.Model == MaterialModel::Legacy;
    const float diffuseFraction = legacy ? 1 - std::clamp(material.Reflectivity, 0.0f, 1.0f) : 1;
    if (legacy)
        material.Model = MaterialModel::Diffuse;
    const SurfaceBsdf bsdf(material, Facing(vertex, view), view, vertex.SurfaceColor, mode);
    auto result = bsdf.Evaluate(outgoing, importance);
    result.Value = result.Value * diffuseFraction;
    result.Pdf *= diffuseFraction;
    result.ReversePdf *= diffuseFraction;
    return result;
}
SurfaceSample SampleBidirectionalScattering(const Scene& scene, const BdptVertex& vertex, Float2 variates,
                                            float lobeVariate, TransportMode mode, bool importance)
{
    if (vertex.Type == BdptVertexType::Volume)
    {
        const auto phase = SampleSubsurfaceDirection(-vertex.ToPrevious,
            scene.Materials[vertex.Hit.MaterialIndex].SubsurfaceAnisotropy, variates, importance);
        return {phase.Direction, Float3(phase.Weight, phase.Weight, phase.Weight), phase.Pdf,
                phase.Pdf, 1, ScatteringGlossy};
    }
    if (vertex.Type != BdptVertexType::Surface)
        return {};
    Material material = scene.Materials[vertex.Hit.MaterialIndex];
    const auto hit = Facing(vertex, vertex.ToPrevious);
    const bool legacy = material.Model == MaterialModel::Legacy;
    const float mirrorFraction = legacy ? std::clamp(material.Reflectivity, 0.0f, 1.0f) : 0;
    if (legacy && lobeVariate < mirrorFraction)
    {
        const Float3 direction = Normalize(Reflect(-vertex.ToPrevious, hit.Normal));
        const float viewGeometry = Dot(vertex.ToPrevious, hit.GeometricNormal);
        const float lightGeometry = Dot(direction, hit.GeometricNormal);
        if (!(viewGeometry * lightGeometry > 0) || Dot(vertex.ToPrevious, hit.Normal) <= 0)
            return {};
        float correction = 1;
        if (mode == TransportMode::Importance)
        {
            const float denominator = std::abs(viewGeometry * Dot(direction, hit.Normal));
            if (!(denominator > 0))
                return {};
            correction = std::abs(lightGeometry * Dot(vertex.ToPrevious, hit.Normal)) / denominator;
        }
        return {direction, Float3(correction, correction, correction), mirrorFraction, mirrorFraction,
                1,         ScatteringReflection | ScatteringDelta};
    }
    if (legacy)
        material.Model = MaterialModel::Diffuse;
    const SurfaceBsdf bsdf(material, hit, vertex.ToPrevious, vertex.SurfaceColor, mode);
    auto result = bsdf.Sample(variates, lobeVariate, importance);
    result.Pdf *= 1 - mirrorFraction;
    result.ReversePdf *= 1 - mirrorFraction;
    return result;
}

BdptSubpath GenerateCameraSubpath(const Scene& scene, const SceneAccelerator& accelerator,
                                  const CameraFrame& camera, Float2 raster, const SampleContext& samples,
                                  BdptWalkOptions options, AtomicStatistics& statistics,
                                  const CancellationProbe& cancellation,
                                  const PreparedEmissionScene* emitters)
{
    BdptSubpath path;
    if (!Valid(options))
        return path;
    path.VertexLimit = options.MaximumVertices;
    if (cancellation && cancellation())
    {
        path.Termination = BdptTermination::Cancelled;
        return path;
    }
    path.Camera = SampleCameraEndpoint(camera, raster, Sample2D(samples, LensSampleDimension));
    if (!(path.Camera.Flags & CameraEndpointValid))
    {
        path.Termination = BdptTermination::NullEndpoint;
        return path;
    }
    auto& root = path.Vertices[path.Count++];
    root.Type = BdptVertexType::Camera;
    root.Position = path.Camera.Origin;
    root.Hit.GeometricNormal = camera.Forward;
    root.Throughput = Float3(1, 1, 1) * path.Camera.Weight;
    root.ArrivingDensity =
        MakePathDensity(path.Camera.PositionPdf,
                        path.Camera.Flags & CameraDeltaPosition ? PathMeasure::Dirac : PathMeasure::Area);
    Walk(scene, accelerator, {path.Camera.Origin, path.Camera.Direction}, root.Throughput,
         MakePathDensity(path.Camera.DirectionPdf, PathMeasure::SolidAngle), samples, options, statistics,
         cancellation, path, emitters);
    return path;
}
BdptSubpath GenerateLightSubpath(const Scene& scene, const SceneAccelerator& accelerator,
                                 const PreparedEmissionScene& emitters, const SampleContext& samples,
                                 BdptWalkOptions options, AtomicStatistics& statistics,
                                 const CancellationProbe& cancellation)
{
    BdptSubpath path;
    path.Mode = TransportMode::Importance;
    if (!Valid(options))
        return path;
    path.VertexLimit = options.MaximumVertices;
    if (cancellation && cancellation())
    {
        path.Termination = BdptTermination::Cancelled;
        return path;
    }
    path.Light = SampleEmitterEndpoint(emitters, Sample2D(samples, SourceDimension).X,
                                       Sample2D(samples, SourcePositionDimension),
                                       Sample2D(samples, SourceDirectionDimension));
    const auto& emission = path.Light.Emission;
    if (!(emission.Flags & EmissionValid))
    {
        path.Termination = BdptTermination::NullEndpoint;
        return path;
    }
    auto& root = path.Vertices[path.Count++];
    root.Type = BdptVertexType::Emitter;
    root.Position = emission.Origin;
    root.Hit.GeometricNormal = emission.Normal;
    root.EmitterIndex = path.Light.EmitterIndex;
    root.Hit.PrimitiveIndex = path.Light.Primitive;
    if (emission.Flags & EmissionInfinite)
    {
        root.ArrivingDensity = MakePathDensity(emission.DirectionPdf, emission.Flags & EmissionDeltaDirection
                                                                          ? PathMeasure::Dirac
                                                                          : PathMeasure::SolidAngle);
        root.Throughput = emission.Value / (path.Light.SelectionProbability * emission.DirectionPdf);
    }
    else
    {
        root.ArrivingDensity =
            MakePathDensity(emission.PositionPdf,
                            emission.Flags & EmissionDeltaPosition ? PathMeasure::Dirac : PathMeasure::Area);
        root.Throughput = emission.Value / (path.Light.SelectionProbability * emission.PositionPdf);
    }
    const Float3 origin =
        emission.Flags & EmissionDeltaPosition ? emission.Origin : Offset(root, emission.Direction);
    Walk(scene, accelerator, {origin, emission.Direction}, emission.Weight,
         MakePathDensity(emission.DirectionPdf, emission.Flags & EmissionDeltaDirection
                                                    ? PathMeasure::Dirac
                                                    : PathMeasure::SolidAngle),
         samples, options, statistics, cancellation, path, &emitters);
    return path;
}
namespace
{
// Rebuild the probability of EVERY cut on the completed path. This intentionally
// favors an explicit O(n) derivation over patching a few cached ratios: discarded
// continuation events at the two connection vertices must not mask valid cuts.
float ConnectionMis(const Scene& scene, const PreparedEmissionScene& emitters, const BdptSubpath& light,
                    uint32_t s, const BdptSubpath& camera, uint32_t t, const BdptVertex& cameraRoot,
                    const CameraEndpointEvaluation& cameraPdf, bool importance, MisHeuristic heuristic)
{
    const uint32_t count = s + t;
    // Directly visible emission is sampled only by the camera (s=0,t=2).
    // There is no light-to-lens strategy with zero scattering vertices.
    if (count == 2)
        return s == 0 ? 1.0f : 0.0f;
    std::array<const BdptVertex*, MaximumBidirectionalVertices> vertex{};
    for (uint32_t i = 0; i < s; ++i)
        vertex[i] = &light.Vertices[i];
    for (uint32_t i = 0; i < t; ++i)
        vertex[s + i] = &camera.Vertices[t - 1 - i];
    vertex[count - 1] = &cameraRoot;
    const bool infinite = (light.Light.Emission.Flags & EmissionInfinite) != 0;
    std::array<BidirectionalVertexPdf, MaximumBidirectionalVertices> pdf{};
    auto direction = [&](uint32_t from, uint32_t to) {
        if (to == 0 && infinite)
            return -light.Light.Emission.Direction;
        return Normalize(vertex[to]->Position - vertex[from]->Position);
    };
    auto convert = [&](PathDensity density, uint32_t from, uint32_t to) {
        if (to == 0 && infinite) return density;
        return ConvertToVertex(scene, density, *vertex[from], *vertex[to]);
    };
    for (uint32_t i = 1; i + 1 < count; ++i)
    {
        float forward = 0, reverse = 0;
        uint32_t event = ScatteringNone;
        if (i == s - 1 || i == s)
        {
            const auto evaluation =
                EvaluateBidirectionalScattering(scene, *vertex[i], direction(i, i - 1), direction(i, i + 1),
                                                TransportMode::Importance, importance);
            forward = evaluation.Pdf;
            reverse = evaluation.ReversePdf;
        }
        else
        {
            const auto& sample = vertex[i]->Continuation;
            forward = i < s ? sample.Pdf : sample.ReversePdf;
            reverse = i < s ? sample.ReversePdf : sample.Pdf;
            event = sample.Event;
        }
        if (event & ScatteringDelta)
            pdf[i].Flags = PathDeltaScattering;
        pdf[i + 1].Forward = convert(DirectionDensity(forward, event), i, i + 1);
        pdf[i - 1].Reverse = convert(DirectionDensity(reverse, event), i, i - 1);
    }
    const Float3 emittedDirection = infinite ? light.Light.Emission.Direction : direction(0, 1);
    const auto emission = EmitterEndpointPdf(emitters, light.Light.EmitterIndex,
                                             vertex[0]->Hit.GeometricNormal, emittedDirection);
    const PathDensity positionPdf =
        MakePathDensity(emission.Flags & EmissionDeltaPosition ? 1 : emission.Position,
                        emission.Flags & EmissionDeltaPosition ? PathMeasure::Dirac : PathMeasure::Area);
    const PathDensity emissionDirection = MakePathDensity(
        emission.Flags & EmissionDeltaDirection ? 1 : emission.Direction,
        emission.Flags & EmissionDeltaDirection ? PathMeasure::Dirac : PathMeasure::SolidAngle);
    pdf[1].Forward = infinite
                         ? ProjectedDiskDensity(positionPdf, vertex[1]->Hit.GeometricNormal, emittedDirection)
                         : convert(emissionDirection, 0, 1);
    if (emission.Flags & (EmissionDeltaPosition | EmissionDeltaDirection))
        pdf[0].Flags = PathEndpointNotIntersectable;
    pdf[count - 2].Reverse =
        convert(MakePathDensity(cameraPdf.DirectionPdf, PathMeasure::SolidAngle), count - 1, count - 2);
    if (cameraPdf.Flags & CameraDeltaPosition)
        pdf[count - 1].Flags = PathEndpointNotIntersectable;
    std::array<BidirectionalStrategy, MaximumBidirectionalStrategies> strategies{};
    for (uint32_t cut = 0; cut < count; ++cut)
    {
        auto& strategy = strategies[cut];
        strategy.Enabled = cut <= light.VertexLimit && count - cut <= camera.VertexLimit;
        strategy.LightEndpoint =
            cut == 0 ? MakePathDensity(1, PathMeasure::Dirac) : (infinite ? emissionDirection : positionPdf);
        strategy.LightSelection = cut == 0 ? 1 : emission.SelectionProbability;
        strategy.CameraEndpoint =
            MakePathDensity(cameraPdf.Flags & CameraDeltaPosition ? 1 : cameraPdf.PositionPdf,
                            cameraPdf.Flags & CameraDeltaPosition ? PathMeasure::Dirac : PathMeasure::Area);
    }
    strategies[count].Enabled = 0; // The camera is explicitly sampled, never intersected.
    return BidirectionalMisWeights(pdf.data(), count, strategies.data(), heuristic).Values[s];
}
bool OccludedScene(const SceneAccelerator& accelerator, const PreparedEmissionScene& emitters, const Ray& ray,
                   AtomicStatistics& statistics)
{
    uint64_t tests = 0;
    bool blocked = accelerator.Occluded(ray, tests);
    if (!blocked)
        for (const auto& source : emitters.Sources)
            if (source.Kind == EmitterKind::Rectangle)
            {
                ++tests;
                float distance = ray.MaxDistance;
                if (IntersectRectangle(source, ray, distance))
                {
                    blocked = true;
                    break;
                }
            }
    statistics.PrimitiveTests.fetch_add(tests, std::memory_order_relaxed);
    return blocked;
}
bool VisibleConnection(const SceneAccelerator& accelerator, const PreparedEmissionScene& emitters,
                       Float3 origin, Float3 target, AtomicStatistics& statistics)
{
    const Float3 segment = target - origin;
    const float length = std::sqrt(Dot(segment, segment));
    if (!(length > 2 * RayEpsilon) || !std::isfinite(length))
        return false;
    statistics.ShadowRays.fetch_add(1, std::memory_order_relaxed);
    return !OccludedScene(accelerator, emitters, {origin, segment / length, RayEpsilon, length - RayEpsilon},
                          statistics);
}
} // namespace

BdptConnection ConnectBidirectionalSubpaths(const Scene& scene, const SceneAccelerator& accelerator,
                                            const PreparedEmissionScene& emitters, const CameraFrame& frame,
                                            const BdptSubpath& light, uint32_t s, const BdptSubpath& camera,
                                            uint32_t t, Float2 lensVariates, bool importance,
                                            MisHeuristic heuristic, AtomicStatistics& statistics,
                                            const CancellationProbe& cancellation)
{
    BdptConnection result;
    if (light.Count > MaximumBidirectionalVertices || camera.Count > MaximumBidirectionalVertices || s < 2 ||
        t < 1 || s > light.Count || t > camera.Count || s > MaximumBidirectionalVertices - t ||
        light.VertexLimit < s || camera.VertexLimit < t || light.Mode != TransportMode::Importance ||
        camera.Mode != TransportMode::Radiance || light.Light.EmitterIndex >= emitters.Sources.size() ||
        (cancellation && cancellation()))
        return result;
    if (light.Termination == BdptTermination::Cancelled || camera.Termination == BdptTermination::Cancelled ||
        light.Termination == BdptTermination::NumericalFailure ||
        camera.Termination == BdptTermination::NumericalFailure)
        return result;
    const auto& l = light.Vertices[s - 1];
    const auto& c = camera.Vertices[t - 1];
    if ((l.Type != BdptVertexType::Surface && l.Type != BdptVertexType::Volume) ||
        (t > 1 && c.Type != BdptVertexType::Surface && c.Type != BdptVertexType::Volume))
        return result;
    BdptVertex cameraRoot = camera.Vertices[0];
    CameraEndpointEvaluation cameraPdf;
    Float3 value, lightDirection, cameraTarget;
    result.Splat = t == 1;
    if (t == 1)
    {
        const auto connection = SampleCameraConnection(frame, l.Position, lensVariates);
        if (!(connection.Flags & CameraEndpointValid))
            return result;
        lightDirection = connection.Direction;
        const auto bsdf = EvaluateBidirectionalScattering(scene, l, l.ToPrevious, lightDirection,
                                                          TransportMode::Importance, importance);
        const float lightCosine = l.Type == BdptVertexType::Volume ? 1 :
            std::abs(Dot(l.Hit.Normal, lightDirection));
        cameraRoot.Position = cameraTarget = connection.Position;
        cameraRoot.Hit.GeometricNormal = frame.Forward;
        value = Multiply(l.Throughput, bsdf.Value) *
                (lightCosine * connection.Weight * SegmentTransmittance(scene, l, cameraRoot));
        cameraPdf = EvaluateCameraEndpoint(frame, connection.LensPoint, -lightDirection);
        result.Raster = connection.Raster;
    }
    else
    {
        const Float3 segment = c.Position - l.Position;
        const float squaredDistance = Dot(segment, segment);
        if (!(squaredDistance > 0) || !std::isfinite(squaredDistance))
            return result;
        lightDirection = segment / std::sqrt(squaredDistance);
        const auto fLight = EvaluateBidirectionalScattering(scene, l, l.ToPrevious, lightDirection,
                                                            TransportMode::Importance, importance);
        const auto fCamera = EvaluateBidirectionalScattering(scene, c, c.ToPrevious, -lightDirection,
                                                             TransportMode::Radiance, importance);
        const float lightCosine = l.Type == BdptVertexType::Volume ? 1 :
            std::abs(Dot(l.Hit.Normal, lightDirection));
        const float cameraCosine = c.Type == BdptVertexType::Volume ? 1 :
            std::abs(Dot(c.Hit.Normal, lightDirection));
        const float geometry = lightCosine * cameraCosine / squaredDistance;
        value =
            Multiply(Multiply(l.Throughput, fLight.Value), Multiply(c.Throughput, fCamera.Value)) *
            (geometry * SegmentTransmittance(scene, l, c));
        cameraTarget = Offset(c, -lightDirection);
        const auto& endpoint = camera.Camera;
        cameraPdf = {endpoint.Raster, endpoint.Importance, endpoint.PositionPdf, endpoint.DirectionPdf,
                     endpoint.Flags};
        result.Raster = endpoint.Raster;
    }
    if (!Finite(value) || !Positive(value) || !(cameraPdf.Flags & CameraEndpointValid) ||
        !VisibleConnection(accelerator, emitters, Offset(l, lightDirection), cameraTarget, statistics))
        return result;
    result.MisWeight =
        ConnectionMis(scene, emitters, light, s, camera, t, cameraRoot, cameraPdf, importance, heuristic);
    result.Radiance = value * result.MisWeight;
    return result;
}
BdptConnection ConnectBidirectionalEmitter(const Scene& scene, const SceneAccelerator& accelerator,
                                           const PreparedEmissionScene& emitters, const BdptSubpath& light,
                                           const BdptSubpath& camera, uint32_t t, bool importance,
                                           MisHeuristic heuristic, AtomicStatistics& statistics,
                                           const CancellationProbe& cancellation)
{
    BdptConnection result;
    if (light.Count == 0 || t < 2 || t > camera.Count || t >= MaximumBidirectionalVertices ||
        light.Light.EmitterIndex >= emitters.Sources.size() || (cancellation && cancellation()))
        return result;
    const auto& c = camera.Vertices[t - 1];
    if (c.Type != BdptVertexType::Surface && c.Type != BdptVertexType::Volume)
        return result;
    const auto& emission = light.Light.Emission;
    const auto& source = emitters.Sources[light.Light.EmitterIndex];
    const bool infinite = (emission.Flags & EmissionInfinite) != 0;
    Float3 direction, value;
    float geometry = 1;
    if (infinite)
        direction = -emission.Direction;
    else
    {
        const Float3 segment = light.Vertices[0].Position - c.Position;
        const float squaredDistance = Dot(segment, segment);
        if (!(squaredDistance > 0) || !std::isfinite(squaredDistance))
            return result;
        direction = segment / std::sqrt(squaredDistance);
        geometry = (emission.Flags & EmissionDeltaPosition
                        ? 1
                        : std::abs(Dot(light.Vertices[0].Hit.GeometricNormal, direction))) /
                   squaredDistance;
    }
    const auto bsdf = EvaluateBidirectionalScattering(scene, c, c.ToPrevious, direction,
                                                      TransportMode::Radiance, importance);
    const float density = infinite ? emission.DirectionPdf : emission.PositionPdf;
    value =
        Multiply(Multiply(c.Throughput, bsdf.Value), EmitterValue(source, -direction)) *
        ((c.Type == BdptVertexType::Volume ? 1 : std::abs(Dot(c.Hit.Normal, direction))) * geometry /
         (light.Light.SelectionProbability * density) *
         (infinite ? 1 : SegmentTransmittance(scene, c, light.Vertices[0])));
    if (!Finite(value) || !Positive(value))
        return result;
    if (infinite)
    {
        statistics.ShadowRays.fetch_add(1, std::memory_order_relaxed);
        if (OccludedScene(accelerator, emitters,
                          {Offset(c, direction), direction, RayEpsilon, MaximumRayDistance}, statistics))
            return result;
    }
    else if (!VisibleConnection(accelerator, emitters, Offset(c, direction),
                                emission.Flags & EmissionDeltaPosition
                                    ? light.Vertices[0].Position
                                    : Offset(light.Vertices[0], -direction),
                                statistics))
        return result;
    const auto& endpoint = camera.Camera;
    const CameraEndpointEvaluation cameraPdf{endpoint.Raster, endpoint.Importance, endpoint.PositionPdf,
                                             endpoint.DirectionPdf, endpoint.Flags};
    result.MisWeight = ConnectionMis(scene, emitters, light, 1, camera, t, camera.Vertices[0], cameraPdf,
                                     importance, heuristic);
    result.Raster = endpoint.Raster;
    result.Radiance = value * result.MisWeight;
    return result;
}

BdptConnection EvaluateBidirectionalEmitterHit(const Scene& scene, const PreparedEmissionScene& emitters,
                                               const BdptSubpath& camera, uint32_t t, uint32_t lightLimit,
                                               bool importance, MisHeuristic heuristic)
{
    BdptConnection result;
    if (t < 2 || t > camera.Count || t > MaximumBidirectionalVertices)
        return result;
    const auto& hit = camera.Vertices[t - 1];
    if (hit.EmitterIndex >= emitters.Sources.size())
        return result;
    const auto& source = emitters.Sources[hit.EmitterIndex];
    const Float3 value = Multiply(hit.Throughput, EmitterValue(source, hit.ToPrevious));
    if (!Finite(value) || !Positive(value))
        return result;
    // Only endpoint metadata is needed for the s=0 comparison. The completed
    // path itself is the reversed camera prefix, including its emitter vertex.
    BdptSubpath endpoint;
    endpoint.Mode = TransportMode::Importance;
    endpoint.Count = 1;
    endpoint.VertexLimit = lightLimit;
    endpoint.Vertices[0] = hit;
    endpoint.Light.EmitterIndex = hit.EmitterIndex;
    const auto density =
        EmitterEndpointPdf(emitters, hit.EmitterIndex, hit.Hit.GeometricNormal, hit.ToPrevious);
    endpoint.Light.Emission.Flags = density.Flags;
    endpoint.Light.Emission.Direction = hit.ToPrevious;
    const auto& c = camera.Camera;
    const CameraEndpointEvaluation cameraPdf{c.Raster, c.Importance, c.PositionPdf, c.DirectionPdf, c.Flags};
    result.MisWeight = ConnectionMis(scene, emitters, endpoint, 0, camera, t, camera.Vertices[0], cameraPdf,
                                     importance, heuristic);
    result.Raster = c.Raster;
    result.Radiance = value * result.MisWeight;
    return result;
}
bool RenderBidirectionalFrame(const Scene& scene, const SceneAccelerator& accelerator,
                              const CameraFrame& camera, const RenderSettings& settings, LinearFilm& film,
                              AtomicStatistics& statistics, const CancellationProbe& cancellation,
                              const BdptDispatcher& dispatch, std::string& error)
{
    if (settings.MaxReflectionBounces > MaximumBidirectionalReflections || settings.SamplesPerPixel == 0 ||
        settings.Adaptive.Enabled || settings.Transport.RussianRoulette ||
        !settings.Transport.NextEventEstimation)
    {
        error =
            "CPU BDPT currently requires fixed samples/depth, NEE, no roulette, and at most 32 reflections.";
        return false;
    }
    if (!settings.ShadowsEnabled || settings.ShadowStrength != 1)
    {
        error = "BDPT requires full physical visibility for every connection strategy.";
        return false;
    }
    const PreparedEmissionScene emitters = PrepareEmitters(scene);
    for (const auto& source : emitters.Sources)
        if (!(source.Flags & EmitterCastsShadows))
        {
            error = "BDPT requires shadow casting on every emitter.";
            return false;
        }
    const uint32_t maximumVertices = settings.MaxReflectionBounces + 1 + 2; // First surface + two endpoints.
    RenderSettings full = settings;
    full.Width = FullImageWidth(settings);
    full.Height = FullImageHeight(settings);
    full.TileX = full.TileY = full.FullWidth = full.FullHeight = 0;
    const uint32_t pathCount = full.Width * full.Height;
    if (film.Width != settings.Width || film.Height != settings.Height ||
        film.Pixels.size() != size_t(settings.Width) * settings.Height || !dispatch)
    {
        error = "BDPT film/dispatcher does not match the requested output.";
        return false;
    }
    struct Splat
    {
        Float3 Value;
        Float2 Raster;
    };
    struct PathContributions
    {
        Float3 CameraValue;
        uint32_t CameraPixel = 0, SplatCount = 0;
        bool HasCamera = false, Failed = false;
        std::array<Splat, MaximumBidirectionalVertices> Splats{};
    };
    std::vector<PathContributions> batch(PathBatchSize);
    std::vector<Float3> pass(film.Pixels.size());
    const auto inside = [&](uint32_t x, uint32_t y) {
        return x >= settings.TileX && y >= settings.TileY && x - settings.TileX < settings.Width &&
               y - settings.TileY < settings.Height;
    };
    for (uint32_t sample = 0; sample < settings.SamplesPerPixel; ++sample)
    {
        std::fill(pass.begin(), pass.end(), Float3());
        for (uint32_t first = 0; first < pathCount; first += PathBatchSize)
        {
            if (cancellation && cancellation())
                return false;
            const uint32_t count = std::min(PathBatchSize, pathCount - first);
            std::fill(batch.begin(), batch.begin() + count, PathContributions{});
            dispatch(count, [&](uint32_t slot) {
                const uint32_t index = first + slot, x = index % full.Width, y = index / full.Width;
                const auto samples = MakeSampleContext(full, x, y, sample);
                const Float2 jitter = Sample2D(samples, 0);
                const Float2 raster = CameraPixelRaster(x, y, full.Width, full.Height, jitter);
                auto& output = batch[slot];
                output.HasCamera = inside(x, y);
                output.CameraPixel =
                    output.HasCamera ? (y - settings.TileY) * settings.Width + x - settings.TileX : 0;
                const auto light = GenerateLightSubpath(
                    scene, accelerator, emitters, samples,
                    {maximumVertices, settings.Transport.ImportanceSampling}, statistics, cancellation);
                auto eye = GenerateCameraSubpath(
                    scene, accelerator, camera, raster, samples,
                    {output.HasCamera ? maximumVertices : 1, settings.Transport.ImportanceSampling},
                    statistics, cancellation, &emitters);
                // A cropped-out camera ray need not be traced; alternative
                // camera strategies still exist at every retained splat pixel.
                eye.VertexLimit = maximumVertices;
                output.Failed = light.Termination == BdptTermination::NumericalFailure ||
                                eye.Termination == BdptTermination::NumericalFailure;
                if (output.Failed || (cancellation && cancellation()))
                    return;
                if (output.HasCamera)
                    for (uint32_t t = 2; t <= eye.Count; ++t)
                    {
                        output.CameraValue =
                            output.CameraValue +
                            EvaluateBidirectionalEmitterHit(scene, emitters, eye, t, maximumVertices,
                                                            settings.Transport.ImportanceSampling,
                                                            settings.Transport.Mis)
                                .Radiance;
                        if (t + 1 <= maximumVertices)
                            output.CameraValue =
                                output.CameraValue +
                                ConnectBidirectionalEmitter(scene, accelerator, emitters, light, eye, t,
                                                            settings.Transport.ImportanceSampling,
                                                            settings.Transport.Mis, statistics, cancellation)
                                    .Radiance;
                    }
                for (uint32_t s = 2; s <= light.Count && s + 1 <= maximumVertices; ++s)
                {
                    const Float2 lens =
                        Sample2D(samples, ConnectionLensDimension ^ (s * BounceSeedMultiplier));
                    const auto connection =
                        ConnectBidirectionalSubpaths(scene, accelerator, emitters, camera, light, s, eye, 1,
                                                     lens, settings.Transport.ImportanceSampling,
                                                     settings.Transport.Mis, statistics, cancellation);
                    if (Positive(connection.Radiance))
                        output.Splats[output.SplatCount++] = {connection.Radiance, connection.Raster};
                    if (output.HasCamera)
                        for (uint32_t t = 2; t <= eye.Count && s + t <= maximumVertices; ++t)
                            output.CameraValue =
                                output.CameraValue + ConnectBidirectionalSubpaths(
                                                         scene, accelerator, emitters, camera, light, s, eye,
                                                         t, {}, settings.Transport.ImportanceSampling,
                                                         settings.Transport.Mis, statistics, cancellation)
                                                         .Radiance;
                }
            });
            if (cancellation && cancellation())
                return false;
            // Fixed global-path order avoids float atomics and makes crop and
            // worker-count changes bit-identical, without storing a whole pass's paths.
            for (uint32_t slot = 0; slot < count; ++slot)
            {
                const auto& output = batch[slot];
                if (output.Failed)
                {
                    error = "BDPT encountered an unrepresentable path density or throughput.";
                    return false;
                }
                if (output.HasCamera)
                    pass[output.CameraPixel] = pass[output.CameraPixel] + output.CameraValue;
                for (uint32_t i = 0; i < output.SplatCount; ++i)
                {
                    const auto& splat = output.Splats[i];
                    const uint32_t x = std::min(uint32_t(splat.Raster.X * full.Width), full.Width - 1);
                    const uint32_t y = std::min(uint32_t(splat.Raster.Y * full.Height), full.Height - 1);
                    if (inside(x, y))
                    {
                        auto& pixel = pass[size_t(y - settings.TileY) * settings.Width + x - settings.TileX];
                        pixel = pixel + splat.Value;
                    }
                }
            }
        }
        // One independent pass is the sample, including every incoming splat.
        // N image-wide light paths per pass cancel the normalized film-cell area;
        // never divide these splats by N a second time.
        for (const auto& value : pass)
            if (!Finite(value))
            {
                error = "BDPT film overflow.";
                return false;
            }
        for (size_t pixel = 0; pixel < pass.size(); ++pixel)
            AccumulateFilmSample(film.Pixels[pixel], pass[pixel]);
    }
    statistics.CompletedPixels.fetch_add(film.Pixels.size(), std::memory_order_relaxed);
    return true;
}
} // namespace Arvo::RayTracing::Cpu
