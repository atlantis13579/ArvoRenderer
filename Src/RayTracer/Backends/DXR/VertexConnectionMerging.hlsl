// Independent DXR vertex connection and merging for offline rendering.
// Preview runs short camera paths; neither VCM path construction nor MIS calls another
// integrator. Georgiev et al. (2012), "Light Transport Simulation with Vertex
// Connection and Merging"; Veach (1997), chapters 5 and 10.
// Fixed-depth walks retain null draws, geometric Jacobians and delta branch
// masses. Connections later re-evaluate only the directions they change.
#include "VertexConnectionMergingMis.hlsl"
#include "SubsurfaceScattering.hlsl"

static const uint VcmSensor = 0, VcmEmitter = 1, VcmSurface = 2, VcmEscape = 3, VcmVolume = 5;
static const uint VcmInvalid = 0xffffffffu;
static const uint VcmEndInvalid = 0, VcmEndNull = 1, VcmEndLimit = 2;
static const uint VcmEndEscape = 3, VcmEndAbsorbed = 4, VcmEndNumeric = 5, VcmEndEmitter = 6;
static const uint VcmCameraStage = 0, VcmLightStage = 1, VcmCouplingStage = 2;
static const uint VcmInitializeStage = 3, VcmGridStage = 4, VcmSortStage = 5, VcmEstimateStage = 6;
static const uint VcmMergeStage = 7, VcmSplatStage = 8, VcmCommitStage = 9;
static const uint VcmArithmeticError = 1, VcmControlsError = 2;
static const uint VcmLensDimension = 0x4c454e53u, VcmSourceDimension = 0x534f5552u;
static const uint VcmPositionDimension = 0x504f534eu, VcmDirectionDimension = 0x44495245u;
static const uint VcmScatterDimension = 0x53434154u, VcmLobeDimension = 0x4c4f4245u;
static const uint VcmInteriorFlightDimension = 0x5643464cu, VcmInteriorPhaseDimension = 0x56435048u;
static const uint VcmVertexScramble = 0x9e3779b9u, VcmLightStream = 0x56434d4cu;
static const float Pi = 3.14159265358979323846f;
static const float RayEpsilon = 1e-4f, MaximumRayDistance = 1e30f, BasisParallelThreshold = 0.999f;
static const uint ImportanceBit = 1;

struct VcmPathVertex
{
    float3 Position;
    uint Kind;
    float3 Incoming;
    uint Material;
    float3 ShadingNormal;
    uint FrontFace;
    float3 GeometricNormal;
    uint Emitter;
    float3 Color;
    uint Primitive;
    float3 Throughput;
    float ForwardPdf;
    float3 Outgoing;
    float ReversePdf;
    uint Event;
    float RelativeIndex;
    uint Reserved0, Reserved1;
    VcmDensity Arrival, PreviousArrival;
};
struct VcmPath
{
    uint First, Count, Limit, Mode;
    float3 LaunchDirection;
    float DirectionPdf;
    float3 LaunchWeight;
    float PositionPdf;
    float3 EndpointValue;
    float SourceProbability;
    float3 EndpointNormal;
    uint SourceIndex;
    float2 Raster, LensPoint;
    uint EndpointFlags, End, Reserved0, Reserved1;
};
struct VcmWalkStatistics
{
    uint2 Primary, Shadow, Reflection, Completed;
};
struct VcmWalkResult
{
    VcmPath Path;
    VcmWalkStatistics Statistics;
};
struct VcmArrival
{
    float3 Position, ShadingNormal, GeometricNormal, Color;
    uint Kind, Material, Primitive, Emitter, FrontFace, Interior;
};
struct VcmScattering
{
    float3 Direction, Weight;
    float Pdf, ReversePdf, RelativeIndex;
    uint Event;
};

#include "Surface.hlsl"
#include "MaterialTypes.hlsl"
#include "Globals.hlsl"
#include "InstancePrograms.hlsli"
#include "Triangle.hlsl"
#include "Sphere.hlsl"
#include "TriangleIntersection.hlsl"
#include "MotionTriangle.hlsl"
#include "Select.hlsl"
#include "BsdfTypes.hlsl"
#include "Fresnel.hlsl"
#include "Ggx.hlsl"
#include "Diffuse.hlsl"
#include "Conductor.hlsl"
#include "Dielectric.hlsl"
#include "SurfaceScattering.hlsl"
static const uint PixelSeedMultiplier = 9781, SampleSeedMultiplier = 6271;
static const uint SecondDimensionScramble = 0x68bc21ebu;
uint Hash(uint word)
{
    static const uint FirstMultiplier = 0x7feb352du, SecondMultiplier = 0x846ca68bu;
    static const uint HalfShift = 16, MiddleShift = 15;
    word ^= word >> HalfShift;
    word *= FirstMultiplier;
    word ^= word >> MiddleShift;
    word *= SecondMultiplier;
    return word ^ (word >> HalfShift);
}
float Random01(uint word)
{
    static const uint DiscardBits = 8;
    static const float FractionRange = 16777216.0f;
    return float(Hash(word) >> DiscardBits) / FractionRange;
}
#include "Sequence.hlsl"
#include "LightSelection.hlsl"
#include "LightEmission.hlsl"
#include "SurfaceEmission.hlsl"
#include "EmitterEndpoint.hlsl"
#include "CameraEndpoint.hlsl"

// Scene access and scattering are defined separately from path state transitions.
VcmArrival VcmIntersect(float3 origin, float3 direction);
VcmScattering VcmScatter(VcmPathVertex vertex, uint mode, float2 variates, float lobe);
void VcmStoreVertex(uint index, VcmPathVertex vertex);
VcmPathVertex VcmLoadVertex(uint index);
void VcmReportError(uint flag);
float VcmMeanFreePath(uint material);
float3 VcmInteriorAlbedo(uint material);
float VcmAnisotropy(uint material);
bool VcmImportanceSampling();
double VcmSquaredDistance(float3 from, float3 to);

bool VcmSpectrum(float3 value)
{
    return isfinite(value.x) && isfinite(value.y) && isfinite(value.z) && value.x >= 0 && value.y >= 0 &&
           value.z >= 0;
}
bool VcmVector(float3 value)
{
    return isfinite(value.x) && isfinite(value.y) && isfinite(value.z);
}
VcmPathVertex VcmEmptyVertex()
{
    VcmPathVertex vertex;
    vertex.Position = vertex.Incoming = vertex.ShadingNormal = vertex.GeometricNormal = vertex.Color =
        vertex.Throughput = vertex.Outgoing = float3(0, 0, 0);
    vertex.Kind = VcmEscape;
    vertex.Material = vertex.Primitive = vertex.Emitter = VcmInvalid;
    vertex.FrontFace = vertex.Event = vertex.Reserved1 = 0;
    vertex.Reserved0 = VcmInvalid;
    vertex.ForwardPdf = vertex.ReversePdf = 0;
    vertex.RelativeIndex = 1;
    vertex.Arrival = vertex.PreviousArrival = VcmLogDensity(0, 0);
    return vertex;
}
VcmPath VcmEmptyPath(uint first, uint limit, uint mode)
{
    VcmPath path;
    path.First = first;
    path.Count = 0;
    path.Limit = limit;
    path.Mode = mode;
    path.LaunchDirection = path.LaunchWeight = path.EndpointValue = path.EndpointNormal = float3(0, 0, 0);
    path.DirectionPdf = path.PositionPdf = path.SourceProbability = 0;
    path.SourceIndex = VcmInvalid;
    path.Raster = path.LensPoint = float2(0, 0);
    path.EndpointFlags = path.Reserved0 = path.Reserved1 = 0;
    path.End = limit > 0 && limit <= VcmMaximumVertices ? VcmEndNull : VcmEndInvalid;
    return path;
}
VcmWalkResult VcmEmptyWalk(VcmPath path)
{
    VcmWalkResult result;
    result.Path = path;
    result.Statistics.Primary = result.Statistics.Shadow = result.Statistics.Reflection =
        result.Statistics.Completed = uint2(0, 0);
    return result;
}
float3 VcmOffset(VcmPathVertex vertex, float3 direction)
{
    if (vertex.Kind == VcmVolume) return vertex.Position;
    float side = dot(vertex.GeometricNormal, direction) >= 0 ? 1.0f : -1.0f;
    return vertex.Position + vertex.GeometricNormal * (RayEpsilon * side);
}
uint VcmSegmentInterior(VcmPathVertex source, VcmPathVertex target)
{
    if (source.Kind == VcmVolume) return source.Material;
    if (target.Kind == VcmVolume) return target.Material;
    if (source.Kind == VcmEmitter && source.Reserved0 != VcmInvalid && VcmMeanFreePath(source.Reserved0) > 0)
        return source.Reserved0;
    if (target.Kind == VcmEmitter && target.Reserved0 != VcmInvalid && VcmMeanFreePath(target.Reserved0) > 0)
        return target.Reserved0;
    const float3 edge = target.Position - source.Position;
    if (source.Kind == VcmSurface && VcmMeanFreePath(source.Material) > 0 &&
        dot(source.GeometricNormal, edge) < 0)
        return source.Material;
    if (target.Kind == VcmSurface && VcmMeanFreePath(target.Material) > 0 &&
        dot(target.GeometricNormal, -edge) < 0)
        return target.Material;
    return VcmInvalid;
}
double VcmSegmentLogTransmittance(VcmPathVertex source, VcmPathVertex target)
{
    const uint material = VcmSegmentInterior(source, target);
    return material == VcmInvalid ? 0 :
        -double(length(target.Position - source.Position)) /
        double(VcmMeanFreePath(material)) * VcmInverseLn2;
}
VcmDensity VcmEdgeArrival(VcmDensity angular, VcmPathVertex source, VcmPathVertex target)
{
    if (!VcmHasDensity(angular)) return VcmLogDensity(0, 0);
    const float3 edge = target.Position - source.Position;
    const double squared = VcmSquaredDistance(source.Position, target.Position);
    if (!(squared > 0) || !VcmFinite(squared)) return VcmLogDensity(0, 0);
    VcmDensity density = VcmLogDensity(0, 0);
    if (target.Kind == VcmVolume)
        density = VcmLogDensity(angular.Log2Value - VcmLog2Positive(squared), 1);
    else
        density = VcmSurfaceArrival(angular, edge, target.GeometricNormal);
    if (!VcmHasDensity(density)) return density;
    const uint material = VcmSegmentInterior(source, target);
    if (material == VcmInvalid)
    {
        if (target.Kind == VcmVolume) return VcmLogDensity(0, 0);
        return density;
    }
    const double meanFreePath = double(VcmMeanFreePath(material));
    density.Log2Value += VcmSegmentLogTransmittance(source, target);
    if (target.Kind == VcmVolume)
        density.Log2Value -= VcmLog2Positive(meanFreePath);
    return VcmLogDensity(density.Log2Value, VcmFinite(density.Log2Value) ? 1 : 0);
}
VcmWalkResult VcmContinueWalk(VcmPath path, SampleContext context, float3 origin, float3 direction,
                              float3 throughput)
{
    VcmWalkResult result = VcmEmptyWalk(path);
    uint interior = VcmInvalid;
    while (result.Path.Count < result.Path.Limit)
    {
        uint index = result.Path.Count;
        VcmPathVertex previous = VcmLoadVertex(result.Path.First + index - 1);
        if (result.Path.Mode == TransportRadiance && index == 1)
            ++result.Statistics.Primary.x;
        else
            ++result.Statistics.Reflection.x;
        VcmArrival arrival = VcmIntersect(origin, direction);
        VcmPathVertex vertex = VcmEmptyVertex();
        vertex.Kind = arrival.Kind;
        vertex.Reserved0 = arrival.Interior;
        vertex.Incoming = -direction; // Actual ray, never reconstructed from offset endpoints.
        vertex.Throughput = throughput;
        vertex.Emitter = arrival.Emitter;
        if (index == 1 && arrival.Interior != VcmInvalid)
            interior = arrival.Interior;
        if (interior != VcmInvalid)
        {
            const float endpoint = arrival.Kind == VcmEscape ? MaximumRayDistance :
                length(arrival.Position - origin);
            const float variate = Sample2D(context, VcmInteriorFlightDimension ^
                (index * VcmVertexScramble)).x;
            const SubsurfaceFlight flight = SubsurfaceFreeFlight(VcmMeanFreePath(interior),
                VcmInteriorAlbedo(interior), direction, endpoint, variate);
            if (flight.Scattered != 0)
            {
                vertex.Kind = VcmVolume;
                vertex.Position = origin + direction * flight.Distance;
                vertex.Material = interior;
                vertex.Emitter = vertex.Primitive = VcmInvalid;
                vertex.ShadingNormal = vertex.GeometricNormal = float3(0, 0, 0);
            }
        }
        if (arrival.Kind == VcmEscape && vertex.Kind != VcmVolume)
        {
            vertex.Position = previous.Position;
            vertex.Arrival = VcmDensityFromValue(previous.ForwardPdf);
            VcmStoreVertex(result.Path.First + index, vertex);
            ++result.Path.Count;
            result.Path.End = VcmEndEscape;
            return result;
        }
        if (vertex.Kind != VcmVolume)
        {
            vertex.Position = arrival.Position;
            vertex.ShadingNormal = arrival.ShadingNormal;
            vertex.GeometricNormal = arrival.GeometricNormal;
            vertex.Color = arrival.Color;
            vertex.Material = arrival.Material;
            vertex.Primitive = arrival.Primitive;
            vertex.FrontFace = arrival.FrontFace;
        }
        bool projected = index == 1 && result.Path.Mode == TransportImportance &&
                         (result.Path.EndpointFlags & EmissionInfinite) != 0;
        if (projected)
            vertex.Arrival = VcmDiskArrival(VcmDensityFromValue(result.Path.PositionPdf),
                                            vertex.GeometricNormal, direction);
        else
            vertex.Arrival = VcmEdgeArrival(VcmDensityFromValue(previous.ForwardPdf), previous, vertex);
        ++result.Path.Count;
        VcmStoreVertex(result.Path.First + index, vertex);
        if (!VcmVector(vertex.Position) || !VcmVector(vertex.GeometricNormal) ||
            !VcmVector(vertex.ShadingNormal))
        {
            result.Path.End = VcmEndNumeric;
            VcmReportError(VcmArithmeticError);
            return result;
        }
        if (!VcmHasDensity(vertex.Arrival))
        {
            result.Path.End = VcmEndAbsorbed;
            return result;
        }
        if (vertex.Kind == VcmEmitter)
        {
            result.Path.End = VcmEndEmitter;
            return result;
        }
        if (result.Path.Count == result.Path.Limit)
            break;
        uint bounce = index * VcmVertexScramble;
        float2 scatteringVariates = Sample2D(context, (vertex.Kind == VcmVolume ?
            VcmInteriorPhaseDimension : VcmScatterDimension) ^ bounce);
        float2 lobeVariates = Sample2D(context, VcmLobeDimension ^ bounce);
        VcmScattering scattering;
        if (vertex.Kind == VcmVolume)
        {
            const SubsurfaceDirection phase = SubsurfaceSampleDirection(-vertex.Incoming,
                VcmAnisotropy(vertex.Material), scatteringVariates, VcmImportanceSampling());
            scattering.Direction = phase.Direction;
            scattering.Weight = VcmInteriorAlbedo(vertex.Material) * phase.Weight;
            scattering.Pdf = scattering.ReversePdf = phase.Pdf;
            scattering.RelativeIndex = 1;
            scattering.Event = ScatteringGlossy;
        }
        else
            scattering = VcmScatter(vertex, result.Path.Mode, scatteringVariates, lobeVariates.x);
        if (scattering.Event == 0 || scattering.Pdf == 0)
        {
            result.Path.End = VcmEndAbsorbed;
            return result;
        }
        if (!(scattering.Pdf > 0) || !isfinite(scattering.Pdf) || !(scattering.ReversePdf >= 0) ||
            !isfinite(scattering.ReversePdf) || !VcmVector(scattering.Direction) ||
            !(dot(scattering.Direction, scattering.Direction) > 0) || !VcmSpectrum(scattering.Weight) ||
            !(scattering.RelativeIndex > 0) || !isfinite(scattering.RelativeIndex))
        {
            result.Path.End = VcmEndNumeric;
            VcmReportError(VcmArithmeticError);
            return result;
        }
        vertex.ForwardPdf = scattering.Pdf;
        vertex.ReversePdf = scattering.ReversePdf;
        vertex.Outgoing = scattering.Direction;
        vertex.Event = scattering.Event;
        vertex.RelativeIndex = scattering.RelativeIndex;
        VcmDensity reverse = VcmDensityFromValue(scattering.ReversePdf);
        if (projected)
            vertex.PreviousArrival = reverse; // Returning to an infinite endpoint is angular.
        else
            vertex.PreviousArrival = VcmEdgeArrival(reverse, vertex, previous);
        VcmStoreVertex(result.Path.First + index, vertex);
        throughput = throughput * scattering.Weight;
        if (!VcmSpectrum(throughput))
        {
            result.Path.End = VcmEndNumeric;
            VcmReportError(VcmArithmeticError);
            return result;
        }
        if (throughput.x == 0 && throughput.y == 0 && throughput.z == 0)
        {
            result.Path.End = VcmEndAbsorbed;
            return result;
        }
        direction = scattering.Direction;
        origin = VcmOffset(vertex, direction);
        if (vertex.Kind == VcmSurface && VcmMeanFreePath(vertex.Material) > 0 &&
            (scattering.Event & ScatteringTransmission) != 0)
            interior = vertex.FrontFace != 0 ? vertex.Material : VcmInvalid;
    }
    result.Path.End = VcmEndLimit;
    return result;
}
VcmWalkResult VcmCameraWalk(CameraEndpointFrame frame, float2 raster, float2 lensVariates,
                            SampleContext context, uint first, uint limit)
{
    VcmPath path = VcmEmptyPath(first, limit, TransportRadiance);
    if (path.End == VcmEndInvalid)
        return VcmEmptyWalk(path);
    CameraEndpointSample camera = SampleCameraEndpoint(frame, raster, lensVariates);
    if ((camera.Flags & CameraEndpointValid) == 0)
        return VcmEmptyWalk(path);
    if (!VcmVector(camera.Origin) || !VcmVector(camera.Direction) || !(camera.Weight >= 0) ||
        !isfinite(camera.Weight))
    {
        path.End = VcmEndNumeric;
        VcmReportError(VcmArithmeticError);
        return VcmEmptyWalk(path);
    }
    path.Count = 1;
    path.LaunchDirection = camera.Direction;
    path.LaunchWeight = float3(camera.Weight, camera.Weight, camera.Weight);
    path.PositionPdf = camera.PositionPdf;
    path.DirectionPdf = camera.DirectionPdf;
    path.EndpointValue = float3(camera.Importance, camera.Importance, camera.Importance);
    path.EndpointNormal = frame.Forward;
    path.SourceProbability = 1;
    path.EndpointFlags = camera.Flags;
    path.LensPoint = camera.LensPoint;
    path.Raster = camera.Raster;
    VcmPathVertex root = VcmEmptyVertex();
    root.Kind = VcmSensor;
    root.Position = camera.Origin;
    root.GeometricNormal = root.ShadingNormal = frame.Forward;
    root.Throughput = path.LaunchWeight;
    root.ForwardPdf = camera.DirectionPdf;
    root.Outgoing = camera.Direction;
    root.Arrival = VcmDensityFromValue(camera.PositionPdf);
    VcmStoreVertex(first, root);
    return VcmContinueWalk(path, context, camera.Origin, camera.Direction, path.LaunchWeight);
}
VcmWalkResult VcmLightWalk(EmitterEndpoint endpoint, SampleContext context, uint first, uint limit)
{
    VcmPath path = VcmEmptyPath(first, limit, TransportImportance);
    path.SourceIndex = endpoint.EmitterIndex;
    path.SourceProbability = endpoint.SelectionProbability;
    if (path.End == VcmEndInvalid || (endpoint.Emission.Flags & EmissionValid) == 0)
        return VcmEmptyWalk(path);
    EmissionSample emission = endpoint.Emission;
    if (!(endpoint.SelectionProbability > 0 && endpoint.SelectionProbability <= 1) ||
        !(emission.PositionPdf > 0) || !(emission.DirectionPdf > 0) || !isfinite(emission.PositionPdf) ||
        !isfinite(emission.DirectionPdf) || !VcmSpectrum(emission.Weight) || !VcmSpectrum(emission.Value) ||
        !VcmVector(emission.Origin) || !VcmVector(emission.Direction) ||
        !(dot(emission.Direction, emission.Direction) > 0))
    {
        path.End = VcmEndNumeric;
        VcmReportError(VcmArithmeticError);
        return VcmEmptyWalk(path);
    }
    path.Count = 1;
    path.LaunchDirection = emission.Direction;
    path.LaunchWeight = emission.Weight; // Endpoint sampler already divides by source PMF.
    path.PositionPdf = emission.PositionPdf;
    path.DirectionPdf = emission.DirectionPdf;
    path.EndpointValue = emission.Value;
    path.EndpointNormal = emission.Normal;
    path.EndpointFlags = emission.Flags;
    VcmPathVertex root = VcmEmptyVertex();
    root.Kind = VcmEmitter;
    root.Position = emission.Origin;
    root.Emitter = endpoint.EmitterIndex;
    root.Primitive = endpoint.Primitive;
    root.ShadingNormal = root.GeometricNormal = emission.Normal;
    root.Outgoing = emission.Direction;
    root.ForwardPdf = emission.DirectionPdf;
    // The canonical infinite endpoint is a direction. Its projected disk PDF
    // belongs to the first surface arrival, not to the endpoint itself.
    float endpointPdf =
        (emission.Flags & EmissionInfinite) != 0 ? emission.DirectionPdf : emission.PositionPdf;
    root.Throughput =
        float3(float(double(emission.Value.x) / double(endpoint.SelectionProbability) / double(endpointPdf)),
               float(double(emission.Value.y) / double(endpoint.SelectionProbability) / double(endpointPdf)),
               float(double(emission.Value.z) / double(endpoint.SelectionProbability) / double(endpointPdf)));
    root.Arrival = VcmDensityFromValue(endpointPdf);
    if (!VcmSpectrum(root.Throughput))
    {
        path.Count = 0;
        path.End = VcmEndNumeric;
        VcmReportError(VcmArithmeticError);
        return VcmEmptyWalk(path);
    }
    VcmStoreVertex(first, root);
    float3 origin = (emission.Flags & (EmissionDeltaPosition | EmissionInfinite)) != 0
                        ? emission.Origin
                        : VcmOffset(root, emission.Direction);
    return VcmContinueWalk(path, context, origin, emission.Direction, emission.Weight);
}

struct VcmEvaluation
{
    float3 Value;
    float Pdf, ReversePdf;
};
VcmEvaluation VcmVolumeEvaluation(VcmPathVertex vertex, float3 incoming, float3 outgoing)
{
    VcmEvaluation value;
    const float phase = SubsurfacePhase(dot(-incoming, outgoing), VcmAnisotropy(vertex.Material));
    value.Value = VcmInteriorAlbedo(vertex.Material) * phase;
    value.Pdf = value.ReversePdf = VcmImportanceSampling() ? phase : 1 / (4 * Pi);
    return value;
}
struct VcmCouplingOptions
{
    double Radius, MergingSamples, LightTracingSamples;
    uint MaximumVertices, Heuristic, NextEventEstimation;
};
struct VcmContribution
{
    float3 Radiance;
    uint Flags;
    double MisWeight;
    float2 Raster;
    uint ShadowRays, Reserved0, Reserved1, Reserved2;
};
static const uint VcmContributionValid = 1, VcmContributionSplat = 2, VcmContributionNumeric = 4;
static const float VcmMergeNormalAgreement = 0.9f;
static const double VcmMergePlaneFraction = 0.1L;
VcmEvaluation VcmEvaluate(VcmPathVertex vertex, float3 incoming, float3 outgoing, uint mode);
EmitterEndpointDensity VcmSourceDensity(uint emitter, float3 normal, float3 direction);
float3 VcmSourceValue(uint emitter, float3 direction);
bool VcmVisible(VcmPathVertex from, VcmPathVertex to, float3 direction, bool infinite);

// A merge has two original positions. Forward edges end at the photon-side
// position; reverse edges end at the camera-side position. The scattering
// function at their common vertex belongs to the camera surface.
struct VcmJoin
{
    VcmPath Light, Eye;
    VcmPathVertex Source, Sensor;
    uint S, T, Merge, Count;
    float3 EmissionDirection, CameraDirection;
    float2 Lens;
    EmitterEndpointDensity SourcePdf;
    CameraEndpointEvaluation CameraPdf;
};
VcmPathVertex VcmJoinedVertex(VcmJoin join, uint index, bool photonSide)
{
    if (index == 0)
        return join.Source;
    if (index + 1 == join.Count)
        return join.Sensor;
    if (index < join.S && (join.Merge == 0 || index + 1 < join.S || photonSide))
        return VcmLoadVertex(join.Light.First + index);
    return VcmLoadVertex(join.Eye.First + join.Count - 1 - index);
}
double VcmSquaredDistance(float3 from, float3 to)
{
    double x = VcmWidenFloat(to.x) - VcmWidenFloat(from.x);
    double y = VcmWidenFloat(to.y) - VcmWidenFloat(from.y);
    double z = VcmWidenFloat(to.z) - VcmWidenFloat(from.z);
    return x * x + y * y + z * z;
}
float3 VcmDirection(float3 from, float3 to)
{
    double x = VcmWidenFloat(to.x) - VcmWidenFloat(from.x);
    double y = VcmWidenFloat(to.y) - VcmWidenFloat(from.y);
    double z = VcmWidenFloat(to.z) - VcmWidenFloat(from.z);
    double ax = x < 0 ? -x : x, ay = y < 0 ? -y : y, az = z < 0 ? -z : z;
    double largest = ax > ay ? ax : ay;
    largest = largest > az ? largest : az;
    if (!(largest > 0) || !VcmFinite(largest))
        return float3(0, 0, 0);
    return normalize(float3(float(x / largest), float(y / largest), float(z / largest)));
}
VcmDensity VcmEdgeDensity(VcmJoin join, uint from, uint to, float angular)
{
    VcmDensity density = VcmDensityFromValue(angular);
    if (to == 0 && (join.SourcePdf.Flags & EmissionInfinite) != 0)
        return density;
    bool forward = to > from;
    VcmPathVertex start = VcmJoinedVertex(join, from, !forward);
    VcmPathVertex end = VcmJoinedVertex(join, to, forward);
    return VcmEdgeArrival(density, start, end);
}
VcmStrategy VcmCouplingStrategy(VcmDensity source, VcmDensity camera, double selection, double samples,
                                bool enabled)
{
    VcmStrategy strategy;
    strategy.LightEndpoint = source;
    strategy.CameraEndpoint = camera;
    strategy.LightSelection = selection;
    strategy.SampleCount = samples;
    strategy.Enabled = enabled ? 1 : 0;
    strategy.Reserved0 = strategy.Reserved1 = strategy.Reserved2 = 0;
    return strategy;
}
VcmMisEvaluation VcmCouplingMis(VcmJoin join, VcmCouplingOptions options)
{
    VcmVertexPdf vertices[VcmMaximumVertices];
    VcmStrategy connections[VcmMaximumStrategies], mergings[VcmMaximumStrategies];
    VcmDensity unit = VcmDensityFromValue(1);
    bool infinite = (join.SourcePdf.Flags & EmissionInfinite) != 0;
    float sourcePosition = (join.SourcePdf.Flags & EmissionDeltaPosition) != 0 ? 1 : join.SourcePdf.Position;
    float sourceDirection =
        (join.SourcePdf.Flags & EmissionDeltaDirection) != 0 ? 1 : join.SourcePdf.Direction;
    VcmDensity source = VcmDensityFromValue(infinite ? sourceDirection : sourcePosition);
    VcmDensity camera = VcmDensityFromValue(
        (join.CameraPdf.Flags & CameraDeltaPosition) != 0 ? 1 : join.CameraPdf.PositionPdf);
    for (uint i = 0; i < VcmMaximumVertices; ++i)
    {
        vertices[i].Forward = vertices[i].Reverse = VcmLogDensity(0, 0);
        vertices[i].Flags = vertices[i].Reserved0 = vertices[i].Reserved1 = vertices[i].Reserved2 = 0;
    }
    if ((join.SourcePdf.Flags & (EmissionDeltaPosition | EmissionDeltaDirection)) != 0)
        vertices[0].Flags = VcmEndpointNotIntersectable;
    vertices[join.Count - 1].Flags = VcmEndpointNotIntersectable;
    if (infinite)
    {
        VcmPathVertex first = VcmJoinedVertex(join, 1, true);
        vertices[1].Forward = VcmDiskArrival(VcmDensityFromValue(join.SourcePdf.Position),
                                             first.GeometricNormal, join.EmissionDirection);
    }
    else
        vertices[1].Forward = VcmEdgeDensity(join, 0, 1, sourceDirection);
    vertices[join.Count - 2].Reverse =
        VcmEdgeDensity(join, join.Count - 1, join.Count - 2, join.CameraPdf.DirectionPdf);
    for (uint i = 1; i + 1 < join.Count; ++i)
    {
        VcmPathVertex left = VcmJoinedVertex(join, i, true), right = VcmJoinedVertex(join, i, false);
        if (right.Kind == VcmVolume)
            vertices[i].Flags |= VcmVolumeScattering;
        bool changed = join.S != 0 && (i + 1 == join.S || (join.Merge == 0 && i == join.S));
        float forward, reverse;
        if (changed)
        {
            float3 towardLight, towardCamera;
            if (i < join.S)
                towardLight = left.Incoming;
            else if (i == 1 && infinite)
                towardLight = -join.EmissionDirection;
            else
            {
                VcmPathVertex previous = VcmJoinedVertex(join, i - 1, false);
                towardLight = VcmDirection(left.Position, previous.Position);
            }
            if (i >= join.S || join.Merge != 0)
                towardCamera = right.Incoming;
            else if (i + 2 == join.Count && join.T == 1)
                towardCamera = -join.CameraDirection;
            else
            {
                VcmPathVertex next = VcmJoinedVertex(join, i + 1, true);
                towardCamera = VcmDirection(right.Position, next.Position);
            }
            VcmEvaluation evaluation = VcmEvaluate(right, towardLight, towardCamera, TransportImportance);
            forward = evaluation.Pdf;
            reverse = evaluation.ReversePdf;
        }
        else
        {
            bool fromLight = i < join.S;
            forward = fromLight ? left.ForwardPdf : right.ReversePdf;
            reverse = fromLight ? left.ReversePdf : right.ForwardPdf;
            if ((right.Event & ScatteringDelta) != 0)
                vertices[i].Flags |= VcmDeltaScattering;
        }
        vertices[i + 1].Forward = VcmEdgeDensity(join, i, i + 1, forward);
        vertices[i - 1].Reverse = VcmEdgeDensity(join, i, i - 1, reverse);
    }
    for (uint cut = 0; cut < VcmMaximumStrategies; ++cut)
    {
        bool eyeHit = cut == 0 && join.Count <= join.Eye.Limit;
        bool connect = cut > 0 && cut < join.Count && join.Count > 2 && cut <= join.Light.Limit &&
                       join.Count - cut <= join.Eye.Limit && (cut != 1 || options.NextEventEstimation != 0);
        double population = cut + 1 == join.Count ? options.LightTracingSamples : 1;
        connections[cut] =
            VcmCouplingStrategy(source, camera, join.SourcePdf.SelectionProbability, population, connect);
        if (eyeHit)
            connections[cut] = VcmCouplingStrategy(unit, camera, 1, 1, true);
        bool merge =
            cut >= 2 && cut < join.Count && cut <= join.Light.Limit && join.Count + 1 - cut <= join.Eye.Limit;
        mergings[cut] = VcmCouplingStrategy(source, camera, join.SourcePdf.SelectionProbability,
                                            options.MergingSamples, merge);
    }
    return VcmEvaluateMis(vertices, join.Count, connections, mergings, options.Radius, options.Heuristic,
                          VcmConnections | VcmMergings, join.S, join.Merge);
}
VcmContribution VcmEmptyContribution()
{
    VcmContribution result;
    result.Radiance = float3(0, 0, 0);
    result.Raster = float2(0, 0);
    result.MisWeight = 0;
    result.Flags = result.ShadowRays = result.Reserved0 = result.Reserved1 = result.Reserved2 = 0;
    return result;
}
float VcmContributionChannel(float a, float b, float c, float d, double logarithm)
{
    double x = VcmWidenFloat(a), y = VcmWidenFloat(b), z = VcmWidenFloat(c), w = VcmWidenFloat(d);
    if (!(x >= 0 && y >= 0 && z >= 0 && w >= 0) || !VcmFinite(x) || !VcmFinite(y) || !VcmFinite(z) ||
        !VcmFinite(w))
        return -1;
    if (x == 0 || y == 0 || z == 0 || w == 0)
        return 0;
    logarithm += VcmLog2Positive(x) + VcmLog2Positive(y) + VcmLog2Positive(z) + VcmLog2Positive(w);
    static const double FloatRoundingZeroExponent = -150, FloatOverflowExponent = 128;
    static const double FloatLargestFinite = 3.40282346638528859811704183484516925440e38L;
    if (logarithm < FloatRoundingZeroExponent)
        return 0;
    if (!VcmFinite(logarithm) || logarithm >= FloatOverflowExponent)
        return -1;
    int exponent = int(logarithm);
    if (double(exponent) < logarithm)
        ++exponent;
    double value = VcmPowerOfTwo(exponent) * VcmRelativeExp2(logarithm - double(exponent));
    return value <= FloatLargestFinite ? float(value) : -1;
}
bool VcmMergeSupport(VcmPathVertex photon, VcmPathVertex eye, double radius)
{
    if (!(radius > 0) || photon.Kind != eye.Kind)
        return false;
    double x = VcmWidenFloat(photon.Position.x) - VcmWidenFloat(eye.Position.x);
    double y = VcmWidenFloat(photon.Position.y) - VcmWidenFloat(eye.Position.y);
    double z = VcmWidenFloat(photon.Position.z) - VcmWidenFloat(eye.Position.z);
    if (x * x + y * y + z * z > radius * radius)
        return false;
    if (eye.Kind == VcmVolume)
        return eye.Material == photon.Material;
    if (eye.Kind != VcmSurface ||
        dot(photon.GeometricNormal, eye.GeometricNormal) < VcmMergeNormalAgreement)
        return false;
    double plane = x * eye.GeometricNormal.x + y * eye.GeometricNormal.y + z * eye.GeometricNormal.z;
    return plane * plane <= (VcmMergePlaneFraction * radius) * (VcmMergePlaneFraction * radius);
}
VcmContribution VcmCouple(VcmPath light, uint s, VcmPath eye, uint t, uint merge, CameraEndpointFrame frame,
                          float2 lensVariates, VcmCouplingOptions options)
{
    VcmContribution result = VcmEmptyContribution();
    if (s > light.Count || t == 0 || t > eye.Count || merge > 1 || eye.Mode != TransportRadiance ||
        (s > 0 && light.Mode != TransportImportance) || light.Count > VcmMaximumVertices ||
        eye.Count > VcmMaximumVertices || light.Limit > VcmMaximumVertices ||
        eye.Limit > VcmMaximumVertices || s > light.Limit || t > eye.Limit ||
        options.MaximumVertices > VcmMaximumVertices || !(options.Radius >= 0) ||
        !VcmFinite(options.Radius) || !(options.MergingSamples >= 0) || !VcmFinite(options.MergingSamples) ||
        !(options.LightTracingSamples >= 0) || !VcmFinite(options.LightTracingSamples) ||
        options.Heuristic > VcmPowerMis)
        return result;
    uint count = s + t - merge;
    if (count < 2 || count > options.MaximumVertices ||
        (merge != 0 && (s < 2 || t < 2 || options.MergingSamples == 0 || options.Radius == 0)) ||
        (s == 1 && (t == 1 || options.NextEventEstimation == 0)) || (s == 0 && merge != 0))
        return result;
    VcmJoin join;
    join.Light = light;
    join.Eye = eye;
    join.S = s;
    join.T = t;
    join.Merge = merge;
    join.Count = count;
    join.Sensor = VcmLoadVertex(eye.First);
    join.Lens = eye.LensPoint;
    VcmPathVertex cameraEnd = VcmLoadVertex(eye.First + t - 1);
    VcmPathVertex lightEnd = VcmEmptyVertex();
    if (s > 0)
    {
        join.Source = VcmLoadVertex(light.First);
        lightEnd = VcmLoadVertex(light.First + s - 1);
        if (s > 1 && lightEnd.Kind != VcmSurface && lightEnd.Kind != VcmVolume)
            return result;
        if (t > 1 && cameraEnd.Kind != VcmSurface && cameraEnd.Kind != VcmVolume)
            return result;
    }
    else
        join.Source = cameraEnd;
    if (join.Source.Emitter == VcmInvalid)
        return result;
    CameraConnectionSample projection = EmptyCameraConnection();
    if (t == 1)
    {
        if (!(options.LightTracingSamples > 0) || s < 2)
            return result;
        projection = SampleCameraConnection(frame, lightEnd.Position, lensVariates);
        if ((projection.Flags & CameraEndpointValid) == 0)
            return result;
        join.Sensor.Position = projection.Position;
        join.Sensor.GeometricNormal = join.Sensor.ShadingNormal = frame.Forward;
        join.Lens = projection.LensPoint;
        join.CameraDirection = -projection.Direction;
        result.Raster = projection.Raster;
    }
    else
    {
        VcmPathVertex firstEye = VcmLoadVertex(eye.First + 1);
        join.CameraDirection = -firstEye.Incoming;
        result.Raster = eye.Raster;
    }
    if (s == 0)
        join.EmissionDirection = cameraEnd.Incoming;
    else if (s > 1)
    {
        VcmPathVertex firstLight = VcmLoadVertex(light.First + 1);
        join.EmissionDirection = -firstLight.Incoming;
    }
    else
    {
        join.EmissionDirection = (light.EndpointFlags & EmissionInfinite) != 0
                                     ? light.LaunchDirection
                                     : VcmDirection(join.Source.Position, cameraEnd.Position);
    }
    join.SourcePdf =
        VcmSourceDensity(join.Source.Emitter, join.Source.GeometricNormal, join.EmissionDirection);
    join.CameraPdf = EvaluateCameraEndpoint(frame, join.Lens, join.CameraDirection);
    if ((join.SourcePdf.Flags & EmissionValid) == 0 || (join.CameraPdf.Flags & CameraEndpointValid) == 0)
        return result;
    bool infinite = (join.SourcePdf.Flags & EmissionInfinite) != 0;
    float3 a = float3(1, 1, 1), b = a, c = a, d = a;
    double logScale = 0;
    if (merge != 0)
    {
        if (!VcmMergeSupport(lightEnd, cameraEnd, options.Radius))
            return result;
        const bool volume = cameraEnd.Kind == VcmVolume;
        if (volume && VcmSquaredDistance(lightEnd.Position, cameraEnd.Position) >
            double(2 * RayEpsilon) * (2 * RayEpsilon) &&
            !VcmVisible(cameraEnd, lightEnd,
                VcmDirection(cameraEnd.Position, lightEnd.Position), false))
            return result;
        VcmEvaluation scattering =
            VcmEvaluate(cameraEnd, cameraEnd.Incoming, lightEnd.Incoming, TransportRadiance);
        double shading = volume ? 1 : abs(dot(cameraEnd.ShadingNormal, lightEnd.Incoming));
        double geometry = volume ? 1 : abs(dot(cameraEnd.GeometricNormal, lightEnd.Incoming));
        if (!(shading > 0 && geometry > 0))
            return result;
        a = lightEnd.Throughput;
        b = cameraEnd.Throughput;
        c = scattering.Value;
        VcmDensity measure = VcmDiskArea(options.Radius);
        if (volume) measure = VcmSphereVolume(options.Radius);
        logScale = VcmLog2Positive(shading) - VcmLog2Positive(geometry) - measure.Log2Value -
                   VcmLog2Positive(options.MergingSamples);
    }
    else if (s == 0)
    {
        a = cameraEnd.Throughput;
        b = VcmSourceValue(join.Source.Emitter, join.EmissionDirection);
    }
    else if (t == 1)
    {
        VcmEvaluation scattering =
            VcmEvaluate(lightEnd, lightEnd.Incoming, projection.Direction, TransportImportance);
        double cosine = lightEnd.Kind == VcmVolume ? 1 :
            abs(dot(lightEnd.ShadingNormal, projection.Direction));
        if (!(cosine > 0 && projection.Weight > 0))
            return result;
        a = lightEnd.Throughput;
        b = scattering.Value;
        logScale = VcmLog2Positive(cosine) + VcmLog2Positive(projection.Weight) +
            VcmSegmentLogTransmittance(lightEnd, join.Sensor);
        result.ShadowRays = 1;
        if (!VcmVisible(lightEnd, join.Sensor, projection.Direction, false))
            return result;
    }
    else if (s == 1)
    {
        float3 toLight = -join.EmissionDirection;
        VcmEvaluation scattering = VcmEvaluate(cameraEnd, cameraEnd.Incoming, toLight, TransportRadiance);
        double cosine = cameraEnd.Kind == VcmVolume ? 1 :
            abs(dot(cameraEnd.ShadingNormal, toLight));
        double sourceCosine = (join.SourcePdf.Flags & EmissionDeltaPosition) != 0 || infinite
                                  ? 1
                                  : abs(dot(join.Source.GeometricNormal, join.EmissionDirection));
        double endpoint =
            infinite ? ((join.SourcePdf.Flags & EmissionDeltaDirection) != 0 ? 1 : join.SourcePdf.Direction)
                     : ((join.SourcePdf.Flags & EmissionDeltaPosition) != 0 ? 1 : join.SourcePdf.Position);
        if (!(cosine > 0 && sourceCosine > 0 && endpoint > 0 && join.SourcePdf.SelectionProbability > 0))
            return result;
        a = cameraEnd.Throughput;
        b = scattering.Value;
        c = VcmSourceValue(join.Source.Emitter, join.EmissionDirection);
        logScale = VcmLog2Positive(cosine) + VcmLog2Positive(sourceCosine) - VcmLog2Positive(endpoint) -
                   VcmLog2Positive(join.SourcePdf.SelectionProbability);
        if (!infinite)
        {
            double distance = VcmSquaredDistance(join.Source.Position, cameraEnd.Position);
            if (!(distance > 0))
                return result;
            logScale -= VcmLog2Positive(distance);
            logScale += VcmSegmentLogTransmittance(cameraEnd, join.Source);
        }
        result.ShadowRays = 1;
        if (!VcmVisible(cameraEnd, join.Source, toLight, infinite))
            return result;
    }
    else
    {
        float3 direction = VcmDirection(lightEnd.Position, cameraEnd.Position);
        VcmEvaluation lightValue = VcmEvaluate(lightEnd, lightEnd.Incoming, direction, TransportImportance);
        VcmEvaluation eyeValue = VcmEvaluate(cameraEnd, cameraEnd.Incoming, -direction, TransportRadiance);
        double leftCosine = lightEnd.Kind == VcmVolume ? 1 :
            abs(dot(lightEnd.ShadingNormal, direction));
        double rightCosine = cameraEnd.Kind == VcmVolume ? 1 :
            abs(dot(cameraEnd.ShadingNormal, direction));
        double distance = VcmSquaredDistance(lightEnd.Position, cameraEnd.Position);
        if (!(leftCosine > 0 && rightCosine > 0 && distance > 0))
            return result;
        a = lightEnd.Throughput;
        b = lightValue.Value;
        c = cameraEnd.Throughput;
        d = eyeValue.Value;
        logScale = VcmLog2Positive(leftCosine) + VcmLog2Positive(rightCosine) - VcmLog2Positive(distance) +
            VcmSegmentLogTransmittance(lightEnd, cameraEnd);
        result.ShadowRays = 1;
        if (!VcmVisible(lightEnd, cameraEnd, direction, false))
            return result;
    }
    VcmMisEvaluation mis = VcmCouplingMis(join, options);
    if (!VcmHasDensity(mis.SelectedLogWeight))
        return result;
    result.MisWeight = VcmRelativeExp2(mis.SelectedLogWeight.Log2Value);
    logScale += mis.SelectedLogWeight.Log2Value;
    result.Radiance = float3(VcmContributionChannel(a.x, b.x, c.x, d.x, logScale),
                             VcmContributionChannel(a.y, b.y, c.y, d.y, logScale),
                             VcmContributionChannel(a.z, b.z, c.z, d.z, logScale));
    if (!VcmSpectrum(result.Radiance))
    {
        result.Radiance = float3(0, 0, 0);
        result.Flags = VcmContributionNumeric;
        VcmReportError(VcmArithmeticError);
        return result;
    }
    result.Flags = VcmContributionValid | (t == 1 ? VcmContributionSplat : 0);
    return result;
}

// A sorted spatial hash retains complete cell coordinates, so collisions cost
// work but never add photons from another cell. Stable photon identities also
// fix summation order across dispatch sizes. Batcher (1968), "Sorting networks
// and their applications", supplies the compare-exchange network only.
// Pass order: light walks -> grid -> sort; camera-cache batches -> estimate ->
// advance merges until complete; then splats -> sort -> commit until complete.
// Preserve a camera cache until all its merge cursors finish, and preserve the
// photon index until ALL camera batches finish: splat indexing reuses it.
// The host inserts UAV barriers after producers and each sorting distance.
// Frame statistics.Completed is a completion flag, not a new camera sample.
struct VcmIndexEntry
{
    uint Key, Identity;
    int CellX, CellY, CellZ;
    uint Reserved0, Reserved1, Reserved2;
};
struct VcmFramePixel
{
    double SumRed, SumGreen, SumBlue;
    double CameraRed, CameraGreen, CameraBlue;
    double SplatRed, SplatGreen, SplatBlue;
    uint CompletedPasses, MergePrefix, Neighbor, Cursor;
    uint Phase, Reserved0, Reserved1, Reserved2;
};
struct VcmSplat
{
    float3 Radiance;
    uint FullPixel;
};
struct VcmFrameSettings
{
    double InitialRadius, Alpha;
    uint LightPathBase, LightPathCount, CameraPathBase, LensPathIndex;
    uint IndexCapacity, BucketCount, WorkBegin, WorkCount;
    uint CameraPixelBegin, SortSpan, SortDistance, Reserved;
};
static const uint VcmReadyForSplats = 0, VcmMerging = 1, VcmReducingSplats = 2, VcmPassComplete = 3;
static const uint VcmAwaitingEstimate = 4;
static const uint VcmMergeWorkBudget = 64, VcmSplatReductionBudget = 1024;
static const uint VcmNeighborSide = 3, VcmNeighborCount = VcmNeighborSide * VcmNeighborSide * VcmNeighborSide;
static const uint VcmPairDimension = 0x50414952u, VcmSplatLensDimension = 0x534c454eu;
VcmPath VcmLoadPath(uint index);
VcmIndexEntry VcmLoadIndex(uint index);
void VcmStoreIndex(uint index, VcmIndexEntry entry);
VcmSplat VcmLoadSplat(uint index);

VcmIndexEntry VcmEmptyIndex()
{
    VcmIndexEntry entry;
    entry.Key = entry.Identity = VcmInvalid;
    entry.CellX = entry.CellY = entry.CellZ = 0;
    entry.Reserved0 = entry.Reserved1 = entry.Reserved2 = 0;
    return entry;
}
VcmFramePixel VcmEmptyFramePixel()
{
    VcmFramePixel pixel;
    pixel.SumRed = pixel.SumGreen = pixel.SumBlue = 0;
    pixel.CameraRed = pixel.CameraGreen = pixel.CameraBlue = 0;
    pixel.SplatRed = pixel.SplatGreen = pixel.SplatBlue = 0;
    pixel.CompletedPasses = pixel.MergePrefix = pixel.Neighbor = 0;
    pixel.Cursor = VcmInvalid;
    pixel.Phase = VcmAwaitingEstimate;
    pixel.Reserved0 = pixel.Reserved1 = pixel.Reserved2 = 0;
    return pixel;
}
double VcmFrameRadius(double initial, double alpha, uint pass)
{
    // Global-radius VCM: r_i = r_0 / (i + 1)^((1-alpha)/2).
    // This is a pass schedule, not a per-pixel PPM radius update.
    return initial * VcmRelativeExp2(-(1 - alpha) / 2 * VcmLog2Positive(double(pass) + 1));
}
uint VcmPhotonSlots(uint maximumVertices)
{
    return maximumVertices > 2 ? maximumVertices - 2 : 0;
}
bool VcmCellAxis(float position, float origin, double radius, out int cell)
{
    static const double CellLimit = 2147483645.0L; // Leave signed room for both neighbors.
    double coordinate = (VcmWidenFloat(position) - VcmWidenFloat(origin)) / radius;
    cell = 0;
    if (!VcmFinite(coordinate) || abs(coordinate) > CellLimit)
        return false;
    cell = int(coordinate);
    if (double(cell) > coordinate)
        --cell; // Integer conversion truncates; a grid requires floor, including negative coordinates.
    return true;
}
uint VcmCellBucket(int x, int y, int z, uint buckets)
{
    static const uint XMultiplier = 0x8da6b343u, YMultiplier = 0xd8163841u, ZMultiplier = 0xcb1ab31fu;
    return ((uint(x) * XMultiplier) ^ (uint(y) * YMultiplier) ^ (uint(z) * ZMultiplier)) & (buckets - 1);
}
VcmIndexEntry VcmGridEntry(uint identity, VcmFrameSettings settings, uint maximumVertices, float3 origin,
                           double radius)
{
    VcmIndexEntry entry = VcmEmptyIndex();
    uint slots = VcmPhotonSlots(maximumVertices);
    if (slots == 0 || identity / slots >= settings.LightPathCount || !(radius > 0))
        return entry;
    VcmPath light = VcmLoadPath(settings.LightPathBase + identity / slots);
    uint prefix = identity % slots + 2;
    if (prefix > light.Count)
        return entry;
    VcmPathVertex vertex = VcmLoadVertex(light.First + prefix - 1);
    if (vertex.Kind != VcmSurface && vertex.Kind != VcmVolume)
        return entry;
    if (!VcmCellAxis(vertex.Position.x, origin.x, radius, entry.CellX) ||
        !VcmCellAxis(vertex.Position.y, origin.y, radius, entry.CellY) ||
        !VcmCellAxis(vertex.Position.z, origin.z, radius, entry.CellZ))
    {
        VcmReportError(VcmArithmeticError);
        return VcmEmptyIndex();
    }
    entry.Key = VcmCellBucket(entry.CellX, entry.CellY, entry.CellZ, settings.BucketCount);
    entry.Identity = identity;
    return entry;
}
bool VcmIndexLess(VcmIndexEntry a, VcmIndexEntry b)
{
    return a.Key < b.Key || (a.Key == b.Key && a.Identity < b.Identity);
}
void VcmSortPair(uint item, uint span, uint distance)
{
    uint partner = item ^ distance;
    if (partner <= item)
        return; // Exactly one invocation owns both entries of this disjoint pair.
    VcmIndexEntry a = VcmLoadIndex(item), b = VcmLoadIndex(partner);
    bool ascending = (item & span) == 0;
    if (ascending ? VcmIndexLess(b, a) : VcmIndexLess(a, b))
    {
        VcmStoreIndex(item, b);
        VcmStoreIndex(partner, a);
    }
}
uint VcmLowerKey(uint key, uint capacity)
{
    uint lower = 0, upper = capacity;
    while (lower < upper)
    {
        uint middle = lower + (upper - lower) / 2;
        if (VcmLoadIndex(middle).Key < key)
            lower = middle + 1;
        else
            upper = middle;
    }
    return lower;
}
void VcmAddCamera(inout VcmFramePixel pixel, VcmContribution value)
{
    if ((value.Flags & VcmContributionValid) == 0)
        return;
    pixel.CameraRed += VcmWidenFloat(value.Radiance.x);
    pixel.CameraGreen += VcmWidenFloat(value.Radiance.y);
    pixel.CameraBlue += VcmWidenFloat(value.Radiance.z);
}
VcmFramePixel VcmEstimateCamera(VcmFramePixel pixel, VcmPath light, VcmPath eye, CameraEndpointFrame frame,
                                VcmCouplingOptions options, uint pass, out uint shadowRays)
{
    shadowRays = 0;
    if (pixel.CompletedPasses != pass)
    {
        VcmReportError(VcmControlsError);
        return pixel;
    }
    pixel.CameraRed = pixel.CameraGreen = pixel.CameraBlue = 0;
    pixel.SplatRed = pixel.SplatGreen = pixel.SplatBlue = 0;
    pixel.MergePrefix = 2;
    pixel.Neighbor = 0;
    pixel.Cursor = VcmInvalid;
    pixel.Phase = options.Radius > 0 && options.MergingSamples > 0 ? VcmMerging : VcmReadyForSplats;
    for (uint t = 2; t <= eye.Count && t <= options.MaximumVertices; ++t)
    {
        VcmAddCamera(pixel, VcmCouple(light, 0, eye, t, 0, frame, float2(0.5f, 0.5f), options));
        uint maximumLight = min(light.Count, options.MaximumVertices - t);
        for (uint s = 1; s <= maximumLight; ++s)
        {
            VcmContribution value = VcmCouple(light, s, eye, t, 0, frame, float2(0.5f, 0.5f), options);
            shadowRays += value.ShadowRays;
            VcmAddCamera(pixel, value);
        }
    }
    return pixel;
}
VcmFramePixel VcmAdvanceMerges(VcmFramePixel pixel, VcmPath eye, VcmFrameSettings settings, float3 origin,
                               CameraEndpointFrame frame, VcmCouplingOptions options, uint budget)
{
    if (pixel.Phase != VcmMerging && pixel.Phase != VcmReadyForSplats)
    {
        VcmReportError(VcmControlsError);
        return pixel;
    }
    // One step visits one bucket candidate or advances an empty cell/prefix.
    // Persisting the cursor bounds dispatch work even when all photons collide.
    for (uint step = 0; step < budget && pixel.Phase == VcmMerging; ++step)
    {
        if (pixel.MergePrefix > eye.Count || pixel.MergePrefix >= options.MaximumVertices)
        {
            pixel.Phase = VcmReadyForSplats;
            pixel.Cursor = VcmInvalid;
            break;
        }
        VcmPathVertex vertex = VcmLoadVertex(eye.First + pixel.MergePrefix - 1);
        if ((vertex.Kind != VcmSurface && vertex.Kind != VcmVolume) ||
            pixel.Neighbor == VcmNeighborCount)
        {
            ++pixel.MergePrefix;
            pixel.Neighbor = 0;
            pixel.Cursor = VcmInvalid;
            continue;
        }
        int x, y, z;
        if (!VcmCellAxis(vertex.Position.x, origin.x, options.Radius, x) ||
            !VcmCellAxis(vertex.Position.y, origin.y, options.Radius, y) ||
            !VcmCellAxis(vertex.Position.z, origin.z, options.Radius, z))
        {
            VcmReportError(VcmArithmeticError);
            pixel.Phase = VcmReadyForSplats;
            break;
        }
        x += int(pixel.Neighbor % VcmNeighborSide) - 1;
        y += int((pixel.Neighbor / VcmNeighborSide) % VcmNeighborSide) - 1;
        z += int(pixel.Neighbor / (VcmNeighborSide * VcmNeighborSide)) - 1;
        uint key = VcmCellBucket(x, y, z, settings.BucketCount);
        if (pixel.Cursor == VcmInvalid)
            pixel.Cursor = VcmLowerKey(key, settings.IndexCapacity);
        if (pixel.Cursor >= settings.IndexCapacity || VcmLoadIndex(pixel.Cursor).Key != key)
        {
            ++pixel.Neighbor;
            pixel.Cursor = VcmInvalid;
            continue;
        }
        VcmIndexEntry entry = VcmLoadIndex(pixel.Cursor++);
        if (entry.CellX != x || entry.CellY != y || entry.CellZ != z)
            continue;
        uint slots = VcmPhotonSlots(options.MaximumVertices);
        uint s = entry.Identity % slots + 2;
        if (s + pixel.MergePrefix - 1 > options.MaximumVertices)
            continue;
        VcmPath light = VcmLoadPath(settings.LightPathBase + entry.Identity / slots);
        VcmAddCamera(pixel,
                     VcmCouple(light, s, eye, pixel.MergePrefix, 1, frame, float2(0.5f, 0.5f), options));
    }
    return pixel;
}

VcmSplat VcmEvaluateSplat(uint identity, VcmFrameSettings settings, CameraEndpointFrame frame,
                          VcmCouplingOptions options, float2 lens, uint fullWidth, uint fullHeight,
                          out uint shadowRays)
{
    VcmSplat splat;
    splat.Radiance = float3(0, 0, 0);
    splat.FullPixel = VcmInvalid;
    shadowRays = 0;
    uint slots = VcmPhotonSlots(options.MaximumVertices);
    if (slots == 0 || identity / slots >= settings.LightPathCount || !(options.LightTracingSamples > 0))
        return splat;
    VcmPath light = VcmLoadPath(settings.LightPathBase + identity / slots);
    uint prefix = identity % slots + 2;
    if (prefix > light.Count)
        return splat;
    VcmContribution value =
        VcmCouple(light, prefix, VcmLoadPath(settings.LensPathIndex), 1, 0, frame, lens, options);
    shadowRays = value.ShadowRays;
    if ((value.Flags & VcmContributionValid) == 0)
        return splat;
    if (!isfinite(value.Raster.x) || !isfinite(value.Raster.y) || value.Raster.x < 0 || value.Raster.y < 0 ||
        value.Raster.x > 1 || value.Raster.y > 1)
    {
        VcmReportError(VcmArithmeticError);
        return splat;
    }
    // Keep full-film keys even for a crop. Normalize the summed light-tracing
    // estimator once, by attempted paths / full-film pixels, during commit.
    uint x = min(uint(value.Raster.x * fullWidth), fullWidth - 1);
    uint y = min(uint(value.Raster.y * fullHeight), fullHeight - 1);
    splat.FullPixel = y * fullWidth + x;
    splat.Radiance = value.Radiance;
    return splat;
}
VcmFramePixel VcmCommitFrame(VcmFramePixel pixel, uint fullPixel, uint capacity, double lightSamples,
                             uint pass, uint budget)
{
    if (pixel.Phase == VcmPassComplete && pixel.CompletedPasses == pass + 1)
        return pixel; // Finished pixels can share a dispatch with unfinished neighbors.
    if (pixel.CompletedPasses != pass ||
        (pixel.Phase != VcmReadyForSplats && pixel.Phase != VcmReducingSplats))
    {
        VcmReportError(VcmControlsError);
        return pixel;
    }
    if (pixel.Phase == VcmReadyForSplats)
    {
        pixel.Cursor = VcmLowerKey(fullPixel, capacity);
        pixel.Phase = VcmReducingSplats;
    }
    for (uint step = 0; step < budget; ++step)
    {
        if (pixel.Cursor >= capacity || VcmLoadIndex(pixel.Cursor).Key != fullPixel)
        {
            double normalization = lightSamples > 0 ? 1 / lightSamples : 0;
            pixel.SumRed += pixel.CameraRed + pixel.SplatRed * normalization;
            pixel.SumGreen += pixel.CameraGreen + pixel.SplatGreen * normalization;
            pixel.SumBlue += pixel.CameraBlue + pixel.SplatBlue * normalization;
            ++pixel.CompletedPasses;
            pixel.Phase = VcmPassComplete;
            pixel.Cursor = VcmInvalid;
            break;
        }
        VcmSplat splat = VcmLoadSplat(VcmLoadIndex(pixel.Cursor++).Identity);
        pixel.SplatRed += VcmWidenFloat(splat.Radiance.x);
        pixel.SplatGreen += VcmWidenFloat(splat.Radiance.y);
        pixel.SplatBlue += VcmWidenFloat(splat.Radiance.z);
    }
    return pixel;
}

struct VcmTexture
{
    uint OffsetTexels, Width, Height, Reserved;
};
StructuredBuffer<MaterialData> Materials : register(t3);
StructuredBuffer<VcmTexture> Textures : register(t5);
StructuredBuffer<uint> TextureTexels : register(t6);
StructuredBuffer<PackedEmitter> Emitters : register(t13);
RWStructuredBuffer<VcmPathVertex> VcmVertices : register(u5);
RWStructuredBuffer<VcmPath> VcmPaths : register(u6);
RWByteAddressBuffer VcmErrors : register(u7);
RWStructuredBuffer<VcmWalkStatistics> VcmStatistics : register(u8);
struct VcmCouplingQuery
{
    uint LightPath, EyePath, S, T;
    float2 Lens;
    uint Merge, Reserved;
};
StructuredBuffer<VcmCouplingQuery> VcmQueries : register(t14);
RWStructuredBuffer<VcmContribution> VcmContributions : register(u9);
RWStructuredBuffer<VcmIndexEntry> VcmIndex : register(u10);
RWStructuredBuffer<VcmFramePixel> VcmFramePixels : register(u11);
RWStructuredBuffer<VcmSplat> VcmSplats : register(u12);
struct VcmLinearFilmPixel
{
    float3 MeanRadiance;
    float LuminanceM2;
    uint SampleCount;
};
RWStructuredBuffer<uint> OutputPixels : register(u0);
RWStructuredBuffer<VcmLinearFilmPixel> LinearFilm : register(u2);
cbuffer VcmFrameControls : register(b3)
{
    VcmFrameSettings Frame;
};
cbuffer VcmWalkControls : register(b1)
{
    float3 EmissionCenter;
    float EmissionRadius;
    uint EmitterCount, PathBegin, PathCount, OutputBegin;
    uint PassIndex, PopulationSeed, VertexLimit, Stage;
    // Root descriptors have no resource extent. The RHI supplies allocation
    // capacities explicitly; GetDimensions is not valid for this binding model.
    // https://microsoft.github.io/DirectX-Specs/d3d/ResourceBinding.html
    uint PathCapacity, VertexCapacity, StatisticsCapacity, SourceCapacity;
};
cbuffer VcmCouplingControls : register(b2)
{
    double MergeRadius, MergingSamples, LightTracingSamples;
    uint CouplingVertexLimit, CouplingHeuristic, CouplingNee, CouplingCount;
    uint CouplingBegin, ContributionBegin;
    uint QueryCapacity, ContributionCapacity, ReservedCoupling0, ReservedCoupling1;
};
void VcmReportError(uint flag)
{
    uint ignored;
    VcmErrors.InterlockedOr(0, flag, ignored);
}
VcmPath VcmLoadPath(uint index)
{
    return VcmPaths[index];
}
VcmIndexEntry VcmLoadIndex(uint index)
{
    return VcmIndex[index];
}
void VcmStoreIndex(uint index, VcmIndexEntry entry)
{
    VcmIndex[index] = entry;
}
VcmSplat VcmLoadSplat(uint index)
{
    return VcmSplats[index];
}
void VcmStoreVertex(uint index, VcmPathVertex vertex)
{
    VcmVertices[index] = vertex;
}
VcmPathVertex VcmLoadVertex(uint index)
{
    return VcmVertices[index];
}
float VcmMeanFreePath(uint material) { return Materials[material].Optical.w; }
float3 VcmInteriorAlbedo(uint material)
{
    return Materials[material].AlbedoReflectivity.xyz * Materials[material].ConductorExtinction.w;
}
float VcmAnisotropy(uint material) { return Materials[material].ConductorIndex.w; }
bool VcmImportanceSampling() { return (TransportFlags & ImportanceBit) != 0; }
float VcmLinearChannel(uint value)
{
    static const float ByteMaximum = 255, Breakpoint = 0.04045f, LinearSlope = 12.92f;
    static const float GammaOffset = 0.055f, GammaScale = 1.055f, GammaPower = 2.4f;
    float encoded = value / ByteMaximum;
    return encoded <= Breakpoint ? encoded / LinearSlope
                                 : pow((encoded + GammaOffset) / GammaScale, GammaPower);
}
float3 VcmTextureValue(VcmTexture texture, int x, int y)
{
    x = (x % int(texture.Width) + int(texture.Width)) % int(texture.Width);
    y = (y % int(texture.Height) + int(texture.Height)) % int(texture.Height);
    uint packed = TextureTexels[texture.OffsetTexels + uint(y) * texture.Width + uint(x)];
    static const uint ByteMask = 0xff, GreenShift = 8, BlueShift = 16;
    return float3(VcmLinearChannel(packed & ByteMask), VcmLinearChannel((packed >> GreenShift) & ByteMask),
                  VcmLinearChannel((packed >> BlueShift) & ByteMask));
}
float3 VcmSurfaceColor(Payload hit)
{
    uint textureIndex = Materials[hit.MaterialIndex].DiffuseTextureIndex;
    if (textureIndex >= TextureCount || (hit.Flags & SurfaceTexCoordFlag) == 0 ||
        !all(isfinite(hit.TexCoord)))
        return hit.Color;
    VcmTexture texture = Textures[textureIndex];
    if (texture.Width == 0 || texture.Height == 0 || texture.OffsetTexels > TextureTexelCount ||
        texture.Height > (TextureTexelCount - texture.OffsetTexels) / texture.Width)
        return hit.Color;
    float2 coordinate =
        frac(float2(hit.TexCoord.x, 1 - hit.TexCoord.y)) * float2(texture.Width, texture.Height) - 0.5f;
    int x = int(floor(coordinate.x)), y = int(floor(coordinate.y));
    float2 weight = frac(coordinate);
    float3 lower = lerp(VcmTextureValue(texture, x, y), VcmTextureValue(texture, x + 1, y), weight.x);
    float3 upper = lerp(VcmTextureValue(texture, x, y + 1), VcmTextureValue(texture, x + 1, y + 1), weight.x);
    return hit.Color * lerp(lower, upper, weight.y);
}
VcmArrival VcmSceneArrival(float3 origin, float3 direction, float maximum)
{
    VcmArrival result = (VcmArrival)0;
    result.Kind = VcmEscape;
    result.Material = result.Primitive = result.Emitter = VcmInvalid;
    result.Interior = VcmInvalid;
    Payload hit = CastRay(origin, direction, RayEpsilon, maximum);
    bool surface = (hit.Flags & SurfaceHitFlag) != 0;
    if (surface && hit.MaterialIndex >= MaterialCount)
    {
        VcmReportError(VcmControlsError);
        return result;
    }
    float distance = asfloat(hit.DistanceBits);
    if (surface && (hit.Flags & SurfaceFrontFaceFlag) == 0 &&
        VcmMeanFreePath(hit.MaterialIndex) > 0)
        result.Interior = hit.MaterialIndex;
    // A rectangle's display proxy may differ by one ray offset. Only that
    // tagged proxy permits the tolerance; an ordinary occluder stays exact.
    bool proxy = surface && Materials[hit.MaterialIndex].SampledEmitter != 0;
    float nearest = distance + (proxy ? RayEpsilon : 0);
    for (uint source = 0; source < EmitterCount; ++source)
    {
        PackedEmitter emitter = Emitters[source];
        if (result.Kind != VcmEmitter && surface && emitter.Primitive == hit.PrimitiveIndex &&
            (emitter.Kind == PackedEmitterTriangle || emitter.Kind == PackedEmitterSphere))
            result.Emitter = source;
        if (result.Kind != VcmEmitter && !surface && emitter.Kind == PackedEmitterEnvironment)
            result.Emitter = source;
        if (emitter.Kind != PackedEmitterRectangle)
            continue;
        float projection = dot(direction, emitter.Direction);
        if (projection == 0)
            continue;
        float candidate = dot(emitter.Origin - origin, emitter.Direction) / projection;
        if (!(candidate > RayEpsilon && candidate < nearest))
            continue;
        LightEmissionAxes axes = EmissionAxes(emitter.Direction);
        float3 relative = origin + direction * candidate - emitter.Origin;
        if (abs(dot(relative, axes.Right)) > emitter.Width / 2 ||
            abs(dot(relative, axes.Up)) > emitter.Height / 2)
            continue;
        nearest = candidate;
        result.Kind = VcmEmitter;
        result.Emitter = source;
        result.Primitive = emitter.Primitive;
        result.Position = origin + direction * candidate;
        result.GeometricNormal = result.ShadingNormal = emitter.Direction;
        result.FrontFace = projection < 0 ? 1 : 0;
    }
    if (result.Kind == VcmEmitter || !surface)
        return result;
    result.Kind = VcmSurface;
    result.Position = origin + direction * distance;
    result.ShadingNormal = hit.Normal;
    result.GeometricNormal = hit.GeometricNormal;
    result.Color = VcmSurfaceColor(hit);
    result.Material = hit.MaterialIndex;
    result.Primitive = hit.PrimitiveIndex;
    result.FrontFace = (hit.Flags & SurfaceFrontFaceFlag) != 0 ? 1 : 0;
    return result;
}
VcmArrival VcmIntersect(float3 origin, float3 direction)
{
    return VcmSceneArrival(origin, direction, MaximumRayDistance);
}
VcmScattering VcmScatter(VcmPathVertex vertex, uint mode, float2 variates, float lobe)
{
    VcmScattering result = (VcmScattering)0;
    MaterialData material = Materials[vertex.Material];
    float mirror = material.Model == MaterialLegacy ? saturate(material.AlbedoReflectivity.w) : 0;
    if (material.Model == MaterialLegacy)
        material.Model = MaterialDiffuse;
    Payload hit = MissPayload(0);
    hit.Normal = vertex.ShadingNormal;
    hit.GeometricNormal = vertex.GeometricNormal;
    hit.Flags = vertex.FrontFace != 0 ? SurfaceFrontFaceFlag : 0;
    SurfaceBsdf surface = MakeSurfaceBsdf(material, hit, vertex.Incoming, vertex.Color, mode);
    if (lobe < mirror)
    {
        float3 outgoing = reflect(-vertex.Incoming, vertex.ShadingNormal);
        if (surface.View.z <= 0 || !SurfaceDirectionValid(surface, outgoing, dot(outgoing, surface.Normal)))
            return result;
        result.Direction = outgoing;
        result.Weight = SurfaceAdjointCorrection(surface, outgoing).xxx;
        result.Pdf = result.ReversePdf = mirror;
        result.RelativeIndex = 1;
        result.Event = ScatteringReflection | ScatteringDelta;
        return result;
    }
    if (mirror >= 1)
        return result;
    SurfaceSample sampled = SampleSurface(surface, variates, (lobe - mirror) / (1 - mirror));
    result.Direction = sampled.Direction;
    result.Weight = sampled.Weight;
    result.Pdf = sampled.Pdf * (1 - mirror);
    result.ReversePdf = sampled.ReversePdf * (1 - mirror);
    result.RelativeIndex = sampled.RelativeIndex;
    result.Event = sampled.Event;
    return result;
}
VcmEvaluation VcmEvaluate(VcmPathVertex vertex, float3 incoming, float3 outgoing, uint mode)
{
    VcmEvaluation result = (VcmEvaluation)0;
    if (vertex.Kind == VcmVolume)
        return VcmVolumeEvaluation(vertex, incoming, outgoing);
    if (vertex.Kind != VcmSurface || vertex.Material >= MaterialCount)
        return result;
    MaterialData material = Materials[vertex.Material];
    float continuous = material.Model == MaterialLegacy ? 1 - saturate(material.AlbedoReflectivity.w) : 1;
    if (material.Model == MaterialLegacy)
        material.Model = MaterialDiffuse;
    Payload hit = MissPayload(0);
    bool front = dot(incoming, vertex.GeometricNormal) > 0;
    // Flip the shading frame with the geometric medium side, not whenever a
    // grazing direction falls below the interpolated normal's hemisphere.
    hit.Normal = front == (vertex.FrontFace != 0) ? vertex.ShadingNormal : -vertex.ShadingNormal;
    hit.GeometricNormal = vertex.GeometricNormal;
    hit.Flags = front ? SurfaceFrontFaceFlag : 0;
    SurfaceBsdf surface = MakeSurfaceBsdf(material, hit, incoming, vertex.Color, mode);
    SurfaceEvaluation value = EvaluateSurface(surface, outgoing);
    result.Value = value.Value * continuous;
    result.Pdf = value.Pdf * continuous;
    result.ReversePdf = value.ReversePdf * continuous;
    return result;
}
EmitterEndpointDensity VcmSourceDensity(uint emitter, float3 normal, float3 direction)
{
    EmissionDomain domain;
    domain.Center = EmissionCenter;
    domain.Radius = EmissionRadius;
    return EmitterEndpointPdf(Emitters, EmitterCount, domain, emitter, normal, direction);
}
float3 VcmSourceValue(uint emitter, float3 direction)
{
    if (emitter >= EmitterCount)
        return float3(0, 0, 0);
    PackedEmitter source = Emitters[emitter];
    if (source.Kind == PackedEmitterEnvironment)
        return lerp(source.BackgroundBottom, source.BackgroundTop, (1 - direction.y) / 2);
    if (source.Kind == PackedEmitterRectangle)
        return dot(source.Direction, direction) > 0 ? source.Strength / (source.Width * source.Height)
                                                    : float3(0, 0, 0);
    return source.Strength;
}
bool VcmVisible(VcmPathVertex from, VcmPathVertex to, float3 direction, bool infinite)
{
    float3 origin = from.Kind == VcmSurface ? VcmOffset(from, direction) : from.Position;
    float maximum = MaximumRayDistance;
    if (!infinite)
    {
        float3 end = to.Kind == VcmSurface ? VcmOffset(to, -direction) : to.Position;
        float3 segment = end - origin;
        float distance = CameraLength(segment);
        if (!(distance > RayEpsilon * 2))
            return true;
        maximum = distance - RayEpsilon;
        direction = segment / distance;
    }
    VcmArrival blocker = VcmSceneArrival(origin, direction, maximum);
    return blocker.Kind == VcmEscape;
}
CameraEndpointFrame VcmCameraFrame()
{
    CameraEndpointFrame frame;
    frame.Position = CameraPosition;
    frame.Forward = CameraForward;
    frame.Right = CameraRight;
    frame.Up = CameraUp;
    frame.HalfWidth = HalfWidth;
    frame.HalfHeight = HalfHeight;
    frame.LensRadius = LensRadius;
    frame.FocusDistance = FocusDistance;
    return frame;
}
uint VcmDisplayChannel(float radiance)
{
    static const float SrgbBreakpoint = 0.0031308f, SrgbSlope = 12.92f;
    static const float SrgbScale = 1.055f, SrgbOffset = 0.055f, SrgbExponent = 2.4f;
    static const float ByteMaximum = 255, RoundNearest = 0.5f;
    float display = GammaCorrection == 0 ? radiance
                                         : (radiance <= SrgbBreakpoint
                                                ? SrgbSlope * radiance
                                                : SrgbScale * pow(radiance, 1 / SrgbExponent) - SrgbOffset);
    return uint(saturate(display) * ByteMaximum + RoundNearest);
}
void VcmPublishFilm(uint pixel, VcmFramePixel state)
{
    double samples = double(state.CompletedPasses);
    float3 radiance = float3(float(state.SumRed / samples), float(state.SumGreen / samples),
                             float(state.SumBlue / samples));
    if (!VcmSpectrum(radiance))
    {
        VcmReportError(VcmArithmeticError);
        return;
    }
    VcmLinearFilmPixel film;
    film.MeanRadiance = radiance;
    film.LuminanceM2 = 0; // Radius changes across passes: this is not an IID variance estimate.
    film.SampleCount = state.CompletedPasses;
    LinearFilm[pixel] = film;
    float peak = max(1, max(radiance.x, max(radiance.y, radiance.z)));
    float3 display = radiance / peak;
    static const uint GreenShift = 8, BlueShift = 16, OpaqueAlpha = 0xff000000u;
    OutputPixels[pixel] = VcmDisplayChannel(display.x) | (VcmDisplayChannel(display.y) << GreenShift) |
                          (VcmDisplayChannel(display.z) << BlueShift) | OpaqueAlpha;
}
bool VcmPowerOfTwoCount(uint count)
{
    return count != 0 && (count & (count - 1)) == 0;
}
SampleContext VcmLightContext(uint path)
{
    SampleContext context = MakeSampleContext(path, 0, 1, PassIndex, SamplesPerPixel, SamplingSequence, SamplingSeed);
    context.PixelSeed = Hash(PopulationSeed ^ VcmLightStream ^ (path * PixelSeedMultiplier));
    return context;
}
void VcmFrameDispatch(uint lane, uint stage)
{
    if (lane >= Frame.WorkCount)
        return;
    uint slots = VcmPhotonSlots(VertexLimit);
    if (Stage != stage || VertexLimit < 2 || VertexLimit > VcmMaximumVertices ||
        CouplingVertexLimit != VertexLimit || CouplingHeuristic > VcmPowerMis || CouplingNee > 1 ||
        SamplesPerPixel == 0 || PassIndex >= SamplesPerPixel || SamplingSequence > BlueNoiseSequence ||
        !VcmFinite(Frame.InitialRadius) || Frame.InitialRadius < 0 || !VcmFinite(Frame.Alpha) ||
        !(Frame.Alpha > 0 && Frame.Alpha <= 1) || !VcmPowerOfTwoCount(Frame.IndexCapacity) ||
        !VcmPowerOfTwoCount(Frame.BucketCount) ||
        (slots != 0 && Frame.LightPathCount > Frame.IndexCapacity / slots) ||
        Frame.LightPathBase > PathCapacity || Frame.LightPathCount > PathCapacity - Frame.LightPathBase ||
        Frame.LensPathIndex >= PathCapacity || Frame.LensPathIndex >= VertexCapacity / VcmMaximumVertices ||
        Frame.LightPathBase > VertexCapacity / VcmMaximumVertices ||
        Frame.LightPathCount > VertexCapacity / VcmMaximumVertices - Frame.LightPathBase ||
        Frame.WorkBegin > VcmInvalid - Frame.WorkCount || EmitterCount > SourceCapacity ||
        EmitterCount > UniformEmitterLimit || Width == 0 || Height == 0 || Width > VcmInvalid / Height ||
        FullWidth == 0 || FullHeight == 0 || FullWidth > (VcmInvalid - 1) / FullHeight || TileX > FullWidth ||
        Width > FullWidth - TileX || TileY > FullHeight || Height > FullHeight - TileY)
    {
        VcmReportError(VcmControlsError);
        return;
    }
    uint item = Frame.WorkBegin + lane, end = Frame.WorkBegin + Frame.WorkCount;
    bool cameraWork = stage == VcmEstimateStage || stage == VcmMergeStage;
    bool transport = cameraWork || stage == VcmSplatStage || stage == VcmCommitStage;
    uint extent = (stage == VcmInitializeStage || stage == VcmCommitStage || cameraWork)
                      ? Width * Height
                      : Frame.IndexCapacity;
    if (end > extent || (transport && Frame.WorkCount > StatisticsCapacity) ||
        (cameraWork && (Frame.CameraPixelBegin > extent - end || Frame.CameraPathBase > PathCapacity ||
                        end > PathCapacity - Frame.CameraPathBase ||
                        Frame.CameraPathBase > VertexCapacity / VcmMaximumVertices ||
                        end > VertexCapacity / VcmMaximumVertices - Frame.CameraPathBase)))
    {
        VcmReportError(VcmControlsError);
        return;
    }
    VcmCouplingOptions options;
    options.Radius = VcmFrameRadius(Frame.InitialRadius, Frame.Alpha, PassIndex);
    options.MergingSamples = double(Frame.LightPathCount);
    options.LightTracingSamples = double(Frame.LightPathCount) / double(FullWidth * FullHeight);
    options.MaximumVertices = VertexLimit;
    options.Heuristic = CouplingHeuristic;
    options.NextEventEstimation = CouplingNee;
    CameraEndpointFrame camera = VcmCameraFrame();
    if (stage == VcmInitializeStage)
    {
        VcmFramePixels[item] = VcmEmptyFramePixel();
        OutputPixels[item] = 0;
        LinearFilm[item] = (VcmLinearFilmPixel)0;
        if (item == 0)
        {
            SampleContext context = MakeSampleContext(0, 0, FullWidth, 0, SamplesPerPixel, SamplingSequence, SamplingSeed);
            VcmPath sensor = VcmCameraWalk(camera, float2(0.5f, 0.5f), float2(0.5f, 0.5f), context,
                                           Frame.LensPathIndex * VcmMaximumVertices, 1)
                                 .Path;
            sensor.Limit = VertexLimit;
            VcmPaths[Frame.LensPathIndex] = sensor;
        }
        return;
    }
    if (stage == VcmGridStage)
    {
        VcmIndex[item] = VcmGridEntry(item, Frame, VertexLimit, EmissionCenter, options.Radius);
        return;
    }
    if (stage == VcmSortStage)
    {
        if (!VcmPowerOfTwoCount(Frame.SortSpan) || Frame.SortSpan < 2 ||
            Frame.SortSpan > Frame.IndexCapacity || !VcmPowerOfTwoCount(Frame.SortDistance) ||
            Frame.SortDistance >= Frame.SortSpan)
            VcmReportError(VcmControlsError);
        else
            VcmSortPair(item, Frame.SortSpan, Frame.SortDistance);
        return;
    }
    VcmWalkStatistics statistics = VcmEmptyWalk(VcmEmptyPath(0, 0, 0)).Statistics;
    if (cameraWork)
    {
        uint pixel = Frame.CameraPixelBegin + item;
        VcmPath eye = VcmPaths[Frame.CameraPathBase + item];
        VcmFramePixel state = VcmFramePixels[pixel];
        if (stage == VcmEstimateStage)
        {
            uint x = TileX + pixel % Width, y = TileY + pixel / Width;
            SampleContext context =
                MakeSampleContext(x, y, FullWidth, PassIndex, SamplesPerPixel, SamplingSequence, SamplingSeed);
            VcmPath light = VcmEmptyPath(0, VertexLimit, TransportImportance);
            if (Frame.LightPathCount > 0)
            {
                uint paired = min(uint(Sample2D(context, VcmPairDimension).x * Frame.LightPathCount),
                                  Frame.LightPathCount - 1);
                light = VcmPaths[Frame.LightPathBase + paired];
            }
            state = VcmEstimateCamera(state, light, eye, camera, options, PassIndex, statistics.Shadow.x);
        }
        else if (state.CompletedPasses != PassIndex)
            VcmReportError(VcmControlsError);
        else
            state = VcmAdvanceMerges(state, eye, Frame, EmissionCenter, camera, options, VcmMergeWorkBudget);
        VcmFramePixels[pixel] = state;
        statistics.Completed.x = state.Phase == VcmMerging ? 0 : 1;
    }
    else if (stage == VcmSplatStage)
    {
        uint prefix = slots == 0 ? 0 : item % slots + 2;
        SampleContext context = VcmLightContext(slots == 0 ? 0 : item / slots);
        VcmSplat splat =
            VcmEvaluateSplat(item, Frame, camera, options,
                             Sample2D(context, VcmSplatLensDimension ^ (prefix * VcmVertexScramble)),
                             FullWidth, FullHeight, statistics.Shadow.x);
        VcmSplats[item] = splat;
        VcmIndexEntry entry = VcmEmptyIndex();
        if (splat.FullPixel != VcmInvalid)
        {
            entry.Key = splat.FullPixel;
            entry.Identity = item;
        }
        VcmIndex[item] = entry;
        statistics.Completed.x = 1;
    }
    else if (stage == VcmCommitStage)
    {
        uint fullPixel = (TileY + item / Width) * FullWidth + TileX + item % Width;
        VcmFramePixel state = VcmCommitFrame(VcmFramePixels[item], fullPixel, Frame.IndexCapacity,
                                             options.LightTracingSamples, PassIndex, VcmSplatReductionBudget);
        VcmFramePixels[item] = state;
        statistics.Completed.x = state.Phase == VcmPassComplete ? 1 : 0;
        if (statistics.Completed.x != 0)
            VcmPublishFilm(item, state);
    }
    VcmStatistics[lane] = statistics;
}
void VcmTransportDispatch(uint lane, uint stage)
{
    if (Stage != stage)
    {
        VcmReportError(VcmControlsError);
        return;
    }
    if (stage == VcmCouplingStage)
    {
        if (lane >= CouplingCount)
            return;
        if (CouplingBegin > QueryCapacity || CouplingCount > QueryCapacity - CouplingBegin ||
            ContributionBegin > ContributionCapacity ||
            CouplingCount > ContributionCapacity - ContributionBegin || EmitterCount > SourceCapacity ||
            EmitterCount > UniformEmitterLimit)
        {
            VcmReportError(VcmControlsError);
            return;
        }
        VcmCouplingQuery query = VcmQueries[CouplingBegin + lane];
        if (query.LightPath >= PathCapacity || query.EyePath >= PathCapacity)
        {
            VcmReportError(VcmControlsError);
            return;
        }
        VcmPath light = VcmPaths[query.LightPath], eye = VcmPaths[query.EyePath];
        if (light.First > VertexCapacity || light.Count > VertexCapacity - light.First ||
            eye.First > VertexCapacity || eye.Count > VertexCapacity - eye.First)
        {
            VcmReportError(VcmControlsError);
            return;
        }
        VcmCouplingOptions options;
        options.Radius = MergeRadius;
        options.MergingSamples = MergingSamples;
        options.LightTracingSamples = LightTracingSamples;
        options.MaximumVertices = CouplingVertexLimit;
        options.Heuristic = CouplingHeuristic;
        options.NextEventEstimation = CouplingNee;
        VcmContributions[ContributionBegin + lane] =
            VcmCouple(light, query.S, eye, query.T, query.Merge, VcmCameraFrame(), query.Lens, options);
        return;
    }
    if (lane >= PathCount)
        return;
    if (VertexLimit == 0 || VertexLimit > VcmMaximumVertices ||
        SamplesPerPixel == 0 || PassIndex >= SamplesPerPixel || SamplingSequence > BlueNoiseSequence ||
        EmitterCount > SourceCapacity || EmitterCount > UniformEmitterLimit ||
        OutputBegin > PathCapacity || PathCount > PathCapacity - OutputBegin ||
        PathCount > StatisticsCapacity || OutputBegin > VertexCapacity / VcmMaximumVertices ||
        PathCount > VertexCapacity / VcmMaximumVertices - OutputBegin || PathBegin > VcmInvalid - PathCount)
    {
        VcmReportError(VcmControlsError);
        return;
    }
    uint output = OutputBegin + lane, pathIndex = PathBegin + lane;
    VcmWalkResult result;
    if (stage == VcmCameraStage)
    {
        if (Width == 0 || Height == 0 || Width > VcmInvalid / Height || pathIndex >= Width * Height ||
            FullHeight == 0 || FullWidth > VcmInvalid / FullHeight || TileX > FullWidth ||
            Width > FullWidth - TileX || TileY > FullHeight || Height > FullHeight - TileY)
        {
            VcmReportError(VcmControlsError);
            return;
        }
        uint x = TileX + pathIndex % Width, y = TileY + pathIndex / Width;
        SampleContext context =
            MakeSampleContext(x, y, FullWidth, PassIndex, SamplesPerPixel, SamplingSequence, SamplingSeed);
        float2 raster = CameraPixelRaster(x, y, FullWidth, FullHeight, Sample2D(context, 0));
        result = VcmCameraWalk(VcmCameraFrame(), raster, Sample2D(context, VcmLensDimension), context,
                               output * VcmMaximumVertices, VertexLimit);
    }
    else
    {
        SampleContext context = VcmLightContext(pathIndex);
        EmissionDomain domain;
        domain.Center = EmissionCenter;
        domain.Radius = EmissionRadius;
        float2 selection = Sample2D(context, VcmSourceDimension);
        EmitterEndpoint endpoint = SampleEmitterEndpoint(Emitters, EmitterCount, domain, selection.x,
                                                         Sample2D(context, VcmPositionDimension),
                                                         Sample2D(context, VcmDirectionDimension));
        result = VcmLightWalk(endpoint, context, output * VcmMaximumVertices, VertexLimit);
    }
    result.Statistics.Completed = uint2(1, 0);
    VcmPaths[output] = result.Path;
    VcmStatistics[lane] = result.Statistics;
}

// Each entry specializes the same DXR estimator at a dispatch boundary. The
// native compiler no longer has to schedule every walk, VC/VM estimator and
// reduction in one RayGen. Storage, accumulation order and work budgets are
// unchanged across these offline stages; preview does not dispatch VCM.
[shader("raygeneration")]
void RayGen() { VcmTransportDispatch(DispatchRaysIndex().x, VcmCameraStage); }
[shader("raygeneration")]
void VcmLight() { VcmTransportDispatch(DispatchRaysIndex().x, VcmLightStage); }
[shader("raygeneration")]
void VcmCoupling() { VcmTransportDispatch(DispatchRaysIndex().x, VcmCouplingStage); }
[shader("raygeneration")]
void VcmInitialize() { VcmFrameDispatch(DispatchRaysIndex().x, VcmInitializeStage); }
[shader("raygeneration")]
void VcmGrid() { VcmFrameDispatch(DispatchRaysIndex().x, VcmGridStage); }
[shader("raygeneration")]
void VcmSort() { VcmFrameDispatch(DispatchRaysIndex().x, VcmSortStage); }
[shader("raygeneration")]
void VcmEstimate() { VcmFrameDispatch(DispatchRaysIndex().x, VcmEstimateStage); }
[shader("raygeneration")]
void VcmMerge() { VcmFrameDispatch(DispatchRaysIndex().x, VcmMergeStage); }
[shader("raygeneration")]
void VcmSplat() { VcmFrameDispatch(DispatchRaysIndex().x, VcmSplatStage); }
[shader("raygeneration")]
void VcmCommit() { VcmFrameDispatch(DispatchRaysIndex().x, VcmCommitStage); }
