// Independent Metal VCM transport. Georgiev et al. (2012), "Light Transport
// Simulation with Vertex Connection and Merging"; Veach (1997), chapters 5/10.
// https://www.iliyan.com/publications/ImplementingVCM/ImplementingVCM_TechRep2012_rev2.pdf
// Radiance and importance walks retain actual sampled directions and generalized
// forward/reverse arrival densities. Delta branch masses acquire the geometric
// Jacobian too. Reconnection must evaluate its changed endpoints separately.
// Shared includes below are Metal math, geometry, material and sampling leaves;
// no other transport integrator supplies a walk or estimator.
#include <metal_stdlib>
using namespace metal;
constant float Pi = 3.14159265358979323846f;
constant float RayEpsilon = 1e-4f, MaximumRayDistance = 1e30f, BasisParallelThreshold = 0.999f;
constant uint PixelSeedMultiplier = 9781u, SampleSeedMultiplier = 6271u;
constant uint BounceSeedMultiplier = 0x9e3779b9u, SecondDimensionScramble = 0x68bc21ebu;
constant uint AvalancheMultiplierA = 0x7feb352du, AvalancheMultiplierB = 0x846ca68bu;
constant uint RandomMantissaMask = 0x00ffffffu, ImportanceBit = 1u;
constant float RandomMantissaRange = 16777216.0f;
constant uint VcmCameraStream = 0x56434341u, VcmLightStream = 0x56434c49u;
constant uint VcmRasterDomain = 0x52415354u, VcmLensDomain = 0x4c454e53u;
constant uint VcmSourceDomain = 0x534f5552u, VcmPositionDomain = 0x504f5349u;
constant uint VcmDirectionDomain = 0x44495245u, VcmBranchDomain = 0x4252414eu;
constant uint VcmInteriorFlightDomain = 0x5643464cu, VcmInteriorPhaseDomain = 0x56435048u;
constant uint VcmCameraVertex = 0, VcmEmitterVertex = 1, VcmSurfaceVertex = 2, VcmEscapeVertex = 3;
constant uint VcmVolumeVertex = 5;
constant uint VcmEndInvalid = 0, VcmEndDepth = 1, VcmEndEscape = 2, VcmEndSource = 3;
constant uint VcmEndAbsorbed = 4, VcmEndNull = 5, VcmEndError = 6;
// Bit zero is owned by native instance traversal.
constant uint VcmBoundsError = 2, VcmNumericalError = 4, VcmControlError = 8;
constant uint VcmRouletteBit = 4;
constant uint VcmInvalidIndex = 0xffffffffu;
#include "SurfaceTypes.metal"
#include "ShadingTypes.metal"
#include "ExtendedFloat.metal"
#include "VertexConnectionMergingMis.metal"
#include "Hash.metal"
#include "Sequence.metal"
void IncrementCounter(thread uint2& value)
{
    if (++value.x == 0)
        ++value.y;
}
#include "TraversalTypes.metal"
#include "Triangle.metal"
#include "Sphere.metal"
#include "Bvh.metal"
#include "KdTree.metal"
#include "Bih.metal"
#include "Dispatch.metal"
#include "BsdfTypes.metal"
#include "Fresnel.metal"
#include "Ggx.metal"
#include "Diffuse.metal"
#include "Conductor.metal"
#include "Dielectric.metal"
#include "SurfaceScattering.metal"
#include "SubsurfaceScattering.metal"
#include "LightSelection.metal"
#include "LightEmission.metal"
#include "SurfaceEmission.metal"
#include "EmitterEndpoint.metal"
#include "CameraEndpoint.metal"

struct VcmSceneControls
{
    packed_float3 CameraPosition;
    uint FullWidth;
    packed_float3 CameraForward;
    uint FullHeight;
    packed_float3 CameraRight;
    uint TotalSamples;
    packed_float3 CameraUp;
    uint MaxReflections;
    float HalfWidth, HalfHeight, LensRadius, FocusDistance;
    uint MaterialCount, NodeCount, TriangleCount, Acceleration;
    uint InstanceNodeCount, TextureCount, TexelCount, TransportFlags;
    packed_float3 EmissionCenter;
    float EmissionRadius;
    uint SourceCount, Sequence, Seed, SamplingSeed;
};
struct VcmWalkDispatch
{
    uint FirstPath, PathCount, Pass, OutputPathBegin;
    uint OutputVertexBegin, VertexStride, PathCapacity, VertexCapacity;
    uint StatisticsCapacity, SourceCapacity, LightPopulation, Reserved;
};
struct VcmWalkVertex
{
    packed_float3 Position;
    uint Kind;
    packed_float3 Normal;
    uint FrontFace;
    packed_float3 Geometry;
    uint Material;
    packed_float3 Incoming;
    uint Primitive;
    packed_float3 Color;
    uint Source;
    packed_float3 Outgoing;
    uint Event;
    VcmWideVector Throughput;
    // Source-vertex Arrival includes its selection PMF. Other arrivals contain
    // only the preceding directional law and geometry; no accumulated product.
    VcmNumber Arrival, ReverseArrival, ForwardAngular, ReverseAngular;
    float2 TexCoord;
    uint HasTexCoord, Reserved;
};
struct VcmWalk
{
    uint Begin, Count, Mode, End;
    uint Identity, Pass, Source, EndpointFlags;
    float2 Raster, LensPoint;
    packed_float3 LaunchDirection;
    uint Reserved;
    // Camera DirectionPdf is full-film, not conditioned on one pixel. For an
    // infinite source PositionPdf is the launch disk and DirectionPdf angular.
    // Keep the source PMF separate when assembling each complete MIS strategy.
    VcmNumber PositionPdf, DirectionPdf, Selection;
};
struct VcmWalkStatistics
{
    uint2 Camera, Transport, Tests, Shadow;
};
static_assert(sizeof(VcmSceneControls) == 144 && sizeof(VcmWalkDispatch) == 48 &&
                  sizeof(VcmWalkVertex) == 224 && sizeof(VcmWalk) == 112 && sizeof(VcmWalkStatistics) == 32,
              "Metal VCM transport ABI mismatch");
struct VcmScene
{
    SceneGeometryBuffers Geometry;
    device const PackedMaterial* Materials;
    device const PackedTextureDescriptor* Textures;
    device const uint* Texels;
    device const PackedEmitter* Sources;
    constant VcmSceneControls* Controls;
};
void VcmReport(VcmScene scene, uint error)
{
    atomic_fetch_or_explicit(scene.Geometry.Error, error, memory_order_relaxed);
}
VcmWideVector VcmScaleColor(VcmWideVector color, VcmNumber factor)
{
    return {WideMath::Multiply(color.X, factor), WideMath::Multiply(color.Y, factor),
            WideMath::Multiply(color.Z, factor)};
}
VcmWideVector VcmMultiplyColor(VcmWideVector a, VcmWideVector b)
{
    return {WideMath::Multiply(a.X, b.X), WideMath::Multiply(a.Y, b.Y), WideMath::Multiply(a.Z, b.Z)};
}
bool VcmBlack(VcmWideVector color)
{
    return !VcmPositive(color.X) && !VcmPositive(color.Y) && !VcmPositive(color.Z);
}
bool VcmNonnegative(float3 value)
{
    return VcmFiniteVector(value) && value.x >= 0 && value.y >= 0 && value.z >= 0;
}
VcmWalkVertex VcmEmptyVertex()
{
    VcmWalkVertex vertex = {};
    vertex.Material = vertex.Primitive = vertex.Source = VcmInvalidIndex;
    vertex.Reserved = VcmInvalidIndex;
    vertex.Color = packed_float3(float3(1));
    return vertex;
}
float VcmMeanFreePath(VcmScene scene, uint material)
{
    return scene.Materials[material].Optical.w;
}
float3 VcmInteriorAlbedo(VcmScene scene, uint material)
{
    PackedMaterial medium = scene.Materials[material];
    return medium.AlbedoReflectivity.xyz * medium.ConductorExtinction.w;
}
float VcmAnisotropy(VcmScene scene, uint material)
{
    return scene.Materials[material].ConductorIndex.w;
}
uint VcmSegmentInterior(VcmScene scene, VcmWalkVertex source, VcmWalkVertex target)
{
    if (source.Kind == VcmVolumeVertex) return source.Material;
    if (target.Kind == VcmVolumeVertex) return target.Material;
    if (source.Kind == VcmEmitterVertex && source.Reserved != VcmInvalidIndex &&
        VcmMeanFreePath(scene, source.Reserved) > 0)
        return source.Reserved;
    if (target.Kind == VcmEmitterVertex && target.Reserved != VcmInvalidIndex &&
        VcmMeanFreePath(scene, target.Reserved) > 0)
        return target.Reserved;
    const float3 edge = float3(target.Position) - float3(source.Position);
    if (source.Kind == VcmSurfaceVertex && VcmMeanFreePath(scene, source.Material) > 0 &&
        dot(float3(source.Geometry), edge) < 0)
        return source.Material;
    if (target.Kind == VcmSurfaceVertex && VcmMeanFreePath(scene, target.Material) > 0 &&
        dot(float3(target.Geometry), -edge) < 0)
        return target.Material;
    return VcmInvalidIndex;
}
VcmNumber VcmSegmentTransmittance(VcmScene scene, VcmWalkVertex source, VcmWalkVertex target)
{
    const uint material = VcmSegmentInterior(scene, source, target);
    if (material == VcmInvalidIndex) return VcmOne();
    const float optical = CameraLength(float3(target.Position) - float3(source.Position)) /
        VcmMeanFreePath(scene, material);
    if (!(optical >= 0) || !VcmFiniteScalar(optical)) return VcmZero();
    const float binary = -optical * 1.44269504088896340736f;
    if (binary < -float(VcmDensityExponentLimit)) return VcmZero();
    const int exponent = int(floor(binary));
    return WideMath::Shift(WideMath::FromFloat(exp2(binary - float(exponent))), exponent);
}
VcmNumber VcmEdgeArrival(VcmScene scene, VcmNumber angular, VcmWalkVertex source, VcmWalkVertex target)
{
    if (!VcmUsableDensity(angular)) return VcmZero();
    const float3 edge = float3(target.Position) - float3(source.Position);
    VcmNumber density = VcmZero();
    if (target.Kind == VcmVolumeVertex)
    {
        const VcmWideVector displacement = VcmDecodeVector(edge);
        const VcmNumber squared = VcmDot(displacement, displacement);
        if (!VcmPositive(squared)) return VcmZero();
        density = WideMath::Divide(angular, squared);
    }
    else
        density = VcmSurfaceArrival(angular, edge, float3(target.Geometry));
    const uint material = VcmSegmentInterior(scene, source, target);
    if (material == VcmInvalidIndex)
        return target.Kind == VcmVolumeVertex ? VcmZero() : density;
    density = WideMath::Multiply(density, VcmSegmentTransmittance(scene, source, target));
    if (target.Kind == VcmVolumeVertex)
        density = WideMath::Divide(density, VcmDensityFromFloat(VcmMeanFreePath(scene, material)));
    return density;
}
CameraEndpointFrame VcmCamera(VcmScene scene)
{
    constant VcmSceneControls& c = *scene.Controls;
    return {float3(c.CameraPosition),
            float3(c.CameraForward),
            float3(c.CameraRight),
            float3(c.CameraUp),
            c.HalfWidth,
            c.HalfHeight,
            c.LensRadius,
            c.FocusDistance};
}
EmissionDomain VcmDomain(VcmScene scene)
{
    return {float3(scene.Controls->EmissionCenter), scene.Controls->EmissionRadius};
}
SampleContext VcmSamples(VcmScene scene, VcmWalkDispatch dispatch, uint identity, uint mode)
{
    constant VcmSceneControls& c = *scene.Controls;
    if (mode == TransportImportance)
    {
        // One globally stratified light population per pass, independent of
        // cache slot, dispatch size, camera pixel and command ordering.
        return {dispatch.Pass,
                c.Seed,
                Hash(c.Seed ^ VcmLightStream ^ dispatch.Pass),
                identity,
                dispatch.LightPopulation,
                c.Sequence,
                c.SamplingSeed};
    }
    SampleContext samples = MakeSampleContext(identity % c.FullWidth, identity / c.FullWidth, c.FullWidth,
                                              dispatch.Pass, c.TotalSamples, c.Sequence, c.SamplingSeed);
    samples.PixelSeed ^= Hash(c.Seed ^ VcmCameraStream);
    return samples;
}
float VcmLinearChannel(uint value)
{
    const float ByteMaximum = 255, SrgbBreakpoint = 0.04045f, SrgbLinearSlope = 12.92f;
    const float SrgbOffset = 0.055f, SrgbScale = 1.055f, SrgbPower = 2.4f;
    float encoded = float(value) / ByteMaximum;
    return encoded <= SrgbBreakpoint ? encoded / SrgbLinearSlope
                                     : pow((encoded + SrgbOffset) / SrgbScale, SrgbPower);
}
float3 VcmTextureColor(VcmScene scene, HitData hit)
{
    uint textureIndex = scene.Materials[hit.MaterialIndex].DiffuseTextureIndex;
    if (!hit.HasTexCoord || textureIndex >= scene.Controls->TextureCount || !isfinite(hit.TexCoord.x) ||
        !isfinite(hit.TexCoord.y))
        return hit.Color;
    PackedTextureDescriptor texture = scene.Textures[textureIndex];
    uint available = scene.Controls->TexelCount;
    if (!texture.Width || !texture.Height || texture.OffsetTexels > available ||
        texture.Height > (available - texture.OffsetTexels) / texture.Width)
    {
        VcmReport(scene, VcmBoundsError);
        return float3(0);
    }
    float2 point =
        float2(fract(hit.TexCoord.x) * texture.Width, fract(1 - hit.TexCoord.y) * texture.Height) - 0.5f;
    float2 fraction = float2(fract(point.x), fract(point.y));
    float3 result = float3(0);
    const uint ChannelMask = 0xff, ChannelBits = 8;
    for (uint row = 0; row < 2; ++row)
        for (uint column = 0; column < 2; ++column)
        {
            // Wrapped bilinear support; integer coordinates never exceed one
            // texel outside [0,size) after reducing UVs to a fractional period.
            int x = int(floor(point.x)) + int(column), y = int(floor(point.y)) + int(row);
            uint wrappedX = x < 0 ? texture.Width - 1 : uint(x) % texture.Width;
            uint wrappedY = y < 0 ? texture.Height - 1 : uint(y) % texture.Height;
            uint encoded = scene.Texels[texture.OffsetTexels + wrappedY * texture.Width + wrappedX];
            float3 color = float3(VcmLinearChannel(encoded & ChannelMask),
                                  VcmLinearChannel((encoded >> ChannelBits) & ChannelMask),
                                  VcmLinearChannel((encoded >> (ChannelBits * 2)) & ChannelMask));
            result += color * ((column ? fraction.x : 1 - fraction.x) * (row ? fraction.y : 1 - fraction.y));
        }
    return result * hit.Color;
}
// The nearest analytic rectangle is an opaque endpoint on either side. Its
// emitted radiance is one-sided, which the later estimator evaluates separately.
// Only an explicitly marked proxy permits the small coincidence tolerance.
VcmWalkVertex VcmIntersect(VcmScene scene, RayData ray, thread VcmWalkStatistics& statistics)
{
    HitData hit = {};
    hit.Distance = ray.MaximumDistance;
    hit.Color = float3(1);
    bool surface = IntersectScene(scene.Geometry, ray, false, hit, statistics.Tests);
    if (surface && hit.MaterialIndex >= scene.Controls->MaterialCount)
    {
        VcmReport(scene, VcmBoundsError);
        return VcmEmptyVertex();
    }
    bool proxy = surface && scene.Materials[hit.MaterialIndex].SampledEmitter != 0;
    float limit = hit.Distance + (proxy ? RayEpsilon : 0);
    uint rectangle = VcmInvalidIndex;
    for (uint index = 0; index < scene.Controls->SourceCount; ++index)
    {
        PackedEmitter source = scene.Sources[index];
        if (source.Kind != PackedEmitterRectangle)
            continue;
        float3 normal = VcmUnitVector(float3(source.Direction));
        float denominator = dot(ray.Direction, normal);
        if (denominator == 0)
            continue;
        float distance = dot(float3(source.Origin) - ray.Origin, normal) / denominator;
        if (!(distance > ray.MinimumDistance && distance < ray.MaximumDistance && distance <= limit))
            continue;
        float3 tangent, bitangent;
        EmissionFrame(normal, tangent, bitangent);
        float3 relative = (ray.Origin - float3(source.Origin)) + ray.Direction * distance;
        if (abs(dot(relative, tangent)) > source.Width / 2 ||
            abs(dot(relative, bitangent)) > source.Height / 2)
            continue;
        if (rectangle != VcmInvalidIndex && distance == limit)
            continue; // Stable catalog order resolves coincident source surfaces.
        rectangle = index;
        limit = distance;
    }
    VcmWalkVertex vertex = VcmEmptyVertex();
    vertex.Incoming = packed_float3(-ray.Direction);
    if (surface && !hit.FrontFace && VcmMeanFreePath(scene, hit.MaterialIndex) > 0)
        vertex.Reserved = hit.MaterialIndex;
    if (rectangle != VcmInvalidIndex)
    {
        vertex.Kind = VcmEmitterVertex;
        vertex.Source = rectangle;
        vertex.Position = packed_float3(ray.Origin + ray.Direction * limit);
        vertex.Geometry = packed_float3(VcmUnitVector(float3(scene.Sources[rectangle].Direction)));
        vertex.FrontFace = dot(float3(vertex.Geometry), ray.Direction) < 0;
        vertex.Normal = packed_float3(vertex.FrontFace ? float3(vertex.Geometry) : -float3(vertex.Geometry));
    }
    else if (surface)
    {
        vertex.Kind = VcmSurfaceVertex;
        vertex.Position = packed_float3(ray.Origin + ray.Direction * hit.Distance);
        vertex.Normal = packed_float3(hit.Normal);
        vertex.Geometry = packed_float3(hit.GeometricNormal);
        vertex.FrontFace = hit.FrontFace;
        vertex.Material = hit.MaterialIndex;
        vertex.Primitive = hit.PrimitiveIndex;
        vertex.Color = packed_float3(VcmTextureColor(scene, hit));
        vertex.TexCoord = hit.TexCoord;
        vertex.HasTexCoord = hit.HasTexCoord;
        for (uint index = 0; index < scene.Controls->SourceCount; ++index)
            if (scene.Sources[index].Primitive == hit.PrimitiveIndex)
            {
                vertex.Source = index;
                break;
            }
    }
    else
    {
        vertex.Kind = VcmEscapeVertex;
        vertex.Position = packed_float3(ray.Direction); // Angular endpoint, not a finite point.
        for (uint index = 0; index < scene.Controls->SourceCount; ++index)
            if (scene.Sources[index].Kind == PackedEmitterEnvironment)
            {
                vertex.Source = index;
                break;
            }
    }
    return vertex;
}
SurfaceBsdf VcmScatteringFrame(VcmScene scene, VcmWalkVertex vertex, float3 view, uint mode)
{
    HitData hit = {};
    hit.Normal = dot(float3(vertex.Normal), view) >= 0 ? float3(vertex.Normal) : -float3(vertex.Normal);
    hit.GeometricNormal = float3(vertex.Geometry);
    hit.FrontFace = dot(hit.GeometricNormal, view) > 0;
    PackedMaterial material = scene.Materials[vertex.Material];
    if (material.Model == MaterialLegacy)
        material.Model = MaterialDiffuse;
    return MakeSurfaceBsdf(material, hit, view, float3(vertex.Color), scene.Controls->TransportFlags, mode);
}
SurfaceSample VcmScatter(VcmScene scene, VcmWalkVertex vertex, uint mode, float2 direction, float branch)
{
    if (vertex.Kind == VcmVolumeVertex)
    {
        const SubsurfaceDirection phase = SubsurfaceSampleDirection(-float3(vertex.Incoming),
            VcmAnisotropy(scene, vertex.Material), direction,
            (scene.Controls->TransportFlags & ImportanceBit) != 0);
        return {phase.Direction, VcmInteriorAlbedo(scene, vertex.Material) * phase.Weight,
                phase.Pdf, phase.Pdf, 1, ScatteringGlossy};
    }
    SurfaceBsdf surface = VcmScatteringFrame(scene, vertex, float3(vertex.Incoming), mode);
    PackedMaterial material = scene.Materials[vertex.Material];
    bool legacy = material.Model == MaterialLegacy;
    float mirror = legacy ? clamp(material.AlbedoReflectivity.w, 0.0f, 1.0f) : 0;
    if (legacy && branch < mirror)
    {
        float3 outgoing = VcmUnitVector(reflect(-float3(vertex.Incoming), surface.Normal));
        float cosine = dot(surface.Normal, outgoing);
        if (!(cosine > 0) || !SurfaceDirectionValid(surface, outgoing, cosine))
            return {};
        return {outgoing, float3(SurfaceAdjointCorrection(surface, outgoing)),
                mirror,   mirror,
                1,        ScatteringReflection | ScatteringDelta};
    }
    SurfaceSample sampled = SampleSurface(surface, direction, branch);
    if (legacy)
    {
        // Selection mass appears in both PDFs; the physical mixture weight
        // cancels it in throughput. The diffuse branch has no secondary lobe.
        sampled.Pdf *= 1 - mirror;
        sampled.ReversePdf *= 1 - mirror;
    }
    return sampled;
}
VcmNumber VcmBackArrival(VcmScene scene, VcmWalkVertex previous, VcmWalkVertex current,
                         VcmNumber reverse)
{
    if (previous.Kind == VcmEscapeVertex)
        return reverse;
    return VcmEdgeArrival(scene, reverse, current, previous);
}
void VcmExtend(VcmScene scene, SampleContext samples, RayData ray, VcmWideVector throughput,
               device VcmWalkVertex* vertices, thread VcmWalk& path, thread VcmWalkStatistics& statistics)
{
    uint interior = VcmInvalidIndex;
    for (uint bounce = 0; bounce <= scene.Controls->MaxReflections; ++bounce)
    {
        if (path.Mode == TransportRadiance && bounce == 0)
            IncrementCounter(statistics.Camera);
        else
            IncrementCounter(statistics.Transport);
        VcmWalkVertex previous = vertices[path.Begin + path.Count - 1];
        VcmWalkVertex vertex = VcmIntersect(scene, ray, statistics);
        if (atomic_load_explicit(scene.Geometry.Error, memory_order_relaxed) != 0)
        {
            path.End = VcmEndError;
            return;
        }
        if (path.Count == 1 && vertex.Reserved != VcmInvalidIndex)
            interior = vertex.Reserved;
        if (interior != VcmInvalidIndex)
        {
            const float endpoint = vertex.Kind == VcmEscapeVertex ? MaximumRayDistance :
                CameraLength(float3(vertex.Position) - ray.Origin);
            const SubsurfaceFlight flight = SubsurfaceFreeFlight(VcmMeanFreePath(scene, interior),
                VcmInteriorAlbedo(scene, interior), ray.Direction, endpoint,
                Sample2D(samples, VcmInteriorFlightDomain ^ ((bounce + 1) * BounceSeedMultiplier)).x);
            if (flight.Scattered)
            {
                vertex.Kind = VcmVolumeVertex;
                vertex.Position = packed_float3(ray.Origin + ray.Direction * flight.Distance);
                vertex.Material = interior;
                vertex.Primitive = vertex.Source = VcmInvalidIndex;
                vertex.Normal = vertex.Geometry = packed_float3(float3(0));
                vertex.FrontFace = 0;
            }
        }
        if (!VcmFiniteVector(float3(vertex.Position)) || !VcmFiniteVector(float3(vertex.Normal)) ||
            !VcmFiniteVector(float3(vertex.Geometry)) || !VcmNonnegative(float3(vertex.Color)))
        {
            VcmReport(scene, VcmNumericalError);
            path.End = VcmEndError;
            return;
        }
        vertex.Throughput = throughput;
        if (vertex.Kind == VcmEscapeVertex)
            vertex.Arrival = previous.ForwardAngular;
        else if (path.Count == 1 && path.Mode == TransportImportance &&
                 (path.EndpointFlags & EmissionInfinite) != 0)
            vertex.Arrival =
                VcmProjectedDiskArrival(path.PositionPdf, float3(vertex.Geometry), ray.Direction);
        else
            vertex.Arrival = VcmEdgeArrival(scene, previous.ForwardAngular, previous, vertex);
        uint output = path.Begin + path.Count++;
        vertices[output] = vertex;
        if (vertex.Kind == VcmEscapeVertex || vertex.Kind == VcmEmitterVertex)
        {
            path.End = vertex.Kind == VcmEscapeVertex ? VcmEndEscape : VcmEndSource;
            return;
        }
        if (bounce == scene.Controls->MaxReflections)
        {
            path.End = VcmEndDepth;
            return;
        }
        uint dimension = (bounce + 1) * BounceSeedMultiplier;
        SurfaceSample event = VcmScatter(scene, vertex, path.Mode,
                                         Sample2D(samples, dimension ^
                                             (vertex.Kind == VcmVolumeVertex ? VcmInteriorPhaseDomain : 0)),
                                         Sample2D(samples, dimension ^ VcmBranchDomain).x);
        if (event.Event == 0)
        {
            path.End = VcmEndNull;
            return;
        }
        if (!VcmNonnegative(event.Weight) || !VcmFiniteVector(event.Direction) ||
            !VcmUsableDensity(VcmDensityFromFloat(event.Pdf)) || !VcmFiniteScalar(event.ReversePdf) ||
            event.ReversePdf < 0)
        {
            VcmReport(scene, VcmNumericalError);
            path.End = VcmEndError;
            return;
        }
        vertex.Outgoing = packed_float3(event.Direction);
        vertex.Event = event.Event;
        vertex.ForwardAngular = VcmDensityFromFloat(event.Pdf);
        vertex.ReverseAngular = VcmDensityFromFloat(event.ReversePdf);
        vertices[output] = vertex;
        vertices[output - 1].ReverseArrival = VcmBackArrival(scene, previous, vertex, vertex.ReverseAngular);
        throughput = VcmMultiplyColor(throughput, VcmDecodeVector(event.Weight));
        if (VcmBlack(throughput))
        {
            path.End = VcmEndAbsorbed;
            return;
        }
        float3 origin = float3(vertex.Position);
        if (vertex.Kind == VcmSurfaceVertex)
        {
            SurfaceBsdf surface = VcmScatteringFrame(scene, vertex, float3(vertex.Incoming), path.Mode);
            origin = SurfaceOffset(surface, origin, event.Direction);
            if (VcmMeanFreePath(scene, vertex.Material) > 0 &&
                (event.Event & ScatteringTransmission) != 0)
                interior = vertex.FrontFace ? vertex.Material : VcmInvalidIndex;
        }
        ray = {origin, event.Direction, RayEpsilon, MaximumRayDistance};
    }
}
void VcmGenerateWalk(VcmScene scene, VcmWalkDispatch dispatch, uint lane, uint mode,
                     device VcmWalkVertex* vertices, device VcmWalk* paths,
                     device VcmWalkStatistics* statistics)
{
    if (lane >= dispatch.PathCount)
        return;
    constant VcmSceneControls& c = *scene.Controls;
    // Check extents before every resource-relative index is formed. A cache
    // slot is never the stochastic identity of its path.
    if (c.MaxReflections > VcmMaximumReflections)
    {
        VcmReport(scene, VcmControlError);
        return;
    }
    if (dispatch.OutputPathBegin > dispatch.PathCapacity ||
        lane >= dispatch.PathCapacity - dispatch.OutputPathBegin || lane >= dispatch.StatisticsCapacity ||
        dispatch.VertexStride < c.MaxReflections + 2 ||
        dispatch.OutputVertexBegin > dispatch.VertexCapacity ||
        lane >= (dispatch.VertexCapacity - dispatch.OutputVertexBegin) / dispatch.VertexStride ||
        c.SourceCount > dispatch.SourceCapacity)
    {
        VcmReport(scene, VcmBoundsError);
        return;
    }
    uint slot = dispatch.OutputPathBegin + lane;
    VcmWalk path = {};
    path.Begin = dispatch.OutputVertexBegin + lane * dispatch.VertexStride;
    path.Mode = mode;
    path.Source = VcmInvalidIndex;
    path.Pass = dispatch.Pass;
    paths[slot] = path;
    statistics[lane] = {};
    if (c.Sequence > BlueNoiseSequence || c.TotalSamples == 0 || c.SourceCount > MaximumUniformSources ||
        (c.TransportFlags & VcmRouletteBit) != 0 || dispatch.Pass >= c.TotalSamples || c.FullWidth == 0 ||
        c.FullHeight == 0 || c.FullHeight > VcmInvalidIndex / c.FullWidth ||
        dispatch.FirstPath > VcmInvalidIndex - lane)
    {
        VcmReport(scene, VcmControlError);
        return;
    }
    uint population = mode == TransportRadiance ? c.FullWidth * c.FullHeight : dispatch.LightPopulation;
    path.Identity = dispatch.FirstPath + lane;
    if (path.Identity >= population)
    {
        VcmReport(scene, VcmControlError);
        return;
    }
    SampleContext samples = VcmSamples(scene, dispatch, path.Identity, mode);
    VcmWalkVertex endpoint = VcmEmptyVertex();
    VcmWideVector throughput = {};
    RayData ray = {};
    ray.MinimumDistance = RayEpsilon;
    ray.MaximumDistance = MaximumRayDistance;
    if (mode == TransportRadiance)
    {
        float2 raster = CameraPixelRaster(path.Identity % c.FullWidth, path.Identity / c.FullWidth,
                                          c.FullWidth, c.FullHeight, Sample2D(samples, VcmRasterDomain));
        CameraEndpointSample camera =
            SampleCameraEndpoint(VcmCamera(scene), raster, Sample2D(samples, VcmLensDomain));
        if ((camera.Flags & CameraEndpointValid) == 0)
        {
            path.End = VcmEndNull;
            paths[slot] = path;
            return;
        }
        path.Raster = camera.Raster;
        path.LensPoint = camera.LensPoint;
        path.PositionPdf = VcmDensityFromFloat(camera.PositionPdf);
        path.DirectionPdf = VcmDensityFromFloat(camera.DirectionPdf);
        path.Selection = VcmOne();
        path.EndpointFlags = camera.Flags;
        endpoint.Kind = VcmCameraVertex;
        endpoint.Position = packed_float3(camera.Origin);
        endpoint.Normal = endpoint.Geometry = c.CameraForward;
        endpoint.Arrival = path.PositionPdf;
        endpoint.Throughput = VcmDecodeVector(float3(camera.Weight));
        ray.Origin = camera.Origin;
        ray.Direction = camera.Direction;
        throughput = endpoint.Throughput;
    }
    else
    {
        EmitterEndpoint source = SampleEmitterEndpoint(
            scene.Sources, c.SourceCount, VcmDomain(scene), Sample2D(samples, VcmSourceDomain).x,
            Sample2D(samples, VcmPositionDomain), Sample2D(samples, VcmDirectionDomain));
        if ((source.Emission.Flags & EmissionValid) == 0)
        {
            path.End = VcmEndNull;
            paths[slot] = path;
            return;
        }
        EmissionSample emission = source.Emission;
        path.Source = source.EmitterIndex;
        path.EndpointFlags = emission.Flags;
        path.PositionPdf = VcmDensityFromFloat(emission.PositionPdf);
        path.DirectionPdf = VcmDensityFromFloat(emission.DirectionPdf);
        path.Selection = VcmDensityFromFloat(source.SelectionProbability);
        if (!VcmNonnegative(emission.Value) || !VcmUsableDensity(path.PositionPdf) ||
            !VcmUsableDensity(path.DirectionPdf) || !VcmUsableDensity(path.Selection))
        {
            VcmReport(scene, VcmNumericalError);
            path.End = VcmEndError;
            paths[slot] = path;
            return;
        }
        bool infinite = (emission.Flags & EmissionInfinite) != 0;
        endpoint.Kind = infinite ? VcmEscapeVertex : VcmEmitterVertex;
        endpoint.Position = packed_float3(emission.Origin);
        endpoint.Normal = endpoint.Geometry = packed_float3(emission.Normal);
        endpoint.Source = source.EmitterIndex;
        endpoint.Primitive = source.Primitive;
        endpoint.Arrival =
            WideMath::Multiply(path.Selection, infinite ? path.DirectionPdf : path.PositionPdf);
        endpoint.Throughput =
            VcmScaleColor(VcmDecodeVector(emission.Value), WideMath::Divide(VcmOne(), endpoint.Arrival));
        VcmNumber launch = WideMath::Divide(VcmOne(), infinite ? path.PositionPdf : path.DirectionPdf);
        if (!infinite && (emission.Flags & EmissionDeltaPosition) == 0)
            launch = WideMath::Multiply(launch,
                                        VcmDensityFromFloat(abs(dot(emission.Normal, emission.Direction))));
        throughput = VcmScaleColor(endpoint.Throughput, launch);
        ray.Origin = emission.Origin;
        if (!infinite && (emission.Flags & EmissionDeltaPosition) == 0)
            ray.Origin +=
                emission.Normal * (dot(emission.Normal, emission.Direction) >= 0 ? RayEpsilon : -RayEpsilon);
        ray.Direction = emission.Direction;
    }
    endpoint.Outgoing = packed_float3(ray.Direction);
    endpoint.ForwardAngular = path.DirectionPdf;
    vertices[path.Begin] = endpoint;
    path.Count = 1;
    path.LaunchDirection = packed_float3(ray.Direction);
    VcmWalkStatistics local = {};
    if (VcmBlack(throughput))
        path.End = VcmEndAbsorbed;
    else
        VcmExtend(scene, samples, ray, throughput, vertices, path, local);
    paths[slot] = path;
    statistics[lane] = local;
}

struct VcmCouplingControls
{
    VcmNumber Radius, MergingSamples, LightTracingSamples;
    uint MaximumVertices, LightVertexLimit, CameraVertexLimit, Heuristic;
    uint Families, NextEventEstimation, PathCapacity, VertexCapacity;
    uint SourceCapacity, QueryCount, QueryBegin, OutputCapacity;
    uint QueryCapacity, StatisticsCapacity, Reserved0, Reserved1;
};
struct VcmCouplingQuery
{
    uint LightPath, CameraPath, LightCount, CameraCount;
    uint Merge, Reserved;
    float2 LensVariates;
};
struct VcmContribution
{
    VcmWideVector Radiance;
    VcmNumber Weight;
    float2 Raster;
    uint Splat, Accepted;
};
static_assert(sizeof(VcmCouplingControls) == 112 && sizeof(VcmCouplingQuery) == 32 &&
                  sizeof(VcmContribution) == 80,
              "Metal VCM coupling ABI mismatch");
// A complete path is a view of two immutable caches. At a merge, the incoming
// light edge ends at the photon position and the camera edge ends at the camera
// surface. Collapsing both into one geometric point changes their actual PDFs.
struct VcmJoinedPath
{
    device const VcmWalkVertex* Light;
    device const VcmWalkVertex* Camera;
    uint S, T, Count, Merge;
    VcmWalkVertex Sensor;
    uint FreshSensor;
    float3 Bridge;
    VcmNumber CameraPositionPdf, CameraDirectionPdf;
};
VcmWalkVertex VcmJoinedVertex(VcmJoinedPath path, uint index, bool lightSide = false)
{
    if (path.FreshSensor && index + 1 == path.Count)
        return path.Sensor;
    uint lightEnd = path.S - (path.Merge ? 1 : 0);
    if (index < lightEnd || (path.Merge && lightSide && index == lightEnd))
        return path.Light[index];
    return path.Camera[path.Count - 1 - index];
}
float3 VcmTowardLight(VcmJoinedPath path, uint index)
{
    if (index == 0)
        return float3(0);
    if (index < path.S)
        return float3(path.Light[index].Incoming);
    if (!path.Merge && index == path.S)
        return -path.Bridge;
    return float3(VcmJoinedVertex(path, index).Outgoing);
}
float3 VcmTowardCamera(VcmJoinedPath path, uint index)
{
    if (index + 1 == path.Count)
        return float3(0);
    if (index + 1 < path.S)
        return float3(path.Light[index].Outgoing);
    if (!path.Merge && path.S && index + 1 == path.S)
        return path.Bridge;
    return float3(VcmJoinedVertex(path, index).Incoming);
}
SurfaceEvaluation VcmEvaluateContinuous(VcmScene scene, VcmWalkVertex vertex, float3 view, float3 outgoing,
                                        uint mode)
{
    if (vertex.Kind == VcmVolumeVertex)
    {
        const float phase = SubsurfacePhase(dot(-view, outgoing),
                                            VcmAnisotropy(scene, vertex.Material));
        const float pdf = (scene.Controls->TransportFlags & ImportanceBit) != 0 ?
            phase : 1 / (4 * Pi);
        return {VcmInteriorAlbedo(scene, vertex.Material) * phase, pdf, pdf};
    }
    if (vertex.Material >= scene.Controls->MaterialCount)
        return {};
    SurfaceEvaluation evaluated = EvaluateSurface(VcmScatteringFrame(scene, vertex, view, mode), outgoing);
    PackedMaterial material = scene.Materials[vertex.Material];
    if (material.Model == MaterialLegacy)
    {
        float continuous = 1 - clamp(material.AlbedoReflectivity.w, 0.0f, 1.0f);
        evaluated.Value *= continuous;
        evaluated.Pdf *= continuous;
        evaluated.ReversePdf *= continuous;
    }
    if (!VcmNonnegative(evaluated.Value) || !VcmFiniteScalar(evaluated.Pdf) ||
        !VcmFiniteScalar(evaluated.ReversePdf) || evaluated.Pdf < 0 || evaluated.ReversePdf < 0)
    {
        VcmReport(scene, VcmNumericalError);
        return {};
    }
    return evaluated;
}
struct VcmSourceMeasure
{
    VcmWideVector Value;
    VcmNumber Position, Direction, Selection;
    uint Flags;
};
VcmSourceMeasure VcmEvaluateSource(VcmScene scene, VcmWalkVertex vertex, float3 outgoing)
{
    VcmSourceMeasure measure = {};
    if (vertex.Source >= scene.Controls->SourceCount)
    {
        // An authored emissive proxy outside its analytic support has no light
        // sampling strategy. It can still be evaluated as a material hit.
        if (vertex.Material < scene.Controls->MaterialCount)
            measure.Value = VcmDecodeVector(scene.Materials[vertex.Material].Emission.xyz);
        return measure;
    }
    PackedEmitter source = scene.Sources[vertex.Source];
    EmitterEndpointDensity pdf =
        EmitterEndpointPdf(scene.Sources, scene.Controls->SourceCount, VcmDomain(scene), vertex.Source,
                           float3(vertex.Geometry), outgoing);
    measure.Position = VcmDensityFromFloat((pdf.Flags & EmissionDeltaPosition) ? 1 : pdf.Position);
    measure.Direction = VcmDensityFromFloat((pdf.Flags & EmissionDeltaDirection) ? 1 : pdf.Direction);
    measure.Selection = VcmDensityFromFloat(pdf.SelectionProbability);
    measure.Flags = pdf.Flags;
    measure.Value = VcmDecodeVector(float3(source.Strength));
    if (source.Kind == PackedEmitterRectangle)
    {
        if (dot(float3(source.Direction), outgoing) <= 0)
            measure.Value = {};
        else
            measure.Value = VcmScaleColor(measure.Value, measure.Position);
    }
    else if (source.Kind == PackedEmitterEnvironment)
    {
        // A visible environment has an angular measure even in an empty scene,
        // where no positive launch disk exists for light-path generation.
        measure.Flags = EmissionValid | EmissionInfinite;
        measure.Direction = VcmDensityFromFloat(
            EmissionGradientPdf(EmissionGradientWeights(EndpointAnalytic(source)), -outgoing.y));
        float blend = clamp((1 - outgoing.y) / 2, 0.0f, 1.0f);
        measure.Value = VcmDecodeVector(float3(source.BackgroundBottom) * (1 - blend) +
                                        float3(source.BackgroundTop) * blend);
    }
    return measure;
}
VcmWeights VcmJoinedWeights(VcmScene scene, VcmJoinedPath path, VcmSourceMeasure source,
                            VcmCouplingControls controls)
{
    VcmVertexPdf densities[VcmMaximumVertices] = {};
    VcmStrategy connections[VcmMaximumStrategies] = {}, mergings[VcmMaximumStrategies] = {};
    const bool infinite = (source.Flags & EmissionInfinite) != 0;
    densities[0].Forward = source.Direction;
    if ((source.Flags & (EmissionDeltaPosition | EmissionDeltaDirection)) != 0)
        densities[0].Flags = VcmNotIntersectable;
    densities[path.Count - 1].Reverse = path.CameraDirectionPdf;
    densities[path.Count - 1].Flags = VcmNotIntersectable; // No random lens-hit estimator.
    for (uint i = 1; i + 1 < path.Count; ++i)
    {
        VcmWalkVertex vertex = VcmJoinedVertex(path, i);
        if (vertex.Kind == VcmVolumeVertex)
            densities[i].Flags |= VcmVolumeScattering;
        bool originalLight = i + 1 < path.S;
        bool originalCamera = i >= path.S && path.Count - i < path.T;
        if (originalLight || originalCamera)
        {
            // Unchanged events keep sampled Snell/mirror directions and branch
            // masses. Reconstructing them from offset hit positions is incorrect.
            densities[i].Forward = originalLight ? vertex.ForwardAngular : vertex.ReverseAngular;
            densities[i].Reverse = originalLight ? vertex.ReverseAngular : vertex.ForwardAngular;
            if ((vertex.Event & ScatteringDelta) != 0)
                densities[i].Flags |= VcmDelta;
        }
        else
        {
            SurfaceEvaluation evaluated = VcmEvaluateContinuous(scene, vertex, VcmTowardCamera(path, i),
                                                                VcmTowardLight(path, i), TransportRadiance);
            densities[i].Forward = VcmDensityFromFloat(evaluated.ReversePdf);
            densities[i].Reverse = VcmDensityFromFloat(evaluated.Pdf);
        }
    }
    // Convert in opposite iteration orders so each neighboring entry still
    // contains its angular PDF when consumed. No extra angular arrays are needed.
    for (uint i = path.Count - 1; i > 0; --i)
    {
        VcmWalkVertex to = VcmJoinedVertex(path, i, true), from = VcmJoinedVertex(path, i - 1);
        densities[i].Forward = i == 1 && infinite
            ? VcmProjectedDiskArrival(source.Position, float3(to.Geometry), VcmTowardCamera(path, 0))
            : VcmEdgeArrival(scene, densities[i - 1].Forward, from, to);
    }
    for (uint i = 0; i + 1 < path.Count; ++i)
    {
        VcmWalkVertex to = VcmJoinedVertex(path, i), from = VcmJoinedVertex(path, i + 1, true);
        densities[i].Reverse = i == 0 && infinite
            ? densities[i + 1].Reverse
            : VcmEdgeArrival(scene, densities[i + 1].Reverse, from, to);
    }
    for (uint cut = 0; cut < path.Count; ++cut)
    {
        VcmStrategy strategy = {};
        strategy.LightEndpoint = cut == 0 ? VcmOne() : infinite ? source.Direction : source.Position;
        strategy.LightSelection = cut == 0 ? VcmOne() : source.Selection;
        strategy.CameraEndpoint = path.CameraPositionPdf;
        strategy.SampleCount = cut + 1 == path.Count ? controls.LightTracingSamples : VcmOne();
        strategy.Enabled = cut <= controls.LightVertexLimit && path.Count - cut <= controls.CameraVertexLimit;
        if ((cut == 1 && (!controls.NextEventEstimation || path.Count == 2)) ||
            (cut + 1 == path.Count && cut < 2))
            strategy.Enabled = 0;
        connections[cut] = strategy;
        // Merge support depends on n+1 total sampled vertices; do not inherit
        // the NEE/lens-connection enable bit from the connection in this slot.
        strategy.Enabled = cut >= 2 && cut <= controls.LightVertexLimit &&
                           path.Count + 1 - cut <= controls.CameraVertexLimit;
        strategy.SampleCount = controls.MergingSamples;
        mergings[cut] = strategy;
    }
    return VcmJointWeights(densities, path.Count, connections, mergings, controls.Radius, controls.Heuristic,
                           controls.Families);
}
VcmNumber VcmAbsoluteDot(float3 a, float3 b)
{
    return WideMath::Absolute(VcmDot(VcmDecodeVector(a), VcmDecodeVector(b)));
}
bool VcmVisibleSegment(VcmScene scene, VcmWalkVertex start, float3 direction, float3 endpoint, bool infinite,
                       thread VcmWalkStatistics& statistics)
{
    float3 normal = float3(start.Geometry);
    float3 origin =
        float3(start.Position) + normal * (dot(normal, direction) >= 0 ? RayEpsilon : -RayEpsilon);
    RayData ray = {origin, direction, RayEpsilon, MaximumRayDistance};
    if (!infinite)
    {
        float3 displacement = endpoint - origin;
        float distance = CameraLength(displacement);
        if (!(distance > 2 * RayEpsilon) || !VcmFiniteScalar(distance))
            return false;
        ray.Direction = VcmUnitVector(displacement);
        ray.MaximumDistance = distance - RayEpsilon;
    }
    IncrementCounter(statistics.Shadow);
    VcmWalkVertex obstruction = VcmIntersect(scene, ray, statistics);
    return obstruction.Kind == VcmEscapeVertex &&
           atomic_load_explicit(scene.Geometry.Error, memory_order_relaxed) == 0;
}
bool VcmMergeSupport(VcmWalkVertex photon, VcmWalkVertex camera, VcmNumber radius)
{
    if (photon.Kind != camera.Kind || !VcmUsableDensity(radius))
        return false;
    float3 separation = float3(photon.Position) - float3(camera.Position);
    if (!VcmFiniteVector(separation))
        return false;
    VcmWideVector difference = VcmDecodeVector(separation);
    if (WideMath::Compare(VcmDot(difference, difference), WideMath::Multiply(radius, radius)) > 0)
        return false;
    if (camera.Kind == VcmVolumeVertex)
        return camera.Material == photon.Material;
    if (camera.Kind != VcmSurfaceVertex)
        return false;
    // Restrict a finite disk to one local sheet. These geometric rejection
    // tolerances are distinct from the radius and never change the kernel area.
    const float NormalAgreement = 0.9f, MaximumNormalOffsetFraction = 0.1f;
    float3 lightNormal = float3(photon.Geometry), cameraNormal = float3(camera.Geometry);
    if (dot(lightNormal, cameraNormal) < NormalAgreement)
        return false;
    VcmNumber thickness = WideMath::Multiply(radius, VcmDensityFromFloat(MaximumNormalOffsetFraction));
    return WideMath::Compare(VcmAbsoluteDot(separation, lightNormal), thickness) <= 0 &&
           WideMath::Compare(VcmAbsoluteDot(separation, cameraNormal), thickness) <= 0;
}
VcmContribution VcmCouple(VcmScene scene, device const VcmWalkVertex* vertices, VcmWalk light, VcmWalk camera,
                          VcmCouplingQuery query, VcmCouplingControls controls,
                          thread VcmWalkStatistics& statistics)
{
    VcmContribution result = {};
    uint s = query.LightCount, t = query.CameraCount;
    if (t == 0 || s + t < 2 || s > VcmMaximumVertices || t > VcmMaximumVertices ||
        (query.Merge && (s < 2 || t < 2)))
        return result;
    VcmJoinedPath path = {};
    path.Light = vertices + light.Begin;
    path.Camera = vertices + camera.Begin;
    path.S = s;
    path.T = t;
    path.Count = s + t - (query.Merge ? 1 : 0);
    path.Merge = query.Merge;
    if (path.Count > controls.MaximumVertices || path.Count > VcmMaximumVertices ||
        s > controls.LightVertexLimit || t > controls.CameraVertexLimit || (s == 1 && t == 1) ||
        (query.Merge ? (controls.Families & VcmMergings) == 0 : (controls.Families & VcmConnections) == 0))
        return result;
    VcmWalkVertex lightEnd = s ? path.Light[s - 1] : VcmEmptyVertex();
    VcmWalkVertex cameraEnd = t > 1 ? path.Camera[t - 1] : VcmEmptyVertex();
    path.CameraPositionPdf = camera.PositionPdf;
    path.CameraDirectionPdf = camera.DirectionPdf;
    result.Raster = camera.Raster;
    VcmWideVector value = {};
    if (t == 1)
    {
        if (s < 2 || (lightEnd.Kind != VcmSurfaceVertex && lightEnd.Kind != VcmVolumeVertex))
            return result;
        CameraConnectionSample sensor =
            SampleCameraConnection(VcmCamera(scene), float3(lightEnd.Position), query.LensVariates);
        if ((sensor.Flags & CameraEndpointValid) == 0)
            return result;
        path.FreshSensor = 1;
        path.Sensor = VcmEmptyVertex();
        path.Sensor.Kind = VcmCameraVertex;
        path.Sensor.Position = packed_float3(sensor.Position);
        path.Sensor.Geometry = path.Sensor.Normal = scene.Controls->CameraForward;
        path.Sensor.Outgoing = packed_float3(-sensor.Direction);
        path.Bridge = sensor.Direction;
        CameraEndpointEvaluation response =
            EvaluateCameraEndpoint(VcmCamera(scene), sensor.LensPoint, -sensor.Direction);
        path.CameraDirectionPdf = VcmDensityFromFloat(response.DirectionPdf);
        path.CameraPositionPdf = VcmDensityFromFloat(sensor.PositionPdf);
        result.Raster = sensor.Raster;
        result.Splat = 1;
        SurfaceEvaluation scattering = VcmEvaluateContinuous(scene, lightEnd, float3(lightEnd.Incoming),
                                                             sensor.Direction, TransportImportance);
        value = VcmMultiplyColor(lightEnd.Throughput, VcmDecodeVector(scattering.Value));
        const VcmNumber cosine = lightEnd.Kind == VcmVolumeVertex ? VcmOne() :
            VcmAbsoluteDot(float3(lightEnd.Normal), sensor.Direction);
        value = VcmScaleColor(value, WideMath::Multiply(cosine,
            WideMath::Multiply(VcmDensityFromFloat(sensor.Weight),
                               VcmSegmentTransmittance(scene, lightEnd, path.Sensor))));
        if (VcmBlack(value) ||
            !VcmVisibleSegment(scene, lightEnd, sensor.Direction, sensor.Position, false, statistics))
            return result;
    }
    else if (s == 0)
    {
        VcmSourceMeasure hit = VcmEvaluateSource(scene, cameraEnd, float3(cameraEnd.Incoming));
        value = VcmMultiplyColor(cameraEnd.Throughput, hit.Value);
        if (VcmBlack(value))
            return result;
    }
    else if (query.Merge)
    {
        if (!VcmUsableDensity(controls.MergingSamples) ||
            !VcmMergeSupport(lightEnd, cameraEnd, controls.Radius))
            return result;
        const bool volume = cameraEnd.Kind == VcmVolumeVertex;
        if (volume && CameraLength(float3(lightEnd.Position) - float3(cameraEnd.Position)) > 2 * RayEpsilon &&
            !VcmVisibleSegment(scene, cameraEnd,
                VcmUnitVector(float3(lightEnd.Position) - float3(cameraEnd.Position)),
                float3(lightEnd.Position), false, statistics))
            return result;
        float3 photonDirection = float3(lightEnd.Incoming);
        SurfaceEvaluation scattering = VcmEvaluateContinuous(scene, cameraEnd, float3(cameraEnd.Incoming),
                                                             photonDirection, TransportRadiance);
        VcmNumber geometryCosine = volume ? VcmOne() :
            VcmAbsoluteDot(float3(cameraEnd.Geometry), photonDirection);
        if (!VcmPositive(geometryCosine))
            return result;
        VcmNumber correction = volume ? VcmOne() :
            WideMath::Divide(VcmAbsoluteDot(float3(cameraEnd.Normal), photonDirection), geometryCosine);
        VcmNumber measure = volume ? VcmSphereVolume(controls.Radius) : VcmDiskArea(controls.Radius);
        VcmNumber normalization = WideMath::Multiply(controls.MergingSamples, measure);
        value = VcmMultiplyColor(VcmMultiplyColor(lightEnd.Throughput, cameraEnd.Throughput),
                                 VcmDecodeVector(scattering.Value));
        value = VcmScaleColor(value, WideMath::Divide(correction, normalization));
        if (VcmBlack(value))
            return result;
    }
    else
    {
        if ((cameraEnd.Kind != VcmSurfaceVertex && cameraEnd.Kind != VcmVolumeVertex) ||
            (s > 1 && lightEnd.Kind != VcmSurfaceVertex && lightEnd.Kind != VcmVolumeVertex) ||
            (s == 1 && !controls.NextEventEstimation))
            return result;
        bool infinite = s == 1 && (light.EndpointFlags & EmissionInfinite) != 0;
        float3 displacement = float3(cameraEnd.Position) - float3(lightEnd.Position);
        path.Bridge = infinite ? float3(lightEnd.Outgoing) : VcmUnitVector(displacement);
        VcmNumber inverseSquared = VcmOne();
        if (!infinite)
        {
            if (!VcmFiniteVector(displacement))
            {
                VcmReport(scene, VcmNumericalError);
                return result;
            }
            VcmWideVector wide = VcmDecodeVector(displacement);
            VcmNumber squared = VcmDot(wide, wide);
            if (!VcmPositive(squared))
                return result;
            inverseSquared = WideMath::Divide(VcmOne(), squared);
        }
        SurfaceEvaluation right = VcmEvaluateContinuous(scene, cameraEnd, float3(cameraEnd.Incoming),
                                                        -path.Bridge, TransportRadiance);
        value = VcmMultiplyColor(cameraEnd.Throughput, VcmDecodeVector(right.Value));
        VcmNumber geometry = WideMath::Multiply(inverseSquared,
            cameraEnd.Kind == VcmVolumeVertex ? VcmOne() :
            VcmAbsoluteDot(float3(cameraEnd.Normal), path.Bridge));
        if (s == 1)
        {
            VcmSourceMeasure source = VcmEvaluateSource(scene, lightEnd, path.Bridge);
            VcmNumber proposal =
                WideMath::Multiply(source.Selection, infinite ? source.Direction : source.Position);
            if (!VcmPositive(proposal))
                return result;
            value =
                VcmMultiplyColor(value, VcmScaleColor(source.Value, WideMath::Divide(VcmOne(), proposal)));
            if (!infinite && (source.Flags & EmissionDeltaPosition) == 0)
                geometry =
                    WideMath::Multiply(geometry, VcmAbsoluteDot(float3(lightEnd.Geometry), path.Bridge));
        }
        else
        {
            SurfaceEvaluation left = VcmEvaluateContinuous(scene, lightEnd, float3(lightEnd.Incoming),
                                                           path.Bridge, TransportImportance);
            value =
                VcmMultiplyColor(value, VcmMultiplyColor(lightEnd.Throughput, VcmDecodeVector(left.Value)));
            geometry = WideMath::Multiply(geometry,
                lightEnd.Kind == VcmVolumeVertex ? VcmOne() :
                VcmAbsoluteDot(float3(lightEnd.Normal), path.Bridge));
        }
        if (!infinite)
            geometry = WideMath::Multiply(geometry,
                VcmSegmentTransmittance(scene, lightEnd, cameraEnd));
        value = VcmScaleColor(value, geometry);
        if (VcmBlack(value) || !VcmVisibleSegment(scene, cameraEnd, -path.Bridge, float3(lightEnd.Position),
                                                  infinite, statistics))
            return result;
    }
    VcmSourceMeasure source =
        VcmEvaluateSource(scene, VcmJoinedVertex(path, 0, true), VcmTowardCamera(path, 0));
    VcmWeights weights = VcmJoinedWeights(scene, path, source, controls);
    result.Weight = query.Merge ? weights.Merging[s] : weights.Connection[s];
    result.Radiance = VcmScaleColor(value, result.Weight);
    result.Accepted = !VcmBlack(result.Radiance);
    // Lens splats remain unnormalized by LightTracingSamples here. The frame
    // averages that population; merge values already include 1/(N_light*pi*r^2).
    return result;
}

// Fixed buffer slots match the planned Metal VCM RHI. Scene resources are
// immutable; each lane owns one path and its entire bounded vertex range.
kernel void ArvoVcmCameraWalks(
    device const PackedAccelerationNode* nodes [[buffer(0)]], device const uint* references [[buffer(1)]],
    device const PackedTriangle* triangles [[buffer(2)]], device const PackedSphere* spheres [[buffer(3)]],
    device const PackedMaterial* materials [[buffer(4)]], device const PackedEmitter* sources [[buffer(5)]],
    constant VcmSceneControls& controls [[buffer(6)]],
    device const PackedTextureDescriptor* textures [[buffer(7)]], device const uint* texels [[buffer(8)]],
    device const InstanceRecord* instances [[buffer(9)]],
    device const InstanceNode* instanceNodes [[buffer(10)]], device atomic_uint* errors [[buffer(11)]],
    constant VcmWalkDispatch& dispatch [[buffer(12)]], device VcmWalkVertex* vertices [[buffer(13)]],
    device VcmWalk* paths [[buffer(14)]], device VcmWalkStatistics* statistics [[buffer(15)]],
    uint lane [[thread_position_in_grid]])
{
    VcmScene scene = {{{nodes, references, triangles, spheres, controls.NodeCount, controls.TriangleCount,
                        controls.Acceleration},
                       instances,
                       instanceNodes,
                       errors,
                       controls.InstanceNodeCount},
                      materials,
                      textures,
                      texels,
                      sources,
                      &controls};
    VcmGenerateWalk(scene, dispatch, lane, TransportRadiance, vertices, paths, statistics);
}
kernel void ArvoVcmLightWalks(
    device const PackedAccelerationNode* nodes [[buffer(0)]], device const uint* references [[buffer(1)]],
    device const PackedTriangle* triangles [[buffer(2)]], device const PackedSphere* spheres [[buffer(3)]],
    device const PackedMaterial* materials [[buffer(4)]], device const PackedEmitter* sources [[buffer(5)]],
    constant VcmSceneControls& controls [[buffer(6)]],
    device const PackedTextureDescriptor* textures [[buffer(7)]], device const uint* texels [[buffer(8)]],
    device const InstanceRecord* instances [[buffer(9)]],
    device const InstanceNode* instanceNodes [[buffer(10)]], device atomic_uint* errors [[buffer(11)]],
    constant VcmWalkDispatch& dispatch [[buffer(12)]], device VcmWalkVertex* vertices [[buffer(13)]],
    device VcmWalk* paths [[buffer(14)]], device VcmWalkStatistics* statistics [[buffer(15)]],
    uint lane [[thread_position_in_grid]])
{
    VcmScene scene = {{{nodes, references, triangles, spheres, controls.NodeCount, controls.TriangleCount,
                        controls.Acceleration},
                       instances,
                       instanceNodes,
                       errors,
                       controls.InstanceNodeCount},
                      materials,
                      textures,
                      texels,
                      sources,
                      &controls};
    VcmGenerateWalk(scene, dispatch, lane, TransportImportance, vertices, paths, statistics);
}
bool VcmCachedPrefix(VcmScene scene, VcmWalk path, uint requested, uint capacity, uint mode)
{
    if (path.Count > VcmMaximumVertices || path.Begin > capacity || path.Count > capacity - path.Begin)
    {
        VcmReport(scene, VcmBoundsError);
        return false;
    }
    if (path.Mode != mode || path.End == VcmEndInvalid || path.End == VcmEndError)
    {
        VcmReport(scene, VcmControlError);
        return false;
    }
    return requested <= path.Count;
}
bool VcmNonnegativeNumber(VcmNumber value)
{
    return VcmUsableDensity(value) || (value.High == 0 && value.Low == 0);
}
kernel void ArvoVcmEvaluateCouplings(
    device const PackedAccelerationNode* nodes [[buffer(0)]], device const uint* references [[buffer(1)]],
    device const PackedTriangle* triangles [[buffer(2)]], device const PackedSphere* spheres [[buffer(3)]],
    device const PackedMaterial* materials [[buffer(4)]], device const PackedEmitter* sources [[buffer(5)]],
    constant VcmSceneControls& controls [[buffer(6)]],
    device const PackedTextureDescriptor* textures [[buffer(7)]], device const uint* texels [[buffer(8)]],
    device const InstanceRecord* instances [[buffer(9)]],
    device const InstanceNode* instanceNodes [[buffer(10)]], device atomic_uint* errors [[buffer(11)]],
    constant VcmCouplingControls& coupling [[buffer(12)]],
    device const VcmWalkVertex* vertices [[buffer(13)]], device const VcmWalk* paths [[buffer(14)]],
    device VcmWalkStatistics* statistics [[buffer(15)]],
    device const VcmCouplingQuery* queries [[buffer(16)]],
    device VcmContribution* contributions [[buffer(17)]], uint lane [[thread_position_in_grid]])
{
    if (lane >= coupling.QueryCount)
        return;
    VcmScene scene = {{{nodes, references, triangles, spheres, controls.NodeCount, controls.TriangleCount,
                        controls.Acceleration},
                       instances,
                       instanceNodes,
                       errors,
                       controls.InstanceNodeCount},
                      materials,
                      textures,
                      texels,
                      sources,
                      &controls};
    if (lane >= coupling.OutputCapacity || lane >= coupling.StatisticsCapacity ||
        coupling.QueryBegin > coupling.QueryCapacity ||
        lane >= coupling.QueryCapacity - coupling.QueryBegin ||
        controls.SourceCount > coupling.SourceCapacity)
    {
        VcmReport(scene, VcmBoundsError);
        return;
    }
    contributions[lane] = {};
    statistics[lane] = {};
    if (coupling.MaximumVertices < 2 || coupling.MaximumVertices > VcmMaximumVertices ||
        coupling.LightVertexLimit > VcmMaximumVertices || coupling.CameraVertexLimit > VcmMaximumVertices ||
        coupling.Heuristic > VcmPower || (coupling.Families & ~(VcmConnections | VcmMergings)) != 0 ||
        !VcmNonnegativeNumber(coupling.Radius) || !VcmNonnegativeNumber(coupling.MergingSamples) ||
        !VcmNonnegativeNumber(coupling.LightTracingSamples))
    {
        VcmReport(scene, VcmControlError);
        return;
    }
    VcmCouplingQuery query = queries[coupling.QueryBegin + lane];
    if (query.Merge > 1)
    {
        VcmReport(scene, VcmControlError);
        return;
    }
    VcmWalk light = {}, camera = {};
    if (query.LightCount)
    {
        if (query.LightPath >= coupling.PathCapacity)
        {
            VcmReport(scene, VcmBoundsError);
            return;
        }
        light = paths[query.LightPath];
        if (!VcmCachedPrefix(scene, light, query.LightCount, coupling.VertexCapacity, TransportImportance))
            return;
    }
    if (query.CameraCount > 1)
    {
        if (query.CameraPath >= coupling.PathCapacity)
        {
            VcmReport(scene, VcmBoundsError);
            return;
        }
        camera = paths[query.CameraPath];
        if (!VcmCachedPrefix(scene, camera, query.CameraCount, coupling.VertexCapacity, TransportRadiance))
            return;
    }
    VcmWalkStatistics work = {};
    VcmContribution value = VcmCouple(scene, vertices, light, camera, query, coupling, work);
    if (atomic_load_explicit(errors, memory_order_relaxed) == 0)
        contributions[lane] = value;
    statistics[lane] = work;
}

// Frame execution keeps the light population global, even for a cropped film.
// A pixel owns its accumulation; no floating-point atomic reduction is used.
// Work resumes after a bounded number of couplings/cell advances. The host must
// finish every camera batch before reusing the spatial index for lens splats,
// and finish a pass before publishing its film or changing the merge radius.
constant uint VcmFrameWorkBudget = 8, VcmGridCoordinateBits = 20;
constant uint VcmGridSide = 3, VcmGridNeighbors = VcmGridSide * VcmGridSide * VcmGridSide;
constant uint VcmCommitted = 0, VcmConnecting = 1, VcmMerging = 2, VcmCameraDone = 3;
constant uint VcmPendingCamera = 1, VcmPendingFilm = 2, VcmSpatialIndex = 0, VcmSplatIndex = 1;
constant uint VcmPairStream = 0x50414952u, VcmLensConnectionDomain = 0x4c434f4eu;
constant uint VcmPairAttempts = 32, VcmUintHalfBits = 16, VcmUintHalfMask = 0xffffu;
struct VcmFrameControls
{
    uint TileX, TileY, Width, Height;
    uint Pass, LightBegin, LightCount, SlotsPerLight;
    uint CameraBegin, CameraCount, IndexCapacity, PixelCapacity;
    uint SplatCapacity, StatisticsCapacity, WorkBegin, WorkCount;
    uint SortWidth, SortStride, SortKind, Reserved;
};
struct VcmIndexEntry
{
    int CellX, CellY, CellZ;
    uint Path, Vertex, Identity, Key, Valid;
};
struct VcmSplat
{
    VcmWideVector Radiance;
    uint Pixel, Identity, Vertex, Valid;
};
struct VcmPixelState
{
    VcmWideVector Sum, Estimate;
    uint CameraPath, LightPath, ConnectionS, ConnectionT;
    uint MergeT, Neighbor, Cursor, Phase;
    uint SplatCursor, CompletedPasses, Pass, Reserved;
};
struct VcmFilmPixel
{
    packed_float3 Mean;
    uint Samples;
};
static_assert(sizeof(VcmFrameControls) == 80 && sizeof(VcmIndexEntry) == 32 && sizeof(VcmSplat) == 64 &&
                  sizeof(VcmPixelState) == 144 && sizeof(VcmFilmPixel) == 16,
              "Metal VCM frame ABI mismatch");
VcmNumber VcmFromUint(uint value)
{
    return WideMath::Add(
        WideMath::Shift(WideMath::FromFloat(float(value >> VcmUintHalfBits)), VcmUintHalfBits),
        WideMath::FromFloat(float(value & VcmUintHalfMask)));
}
VcmWideVector VcmAddColor(VcmWideVector a, VcmWideVector b)
{
    return {WideMath::Add(a.X, b.X), WideMath::Add(a.Y, b.Y), WideMath::Add(a.Z, b.Z)};
}
void VcmFrameError(device atomic_uint* errors, uint flag)
{
    atomic_fetch_or_explicit(errors, flag, memory_order_relaxed);
}
bool VcmFrameValid(constant VcmSceneControls& c, constant VcmCouplingControls& coupling,
                   constant VcmFrameControls& f, device atomic_uint* errors)
{
    if (atomic_load_explicit(errors, memory_order_relaxed))
        return false;
    if (!c.FullWidth || !c.FullHeight || c.FullHeight > VcmInvalidIndex / c.FullWidth || !f.Width ||
        !f.Height || f.TileX > c.FullWidth || f.Width > c.FullWidth - f.TileX || f.TileY > c.FullHeight ||
        f.Height > c.FullHeight - f.TileY || f.Height > f.PixelCapacity / f.Width ||
        f.LightBegin > coupling.PathCapacity || f.LightCount > coupling.PathCapacity - f.LightBegin ||
        f.CameraBegin > coupling.PathCapacity || f.CameraCount > coupling.PathCapacity - f.CameraBegin ||
        c.SourceCount > coupling.SourceCapacity || !f.SlotsPerLight ||
        f.LightCount > f.SplatCapacity / f.SlotsPerLight || f.SplatCapacity > f.IndexCapacity ||
        !f.IndexCapacity || (f.IndexCapacity & (f.IndexCapacity - 1)))
    {
        VcmFrameError(errors, VcmBoundsError);
        return false;
    }
    if (!f.LightCount || f.Pass >= c.TotalSamples || c.MaxReflections > VcmMaximumReflections ||
        f.SlotsPerLight != c.MaxReflections + 1 || coupling.Heuristic > VcmPower ||
        (coupling.Families & ~(VcmConnections | VcmMergings)) || !VcmNonnegativeNumber(coupling.Radius) ||
        !VcmFiniteVector(float3(c.EmissionCenter)) || !isfinite(c.EmissionRadius) || c.EmissionRadius < 0 ||
        (f.CameraCount && f.CameraBegin < f.LightBegin + f.LightCount &&
         f.LightBegin < f.CameraBegin + f.CameraCount))
    {
        VcmFrameError(errors, VcmControlError);
        return false;
    }
    return true;
}
bool VcmFrameWorkValid(constant VcmFrameControls& f, uint capacity, device atomic_uint* errors)
{
    if (f.WorkBegin > capacity || f.WorkCount > capacity - f.WorkBegin)
    {
        VcmFrameError(errors, VcmBoundsError);
        return false;
    }
    return true;
}
VcmCouplingControls VcmFrameCoupling(constant VcmSceneControls& c, VcmCouplingControls coupling,
                                     constant VcmFrameControls& frame)
{
    coupling.MergingSamples = VcmFromUint(frame.LightCount);
    coupling.LightTracingSamples =
        WideMath::Divide(coupling.MergingSamples, VcmFromUint(c.FullWidth * c.FullHeight));
    coupling.MaximumVertices = c.MaxReflections + 3;
    coupling.LightVertexLimit = coupling.CameraVertexLimit = c.MaxReflections + 2;
    return coupling;
}
// Origin zero and a binary cell width make quantization exact for binary32
// positions, including negative subnormals on devices with native FTZ. Very
// distant scenes use coarser cells; resumable queries tolerate their occupancy.
int VcmGridExponent(constant VcmSceneControls& c, VcmNumber radius)
{
    VcmNumber bound = WideMath::Absolute(WideMath::FromFloat(c.EmissionCenter.x));
    VcmNumber y = WideMath::Absolute(WideMath::FromFloat(c.EmissionCenter.y));
    VcmNumber z = WideMath::Absolute(WideMath::FromFloat(c.EmissionCenter.z));
    if (WideMath::Compare(y, bound) > 0)
        bound = y;
    if (WideMath::Compare(z, bound) > 0)
        bound = z;
    bound = WideMath::Add(bound, WideMath::FromFloat(c.EmissionRadius));
    int exponent = VcmPositive(radius) ? radius.Exponent : 0;
    return VcmPositive(bound) ? max(exponent, bound.Exponent - int(VcmGridCoordinateBits)) : exponent;
}
bool VcmGridAxis(float position, int exponent, thread int& cell)
{
    if (!isfinite(position))
        return false;
    VcmNumber coordinate = WideMath::FromFloat(position);
    bool negative = coordinate.High < 0;
    coordinate = WideMath::Absolute(coordinate);
    if (!VcmPositive(coordinate))
    {
        cell = 0;
        return true;
    }
    int shift = coordinate.Exponent - exponent;
    if (shift > int(VcmGridCoordinateBits))
        return false;
    if (shift <= 0)
    {
        cell = negative ? -1 : 0;
        return true;
    }
    float scaled = ldexp(coordinate.High, shift);
    int whole = int(scaled);
    cell = negative ? -whole - (scaled > float(whole) ? 1 : 0) : whole;
    return true;
}
bool VcmGridCell(float3 position, int exponent, thread VcmIndexEntry& entry)
{
    return VcmGridAxis(position.x, exponent, entry.CellX) && VcmGridAxis(position.y, exponent, entry.CellY) &&
           VcmGridAxis(position.z, exponent, entry.CellZ);
}
int VcmCellCompare(VcmIndexEntry a, VcmIndexEntry b)
{
    if (a.CellX != b.CellX)
        return a.CellX < b.CellX ? -1 : 1;
    if (a.CellY != b.CellY)
        return a.CellY < b.CellY ? -1 : 1;
    if (a.CellZ != b.CellZ)
        return a.CellZ < b.CellZ ? -1 : 1;
    return 0;
}
bool VcmIndexBefore(VcmIndexEntry a, VcmIndexEntry b, uint kind)
{
    if (a.Valid != b.Valid)
        return a.Valid > b.Valid;
    if (!a.Valid)
        return false;
    if (kind == VcmSpatialIndex)
    {
        int order = VcmCellCompare(a, b);
        if (order)
            return order < 0;
    }
    else if (a.Key != b.Key)
        return a.Key < b.Key;
    return a.Identity != b.Identity ? a.Identity < b.Identity : a.Vertex < b.Vertex;
}
uint VcmLowerBound(device const VcmIndexEntry* index, uint capacity, VcmIndexEntry key, uint kind)
{
    uint first = 0, count = capacity;
    while (count)
    {
        uint half = count / 2, middle = first + half;
        VcmIndexEntry entry = index[middle];
        bool less =
            entry.Valid && (kind == VcmSpatialIndex ? VcmCellCompare(entry, key) < 0 : entry.Key < key.Key);
        if (less)
        {
            first = middle + 1;
            count -= half + 1;
        }
        else
            count = half;
    }
    return first;
}
bool VcmCropPixel(constant VcmSceneControls& c, constant VcmFrameControls& f, uint identity,
                  thread uint& pixel)
{
    uint x = identity % c.FullWidth, y = identity / c.FullWidth;
    if (x < f.TileX || x - f.TileX >= f.Width || y < f.TileY || y - f.TileY >= f.Height)
        return false;
    pixel = (y - f.TileY) * f.Width + x - f.TileX;
    return true;
}
uint VcmPairLight(uint identity, uint seed, uint pass, uint count)
{
    uint threshold = (0u - count) % count;
    uint state = Hash(identity ^ Hash(seed ^ VcmPairStream) ^ Hash(pass));
    for (uint attempt = 0; attempt < VcmPairAttempts; ++attempt)
    {
        // Rejection avoids modulo bias. No cache address enters this stream.
        uint value = Hash(state + attempt * BounceSeedMultiplier);
        if (value >= threshold)
            return value % count;
    }
    return VcmInvalidIndex;
}
bool VcmFramePath(VcmScene scene, VcmWalk path, uint identity, uint mode,
                  constant VcmCouplingControls& coupling, constant VcmFrameControls& f)
{
    if (!VcmCachedPrefix(scene, path, 0, coupling.VertexCapacity, mode))
        return false;
    if (path.Identity != identity || path.Pass != f.Pass || path.Count > scene.Controls->MaxReflections + 2)
    {
        VcmReport(scene, VcmControlError);
        return false;
    }
    return true;
}

// Slots 0..11 retain the scene ABI. Frame-specific buffers use 12: coupling,
// 13: vertices, 14: paths, 15: statistics, 16: frame, 17: index, 18: pixel state,
// 19: splats, 20: film, 21: pending flags. Unused slots need not be bound.
kernel void ArvoVcmBuildGrid(constant VcmSceneControls& c [[buffer(6)]],
                             device atomic_uint* errors [[buffer(11)]],
                             constant VcmCouplingControls& coupling [[buffer(12)]],
                             device const VcmWalkVertex* vertices [[buffer(13)]],
                             device const VcmWalk* paths [[buffer(14)]],
                             constant VcmFrameControls& f [[buffer(16)]],
                             device VcmIndexEntry* index [[buffer(17)]],
                             uint lane [[thread_position_in_grid]])
{
    if (lane >= f.WorkCount || !VcmFrameValid(c, coupling, f, errors))
        return;
    if (!VcmFrameWorkValid(f, f.IndexCapacity, errors))
        return;
    uint slot = f.WorkBegin + lane;
    index[slot] = {};
    if (slot >= f.LightCount * f.SlotsPerLight)
        return;
    uint ordinal = slot / f.SlotsPerLight, vertex = slot % f.SlotsPerLight + 1;
    VcmWalk path = paths[f.LightBegin + ordinal];
    if (path.Begin > coupling.VertexCapacity || path.Count > coupling.VertexCapacity - path.Begin)
    {
        VcmFrameError(errors, VcmBoundsError);
        return;
    }
    if (path.Mode != TransportImportance || path.Identity != ordinal || path.Pass != f.Pass ||
        path.Count > c.MaxReflections + 2 || path.End == VcmEndInvalid || path.End == VcmEndError)
    {
        VcmFrameError(errors, VcmControlError);
        return;
    }
    if (vertex >= path.Count || (vertices[path.Begin + vertex].Kind != VcmSurfaceVertex &&
                                 vertices[path.Begin + vertex].Kind != VcmVolumeVertex))
        return;
    VcmIndexEntry entry = {};
    if (!VcmGridCell(float3(vertices[path.Begin + vertex].Position), VcmGridExponent(c, coupling.Radius),
                     entry))
    {
        VcmFrameError(errors, VcmNumericalError);
        return;
    }
    entry.Path = f.LightBegin + ordinal;
    entry.Vertex = vertex;
    entry.Identity = ordinal;
    entry.Valid = 1;
    index[slot] = entry;
}
// Batcher (1968), "Sorting networks and their applications". Each invocation
// owns one disjoint compare/exchange pair; a dispatch boundary separates network
// levels. Identity and vertex break ties, making reduction independent of lanes.
kernel void ArvoVcmSortIndex(device atomic_uint* errors [[buffer(11)]],
                             constant VcmFrameControls& f [[buffer(16)]],
                             device VcmIndexEntry* index [[buffer(17)]],
                             uint lane [[thread_position_in_grid]])
{
    if (lane >= f.WorkCount)
        return;
    if (!f.IndexCapacity || (f.IndexCapacity & (f.IndexCapacity - 1)) || f.WorkBegin > f.IndexCapacity ||
        f.WorkCount > f.IndexCapacity - f.WorkBegin || f.SortWidth < 2 || f.SortWidth > f.IndexCapacity ||
        (f.SortWidth & (f.SortWidth - 1)) || !f.SortStride || f.SortStride >= f.SortWidth ||
        (f.SortStride & (f.SortStride - 1)) || f.SortKind > VcmSplatIndex)
    {
        VcmFrameError(errors, VcmControlError);
        return;
    }
    uint i = f.WorkBegin + lane, partner = i ^ f.SortStride;
    if (i >= partner)
        return;
    VcmIndexEntry a = index[i], b = index[partner];
    bool ascending = (i & f.SortWidth) == 0;
    if (ascending ? VcmIndexBefore(b, a, f.SortKind) : VcmIndexBefore(a, b, f.SortKind))
    {
        index[i] = b;
        index[partner] = a;
    }
}
kernel void ArvoVcmClearFilm(constant VcmSceneControls& c [[buffer(6)]],
                             device atomic_uint* errors [[buffer(11)]],
                             constant VcmCouplingControls& coupling [[buffer(12)]],
                             constant VcmFrameControls& f [[buffer(16)]],
                             device VcmPixelState* pixels [[buffer(18)]],
                             device VcmFilmPixel* film [[buffer(20)]], uint lane [[thread_position_in_grid]])
{
    if (lane >= f.WorkCount || !VcmFrameValid(c, coupling, f, errors))
        return;
    if (!VcmFrameWorkValid(f, f.PixelCapacity, errors))
        return;
    uint pixel = f.WorkBegin + lane;
    if (pixel >= f.Width * f.Height)
        return;
    pixels[pixel] = {};
    film[pixel] = {};
}

void VcmAdvanceCamera(VcmScene scene, device const VcmWalkVertex* vertices, device const VcmWalk* paths,
                      device const VcmIndexEntry* index, constant VcmCouplingControls& input,
                      constant VcmFrameControls& f, uint pathIndex, thread VcmPixelState& state,
                      thread VcmWalkStatistics& stats)
{
    VcmWalk camera = paths[pathIndex];
    if (!VcmFramePath(scene, camera, camera.Identity, TransportRadiance, input, f))
        return;
    if (state.Phase == VcmCommitted && state.CompletedPasses == f.Pass)
    {
        uint selected = VcmPairLight(camera.Identity, scene.Controls->Seed, f.Pass, f.LightCount);
        if (selected == VcmInvalidIndex)
        {
            VcmReport(scene, VcmNumericalError);
            return;
        }
        state.Estimate = {};
        state.CameraPath = pathIndex;
        state.LightPath = f.LightBegin + selected;
        state.ConnectionS = 0;
        state.ConnectionT = state.MergeT = 2;
        state.Neighbor = 0;
        state.Cursor = state.SplatCursor = VcmInvalidIndex;
        state.Pass = f.Pass;
        state.Phase = camera.Count >= 2 ? VcmConnecting : VcmCameraDone;
    }
    if (state.Pass != f.Pass || state.CompletedPasses != f.Pass || state.CameraPath != pathIndex ||
        state.LightPath < f.LightBegin || state.LightPath - f.LightBegin >= f.LightCount ||
        state.Phase < VcmConnecting || state.Phase > VcmCameraDone)
    {
        VcmReport(scene, VcmControlError);
        return;
    }
    VcmWalk light = paths[state.LightPath];
    if (!VcmFramePath(scene, light, state.LightPath - f.LightBegin, TransportImportance, input, f))
        return;
    if ((state.Phase == VcmConnecting &&
         (state.ConnectionS > light.Count || state.ConnectionT < 2 || state.ConnectionT > camera.Count)) ||
        (state.Phase == VcmMerging && (state.MergeT < 2 || state.Neighbor >= VcmGridNeighbors ||
                                       (state.Cursor != VcmInvalidIndex && state.Cursor > f.IndexCapacity))))
    {
        VcmReport(scene, VcmControlError);
        return;
    }
    VcmCouplingControls coupling = VcmFrameCoupling(*scene.Controls, input, f);
    for (uint work = 0; work < VcmFrameWorkBudget && state.Phase != VcmCameraDone; ++work)
    {
        if (state.Phase == VcmConnecting)
        {
            VcmCouplingQuery query = {state.LightPath, pathIndex, state.ConnectionS, state.ConnectionT, 0, 0,
                                      float2(0)};
            state.Estimate = VcmAddColor(
                state.Estimate, VcmCouple(scene, vertices, light, camera, query, coupling, stats).Radiance);
            if (++state.ConnectionT > camera.Count)
            {
                state.ConnectionT = 2;
                if (++state.ConnectionS > light.Count)
                    state.Phase = VcmMerging;
            }
        }
        else
        {
            if (state.MergeT > camera.Count || !(coupling.Families & VcmMergings) ||
                !VcmPositive(coupling.Radius))
            {
                state.Phase = VcmCameraDone;
                continue;
            }
            VcmWalkVertex vertex = vertices[camera.Begin + state.MergeT - 1];
            if (vertex.Kind != VcmSurfaceVertex && vertex.Kind != VcmVolumeVertex)
            {
                ++state.MergeT;
                continue;
            }
            VcmIndexEntry cell = {};
            if (!VcmGridCell(float3(vertex.Position), VcmGridExponent(*scene.Controls, coupling.Radius),
                             cell))
            {
                VcmReport(scene, VcmNumericalError);
                return;
            }
            cell.CellX += int(state.Neighbor % VcmGridSide) - 1;
            cell.CellY += int((state.Neighbor / VcmGridSide) % VcmGridSide) - 1;
            cell.CellZ += int(state.Neighbor / (VcmGridSide * VcmGridSide)) - 1;
            if (state.Cursor == VcmInvalidIndex)
                state.Cursor = VcmLowerBound(index, f.IndexCapacity, cell, VcmSpatialIndex);
            if (state.Cursor >= f.IndexCapacity || !index[state.Cursor].Valid ||
                VcmCellCompare(index[state.Cursor], cell))
            {
                state.Cursor = VcmInvalidIndex;
                if (++state.Neighbor == VcmGridNeighbors)
                {
                    state.Neighbor = 0;
                    ++state.MergeT;
                }
                continue;
            }
            VcmIndexEntry candidate = index[state.Cursor++];
            if (candidate.Path < f.LightBegin || candidate.Path - f.LightBegin >= f.LightCount)
            {
                VcmReport(scene, VcmBoundsError);
                return;
            }
            VcmWalk photon = paths[candidate.Path];
            if (!VcmFramePath(scene, photon, candidate.Identity, TransportImportance, input, f))
                return;
            if (!candidate.Vertex || candidate.Vertex >= photon.Count)
            {
                VcmReport(scene, VcmBoundsError);
                return;
            }
            VcmCouplingQuery query = {candidate.Path, pathIndex, candidate.Vertex + 1, state.MergeT, 1, 0,
                                      float2(0)};
            state.Estimate = VcmAddColor(
                state.Estimate, VcmCouple(scene, vertices, photon, camera, query, coupling, stats).Radiance);
        }
    }
}
kernel void ArvoVcmCameraFrame(
    device const PackedAccelerationNode* nodes [[buffer(0)]], device const uint* references [[buffer(1)]],
    device const PackedTriangle* triangles [[buffer(2)]], device const PackedSphere* spheres [[buffer(3)]],
    device const PackedMaterial* materials [[buffer(4)]], device const PackedEmitter* sources [[buffer(5)]],
    constant VcmSceneControls& controls [[buffer(6)]],
    device const PackedTextureDescriptor* textures [[buffer(7)]], device const uint* texels [[buffer(8)]],
    device const InstanceRecord* instances [[buffer(9)]],
    device const InstanceNode* instanceNodes [[buffer(10)]], device atomic_uint* errors [[buffer(11)]],
    constant VcmCouplingControls& coupling [[buffer(12)]],
    device const VcmWalkVertex* vertices [[buffer(13)]], device const VcmWalk* paths [[buffer(14)]],
    device VcmWalkStatistics* statistics [[buffer(15)]], constant VcmFrameControls& f [[buffer(16)]],
    device const VcmIndexEntry* index [[buffer(17)]], device VcmPixelState* pixels [[buffer(18)]],
    device atomic_uint* pending [[buffer(21)]], uint lane [[thread_position_in_grid]])
{
    if (lane >= f.CameraCount || !VcmFrameValid(controls, coupling, f, errors))
        return;
    if (lane >= f.StatisticsCapacity)
    {
        VcmFrameError(errors, VcmBoundsError);
        return;
    }
    uint pathIndex = f.CameraBegin + lane, pixel = 0;
    uint firstIdentity = paths[f.CameraBegin].Identity, population = controls.FullWidth * controls.FullHeight;
    if (firstIdentity >= population || f.CameraCount > population - firstIdentity ||
        paths[pathIndex].Identity != firstIdentity + lane ||
        !VcmCropPixel(controls, f, paths[pathIndex].Identity, pixel))
    {
        VcmFrameError(errors, VcmControlError);
        return;
    }
    VcmScene scene = {{{nodes, references, triangles, spheres, controls.NodeCount, controls.TriangleCount,
                        controls.Acceleration},
                       instances,
                       instanceNodes,
                       errors,
                       controls.InstanceNodeCount},
                      materials,
                      textures,
                      texels,
                      sources,
                      &controls};
    VcmPixelState state = pixels[pixel];
    VcmWalkStatistics stats = {};
    VcmAdvanceCamera(scene, vertices, paths, index, coupling, f, pathIndex, state, stats);
    if (!atomic_load_explicit(errors, memory_order_relaxed))
        pixels[pixel] = state;
    statistics[lane] = stats;
    if (state.Phase != VcmCameraDone)
        atomic_fetch_or_explicit(pending, VcmPendingCamera, memory_order_relaxed);
}
kernel void ArvoVcmProjectLightPaths(
    device const PackedAccelerationNode* nodes [[buffer(0)]], device const uint* references [[buffer(1)]],
    device const PackedTriangle* triangles [[buffer(2)]], device const PackedSphere* spheres [[buffer(3)]],
    device const PackedMaterial* materials [[buffer(4)]], device const PackedEmitter* sources [[buffer(5)]],
    constant VcmSceneControls& controls [[buffer(6)]],
    device const PackedTextureDescriptor* textures [[buffer(7)]], device const uint* texels [[buffer(8)]],
    device const InstanceRecord* instances [[buffer(9)]],
    device const InstanceNode* instanceNodes [[buffer(10)]], device atomic_uint* errors [[buffer(11)]],
    constant VcmCouplingControls& input [[buffer(12)]], device const VcmWalkVertex* vertices [[buffer(13)]],
    device const VcmWalk* paths [[buffer(14)]], device VcmWalkStatistics* statistics [[buffer(15)]],
    constant VcmFrameControls& f [[buffer(16)]], device VcmIndexEntry* index [[buffer(17)]],
    device VcmSplat* splats [[buffer(19)]], uint lane [[thread_position_in_grid]])
{
    if (lane >= f.WorkCount || !VcmFrameValid(controls, input, f, errors))
        return;
    if (!VcmFrameWorkValid(f, f.IndexCapacity, errors))
        return;
    if (lane >= f.StatisticsCapacity)
    {
        VcmFrameError(errors, VcmBoundsError);
        return;
    }
    uint slot = f.WorkBegin + lane;
    index[slot] = {};
    statistics[lane] = {};
    if (slot >= f.LightCount * f.SlotsPerLight)
        return;
    splats[slot] = {};
    uint identity = slot / f.SlotsPerLight, vertex = slot % f.SlotsPerLight + 1;
    VcmWalk light = paths[f.LightBegin + identity];
    VcmScene scene = {{{nodes, references, triangles, spheres, controls.NodeCount, controls.TriangleCount,
                        controls.Acceleration},
                       instances,
                       instanceNodes,
                       errors,
                       controls.InstanceNodeCount},
                      materials,
                      textures,
                      texels,
                      sources,
                      &controls};
    if (!VcmFramePath(scene, light, identity, TransportImportance, input, f))
        return;
    if (vertex >= light.Count || (vertices[light.Begin + vertex].Kind != VcmSurfaceVertex &&
                                  vertices[light.Begin + vertex].Kind != VcmVolumeVertex))
        return;
    VcmWalkDispatch dispatch = {};
    dispatch.Pass = f.Pass;
    dispatch.LightPopulation = f.LightCount;
    float2 lens = Sample2D(VcmSamples(scene, dispatch, identity, TransportImportance),
                           VcmLensConnectionDomain ^ (vertex * BounceSeedMultiplier));
    VcmCouplingQuery query = {f.LightBegin + identity, 0, vertex + 1, 1, 0, 0, lens};
    VcmCouplingControls coupling = VcmFrameCoupling(controls, input, f);
    VcmWalkStatistics stats = {};
    VcmContribution value = VcmCouple(scene, vertices, light, {}, query, coupling, stats);
    statistics[lane] = stats;
    if (!value.Accepted || !value.Splat || atomic_load_explicit(errors, memory_order_relaxed))
        return;
    if (!isfinite(value.Raster.x) || !isfinite(value.Raster.y) || value.Raster.x < 0 || value.Raster.x >= 1 ||
        value.Raster.y < 0 || value.Raster.y >= 1)
        return;
    uint x = min(uint(value.Raster.x * controls.FullWidth), controls.FullWidth - 1);
    uint y = min(uint(value.Raster.y * controls.FullHeight), controls.FullHeight - 1);
    uint fullPixel = y * controls.FullWidth + x, pixel = 0;
    if (!VcmCropPixel(controls, f, fullPixel, pixel))
        return;
    splats[slot] = {VcmScaleColor(value.Radiance, WideMath::Divide(VcmOne(), coupling.LightTracingSamples)),
                    fullPixel, identity, vertex, 1};
    VcmIndexEntry entry = {};
    entry.Path = slot;
    entry.Vertex = vertex;
    entry.Identity = identity;
    entry.Key = fullPixel;
    entry.Valid = 1;
    index[slot] = entry;
}
kernel void ArvoVcmCommitFilm(
    constant VcmSceneControls& c [[buffer(6)]], device atomic_uint* errors [[buffer(11)]],
    constant VcmCouplingControls& coupling [[buffer(12)]], constant VcmFrameControls& f [[buffer(16)]],
    device const VcmIndexEntry* index [[buffer(17)]], device VcmPixelState* pixels [[buffer(18)]],
    device const VcmSplat* splats [[buffer(19)]], device VcmFilmPixel* film [[buffer(20)]],
    device atomic_uint* pending [[buffer(21)]], uint lane [[thread_position_in_grid]])
{
    if (lane >= f.WorkCount || !VcmFrameValid(c, coupling, f, errors))
        return;
    if (!VcmFrameWorkValid(f, f.PixelCapacity, errors))
        return;
    uint pixel = f.WorkBegin + lane;
    if (pixel >= f.Width * f.Height)
        return;
    VcmPixelState state = pixels[pixel];
    if (state.Phase == VcmCommitted && state.CompletedPasses == f.Pass + 1)
        return;
    if (state.Phase != VcmCameraDone || state.Pass != f.Pass || state.CompletedPasses != f.Pass)
    {
        VcmFrameError(errors, VcmControlError);
        return;
    }
    VcmIndexEntry key = {};
    key.Key = (f.TileY + pixel / f.Width) * c.FullWidth + f.TileX + pixel % f.Width;
    if (state.SplatCursor == VcmInvalidIndex)
        state.SplatCursor = VcmLowerBound(index, f.IndexCapacity, key, VcmSplatIndex);
    for (uint work = 0; work < VcmFrameWorkBudget; ++work)
    {
        if (state.SplatCursor >= f.IndexCapacity || !index[state.SplatCursor].Valid ||
            index[state.SplatCursor].Key != key.Key)
        {
            state.Sum = VcmAddColor(state.Sum, state.Estimate);
            ++state.CompletedPasses;
            state.Phase = VcmCommitted;
            VcmWideVector mean =
                VcmScaleColor(state.Sum, WideMath::Divide(VcmOne(), VcmFromUint(state.CompletedPasses)));
            // Progressive radii correlate estimates. Do not invent per-ray M2
            // or feed this pass mean to an independent-sample adaptive test.
            float3 encoded =
                float3(as_type<float>(WideMath::Encode(mean.X)), as_type<float>(WideMath::Encode(mean.Y)),
                       as_type<float>(WideMath::Encode(mean.Z)));
            if (!VcmNonnegative(encoded))
            {
                VcmFrameError(errors, VcmNumericalError);
                return;
            }
            film[pixel] = {packed_float3(encoded), state.CompletedPasses};
            break;
        }
        VcmIndexEntry entry = index[state.SplatCursor++];
        if (entry.Path >= f.SplatCapacity)
        {
            VcmFrameError(errors, VcmBoundsError);
            return;
        }
        VcmSplat splat = splats[entry.Path];
        if (!splat.Valid || splat.Pixel != key.Key || splat.Identity != entry.Identity ||
            splat.Vertex != entry.Vertex)
        {
            VcmFrameError(errors, VcmControlError);
            return;
        }
        state.Estimate = VcmAddColor(state.Estimate, splat.Radiance);
    }
    pixels[pixel] = state;
    if (state.Phase != VcmCommitted)
        atomic_fetch_or_explicit(pending, VcmPendingFilm, memory_order_relaxed);
}
