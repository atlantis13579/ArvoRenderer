#include "PointBasedGlobalIllumination.h"
#include "CpuTransport.h"
#include "SurfaceEmission.h"
#include "SubsurfaceScattering.h"
#include "SurfaceScattering.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace Arvo::RayTracing::Cpu
{
/*
Independent point-based GI: geometry -> diffuse bake -> radiance hierarchy ->
direction-buffer visibility/integration -> camera reconstruction. Christensen,
"Point-Based Approximate Color Bleeding" (2008), and Bunnell, GPU Gems 2, ch. 14:
https://developer.nvidia.com/gpugems/gpugems2/part-ii-shading-lighting-and-shadows/chapter-14-dynamic-ambient-occlusion-and

This is a biased surface-radiance discretization, separate from photon density
estimation and from path tracing. Geometry is camera independent. Triangles use
longest-edge bisection; sphere quadrature uses equal-solid-angle Fibonacci points
and the affine surface Jacobian. Equivalent-area disks are an approximation to
their patches, especially at silhouettes; holes and overlaps are a persistent
coverage bias, not an unbiased quadrature rule.

Dependency contract for the remaining transport phases: bake both oriented sides
without doubling geometric area; retain all opaque surfels for visibility; keep
direct source emission separate from reflected cache radiance to avoid counting
NEE twice. Each diffuse bounce reads a complete previous bake. A direction buffer
must resolve overlapping disks front to back before solid-angle integration.
Hierarchical aggregation needs angular-size and normal-cone error controls; a
plain unoccluded sum of disk form factors is not a visibility solution.

The cache stores diffuse reflection and the diffuse exit radiance of subsurface
random walks. Each dielectric subsurface surfel traces through its closed solid,
then queries exterior direct or cached indirect irradiance at the exit. This is
a nonlocal transport operator, not a diffuse reinterpretation of the boundary.
Other directional BRDFs and displacement require their own representations.
Camera samples, bake
samples, cache density and hierarchy error are separate controls. A shared bake
is not a collection of independent pixel samples for adaptive stopping.
The implementation resolves the nearest disk per sampled hemisphere direction.
No extra area factor belongs in this angular integral. Intrinsic emission enters
only the direct estimator; cached radiance contains reflection. Bounce orders
use Jacobi iteration and are added once, rather than recursively adding totals.
Camera samples query the actual first surface, then gather reflected radiance
from the same global bake. Direct emitter evaluation is separate, including
visible analytic rectangles without mesh proxies. Camera variance is conditional
on the shared bake; it cannot certify cache error or enable adaptive stopping.
TODO: hierarchical radiance approximation; other backends remain independent.
*/
namespace
{
constexpr uint32_t Dimensions = 3, AffineColumns = 4, TriangleCorners = 3;
constexpr uint32_t MinimumSpherePoints = 32, LeafSurfelCount = 8, CancellationPeriod = 1024;
constexpr uint32_t MaximumTriangleSplits = 64;
constexpr double SurfacePi = 3.14159265358979323846;
constexpr double GoldenRatioConjugate = 0.61803398874989484820;
constexpr double NormalConeRoundoff = 8 * std::numeric_limits<float>::epsilon();
using Vector = std::array<double, Dimensions>;
Vector V(Float3 p)
{
    return {p.X, p.Y, p.Z};
}
Vector Plus(Vector a, Vector b)
{
    return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}
Vector Minus(Vector a, Vector b)
{
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}
Vector Times(Vector a, double s)
{
    return {a[0] * s, a[1] * s, a[2] * s};
}
double Length(Vector a)
{
    return std::hypot(a[0], a[1], a[2]);
}
double Inner(Vector a, Vector b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
Vector Crossed(Vector a, Vector b)
{
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
Float3 Stored(Vector value)
{
    for (double component : value)
        if (!std::isfinite(component) || std::abs(component) > std::numeric_limits<float>::max())
            throw std::runtime_error("Point GI surface coordinates exceed finite storage.");
    return {float(value[0]), float(value[1]), float(value[2])};
}
Float3 Unit(Vector value)
{
    double length = Length(value);
    if (!(length > 0) || !std::isfinite(length))
        throw std::runtime_error("Point GI surface normal is undefined.");
    return Stored(Times(value, 1 / length));
}
struct Affine
{
    std::array<Vector, Dimensions> Columns{{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
    std::array<Vector, Dimensions> Cofactors = Columns;
    Vector Translation{};
    double Orientation = 1;
    Vector Direction(Vector v) const
    {
        return Plus(Plus(Times(Columns[0], v[0]), Times(Columns[1], v[1])), Times(Columns[2], v[2]));
    }
    Vector Point(Vector v) const { return Plus(Direction(v), Translation); }
    Vector Cofactor(Vector v) const
    {
        return Plus(Plus(Times(Cofactors[0], v[0]), Times(Cofactors[1], v[1])), Times(Cofactors[2], v[2]));
    }
    Float3 Normal(Vector v) const { return Unit(Times(Cofactor(v), Orientation)); }
};
Affine InstanceFrame(const GeometryInstance& instance)
{
    Affine result;
    for (uint32_t row = 0; row < Dimensions; ++row)
    {
        for (uint32_t column = 0; column < Dimensions; ++column)
            result.Columns[column][row] = instance.ObjectToWorld[row * AffineColumns + column];
        result.Translation[row] = instance.ObjectToWorld[row * AffineColumns + Dimensions];
    }
    for (uint32_t axis = 0; axis < Dimensions; ++axis)
        result.Cofactors[axis] =
            Crossed(result.Columns[(axis + 1) % Dimensions], result.Columns[(axis + 2) % Dimensions]);
    double determinant = Inner(result.Columns[0], result.Cofactors[0]);
    if (!std::isfinite(determinant) || determinant == 0)
        throw std::runtime_error("Point GI instance transform must be finite and invertible.");
    // Match ray instancing: inverse-transpose normals do not follow reflected
    // world-space triangle winding. The determinant sign is essential here.
    result.Orientation = determinant < 0 ? -1 : 1;
    return result;
}
struct Patch
{
    std::array<Vector, TriangleCorners> Barycentrics{{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
    uint32_t Depth = 0;
};
class Bake
{
    const Scene& SceneData;
    const PointGiBakeGeometry& Options;
    const Detail::CancellationProbe& Cancelled;
    uint64_t Work = 0;

  public:
    PointGiSurfaceCache Cache;
    Bake(const Scene& scene, const PointGiBakeGeometry& options, const Detail::CancellationProbe& cancelled)
        : SceneData(scene), Options(options), Cancelled(cancelled)
    {
    }
    void Poll()
    {
        if (Work++ % CancellationPeriod == 0 && Cancelled && Cancelled())
            throw std::runtime_error("Point GI surface bake was cancelled.");
    }
    void Add(PointGiSurfel surfel)
    {
        Poll();
        if (Cache.Surfels.size() >= Options.MaximumSurfels)
            throw std::runtime_error("Point GI surface budget cannot represent the requested patch area.");
        if (!(surfel.Area > 0) || !std::isfinite(surfel.Area))
            throw std::runtime_error("Point GI surface area is not finite and positive.");
        surfel.Radius = std::sqrt(surfel.Area / SurfacePi);
        Cache.Surfels.push_back(surfel);
    }
    void TrianglePatches(const Triangle& triangle, const Affine& frame, uint32_t primitive)
    {
        Poll();
        if (triangle.MaterialIndex >= SceneData.Materials.size())
            throw std::runtime_error("Point GI triangle material index is invalid.");
        Vector origin = V(triangle.Position0), e1 = Minus(V(triangle.Position1), origin),
               e2 = Minus(V(triangle.Position2), origin);
        Vector worldE1 = frame.Direction(e1), worldE2 = frame.Direction(e2);
        double twiceArea = Length(Crossed(worldE1, worldE2));
        if (!std::isfinite(twiceArea))
            throw std::runtime_error("Point GI triangle area is not finite.");
        if (twiceArea == 0)
            return;
        Float3 geometry = frame.Normal(Crossed(e1, e2));
        std::vector<Patch> pending(1);
        uint32_t ordinal = 0;
        const std::array<Vector, TriangleCorners> colors{V(triangle.Color0), V(triangle.Color1),
                                                         V(triangle.Color2)};
        const std::array<Vector, TriangleCorners> normals{V(triangle.Normal0), V(triangle.Normal1),
                                                          V(triangle.Normal2)};
        const std::array<Float2, TriangleCorners> uv{triangle.TexCoord0, triangle.TexCoord1,
                                                     triangle.TexCoord2};
        while (!pending.empty())
        {
            Poll();
            Patch patch = pending.back();
            pending.pop_back();
            std::array<Vector, TriangleCorners> relative;
            for (uint32_t corner = 0; corner < TriangleCorners; ++corner)
                relative[corner] = Plus(Times(worldE1, patch.Barycentrics[corner][1]),
                                        Times(worldE2, patch.Barycentrics[corner][2]));
            // Each binary split halves barycentric area exactly, regardless of
            // which edge is longest. Avoid subtracting far-translated positions.
            double area = std::ldexp(twiceArea / 2, -int(patch.Depth));
            if (area > Options.MaximumArea)
            {
                if (patch.Depth >= MaximumTriangleSplits ||
                    Cache.Surfels.size() + pending.size() + 2 > Options.MaximumSurfels)
                    throw std::runtime_error("Point GI triangle subdivision exceeds its finite budget.");
                uint32_t edge = 0;
                double longest = 0;
                for (uint32_t corner = 0; corner < TriangleCorners; ++corner)
                {
                    double length = Length(Minus(relative[(corner + 1) % TriangleCorners], relative[corner]));
                    if (length > longest)
                    {
                        longest = length;
                        edge = corner;
                    }
                }
                uint32_t next = (edge + 1) % TriangleCorners;
                Vector midpoint = Times(Plus(patch.Barycentrics[edge], patch.Barycentrics[next]), 0.5);
                Patch first = patch, second = patch;
                first.Barycentrics[next] = midpoint;
                second.Barycentrics[edge] = midpoint;
                first.Depth = second.Depth = patch.Depth + 1;
                pending.push_back(second);
                pending.push_back(first);
                continue;
            }
            Vector weights =
                Times(Plus(Plus(patch.Barycentrics[0], patch.Barycentrics[1]), patch.Barycentrics[2]),
                      1.0 / TriangleCorners);
            PointGiSurfel s;
            s.Position =
                Stored(frame.Point(Plus(origin, Plus(Times(e1, weights[1]), Times(e2, weights[2])))));
            s.GeometryNormal = s.ShadingNormal = geometry;
            s.Color = Stored(Plus(Plus(Times(colors[0], weights[0]), Times(colors[1], weights[1])),
                                  Times(colors[2], weights[2])));
            if (triangle.SmoothNormals)
            {
                Vector normal = Plus(Plus(Times(normals[0], weights[0]), Times(normals[1], weights[1])),
                                     Times(normals[2], weights[2]));
                if (Length(normal) > 0)
                    s.ShadingNormal = frame.Normal(normal);
            }
            if (Dot(s.ShadingNormal, s.GeometryNormal) < 0)
                s.ShadingNormal = -s.ShadingNormal;
            s.TexCoord = {float(weights[0] * uv[0].X + weights[1] * uv[1].X + weights[2] * uv[2].X),
                          float(weights[0] * uv[0].Y + weights[1] * uv[1].Y + weights[2] * uv[2].Y)};
            if (!std::isfinite(s.TexCoord.X) || !std::isfinite(s.TexCoord.Y))
                throw std::runtime_error("Point GI triangle texture coordinates are not finite.");
            s.HasTexCoord = true;
            s.Material = triangle.MaterialIndex;
            s.Primitive = primitive;
            s.Patch = ordinal++;
            s.Area = area;
            Add(s);
        }
    }
    void SpherePoints(const Sphere& sphere, const Affine& frame, uint32_t primitive)
    {
        Poll();
        if (sphere.MaterialIndex >= SceneData.Materials.size() || !std::isfinite(sphere.Radius) ||
            sphere.Radius <= 0)
            throw std::runtime_error("Point GI sphere parameters are invalid.");
        double oneNorm = 0, infinityNorm = 0;
        for (uint32_t axis = 0; axis < Dimensions; ++axis)
        {
            double column = 0, row = 0;
            for (uint32_t other = 0; other < Dimensions; ++other)
            {
                column += std::abs(frame.Cofactors[axis][other]);
                row += std::abs(frame.Cofactors[other][axis]);
            }
            oneNorm = std::max(oneNorm, column);
            infinityNorm = std::max(infinityNorm, row);
        }
        double radiusSquared = double(sphere.Radius) * sphere.Radius;
        double measure = 4 * SurfacePi * radiusSquared;
        // ||C*n|| <= sqrt(||C||_1 ||C||_inf) bounds every quadrature weight.
        double maximumArea = measure * std::sqrt(oneNorm * infinityNorm);
        double required = std::max(double(MinimumSpherePoints), std::ceil(maximumArea / Options.MaximumArea));
        if (!std::isfinite(required) || required > Options.MaximumSurfels - Cache.Surfels.size())
            throw std::runtime_error("Point GI sphere quadrature exceeds the surface budget.");
        uint32_t count = uint32_t(required);
        for (uint32_t i = 0; i < count; ++i)
        {
            double latitude = 1 - 2 * ((double(i) + 0.5) / count);
            double turn = double(i) * GoldenRatioConjugate;
            turn -= std::floor(turn);
            double azimuth = 2 * SurfacePi * turn, radial = std::sqrt(std::max(0.0, 1 - latitude * latitude));
            Vector normal{radial * std::cos(azimuth), latitude, radial * std::sin(azimuth)};
            PointGiSurfel s;
            s.Position = Stored(frame.Point(Plus(V(sphere.Center), Times(normal, sphere.Radius))));
            s.GeometryNormal = s.ShadingNormal = frame.Normal(normal);
            s.Color = {1, 1, 1};
            s.Material = sphere.MaterialIndex;
            s.Primitive = primitive;
            s.Patch = i;
            s.Area = (measure / count) * Length(frame.Cofactor(normal));
            Add(s);
        }
    }
    void Geometry(const SceneGeometry& geometry, const Affine& frame, uint64_t& primitive)
    {
        if (primitive + geometry.Triangles.size() + geometry.Spheres.size() >= InvalidRayPrimitiveIndex)
            throw std::runtime_error("Point GI primitive identity exceeds its storage.");
        for (const auto& triangle : geometry.Triangles)
            TrianglePatches(triangle, frame, uint32_t(primitive++));
        for (const auto& sphere : geometry.Spheres)
            SpherePoints(sphere, frame, uint32_t(primitive++));
    }
    uint32_t Hierarchy(uint32_t begin, uint32_t count)
    {
        Poll();
        uint32_t slot = uint32_t(Cache.Nodes.size());
        Cache.Nodes.emplace_back();
        PointGiNode node;
        node.Begin = begin;
        node.Count = count;
        node.Low.fill(INFINITY);
        node.High.fill(-INFINITY);
        const Vector anchor = V(Cache.Surfels[Cache.Order[begin]].Position);
        Vector normalSum{};
        for (uint32_t i = begin; i < begin + count; ++i)
        {
            Poll();
            const auto& s = Cache.Surfels[Cache.Order[i]];
            Vector point = V(s.Position);
            node.Area += s.Area;
            node.Center = Plus(node.Center, Times(Minus(point, anchor), s.Area));
            normalSum = Plus(normalSum, Times(V(s.GeometryNormal), s.Area));
            for (uint32_t axis = 0; axis < Dimensions; ++axis)
            {
                node.Low[axis] = std::min(node.Low[axis], std::nextafter(point[axis] - s.Radius, -INFINITY));
                node.High[axis] = std::max(node.High[axis], std::nextafter(point[axis] + s.Radius, INFINITY));
            }
        }
        // Relative moments avoid losing a tiny cluster's extent at a distant
        // origin. Outward-rounded bounds must enclose even sub-ULP disks.
        node.Center = Plus(anchor, Times(node.Center, 1 / node.Area));
        node.NormalAxis = Length(normalSum) > 0 ? Unit(normalSum) : Float3(0, 0, 1);
        node.MinimumNormalCosine = 1;
        for (uint32_t i = begin; i < begin + count; ++i)
        {
            const auto& s = Cache.Surfels[Cache.Order[i]];
            node.Radius = std::max(
                node.Radius, std::nextafter(Length(Minus(V(s.Position), node.Center)) + s.Radius, INFINITY));
            Vector axis = V(node.NormalAxis), normal = V(s.GeometryNormal);
            double cosine = Inner(axis, normal) / (Length(axis) * Length(normal));
            node.MinimumNormalCosine =
                std::min(node.MinimumNormalCosine, std::clamp(cosine - NormalConeRoundoff, -1.0, 1.0));
        }
        if (count > LeafSurfelCount)
        {
            uint32_t axis = 0;
            for (uint32_t candidate = 1; candidate < Dimensions; ++candidate)
                if (node.High[candidate] - node.Low[candidate] > node.High[axis] - node.Low[axis])
                    axis = candidate;
            std::stable_sort(Cache.Order.begin() + begin, Cache.Order.begin() + begin + count,
                             [&](uint32_t a, uint32_t b) {
                                 double x = V(Cache.Surfels[a].Position)[axis],
                                        y = V(Cache.Surfels[b].Position)[axis];
                                 return x != y ? x < y : a < b;
                             });
            uint32_t half = count / 2;
            node.Left = Hierarchy(begin, half);
            node.Right = Hierarchy(begin + half, count - half);
        }
        Cache.Nodes[slot] = node;
        return slot;
    }
};
} // namespace
bool BuildPointGiSurfaceCache(const Scene& scene, const PointGiBakeGeometry& options,
                              PointGiSurfaceCache& output, const Detail::CancellationProbe& cancelled,
                              std::string& error)
{
    output = {};
    if (!(options.MaximumArea > 0) || !std::isfinite(options.MaximumArea) || !options.MaximumSurfels ||
        options.MaximumSurfels > (InvalidRayPrimitiveIndex - 1) / 2)
    {
        error = "Point GI patch area and surface budget must be finite and positive.";
        return false;
    }
    if (!ValidateInstances(scene, error))
        return false;
    try
    {
        Bake bake(scene, options, cancelled);
        uint64_t primitive = 0;
        bake.Geometry(scene, {}, primitive);
        for (const auto& instance : scene.Instances)
            bake.Geometry(*scene.Geometries[instance.GeometryIndex], InstanceFrame(instance), primitive);
        bake.Cache.Order.resize(bake.Cache.Surfels.size());
        std::iota(bake.Cache.Order.begin(), bake.Cache.Order.end(), 0u);
        if (!bake.Cache.Order.empty())
            bake.Hierarchy(0, uint32_t(bake.Cache.Order.size()));
        if (cancelled && cancelled())
        {
            error = "Point GI surface bake was cancelled.";
            return false;
        }
        output = std::move(bake.Cache);
        error.clear();
        return true;
    }
    catch (const std::exception& exception)
    {
        error = exception.what();
        return false;
    }
}
namespace
{
constexpr uint32_t DiskTraversalStack = 64, MaximumDiffuseBounces = 32;
constexpr uint32_t SourceDomain = 0x534f5552u, PositionDomain = 0x504f534eu, GatherDomain = 0x47415448u;
constexpr uint32_t SubsurfaceInterfaceDomain = 0x53534946u, SubsurfaceBranchDomain = 0x53534252u;
constexpr uint32_t SubsurfaceFlightDomain = 0x5353464cu, SubsurfacePhaseDomain = 0x53535048u;
constexpr uint32_t BackSideSeed = 0x4241434bu; // Independent, stable streams for the two orientations.
constexpr double NormalLengthTolerance = 64 * std::numeric_limits<float>::epsilon();
void RequireColor(Vector value)
{
    for (double component : value)
        if (!std::isfinite(component) || component < 0)
            throw std::runtime_error("Point GI radiometry must be finite and nonnegative.");
}
void ValidateLighting(const PointGiLightingOptions& options)
{
    if (!options.DirectSamples || !options.GatherSamples || options.IndirectBounces > MaximumDiffuseBounces ||
        options.Sequence >= SampleSequence::Count)
        throw std::runtime_error("Point GI sampling or diffuse bounce controls are invalid.");
}
void RequireNormal(Float3 normal)
{
    if (std::abs(Length(V(normal)) - 1) > NormalLengthTolerance || !std::isfinite(Length(V(normal))))
        throw std::runtime_error("Point GI normals must be finite and unit length.");
}
void PollLighting(uint64_t& work, const Detail::CancellationProbe& cancelled)
{
    if (work++ % CancellationPeriod == 0 && cancelled && cancelled())
        throw std::runtime_error("Point GI lighting bake was cancelled.");
}
SampleContext BakeSamples(uint32_t identity, bool front, uint32_t sample, uint32_t count,
                          const PointGiLightingOptions& options)
{
    return {identity, front ? 0u : 1u, Hash(identity ^ Hash(options.Seed) ^ (front ? 0u : BackSideSeed)),
            sample,   count,           options.Sequence, options.SamplingSeed};
}
bool BoxHit(const PointGiNode& node, const Ray& ray, double far)
{
    double near = ray.MinDistance;
    const Vector origin = V(ray.Origin), direction = V(ray.Direction);
    for (uint32_t axis = 0; axis < Dimensions; ++axis)
    {
        if (!std::isfinite(node.Low[axis]) || !std::isfinite(node.High[axis]) ||
            node.Low[axis] > node.High[axis])
            throw std::runtime_error("Point GI hierarchy bounds are invalid.");
        if (direction[axis] == 0)
        {
            if (origin[axis] < node.Low[axis] || origin[axis] > node.High[axis])
                return false;
            continue;
        }
        double a = (node.Low[axis] - origin[axis]) / direction[axis];
        double b = (node.High[axis] - origin[axis]) / direction[axis];
        if (a > b)
            std::swap(a, b);
        near = std::max(near, a);
        far = std::min(far, b);
        if (near > far)
            return false;
    }
    return true;
}
PointGiDiskHit DiskHit(const PointGiSurfaceCache& cache, const Ray& ray, uint32_t ignored, uint64_t& tests)
{
    PointGiDiskHit hit;
    if (cache.Surfels.empty())
        return hit;
    std::array<uint32_t, DiskTraversalStack> stack{};
    uint32_t pending = 1;
    size_t visited = 0;
    while (pending)
    {
        uint32_t index = stack[--pending];
        if (index >= cache.Nodes.size() || ++visited > cache.Nodes.size())
            throw std::runtime_error("Point GI hierarchy has an invalid child or cycle.");
        const auto& node = cache.Nodes[index];
        if (!BoxHit(node, ray, std::min(hit.Distance, double(ray.MaxDistance))))
            continue;
        if (node.Left != InvalidRayPrimitiveIndex)
        {
            if (pending + 2 > stack.size())
                throw std::runtime_error("Point GI hierarchy exceeds its balanced traversal stack.");
            stack[pending++] = node.Right;
            stack[pending++] = node.Left;
            continue;
        }
        if (uint64_t(node.Begin) + node.Count > cache.Order.size())
            throw std::runtime_error("Point GI hierarchy leaf range is invalid.");
        for (uint32_t i = 0; i < node.Count; ++i)
        {
            uint32_t identity = cache.Order[node.Begin + i];
            if (identity >= cache.Surfels.size())
                throw std::runtime_error("Point GI hierarchy references an invalid surface.");
            if (identity == ignored)
                continue;
            ++tests;
            const auto& s = cache.Surfels[identity];
            const Vector normal = V(s.GeometryNormal), direction = V(ray.Direction);
            double denominator = Inner(normal, direction);
            if (denominator == 0)
                continue;
            Vector relative = Minus(V(ray.Origin), V(s.Position));
            double distance = -Inner(relative, normal) / denominator;
            if (!(distance > ray.MinDistance && distance < ray.MaxDistance) || distance > hit.Distance ||
                (distance == hit.Distance && identity >= hit.Surfel))
                continue;
            Vector radial = Plus(relative, Times(direction, distance));
            if (Inner(radial, radial) <= s.Radius * s.Radius)
                hit = {identity, distance, denominator < 0};
        }
    }
    return hit;
}
// Analytic rectangles are opaque even without a mesh proxy. Their backside
// blocks transport although the source emits into its front hemisphere only.
struct RectangleHit
{
    const PackedEmitter* Source = nullptr;
    double Distance = 0;
    bool Front = false;
};
RectangleHit FindRectangle(const PreparedEmissionScene& sources, const Ray& ray, double maximum,
                           uint64_t& tests)
{
    RectangleHit hit;
    for (const auto& source : sources.Sources)
    {
        if (source.Kind != EmitterKind::Rectangle)
            continue;
        ++tests;
        Float3 normal = Unit(V(source.Direction));
        Vector n = V(normal), direction = V(ray.Direction);
        double denominator = Inner(n, direction);
        if (denominator == 0)
            continue;
        Vector relative = Minus(V(ray.Origin), V(source.Origin));
        double distance = -Inner(relative, n) / denominator;
        if (!(distance > ray.MinDistance && distance < maximum))
            continue;
        Float3 helper = std::abs(normal.Y) < BasisParallelThreshold ? Float3(0, 1, 0) : Float3(1, 0, 0);
        Vector tangent = V(Unit(Crossed(V(helper), n))), bitangent = Crossed(n, tangent);
        Vector point = Plus(relative, Times(direction, distance));
        if (std::abs(Inner(point, tangent)) <= source.Width / 2 &&
            std::abs(Inner(point, bitangent)) <= source.Height / 2)
        {
            hit = {&source, distance, denominator < 0};
            maximum = distance;
        }
    }
    return hit;
}
bool RectangleBlocks(const PreparedEmissionScene& sources, const Ray& ray, double maximum, uint64_t& tests)
{
    return FindRectangle(sources, ray, maximum, tests).Source != nullptr;
}
Vector Reflectance(const Scene& scene, const PointGiSurfel& s)
{
    if (s.Material >= scene.Materials.size())
        throw std::runtime_error("Point GI surface references an invalid material.");
    const auto& material = scene.Materials[s.Material];
    if (material.SampledEmitter)
        return {};
    if (material.Model != MaterialModel::Diffuse &&
        !(material.Model == MaterialModel::Legacy && material.Reflectivity == 0 &&
          material.SpecularColor.X == 0 && material.SpecularColor.Y == 0 && material.SpecularColor.Z == 0))
        throw std::runtime_error(
            "Point GI diffuse cache does not support directional or transmissive BSDFs.");
    Vector albedo = V(material.Albedo), color = V(s.Color);
    Vector texture = s.HasTexCoord ? V(SampleDiffuseTexture(scene, material.DiffuseTextureIndex, s.TexCoord))
                                   : Vector{1, 1, 1};
    RequireColor(albedo);
    RequireColor(color);
    RequireColor(texture);
    Vector result{};
    for (uint32_t channel = 0; channel < Dimensions; ++channel)
        result[channel] = std::clamp(albedo[channel] * color[channel] * texture[channel], 0.0, 1.0);
    return result;
}
Vector Reflected(Vector rho, Vector irradiance)
{
    Vector result{};
    for (uint32_t channel = 0; channel < Dimensions; ++channel)
        result[channel] = rho[channel] * irradiance[channel] / SurfacePi;
    RequireColor(result);
    return result;
}
Vector DirectIrradiance(const SceneAccelerator& accelerator, const PreparedEmissionScene& sources,
                        const PointGiSurfel& receiver, uint32_t identity, bool front,
                        const PointGiLightingOptions& options, AtomicStatistics& statistics, uint64_t& work,
                        const Detail::CancellationProbe& cancelled,
                        const SampleContext* cameraSample = nullptr)
{
    Vector result{};
    if (sources.Sources.empty())
        return result;
    const Float3 normal = front ? receiver.ShadingNormal : -receiver.ShadingNormal;
    const Float3 geometric = front ? receiver.GeometryNormal : -receiver.GeometryNormal;
    const uint32_t sampleCount = cameraSample ? 1 : options.DirectSamples;
    for (uint32_t sample = 0; sample < sampleCount; ++sample)
    {
        PollLighting(work, cancelled);
        const auto random =
            cameraSample ? *cameraSample : BakeSamples(identity, front, sample, sampleCount, options);
        const auto selection =
            SelectEmitter(uint32_t(sources.Sources.size()), Sample2D(random, SourceDomain).X);
        if (!(selection.Probability > 0))
            throw std::runtime_error("Point GI source selection exceeds its representable domain.");
        const auto& source = sources.Sources[selection.Index];
        const Float2 uv = Sample2D(random, PositionDomain);
        Float3 direction, endpoint;
        Vector incoming = V(source.Strength);
        double weight = 1 / double(selection.Probability);
        bool infinite = source.Kind == EmitterKind::Directional || source.Kind == EmitterKind::Environment;
        if (source.Kind == EmitterKind::Directional)
            direction = Unit(V(source.Direction));
        else if (source.Kind == EmitterKind::Environment)
        {
            direction = SampleDiffuseDirection(normal, uv.X, uv.Y, options.ImportanceSampling);
            double cosine = Dot(normal, direction),
                   pdf = DiffusePdf(float(cosine), options.ImportanceSampling);
            if (!(pdf > 0))
                continue;
            weight /= pdf;
            double latitude = std::clamp((double(direction.Y) + 1) / 2, 0.0, 1.0);
            incoming = Plus(Times(V(source.BackgroundBottom), 1 - latitude),
                            Times(V(source.BackgroundTop), latitude));
        }
        else
        {
            EmissionSample point;
            if (source.Kind == EmitterKind::Point)
                point.Origin = source.Origin;
            else if (source.Kind == EmitterKind::Rectangle)
            {
                EmissionSource rectangle;
                rectangle.Kind = EmissionKind::Rectangle;
                rectangle.Position = source.Origin;
                rectangle.Direction = source.Direction;
                rectangle.Strength = source.Strength;
                rectangle.Size = {source.Width, source.Height};
                point = SampleLightEmission(rectangle, {}, uv, {0.5f, 0.5f});
            }
            else
                point = SampleSurfaceEmission(
                    {source.Kind == EmitterKind::Triangle ? SurfaceEmitterKind::Triangle
                                                          : SurfaceEmitterKind::Ellipsoid,
                     source.Origin, source.AxisX, source.AxisY, source.AxisZ, source.Strength},
                    uv, {0.5f, 0.5f});
            if (source.Kind != EmitterKind::Point &&
                (!(point.Flags & EmissionValid) || !(point.PositionPdf > 0) ||
                 !std::isfinite(point.PositionPdf)))
                throw std::runtime_error("Point GI finite emitter sample exceeds its representable domain.");
            endpoint = point.Origin;
            Vector delta = Minus(V(endpoint), V(receiver.Position));
            double distance = Length(delta);
            if (!(distance > RayEpsilon))
                continue;
            direction = Stored(Times(delta, 1 / distance));
            weight /= distance * distance;
            if (source.Kind != EmitterKind::Point)
            {
                double emitterCosine = Dot(point.Normal, -direction);
                emitterCosine = source.Kind == EmitterKind::Rectangle ? std::max(0.0, emitterCosine)
                                                                      : std::abs(emitterCosine);
                weight *= emitterCosine / point.PositionPdf;
                incoming = V(point.Value);
            }
        }
        double cosine = Dot(normal, direction);
        if (!(cosine > 0 && Dot(geometric, direction) > 0 && weight > 0))
            continue;
        Ray shadow{receiver.Position + geometric * RayEpsilon, direction, RayEpsilon, MaximumRayDistance};
        if (!infinite)
        {
            Vector delta = Minus(V(endpoint), V(shadow.Origin));
            double distance = Length(delta);
            if (!(distance > 2 * RayEpsilon) || distance > MaximumRayDistance)
                continue;
            shadow.Direction = Stored(Times(delta, 1 / distance));
            shadow.MaxDistance = float(distance - RayEpsilon);
        }
        uint64_t tests = 0;
        bool blocked = accelerator.Occluded(shadow, tests) ||
                       RectangleBlocks(sources, shadow, shadow.MaxDistance, tests);
        ++statistics.ShadowRays;
        statistics.PrimitiveTests += tests;
        if (!blocked)
            result = Plus(result, Times(incoming, weight * cosine / sampleCount));
    }
    RequireColor(result);
    return result;
}
Vector Gather(const PointGiSurfaceCache& cache, const std::vector<PointGiRadiance>& radiance,
              const PreparedEmissionScene& sources, const PointGiSurfel& receiver, uint32_t identity,
              bool front, const PointGiLightingOptions& options, AtomicStatistics& statistics, uint64_t& work,
              const Detail::CancellationProbe& cancelled, const SampleContext* cameraSample = nullptr)
{
    Vector result{};
    const Float3 normal = front ? receiver.ShadingNormal : -receiver.ShadingNormal;
    const Float3 geometric = front ? receiver.GeometryNormal : -receiver.GeometryNormal;
    const uint32_t sampleCount = cameraSample ? 1 : options.GatherSamples;
    for (uint32_t sample = 0; sample < sampleCount; ++sample)
    {
        PollLighting(work, cancelled);
        const auto random =
            cameraSample ? *cameraSample : BakeSamples(identity, front, sample, sampleCount, options);
        const Float2 uv = Sample2D(random, GatherDomain);
        Float3 direction = SampleDiffuseDirection(normal, uv.X, uv.Y, options.ImportanceSampling);
        double cosine = Dot(normal, direction), pdf = DiffusePdf(float(cosine), options.ImportanceSampling);
        if (!(pdf > 0 && Dot(geometric, direction) > 0))
            continue;
        Ray ray{receiver.Position + geometric * RayEpsilon, direction, RayEpsilon, MaximumRayDistance};
        uint64_t tests = 0;
        const auto hit = DiskHit(cache, ray, identity, tests);
        bool blocked = RectangleBlocks(sources, ray, std::min(hit.Distance, double(ray.MaxDistance)), tests);
        ++statistics.ReflectionRays;
        statistics.PrimitiveTests += tests;
        if (!blocked && hit.Surfel != InvalidRayPrimitiveIndex)
        {
            const auto& value = radiance[hit.Surfel];
            result = Plus(result, Times(hit.Front ? value.Front : value.Back, cosine / (pdf * sampleCount)));
        }
    }
    RequireColor(result);
    return result;
}
struct SubsurfaceExit
{
    PointGiSurfel Receiver;
    Vector Throughput{};
    bool Reached = false;
};
SubsurfaceExit TraceSubsurface(const Scene& scene, const SceneAccelerator& accelerator,
                               const SurfaceHit& entry, Float3 position, Ray ray, bool cameraInside,
                               const SampleContext& random, bool importance, AtomicStatistics& statistics,
                               uint64_t& work, const Detail::CancellationProbe& cancelled)
{
    SubsurfaceExit result;
    const uint32_t materialIndex = entry.MaterialIndex;
    const Material& material = scene.Materials.at(materialIndex);
    Float3 throughput{1, 1, 1};
    if (!cameraInside)
    {
        const SurfaceBsdf interface(material, entry, -ray.Direction, entry.Color);
        const SurfaceSample launch = interface.Sample(Sample2D(random, SubsurfaceInterfaceDomain),
            Sample2D(random, SubsurfaceBranchDomain).X, importance);
        if ((launch.Event & ScatteringTransmission) == 0 || launch.Pdf <= 0)
            return result;
        throughput = Multiply(throughput, launch.Weight);
        ray = {interface.Offset(position, launch.Direction), launch.Direction,
               RayEpsilon, MaximumRayDistance};
    }
    for (uint32_t bounce = 0; bounce < MaximumDiffuseBounces; ++bounce)
    {
        PollLighting(work, cancelled);
        uint64_t tests = 0;
        SurfaceHit boundary;
        const bool found = accelerator.Intersect(ray, boundary, tests);
        ++statistics.ReflectionRays;
        statistics.PrimitiveTests += tests;
        if (!found || boundary.MaterialIndex != materialIndex)
            return result;
        const uint32_t dimension = (bounce + 1) * BounceSeedMultiplier;
        const auto flight = SampleSubsurfaceFlight(material.SubsurfaceMeanFreePath,
            material.Albedo * material.SubsurfaceAlbedoScale, ray.Direction, boundary.Distance,
            Sample2D(random, dimension ^ SubsurfaceFlightDomain).X);
        throughput = Multiply(throughput, flight.Weight);
        if (!(throughput.X > 0 || throughput.Y > 0 || throughput.Z > 0))
            return result;
        if (flight.Scattered)
        {
            const auto phase = SampleSubsurfaceDirection(ray.Direction, material.SubsurfaceAnisotropy,
                Sample2D(random, dimension ^ SubsurfacePhaseDomain), importance);
            if (!(phase.Pdf > 0)) return result;
            throughput = throughput * phase.Weight;
            ray = {ray.Origin + ray.Direction * flight.Distance, phase.Direction,
                   RayEpsilon, MaximumRayDistance};
            continue;
        }
        const Float3 boundaryPosition = ray.Origin + ray.Direction * boundary.Distance;
        const SurfaceBsdf interface(material, boundary, -ray.Direction, boundary.Color);
        const SurfaceSample event = interface.Sample(Sample2D(random, dimension ^ SubsurfaceInterfaceDomain),
            Sample2D(random, dimension ^ SubsurfaceBranchDomain).X, importance);
        if (event.Event == ScatteringNone || !(event.Pdf > 0))
            return result;
        throughput = Multiply(throughput, event.Weight);
        if (!(throughput.X > 0 || throughput.Y > 0 || throughput.Z > 0))
            return result;
        if ((event.Event & ScatteringTransmission) == 0)
        {
            ray = {interface.Offset(boundaryPosition, event.Direction), event.Direction,
                   RayEpsilon, MaximumRayDistance};
            continue;
        }
        auto& receiver = result.Receiver;
        receiver.Position = boundaryPosition;
        receiver.GeometryNormal = boundary.GeometricNormal;
        receiver.ShadingNormal = Dot(boundary.Normal, receiver.GeometryNormal) >= 0 ?
            boundary.Normal : -boundary.Normal;
        receiver.Material = materialIndex;
        receiver.Primitive = boundary.PrimitiveIndex;
        receiver.Color = boundary.Color;
        receiver.TexCoord = boundary.TexCoord;
        receiver.HasTexCoord = boundary.HasTexCoord;
        result.Throughput = V(throughput);
        RequireColor(result.Throughput);
        result.Reached = true;
        return result;
    }
    return result;
}
SurfaceHit SurfelHit(const PointGiSurfel& surfel)
{
    SurfaceHit hit;
    hit.Normal = surfel.ShadingNormal;
    hit.GeometricNormal = surfel.GeometryNormal;
    hit.Color = surfel.Color;
    hit.TexCoord = surfel.TexCoord;
    hit.MaterialIndex = surfel.Material;
    hit.PrimitiveIndex = surfel.Primitive;
    hit.HasTexCoord = surfel.HasTexCoord;
    hit.FrontFace = true;
    return hit;
}
Vector SubsurfaceIncident(const Scene& scene, const SceneAccelerator& accelerator,
                          const PointGiSurfaceCache& cache, const PreparedEmissionScene& sources,
                          const std::vector<PointGiRadiance>* previous, const SurfaceHit& entry,
                          Float3 position, Ray ray, bool cameraInside, const SampleContext& random,
                          const PointGiLightingOptions& options, AtomicStatistics& statistics,
                          uint64_t& work, const Detail::CancellationProbe& cancelled)
{
    const auto exit = TraceSubsurface(scene, accelerator, entry, position, ray, cameraInside,
                                      random, options.ImportanceSampling, statistics, work, cancelled);
    if (!exit.Reached) return {};
    Vector incident = previous ? Gather(cache, *previous, sources, exit.Receiver,
                                        InvalidRayPrimitiveIndex, true, options, statistics, work,
                                        cancelled, &random) :
                                 DirectIrradiance(accelerator, sources, exit.Receiver,
                                                  InvalidRayPrimitiveIndex, true, options, statistics,
                                                  work, cancelled, &random);
    Vector outgoing{};
    for (uint32_t channel = 0; channel < Dimensions; ++channel)
        outgoing[channel] = exit.Throughput[channel] * incident[channel] / SurfacePi;
    RequireColor(outgoing);
    return outgoing;
}
} // namespace
bool IntersectPointGiDisks(const PointGiSurfaceCache& cache, const Ray& ray, uint32_t ignored,
                           PointGiDiskHit& output, uint64_t& diskTests, std::string& error)
{
    output = {};
    try
    {
        if (!std::isfinite(Length(V(ray.Origin))) || !std::isfinite(Length(V(ray.Direction))) ||
            !(Length(V(ray.Direction)) > 0) || !(ray.MinDistance >= 0 && ray.MaxDistance > ray.MinDistance))
            throw std::runtime_error("Point GI disk ray is invalid.");
        auto hit = DiskHit(cache, ray, ignored, diskTests);
        output = hit;
        error.clear();
        return true;
    }
    catch (const std::exception& exception)
    {
        error = exception.what();
        return false;
    }
}
bool GatherPointGiIrradiance(const PointGiSurfaceCache& cache, const std::vector<PointGiRadiance>& radiance,
                             const PreparedEmissionScene& sources, const PointGiSurfel& receiver,
                             uint32_t identity, bool front, const PointGiLightingOptions& options,
                             Vector& output, AtomicStatistics& statistics,
                             const Detail::CancellationProbe& cancelled, std::string& error)
{
    output = {};
    try
    {
        ValidateLighting(options);
        RequireNormal(receiver.GeometryNormal);
        RequireNormal(receiver.ShadingNormal);
        if (!std::isfinite(Length(V(receiver.Position))) ||
            Dot(receiver.GeometryNormal, receiver.ShadingNormal) < 0)
            throw std::runtime_error("Point GI receiver position or normal orientation is invalid.");
        if (radiance.size() != cache.Surfels.size())
            throw std::runtime_error("Point GI radiance and surface cache sizes differ.");
        for (const auto& value : radiance)
        {
            RequireColor(value.Front);
            RequireColor(value.Back);
        }
        uint64_t work = 0;
        auto result =
            Gather(cache, radiance, sources, receiver, identity, front, options, statistics, work, cancelled);
        if (cancelled && cancelled())
            throw std::runtime_error("Point GI lighting gather was cancelled.");
        output = result;
        error.clear();
        return true;
    }
    catch (const std::exception& exception)
    {
        error = exception.what();
        return false;
    }
}
bool BakePointGiDiffuse(const Scene& scene, const SceneAccelerator& accelerator,
                        const PointGiSurfaceCache& cache, const PointGiLightingOptions& options,
                        PointGiLighting& output, AtomicStatistics& statistics,
                        const Detail::CancellationProbe& cancelled, std::string& error)
{
    output = {};
    try
    {
        ValidateLighting(options);
        uint64_t work = 0;
        PollLighting(work, cancelled);
        const auto sources = PrepareEmitters(scene);
        if (sources.Sources.size() > MaximumUniformEmitters)
            throw std::runtime_error("Point GI source catalog exceeds uniform selection precision.");
        for (const auto& source : sources.Sources)
            if (!(source.Flags & EmitterCastsShadows))
                throw std::runtime_error("Point GI requires physical visibility for every source.");
        std::vector<Vector> reflectance;
        std::vector<bool> subsurface;
        reflectance.reserve(cache.Surfels.size());
        subsurface.reserve(cache.Surfels.size());
        for (const auto& s : cache.Surfels)
        {
            PollLighting(work, cancelled);
            RequireNormal(s.GeometryNormal);
            RequireNormal(s.ShadingNormal);
            const bool interior = scene.Materials.at(s.Material).SubsurfaceMeanFreePath > 0;
            if (interior && scene.Materials[s.Material].Model != MaterialModel::Dielectric)
                throw std::runtime_error("Point GI subsurface transport requires a dielectric solid.");
            subsurface.push_back(interior);
            reflectance.push_back(interior ? Vector{} : Reflectance(scene, s));
        }
        PointGiLighting result;
        result.Direct.resize(cache.Surfels.size());
        for (uint32_t i = 0; i < cache.Surfels.size(); ++i)
        {
            PollLighting(work, cancelled);
            if (!subsurface[i] && reflectance[i] == Vector{})
                continue;
            for (bool front : {true, false})
            {
                auto& side = front ? result.Direct[i].Front : result.Direct[i].Back;
                if (subsurface[i])
                {
                    if (!front) continue;
                    const auto& surfel = cache.Surfels[i];
                    const SurfaceHit entry = SurfelHit(surfel);
                    Ray ray{surfel.Position + surfel.GeometryNormal * RayEpsilon,
                            -surfel.GeometryNormal, RayEpsilon, MaximumRayDistance};
                    for (uint32_t sample = 0; sample < options.DirectSamples; ++sample)
                    {
                        const auto random = BakeSamples(i, true, sample, options.DirectSamples, options);
                        side = Plus(side, Times(SubsurfaceIncident(scene, accelerator, cache, sources,
                            nullptr, entry, surfel.Position, ray, false, random, options, statistics,
                            work, cancelled), 1 / double(options.DirectSamples)));
                    }
                }
                else
                    side = Reflected(reflectance[i], DirectIrradiance(accelerator, sources, cache.Surfels[i],
                                                                      i, front, options, statistics, work,
                                                                      cancelled));
            }
        }
        result.Total = result.Direct;
        result.CompletedOrders = 1;
        auto previous = result.Direct;
        for (uint32_t bounce = 0; bounce < options.IndirectBounces; ++bounce)
        {
            std::vector<PointGiRadiance> next(cache.Surfels.size());
            for (uint32_t i = 0; i < cache.Surfels.size(); ++i)
            {
                PollLighting(work, cancelled);
                if (!subsurface[i] && reflectance[i] == Vector{})
                    continue;
                for (bool front : {true, false})
                {
                    auto& side = front ? next[i].Front : next[i].Back;
                    if (subsurface[i])
                    {
                        if (!front) continue;
                        const auto& surfel = cache.Surfels[i];
                        const SurfaceHit entry = SurfelHit(surfel);
                        Ray ray{surfel.Position + surfel.GeometryNormal * RayEpsilon,
                                -surfel.GeometryNormal, RayEpsilon, MaximumRayDistance};
                        for (uint32_t sample = 0; sample < options.GatherSamples; ++sample)
                        {
                            const auto random = BakeSamples(i, true, sample, options.GatherSamples, options);
                            side = Plus(side, Times(SubsurfaceIncident(scene, accelerator, cache, sources,
                                &previous, entry, surfel.Position, ray, false, random, options, statistics,
                                work, cancelled), 1 / double(options.GatherSamples)));
                        }
                    }
                    else
                        side = Reflected(reflectance[i], Gather(cache, previous, sources, cache.Surfels[i], i,
                                                                front, options, statistics, work, cancelled));
                    auto& total = front ? result.Total[i].Front : result.Total[i].Back;
                    total = Plus(total, side);
                    RequireColor(total);
                }
            }
            previous.swap(next);
            ++result.CompletedOrders;
        }
        if (cancelled && cancelled())
            throw std::runtime_error("Point GI lighting bake was cancelled.");
        output = std::move(result);
        error.clear();
        return true;
    }
    catch (const std::exception& exception)
    {
        error = exception.what();
        return false;
    }
}
bool RenderPointGiFrame(const Scene& scene, const SceneAccelerator& accelerator, const CameraFrame& camera,
                        const RenderSettings& settings, LinearFilm& output, AtomicStatistics& statistics,
                        const Detail::CancellationProbe& cancelled, const PointGiDispatcher& dispatch,
                        std::string& error)
{
    output = {};
    try
    {
        if (!ValidateTileSettings(settings, error))
            return false;
        if (settings.Integrator != Algorithm::PointBasedGlobalIllumination || !dispatch ||
            !settings.SamplesPerPixel || settings.MaxReflectionBounces > MaximumDiffuseBounces ||
            settings.Adaptive.Enabled || !settings.ShadowsEnabled || settings.ShadowStrength != 1)
            throw std::runtime_error(
                "Point GI requires bounded diffuse orders, fixed camera samples and full visibility.");
        if (!Detail::ValidateMaterialTransport(scene, BackendType::CPU, settings.Integrator, error))
            return false;
        const auto sources = PrepareEmitters(scene);
        for (const auto& source : sources.Sources)
            if (!(source.Flags & EmitterCastsShadows))
                throw std::runtime_error("Point GI requires shadows on every source.");
        PointGiLightingOptions options;
        options.DirectSamples = settings.PointGi.DirectSamples;
        options.GatherSamples = settings.PointGi.GatherSamples;
        options.Seed = settings.PointGi.Seed ^ settings.SamplingSeed;
        options.SamplingSeed = settings.SamplingSeed;
        options.Sequence = settings.Sampling;
        options.ImportanceSampling = settings.Transport.ImportanceSampling;
        PointGiSurfaceCache cache;
        PointGiLighting lighting;
        if (settings.MaxReflectionBounces)
        {
            options.IndirectBounces = settings.MaxReflectionBounces - 1;
            if (!BuildPointGiSurfaceCache(scene,
                                          {settings.PointGi.MaximumArea, settings.PointGi.MaximumSurfels},
                                          cache, cancelled, error) ||
                !BakePointGiDiffuse(scene, accelerator, cache, options, lighting, statistics, cancelled,
                                    error))
                return false;
        }
        LinearFilm film(settings.Width, settings.Height);
        std::atomic<uint32_t> completed{0};
        dispatch(settings.Width * settings.Height, [&](uint32_t pixel) {
            const uint32_t x = pixel % settings.Width, y = pixel / settings.Width;
            auto& target = film.Pixels[pixel];
            uint64_t work = 0;
            for (uint32_t sample = 0; sample < settings.SamplesPerPixel; ++sample)
            {
                if (cancelled && cancelled())
                    return;
                auto random = MakeSampleContext(settings, x, y, sample);
                Ray ray = MakeCameraRay(camera, settings, random);
                if (camera.Lens.Radius > 0)
                    ApplyThinLens(ray, camera, Sample2D(random, LensSampleDimension));
                // Decorrelate camera transport from the fixed surface bake. The
                // raster/lens streams remain identical to the other cameras.
                random.PixelSeed = Hash(random.PixelSeed ^ Hash(settings.PointGi.Seed) ^ GatherDomain);
                ++statistics.PrimaryRays;
                uint64_t tests = 0;
                SurfaceHit hit;
                const bool surface = accelerator.Intersect(ray, hit, tests);
                const bool sampledProxy = surface && hit.MaterialIndex < scene.Materials.size() &&
                                          scene.Materials[hit.MaterialIndex].SampledEmitter;
                double limit = surface ? std::min(double(ray.MaxDistance), double(hit.Distance) +
                                                                         (sampledProxy ? RayEpsilon : 0))
                                       : ray.MaxDistance;
                const auto rectangle = FindRectangle(sources, ray, limit, tests);
                statistics.PrimitiveTests += tests;
                Vector value{};
                if (rectangle.Source)
                {
                    if (rectangle.Front)
                        value = Times(V(rectangle.Source->Strength),
                                      1 / (double(rectangle.Source->Width) * rectangle.Source->Height));
                }
                else if (!surface)
                    value = V(Background(scene, ray));
                else
                {
                    const auto& material = scene.Materials.at(hit.MaterialIndex);
                    value = V(material.Emission);
                    PointGiSurfel receiver;
                    receiver.Position = ray.Origin + ray.Direction * hit.Distance;
                    receiver.GeometryNormal = hit.FrontFace ? hit.GeometricNormal : -hit.GeometricNormal;
                    receiver.ShadingNormal = hit.Normal;
                    if (Dot(receiver.GeometryNormal, receiver.ShadingNormal) < 0)
                        receiver.ShadingNormal = -receiver.ShadingNormal;
                    if (Dot(receiver.ShadingNormal, -ray.Direction) <= 0)
                        receiver.ShadingNormal = receiver.GeometryNormal;
                    receiver.Material = hit.MaterialIndex;
                    receiver.Primitive = hit.PrimitiveIndex;
                    receiver.Color = hit.Color;
                    receiver.HasTexCoord = hit.HasTexCoord;
                    receiver.TexCoord = hit.TexCoord;
                    if (material.SubsurfaceMeanFreePath > 0)
                    {
                        const auto exit = TraceSubsurface(scene, accelerator, hit, receiver.Position,
                            ray, !hit.FrontFace, random, options.ImportanceSampling, statistics,
                            work, cancelled);
                        if (exit.Reached)
                        {
                            Vector irradiance = DirectIrradiance(accelerator, sources, exit.Receiver,
                                InvalidRayPrimitiveIndex, true, options, statistics, work, cancelled, &random);
                            if (settings.MaxReflectionBounces)
                                irradiance = Plus(irradiance, Gather(cache, lighting.Total, sources,
                                    exit.Receiver, InvalidRayPrimitiveIndex, true, options,
                                    statistics, work, cancelled, &random));
                            Vector contribution{};
                            for (uint32_t channel = 0; channel < Dimensions; ++channel)
                                contribution[channel] = exit.Throughput[channel] * irradiance[channel] / SurfacePi;
                            value = Plus(value, contribution);
                        }
                    }
                    else
                    {
                        const Vector rho = Reflectance(scene, receiver);
                        if (rho != Vector{} && Dot(receiver.ShadingNormal, -ray.Direction) > 0)
                        {
                            Vector irradiance =
                                DirectIrradiance(accelerator, sources, receiver, InvalidRayPrimitiveIndex,
                                                 true, options, statistics, work, cancelled, &random);
                            if (settings.MaxReflectionBounces)
                                irradiance = Plus(irradiance, Gather(cache, lighting.Total, sources, receiver,
                                                                     InvalidRayPrimitiveIndex, true, options,
                                                                     statistics, work, cancelled, &random));
                            value = Plus(value, Reflected(rho, irradiance));
                        }
                    }
                }
                RequireColor(value);
                AccumulateFilmSample(target, Stored(value));
                RequireColor(V(target.MeanRadiance));
                if (!std::isfinite(target.LuminanceM2))
                    throw std::runtime_error("Point GI camera moments exceed finite film storage.");
            }
            ++statistics.CompletedPixels;
            ++completed;
        });
        if (cancelled && cancelled())
            throw std::runtime_error("Point GI camera rendering was cancelled.");
        if (completed != settings.Width * settings.Height)
            throw std::runtime_error("Point GI camera dispatcher did not finish the image.");
        output = std::move(film);
        error.clear();
        return true;
    }
    catch (const std::exception& exception)
    {
        error = exception.what();
        return false;
    }
}
} // namespace Arvo::RayTracing::Cpu
