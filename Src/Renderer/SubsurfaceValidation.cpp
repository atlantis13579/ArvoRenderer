#include "Renderer/SubsurfaceValidation.h"
#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <stdexcept>
#include <tuple>

namespace Arvo::RayTracing
{
namespace
{
constexpr size_t NoRegion = std::numeric_limits<size_t>::max();
using Point = std::array<double, 3>;
using Transform = std::array<float, 12>;
constexpr Transform Identity{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
double BoundaryClearance(double coordinateScale)
{
    // All five PT backends currently use this world-space surface offset.
    // Leave room for both interfaces, in addition to transform roundoff.
    constexpr double SurfaceRayOffset = 1.e-4, TransformUlps = 32;
    return 2 * SurfaceRayOffset + TransformUlps * std::numeric_limits<float>::epsilon() * coordinateScale;
}
struct Bounds
{
    Point Low{INFINITY, INFINITY, INFINITY}, High{-INFINITY, -INFINITY, -INFINITY};
    void Add(Point point)
    {
        for (size_t axis = 0; axis < point.size(); ++axis)
        {
            if (!std::isfinite(point[axis]))
                throw std::runtime_error("SSS geometry must be finite");
            Low[axis] = std::min(Low[axis], point[axis]);
            High[axis] = std::max(High[axis], point[axis]);
        }
    }
};
Point Position(Float3 p)
{
    return {p.X, p.Y, p.Z};
}
Point World(Point p, const Transform& transform)
{
    Point result{};
    constexpr size_t AffineStride = 4;
    for (size_t row = 0; row < result.size(); ++row)
        result[row] = transform[row * AffineStride] * p[0] + transform[row * AffineStride + 1] * p[1] +
                      transform[row * AffineStride + 2] * p[2] + transform[row * AffineStride + 3];
    return result;
}
Point Difference(Point a, Point b)
{
    for (size_t axis = 0; axis < a.size(); ++axis)
        a[axis] -= b[axis];
    return a;
}
double Determinant(Point a, Point b, Point c)
{
    return a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) +
           a[2] * (b[0] * c[1] - b[1] * c[0]);
}
Point CrossProduct(Point a, Point b)
{
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
struct Primitive
{
    Bounds Box;
    size_t Region = NoRegion;
    std::vector<Point> Vertices, Edges, Normals;
};
struct Region
{
    Bounds Box;
    uint32_t Material = 0;
    std::vector<Point> Vertices, Edges, Normals;
};
struct Edge
{
    size_t Triangle = 0;
    bool Forward = false;
    unsigned Count = 0;
};
void Require(bool valid, const char* message)
{
    if (!valid)
        throw std::runtime_error(message);
}
bool InteriorOverlap(const Bounds& region, const Bounds& primitive)
{
    // Foreign surfaces must have a positive gap: coincident contacts do not
    // define which material interface a single nearest-hit query should cross.
    for (size_t axis = 0; axis < region.Low.size(); ++axis)
    {
        const double scale = std::max({1.0, std::abs(region.Low[axis]), std::abs(region.High[axis])});
        const double tolerance = BoundaryClearance(scale);
        if (primitive.High[axis] < region.Low[axis] - tolerance ||
            primitive.Low[axis] > region.High[axis] + tolerance)
            return false;
    }
    return true;
}
bool Separated(Point axis, const Region& region, const Primitive& primitive)
{
    const double square = axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2];
    if (!(square > std::numeric_limits<double>::min()))
        return false;
    const double inverseLength = 1 / std::sqrt(square);
    for (auto& value : axis)
        value *= inverseLength;
    const auto interval = [&](const auto& points) {
        std::pair<double, double> range{INFINITY, -INFINITY};
        for (const auto& point : points)
        {
            const double distance = point[0] * axis[0] + point[1] * axis[1] + point[2] * axis[2];
            range.first = std::min(range.first, distance);
            range.second = std::max(range.second, distance);
        }
        return range;
    };
    const auto a = interval(region.Vertices), b = interval(primitive.Vertices);
    const double scale =
        std::max({1.0, std::abs(a.first), std::abs(a.second), std::abs(b.first), std::abs(b.second)});
    const double tolerance = BoundaryClearance(scale);
    return a.second < b.first - tolerance || b.second < a.first - tolerance;
}
bool PotentialInteriorOverlap(const Region& region, const Primitive& primitive)
{
    if (!InteriorOverlap(region.Box, primitive.Box))
        return false;
    // A separating-axis certificate avoids rejecting the two rotated Cornell
    // boxes merely because their world AABBs overlap. Nonconvex shapes and
    // affine spheres retain conservative convex bounds, never false admission.
    for (const auto& normal : region.Normals)
        if (Separated(normal, region, primitive))
            return false;
    for (const auto& normal : primitive.Normals)
        if (Separated(normal, region, primitive))
            return false;
    for (const auto& edge : region.Edges)
        for (const auto& other : primitive.Edges)
            if (Separated(CrossProduct(edge, other), region, primitive))
                return false;
    return true;
}
void AddPrimitive(Primitive primitive, std::vector<Primitive>& primitives, std::vector<Region>& regions)
{
    if (primitive.Region != NoRegion)
    {
        auto& region = regions[primitive.Region];
        region.Box.Add(primitive.Box.Low);
        region.Box.Add(primitive.Box.High);
        region.Vertices.insert(region.Vertices.end(), primitive.Vertices.begin(), primitive.Vertices.end());
        region.Edges.insert(region.Edges.end(), primitive.Edges.begin(), primitive.Edges.end());
        region.Normals.insert(region.Normals.end(), primitive.Normals.begin(), primitive.Normals.end());
    }
    primitives.push_back(std::move(primitive));
}
void AppendGeometry(const SceneGeometry& geometry, const Scene& scene, const Transform& transform,
                    std::vector<Primitive>& primitives, std::vector<Region>& regions)
{
    const auto subsurface = [&](uint32_t material) {
        Require(material < scene.Materials.size(), "SSS validation found an invalid material index");
        return scene.Materials[material].SubsurfaceMeanFreePath > 0;
    };
    std::vector<size_t> parent(geometry.Triangles.size());
    std::iota(parent.begin(), parent.end(), 0);
    const auto root = [&](size_t triangle) {
        while (parent[triangle] != triangle)
        {
            parent[triangle] = parent[parent[triangle]];
            triangle = parent[triangle];
        }
        return triangle;
    };
    std::map<Point, size_t> vertices;
    std::map<std::tuple<uint32_t, size_t, size_t>, Edge> edges;
    for (size_t index = 0; index < geometry.Triangles.size(); ++index)
    {
        const auto& triangle = geometry.Triangles[index];
        if (!subsurface(triangle.MaterialIndex))
            continue;
        const Point points[]{Position(triangle.Position0), Position(triangle.Position1),
                             Position(triangle.Position2)};
        size_t ids[std::size(points)];
        Bounds finite;
        for (size_t corner = 0; corner < std::size(points); ++corner)
        {
            finite.Add(points[corner]);
            const auto inserted = vertices.emplace(points[corner], vertices.size());
            ids[corner] = inserted.first->second;
        }
        for (size_t corner = 0; corner < std::size(points); ++corner)
        {
            const size_t a = ids[corner], b = ids[(corner + 1) % std::size(points)];
            Require(a != b, "SSS triangles must have distinct vertices");
            auto& edge = edges[{triangle.MaterialIndex, std::min(a, b), std::max(a, b)}];
            if (!edge.Count)
                edge = {index, a < b, 1};
            else
            {
                Require(edge.Count == 1 && edge.Forward != (a < b),
                        "SSS requires consistently oriented manifold surfaces; remove duplicate faces");
                ++edge.Count;
                parent[root(index)] = root(edge.Triangle);
            }
        }
    }
    for (const auto& edge : edges)
        Require(edge.second.Count == 2,
                "SSS requires a closed solid with one material on every boundary face");
    struct Component
    {
        size_t Region;
        Point Reference;
        double SignedVolume = 0;
    };
    std::map<size_t, Component> components;
    for (size_t index = 0; index < geometry.Triangles.size(); ++index)
    {
        const auto& triangle = geometry.Triangles[index];
        Primitive primitive;
        const Point points[]{Position(triangle.Position0), Position(triangle.Position1),
                             Position(triangle.Position2)};
        for (const auto& point : points)
        {
            const auto world = World(point, transform);
            primitive.Box.Add(world);
            primitive.Vertices.push_back(world);
        }
        for (size_t corner = 0; corner < primitive.Vertices.size(); ++corner)
            primitive.Edges.push_back(Difference(primitive.Vertices[(corner + 1) % primitive.Vertices.size()],
                                                 primitive.Vertices[corner]));
        primitive.Normals.push_back(CrossProduct(primitive.Edges[0], primitive.Edges[1]));
        if (subsurface(triangle.MaterialIndex))
        {
            const auto inserted = components.emplace(root(index), Component{regions.size(), points[0], 0});
            if (inserted.second)
                regions.push_back({{}, triangle.MaterialIndex});
            auto& component = inserted.first->second;
            primitive.Region = component.Region;
            component.SignedVolume += Determinant(Difference(points[0], component.Reference),
                                                  Difference(points[1], component.Reference),
                                                  Difference(points[2], component.Reference));
        }
        AddPrimitive(std::move(primitive), primitives, regions);
    }
    for (const auto& entry : components)
        Require(entry.second.SignedVolume > 0 && std::isfinite(entry.second.SignedVolume),
                "SSS requires outward-facing closed surfaces with positive volume");
    for (const auto& sphere : geometry.Spheres)
    {
        Primitive primitive;
        if (subsurface(sphere.MaterialIndex))
        {
            Require(sphere.Radius > 0 && std::isfinite(sphere.Radius),
                    "SSS spheres require a positive finite radius");
            primitive.Region = regions.size();
            regions.push_back({{}, sphere.MaterialIndex});
        }
        // Transformed local boxes conservatively enclose affine sphere instances.
        for (unsigned corner = 0; corner < (1u << 3); ++corner)
        {
            Point point = Position(sphere.Center);
            for (unsigned axis = 0; axis < point.size(); ++axis)
                point[axis] += ((corner >> axis) & 1 ? 1 : -1) * double(sphere.Radius);
            const auto world = World(point, transform);
            primitive.Box.Add(world);
            primitive.Vertices.push_back(world);
        }
        for (unsigned axis = 0; axis < 3; ++axis)
            primitive.Edges.push_back(
                Difference(primitive.Vertices[size_t{1} << axis], primitive.Vertices[0]));
        for (size_t axis = 0; axis < primitive.Edges.size(); ++axis)
            primitive.Normals.push_back(
                CrossProduct(primitive.Edges[axis], primitive.Edges[(axis + 1) % primitive.Edges.size()]));
        AddPrimitive(std::move(primitive), primitives, regions);
    }
}
} // namespace
bool HasSubsurfaceMaterials(const Scene& scene)
{
    return std::any_of(scene.Materials.begin(), scene.Materials.end(),
                       [](const auto& material) { return material.SubsurfaceMeanFreePath != 0; });
}
bool ValidateSubsurfaceScene(const Scene& scene, const RenderSettings& settings, std::string& error)
{
    if (!HasSubsurfaceMaterials(scene))
        return true;
    try
    {
        Require(settings.Integrator == Algorithm::PathTracing ||
                    settings.Integrator == Algorithm::DistributedRayTracing ||
                    settings.Integrator == Algorithm::BidirectionalPathTracing ||
                    settings.Integrator == Algorithm::MetropolisLightTransport ||
                    settings.Integrator == Algorithm::PhotonMapping ||
                    settings.Integrator == Algorithm::ProgressivePhotonMapping ||
                    settings.Integrator == Algorithm::VertexConnectionMerging ||
                    settings.Integrator == Algorithm::PointBasedGlobalIllumination,
                "Subsurface random walks are not implemented for this integrator");
        Require(settings.ShadowsEnabled && settings.ShadowStrength == 1,
                "SSS requires full geometric shadow visibility");
        Require(!HasSceneMotion(scene),
                "SSS solid validation currently requires static geometry; camera motion is supported");
        for (const auto& material : scene.Materials)
            if (!ValidateMaterialScattering(material, error))
                return false;
        for (const auto& light : scene.PointLights)
            Require(light.CastShadows, "SSS lights must cast shadows");
        for (const auto& light : scene.DirectionalLights)
            Require(light.CastShadows, "SSS lights must cast shadows");
        for (const auto& light : scene.RectAreaLights)
            Require(light.CastShadows, "SSS lights must cast shadows");
        std::vector<Primitive> primitives;
        std::vector<Region> regions;
        AppendGeometry(scene, scene, Identity, primitives, regions);
        for (const auto& instance : scene.Instances)
        {
            Require(instance.GeometryIndex < scene.Geometries.size() &&
                        scene.Geometries[instance.GeometryIndex],
                    "SSS validation found an invalid instance geometry");
            AppendGeometry(*scene.Geometries[instance.GeometryIndex], scene, instance.ObjectToWorld,
                           primitives, regions);
        }
        for (size_t index = 0; index < regions.size(); ++index)
            for (const auto& primitive : primitives)
                if (primitive.Region != index && PotentialInteriorOverlap(regions[index], primitive))
                    throw std::runtime_error(
                        "SSS solid overlaps or is too close to foreign geometry (material " +
                        std::to_string(regions[index].Material) + ")");
        return true;
    }
    catch (const std::exception& failure)
    {
        error = failure.what();
        return false;
    }
}
} // namespace Arvo::RayTracing
