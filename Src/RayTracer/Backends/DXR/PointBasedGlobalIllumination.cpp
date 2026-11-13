#include "PointGiGeometry.h"
#include "InstancingBuilder.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>

namespace Arvo::RayTracing::DxrPointGi
{
namespace
{
// Independent DXR surface discretization. Uniform barycentric tessellation
// gives n^2 equal-area triangle patches. Sphere quadrature uses equal solid
// angles and the affine cofactor Jacobian. Reflections are baked by DXR itself;
// this host stage creates only geometry and a balanced disk-bounds hierarchy.
// Christensen (2008), "Point-Based Approximate Color Bleeding":
// https://graphics.pixar.com/library/PointBasedColorBleeding/paper.pdf
constexpr double Pi = 3.1415926535897932384626433832795;
constexpr uint32_t LeafSize = 8, CancellationInterval = 1024;
struct Wide
{
    double X = 0, Y = 0, Z = 0;
    double Axis(uint32_t axis) const { return axis == 0 ? X : axis == 1 ? Y : Z; }
    Wide operator+(Wide r) const { return {X + r.X, Y + r.Y, Z + r.Z}; }
    Wide operator-(Wide r) const { return {X - r.X, Y - r.Y, Z - r.Z}; }
    Wide operator*(double scale) const { return {X * scale, Y * scale, Z * scale}; }
};
Wide WideOf(Float3 v)
{
    return {v.X, v.Y, v.Z};
}
Wide WideOf(Vector v)
{
    return {v.X, v.Y, v.Z};
}
Vector Pack(Wide v)
{
    return {float(v.X), float(v.Y), float(v.Z)};
}
Wide CrossWide(Wide a, Wide b)
{
    return {a.Y * b.Z - a.Z * b.Y, a.Z * b.X - a.X * b.Z, a.X * b.Y - a.Y * b.X};
}
double DotWide(Wide a, Wide b)
{
    return a.X * b.X + a.Y * b.Y + a.Z * b.Z;
}
double Magnitude(Wide v)
{
    return std::hypot(v.X, v.Y, v.Z);
}
Wide Direction(Wide v)
{
    double length = Magnitude(v);
    return length > 0 ? v * (1 / length) : Wide{};
}
bool Representable(Vector v)
{
    return std::isfinite(v.X) && std::isfinite(v.Y) && std::isfinite(v.Z);
}
struct Affine
{
    Wide Columns[3]{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}, Offset{};
    Wide Cofactors[3]{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    double Sign = 1;
    explicit Affine(const GeometryInstance* instance)
    {
        if (!instance)
            return;
        const auto& m = instance->ObjectToWorld;
        for (uint32_t column = 0; column < 3; ++column)
            Columns[column] = {m[column], m[column + 4], m[column + 8]};
        Offset = {m[3], m[7], m[11]};
        Cofactors[0] = CrossWide(Columns[1], Columns[2]);
        Cofactors[1] = CrossWide(Columns[2], Columns[0]);
        Cofactors[2] = CrossWide(Columns[0], Columns[1]);
        Sign = DotWide(Columns[0], Cofactors[0]) < 0 ? -1 : 1;
    }
    Wide Point(Wide p) const { return Offset + Columns[0] * p.X + Columns[1] * p.Y + Columns[2] * p.Z; }
    Wide Vector(Wide p) const { return Columns[0] * p.X + Columns[1] * p.Y + Columns[2] * p.Z; }
    Wide AreaVector(Wide n) const { return Cofactors[0] * n.X + Cofactors[1] * n.Y + Cofactors[2] * n.Z; }
    Wide Normal(Wide n) const { return Direction(AreaVector(n)) * Sign; }
    double MaximumAreaScale() const
    {
        double largestRow = 0, largestColumn = 0;
        for (uint32_t row = 0; row < 3; ++row)
            largestRow =
                std::max(largestRow, std::abs(Cofactors[0].Axis(row)) + std::abs(Cofactors[1].Axis(row)) +
                                         std::abs(Cofactors[2].Axis(row)));
        for (const auto& column : Cofactors)
            largestColumn =
                std::max(largestColumn, std::abs(column.X) + std::abs(column.Y) + std::abs(column.Z));
        return std::sqrt(largestRow * largestColumn);
    }
};
class CacheBuilder
{
  public:
    CacheBuilder(const Scene& scene, const PointGiOptions& options,
                 const Detail::CancellationProbe& cancelled, std::string& error)
        : SceneData(scene), Options(options), Cancelled(cancelled), Error(error)
    {
    }
    bool Build(SurfaceCache& result)
    {
        if (!AddGeometry(SceneData, Affine(nullptr)))
            return false;
        for (const auto& instance : SceneData.Instances)
            if (!AddGeometry(*SceneData.Geometries[instance.GeometryIndex], Affine(&instance)))
                return false;
        Cache.Indices.resize(Cache.Surfaces.size());
        std::iota(Cache.Indices.begin(), Cache.Indices.end(), 0u);
        if (!Cache.Indices.empty() && BuildNode(0, uint32_t(Cache.Indices.size())) == Invalid)
            return false;
        if (Cancelled && Cancelled())
        {
            Error = "DXR Point GI geometry cancelled";
            return false;
        }
        result = std::move(Cache);
        Error.clear();
        return true;
    }

  private:
    bool Tick()
    {
        if (Iterations++ % CancellationInterval == 0 && Cancelled && Cancelled())
        {
            Error = "DXR Point GI geometry cancelled";
            return false;
        }
        return true;
    }
    bool ReservePopulation(double count) const
    {
        if (!(count >= 1) || !std::isfinite(count) || count > Options.MaximumSurfels - Cache.Surfaces.size())
        {
            Error = "DXR Point GI surface budget exceeded; increase patch area or budget";
            return false;
        }
        return true;
    }
    bool Emit(Surface surface)
    {
        if (!Representable(surface.Position) || !Representable(surface.Geometry) ||
            !Representable(surface.Shading) || !Representable(surface.Color) ||
            !std::isfinite(surface.TexCoord[0]) || !std::isfinite(surface.TexCoord[1]) ||
            !(surface.Area > 0) || !std::isfinite(surface.Area) || !(surface.RadiusSquared > 0) ||
            !std::isfinite(surface.RadiusSquared))
        {
            Error = "DXR Point GI surface has nonrepresentable attributes";
            return false;
        }
        Cache.Surfaces.push_back(surface);
        return true;
    }
    bool AddTriangle(const Triangle& triangle, const Affine& transform)
    {
        const Wide worldA = transform.Point(WideOf(triangle.Position0));
        // Differential area is translation invariant, including at coordinates
        // whose spacing is larger than the transformed triangle's edge lengths.
        const Wide edgeB = transform.Vector(WideOf(triangle.Position1) - WideOf(triangle.Position0));
        const Wide edgeC = transform.Vector(WideOf(triangle.Position2) - WideOf(triangle.Position0));
        const Wide cross = CrossWide(edgeB, edgeC);
        const double area = Magnitude(cross) / 2;
        if (area == 0)
            return true;
        if (!std::isfinite(area))
        {
            Error = "DXR Point GI triangle area is nonfinite";
            return false;
        }
        double divisions = std::ceil(std::sqrt(area / Options.MaximumArea));
        if (area / (divisions * divisions) > Options.MaximumArea)
            ++divisions;
        if (!ReservePopulation(divisions * divisions))
            return false;
        const uint32_t count = uint32_t(divisions);
        const Wide geometric = Direction(cross) * transform.Sign;
        uint32_t patchIndex = 0;
        const auto patch = [&](double baryB, double baryC) {
            if (!Tick())
                return false;
            double baryA = 1 - baryB - baryC;
            Surface surface{};
            surface.Position = Pack(worldA + edgeB * baryB + edgeC * baryC);
            surface.Geometry = Pack(geometric);
            Wide normal = geometric;
            if (triangle.SmoothNormals)
                normal =
                    transform.Normal(WideOf(triangle.Normal0) * baryA + WideOf(triangle.Normal1) * baryB +
                                     WideOf(triangle.Normal2) * baryC);
            if (!(Magnitude(normal) > 0))
                normal = geometric;
            if (DotWide(normal, geometric) < 0)
                normal = normal * -1;
            surface.Shading = Pack(normal);
            surface.Color = Pack(WideOf(triangle.Color0) * baryA + WideOf(triangle.Color1) * baryB +
                                 WideOf(triangle.Color2) * baryC);
            surface.TexCoord[0] =
                float(double(triangle.TexCoord0.X) * baryA + double(triangle.TexCoord1.X) * baryB +
                      double(triangle.TexCoord2.X) * baryC);
            surface.TexCoord[1] =
                float(double(triangle.TexCoord0.Y) * baryA + double(triangle.TexCoord1.Y) * baryB +
                      double(triangle.TexCoord2.Y) * baryC);
            surface.HasTexCoord = 1;
            surface.Material = triangle.MaterialIndex;
            surface.Primitive = uint32_t(Primitive);
            surface.Patch = patchIndex++;
            surface.Area = area / (divisions * divisions);
            surface.RadiusSquared = surface.Area / Pi;
            return Emit(surface);
        };
        for (uint32_t row = 0; row < count; ++row)
            for (uint32_t column = 0; column < count - row; ++column)
            {
                constexpr double FirstCentroid = 1.0 / 3, SecondCentroid = 2.0 / 3;
                if (!patch((double(row) + FirstCentroid) / divisions,
                           (double(column) + FirstCentroid) / divisions))
                    return false;
                if (row + column + 1 < count && !patch((double(row) + SecondCentroid) / divisions,
                                                       (double(column) + SecondCentroid) / divisions))
                    return false;
            }
        return true;
    }
    bool AddSphere(const Sphere& sphere, const Affine& transform)
    {
        if (!(sphere.Radius > 0) || !std::isfinite(sphere.Radius))
        {
            Error = "DXR Point GI sphere radius must be positive and finite";
            return false;
        }
        const double radiusSquared = double(sphere.Radius) * sphere.Radius;
        const double bound = 4 * Pi * radiusSquared * transform.MaximumAreaScale();
        const double population = std::max(1.0, std::ceil(bound / Options.MaximumArea));
        if (!ReservePopulation(population))
            return false;
        const uint32_t count = uint32_t(population);
        const double goldenTurn = (3 - std::sqrt(5.0)) / 2;
        for (uint32_t i = 0; i < count; ++i)
        {
            if (!Tick())
                return false;
            const double z = 1 - 2 * (double(i) + 0.5) / population;
            const double radial = std::sqrt(std::max(0.0, 1 - z * z));
            const double angle = 2 * Pi * std::fmod(double(i) * goldenTurn, 1.0);
            const Wide direction{radial * std::cos(angle), radial * std::sin(angle), z};
            Surface surface{};
            surface.Position = Pack(transform.Point(WideOf(sphere.Center) + direction * sphere.Radius));
            surface.Geometry = surface.Shading = Pack(transform.Normal(direction));
            surface.Color = {1, 1, 1};
            surface.Material = sphere.MaterialIndex;
            surface.Primitive = uint32_t(Primitive);
            surface.Patch = i;
            surface.Area = (4 * Pi * radiusSquared / population) * Magnitude(transform.AreaVector(direction));
            surface.RadiusSquared = surface.Area / Pi;
            if (!Emit(surface))
                return false;
        }
        return true;
    }
    bool AddGeometry(const SceneGeometry& geometry, const Affine& transform)
    {
        for (const auto& triangle : geometry.Triangles)
        {
            if (!Tick())
                return false;
            if (Primitive >= Invalid || triangle.MaterialIndex >= SceneData.Materials.size())
            {
                Error = "DXR Point GI triangle identifier is invalid";
                return false;
            }
            if (!AddTriangle(triangle, transform))
                return false;
            ++Primitive;
        }
        for (const auto& sphere : geometry.Spheres)
        {
            if (Primitive >= Invalid || sphere.MaterialIndex >= SceneData.Materials.size())
            {
                Error = "DXR Point GI sphere identifier is invalid";
                return false;
            }
            if (!AddSphere(sphere, transform))
                return false;
            ++Primitive;
        }
        return true;
    }
    uint32_t BuildNode(uint32_t begin, uint32_t end)
    {
        if (!Tick())
            return Invalid;
        Node node{};
        for (uint32_t axis = 0; axis < 3; ++axis)
        {
            node.Minimum[axis] = std::numeric_limits<double>::infinity();
            node.Maximum[axis] = -node.Minimum[axis];
        }
        for (uint32_t cursor = begin; cursor < end; ++cursor)
        {
            if (!Tick())
                return Invalid;
            const auto& surface = Cache.Surfaces[Cache.Indices[cursor]];
            const Wide center = WideOf(surface.Position);
            const double radius = std::sqrt(surface.RadiusSquared);
            for (uint32_t axis = 0; axis < 3; ++axis)
            {
                node.Minimum[axis] =
                    std::min(node.Minimum[axis], std::nextafter(center.Axis(axis) - radius,
                                                                -std::numeric_limits<double>::infinity()));
                node.Maximum[axis] =
                    std::max(node.Maximum[axis], std::nextafter(center.Axis(axis) + radius,
                                                                std::numeric_limits<double>::infinity()));
            }
        }
        const uint32_t identity = uint32_t(Cache.Nodes.size());
        Cache.Nodes.push_back(node);
        if (end - begin <= LeafSize)
        {
            Cache.Nodes[identity].Begin = begin;
            Cache.Nodes[identity].Count = end - begin;
        }
        else
        {
            uint32_t axis = 0;
            for (uint32_t dimension = 1; dimension < 3; ++dimension)
                if (node.Maximum[dimension] - node.Minimum[dimension] >
                    node.Maximum[axis] - node.Minimum[axis])
                    axis = dimension;
            const uint32_t middle = begin + (end - begin) / 2;
            std::nth_element(Cache.Indices.begin() + begin, Cache.Indices.begin() + middle,
                             Cache.Indices.begin() + end, [&](uint32_t a, uint32_t b) {
                                 const double x = WideOf(Cache.Surfaces[a].Position).Axis(axis);
                                 const double y = WideOf(Cache.Surfaces[b].Position).Axis(axis);
                                 return x < y || (x == y && a < b);
                             });
            const uint32_t left = BuildNode(begin, middle), right = BuildNode(middle, end);
            if (left == Invalid || right == Invalid)
                return Invalid;
            Cache.Nodes[identity].Left = left;
            Cache.Nodes[identity].Right = right;
        }
        return identity;
    }
    const Scene& SceneData;
    const PointGiOptions& Options;
    const Detail::CancellationProbe& Cancelled;
    std::string& Error;
    SurfaceCache Cache;
    uint64_t Primitive = 0, Iterations = 0;
};
} // namespace
bool BuildSurfaceCache(const Scene& scene, const PointGiOptions& options,
                       const Detail::CancellationProbe& cancelled, SurfaceCache& result, std::string& error)
{
    if (!ValidatePointGiOptions(options, error) || !DxrInstances::Validate(scene, error))
        return false;
    return CacheBuilder(scene, options, cancelled, error).Build(result);
}
} // namespace Arvo::RayTracing::DxrPointGi
