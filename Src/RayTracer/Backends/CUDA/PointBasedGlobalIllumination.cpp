#include "PointGiGeometry.h"
#include "InstancingBuilder.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace Arvo::RayTracing::CudaPointGi
{
/*
Independent CUDA Point GI surface discretization. Christensen (2008),
"Point-Based Approximate Color Bleeding": a view-independent diffuse bake is
resolved by directional visibility before camera reconstruction. This backend
builds its own world-space disks and hierarchy on the host, then uploads them
for CUDA direct baking, Jacobi reflection orders and camera integration.

Triangle subdivision halves barycentric area at each longest-edge split.
Fibonacci sphere quadrature has weights dA = r^2 |cofactor(A) n| dOmega.
Both disk sides share one area. Mirrored transforms use inverse-transpose
normals, preserving traversal's front/back convention. Disk coverage remains
an approximation; nearest visible disks must not be summed through occluders.
*/
namespace
{
constexpr uint32_t Axes = 3, MatrixStride = 4, Corners = 3;
constexpr uint32_t MinimumSphereSamples = 32, LeafSize = 8, PollInterval = 1024, SplitLimit = 64;
constexpr double Pi = 3.1415926535897932384626433832795;
constexpr double GoldenTurn = 0.61803398874989484820;
using D3 = std::array<double, Axes>;
D3 Load(Float3 v)
{
    return {v.X, v.Y, v.Z};
}
D3 Load(Vector v)
{
    return {v.X, v.Y, v.Z};
}
D3 Add(D3 a, D3 b)
{
    return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}
D3 Sub(D3 a, D3 b)
{
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}
D3 Scale(D3 a, double b)
{
    return {a[0] * b, a[1] * b, a[2] * b};
}
double DotD(D3 a, D3 b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
D3 CrossD(D3 a, D3 b)
{
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
double Norm(D3 a)
{
    return std::hypot(a[0], a[1], a[2]);
}
Vector Store(D3 a)
{
    for (double component : a)
        if (!std::isfinite(component) || std::abs(component) > std::numeric_limits<float>::max())
            throw std::runtime_error("CUDA Point GI attributes exceed finite upload storage.");
    return {float(a[0]), float(a[1]), float(a[2])};
}
Vector Normal(D3 a)
{
    const double length = Norm(a);
    if (!(length > 0) || !std::isfinite(length))
        throw std::runtime_error("CUDA Point GI encountered an undefined surface normal.");
    return Store(Scale(a, 1 / length));
}
struct Frame
{
    std::array<D3, Axes> A, Cofactor;
    D3 Offset;
    double Sign;
    explicit Frame(const GeometryInstance& instance)
    {
        for (uint32_t row = 0; row < Axes; ++row)
        {
            Offset[row] = instance.ObjectToWorld[row * MatrixStride + Axes];
            for (uint32_t column = 0; column < Axes; ++column)
                A[column][row] = instance.ObjectToWorld[row * MatrixStride + column];
        }
        Cofactor = {CrossD(A[1], A[2]), CrossD(A[2], A[0]), CrossD(A[0], A[1])};
        const double determinant = DotD(A[0], Cofactor[0]);
        if (!std::isfinite(determinant) || determinant == 0)
            throw std::runtime_error("CUDA Point GI needs a finite invertible affine transform.");
        Sign = determinant < 0 ? -1 : 1;
    }
    D3 Product(const std::array<D3, Axes>& columns, D3 vector) const
    {
        return Add(Add(Scale(columns[0], vector[0]), Scale(columns[1], vector[1])),
                   Scale(columns[2], vector[2]));
    }
    D3 Direction(D3 v) const { return Product(A, v); }
    Vector Point(D3 v) const { return Store(Add(Offset, Direction(v))); }
    Vector OrientedNormal(D3 n) const { return Normal(Scale(Product(Cofactor, n), Sign)); }
};
struct TriangleCell
{
    std::array<D3, Corners> Bary{{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
    uint32_t Depth = 0;
};
class GeometryBuilder
{
    const Scene& World;
    const PointGiOptions& Options;
    const Detail::CancellationProbe& Cancel;
    uint64_t Work = 0;

  public:
    GeometryCache Cache;
    GeometryBuilder(const Scene& scene, const PointGiOptions& options,
                    const Detail::CancellationProbe& cancelled)
        : World(scene), Options(options), Cancel(cancelled)
    {
    }
    void Poll()
    {
        if (Work++ % PollInterval == 0 && Cancel && Cancel())
            throw std::runtime_error("CUDA Point GI geometry bake was cancelled.");
    }
    void Append(Surfel point)
    {
        Poll();
        if (Cache.Surfels.size() >= Options.MaximumSurfels)
            throw std::runtime_error(
                "CUDA Point GI surface budget is too small for the requested patch area.");
        if (!(point.Area > 0) || !std::isfinite(point.Area) || point.Material >= World.Materials.size())
            throw std::runtime_error("CUDA Point GI surface area or material is invalid.");
        point.Radius = std::sqrt(point.Area / Pi);
        Cache.Surfels.push_back(point);
    }
    void TrianglePoints(const Triangle& triangle, const Frame& frame, uint32_t primitive)
    {
        Poll();
        const D3 base = Load(triangle.Position0), a = Sub(Load(triangle.Position1), base),
                 b = Sub(Load(triangle.Position2), base);
        const D3 worldA = frame.Direction(a), worldB = frame.Direction(b);
        const double area = Norm(CrossD(worldA, worldB)) / 2;
        if (!std::isfinite(area))
            throw std::runtime_error("CUDA Point GI triangle area is not finite.");
        if (area == 0)
            return;
        const Vector ng = frame.OrientedNormal(CrossD(a, b));
        std::vector<TriangleCell> stack(1);
        uint32_t ordinal = 0;
        while (!stack.empty())
        {
            Poll();
            TriangleCell cell = stack.back();
            stack.pop_back();
            const double cellArea = std::ldexp(area, -int(cell.Depth));
            if (cellArea > Options.MaximumArea)
            {
                if (cell.Depth >= SplitLimit ||
                    Cache.Surfels.size() + stack.size() + 2 > Options.MaximumSurfels)
                    throw std::runtime_error(
                        "CUDA Point GI triangle refinement exhausted its finite budget.");
                uint32_t edge = 0;
                double longest = -1;
                for (uint32_t corner = 0; corner < Corners; ++corner)
                {
                    D3 weights = Sub(cell.Bary[(corner + 1) % Corners], cell.Bary[corner]);
                    double length = Norm(Add(Scale(worldA, weights[1]), Scale(worldB, weights[2])));
                    if (length > longest)
                    {
                        longest = length;
                        edge = corner;
                    }
                }
                const uint32_t next = (edge + 1) % Corners;
                D3 midpoint = Scale(Add(cell.Bary[edge], cell.Bary[next]), 0.5);
                TriangleCell left = cell, right = cell;
                left.Bary[next] = right.Bary[edge] = midpoint;
                left.Depth = right.Depth = cell.Depth + 1;
                stack.push_back(right);
                stack.push_back(left);
                continue;
            }
            const D3 w = Scale(Add(Add(cell.Bary[0], cell.Bary[1]), cell.Bary[2]), 1.0 / Corners);
            const auto blend = [&](Float3 x, Float3 y, Float3 z) {
                return Add(Add(Scale(Load(x), w[0]), Scale(Load(y), w[1])), Scale(Load(z), w[2]));
            };
            Surfel point{};
            point.Position = frame.Point(Add(base, Add(Scale(a, w[1]), Scale(b, w[2]))));
            point.GeometryNormal = point.ShadingNormal = ng;
            if (triangle.SmoothNormals)
            {
                const D3 smooth = blend(triangle.Normal0, triangle.Normal1, triangle.Normal2);
                if (Norm(smooth) > 0)
                    point.ShadingNormal = frame.OrientedNormal(smooth);
                if (DotD(Load(point.ShadingNormal), Load(ng)) < 0)
                    point.ShadingNormal = Store(Scale(Load(point.ShadingNormal), -1));
            }
            point.Color = Store(blend(triangle.Color0, triangle.Color1, triangle.Color2));
            point.TexCoord[0] = float(triangle.TexCoord0.X * w[0] + triangle.TexCoord1.X * w[1] +
                                      triangle.TexCoord2.X * w[2]);
            point.TexCoord[1] = float(triangle.TexCoord0.Y * w[0] + triangle.TexCoord1.Y * w[1] +
                                      triangle.TexCoord2.Y * w[2]);
            if (!std::isfinite(point.TexCoord[0]) || !std::isfinite(point.TexCoord[1]))
                throw std::runtime_error("CUDA Point GI texture coordinates are not finite.");
            point.HasTexCoord = 1;
            point.Material = triangle.MaterialIndex;
            point.Primitive = primitive;
            point.Patch = ordinal++;
            point.Area = cellArea;
            Append(point);
        }
    }
    void SpherePoints(const Sphere& sphere, const Frame& frame, uint32_t primitive)
    {
        Poll();
        if (!(sphere.Radius > 0) || !std::isfinite(sphere.Radius))
            throw std::runtime_error("CUDA Point GI sphere radius must be finite and positive.");
        double rowBound = 0, columnBound = 0;
        for (uint32_t axis = 0; axis < Axes; ++axis)
        {
            double row = 0, column = 0;
            for (uint32_t k = 0; k < Axes; ++k)
            {
                column += std::abs(frame.Cofactor[axis][k]);
                row += std::abs(frame.Cofactor[k][axis]);
            }
            columnBound = std::max(columnBound, column);
            rowBound = std::max(rowBound, row);
        }
        const double sphereMeasure = 4 * Pi * double(sphere.Radius) * sphere.Radius;
        const double countBound =
            std::max(double(MinimumSphereSamples),
                     std::ceil(sphereMeasure * std::sqrt(rowBound * columnBound) / Options.MaximumArea));
        if (!std::isfinite(countBound) || countBound > Options.MaximumSurfels - Cache.Surfels.size())
            throw std::runtime_error("CUDA Point GI ellipsoid quadrature exceeds its surface budget.");
        const uint32_t count = uint32_t(countBound);
        for (uint32_t sample = 0; sample < count; ++sample)
        {
            const double y = 1 - 2 * (double(sample) + 0.5) / count;
            const double turn = sample * GoldenTurn;
            const double angle = 2 * Pi * (turn - std::floor(turn)),
                         radial = std::sqrt(std::max(0.0, 1 - y * y));
            const D3 n{radial * std::cos(angle), y, radial * std::sin(angle)};
            Surfel point{};
            point.Position = frame.Point(Add(Load(sphere.Center), Scale(n, sphere.Radius)));
            point.GeometryNormal = point.ShadingNormal = frame.OrientedNormal(n);
            point.Color = {1, 1, 1};
            point.Material = sphere.MaterialIndex;
            point.Primitive = primitive;
            point.Patch = sample;
            point.Area = sphereMeasure * Norm(frame.Product(frame.Cofactor, n)) / count;
            Append(point);
        }
    }
    void AddGeometry(const SceneGeometry& geometry, const Frame& frame, uint64_t& primitive)
    {
        if (primitive + geometry.Triangles.size() + geometry.Spheres.size() >= Invalid)
            throw std::runtime_error("CUDA Point GI primitive identifiers exceed 32-bit storage.");
        for (const auto& triangle : geometry.Triangles)
            TrianglePoints(triangle, frame, uint32_t(primitive++));
        for (const auto& sphere : geometry.Spheres)
            SpherePoints(sphere, frame, uint32_t(primitive++));
    }
    uint32_t Tree(uint32_t begin, uint32_t count)
    {
        Poll();
        const uint32_t slot = uint32_t(Cache.Nodes.size());
        Cache.Nodes.emplace_back();
        Node node;
        node.Begin = begin;
        node.Count = count;
        for (uint32_t axis = 0; axis < Axes; ++axis)
        {
            node.Low[axis] = INFINITY;
            node.High[axis] = -INFINITY;
        }
        for (uint32_t i = begin; i < begin + count; ++i)
        {
            Poll();
            const auto& point = Cache.Surfels[Cache.Order[i]];
            const D3 p = Load(point.Position);
            for (uint32_t axis = 0; axis < Axes; ++axis)
            {
                node.Low[axis] = std::min(node.Low[axis], std::nextafter(p[axis] - point.Radius, -INFINITY));
                node.High[axis] = std::max(node.High[axis], std::nextafter(p[axis] + point.Radius, INFINITY));
            }
        }
        if (count > LeafSize)
        {
            uint32_t axis = 0;
            for (uint32_t candidate = 1; candidate < Axes; ++candidate)
                if (node.High[candidate] - node.Low[candidate] > node.High[axis] - node.Low[axis])
                    axis = candidate;
            std::stable_sort(Cache.Order.begin() + begin, Cache.Order.begin() + begin + count,
                             [&](uint32_t a, uint32_t b) {
                                 const double x = Load(Cache.Surfels[a].Position)[axis],
                                              y = Load(Cache.Surfels[b].Position)[axis];
                                 return x == y ? a < b : x < y;
                             });
            const uint32_t half = count / 2;
            node.Left = Tree(begin, half);
            node.Right = Tree(begin + half, count - half);
        }
        Cache.Nodes[slot] = node;
        return slot;
    }
};
} // namespace
bool BuildGeometry(const Scene& scene, const PointGiOptions& options, GeometryCache& output,
                   const Detail::CancellationProbe& cancelled, std::string& error)
{
    output = {};
    if (!ValidatePointGiOptions(options, error) || !CudaInstances::Validate(scene, error))
        return false;
    try
    {
        GeometryBuilder builder(scene, options, cancelled);
        uint64_t primitive = 0;
        builder.AddGeometry(scene, Frame(GeometryInstance{}), primitive);
        for (const auto& instance : scene.Instances)
            builder.AddGeometry(*scene.Geometries[instance.GeometryIndex], Frame(instance), primitive);
        builder.Cache.Order.resize(builder.Cache.Surfels.size());
        std::iota(builder.Cache.Order.begin(), builder.Cache.Order.end(), 0u);
        if (!builder.Cache.Order.empty())
            builder.Tree(0, uint32_t(builder.Cache.Order.size()));
        if (cancelled && cancelled())
            throw std::runtime_error("CUDA Point GI geometry bake was cancelled.");
        output = std::move(builder.Cache);
        error.clear();
        return true;
    }
    catch (const std::exception& exception)
    {
        error = exception.what();
        return false;
    }
}
} // namespace Arvo::RayTracing::CudaPointGi
