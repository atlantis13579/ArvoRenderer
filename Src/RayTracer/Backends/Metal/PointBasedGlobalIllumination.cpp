#include "PointGiGeometry.h"
#include "InstancingBuilder.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>

namespace Arvo::RayTracing::MetalPointGi
{
namespace
{
// Independent Metal geometry bake. Four-way midpoint subdivision retains each
// triangle's barycentric attributes. Equal-solid-angle latitude/azimuth cells
// integrate affine spheres with the cofactor Jacobian. Equivalent-area disks
// are a geometric approximation, not an exact replacement for the input mesh.
// Christensen (2008), "Point-Based Approximate Color Bleeding":
// https://graphics.pixar.com/library/PointBasedColorBleeding/paper.pdf
constexpr double Pi = 3.1415926535897932384626433832795;
constexpr uint32_t LeafSize = 8, PollPeriod = 1024;
constexpr int BoundPaddingBits = 44;
using Triple = std::array<double, 3>;
Triple Add(Triple a, Triple b)
{
    return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}
Triple Subtract(Triple a, Triple b)
{
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}
Triple Multiply(Triple a, double scale)
{
    return {a[0] * scale, a[1] * scale, a[2] * scale};
}
double DotWide(Triple a, Triple b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
Triple CrossWide(Triple a, Triple b)
{
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
double Length(Triple v)
{
    return std::hypot(v[0], v[1], v[2]);
}
Triple Unit(Triple v)
{
    double length = Length(v);
    return length > 0 ? Multiply(v, 1 / length) : Triple{};
}
Triple Components(Float3 v)
{
    return {v.X, v.Y, v.Z};
}
Triple Components(Vector v)
{
    return {v.X, v.Y, v.Z};
}
Vector Pack(Triple v)
{
    return {float(v[0]), float(v[1]), float(v[2])};
}
bool Finite(Vector v)
{
    return std::isfinite(v.X) && std::isfinite(v.Y) && std::isfinite(v.Z);
}
Number PackNumber(double value)
{
    if (value == 0)
        return {};
    int exponent;
    double mantissa = std::frexp(value, &exponent);
    float high = float(mantissa), low = float(mantissa - double(high));
    if (std::abs(high) >= 1)
    {
        high /= 2;
        low /= 2;
        ++exponent;
    }
    return {high, low, exponent, 0};
}
struct Transform
{
    Triple Columns[3]{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}, Translation{};
    Triple Cofactors[3]{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    double Sign = 1;
    void Set(const GeometryInstance& instance)
    {
        const auto& m = instance.ObjectToWorld;
        for (uint32_t axis = 0; axis < 3; ++axis)
            Columns[axis] = {m[axis], m[axis + 4], m[axis + 8]};
        Translation = {m[3], m[7], m[11]};
        Cofactors[0] = CrossWide(Columns[1], Columns[2]);
        Cofactors[1] = CrossWide(Columns[2], Columns[0]);
        Cofactors[2] = CrossWide(Columns[0], Columns[1]);
        Sign = DotWide(Columns[0], Cofactors[0]) < 0 ? -1 : 1;
    }
    Triple Linear(Triple v) const
    {
        return Add(Add(Multiply(Columns[0], v[0]), Multiply(Columns[1], v[1])), Multiply(Columns[2], v[2]));
    }
    Triple Map(Triple v) const { return Add(Translation, Linear(v)); }
    Triple AreaNormal(Triple v) const
    {
        return Add(Add(Multiply(Cofactors[0], v[0]), Multiply(Cofactors[1], v[1])),
                   Multiply(Cofactors[2], v[2]));
    }
    Triple Normal(Triple v) const { return Multiply(Unit(AreaNormal(v)), Sign); }
};
struct Work
{
    const Scene& SceneData;
    const PointGiOptions& Options;
    const Detail::CancellationProbe& Cancelled;
    std::string& Error;
    Geometry Output;
    std::vector<double> Radii;
    uint64_t Primitive = 0, Steps = 0;
    bool Poll()
    {
        if (Steps++ % PollPeriod == 0 && Cancelled && Cancelled())
        {
            Error = "Metal Point GI geometry cancelled";
            return false;
        }
        return true;
    }
    bool Capacity(uint64_t additional)
    {
        if (additional > Options.MaximumSurfels - Output.Surfaces.size())
        {
            Error = "Metal Point GI surface budget exceeded; increase patch area or budget";
            return false;
        }
        return true;
    }
    bool Append(Surface surface, double area)
    {
        if (!Capacity(1))
            return false;
        if (!(area > 0) || !std::isfinite(area) || !Finite(surface.Position) ||
            !Finite(surface.GeometricNormal) || !Finite(surface.ShadingNormal) || !Finite(surface.Color) ||
            !std::isfinite(surface.TexCoord[0]) || !std::isfinite(surface.TexCoord[1]))
        {
            Error = "Metal Point GI surface attributes are nonfinite or unrepresentable";
            return false;
        }
        surface.Area = PackNumber(area);
        surface.RadiusSquared = PackNumber(area / Pi);
        Output.Surfaces.push_back(surface);
        Radii.push_back(std::sqrt(area / Pi));
        return true;
    }
    bool Triangle(const RayTracing::Triangle& triangle, const Transform& transform)
    {
        struct Corner
        {
            Triple Position, Barycentric;
        };
        using Patch = std::array<Corner, 3>;
        // Keep subdivision offsets separate from translation so tiny world-area
        // patches survive even when their uploaded centers round to one point.
        const Triple anchor = Components(triangle.Position0), base = transform.Map(anchor);
        const Triple p[3]{{},
                          transform.Linear(Subtract(Components(triangle.Position1), anchor)),
                          transform.Linear(Subtract(Components(triangle.Position2), anchor))};
        const Triple geometric =
            Multiply(Unit(CrossWide(Subtract(p[1], p[0]), Subtract(p[2], p[0]))), transform.Sign);
        std::vector<Patch> stack{{Corner{p[0], {1, 0, 0}}, Corner{p[1], {0, 1, 0}}, Corner{p[2], {0, 0, 1}}}};
        uint32_t patchId = 0;
        while (!stack.empty())
        {
            if (!Poll())
                return false;
            const Patch patch = stack.back();
            stack.pop_back();
            const double area = Length(CrossWide(Subtract(patch[1].Position, patch[0].Position),
                                                 Subtract(patch[2].Position, patch[0].Position))) /
                                2;
            if (area == 0)
                continue;
            if (!std::isfinite(area))
            {
                Error = "Metal Point GI triangle has nonfinite area";
                return false;
            }
            if (area > Options.MaximumArea)
            {
                constexpr uint32_t Subpatches = 4;
                if (!Capacity(stack.size() + Subpatches))
                    return false;
                Corner midpoint[3];
                for (uint32_t edge = 0; edge < 3; ++edge)
                {
                    const auto& a = patch[edge];
                    const auto& b = patch[(edge + 1) % 3];
                    midpoint[edge] = {Multiply(Add(a.Position, b.Position), 0.5),
                                      Multiply(Add(a.Barycentric, b.Barycentric), 0.5)};
                    if (midpoint[edge].Position == a.Position || midpoint[edge].Position == b.Position)
                    {
                        Error = "Metal Point GI subdivision cannot resolve a midpoint";
                        return false;
                    }
                }
                stack.push_back({midpoint[0], midpoint[1], midpoint[2]});
                stack.push_back({midpoint[2], midpoint[1], patch[2]});
                stack.push_back({midpoint[0], patch[1], midpoint[1]});
                stack.push_back({patch[0], midpoint[0], midpoint[2]});
                continue;
            }
            const Triple bary =
                Multiply(Add(Add(patch[0].Barycentric, patch[1].Barycentric), patch[2].Barycentric), 1.0 / 3);
            const auto interpolate = [&](Float3 a, Float3 b, Float3 c) {
                return Add(Add(Multiply(Components(a), bary[0]), Multiply(Components(b), bary[1])),
                           Multiply(Components(c), bary[2]));
            };
            Surface surface{};
            surface.Position = Pack(Add(
                base, Multiply(Add(Add(patch[0].Position, patch[1].Position), patch[2].Position), 1.0 / 3)));
            surface.GeometricNormal = Pack(geometric);
            Triple normal =
                triangle.SmoothNormals
                    ? transform.Normal(interpolate(triangle.Normal0, triangle.Normal1, triangle.Normal2))
                    : geometric;
            if (Length(normal) == 0)
                normal = geometric;
            if (DotWide(normal, geometric) < 0)
                normal = Multiply(normal, -1);
            surface.ShadingNormal = Pack(normal);
            surface.Color = Pack(interpolate(triangle.Color0, triangle.Color1, triangle.Color2));
            surface.TexCoord[0] = float(bary[0] * triangle.TexCoord0.X + bary[1] * triangle.TexCoord1.X +
                                        bary[2] * triangle.TexCoord2.X);
            surface.TexCoord[1] = float(bary[0] * triangle.TexCoord0.Y + bary[1] * triangle.TexCoord1.Y +
                                        bary[2] * triangle.TexCoord2.Y);
            surface.Material = triangle.MaterialIndex;
            surface.Primitive = uint32_t(Primitive);
            surface.Patch = patchId++;
            surface.HasTexCoord = 1;
            if (!Append(surface, area))
                return false;
        }
        return true;
    }
    bool Sphere(const RayTracing::Sphere& sphere, const Transform& transform)
    {
        if (!(sphere.Radius > 0) || !std::isfinite(sphere.Radius))
        {
            Error = "Metal Point GI sphere radius is invalid";
            return false;
        }
        const double radiusSquared = double(sphere.Radius) * sphere.Radius;
        const double areaBound = 4 * Pi * radiusSquared *
                                 std::hypot(Length(transform.Cofactors[0]), Length(transform.Cofactors[1]),
                                            Length(transform.Cofactors[2]));
        const double required = std::max(1.0, std::ceil(areaBound / Options.MaximumArea));
        if (!std::isfinite(required) || required > Options.MaximumSurfels)
        {
            Error = "Metal Point GI sphere quadrature exceeds the surface budget";
            return false;
        }
        const uint32_t bands = std::max(1u, uint32_t(std::ceil(std::sqrt(required / 2))));
        const uint32_t slices = std::max(1u, uint32_t(std::ceil(required / bands)));
        const uint64_t count = uint64_t(bands) * slices;
        if (!Capacity(count))
            return false;
        for (uint32_t band = 0; band < bands; ++band)
        {
            const double y = 1 - 2 * (double(band) + 0.5) / bands,
                         circle = std::sqrt(std::max(0.0, 1 - y * y));
            for (uint32_t slice = 0; slice < slices; ++slice)
            {
                if (!Poll())
                    return false;
                const double stagger = (band % 2) ? 0.5 : 0;
                const double angle = 2 * Pi * (double(slice) + stagger) / slices;
                const Triple local{circle * std::cos(angle), y, circle * std::sin(angle)};
                Surface surface{};
                surface.Position =
                    Pack(transform.Map(Add(Components(sphere.Center), Multiply(local, sphere.Radius))));
                surface.GeometricNormal = surface.ShadingNormal = Pack(transform.Normal(local));
                surface.Color = {1, 1, 1};
                surface.Material = sphere.MaterialIndex;
                surface.Primitive = uint32_t(Primitive);
                surface.Patch = band * slices + slice;
                const double area =
                    4 * Pi * radiusSquared * Length(transform.AreaNormal(local)) / double(count);
                if (!Append(surface, area))
                    return false;
            }
        }
        return true;
    }
    bool Object(const SceneGeometry& geometry, const Transform& transform)
    {
        for (const auto& triangle : geometry.Triangles)
        {
            if (!Poll())
                return false;
            if (Primitive >= Invalid || triangle.MaterialIndex >= SceneData.Materials.size())
            {
                Error = "Metal Point GI triangle has an invalid identity";
                return false;
            }
            if (!Triangle(triangle, transform))
                return false;
            ++Primitive;
        }
        for (const auto& sphere : geometry.Spheres)
        {
            if (Primitive >= Invalid || sphere.MaterialIndex >= SceneData.Materials.size())
            {
                Error = "Metal Point GI sphere has an invalid identity";
                return false;
            }
            if (!Sphere(sphere, transform))
                return false;
            ++Primitive;
        }
        return true;
    }
    bool Tree(uint32_t begin, uint32_t count)
    {
        if (!Poll())
            return false;
        Triple minimum{}, maximum{};
        minimum.fill(std::numeric_limits<double>::infinity());
        maximum.fill(-std::numeric_limits<double>::infinity());
        for (uint32_t i = begin; i < begin + count; ++i)
        {
            if (!Poll())
                return false;
            const uint32_t id = Output.Indices[i];
            const Triple center = Components(Output.Surfaces[id].Position);
            for (uint32_t axis = 0; axis < 3; ++axis)
            {
                // Pad beyond the 48-bit upload precision before encoding bounds.
                const double padding =
                    std::ldexp(std::max(std::abs(center[axis]), Radii[id]), -BoundPaddingBits);
                minimum[axis] = std::min(minimum[axis], center[axis] - Radii[id] - padding);
                maximum[axis] = std::max(maximum[axis], center[axis] + Radii[id] + padding);
            }
        }
        Node node{};
        for (uint32_t axis = 0; axis < 3; ++axis)
        {
            node.Minimum[axis] = PackNumber(minimum[axis]);
            node.Maximum[axis] = PackNumber(maximum[axis]);
        }
        const uint32_t index = uint32_t(Output.Nodes.size());
        Output.Nodes.push_back(node);
        if (count <= LeafSize)
        {
            Output.Nodes[index].Begin = begin;
            Output.Nodes[index].Count = count;
        }
        else
        {
            uint32_t splitAxis = 0;
            for (uint32_t axis = 1; axis < 3; ++axis)
                if (maximum[axis] - minimum[axis] > maximum[splitAxis] - minimum[splitAxis])
                    splitAxis = axis;
            const uint32_t leftCount = count / 2;
            std::nth_element(Output.Indices.begin() + begin, Output.Indices.begin() + begin + leftCount,
                             Output.Indices.begin() + begin + count, [&](uint32_t a, uint32_t b) {
                                 const double x = Components(Output.Surfaces[a].Position)[splitAxis],
                                              y = Components(Output.Surfaces[b].Position)[splitAxis];
                                 return x == y ? a < b : x < y;
                             });
            if (!Tree(begin, leftCount) || !Tree(begin + leftCount, count - leftCount))
                return false;
        }
        Output.Nodes[index].Escape = uint32_t(Output.Nodes.size());
        return true;
    }
};
} // namespace
bool BuildGeometry(const Scene& scene, const PointGiOptions& options,
                   const Detail::CancellationProbe& cancelled, Geometry& result, std::string& error)
{
    if (!ValidatePointGiOptions(options, error) || !MetalInstances::Validate(scene, error))
        return false;
    Work work{scene, options, cancelled, error};
    if (!work.Object(scene, Transform{}))
        return false;
    for (const auto& instance : scene.Instances)
    {
        Transform transform;
        transform.Set(instance);
        if (!work.Object(*scene.Geometries[instance.GeometryIndex], transform))
            return false;
    }
    work.Output.Indices.resize(work.Output.Surfaces.size());
    std::iota(work.Output.Indices.begin(), work.Output.Indices.end(), 0u);
    if (!work.Output.Surfaces.empty() && !work.Tree(0, uint32_t(work.Output.Surfaces.size())))
        return false;
    if (cancelled && cancelled())
    {
        error = "Metal Point GI geometry cancelled";
        return false;
    }
    result = std::move(work.Output);
    error.clear();
    return true;
}
} // namespace Arvo::RayTracing::MetalPointGi
