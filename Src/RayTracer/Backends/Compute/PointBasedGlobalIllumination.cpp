#include "PointGiGeometry.h"
#include "InstancingBuilder.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>

namespace Arvo::RayTracing::ComputePointGi
{
namespace
{
// Christensen (2008), "Point-Based Approximate Color Bleeding". This teaching
// https://graphics.pixar.com/library/PointBasedColorBleeding/paper.pdf
// implementation uses equal-area disks, with a stackless disk hierarchy rather
// than an unoccluded sum of distant clusters. Triangle patches preserve UVs;
// affine sphere samples carry their own cofactor area Jacobian.
constexpr double Pi = 3.1415926535897932384626433832795;
const double GoldenAngle = Pi * (3 - std::sqrt(5.0));
constexpr uint32_t LeafCapacity = 8, CancellationPeriod = 1024;
using D = std::array<double, 3>;
D Add(D a, D b)
{
    return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}
D Sub(D a, D b)
{
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}
D Scale(D a, double b)
{
    return {a[0] * b, a[1] * b, a[2] * b};
}
double DotD(D a, D b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
D CrossD(D a, D b)
{
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
double Length(D a)
{
    return std::hypot(a[0], a[1], a[2]);
}
D Unit(D a)
{
    const double n = Length(a);
    return n > 0 ? Scale(a, 1 / n) : D{};
}
D Widen(Float3 v)
{
    return {v.X, v.Y, v.Z};
}
D Widen(Vector v)
{
    return {v.X, v.Y, v.Z};
}
Vector Narrow(D a)
{
    return {float(a[0]), float(a[1]), float(a[2])};
}
bool Finite(Vector v)
{
    return std::isfinite(v.X) && std::isfinite(v.Y) && std::isfinite(v.Z);
}
struct Frame
{
    D X{1, 0, 0}, Y{0, 1, 0}, Z{0, 0, 1}, T{};
    D Cx{1, 0, 0}, Cy{0, 1, 0}, Cz{0, 0, 1};
    double Orientation = 1;
    D VectorOf(D p) const { return Add(Add(Scale(X, p[0]), Scale(Y, p[1])), Scale(Z, p[2])); }
    D Position(D p) const { return Add(VectorOf(p), T); }
    D Cofactor(D n) const { return Add(Add(Scale(Cx, n[0]), Scale(Cy, n[1])), Scale(Cz, n[2])); }
    D Normal(D n) const { return Scale(Unit(Cofactor(n)), Orientation); }
};
Frame MakeFrame(const std::array<float, 12>& m)
{
    Frame f;
    f.X = {m[0], m[4], m[8]};
    f.Y = {m[1], m[5], m[9]};
    f.Z = {m[2], m[6], m[10]};
    f.T = {m[3], m[7], m[11]};
    f.Cx = CrossD(f.Y, f.Z);
    f.Cy = CrossD(f.Z, f.X);
    f.Cz = CrossD(f.X, f.Y);
    f.Orientation = DotD(f.X, f.Cx) < 0 ? -1 : 1;
    return f;
}
struct Builder
{
    const Scene& SceneData;
    const PointGiOptions& Options;
    const Detail::CancellationProbe& Cancelled;
    std::string& Error;
    Geometry Output;
    uint64_t Work = 0, Primitive = 0;

    bool Poll()
    {
        if ((Work++ % CancellationPeriod) == 0 && Cancelled && Cancelled())
        {
            Error = "Compute Point GI geometry cancelled";
            return false;
        }
        return true;
    }
    bool Append(Surface surface)
    {
        if (Output.Surfaces.size() >= Options.MaximumSurfels)
        {
            Error = "Compute Point GI surface budget exceeded; increase patch area or budget";
            return false;
        }
        if (!Finite(surface.Position) || !Finite(surface.Geometry) || !Finite(surface.Normal) ||
            !Finite(surface.Color) || !std::isfinite(surface.TexCoord[0]) ||
            !std::isfinite(surface.TexCoord[1]) || !(surface.Area > 0) || !std::isfinite(surface.Area) ||
            !(surface.Radius > 0) || !std::isfinite(surface.Radius))
        {
            Error = "Compute Point GI surface cannot be represented by the native layout";
            return false;
        }
        Output.Surfaces.push_back(surface);
        return true;
    }
    bool TrianglePatches(const Triangle& triangle, const Frame& frame)
    {
        // Subdivide in a translated local frame; absolute coordinates can erase
        // small edges before their world-area measure has been computed.
        const D anchor = Widen(triangle.Position0), origin = frame.Position(anchor);
        const std::array<D, 3> positions{D{}, frame.VectorOf(Sub(Widen(triangle.Position1), anchor)),
                                         frame.VectorOf(Sub(Widen(triangle.Position2), anchor))};
        const std::array<D, 3> normals{Widen(triangle.Normal0), Widen(triangle.Normal1),
                                       Widen(triangle.Normal2)};
        const std::array<D, 3> colors{Widen(triangle.Color0), Widen(triangle.Color1), Widen(triangle.Color2)};
        const std::array<Float2, 3> uv{triangle.TexCoord0, triangle.TexCoord1, triangle.TexCoord2};
        const D geometric =
            Scale(Unit(CrossD(Sub(positions[1], positions[0]), Sub(positions[2], positions[0]))),
                  frame.Orientation);
        struct Corner
        {
            D P, B;
        };
        using Patch = std::array<Corner, 3>;
        std::vector<Patch> pending{{Corner{positions[0], {1, 0, 0}}, Corner{positions[1], {0, 1, 0}},
                                    Corner{positions[2], {0, 0, 1}}}};
        uint32_t patchIndex = 0;
        while (!pending.empty())
        {
            if (!Poll())
                return false;
            const Patch patch = pending.back();
            pending.pop_back();
            const double area = Length(CrossD(Sub(patch[1].P, patch[0].P), Sub(patch[2].P, patch[0].P))) / 2;
            if (area == 0)
                continue;
            if (!std::isfinite(area))
            {
                Error = "Nonfinite Compute Point GI triangle area";
                return false;
            }
            if (area > Options.MaximumArea)
            {
                if (Output.Surfaces.size() + pending.size() + 2 > Options.MaximumSurfels)
                {
                    Error = "Compute Point GI triangle subdivision exceeds surface budget";
                    return false;
                }
                uint32_t edge = 0;
                double longest = -1;
                for (uint32_t i = 0; i < 3; ++i)
                {
                    const double length = Length(Sub(patch[(i + 1) % 3].P, patch[i].P));
                    if (length > longest)
                    {
                        longest = length;
                        edge = i;
                    }
                }
                const uint32_t a = edge, b = (edge + 1) % 3, c = (edge + 2) % 3;
                const Corner middle{Scale(Add(patch[a].P, patch[b].P), 0.5),
                                    Scale(Add(patch[a].B, patch[b].B), 0.5)};
                if (middle.P == patch[a].P || middle.P == patch[b].P)
                {
                    Error = "Compute Point GI subdivision lost numerical progress";
                    return false;
                }
                pending.push_back({middle, patch[b], patch[c]});
                pending.push_back({patch[a], middle, patch[c]});
                continue;
            }
            const D bary = Scale(Add(Add(patch[0].B, patch[1].B), patch[2].B), 1.0 / 3);
            Surface surface{};
            surface.Position =
                Narrow(Add(origin, Scale(Add(Add(patch[0].P, patch[1].P), patch[2].P), 1.0 / 3)));
            surface.Geometry = Narrow(geometric);
            D normal{}, color{};
            for (uint32_t i = 0; i < 3; ++i)
            {
                normal = Add(normal, Scale(normals[i], bary[i]));
                color = Add(color, Scale(colors[i], bary[i]));
                surface.TexCoord[0] += float(double(uv[i].X) * bary[i]);
                surface.TexCoord[1] += float(double(uv[i].Y) * bary[i]);
            }
            normal = triangle.SmoothNormals ? frame.Normal(normal) : geometric;
            if (!(Length(normal) > 0))
                normal = geometric;
            if (DotD(normal, geometric) < 0)
                normal = Scale(normal, -1);
            surface.Normal = Narrow(normal);
            surface.Color = Narrow(color);
            surface.Material = triangle.MaterialIndex;
            surface.Primitive = uint32_t(Primitive);
            surface.Patch = patchIndex++;
            surface.HasTexCoord = 1;
            surface.Area = area;
            surface.Radius = std::sqrt(area / Pi);
            if (!Append(surface))
                return false;
        }
        return true;
    }
    bool SpherePatches(const Sphere& sphere, const Frame& frame)
    {
        if (!(sphere.Radius > 0) || !std::isfinite(sphere.Radius))
        {
            Error = "Compute Point GI sphere radius must be positive and finite";
            return false;
        }
        const double radiusSquared = double(sphere.Radius) * sphere.Radius;
        // The Frobenius norm bounds every cofactor direction, so no generated
        // patch has more world area than the requested maximum.
        const double cofactorBound = std::hypot(Length(frame.Cx), Length(frame.Cy), Length(frame.Cz));
        const double countBound = std::ceil(4 * Pi * radiusSquared * cofactorBound / Options.MaximumArea);
        if (!std::isfinite(countBound) || countBound > Options.MaximumSurfels - Output.Surfaces.size())
        {
            Error = "Compute Point GI sphere exceeds surface budget";
            return false;
        }
        const uint32_t count = std::max(1u, uint32_t(countBound));
        for (uint32_t i = 0; i < count; ++i)
        {
            if (!Poll())
                return false;
            const double y = 1 - 2 * (double(i) + 0.5) / count;
            const double ring = std::sqrt(std::max(0.0, 1 - y * y)), angle = GoldenAngle * i;
            const D unit{ring * std::cos(angle), y, ring * std::sin(angle)};
            Surface surface{};
            surface.Position = Narrow(frame.Position(Add(Widen(sphere.Center), Scale(unit, sphere.Radius))));
            surface.Geometry = surface.Normal = Narrow(frame.Normal(unit));
            surface.Color = {1, 1, 1};
            surface.Material = sphere.MaterialIndex;
            surface.Primitive = uint32_t(Primitive);
            surface.Patch = i;
            surface.Area = 4 * Pi * radiusSquared * Length(frame.Cofactor(unit)) / count;
            surface.Radius = std::sqrt(surface.Area / Pi);
            if (!Append(surface))
                return false;
        }
        return true;
    }
    bool AddObject(const SceneGeometry& object, const Frame& frame)
    {
        for (const auto& triangle : object.Triangles)
        {
            if (Primitive >= Invalid || triangle.MaterialIndex >= SceneData.Materials.size())
            {
                Error = "Invalid Compute Point GI triangle identifier";
                return false;
            }
            if (!TrianglePatches(triangle, frame))
                return false;
            ++Primitive;
        }
        for (const auto& sphere : object.Spheres)
        {
            if (Primitive >= Invalid || sphere.MaterialIndex >= SceneData.Materials.size())
            {
                Error = "Invalid Compute Point GI sphere identifier";
                return false;
            }
            if (!SpherePatches(sphere, frame))
                return false;
            ++Primitive;
        }
        return true;
    }
    bool MakeBounds(uint32_t begin, uint32_t end)
    {
        if (!Poll())
            return false;
        const uint32_t nodeIndex = uint32_t(Output.Hierarchy.size());
        Bounds node{};
        for (uint32_t axis = 0; axis < 3; ++axis)
        {
            node.Minimum[axis] = std::numeric_limits<double>::infinity();
            node.Maximum[axis] = -node.Minimum[axis];
        }
        for (uint32_t i = begin; i < end; ++i)
        {
            if (!Poll())
                return false;
            const Surface& surface = Output.Surfaces[Output.Indices[i]];
            const D center = Widen(surface.Position);
            for (uint32_t axis = 0; axis < 3; ++axis)
            {
                node.Minimum[axis] =
                    std::min(node.Minimum[axis], std::nextafter(center[axis] - surface.Radius,
                                                                -std::numeric_limits<double>::infinity()));
                node.Maximum[axis] =
                    std::max(node.Maximum[axis], std::nextafter(center[axis] + surface.Radius,
                                                                std::numeric_limits<double>::infinity()));
            }
        }
        node.Begin = begin;
        node.Count = end - begin;
        Output.Hierarchy.push_back(node);
        if (end - begin > LeafCapacity)
        {
            uint32_t axis = 0;
            for (uint32_t k = 1; k < 3; ++k)
                if (node.Maximum[k] - node.Minimum[k] > node.Maximum[axis] - node.Minimum[axis])
                    axis = k;
            const uint32_t middle = begin + (end - begin) / 2;
            std::nth_element(Output.Indices.begin() + begin, Output.Indices.begin() + middle,
                             Output.Indices.begin() + end, [&](uint32_t a, uint32_t b) {
                                 const double x = Widen(Output.Surfaces[a].Position)[axis],
                                              y = Widen(Output.Surfaces[b].Position)[axis];
                                 return x == y ? a < b : x < y;
                             });
            Output.Hierarchy[nodeIndex].Count = 0;
            if (!MakeBounds(begin, middle) || !MakeBounds(middle, end))
                return false;
        }
        Output.Hierarchy[nodeIndex].Escape = uint32_t(Output.Hierarchy.size());
        return true;
    }
};
} // namespace

bool BuildGeometry(const Scene& scene, const PointGiOptions& options,
                   const Detail::CancellationProbe& cancelled, Geometry& result, std::string& error)
{
    if (!ValidatePointGiOptions(options, error) || !ComputeInstances::Validate(scene, error))
        return false;
    Builder builder{scene, options, cancelled, error};
    if (!builder.AddObject(scene, Frame{}))
        return false;
    for (const auto& instance : scene.Instances)
        if (!builder.AddObject(*scene.Geometries[instance.GeometryIndex], MakeFrame(instance.ObjectToWorld)))
            return false;
    builder.Output.Indices.resize(builder.Output.Surfaces.size());
    std::iota(builder.Output.Indices.begin(), builder.Output.Indices.end(), 0u);
    if (!builder.Output.Surfaces.empty() && !builder.MakeBounds(0, uint32_t(builder.Output.Surfaces.size())))
        return false;
    if (cancelled && cancelled())
    {
        error = "Compute Point GI geometry cancelled";
        return false;
    }
    result = std::move(builder.Output);
    error.clear();
    return true;
}
} // namespace Arvo::RayTracing::ComputePointGi
