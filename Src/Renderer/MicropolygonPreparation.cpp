#include "Renderer/MicropolygonPreparation.h"
#include "Renderer/GeometryGuides.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <numeric>
#include <stdexcept>

namespace Arvo::RayTracing
{
namespace
{
namespace M = MicropolygonData;
constexpr float MinimumWavelength = 1.e-4f;
constexpr float MaximumWaveSlope = 1024;
constexpr float Pi = 3.14159265358979323846f;
constexpr uint32_t NoMaterial = std::numeric_limits<uint32_t>::max();
bool Finite(Float3 v)
{
    return std::isfinite(v.X) && std::isfinite(v.Y) && std::isfinite(v.Z);
}
M::Float4 Pack(Float3 v)
{
    return {v.X, v.Y, v.Z, 0};
}
Float3 Unpack(M::Float4 v)
{
    return {v.X, v.Y, v.Z};
}
uint32_t Bits(float v)
{
    // Signed zeros denote the same geometric corner.
    uint32_t bits = 0;
    if (v != 0)
        std::memcpy(&bits, &v, sizeof(bits));
    return bits;
}
struct Affine
{
    std::array<float, 12> Forward, Inverse;
    bool Mirrored = false;
    explicit Affine(const std::array<float, 12>& m) : Forward(m)
    {
        for (float v : m)
            if (!std::isfinite(v))
                throw std::invalid_argument("Nonfinite dicing instance transform");
        const double determinant = double(m[0]) * (double(m[5]) * m[10] - double(m[6]) * m[9]) -
                                   double(m[1]) * (double(m[4]) * m[10] - double(m[6]) * m[8]) +
                                   double(m[2]) * (double(m[4]) * m[9] - double(m[5]) * m[8]);
        if (determinant == 0 || !std::isfinite(determinant))
            throw std::invalid_argument("Singular dicing instance transform");
        Mirrored = determinant < 0;
        Inverse = {float((double(m[5]) * m[10] - double(m[6]) * m[9]) / determinant),
                   float((double(m[2]) * m[9] - double(m[1]) * m[10]) / determinant),
                   float((double(m[1]) * m[6] - double(m[2]) * m[5]) / determinant),
                   0,
                   float((double(m[6]) * m[8] - double(m[4]) * m[10]) / determinant),
                   float((double(m[0]) * m[10] - double(m[2]) * m[8]) / determinant),
                   float((double(m[2]) * m[4] - double(m[0]) * m[6]) / determinant),
                   0,
                   float((double(m[4]) * m[9] - double(m[5]) * m[8]) / determinant),
                   float((double(m[1]) * m[8] - double(m[0]) * m[9]) / determinant),
                   float((double(m[0]) * m[5] - double(m[1]) * m[4]) / determinant),
                   0};
        for (size_t row = 0; row < 3; ++row)
            Inverse[row * 4 + 3] =
                -(Inverse[row * 4] * m[3] + Inverse[row * 4 + 1] * m[7] + Inverse[row * 4 + 2] * m[11]);
        for (float v : Inverse)
            if (!std::isfinite(v))
                throw std::invalid_argument("Unrepresentable dicing inverse transform");
    }
    static Float3 Point(const std::array<float, 12>& m, Float3 p)
    {
        return {m[0] * p.X + m[1] * p.Y + m[2] * p.Z + m[3], m[4] * p.X + m[5] * p.Y + m[6] * p.Z + m[7],
                m[8] * p.X + m[9] * p.Y + m[10] * p.Z + m[11]};
    }
    static Float3 Normal(const std::array<float, 12>& inverse, Float3 n)
    {
        return Normalize({inverse[0] * n.X + inverse[4] * n.Y + inverse[8] * n.Z,
                          inverse[1] * n.X + inverse[5] * n.Y + inverse[9] * n.Z,
                          inverse[2] * n.X + inverse[6] * n.Y + inverse[10] * n.Z});
    }
};
const std::array<float, 12> Identity{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
struct Group
{
    const SceneGeometry* Source = nullptr;
    GeometryInstance Instance;
    Affine Transform{Identity};
    bool Root = false;
    uint32_t MotionDomain = 0;
};
class Components
{
    std::vector<uint32_t> Parent;

  public:
    explicit Components(size_t count) : Parent(count)
    {
        std::iota(Parent.begin(), Parent.end(), 0);
    }
    uint32_t Find(uint32_t id)
    {
        while (Parent[id] != id)
        {
            Parent[id] = Parent[Parent[id]];
            id = Parent[id];
        }
        return id;
    }
    void Join(uint32_t a, uint32_t b)
    {
        a = Find(a);
        b = Find(b);
        Parent[std::max(a, b)] = std::min(a, b);
    }
};
// FNV-1a over explicit little-endian words; no structure padding or pointers.
class Fingerprint
{
    uint64_t Value = 14695981039346656037ull;

  public:
    void Word(uint32_t word)
    {
        for (uint32_t byte = 0; byte < sizeof(word); ++byte)
        {
            Value ^= (word >> (byte * 8)) & 0xffu;
            Value *= 1099511628211ull;
        }
    }
    void Scalar(float value)
    {
        Word(Bits(value));
    }
    void Vector(M::Float4 v)
    {
        Scalar(v.X);
        Scalar(v.Y);
        Scalar(v.Z);
        Scalar(v.W);
    }
    void Vector(M::UInt4 v)
    {
        Word(v.X);
        Word(v.Y);
        Word(v.Z);
        Word(v.W);
    }
    uint64_t Get() const
    {
        return Value;
    }
};
bool Stopped(const Detail::CancellationProbe& cancelled)
{
    return cancelled && cancelled();
}
} // namespace

bool ValidateMicropolygonOptions(const MicropolygonOptions& options, std::string& error)
{
    if (!std::isfinite(options.TargetEdgePixels) || options.TargetEdgePixels < 1 ||
        options.TargetEdgePixels > 16384 || options.MaximumLevel > M::MaximumLevel ||
        !options.TriangleBudget || options.TriangleBudget > M::MaximumOutputTriangles ||
        !std::isfinite(options.Amplitude) || !std::isfinite(options.Wavelength) ||
        options.Wavelength < MinimumWavelength || !std::isfinite(options.PhaseCycles) ||
        std::abs(double(options.Amplitude) / options.Wavelength) * (2 * Pi) > MaximumWaveSlope)
    {
        error = "Micropolygons require a positive bounded pixel target/budget, levels 0..8, and a finite "
                "bounded wave slope";
        return false;
    }
    return true;
}

MicropolygonResult IRayTracingRHI::PrepareMicropolygons(const Scene&, const Camera&, const RenderSettings&,
                                                        const CancellationToken& cancellation) const
{
    MicropolygonResult result;
    if (cancellation.IsCancellationRequested())
        result.Status = RenderStatus::Cancelled;
    else
        result.Error = "Native micropolygon preparation is unavailable on this backend";
    return result;
}

namespace Detail
{
MicropolygonResult PrepareMicropolygons(const Scene& scene, const Camera& camera,
                                        const RenderSettings& settings, const CancellationProbe& cancelled,
                                        const NativeMicropolygonPass& native)
{
    MicropolygonResult result;
    const auto cancel = [&] {
        result.Status = RenderStatus::Cancelled;
        result.Error.clear();
        result.Geometry.reset();
        result.Plan.reset();
        result.InstanceSources.clear();
    };
    if (Stopped(cancelled))
    {
        cancel();
        return result;
    }
    if (!ValidateMicropolygonOptions(settings.Micropolygons, result.Error) ||
        !ValidateTileSettings(settings, result.Error))
        return result;
    if (!settings.Micropolygons.Enabled)
    {
        result.Geometry = std::make_shared<Scene>(scene);
        result.InstanceSources.resize(scene.Instances.size());
        std::iota(result.InstanceSources.begin(), result.InstanceSources.end(), 0u);
        result.Status = RenderStatus::Completed;
        return result;
    }
    try
    {
        const auto& options = settings.Micropolygons;
        const bool displace = options.DisplacementMaterial != NoMaterial && options.Amplitude != 0;
        if (displace && options.DisplacementMaterial >= scene.Materials.size())
            throw std::invalid_argument("Displacement material does not belong to the scene");
        if (!ValidateGeometryGuideRequest(camera, settings, result.Error))
            return result;
        std::vector<Group> groups;
        groups.push_back({&scene, {}, Affine(Identity), true, 0});
        size_t population = scene.Triangles.size();
        for (const auto& instance : scene.Instances)
        {
            if (instance.GeometryIndex >= scene.Geometries.size() ||
                !scene.Geometries[instance.GeometryIndex])
                throw std::invalid_argument("Dicing instance references missing geometry");
            if (!ValidateRigidMotion(instance.Motion, result.Error))
                return result;
            const auto* geometry = scene.Geometries[instance.GeometryIndex].get();
            population += geometry->Triangles.size();
            groups.push_back({geometry, instance, Affine(instance.ObjectToWorld), false,
                              instance.Motion.Enabled ? uint32_t(groups.size()) : 0});
            if (population > options.TriangleBudget)
                throw std::length_error("Source triangles exceed the micropolygon budget");
        }
        if (population > options.TriangleBudget)
            throw std::length_error("Source triangles exceed the micropolygon budget");
        std::vector<M::Patch> patches;
        patches.reserve(population);
        // Corner identity crosses shading/UV/material seams but never joins
        // separate moving instances merely because they overlap at frame zero.
        std::map<std::array<uint32_t, 4>, uint32_t> corners;
        bool selected = false;
        for (uint32_t groupIndex = 0; groupIndex < groups.size(); ++groupIndex)
        {
            const auto& group = groups[groupIndex];
            if (displace)
                for (const auto& sphere : group.Source->Spheres)
                    if (sphere.MaterialIndex == options.DisplacementMaterial)
                        throw std::invalid_argument("Displacement currently requires triangle meshes; "
                                                    "analytic sphere patches are not implemented");
            for (const auto& triangle : group.Source->Triangles)
            {
                if (Stopped(cancelled))
                {
                    cancel();
                    return result;
                }
                if (triangle.MaterialIndex >= scene.Materials.size())
                    throw std::invalid_argument("Dicing triangle references missing material");
                M::Patch patch;
                patch.Metadata = {triangle.MaterialIndex, triangle.SmoothNormals ? M::SmoothNormals : 0, 0,
                                  groupIndex};
                const Float3 positions[]{triangle.Position0, triangle.Position1, triangle.Position2};
                const Float3 normals[]{triangle.Normal0, triangle.Normal1, triangle.Normal2};
                const Float3 colors[]{triangle.Color0, triangle.Color1, triangle.Color2};
                const Float2 texcoords[]{triangle.TexCoord0, triangle.TexCoord1, triangle.TexCoord2};
                uint32_t ids[3]{};
                for (uint32_t corner = 0; corner < 3; ++corner)
                {
                    auto p = Affine::Point(group.Transform.Forward, positions[corner]);
                    if (!Finite(p) || !Finite(normals[corner]) || !Finite(colors[corner]) ||
                        !std::isfinite(texcoords[corner].X) || !std::isfinite(texcoords[corner].Y))
                        throw std::invalid_argument("Nonfinite micropolygon input attributes");
                    // Equal corner keys also receive identical float bits.
                    if (p.X == 0)
                        p.X = 0;
                    if (p.Y == 0)
                        p.Y = 0;
                    if (p.Z == 0)
                        p.Z = 0;
                    const std::array<uint32_t, 4> key{Bits(p.X), Bits(p.Y), Bits(p.Z), group.MotionDomain};
                    ids[corner] = corners.emplace(key, uint32_t(corners.size())).first->second;
                    patch.Corners[corner] = {Pack(p),
                                             Pack(Affine::Normal(group.Transform.Inverse, normals[corner])),
                                             Pack(colors[corner]),
                                             {texcoords[corner].X, texcoords[corner].Y, 0, 0}};
                    if (!Finite(Unpack(patch.Corners[corner].Normal)))
                        throw std::invalid_argument("Unrepresentable transformed dicing normal");
                }
                if (group.Transform.Mirrored)
                {
                    std::swap(patch.Corners[1], patch.Corners[2]);
                    std::swap(ids[1], ids[2]);
                }
                patch.CornerIds = {ids[0], ids[1], ids[2], 0};
                patches.push_back(patch);
            }
        }
        Components connectivity(patches.size());
        std::map<std::pair<uint32_t, uint32_t>, uint32_t> edges;
        for (uint32_t index = 0; index < patches.size(); ++index)
        {
            if (Stopped(cancelled))
            {
                cancel();
                return result;
            }
            const auto ids = patches[index].CornerIds;
            const uint32_t vertices[]{ids.X, ids.Y, ids.Z};
            for (uint32_t edge = 0; edge < 3; ++edge)
            {
                auto a = vertices[edge], b = vertices[(edge + 1) % 3];
                if (a > b)
                    std::swap(a, b);
                const auto inserted = edges.emplace(std::make_pair(a, b), index);
                if (!inserted.second)
                    connectivity.Join(index, inserted.first->second);
            }
        }
        std::map<uint32_t, uint32_t> componentIds;
        for (uint32_t index = 0; index < patches.size(); ++index)
            patches[index].Metadata.Z =
                componentIds.emplace(connectivity.Find(index), uint32_t(componentIds.size())).first->second;
        std::vector<bool> displaced(componentIds.size(), false);
        for (const auto& patch : patches)
            if (displace && patch.Metadata.X == options.DisplacementMaterial)
            {
                displaced[patch.Metadata.Z] = true;
                selected = true;
            }
        if (displace && !selected)
            throw std::invalid_argument("Displacement material has no triangle geometry");
        for (auto& patch : patches)
            if (displaced[patch.Metadata.Z])
            {
                if (scene.Materials[patch.Metadata.X].SampledEmitter)
                    throw std::invalid_argument("A displaced component intersects an analytic emitter proxy; "
                                                "edit the light geometry instead");
                for (const auto& vertex : patch.Corners)
                    if (!std::isfinite(vertex.Position.X / options.Wavelength) ||
                        !std::isfinite(vertex.Position.Y + std::abs(options.Amplitude)) ||
                        !std::isfinite(vertex.Position.Y - std::abs(options.Amplitude)))
                        throw std::invalid_argument(
                            "Displacement cannot be represented at this world-coordinate scale");
                patch.Metadata.Y |= M::Displace;
            }
        M::Controls controls;
        const auto forward = Normalize(camera.Target - camera.Position);
        const auto right = Normalize(Cross(forward, camera.Up));
        const auto up = Cross(right, forward);
        const float halfHeight = std::tan(camera.VerticalFieldOfViewDegrees * Pi / 360);
        controls.CameraPositionTarget = Pack(camera.Position);
        controls.CameraPositionTarget.W = options.TargetEdgePixels;
        controls.CameraForwardHalfHeight = Pack(forward);
        controls.CameraForwardHalfHeight.W = halfHeight;
        controls.CameraRightHalfWidth = Pack(right);
        controls.CameraRightHalfWidth.W =
            halfHeight * float(FullImageWidth(settings)) / FullImageHeight(settings);
        controls.CameraUpAmplitude = Pack(up);
        controls.CameraUpAmplitude.W = options.Amplitude;
        controls.Wave = {options.Wavelength, options.PhaseCycles - std::floor(options.PhaseCycles), 0, 0};
        controls.Populations = {uint32_t(patches.size()), uint32_t(componentIds.size()), 0, M::PlanPass};
        controls.Extent = {FullImageWidth(settings), FullImageHeight(settings), options.MaximumLevel, 0};
        Fingerprint fingerprint;
        constexpr uint32_t PlanFormatVersion = 1;
        fingerprint.Word(PlanFormatVersion);
        fingerprint.Vector(controls.CameraPositionTarget);
        fingerprint.Vector(controls.CameraForwardHalfHeight);
        fingerprint.Vector(controls.CameraRightHalfWidth);
        fingerprint.Vector(controls.CameraUpAmplitude);
        fingerprint.Vector(controls.Wave);
        fingerprint.Vector(controls.Populations);
        fingerprint.Vector(controls.Extent);
        for (const auto& patch : patches)
        {
            fingerprint.Vector(patch.Metadata);
            fingerprint.Vector(patch.CornerIds);
            for (const auto& vertex : patch.Corners)
            {
                fingerprint.Vector(vertex.Position);
                fingerprint.Vector(vertex.Normal);
                fingerprint.Vector(vertex.Color);
                fingerprint.Vector(vertex.TexCoord);
            }
        }
        auto plan = std::make_shared<MicropolygonPlan>();
        plan->Fingerprint = fingerprint.Get();
        if (settings.DicingPlan)
        {
            if (settings.DicingPlan->Fingerprint != plan->Fingerprint ||
                settings.DicingPlan->Components.size() != componentIds.size())
                throw std::invalid_argument(
                    "Micropolygon plan does not match this scene and full-film camera");
            plan->Components = settings.DicingPlan->Components;
        }
        else
        {
            plan->Components.assign(componentIds.size(), 0);
            controls.Dispatch = {0, uint32_t(patches.size()), 0, 0};
            if (!patches.empty() && !native(controls, patches, {}, plan->Components, result.Error))
            {
                if (Stopped(cancelled))
                    cancel();
                return result;
            }
        }
        if (Stopped(cancelled))
        {
            cancel();
            return result;
        }
        if (plan->Components.size() != componentIds.size())
            throw std::runtime_error("Incomplete native dicing plan");
        for (auto word : plan->Components)
        {
            if ((word & ~(M::LevelMask | M::LevelClamped)) || (word & M::LevelMask) > options.MaximumLevel ||
                ((word & M::LevelClamped) && (word & M::LevelMask) != options.MaximumLevel))
                throw std::runtime_error("Invalid native dicing level");
            result.ClampedComponents += (word & M::LevelClamped) != 0;
        }
        std::vector<M::Range> ranges;
        ranges.reserve(patches.size());
        uint64_t total = 0;
        for (const auto& patch : patches)
        {
            const auto word = plan->Components[patch.Metadata.Z];
            const uint32_t segments = 1u << (word & M::LevelMask);
            const uint64_t next = total + uint64_t(segments) * segments;
            if (next > options.TriangleBudget)
                throw std::length_error("Micropolygon plan exceeds the triangle budget; increase the pixel "
                                        "target or reduce the maximum level");
            ranges.push_back({uint32_t(total), uint32_t(next), segments, word});
            total = next;
        }
        std::vector<uint32_t> words(size_t(total) * M::OutputTriangleWords);
        controls.Populations.Z = uint32_t(total);
        controls.Populations.W = M::EmitPass;
        controls.Dispatch = {0, uint32_t(total), 0, 0};
        if (total && !native(controls, patches, ranges, words, result.Error))
        {
            if (Stopped(cancelled))
                cancel();
            return result;
        }
        if (Stopped(cancelled))
        {
            cancel();
            return result;
        }
        if (words.size() != total * M::OutputTriangleWords)
            throw std::runtime_error("Incomplete native micropolygon geometry");
        auto prepared = std::make_shared<Scene>(scene);
        prepared->Triangles.clear();
        prepared->Geometries.clear();
        prepared->Instances.clear();
        std::vector<std::shared_ptr<SceneGeometry>> outputGroups(groups.size());
        for (uint32_t group = 1; group < groups.size(); ++group)
            outputGroups[group] = std::make_shared<SceneGeometry>();
        std::vector<size_t> groupPopulations(groups.size(), 0);
        for (size_t patch = 0; patch < patches.size(); ++patch)
            groupPopulations[patches[patch].Metadata.W] += ranges[patch].End - ranges[patch].First;
        prepared->Triangles.reserve(groupPopulations[0]);
        for (uint32_t group = 1; group < groups.size(); ++group)
            outputGroups[group]->Triangles.reserve(groupPopulations[group]);
        for (size_t patchIndex = 0; patchIndex < patches.size(); ++patchIndex)
        {
            const auto& patch = patches[patchIndex];
            const auto& group = groups[patch.Metadata.W];
            auto& destination = group.Root ? prepared->Triangles : outputGroups[patch.Metadata.W]->Triangles;
            const auto range = ranges[patchIndex];
            for (uint32_t index = range.First; index < range.End; ++index)
            {
                if (Stopped(cancelled))
                {
                    cancel();
                    return result;
                }
                M::OutputTriangle emitted;
                std::memcpy(&emitted, words.data() + size_t(index) * M::OutputTriangleWords, sizeof(emitted));
                for (auto& vertex : emitted.Corners)
                {
                    if (!Finite(Unpack(vertex.Position)) || !Finite(Unpack(vertex.Normal)) ||
                        !Finite(Unpack(vertex.Color)) || !std::isfinite(vertex.TexCoord.X) ||
                        !std::isfinite(vertex.TexCoord.Y))
                        throw std::runtime_error("Native dicing produced nonfinite geometry");
                    if (group.Instance.Motion.Enabled)
                    {
                        // Preserve A * M(t) * p: deformation is authored in the
                        // reference world pose, then stored back in object space.
                        vertex.Position =
                            Pack(Affine::Point(group.Transform.Inverse, Unpack(vertex.Position)));
                        vertex.Normal = Pack(Affine::Normal(group.Transform.Forward, Unpack(vertex.Normal)));
                        if (!Finite(Unpack(vertex.Position)) || !Finite(Unpack(vertex.Normal)))
                            throw std::runtime_error(
                                "Diced geometry cannot be represented in the moving instance frame");
                    }
                }
                if (group.Instance.Motion.Enabled && group.Transform.Mirrored)
                    std::swap(emitted.Corners[1], emitted.Corners[2]);
                const auto& a = emitted.Corners[0];
                const auto& b = emitted.Corners[1];
                const auto& c = emitted.Corners[2];
                Triangle triangle;
                triangle.Position0 = Unpack(a.Position);
                triangle.Position1 = Unpack(b.Position);
                triangle.Position2 = Unpack(c.Position);
                triangle.Normal0 = Unpack(a.Normal);
                triangle.Normal1 = Unpack(b.Normal);
                triangle.Normal2 = Unpack(c.Normal);
                triangle.Color0 = Unpack(a.Color);
                triangle.Color1 = Unpack(b.Color);
                triangle.Color2 = Unpack(c.Color);
                triangle.TexCoord0 = {a.TexCoord.X, a.TexCoord.Y};
                triangle.TexCoord1 = {b.TexCoord.X, b.TexCoord.Y};
                triangle.TexCoord2 = {c.TexCoord.X, c.TexCoord.Y};
                triangle.MaterialIndex = patch.Metadata.X;
                triangle.SmoothNormals = (patch.Metadata.Y & (M::SmoothNormals | M::Displace)) != 0;
                destination.push_back(triangle);
            }
        }
        for (uint32_t groupIndex = 1; groupIndex < groups.size(); ++groupIndex)
        {
            const auto& group = groups[groupIndex];
            if (!outputGroups[groupIndex]->Triangles.empty())
            {
                auto instance = group.Instance;
                instance.GeometryIndex = uint32_t(prepared->Geometries.size());
                if (!instance.Motion.Enabled)
                    instance.ObjectToWorld = Identity;
                prepared->Geometries.push_back(outputGroups[groupIndex]);
                prepared->Instances.push_back(instance);
                result.InstanceSources.push_back(groupIndex - 1);
            }
            if (!group.Source->Spheres.empty())
            {
                auto spheres = std::make_shared<SceneGeometry>();
                spheres->Spheres = group.Source->Spheres;
                auto instance = group.Instance;
                instance.GeometryIndex = uint32_t(prepared->Geometries.size());
                prepared->Geometries.push_back(std::move(spheres));
                prepared->Instances.push_back(instance);
                result.InstanceSources.push_back(groupIndex - 1);
            }
        }
        result.Geometry = std::move(prepared);
        result.Plan = std::move(plan);
        result.InputTriangles = uint32_t(patches.size());
        result.OutputTriangles = uint32_t(total);
        result.Status = RenderStatus::Completed;
    }
    catch (const std::exception& error)
    {
        result.Error = error.what();
    }
    return result;
}
} // namespace Detail
} // namespace Arvo::RayTracing
