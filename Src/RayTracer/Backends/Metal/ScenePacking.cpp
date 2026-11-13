#include "ScenePacking.h"
#include "BvhBuilder.h"
#include "KdTreeBuilder.h"
#include "BihBuilder.h"
#include "SpatialBvhBuilder.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace Arvo::RayTracing::MetalScene
{
namespace
{
// Pack immutable geometry once per object and retain object-local BLAS indices.
// All numerical tree construction remains in this backend's own algorithm files.
constexpr uint32_t CancellationStride = 4096;
constexpr float MinimumVectorLengthSquared = 1.0e-20f;
constexpr float MinimumTriangleAreaSquared = 1.0e-16f;
[[nodiscard]] Vec3 ToVec3(const Float3 &value)
{
    return {value.X, value.Y, value.Z};
}

[[nodiscard]] Vec4 ToVec4(const Float3 &value, float w = 0.0f)
{
    return {value.X, value.Y, value.Z, w};
}

[[nodiscard]] Vec3 Add(const Vec3 &left, const Vec3 &right)
{
    return {left.X + right.X, left.Y + right.Y, left.Z + right.Z};
}

[[nodiscard]] Vec3 Subtract(const Vec3 &left, const Vec3 &right)
{
    return {left.X - right.X, left.Y - right.Y, left.Z - right.Z};
}

[[nodiscard]] Vec3 Multiply(const Vec3 &value, float scalar)
{
    return {value.X * scalar, value.Y * scalar, value.Z * scalar};
}

[[nodiscard]] float DotVector(const Vec3 &left, const Vec3 &right)
{
    return left.X * right.X + left.Y * right.Y + left.Z * right.Z;
}

[[nodiscard]] Vec3 CrossVector(const Vec3 &left, const Vec3 &right)
{
    return {left.Y * right.Z - left.Z * right.Y, left.Z * right.X - left.X * right.Z,
            left.X * right.Y - left.Y * right.X};
}

[[nodiscard]] Vec3 NormalizeVector(const Vec3 &value)
{
    const float lengthSquared = DotVector(value, value);
    if (lengthSquared <= MinimumVectorLengthSquared)
    {
        return {};
    }
    return Multiply(value, 1.0f / std::sqrt(lengthSquared));
}

[[nodiscard]] bool IsFinite(const Float3 &value)
{
    return std::isfinite(value.X) && std::isfinite(value.Y) && std::isfinite(value.Z);
}

[[nodiscard]] bool IsFinite(const Float2 &value)
{
    return std::isfinite(value.X) && std::isfinite(value.Y);
}

struct Bounds
{
    Vec3 Minimum{std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
                 std::numeric_limits<float>::max()};
    Vec3 Maximum{-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(),
                 -std::numeric_limits<float>::max()};

    void Expand(const Vec3 &point)
    {
        Minimum.X = std::min(Minimum.X, point.X);
        Minimum.Y = std::min(Minimum.Y, point.Y);
        Minimum.Z = std::min(Minimum.Z, point.Z);
        Maximum.X = std::max(Maximum.X, point.X);
        Maximum.Y = std::max(Maximum.Y, point.Y);
        Maximum.Z = std::max(Maximum.Z, point.Z);
    }

    void Expand(const Bounds &bounds)
    {
        Expand(bounds.Minimum);
        Expand(bounds.Maximum);
    }

    [[nodiscard]] Vec3 Center() const
    {
        return {float((double(Minimum.X) + Maximum.X) / 2), float((double(Minimum.Y) + Maximum.Y) / 2),
                float((double(Minimum.Z) + Maximum.Z) / 2)};
    }
};

struct BuildPrimitive
{
    Bounds BoundingBox;
    Vec3 Centroid;
    PackedPrimitiveRef Reference;
};

class ScenePacker
{
  public:
    explicit ScenePacker(Detail::CancellationProbe cancellation) : Cancellation_(std::move(cancellation)) {}

    bool Pack(const Scene &scene, AccelerationOptions acceleration, PackedScene &output, std::string &error)
    {
        output = {};
        if (!Detail::BuildRayTextureAtlas(scene, output.TextureAtlas, error))
            return false;
        output.Materials.reserve(scene.Materials.size());
        output.Lights.reserve(scene.DirectionalLights.size() + scene.PointLights.size() +
                              scene.RectAreaLights.size());
        for (std::uint32_t index = 0; index < scene.Materials.size(); ++index)
        {
            if (index % CancellationStride == 0 && IsCancelled())
            {
                return Cancel(error);
            }
            const Material &source = scene.Materials[index];
            PackedMaterial material;
            material.AlbedoReflectivity = ToVec4(source.Albedo, source.Reflectivity);
            material.AmbientShininess = ToVec4(source.AmbientColor, source.Shininess);
            material.Specular = ToVec4(source.SpecularColor);
            material.Emission = ToVec4(source.Emission);
            material.DiffuseTextureIndex = source.DiffuseTextureIndex;
            material.SampledEmitter = source.SampledEmitter ? 1u : 0u;
            material.Model = static_cast<uint32_t>(source.Model);
            material.Optical = {source.MicrofacetSlope, source.ExteriorIndex, source.InteriorIndex, source.SubsurfaceMeanFreePath};
            material.ConductorIndex = ToVec4(source.ConductorIndex, source.SubsurfaceAnisotropy);
            material.ConductorExtinction = ToVec4(source.ConductorExtinction, source.SubsurfaceAlbedoScale);
            output.Materials.push_back(material);
        }
        for (std::uint32_t index = 0; index < scene.DirectionalLights.size(); ++index)
        {
            if (index % CancellationStride == 0 && IsCancelled())
            {
                return Cancel(error);
            }
            const DirectionalLight &source = scene.DirectionalLights[index];
            const Vec3 direction = NormalizeVector(ToVec3(source.DirectionToLight));
            PackedLight light;
            light.Type = LightTypeDirectional;
            light.Direction = direction;
            light.Intensity = source.Intensity;
            light.Color = ToVec3(source.Color);
            light.CastShadows = source.CastShadows ? 1u : 0u;
            output.Lights.push_back(light);
        }
        for (std::uint32_t index = 0; index < scene.PointLights.size(); ++index)
        {
            if (index % CancellationStride == 0 && IsCancelled())
            {
                return Cancel(error);
            }
            const PointLight &source = scene.PointLights[index];
            PackedLight light;
            light.Position = ToVec3(source.Position);
            light.Type = LightTypePoint;
            light.Intensity = source.Intensity;
            light.Color = ToVec3(source.Color);
            light.CastShadows = source.CastShadows ? 1u : 0u;
            output.Lights.push_back(light);
        }
        for (std::uint32_t index = 0; index < scene.RectAreaLights.size(); ++index)
        {
            if (index % CancellationStride == 0 && IsCancelled())
            {
                return Cancel(error);
            }
            const RectAreaLight &source = scene.RectAreaLights[index];
            PackedLight light;
            light.Position = ToVec3(source.Position);
            light.Type = LightTypeRectArea;
            light.Direction = NormalizeVector(ToVec3(source.Direction));
            light.Intensity = source.Intensity;
            light.Color = ToVec3(source.Color);
            light.CastShadows = source.CastShadows ? 1u : 0u;
            light.Width = source.Size.X;
            light.Height = source.Size.Y;
            output.Lights.push_back(light);
        }

        if (scene.Instances.empty())
            return PackObject(scene, acceleration, output, error);
        MetalInstances::Hierarchy hierarchy;
        if (!MetalInstances::Build(scene, Cancellation_, hierarchy, error))
            return false;
        std::vector<MetalInstances::GeometryRange> ranges;
        ranges.reserve(hierarchy.Objects.size());
        constexpr size_t CopyChunk = CancellationStride;
        const auto append = [&](auto &destination, const auto &source) {
            destination.reserve(destination.size() + source.size());
            for (size_t begin = 0; begin < source.size(); begin += CopyChunk)
            {
                if (IsCancelled())
                    return Cancel(error);
                const size_t end = std::min(begin + CopyChunk, source.size());
                destination.insert(destination.end(), source.begin() + begin, source.begin() + end);
            }
            return true;
        };
        for (const SceneGeometry *geometry : hierarchy.Objects)
        {
            if (IsCancelled())
                return Cancel(error);
            PackedScene object;
            if (!PackObject(*geometry, acceleration, object, error))
                return false;
            constexpr uint64_t IndexLimit = std::numeric_limits<uint32_t>::max();
            if (uint64_t(output.Nodes.size()) + object.Nodes.size() > IndexLimit ||
                uint64_t(output.PrimitiveRefs.size()) + object.PrimitiveRefs.size() > IndexLimit ||
                uint64_t(output.Triangles.size()) + object.Triangles.size() > IndexLimit ||
                uint64_t(output.Spheres.size()) + object.Spheres.size() > IndexLimit)
            {
                error = "Metal shared object buffers exceed native 32-bit indexing.";
                return false;
            }
            MetalInstances::GeometryRange range;
            range.NodeBegin = uint32_t(output.Nodes.size());
            range.NodeCount = uint32_t(object.Nodes.size());
            range.ReferenceBegin = uint32_t(output.PrimitiveRefs.size());
            range.TriangleBegin = uint32_t(output.Triangles.size());
            range.SphereBegin = uint32_t(output.Spheres.size());
            range.TriangleCount = uint32_t(object.Triangles.size());
            if (!append(output.Nodes, object.Nodes) || !append(output.PrimitiveRefs, object.PrimitiveRefs) ||
                !append(output.Triangles, object.Triangles) || !append(output.Spheres, object.Spheres))
                return false;
            ranges.push_back(range);
        }
        output.Instances = std::move(hierarchy.Records);
        output.InstanceNodes = std::move(hierarchy.Nodes);
        for (size_t index = 0; index < output.Instances.size(); ++index)
        {
            if (index % CopyChunk == 0 && IsCancelled())
                return Cancel(error);
            auto &instance = output.Instances[index];
            const uint32_t first = instance.Geometry.FirstPrimitive;
            instance.Geometry = ranges[instance.ObjectIndex];
            instance.Geometry.FirstPrimitive = first;
        }
        return true;
    }

    bool PackObject(const SceneGeometry &scene, AccelerationOptions acceleration, PackedScene &output,
                    std::string &error)
    {
        Primitives_.clear();
        Primitives_.reserve(scene.Triangles.size() + scene.Spheres.size());
        output.Triangles.reserve(scene.Triangles.size());
        output.Spheres.reserve(scene.Spheres.size());
        for (std::uint32_t index = 0; index < scene.Triangles.size(); ++index)
        {
            if (index % CancellationStride == 0 && IsCancelled())
            {
                return Cancel(error);
            }
            const Triangle &source = scene.Triangles[index];
            const Vec3 position0 = ToVec3(source.Position0);
            const Vec3 edge1 = Subtract(ToVec3(source.Position1), position0);
            const Vec3 edge2 = Subtract(ToVec3(source.Position2), position0);
            PackedTriangle triangle;
            triangle.Position0 = {position0.X, position0.Y, position0.Z, 0.0f};
            triangle.Edge1 = {edge1.X, edge1.Y, edge1.Z, 0.0f};
            triangle.Edge2 = {edge2.X, edge2.Y, edge2.Z, 0.0f};
            triangle.Normal0 = ToVec4(source.Normal0);
            triangle.Normal1 = ToVec4(source.Normal1);
            triangle.Normal2 = ToVec4(source.Normal2);
            triangle.Color0 = ToVec4(source.Color0, 1.0f);
            triangle.Color1 = ToVec4(source.Color1, 1.0f);
            triangle.Color2 = ToVec4(source.Color2, 1.0f);
            triangle.TexCoord01 = {source.TexCoord0.X, source.TexCoord0.Y, source.TexCoord1.X,
                                   source.TexCoord1.Y};
            triangle.TexCoord2 = {source.TexCoord2.X, source.TexCoord2.Y, 0.0f, 0.0f};
            triangle.MaterialIndex = source.MaterialIndex;
            triangle.Flags = source.SmoothNormals ? 1u : 0u;
            output.Triangles.push_back(triangle);

            BuildPrimitive primitive;
            primitive.BoundingBox.Expand(position0);
            primitive.BoundingBox.Expand(Add(position0, edge1));
            primitive.BoundingBox.Expand(Add(position0, edge2));
            primitive.Centroid = primitive.BoundingBox.Center();
            primitive.Reference.TypeAndIndex = index;
            Primitives_.push_back(primitive);
        }
        for (std::uint32_t index = 0; index < scene.Spheres.size(); ++index)
        {
            if (index % CancellationStride == 0 && IsCancelled())
            {
                return Cancel(error);
            }
            const Sphere &source = scene.Spheres[index];
            PackedSphere sphere;
            sphere.CenterRadius = ToVec4(source.Center, source.Radius);
            sphere.MaterialIndex = source.MaterialIndex;
            output.Spheres.push_back(sphere);

            const Vec3 center = ToVec3(source.Center);
            const Vec3 radius{source.Radius, source.Radius, source.Radius};
            BuildPrimitive primitive;
            primitive.BoundingBox.Expand(Subtract(center, radius));
            primitive.BoundingBox.Expand(Add(center, radius));
            primitive.Centroid = center;
            primitive.Reference.TypeAndIndex = SphereBit | index;
            Primitives_.push_back(primitive);
        }
        if (acceleration.Type == AccelerationType::Bvh &&
            acceleration.BvhPolicy == BvhBuildPolicy::SpatialSplits)
        {
            std::vector<MetalSpatial::Reference> pieces;
            pieces.reserve(Primitives_.size());
            for (size_t index = 0; index < Primitives_.size(); ++index)
            {
                if (index % CancellationStride == 0 && IsCancelled())
                    return Cancel(error);
                const BuildPrimitive &source = Primitives_[index];
                const auto &box = source.BoundingBox;
                pieces.push_back({{box.Minimum.X, box.Minimum.Y, box.Minimum.Z},
                                  {box.Maximum.X, box.Maximum.Y, box.Maximum.Z},
                                  source.Reference.TypeAndIndex});
            }
            MetalSpatial::Hierarchy hierarchy;
            if (!MetalSpatial::Build(pieces, Cancellation_, hierarchy, error))
                return false;
            output.Nodes.resize(hierarchy.Nodes.size());
            for (size_t index = 0; index < hierarchy.Nodes.size(); ++index)
            {
                if (index % CancellationStride == 0 && IsCancelled())
                    return Cancel(error);
                const auto &node = hierarchy.Nodes[index];
                auto &packed = output.Nodes[index];
                packed.MinX = node.Lower[0];
                packed.MinY = node.Lower[1];
                packed.MinZ = node.Lower[2];
                packed.MaxX = node.Upper[0];
                packed.MaxY = node.Upper[1];
                packed.MaxZ = node.Upper[2];
                packed.FirstOrEscape = node.FirstOrEscape;
                packed.PrimitiveCount = node.Count;
            }
            output.PrimitiveRefs.reserve(hierarchy.References.size());
            for (size_t index = 0; index < hierarchy.References.size(); ++index)
            {
                if (index % CancellationStride == 0 && IsCancelled())
                    return Cancel(error);
                output.PrimitiveRefs.push_back({hierarchy.References[index]});
            }
        }
        else if (acceleration.Type == AccelerationType::KdTree)
        {
            std::vector<MetalKd::PrimitiveBounds> spatialPrimitives;
            spatialPrimitives.reserve(Primitives_.size());
            for (std::size_t index = 0; index < Primitives_.size(); ++index)
            {
                if (index % CancellationStride == 0 && IsCancelled())
                    return Cancel(error);
                const BuildPrimitive &primitive = Primitives_[index];
                const Bounds &bounds = primitive.BoundingBox;
                spatialPrimitives.push_back({{bounds.Minimum.X, bounds.Minimum.Y, bounds.Minimum.Z},
                                             {bounds.Maximum.X, bounds.Maximum.Y, bounds.Maximum.Z},
                                             primitive.Reference.TypeAndIndex});
            }
            std::vector<MetalKd::SpatialNode> spatialNodes;
            std::vector<std::uint32_t> spatialReferences;
            if (!MetalKd::Build(spatialPrimitives, Cancellation_, spatialNodes, spatialReferences, error))
                return false;
            output.Nodes.resize(spatialNodes.size());
            for (std::size_t index = 0; index < spatialNodes.size(); ++index)
            {
                if (index % CancellationStride == 0 && IsCancelled())
                    return Cancel(error);
                const MetalKd::SpatialNode &source = spatialNodes[index];
                PackedAccelerationNode &target = output.Nodes[index];
                target.MinX = source.Lower[0];
                target.MinY = source.Lower[1];
                target.MinZ = source.Lower[2];
                target.MaxX = source.Upper[0];
                target.MaxY = source.Upper[1];
                target.MaxZ = source.Upper[2];
                target.FirstOrEscape = source.ReferenceBegin;
                target.PrimitiveCount = source.ReferenceCount;
                target.SplitPosition = source.Plane;
                target.SplitAxis = source.Axis;
                target.AboveChild = source.UpperChild;
            }
            output.PrimitiveRefs.reserve(spatialReferences.size());
            for (std::size_t index = 0; index < spatialReferences.size(); ++index)
            {
                if (index % CancellationStride == 0 && IsCancelled())
                    return Cancel(error);
                output.PrimitiveRefs.push_back({spatialReferences[index]});
            }
        }
        else if (acceleration.Type == AccelerationType::Bih)
        {
            std::vector<MetalBih::Object> objects;
            objects.reserve(Primitives_.size());
            for (std::size_t index = 0; index < Primitives_.size(); ++index)
            {
                if (index % CancellationStride == 0 && IsCancelled())
                    return Cancel(error);
                const BuildPrimitive &primitive = Primitives_[index];
                const auto &bounds = primitive.BoundingBox;
                objects.push_back({{bounds.Minimum.X, bounds.Minimum.Y, bounds.Minimum.Z},
                                   {bounds.Maximum.X, bounds.Maximum.Y, bounds.Maximum.Z},
                                   primitive.Reference.TypeAndIndex});
            }
            std::vector<MetalBih::IntervalNode> intervalNodes;
            std::vector<std::uint32_t> objectReferences;
            if (!MetalBih::Build(objects, Cancellation_, intervalNodes, objectReferences, error))
                return false;
            output.Nodes.resize(intervalNodes.size());
            for (std::size_t index = 0; index < intervalNodes.size(); ++index)
            {
                if (index % CancellationStride == 0 && IsCancelled())
                    return Cancel(error);
                const MetalBih::IntervalNode &node = intervalNodes[index];
                PackedAccelerationNode &packed = output.Nodes[index];
                packed.MinX = node.Lower[0];
                packed.MinY = node.Lower[1];
                packed.MinZ = node.Lower[2];
                packed.MaxX = node.Upper[0];
                packed.MaxY = node.Upper[1];
                packed.MaxZ = node.Upper[2];
                packed.FirstOrEscape = node.ReferenceBegin;
                packed.PrimitiveCount = node.ReferenceCount;
                packed.SplitAxis = node.Axis;
                packed.AboveChild = node.RightChild;
                packed.SplitPosition = node.LeftMaximum;
                packed.SecondPlane = node.RightMinimum;
            }
            output.PrimitiveRefs.reserve(objectReferences.size());
            for (std::size_t index = 0; index < objectReferences.size(); ++index)
            {
                if (index % CancellationStride == 0 && IsCancelled())
                    return Cancel(error);
                output.PrimitiveRefs.push_back({objectReferences[index]});
            }
        }
        else if (!Primitives_.empty())
        {
            std::vector<MetalBvh::Primitive> input;
            input.reserve(Primitives_.size());
            for (size_t index = 0; index < Primitives_.size(); ++index)
            {
                if (index % CancellationStride == 0 && IsCancelled())
                    return Cancel(error);
                const auto &primitive = Primitives_[index];
                const auto &box = primitive.BoundingBox;
                const auto &center = primitive.Centroid;
                input.push_back({{box.Minimum.X, box.Minimum.Y, box.Minimum.Z},
                                 {box.Maximum.X, box.Maximum.Y, box.Maximum.Z},
                                 {center.X, center.Y, center.Z},
                                 primitive.Reference.TypeAndIndex});
            }
            MetalBvh::Tree tree;
            if (!MetalBvh::Build(input, Cancellation_, tree, error))
                return false;
            output.Nodes.resize(tree.Nodes.size());
            for (size_t index = 0; index < tree.Nodes.size(); ++index)
            {
                if (index % CancellationStride == 0 && IsCancelled())
                    return Cancel(error);
                const auto &source = tree.Nodes[index];
                auto &target = output.Nodes[index];
                target.MinX = source.MinX;
                target.MinY = source.MinY;
                target.MinZ = source.MinZ;
                target.MaxX = source.MaxX;
                target.MaxY = source.MaxY;
                target.MaxZ = source.MaxZ;
                target.FirstOrEscape = source.FirstOrEscape;
                target.PrimitiveCount = source.PrimitiveCount;
            }
            output.PrimitiveRefs.reserve(tree.References.size());
            for (size_t index = 0; index < tree.References.size(); ++index)
            {
                if (index % CancellationStride == 0 && IsCancelled())
                    return Cancel(error);
                output.PrimitiveRefs.push_back({tree.References[index]});
            }
        }
        return true;
    }

  private:
    [[nodiscard]] bool IsCancelled() const { return Cancellation_ && Cancellation_(); }

    bool Cancel(std::string &error) const
    {
        error = "Metal scene packing was cancelled.";
        return false;
    }

    Detail::CancellationProbe Cancellation_;
    std::vector<BuildPrimitive> Primitives_;
};

} // namespace

[[nodiscard]] bool ValidateGeometry(const SceneGeometry &geometry, size_t materialCount, std::string &error)
{
    for (const Triangle &triangle : geometry.Triangles)
    {
        if (triangle.MaterialIndex >= materialCount || !IsFinite(triangle.Position0) ||
            !IsFinite(triangle.Position1) || !IsFinite(triangle.Position2) || !IsFinite(triangle.Normal0) ||
            !IsFinite(triangle.Normal1) || !IsFinite(triangle.Normal2) || !IsFinite(triangle.Color0) ||
            !IsFinite(triangle.Color1) || !IsFinite(triangle.Color2) || !IsFinite(triangle.TexCoord0) ||
            !IsFinite(triangle.TexCoord1) || !IsFinite(triangle.TexCoord2))
        {
            error = "Metal triangle data is invalid.";
            return false;
        }
        const Vec3 edge1 = Subtract(ToVec3(triangle.Position1), ToVec3(triangle.Position0));
        const Vec3 edge2 = Subtract(ToVec3(triangle.Position2), ToVec3(triangle.Position0));
        const Vec3 area = CrossVector(edge1, edge2);
        const float edge1LengthSquared = DotVector(edge1, edge1);
        const float edge2LengthSquared = DotVector(edge2, edge2);
        const float areaSquared = DotVector(area, area);
        if (!std::isfinite(edge1LengthSquared) || !std::isfinite(edge2LengthSquared) ||
            !std::isfinite(areaSquared) || areaSquared < MinimumTriangleAreaSquared)
        {
            error = "Metal does not accept degenerate or numerically unsafe triangles.";
            return false;
        }
    }
    for (const Sphere &sphere : geometry.Spheres)
    {
        const float radiusSquared = sphere.Radius * sphere.Radius;
        const Float3 minimum{sphere.Center.X - sphere.Radius, sphere.Center.Y - sphere.Radius,
                             sphere.Center.Z - sphere.Radius};
        const Float3 maximum{sphere.Center.X + sphere.Radius, sphere.Center.Y + sphere.Radius,
                             sphere.Center.Z + sphere.Radius};
        if (sphere.MaterialIndex >= materialCount || !IsFinite(sphere.Center) ||
            !std::isfinite(sphere.Radius) || sphere.Radius <= 0.0f || !std::isfinite(radiusSquared) ||
            !IsFinite(minimum) || !IsFinite(maximum))
        {
            error = "Metal sphere data is invalid.";
            return false;
        }
    }
    return true;
}

bool Pack(const Scene &scene, AccelerationOptions acceleration, const Detail::CancellationProbe &cancellation,
          PackedScene &output, std::string &error)
{
    output = {};
    error.clear();
    if (cancellation && cancellation())
    {
        error = "Metal scene packing was cancelled.";
        return false;
    }
    if (!ValidateAcceleration(acceleration, error) || !MetalInstances::Validate(scene, error) ||
        !ValidateGeometry(scene, scene.Materials.size(), error))
        return false;
    for (const auto &geometry : scene.Geometries)
        if (!ValidateGeometry(*geometry, scene.Materials.size(), error))
            return false;
    PackedScene pending;
    if (!ScenePacker(cancellation).Pack(scene, acceleration, pending, error))
        return false;
    output = std::move(pending);
    return true;
}
} // namespace Arvo::RayTracing::MetalScene
