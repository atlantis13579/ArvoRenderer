#include "Editor/Demos.hpp"
#include "Maths/Vector3.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace Arvo::Editor
{
namespace
{

struct Bounds
{
    Float3 Minimum{
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max()};
    Float3 Maximum{
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest(),
        std::numeric_limits<float>::lowest()};
    bool Valid = false;
};

[[nodiscard]] Bounds ComputeBounds(const std::vector<MeshAsset>& meshes)
{
    Bounds bounds;
    for (const MeshAsset& mesh : meshes)
    {
        for (const Vertex& vertex : mesh.Vertices)
        {
            bounds.Valid = true;
            bounds.Minimum.X = std::min(bounds.Minimum.X, vertex.Position.X);
            bounds.Minimum.Y = std::min(bounds.Minimum.Y, vertex.Position.Y);
            bounds.Minimum.Z = std::min(bounds.Minimum.Z, vertex.Position.Z);
            bounds.Maximum.X = std::max(bounds.Maximum.X, vertex.Position.X);
            bounds.Maximum.Y = std::max(bounds.Maximum.Y, vertex.Position.Y);
            bounds.Maximum.Z = std::max(bounds.Maximum.Z, vertex.Position.Z);
        }
    }
    return bounds;
}

[[nodiscard]] Transform FrameTransform(const Bounds& bounds, float targetExtent)
{
    if (!bounds.Valid)
    {
        return {};
    }
    const Float3 center{
        (bounds.Minimum.X + bounds.Maximum.X) * 0.5f,
        (bounds.Minimum.Y + bounds.Maximum.Y) * 0.5f,
        (bounds.Minimum.Z + bounds.Maximum.Z) * 0.5f};
    const float extent = std::max({
        bounds.Maximum.X - bounds.Minimum.X,
        bounds.Maximum.Y - bounds.Minimum.Y,
        bounds.Maximum.Z - bounds.Minimum.Z});
    const float scale = extent > 1.0e-6f ? targetExtent / extent : 1.0f;

    Transform transform;
    transform.Scale = {scale, scale, scale};
    transform.Translation = {-center.X * scale, -center.Y * scale, -center.Z * scale};
    return transform;
}

[[nodiscard]] Light CornellEmitter(const SceneSnapshot& scene)
{
    const auto material = std::find_if(scene.Materials.begin(), scene.Materials.end(),
        [](const Material& value) { return value.Name == "light"; });
    if (material == scene.Materials.end()) throw std::runtime_error("Cornell light material is missing");
    std::vector<Maths::Vector3> positions;
    Maths::Vector3 weightedCenter, normal;
    float area = 0.0f;
    for (const SceneNode& node : scene.Nodes)
    {
        if (!node.Mesh) continue;
        const auto& mesh = *node.Mesh;
        for (const auto& section : mesh.Sections)
        {
            if (section.MaterialName != material->Name) continue;
            for (std::uint32_t index = section.FirstIndex; index + 2 < section.FirstIndex + section.IndexCount; index += 3)
            {
                Maths::Vector3 points[3];
                for (std::uint32_t corner = 0; corner < 3; ++corner)
                {
                    const auto& position = mesh.Vertices.at(mesh.Indices.at(index + corner)).Position;
                    points[corner] = {position.X, position.Y, position.Z};
                    positions.push_back(points[corner]);
                }
                const auto cross = (points[1] - points[0]).Cross(points[2] - points[0]);
                const float triangleArea = cross.Length() * 0.5f;
                weightedCenter += (points[0] + points[1] + points[2]) * (triangleArea / 3.0f);
                normal += cross;
                area += triangleArea;
            }
        }
    }
    if (area <= 1.0e-8f || normal.LengthSquared() <= 1.0e-12f)
        throw std::runtime_error("Cornell light mesh has no emitting surface");
    const auto center = weightedCenter / area;
    normal = normal.Normalized();
    const auto helper = std::fabs(normal.y) < 0.999f ? Maths::Vector3::UnitY() : Maths::Vector3::UnitX();
    const auto tangent = helper.Cross(normal).Normalized();
    const auto bitangent = normal.Cross(tangent).Normalized();
    float minimumX = std::numeric_limits<float>::max(), maximumX = -minimumX;
    float minimumY = minimumX, maximumY = -minimumX;
    for (const auto& position : positions)
    {
        const auto relative = position - center;
        if (std::fabs(relative.Dot(normal)) > 1.0e-5f)
            throw std::runtime_error("Cornell emitter must be planar");
        minimumX = std::min(minimumX, relative.Dot(tangent));
        maximumX = std::max(maximumX, relative.Dot(tangent));
        minimumY = std::min(minimumY, relative.Dot(bitangent));
        maximumY = std::max(maximumY, relative.Dot(bitangent));
    }
    const float width = maximumX - minimumX, height = maximumY - minimumY;
    if (std::fabs(width * height - area) > area * 1.0e-4f)
        throw std::runtime_error("Cornell emitter must match the sampled rectangle");
    Light light;
    light.Type = LightType::Area;
    light.Position = {center.x, center.y, center.z};
    light.Direction = {normal.x, normal.y, normal.z};
    light.LightColor = material->Emissive;
    // The finite-light estimator multiplies Ke by area / squared distance.
    light.Intensity = area;
    light.AreaSize = {width, height};
    light.CastShadows = true;
    return light;
}

[[nodiscard]] std::filesystem::path ModelPath(DemoId demo)
{
    switch (demo)
    {
    case DemoId::CornellBox:
        return "Scenes/CornellBox/CornellBox-Original.obj";
    case DemoId::SponzaAtrium:
        return "Scenes/Sponza/sponza.obj";
    case DemoId::StanfordDragon:
        return "Scenes/Dragon/Dragon_80K.obj";
    }
    throw std::invalid_argument("Unknown scene");
}

} // namespace

DemoCatalog::DemoCatalog(std::filesystem::path contentRoot)
    : ContentRoot_(std::move(contentRoot))
{
}

const std::vector<DemoDescriptor>& DemoCatalog::Descriptors()
{
    static const std::vector<DemoDescriptor> descriptors = {
        {DemoId::CornellBox, "cornell", "Cornell Box", "A compact scene for shadows and light transport."},
        {DemoId::SponzaAtrium, "sponza", "Sponza Atrium", "A detailed architectural stress scene."},
        {DemoId::StanfordDragon, "dragon", "Stanford Dragon",
         "Clear glass, frosted refraction and subsurface scattering in a ceiling-lit Cornell room."}};
    return descriptors;
}

std::string_view DemoKey(DemoId demo)
{
    for (const auto& descriptor : DemoCatalog::Descriptors())
        if (descriptor.Id == demo) return descriptor.Key;
    throw std::invalid_argument("Unknown demo identifier");
}

bool ParseDemo(std::string_view key, DemoId& demo)
{
    for (const auto& descriptor : DemoCatalog::Descriptors())
        if (descriptor.Key == key)
        {
            demo = descriptor.Id;
            return true;
        }
    return false;
}

SceneSnapshot DemoCatalog::Load(DemoId demo) const
{
    if (demo == DemoId::StanfordDragon) return LoadDragonDemo(ModelPath(demo));
    return LoadModelDemo(demo, ModelPath(demo));
}

SceneSnapshot DemoCatalog::LoadModelDemo(
    DemoId demo,
    const std::filesystem::path& relativePath) const
{
    LoadedModel model = FileResourceLoader::LoadObj(ContentRoot_ / relativePath);
    if (model.Geometry.Meshes.empty())
    {
        throw std::runtime_error("The demo resource contains no triangles: " + relativePath.string());
    }

    SceneSnapshot scene;
    scene.Demo = demo;
    scene.Revision = 1;
    scene.Materials = std::move(model.Materials);

    const float targetExtent = demo == DemoId::SponzaAtrium ? 5.0f : 3.0f;
    const Transform frameTransform = FrameTransform(ComputeBounds(model.Geometry.Meshes), targetExtent);
    SceneNodeId nodeId = 1;
    for (MeshAsset& mesh : model.Geometry.Meshes)
    {
        // Bake the common framing transform so rotating multi-object models uses
        // the normalized model center as its pivot in every render adapter.
        for (Vertex& vertex : mesh.Vertices)
        {
            vertex.Position.X = vertex.Position.X * frameTransform.Scale.X +
                                frameTransform.Translation.X;
            vertex.Position.Y = vertex.Position.Y * frameTransform.Scale.Y +
                                frameTransform.Translation.Y;
            vertex.Position.Z = vertex.Position.Z * frameTransform.Scale.Z +
                                frameTransform.Translation.Z;
        }
        SceneNode node;
        node.Id = nodeId++;
        node.Name = mesh.Name.empty() ? "Mesh " + std::to_string(node.Id) : mesh.Name;
        node.Mesh = std::make_shared<MeshAsset>(std::move(mesh));
        scene.Nodes.push_back(std::move(node));
    }

    if (demo == DemoId::CornellBox)
    {
        for (auto& material : scene.Materials)
            material.SampledEmitter = material.Name == "light";
        scene.SceneCamera.Position = {0.0f, 0.0f, 4.2f};
        scene.SceneCamera.Target = {0.0f, 0.0f, 0.0f};
        scene.Lights.push_back(CornellEmitter(scene));
    }
    else if (demo == DemoId::SponzaAtrium)
    {
        // Place the camera at human height inside the centered atrium and look
        // along its main colonnade instead of at the mostly closed exterior.
        scene.SceneCamera.Position = {-1.15f, -0.55f, 0.0f};
        scene.SceneCamera.Target = {1.6f, -0.45f, 0.0f};
        scene.SceneCamera.VerticalFieldOfViewDegrees = 72.0f;
        scene.Lights.push_back(
            {LightType::Directional, {}, {-0.35f, -1.0f, -0.25f}, {}, 3.0f, {}, true});
    }
    else
    {
        scene.SceneCamera.Position = {0.0f, 1.4f, 4.5f};
        scene.SceneCamera.Target = {0.0f, 0.3f, 0.0f};
        scene.Lights.push_back(
            {LightType::Directional, {}, {-0.35f, -1.0f, -0.25f}, {}, 3.0f, {}, true});
    }
    return scene;
}

} // namespace Arvo::Editor
