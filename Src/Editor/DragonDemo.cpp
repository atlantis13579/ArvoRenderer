#include "Editor/Demos.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace Arvo::Editor
{
namespace
{
constexpr float DragonLength = 1.7f, DragonSpacing = 2.25f, FloorClearance = 0.04f;
// Keep the scan's smallest nonzero faces above native float geometry thresholds.
// Scale the entire optical scene, including interior scattering distances.
constexpr float WorldUnitsPerMeter = 10;
constexpr float GlassIndex = 1.5f, FrostedSlope = 0.14f;
constexpr float JadeMeanFreePath = 0.18f, JadeAlbedoScale = 0.98f, JadeAnisotropy = 0.2f;
constexpr float RoomWidth = 9, RoomDepth = 6, RoomHeight = 3.6f;
constexpr float CeilingLightRadiance = 1;
constexpr Color CeilingLightColor{1, 0.96f, 0.9f, 1};
constexpr unsigned FloorColumns = 12, FloorRows = 8;
constexpr Color FloorIvory{0.82f, 0.84f, 0.87f, 1};
constexpr Color FloorSlate{0.26f, 0.31f, 0.38f, 1};
constexpr Color LeftWallRed{0.63f, 0.065f, 0.05f, 1};
constexpr Color RightWallGreen{0.14f, 0.45f, 0.091f, 1};
constexpr unsigned BackdropPanels = 6;
constexpr Color BackdropBlue{0.18f, 0.38f, 0.65f, 1};
constexpr Color BackdropAmber{0.8f, 0.54f, 0.25f, 1};
constexpr Float3 CameraPosition{0, 1.8f, 7.5f}, CameraTarget{0, 1.5f, 0};
constexpr float CameraVerticalFieldOfView = 45;
constexpr std::array<const char*, 3> DragonMaterials{
    "Clear glass", "Frosted glass", "Jade scattering"};

Float3 WorldPosition(Float3 meters)
{
    return {meters.X * WorldUnitsPerMeter, meters.Y * WorldUnitsPerMeter, meters.Z * WorldUnitsPerMeter};
}

Material DiffuseMaterial(const char* name, Color color)
{
    Material result;
    result.Name = name;
    result.Diffuse = color;
    result.Scattering.Model = RayTracing::MaterialModel::Diffuse;
    return result;
}

void Quad(MeshAsset& mesh, const std::array<Float3, 4>& corners, Float3 normal, const char* material)
{
    const auto firstVertex = static_cast<uint32_t>(mesh.Vertices.size());
    const auto firstIndex = static_cast<uint32_t>(mesh.Indices.size());
    for (const auto& position : corners)
    {
        Vertex vertex;
        vertex.Position = position;
        vertex.Normal = normal;
        mesh.Vertices.push_back(vertex);
    }
    for (const uint32_t corner : {0u, 1u, 2u, 0u, 2u, 3u}) mesh.Indices.push_back(firstVertex + corner);
    mesh.Sections.push_back({material, firstIndex, static_cast<uint32_t>(mesh.Indices.size()) - firstIndex});
}
}

// The 87,130-triangle Stanford dragon is the mesh distributed with Sebastian
// Lague's Coding Adventure ray tracer. Preserve the downloaded OBJ; author only
// framing, lighting and materials here. See Contents/Scenes/Dragon/copyright.txt.
// Dielectric transmission uses Fresnel/Snell transport, not alpha blending.
// The jade object's closed boundary activates each offline backend's interior
// random walk. Preview adapters retain their own bounded approximations.
SceneSnapshot DemoCatalog::LoadDragonDemo(const std::filesystem::path& relativePath) const
{
    LoadedModel model = FileResourceLoader::LoadObj(ContentRoot_ / relativePath);
    if (model.Geometry.Meshes.size() != 1 || model.Geometry.Meshes.front().Vertices.empty())
        throw std::runtime_error("Dragon demo requires the single closed source mesh");
    auto& source = model.Geometry.Meshes.front();
    Float3 low{INFINITY, INFINITY, INFINITY}, high{-INFINITY, -INFINITY, -INFINITY};
    for (const auto& vertex : source.Vertices)
    {
        low.X = std::min(low.X, vertex.Position.X); high.X = std::max(high.X, vertex.Position.X);
        low.Y = std::min(low.Y, vertex.Position.Y); high.Y = std::max(high.Y, vertex.Position.Y);
        low.Z = std::min(low.Z, vertex.Position.Z); high.Z = std::max(high.Z, vertex.Position.Z);
    }
    const float extent = std::max({high.X - low.X, high.Y - low.Y, high.Z - low.Z});
    if (!(extent > 0)) throw std::runtime_error("Dragon demo has empty bounds");
    const float scale = DragonLength * WorldUnitsPerMeter / extent;
    const float centerX = (low.X + high.X) / 2, centerZ = (low.Z + high.Z) / 2;
    for (auto& vertex : source.Vertices)
    {
        const auto p = vertex.Position, n = vertex.Normal;
        // A proper quarter turn exposes the long profile without mirroring the boundary.
        vertex.Position = {(p.Z - centerZ) * scale, (p.Y - low.Y) * scale + FloorClearance * WorldUnitsPerMeter,
                           -(p.X - centerX) * scale};
        vertex.Normal = {n.Z, n.Y, -n.X};
    }

    SceneSnapshot scene;
    scene.Demo = DemoId::StanfordDragon;
    scene.Revision = 1;
    for (size_t index = 0; index < DragonMaterials.size(); ++index)
    {
        Material material;
        material.Name = DragonMaterials[index];
        material.Scattering.Model = RayTracing::MaterialModel::Dielectric;
        material.Scattering.MicrofacetSlope = index == 1 ? FrostedSlope : 0;
        material.Scattering.InteriorIndex = GlassIndex;
        if (index == 2)
        {
            material.Diffuse = {0.65f, 0.92f, 0.75f, 1};
            material.Scattering.SubsurfaceMeanFreePath = JadeMeanFreePath * WorldUnitsPerMeter;
            material.Scattering.SubsurfaceAlbedoScale = JadeAlbedoScale;
            material.Scattering.SubsurfaceAnisotropy = JadeAnisotropy;
        }
        scene.Materials.push_back(material);
        auto mesh = std::make_shared<MeshAsset>(source);
        mesh->Name = material.Name + " dragon";
        for (auto& section : mesh->Sections) section.MaterialName = material.Name;
        SceneNode node;
        node.Id = index + 1;
        node.Name = mesh->Name;
        node.Mesh = std::move(mesh);
        node.LocalTransform.Translation.X = (static_cast<float>(index) - 1) * DragonSpacing * WorldUnitsPerMeter;
        scene.Nodes.push_back(std::move(node));
    }

    scene.Materials.push_back(DiffuseMaterial("Ivory floor", FloorIvory));
    scene.Materials.push_back(DiffuseMaterial("Slate floor", FloorSlate));
    scene.Materials.push_back(DiffuseMaterial("Left wall red", LeftWallRed));
    scene.Materials.push_back(DiffuseMaterial("Right wall green", RightWallGreen));
    scene.Materials.push_back(DiffuseMaterial("Blue backdrop", BackdropBlue));
    scene.Materials.push_back(DiffuseMaterial("Amber backdrop", BackdropAmber));
    auto room = std::make_shared<MeshAsset>();
    room->Name = "Cornell room";
    const float halfWidth = RoomWidth / 2, halfDepth = RoomDepth / 2;
    // Leave the camera-facing side open; room faces and the emitting ceiling point inward.
    // Colored diffuse walls provide reflected color through glass and GI bounces.
    const float cellWidth = RoomWidth / FloorColumns, cellDepth = RoomDepth / FloorRows;
    for (unsigned row = 0; row < FloorRows; ++row)
        for (unsigned column = 0; column < FloorColumns; ++column)
        {
            const float x = column * cellWidth - halfWidth, z = row * cellDepth - halfDepth;
            Quad(*room, {{{x, 0, z}, {x, 0, z + cellDepth}, {x + cellWidth, 0, z + cellDepth},
                          {x + cellWidth, 0, z}}}, {0, 1, 0},
                 (column + row) % 2 ? "Slate floor" : "Ivory floor");
        }
    const float panelWidth = RoomWidth / BackdropPanels;
    for (unsigned panel = 0; panel < BackdropPanels; ++panel)
    {
        const float x = panel * panelWidth - halfWidth;
        Quad(*room, {{{x, 0, -halfDepth}, {x + panelWidth, 0, -halfDepth},
                      {x + panelWidth, RoomHeight, -halfDepth}, {x, RoomHeight, -halfDepth}}},
             {0, 0, 1}, panel % 2 ? "Amber backdrop" : "Blue backdrop");
    }
    Quad(*room, {{{-halfWidth, 0, halfDepth}, {-halfWidth, 0, -halfDepth},
                  {-halfWidth, RoomHeight, -halfDepth}, {-halfWidth, RoomHeight, halfDepth}}},
         {1, 0, 0}, "Left wall red");
    Quad(*room, {{{halfWidth, 0, -halfDepth}, {halfWidth, 0, halfDepth},
                  {halfWidth, RoomHeight, halfDepth}, {halfWidth, RoomHeight, -halfDepth}}},
         {-1, 0, 0}, "Right wall green");
    for (auto& vertex : room->Vertices) vertex.Position = WorldPosition(vertex.Position);
    SceneNode enclosure;
    enclosure.Id = scene.Nodes.size() + 1;
    enclosure.Name = room->Name;
    enclosure.Mesh = std::move(room);
    scene.Nodes.push_back(std::move(enclosure));

    auto emitterMaterial = DiffuseMaterial("Ceiling light", {0, 0, 0, 1});
    emitterMaterial.Emissive = {CeilingLightColor.R * CeilingLightRadiance,
                               CeilingLightColor.G * CeilingLightRadiance,
                               CeilingLightColor.B * CeilingLightRadiance, 1};
    emitterMaterial.SampledEmitter = true;
    scene.Materials.push_back(emitterMaterial);
    auto emitter = std::make_shared<MeshAsset>();
    emitter->Name = emitterMaterial.Name;
    Quad(*emitter, {{{-halfWidth, RoomHeight, -halfDepth}, {halfWidth, RoomHeight, -halfDepth},
                     {halfWidth, RoomHeight, halfDepth}, {-halfWidth, RoomHeight, halfDepth}}},
         {0, -1, 0}, emitterMaterial.Name.c_str());
    for (auto& vertex : emitter->Vertices) vertex.Position = WorldPosition(vertex.Position);
    SceneNode ceilingLight;
    ceilingLight.Id = scene.Nodes.size() + 1;
    ceilingLight.Name = emitter->Name;
    ceilingLight.Mesh = std::move(emitter);
    scene.Nodes.push_back(std::move(ceilingLight));
    // The downward light's tangent runs along Z, so its local size is depth/width.
    // Match mesh emission to the sampled rectangle without counting it twice.
    scene.Lights.push_back({LightType::Area, {0, RoomHeight, 0}, {0, -1, 0}, CeilingLightColor,
                           CeilingLightRadiance * RoomWidth * RoomDepth,
                           {RoomDepth, RoomWidth}, true});

    scene.SceneCamera.Position = WorldPosition(CameraPosition);
    scene.SceneCamera.Target = WorldPosition(CameraTarget);
    scene.SceneCamera.VerticalFieldOfViewDegrees = CameraVerticalFieldOfView;
    for (auto& light : scene.Lights)
    {
        light.Position = WorldPosition(light.Position);
        light.AreaSize.X *= WorldUnitsPerMeter;
        light.AreaSize.Y *= WorldUnitsPerMeter;
        // Area-light intensity is area-integrated; preserve emitted radiance.
        light.Intensity *= WorldUnitsPerMeter * WorldUnitsPerMeter;
    }
    return scene;
}
} // namespace Arvo::Editor
