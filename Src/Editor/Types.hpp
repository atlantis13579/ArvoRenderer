#pragma once

#include "Renderer/RayTracing.h"
#include "Renderer/TemporalReconstruction.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace Arvo::Editor
{

struct Float2
{
    float X = 0.0f;
    float Y = 0.0f;
};

struct Rect
{
    float X = 0.0f;
    float Y = 0.0f;
    float Width = 0.0f;
    float Height = 0.0f;

    [[nodiscard]] bool Contains(float x, float y) const noexcept
    {
        return x >= X && y >= Y && x < X + Width && y < Y + Height;
    }
};

struct Float3
{
    float X = 0.0f;
    float Y = 0.0f;
    float Z = 0.0f;
};

struct Color
{
    float R = 1.0f;
    float G = 1.0f;
    float B = 1.0f;
    float A = 1.0f;
};

struct Transform
{
    Float3 Translation;
    Float3 RotationDegrees;
    Float3 Scale{1.0f, 1.0f, 1.0f};
};

struct Vertex
{
    Float3 Position;
    Float3 Normal{0.0f, 1.0f, 0.0f};
    Float2 TexCoord;
    // RGB is authored in display-sRGB; alpha remains linear.
    Color VertexColor;
};

struct MeshSection
{
    std::string MaterialName;
    std::uint32_t FirstIndex = 0;
    std::uint32_t IndexCount = 0;
};

struct MeshAsset
{
    std::string Name;
    std::vector<Vertex> Vertices;
    std::vector<std::uint32_t> Indices;
    std::vector<MeshSection> Sections;
};

struct ImageAsset
{
    std::uint32_t Width = 0;
    std::uint32_t Height = 0;
    std::vector<std::uint8_t> PixelsRGBA8;

    [[nodiscard]] bool IsValid() const noexcept
    {
        return Width > 0 && Height > 0 &&
               PixelsRGBA8.size() == static_cast<std::size_t>(Width) * Height * 4U;
    }
};

struct Material
{
    std::string Name;
    RayTracing::MaterialScattering Scattering;
    Color Ambient{0.0f, 0.0f, 0.0f, 1.0f};
    Color Diffuse;
    Color Specular{0.0f, 0.0f, 0.0f, 1.0f};
    Color Emissive{0.0f, 0.0f, 0.0f, 1.0f};
    // An explicit area light samples this same emitting surface.
    bool SampledEmitter = false;
    float Shininess = 0.0f;
    float Opacity = 1.0f;
    std::string DiffuseTexture;
    std::string NormalTexture;
    std::string OpacityTexture;
    std::shared_ptr<const ImageAsset> DiffuseImage;
    std::shared_ptr<const ImageAsset> NormalImage;
    std::shared_ptr<const ImageAsset> OpacityImage;
    bool ScatteringEdited = false;
};

using SceneNodeId = std::uint64_t;

struct SceneNode
{
    SceneNodeId Id = 0;
    std::string Name;
    Transform LocalTransform;
    std::shared_ptr<const MeshAsset> Mesh;
    bool Visible = true;
    bool Selectable = true;
    RayTracing::MotionData::RigidMotion Motion;
};

struct Camera
{
    Float3 Position{0.0f, 1.5f, 5.0f};
    Float3 Target{0.0f, 0.5f, 0.0f};
    Float3 Up{0.0f, 1.0f, 0.0f};
    float VerticalFieldOfViewDegrees = 60.0f;
    RayTracing::CameraLens Lens;
    RayTracing::MotionData::Exposure Motion;
};

enum class LightType
{
    Directional,
    Point,
    Area
};

struct Light
{
    LightType Type = LightType::Directional;
    Float3 Position;
    Float3 Direction{-0.4f, -1.0f, -0.3f};
    Color LightColor;
    float Intensity = 1.0f;
    Float2 AreaSize{1.0f, 1.0f};
    bool CastShadows = true;
};

enum class DemoId
{
    CornellBox,
    SponzaAtrium,
    StanfordDragon,
    SponzaGallery
};

[[nodiscard]] constexpr bool IsSponza(DemoId demo)
{
    return demo == DemoId::SponzaAtrium || demo == DemoId::SponzaGallery;
}

struct SceneSnapshot
{
    DemoId Demo = DemoId::CornellBox;
    // Geometry/material changes invalidate resident scene resources.
    std::uint64_t Revision = 0;
    // Camera changes invalidate only rendered pixels, not scene resources.
    std::uint64_t CameraRevision = 0;
    Camera SceneCamera;
    std::vector<Light> Lights;
    std::vector<Material> Materials;
    std::vector<SceneNode> Nodes;
    // Linear radiance seen through openings and by rays that escape the scene.
    Float3 EnvironmentRadiance{0.02f, 0.02f, 0.025f};
};

enum class FillMode
{
    Solid,
    Wireframe,
    SolidWireframe
};

struct ViewportSettings
{
    std::uint32_t Width = 1280;
    std::uint32_t Height = 720;
    FillMode Mode = FillMode::Solid;
    bool ShadowsEnabled = true;
    bool OutlineEnabled = true;
    SceneNodeId SelectedNode = 0;
    RayTracing::Algorithm Integrator = RayTracing::Algorithm::RayTracing;
    RayTracing::RadiometryMode Radiometry = RayTracing::RadiometryMode::RGB;
    RayTracing::TransportOptions Transport;
    RayTracing::SampleSequence Sampling = RayTracing::SampleSequence::Independent;
    uint32_t SamplingSeed = 0;
    RayTracing::AdaptiveOptions Adaptive;
    RayTracing::MediumOptions Medium;
    RayTracing::MicropolygonOptions Micropolygons;
    RayTracing::ReconstructionOptions Reconstruction;
    bool TemporalPreview = false;
    RayTracing::TemporalOptions Temporal;
    RayTracing::MetropolisOptions Metropolis;
    RayTracing::PhotonMappingOptions Photons;
    RayTracing::ProgressivePhotonOptions ProgressivePhotons;
    RayTracing::VertexConnectionMergingOptions Vcm;
    RayTracing::PointGiOptions PointGi;
    RayTracing::AccelerationType Acceleration = RayTracing::AccelerationType::Bvh;
    RayTracing::BvhBuildPolicy BvhPolicy = RayTracing::BvhBuildPolicy::Default;
    RayTracing::GeometryMode Geometry = RayTracing::GeometryMode::Flat;
    std::uint32_t PreviewSamples = 1;
    std::uint32_t MaxBounces = 4;
};

struct Image
{
    std::uint32_t Width = 0;
    std::uint32_t Height = 0;
    // Top-left RGBA8 with display-sRGB RGB and straight linear alpha.
    std::vector<std::uint8_t> PixelsRGBA8;

    [[nodiscard]] bool IsValid() const noexcept
    {
        return Width > 0 && Height > 0 &&
               PixelsRGBA8.size() == static_cast<std::size_t>(Width) * Height * 4U;
    }
};

} // namespace Arvo::Editor
