#pragma once
#include "Renderer/RayTracing.h"
#include <functional>
#include <array>
#include <map>

namespace Arvo::RayTracing::SpectralSceneData
{
// Data/interfaces only. Each backend owns scene collection, spectral fitting
// and filtering. RGB fields with different physical roles never share an entry.
enum class Role : uint32_t
{
    Reflectance,
    Illuminant,
    Nonnegative
};
inline constexpr uint32_t InvalidIndex = std::numeric_limits<uint32_t>::max();
struct Request
{
    Float3 Color;
    Role Kind = Role::Reflectance;
};
struct Material
{
    uint32_t Albedo, Ambient, Specular, Emission, ConductorIndex, ConductorExtinction;
};
struct Triangle
{
    uint32_t Color[3];
};
struct Geometry
{
    uint32_t FirstTriangle, TriangleCount;
};
struct Texture
{
    uint32_t FirstTexel = 0, Width = 0, Height = 0, Reserved = 0;
};
struct Scene
{
    std::vector<Request> Requests;
    std::vector<Material> Materials;
    std::vector<Triangle> Triangles;
    // Slot zero is Scene's direct geometry; slot g+1 is Scene::Geometries[g].
    // Instances reference their original geometry; transforms do not alter color.
    std::vector<Geometry> Geometries;
    std::vector<Texture> Textures;
    std::vector<uint32_t> Texels;
    std::vector<uint32_t> DirectionalLights, PointLights, RectAreaLights;
    uint32_t BackgroundTop = InvalidIndex, BackgroundBottom = InvalidIndex;
    uint32_t WhiteReflectance = InvalidIndex;
};
using Cancellation = std::function<bool()>;
// A data-only native upload contract. Every backend independently packs these
// links and fits Sources on its own device. Primitive padding stores a triangle
// link offset, preserving integer bits and the existing geometry buffer stride.
enum BindingWord : uint32_t
{
    MaterialTable, LightTable, TextureTable, TexelTable,
    BackgroundTop, BackgroundBottom, Absorption, Scattering, HeaderWords
};
struct GpuScene
{
    std::vector<Request> Sources;
    std::vector<uint32_t> References;
    std::map<std::array<float, 9>, uint32_t> TriangleReferences;
    uint32_t SceneSourceCount = 0;
};
struct GpuSpectrum
{
    float Quadratic, Linear, Offset, Scale;
    float Constant, Error;
    uint32_t Status, Iterations;
};
static_assert(sizeof(GpuSpectrum) == 32);
static_assert(sizeof(Request) == 16 && sizeof(Texture) == 16);
} // namespace Arvo::RayTracing::SpectralSceneData

// Builders publish only complete snapshots; cancellation/failure preserves output.
// Ordinary geometry validation is separate. Invalid textures retain the existing
// white fallback, while nonfinite/out-of-domain spectral colors are rejected.
namespace Arvo::RayTracing::Cpu
{
RenderStatus BuildSpectralSceneInputs(const Scene&, const SpectralSceneData::Cancellation&,
                                      SpectralSceneData::Scene&, std::string&);
}
namespace Arvo::RayTracing::ComputeSpectralHost
{
RenderStatus BuildGpuInputs(const Scene&, const MediumOptions&, const SpectralSceneData::Cancellation&,
                            SpectralSceneData::GpuScene&, std::string&);
RenderStatus BuildSceneInputs(const Scene&, const SpectralSceneData::Cancellation&, SpectralSceneData::Scene&,
                              std::string&);
}
namespace Arvo::RayTracing::DxrSpectralHost
{
RenderStatus BuildGpuInputs(const Scene&, const MediumOptions&, const SpectralSceneData::Cancellation&,
                            SpectralSceneData::GpuScene&, std::string&);
RenderStatus BuildSceneInputs(const Scene&, const SpectralSceneData::Cancellation&, SpectralSceneData::Scene&,
                              std::string&);
}
namespace Arvo::RayTracing::CudaSpectralHost
{
RenderStatus BuildGpuInputs(const Scene&, const MediumOptions&, const SpectralSceneData::Cancellation&,
                            SpectralSceneData::GpuScene&, std::string&);
RenderStatus BuildSceneInputs(const Scene&, const SpectralSceneData::Cancellation&, SpectralSceneData::Scene&,
                              std::string&);
}
namespace Arvo::RayTracing::MetalSpectralHost
{
RenderStatus BuildGpuInputs(const Scene&, const MediumOptions&, const SpectralSceneData::Cancellation&,
                            SpectralSceneData::GpuScene&, std::string&);
RenderStatus BuildSceneInputs(const Scene&, const SpectralSceneData::Cancellation&, SpectralSceneData::Scene&,
                              std::string&);
}
