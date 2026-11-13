#include "Renderer/SpectralSceneData.h"
#include <stdexcept>

namespace Arvo::RayTracing::ComputeSpectralHost
{
RenderStatus BuildGpuInputs(const Scene& scene, const MediumOptions& medium,
                            const SpectralSceneData::Cancellation& cancel,
                            SpectralSceneData::GpuScene& output, std::string& error)
{
    namespace S = SpectralSceneData;
    S::Scene source;
    if (!ValidateMediumOptions(Algorithm::PathTracing, medium, error))
        return RenderStatus::Failed;
    const auto collected = BuildSceneInputs(scene, cancel, source, error);
    if (collected != RenderStatus::Completed)
        return collected;
    try
    {
        constexpr uint64_t MaterialWords = sizeof(S::Material) / sizeof(uint32_t);
        constexpr uint64_t TextureWords = sizeof(S::Texture) / sizeof(uint32_t);
        constexpr uint64_t VertexWords = sizeof(S::Triangle) / sizeof(uint32_t);
        constexpr uint32_t MediumSources = 2, PollPeriod = 1024;
        const uint64_t words = S::HeaderWords + source.Materials.size() * MaterialWords +
                               source.Textures.size() * TextureWords + source.Texels.size() +
                               source.DirectionalLights.size() + source.PointLights.size() +
                               source.RectAreaLights.size() + source.Triangles.size() * VertexWords;
        if (words > S::InvalidIndex || source.Requests.size() > S::InvalidIndex - MediumSources)
            throw std::overflow_error("Compute spectral bindings exceed 32-bit indexing.");
        S::GpuScene candidate;
        candidate.SceneSourceCount = uint32_t(source.Requests.size());
        candidate.Sources = std::move(source.Requests);
        candidate.Sources.push_back({medium.Enabled ? medium.Absorption : Float3(), S::Role::Nonnegative});
        candidate.Sources.push_back({medium.Enabled ? medium.Scattering : Float3(), S::Role::Nonnegative});
        auto& links = candidate.References;
        links.resize(S::HeaderWords);
        links[S::BackgroundTop] = source.BackgroundTop;
        links[S::BackgroundBottom] = source.BackgroundBottom;
        links[S::Absorption] = candidate.SceneSourceCount;
        links[S::Scattering] = candidate.SceneSourceCount + 1;
        links[S::MaterialTable] = uint32_t(links.size());
        for (const auto& material : source.Materials)
            links.insert(links.end(),
                         {material.Albedo, material.Ambient, material.Specular, material.Emission,
                          material.ConductorIndex, material.ConductorExtinction});
        links[S::LightTable] = uint32_t(links.size());
        links.insert(links.end(), source.DirectionalLights.begin(), source.DirectionalLights.end());
        links.insert(links.end(), source.PointLights.begin(), source.PointLights.end());
        links.insert(links.end(), source.RectAreaLights.begin(), source.RectAreaLights.end());
        links[S::TextureTable] = uint32_t(links.size());
        for (const auto& texture : source.Textures)
            links.insert(links.end(), {texture.FirstTexel, texture.Width, texture.Height, 0});
        links[S::TexelTable] = uint32_t(links.size());
        links.insert(links.end(), source.Texels.begin(), source.Texels.end());
        uint32_t processed = 0;
        for (const auto& triangle : source.Triangles)
        {
            if (processed++ % PollPeriod == 0 && cancel && cancel())
            {
                error = "Compute spectral binding preparation cancelled.";
                return RenderStatus::Cancelled;
            }
            const Float3 a = candidate.Sources[triangle.Color[0]].Color;
            const Float3 b = candidate.Sources[triangle.Color[1]].Color;
            const Float3 c = candidate.Sources[triangle.Color[2]].Color;
            const std::array<float, 9> key{a.X, a.Y, a.Z, b.X, b.Y, b.Z, c.X, c.Y, c.Z};
            if (candidate.TriangleReferences.emplace(key, uint32_t(links.size())).second)
                links.insert(links.end(), std::begin(triangle.Color), std::end(triangle.Color));
        }
        if (cancel && cancel())
        {
            error = "Compute spectral binding publication cancelled.";
            return RenderStatus::Cancelled;
        }
        output = std::move(candidate);
        error.clear();
        return RenderStatus::Completed;
    }
    catch (const std::exception& failure)
    {
        error = failure.what();
        return RenderStatus::Failed;
    }
}
} // namespace Arvo::RayTracing::ComputeSpectralHost
