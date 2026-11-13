#include "Renderer/SpectralSceneData.h"
#include <stdexcept>

namespace Arvo::RayTracing::CudaSpectralHost
{
RenderStatus BuildGpuInputs(const Scene& scene, const MediumOptions& medium,
                            const SpectralSceneData::Cancellation& cancel,
                            SpectralSceneData::GpuScene& destination, std::string& error)
{
    namespace S = SpectralSceneData;
    if (!ValidateMediumOptions(Algorithm::PathTracing, medium, error))
        return RenderStatus::Failed;
    S::Scene collected;
    RenderStatus status = BuildSceneInputs(scene, cancel, collected, error);
    if (status != RenderStatus::Completed)
        return status;
    try
    {
        constexpr size_t MediumSlots = 2, CancellationBlock = 1024;
        constexpr size_t MaterialSlots = sizeof(S::Material) / sizeof(uint32_t);
        constexpr size_t TextureSlots = sizeof(S::Texture) / sizeof(uint32_t);
        constexpr size_t VertexSlots = sizeof(S::Triangle) / sizeof(uint32_t);
        const uint64_t upper = S::HeaderWords + uint64_t(collected.Materials.size()) * MaterialSlots +
                               uint64_t(collected.Textures.size()) * TextureSlots +
                               uint64_t(collected.Triangles.size()) * VertexSlots +
                               collected.DirectionalLights.size() + collected.PointLights.size() +
                               collected.RectAreaLights.size() + collected.Texels.size();
        if (upper > S::InvalidIndex || collected.Requests.size() > S::InvalidIndex - MediumSlots)
            throw std::overflow_error("CUDA spectral upload cannot use 32-bit links.");
        S::GpuScene candidate;
        candidate.Sources.swap(collected.Requests);
        candidate.SceneSourceCount = uint32_t(candidate.Sources.size());
        const Float3 absorption = medium.Enabled ? medium.Absorption : Float3();
        const Float3 scattering = medium.Enabled ? medium.Scattering : Float3();
        candidate.Sources.push_back({absorption, S::Role::Nonnegative});
        candidate.Sources.push_back({scattering, S::Role::Nonnegative});
        auto& refs = candidate.References;
        refs.resize(S::HeaderWords);
        refs[S::BackgroundTop] = collected.BackgroundTop;
        refs[S::BackgroundBottom] = collected.BackgroundBottom;
        refs[S::Absorption] = candidate.SceneSourceCount;
        refs[S::Scattering] = candidate.SceneSourceCount + 1;
        refs[S::MaterialTable] = uint32_t(refs.size());
        for (const auto& material : collected.Materials)
        {
            refs.push_back(material.Albedo);
            refs.push_back(material.Ambient);
            refs.push_back(material.Specular);
            refs.push_back(material.Emission);
            refs.push_back(material.ConductorIndex);
            refs.push_back(material.ConductorExtinction);
        }
        refs[S::LightTable] = uint32_t(refs.size());
        for (auto color : collected.DirectionalLights)
            refs.push_back(color);
        for (auto color : collected.PointLights)
            refs.push_back(color);
        for (auto color : collected.RectAreaLights)
            refs.push_back(color);
        refs[S::TextureTable] = uint32_t(refs.size());
        for (const auto& texture : collected.Textures)
            refs.insert(refs.end(), {texture.FirstTexel, texture.Width, texture.Height, 0});
        refs[S::TexelTable] = uint32_t(refs.size());
        refs.insert(refs.end(), collected.Texels.begin(), collected.Texels.end());
        size_t triangleIndex = 0;
        for (const auto& triangle : collected.Triangles)
        {
            if (triangleIndex++ % CancellationBlock == 0 && cancel && cancel())
            {
                error = "CUDA spectral references cancelled.";
                return RenderStatus::Cancelled;
            }
            const auto a = candidate.Sources[triangle.Color[0]].Color;
            const auto b = candidate.Sources[triangle.Color[1]].Color;
            const auto c = candidate.Sources[triangle.Color[2]].Color;
            const std::array<float, 9> tuple{a.X, a.Y, a.Z, b.X, b.Y, b.Z, c.X, c.Y, c.Z};
            const auto inserted = candidate.TriangleReferences.emplace(tuple, uint32_t(refs.size()));
            if (!inserted.second)
                continue;
            for (uint32_t entry : triangle.Color)
                refs.push_back(entry);
        }
        if (cancel && cancel())
        {
            error = "CUDA spectral reference publication cancelled.";
            return RenderStatus::Cancelled;
        }
        destination = std::move(candidate);
        error.clear();
        return RenderStatus::Completed;
    }
    catch (const std::exception& exception)
    {
        error = exception.what();
        return RenderStatus::Failed;
    }
}
} // namespace Arvo::RayTracing::CudaSpectralHost
