#include "Renderer/SpectralSceneData.h"
#include <stdexcept>

namespace Arvo::RayTracing::DxrSpectralHost
{
RenderStatus BuildGpuInputs(const Scene& scene, const MediumOptions& medium,
                            const SpectralSceneData::Cancellation& cancelled,
                            SpectralSceneData::GpuScene& output, std::string& error)
{
    namespace S = SpectralSceneData;
    if (!ValidateMediumOptions(Algorithm::PathTracing, medium, error))
        return RenderStatus::Failed;
    S::Scene inputs;
    const RenderStatus status = BuildSceneInputs(scene, cancelled, inputs, error);
    if (status != RenderStatus::Completed)
        return status;
    try
    {
        constexpr uint64_t MaterialFields = sizeof(S::Material) / sizeof(uint32_t);
        constexpr uint64_t TextureFields = sizeof(S::Texture) / sizeof(uint32_t);
        constexpr uint64_t TriangleFields = sizeof(S::Triangle) / sizeof(uint32_t);
        constexpr uint32_t VolumeEntries = 2, CancellationInterval = 1024;
        const uint64_t bound = S::HeaderWords + MaterialFields * inputs.Materials.size() +
                               TextureFields * inputs.Textures.size() +
                               TriangleFields * inputs.Triangles.size() + inputs.Texels.size() +
                               inputs.DirectionalLights.size() + inputs.PointLights.size() +
                               inputs.RectAreaLights.size();
        if (bound > S::InvalidIndex || inputs.Requests.size() > S::InvalidIndex - VolumeEntries)
            throw std::overflow_error("DXR spectral descriptor offsets exceed uint32.");
        S::GpuScene packed;
        packed.SceneSourceCount = uint32_t(inputs.Requests.size());
        packed.Sources = std::move(inputs.Requests);
        packed.Sources.push_back({medium.Enabled ? medium.Absorption : Float3(), S::Role::Nonnegative});
        packed.Sources.push_back({medium.Enabled ? medium.Scattering : Float3(), S::Role::Nonnegative});
        packed.References.resize(S::HeaderWords);
        auto& words = packed.References;
        words[S::BackgroundTop] = inputs.BackgroundTop;
        words[S::BackgroundBottom] = inputs.BackgroundBottom;
        words[S::Absorption] = packed.SceneSourceCount;
        words[S::Scattering] = packed.SceneSourceCount + 1;
        words[S::MaterialTable] = uint32_t(words.size());
        for (const S::Material& m : inputs.Materials)
        {
            const uint32_t fields[]{m.Albedo,   m.Ambient,        m.Specular,
                                    m.Emission, m.ConductorIndex, m.ConductorExtinction};
            words.insert(words.end(), std::begin(fields), std::end(fields));
        }
        words[S::LightTable] = uint32_t(words.size());
        for (const auto* lights : {&inputs.DirectionalLights, &inputs.PointLights, &inputs.RectAreaLights})
            words.insert(words.end(), lights->begin(), lights->end());
        words[S::TextureTable] = uint32_t(words.size());
        for (const S::Texture& texture : inputs.Textures)
        {
            words.push_back(texture.FirstTexel);
            words.push_back(texture.Width);
            words.push_back(texture.Height);
            words.push_back(0);
        }
        words[S::TexelTable] = uint32_t(words.size());
        words.insert(words.end(), inputs.Texels.begin(), inputs.Texels.end());
        for (size_t index = 0; index < inputs.Triangles.size(); ++index)
        {
            if (index % CancellationInterval == 0 && cancelled && cancelled())
            {
                error = "DXR spectral link construction cancelled.";
                return RenderStatus::Cancelled;
            }
            const auto& triangle = inputs.Triangles[index];
            std::array<float, 9> key{};
            constexpr unsigned Components = 3;
            for (unsigned vertex = 0; vertex < Components; ++vertex)
            {
                const Float3 color = packed.Sources[triangle.Color[vertex]].Color;
                key[vertex * Components] = color.X;
                key[vertex * Components + 1] = color.Y;
                key[vertex * Components + 2] = color.Z;
            }
            if (packed.TriangleReferences.find(key) != packed.TriangleReferences.end())
                continue;
            packed.TriangleReferences.emplace(key, uint32_t(words.size()));
            for (uint32_t color : triangle.Color)
                words.push_back(color);
        }
        if (cancelled && cancelled())
        {
            error = "DXR spectral link publication cancelled.";
            return RenderStatus::Cancelled;
        }
        output = std::move(packed);
        error.clear();
        return RenderStatus::Completed;
    }
    catch (const std::exception& exception)
    {
        error = exception.what();
        return RenderStatus::Failed;
    }
}
} // namespace Arvo::RayTracing::DxrSpectralHost
