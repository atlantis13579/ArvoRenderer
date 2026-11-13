#include "Renderer/SpectralSceneData.h"
#include <stdexcept>

namespace Arvo::RayTracing::MetalSpectralHost
{
RenderStatus BuildGpuInputs(const Scene& scene, const MediumOptions& medium,
                            const SpectralSceneData::Cancellation& cancellation,
                            SpectralSceneData::GpuScene& output, std::string& error)
{
    namespace S = SpectralSceneData;
    if (!ValidateMediumOptions(Algorithm::PathTracing, medium, error))
        return RenderStatus::Failed;
    S::Scene input;
    const auto status = BuildSceneInputs(scene, cancellation, input, error);
    if (status != RenderStatus::Completed)
        return status;
    try
    {
        constexpr uint64_t MaterialWidth = sizeof(S::Material) / sizeof(uint32_t);
        constexpr uint64_t TextureWidth = sizeof(S::Texture) / sizeof(uint32_t);
        constexpr uint64_t TriangleWidth = sizeof(S::Triangle) / sizeof(uint32_t);
        constexpr uint32_t MediumCount = 2, PollStride = 1024;
        const uint64_t count = S::HeaderWords + input.Materials.size() * MaterialWidth +
                               input.Textures.size() * TextureWidth + input.Triangles.size() * TriangleWidth +
                               input.Texels.size() + input.DirectionalLights.size() +
                               input.PointLights.size() + input.RectAreaLights.size();
        if (count > S::InvalidIndex || input.Requests.size() > S::InvalidIndex - MediumCount)
            throw std::overflow_error("Metal spectral atlas exceeds addressable uint32 offsets.");
        S::GpuScene prepared;
        prepared.SceneSourceCount = uint32_t(input.Requests.size());
        prepared.Sources = std::move(input.Requests);
        prepared.Sources.push_back({medium.Enabled ? medium.Absorption : Float3(), S::Role::Nonnegative});
        prepared.Sources.push_back({medium.Enabled ? medium.Scattering : Float3(), S::Role::Nonnegative});
        auto& atlas = prepared.References;
        atlas.assign(S::HeaderWords, 0);
        atlas[S::BackgroundTop] = input.BackgroundTop;
        atlas[S::BackgroundBottom] = input.BackgroundBottom;
        atlas[S::Absorption] = prepared.SceneSourceCount;
        atlas[S::Scattering] = prepared.SceneSourceCount + 1;
        atlas[S::MaterialTable] = uint32_t(atlas.size());
        for (const auto& record : input.Materials)
            atlas.insert(atlas.end(), {record.Albedo, record.Ambient, record.Specular, record.Emission,
                                       record.ConductorIndex, record.ConductorExtinction});
        atlas[S::LightTable] = uint32_t(atlas.size());
        for (const auto* table : {&input.DirectionalLights, &input.PointLights, &input.RectAreaLights})
            atlas.insert(atlas.end(), table->begin(), table->end());
        atlas[S::TextureTable] = uint32_t(atlas.size());
        for (const auto& record : input.Textures)
            atlas.insert(atlas.end(), {record.FirstTexel, record.Width, record.Height, record.Reserved});
        atlas[S::TexelTable] = uint32_t(atlas.size());
        atlas.insert(atlas.end(), input.Texels.begin(), input.Texels.end());
        for (size_t index = 0; index < input.Triangles.size(); ++index)
        {
            if (index % PollStride == 0 && cancellation && cancellation())
            {
                error = "Metal spectral atlas construction cancelled.";
                return RenderStatus::Cancelled;
            }
            const auto& triangle = input.Triangles[index];
            const Float3 first = prepared.Sources[triangle.Color[0]].Color;
            const Float3 second = prepared.Sources[triangle.Color[1]].Color;
            const Float3 third = prepared.Sources[triangle.Color[2]].Color;
            std::array<float, 9> identity{first.X,  first.Y, first.Z, second.X, second.Y,
                                          second.Z, third.X, third.Y, third.Z};
            if (prepared.TriangleReferences.count(identity) != 0)
                continue;
            prepared.TriangleReferences.emplace(identity, uint32_t(atlas.size()));
            atlas.insert(atlas.end(), std::begin(triangle.Color), std::end(triangle.Color));
        }
        if (cancellation && cancellation())
        {
            error = "Metal spectral atlas publication cancelled.";
            return RenderStatus::Cancelled;
        }
        output = std::move(prepared);
        error.clear();
        return RenderStatus::Completed;
    }
    catch (const std::exception& exception)
    {
        error = exception.what();
        return RenderStatus::Failed;
    }
}
} // namespace Arvo::RayTracing::MetalSpectralHost
