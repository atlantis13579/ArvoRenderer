#include "Editor/Editor.hpp"
#include "Editor/ModuleAdapters.h"
#include "RayTracer/RayTracer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <thread>

namespace Ray = Arvo::RayTracing;
namespace Editor = Arvo::Editor;

namespace
{
void Check(bool condition, const std::string& message)
{
    if (!condition) throw std::runtime_error(message);
}

void Compare(const std::vector<uint8_t>& reference, const std::vector<uint8_t>& candidate,
    bool exact, const std::string& label)
{
    Check(reference.size() == candidate.size(), label + ": image dimensions differ");
    uint64_t total = 0;
    size_t large = 0;
    unsigned maximum = 0;
    for (size_t i = 0; i < reference.size(); ++i)
    {
        const unsigned difference = static_cast<unsigned>(std::abs(int(reference[i]) - int(candidate[i])));
        total += difference;
        maximum = std::max(maximum, difference);
        if (difference > 4) ++large;
    }
    const double mean = double(total) / reference.size();
    std::cout << label << ": mean=" << mean << " max=" << maximum
              << " large_components=" << large << std::endl;
    Check(exact ? total == 0 : (mean < 1.0 && large < reference.size() / 100 + 1),
        label + ": image parity failed");
}

void WaitForPreview(Editor::EditorCore& editor, uint64_t previousRevision, Editor::DemoId scene)
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
    while (std::chrono::steady_clock::now() < deadline)
    {
        editor.Tick(0.0f);
        if (!editor.IsDemoLoading() && editor.Scene().Demo == scene
            && editor.ViewportBackend() == "renderer.dxr"
            && editor.ViewportRevision() > previousRevision
            && editor.ViewportImage().Width == 96 && editor.ViewportImage().Height == 54)
        {
            return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    throw std::runtime_error("DXR preview did not finish: " + editor.StatusText());
}
}

int main()
{
    try
    {
        Editor::DemoCatalog catalog(ARVO_CONTENT_ROOT);
        Editor::EditorCore editor(ARVO_CONTENT_ROOT);
        Editor::RegisterModuleAdapters(editor);
        editor.ResizeViewport(96, 54);
        editor.SetOutlineEnabled(false);
        editor.SetFillMode(Editor::FillMode::Solid);
        const auto previews = editor.Backends();
        const bool dxrAvailable = std::any_of(previews.begin(), previews.end(),
            [](const auto& backend) { return backend.Id == "renderer.dxr"; });
        if (dxrAvailable) editor.SelectBackend("renderer.dxr");
        else std::cout << "SKIP native DXR preview on this platform/device\n";
        for (const auto demo : {Editor::DemoId::CornellBox, Editor::DemoId::SponzaAtrium})
        {
            const std::string name = demo == Editor::DemoId::CornellBox ? "CornellBox" : "Sponza";
            const auto snapshot = catalog.Load(demo);
            const auto scene = Editor::BuildOfflineScene(snapshot);
            Check(scene && !scene->Triangles.empty(), "Scene conversion produced no geometry");
            if (demo == Editor::DemoId::CornellBox)
            {
                Check(scene->RectAreaLights.size() == 1, "Cornell must have one sampled emitter");
                const auto& light = scene->RectAreaLights.front();
                float area = 0.0f;
                Ray::Float3 center;
                size_t emitterTriangles = 0;
                for (const auto& triangle : scene->Triangles)
                {
                    const auto& material = scene->Materials.at(triangle.MaterialIndex);
                    if (!material.SampledEmitter) continue;
                    ++emitterTriangles;
                    const auto cross = Ray::Cross(triangle.Position1 - triangle.Position0,
                        triangle.Position2 - triangle.Position0);
                    const float triangleArea = std::sqrt(Ray::Dot(cross, cross)) * 0.5f;
                    area += triangleArea;
                    center = center + (triangle.Position0 + triangle.Position1 + triangle.Position2) * (triangleArea / 3.0f);
                    Check(Ray::Dot(Ray::Normalize(cross), light.Direction) > 0.9999f,
                        "Cornell sampled light direction differs from the emitter geometry");
                    Check(std::fabs(material.Emission.X - light.Color.X) < 1.0e-6f
                        && std::fabs(material.Emission.Y - light.Color.Y) < 1.0e-6f
                        && std::fabs(material.Emission.Z - light.Color.Z) < 1.0e-6f,
                        "Cornell sampled radiance differs from material Ke");
                }
                Check(emitterTriangles == 2 && area > 0, "Cornell emitter is not the expected rectangle");
                center = center / area;
                const auto offset = center - light.Position;
                Check(Ray::Dot(offset, offset) < 1.0e-10f
                    && std::fabs(area - light.Intensity) < 1.0e-5f
                    && std::fabs(area - light.Size.X * light.Size.Y) < 1.0e-5f,
                    "Cornell sampled light position or integrated radiance differs from its mesh");
            }

            const auto camera = Editor::BuildOfflineCamera(snapshot);
            std::cout << name << " triangles=" << scene->Triangles.size()
                      << " textures=" << scene->Textures.size() << std::endl;
            for (const auto algorithm : {Ray::Algorithm::RayTracing, Ray::Algorithm::PathTracing})
            {
                const std::string label = name + (algorithm == Ray::Algorithm::RayTracing ? " RT " : " PT ");
                Ray::RenderSettings settings;
                settings.Width = 96;
                settings.Height = 54;
                settings.SamplesPerPixel = 2;
                settings.MaxReflectionBounces = 3;
                settings.Integrator = algorithm;
                auto cpu = Ray::CreateRayTracingRHI(Ray::BackendType::CPU);
                auto reference = cpu->RenderAsync(scene, camera, settings).GetResult();
                Check(reference.Status == Ray::RenderStatus::Completed, reference.Error);
                std::vector<uint8_t> dxrPixels;
                for (const auto type : {Ray::BackendType::CUDA, Ray::BackendType::ComputeShader,
                    Ray::BackendType::DXR, Ray::BackendType::Metal})
                {
                    auto backend = Ray::CreateRayTracingRHI(type);
                    const auto info = backend->GetBackendInfo();
                    if (!info.IsAvailable)
                    {
                        std::cout << "SKIP " << label << Ray::ToString(type) << std::endl;
                        continue;
                    }
                    Check(info.IsNative, "Backend unexpectedly uses a fallback");
                    auto result = backend->RenderAsync(scene, camera, settings).GetResult();
                    Check(result.Status == Ray::RenderStatus::Completed, result.Error);
                    Compare(reference.Output.GetPixelsRGBA8(), result.Output.GetPixelsRGBA8(),
                        false, label + Ray::ToString(type));
                    if (type == Ray::BackendType::DXR) dxrPixels = result.Output.GetPixelsRGBA8();
                }
                if (dxrAvailable)
                {
                    const auto previous = editor.ViewportRevision();
                    editor.SetPreviewIntegrator(algorithm == Ray::Algorithm::PathTracing, 2, 3);
                    editor.SelectDemo(demo);
                    editor.RequestRender();
                    WaitForPreview(editor, previous, demo);
                    Check(!dxrPixels.empty(), "DXR preview exists without a native DXR backend");
                    Compare(dxrPixels, editor.ViewportImage().PixelsRGBA8, true, label + "actual editor DXR preview");
                }
            }
        }
        std::cout << "Scene and preview validation passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Scene validation failed: " << error.what() << '\n';
        return 1;
    }
}
