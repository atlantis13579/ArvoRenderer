#include "RayTracer/RayTracer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace Arvo::RayTracing;

namespace
{
void Check(bool condition, const std::string& message)
{
    if (!condition) throw std::runtime_error(message);
}

void Quad(Scene& scene, Float3 a, Float3 b, Float3 c, Float3 d, uint32_t material)
{
    const Float3 normal = Normalize(Cross(b - a, c - a));
    for (const auto& vertices : {std::array<Float3, 3>{a, b, c}, std::array<Float3, 3>{a, c, d}})
    {
        Triangle triangle;
        triangle.Position0 = vertices[0];
        triangle.Position1 = vertices[1];
        triangle.Position2 = vertices[2];
        triangle.Normal0 = triangle.Normal1 = triangle.Normal2 = normal;
        triangle.MaterialIndex = material;
        scene.Triangles.push_back(triangle);
    }
}

Scene MakeScene()
{
    Scene scene;
    Material white;
    white.Albedo = Float3(0.65f, 0.65f, 0.65f);
    Material red = white;
    red.Albedo = Float3(0.75f, 0.1f, 0.06f);
    Material mirror = white;
    mirror.Reflectivity = 0.7f;
    scene.Materials = {white, red, mirror};
    scene.AmbientIntensity = 0.0f;
    scene.BackgroundTop = Float3(0.22f, 0.3f, 0.45f);
    scene.BackgroundBottom = Float3(0.08f, 0.1f, 0.15f);
    Quad(scene, {-2,-1,2}, {2,-1,2}, {2,-1,-2}, {-2,-1,-2}, 0);
    Quad(scene, {-2,-1,-2}, {2,-1,-2}, {2,2,-2}, {-2,2,-2}, 0);
    Quad(scene, {-2,-1,2}, {-2,-1,-2}, {-2,2,-2}, {-2,2,2}, 1);
    Quad(scene, {0,-1,-0.5f}, {1,-1,-0.5f}, {1,0.4f,-0.5f}, {0,0.4f,-0.5f}, 2);
    RectAreaLight light;
    light.Position = Float3(0,1.8f,0);
    light.Intensity = 2.0f;
    light.Size = Float2(0.8f, 0.8f);
    scene.RectAreaLights.push_back(light);
    return scene;
}

RenderResult Render(IRayTracingRHI& backend, const Scene& scene,
    const Camera& camera, const RenderSettings& settings)
{
    auto result = backend.Render(scene, camera, settings);
    Check(result.Status == RenderStatus::Completed, result.Error);
    Check(result.Output.GetWidth() == settings.Width && result.Output.GetHeight() == settings.Height,
        "Backend returned incorrect tile dimensions");
    Check(result.Statistics.CompletedPixels == uint64_t(settings.Width) * settings.Height,
        "Backend did not complete all pixels");
    return result;
}

void CheckTiles(IRayTracingRHI& backend, const Scene& scene, const Camera& camera,
    const RenderSettings& settings, const Image& reference)
{
    std::vector<uint8_t> assembled(reference.GetPixelsRGBA8().size(), 0);
    for (uint32_t y = 0; y < settings.Height; y += 19)
    {
        for (uint32_t x = 0; x < settings.Width; x += 23)
        {
            auto tile = settings;
            tile.FullWidth = settings.Width;
            tile.FullHeight = settings.Height;
            tile.TileX = x;
            tile.TileY = y;
            tile.Width = std::min(23u, settings.Width - x);
            tile.Height = std::min(19u, settings.Height - y);
            auto rendered = Render(backend, scene, camera, tile);
            for (uint32_t row = 0; row < tile.Height; ++row)
            {
                std::copy_n(rendered.Output.GetPixelsRGBA8().data() + row * tile.Width * 4,
                    tile.Width * 4, assembled.data() + ((y + row) * settings.Width + x) * 4);
            }
        }
    }
    Check(assembled == reference.GetPixelsRGBA8(), "Tiled image differs from full frame");
}

void CheckEmitterAccounting(IRayTracingRHI& backend)
{
    Scene scene;
    Material floor;
    floor.Albedo = Float3(0.5f, 0.5f, 0.5f);
    Material emitter;
    emitter.Albedo = emitter.AmbientColor = Float3();
    emitter.Emission = Float3(0.1f, 0.08f, 0.05f);
    emitter.SampledEmitter = true;
    scene.Materials = {floor, emitter};
    scene.AmbientIntensity = 0.0f;
    scene.BackgroundTop = scene.BackgroundBottom = Float3();
    Quad(scene, {-2,0,2}, {2,0,2}, {2,0,-2}, {-2,0,-2}, 0);
    Quad(scene, {-4,2,-4}, {4,2,-4}, {4,2,4}, {-4,2,4}, 1);
    RectAreaLight light;
    light.Position = Float3(0,2,0);
    light.Color = emitter.Emission;
    light.Size = Float2(8,8);
    light.Intensity = 64.0f;
    scene.RectAreaLights.push_back(light);
    Camera camera;
    camera.Position = Float3(0,1,0);
    camera.Target = Float3(0,0,0);
    camera.Up = Float3(0,0,-1);
    camera.VerticalFieldOfViewDegrees = 20;
    RenderSettings settings;
    settings.Width = 13; settings.Height = 9; settings.SamplesPerPixel = 16;
    settings.Integrator = Algorithm::PathTracing;
    settings.MaxReflectionBounces = 0;
    const auto direct = Render(backend, scene, camera, settings).Output;
    settings.MaxReflectionBounces = 1;
    const auto sampled = Render(backend, scene, camera, settings).Output;
    Check(direct.GetPixelsRGBA8() == sampled.GetPixelsRGBA8(),
        "An explicitly sampled emitter was counted twice after a diffuse bounce");
    scene.Materials[1].SampledEmitter = false;
    const auto unsampled = Render(backend, scene, camera, settings).Output;
    Check(sampled.GetPixelsRGBA8() != unsampled.GetPixelsRGBA8(),
        "Unregistered emissive geometry must still contribute along diffuse paths");
    scene.Materials[1].SampledEmitter = true;
    camera.Target = Float3(0,2,0);
    settings.MaxReflectionBounces = 0;
    const auto visible = Render(backend, scene, camera, settings).Output;
    Check(visible.GetPixelsRGBA8()[0] > 0, "Directly visible sampled emitter disappeared");
    camera.Target = Float3(0,0,0);
    scene.Materials[0].Reflectivity = 1.0f;
    settings.MaxReflectionBounces = 1;
    const auto reflected = Render(backend, scene, camera, settings).Output;
    Check(reflected.GetPixelsRGBA8() == visible.GetPixelsRGBA8(),
        "Sampled emitter disappeared after a specular bounce");
}

void CheckParity(const Image& reference, const Image& candidate)
{
    uint64_t total = 0;
    size_t large = 0;
    const auto& a = reference.GetPixelsRGBA8();
    const auto& b = candidate.GetPixelsRGBA8();
    Check(a.size() == b.size(), "Parity image dimensions differ");
    for (size_t i = 0; i < a.size(); ++i)
    {
        const int difference = std::abs(int(a[i]) - int(b[i]));
        total += difference;
        if (difference > 4) ++large;
    }
    const double mean = double(total) / a.size();
    std::cout << " mean_byte_error=" << mean << " large_components=" << large;
    Check(mean < 1.0 && large < a.size() / 100 + 1,
        "Backend image exceeds CPU parity tolerance");
}
}

int main()
{
    try
    {
        const Scene scene = MakeScene();
        Camera camera;
        camera.Position = Float3(0, 0.6f, 4.0f);
        camera.Target = Float3(0, 0.1f, -0.5f);
        RenderSettings settings;
        settings.Width = 61;
        settings.Height = 43;
        settings.SamplesPerPixel = 16;
        settings.MaxReflectionBounces = 3;
        auto cpu = CreateRayTracingRHI(BackendType::CPU);
        std::array<Image, 2> reference;
        for (unsigned algorithm = 0; algorithm < 2; ++algorithm)
        {
            settings.Integrator = static_cast<Algorithm>(algorithm);
            reference[algorithm] = Render(*cpu, scene, camera, settings).Output;
        }
        Check(reference[0].GetPixelsRGBA8() != reference[1].GetPixelsRGBA8(),
            "Path tracing must differ from direct ray tracing");
        // An unlit diffuse surface must receive indirect environment light.
        Scene indirect = scene;
        indirect.RectAreaLights.clear();
        settings.Integrator = Algorithm::RayTracing;
        const auto direct = Render(*cpu, indirect, camera, settings).Output;
        settings.Integrator = Algorithm::PathTracing;
        const auto bounce = Render(*cpu, indirect, camera, settings).Output;
        uint64_t directEnergy = 0, bounceEnergy = 0;
        for (size_t i = 0; i < direct.GetPixelsRGBA8().size(); i += 4)
        {
            directEnergy += direct.GetPixelsRGBA8()[i];
            bounceEnergy += bounce.GetPixelsRGBA8()[i];
        }
        Check(bounceEnergy > directEnergy + 10000, "Diffuse indirect transport is missing");
        for (const BackendType type : {BackendType::CPU, BackendType::CUDA,
            BackendType::ComputeShader, BackendType::DXR, BackendType::Metal})
        {
            auto backend = CreateRayTracingRHI(type);
            const auto info = backend->GetBackendInfo();
            if (!info.IsAvailable)
            {
                std::cout << "SKIP " << ToString(type) << ": " << info.Description << '\n';
                continue;
            }
            Check(info.IsNative, "Available backend must use its native implementation");
            CheckEmitterAccounting(*backend);
            for (unsigned algorithm = 0; algorithm < 2; ++algorithm)
            {
                settings.Integrator = static_cast<Algorithm>(algorithm);
                const auto result = Render(*backend, scene, camera, settings);
                std::cout << ToString(type) << (algorithm ? " PathTracing" : " RayTracing");
                CheckParity(reference[algorithm], result.Output);
                CheckTiles(*backend, scene, camera, settings, result.Output);
                auto invalid = settings;
                invalid.TileX = settings.Width;
                Check(backend->Render(scene, camera, invalid).Status == RenderStatus::Failed,
                    "Out-of-bounds tile was accepted");
                CancellationSource cancellation;
                cancellation.Cancel();
                Check(backend->Render(scene, camera, settings, cancellation.GetToken()).Status
                    == RenderStatus::Cancelled, "Pre-cancelled render was not cancelled");
                std::cout << " tiles=exact invalid_tile=rejected cancellation=passed\n";
            }
        }
        std::cout << "Integrator validation passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Integrator validation failed: " << error.what() << '\n';
        return 1;
    }
}
