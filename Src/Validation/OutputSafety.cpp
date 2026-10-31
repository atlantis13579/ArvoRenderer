#include "Distributed/Distributed.h"
#include "Distributed/Protocol.h"

#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <thread>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace
{
using Bytes = std::vector<std::uint8_t>;
void Check(bool condition, const std::string& message)
{
    if (!condition) throw std::runtime_error(message);
}
Bytes Read(const std::filesystem::path& path)
{
    std::ifstream stream(path, std::ios::binary);
    Check(bool(stream), "Cannot read test output");
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}
void ExpectFailure(const std::filesystem::path& path, const Bytes& rgba)
{
    bool failed = false;
    try { Arvo::Distributed::WritePng(path, 32, 21, rgba); }
    catch (const std::exception&) { failed = true; }
    Check(failed, "Invalid output target was accepted");
}
}

int main(int argc, char** argv)
{
    try
    {
        const auto root = argc > 1 ? std::filesystem::path(argv[1])
            : std::filesystem::temp_directory_path() / ("ArvoOutputSafety-"
                + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(root);
        Bytes red(32 * 21 * 4, 0), blue = red;
        for (size_t i = 0; i < red.size(); i += 4)
        {
            red[i] = 255; red[i + 3] = 255;
            blue[i + 2] = 255; blue[i + 3] = 255;
        }
        const auto target = root / "result.png";
        Arvo::Distributed::WritePng(root / "red.png", 32, 21, red);
        Arvo::Distributed::WritePng(root / "blue.png", 32, 21, blue);
        const auto redPng = Read(root / "red.png"), bluePng = Read(root / "blue.png");
        const Bytes signature{137,80,78,71,13,10,26,10};
        Check(redPng.size() > 24 && std::equal(signature.begin(), signature.end(), redPng.begin()), "Invalid PNG signature");
        Check(redPng[19] == 32 && redPng[23] == 21, "Invalid PNG dimensions");
        Arvo::Distributed::WritePng(target, 32, 21, red);
        Arvo::Distributed::WritePng(target, 32, 21, blue);
        Check(Read(target) == bluePng, "Existing PNG was not replaced");
        unsigned cancellationChecks = 0;
        bool saveCancelled = false;
        try
        {
            Arvo::Distributed::WritePng(target, 32, 21, red, [&] { return ++cancellationChecks > 8; });
        }
        catch (const std::exception&) { saveCancelled = true; }
        Check(saveCancelled && Read(target) == bluePng, "Cancelled PNG save replaced the previous complete image");
        std::filesystem::create_directories(root / "directory.png");
        ExpectFailure(root / "directory.png", red);
        Check(std::filesystem::is_directory(root / "directory.png"), "A failed publish removed the target directory");
#ifdef _WIN32
        const HANDLE locked = CreateFileW(target.c_str(), GENERIC_READ, FILE_SHARE_READ,
            nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        Check(locked != INVALID_HANDLE_VALUE, "Cannot lock the replacement target");
        bool failed = false;
        try { Arvo::Distributed::WritePng(target, 32, 21, red); }
        catch (const std::exception&) { failed = true; }
        CloseHandle(locked);
        Check(failed && Read(target) == bluePng, "Failed replacement destroyed the old PNG");
#endif
        std::atomic<bool> start{false};
        std::atomic<unsigned> failures{0};
        std::vector<std::thread> writers;
        for (unsigned i = 0; i < 8; ++i)
        {
            writers.emplace_back([&, i] {
                while (!start.load()) std::this_thread::yield();
                try
                {
                    for (unsigned repeat = 0; repeat < 4; ++repeat)
                        Arvo::Distributed::WritePng(target, 32, 21, (i & 1) ? red : blue);
                }
                catch (...) { ++failures; }
            });
        }
        start.store(true);
        for (auto& thread : writers) thread.join();
        const auto concurrent = Read(target);
        Check(!failures.load(), "Concurrent PNG writers failed");
        Check(concurrent == redPng || concurrent == bluePng, "Concurrent publication produced a partial PNG");
        for (const auto& entry : std::filesystem::directory_iterator(root))
            Check(entry.path().filename().string().find(".partial-") == std::string::npos,
                "Temporary PNG leaked after success or failure");

        namespace Ray = Arvo::RayTracing;
        namespace Protocol = Arvo::Distributed::Protocol;
        Ray::Scene scene;
        scene.Materials.emplace_back(); scene.Triangles.emplace_back(); scene.Spheres.emplace_back();
        scene.DirectionalLights.emplace_back(); scene.PointLights.emplace_back(); scene.RectAreaLights.emplace_back();
        Ray::RayTexture texture;
        texture.Width = 2; texture.Height = 1;
        texture.PixelsRGBA8 = std::make_shared<const Bytes>(Bytes{1,2,3,255,4,5,6,255});
        scene.Textures.push_back(texture);
        const auto referenceHash = Protocol::SceneHash(scene);
        const auto changed = [&](auto mutation) {
            Ray::Scene copy = scene;
            mutation(copy);
            Check(Protocol::SceneHash(copy) != referenceHash, "Rendering change omitted from scene hash");
        };
        Check(Protocol::SceneHash(Ray::Scene(scene)) == referenceHash, "Scene hash depends on object addresses");
        changed([](auto& v) { v.Textures[0].Width = 1; v.Textures[0].Height = 2; });
        changed([](auto& v) { v.Textures[0].PixelsRGBA8.reset(); });
        changed([](auto& v) { v.Triangles[0].SmoothNormals = true; });
        changed([](auto& v) { v.Triangles[0].TexCoord2.X = 1; });
        changed([](auto& v) { v.Spheres[0].Radius = 2; });
        changed([](auto& v) { v.Spheres.clear(); });
        changed([](auto& v) { v.BackgroundTop.X = 0.5f; });
        changed([](auto& v) { v.BackgroundBottom.Y = 0.5f; });
        changed([](auto& v) { v.AmbientIntensity = 0.5f; });
        changed([](auto& v) { v.DirectionalLights[0].CastShadows = false; });
        changed([](auto& v) { v.PointLights[0].CastShadows = false; });
        changed([](auto& v) { v.RectAreaLights[0].CastShadows = false; });
        changed([](auto& v) { v.RectAreaLights[0].Size.X = 2; });
        changed([](auto& v) { v.Materials[0].Reflectivity = 0.5f; });
        changed([](auto& v) { v.Materials[0].SampledEmitter = true; });
        std::cout << "PNG replacement, failure preservation, concurrent writers, and scene hash validation passed\n";
        std::cout << "Artifacts: " << root.string() << '\n';
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Output safety validation failed: " << error.what() << '\n';
        return 1;
    }
}
