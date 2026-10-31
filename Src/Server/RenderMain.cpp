#include "Distributed/Distributed.h"
#include "Core/Paths.h"
#include "Distributed/Protocol.h"
#include "Editor/Demos.hpp"
#include "Editor/ModuleAdapters.h"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace
{
std::uint32_t Integer(const std::string& text)
{
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos) throw std::runtime_error("Expected a nonnegative integer");
    const auto value = std::stoull(text);
    if (value > UINT32_MAX) throw std::runtime_error("Integer is too large");
    return static_cast<std::uint32_t>(value);
}

void Report(std::ostream& stream, const Arvo::Distributed::Progress& progress)
{
    stream << "state=" << static_cast<int>(progress.State) << "\ncompleted=" << progress.CompletedTiles
           << "\ntotal=" << progress.TotalTiles << "\nseconds=" << progress.ElapsedSeconds
           << "\noutput=" << progress.OutputPath.string() << "\nerror=" << progress.Error << '\n';
    for (std::size_t index = 0; index < progress.Servers.size(); ++index)
    {
        const auto& server = progress.Servers[index];
        stream << "server" << index << ".url=" << server.Url << '\n'
               << "server" << index << ".backend=" << server.Backend << '\n'
               << "server" << index << ".completed=" << server.CompletedTiles << '\n'
               << "server" << index << ".failures=" << server.FailedAttempts << '\n'
               << "server" << index << ".cpu_process_percent=" << server.CpuPercent << '\n'
               << "server" << index << ".gpu_device0_percent=" << server.GpuPercent << '\n'
               << "server" << index << ".status=" << server.Status << '\n';
    }
}
} // namespace

int main(int argc, char** argv)
{
    try
    {
        Arvo::Distributed::RenderConfig config;
        std::filesystem::path contentRoot = Arvo::Core::ContentRoot(), reportPath;
        config.OutputPath = Arvo::Core::BinaryRoot() / "Renders" / "render.png";
        std::uint32_t cancelAfter = 0;
        if (const char* token = std::getenv("ARVO_WORKER_TOKEN")) config.Token = token;
        for (int index = 1; index < argc; ++index)
        {
            const std::string option = argv[index];
            if (option == "--help")
            {
                std::cout << "ArvoRender --servers http://host:7831,http://host:7832 --scene cornell|sponza\n"
                             "           --backend cpu|cuda|compute|metal|auto --algorithm ray|path\n"
                             "           --width 3840 --height 2160 --tile 256 --samples 16 --bounces 4\n"
                             "           --output image.png [--report report.txt] [--token TOKEN]\n"
                             "           [--content-root Contents] [--cancel-after-ms N] [--retries 2]\n";
                return 0;
            }
            if (++index >= argc) throw std::runtime_error("Missing value for " + option);
            const std::string value = argv[index];
            if (option == "--servers")
            {
                config.Servers.clear();
                std::istringstream servers(value);
                std::string url;
                while (std::getline(servers, url, ',')) if (!url.empty()) config.Servers.push_back(url);
            }
            else if (option == "--scene") config.Scene = value;
            else if (option == "--backend") config.Backend = value;
            else if (option == "--algorithm")
            {
                if (value != "ray" && value != "path") throw std::runtime_error("Unknown algorithm: " + value);
                config.Integrator = value == "path" ? Arvo::RayTracing::Algorithm::PathTracing : Arvo::RayTracing::Algorithm::RayTracing;
            }
            else if (option == "--width") config.Width = Integer(value);
            else if (option == "--height") config.Height = Integer(value);
            else if (option == "--tile") config.TileSize = Integer(value);
            else if (option == "--samples") config.SamplesPerPixel = Integer(value);
            else if (option == "--bounces") config.MaxBounces = Integer(value);
            else if (option == "--retries") config.MaxRetries = Integer(value);
            else if (option == "--cancel-after-ms") cancelAfter = Integer(value);
            else if (option == "--output") config.OutputPath = value;
            else if (option == "--report") reportPath = value;
            else if (option == "--content-root") contentRoot = value;
            else if (option == "--token") config.Token = value;
            else throw std::runtime_error("Unknown argument: " + option);
        }
        if (config.Scene != "cornell" && config.Scene != "sponza") throw std::runtime_error("Select cornell or sponza");
        Arvo::Editor::DemoCatalog catalog(contentRoot);
        const auto scene = catalog.Load(config.Scene == "cornell" ?
            Arvo::Editor::DemoId::CornellBox : Arvo::Editor::DemoId::SponzaAtrium);
        config.Camera = Arvo::Editor::BuildOfflineCamera(scene);
        config.ExpectedSceneHash = Arvo::Distributed::Protocol::SceneHash(*Arvo::Editor::BuildOfflineScene(scene));
        Arvo::Distributed::RenderClient client;
        client.Start(config);
        bool cancelled = false;
        std::uint32_t previous = UINT32_MAX;
        while (client.IsRunning())
        {
            const auto progress = client.Snapshot();
            if (progress.CompletedTiles != previous)
            {
                std::cout << "Tiles " << progress.CompletedTiles << '/' << progress.TotalTiles << std::endl;
                previous = progress.CompletedTiles;
            }
            if (cancelAfter && !cancelled && progress.ElapsedSeconds * 1000 >= cancelAfter)
            { client.Cancel(); cancelled = true; }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        client.Wait();
        const auto progress = client.Snapshot();
        Report(std::cout, progress);
        if (!reportPath.empty())
        {
            if (!reportPath.parent_path().empty()) std::filesystem::create_directories(reportPath.parent_path());
            std::ofstream report(reportPath);
            if (!report) throw std::runtime_error("Cannot write render report");
            Report(report, progress);
        }
        if (progress.State == Arvo::Distributed::JobState::Completed) return 0;
        if (progress.State == Arvo::Distributed::JobState::Cancelled) return 2;
        return 1;
    }
    catch (const std::exception& error)
    {
        std::cerr << "ArvoRender: " << error.what() << std::endl;
        return 1;
    }
}
