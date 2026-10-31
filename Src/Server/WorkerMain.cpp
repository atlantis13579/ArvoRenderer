#include "Distributed/Distributed.h"
#include "Core/Paths.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv)
{
    try
    {
        Arvo::Distributed::WorkerConfig config;
        config.ContentRoot = Arvo::Core::ContentRoot();
        if (const char* token = std::getenv("ARVO_WORKER_TOKEN")) config.Token = token;
        for (int index = 1; index < argc; ++index)
        {
            const std::string option = argv[index];
            if (option == "--help")
            {
                std::cout << "ArvoWorker [--bind 127.0.0.1] [--port 7831] [--backend cpu|cuda|compute|metal]\n"
                             "           [--content-root Contents] [--token TOKEN]\n"
                             "ARVO_WORKER_TOKEN can supply the bearer token without a command-line argument.\n";
                return 0;
            }
            if (++index >= argc) throw std::runtime_error("Missing value for " + option);
            const std::string value = argv[index];
            if (option == "--port")
            {
                std::size_t used = 0;
                const int port = std::stoi(value, &used);
                if (used != value.size() || port < 1 || port > 65535) throw std::runtime_error("Invalid worker port");
                config.Port = static_cast<std::uint16_t>(port);
            }
            else if (option == "--bind") config.BindAddress = value;
            else if (option == "--backend") config.Backend = value;
            else if (option == "--content-root") config.ContentRoot = value;
            else if (option == "--token") config.Token = value;
            else throw std::runtime_error("Unknown argument: " + option);
        }
        return Arvo::Distributed::RunWorker(config);
    }
    catch (const std::exception& error)
    {
        std::cerr << "ArvoWorker: " << error.what() << std::endl;
        return 1;
    }
}
