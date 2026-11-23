#include "Lightmap/Baker.h"
#include <iostream>

int RunLightmap(int argc,char** argv)
{
    try
    {
        if(argc==2 && std::string(argv[1])=="--help")
        { std::cout<<"ArvoEditor --bake-lightmap scene.arvolm result.arvolm [--backend cuda|cpu] (default: cuda)\n"; return 0; }
        if((argc!=4 && argc!=6) || std::string(argv[1])!="--bake-lightmap")
        { std::cerr<<"Usage: ArvoEditor --bake-lightmap scene.arvolm result.arvolm [--backend cuda|cpu] (default: cuda)\n"; return 2; }
        auto backend=Arvo::Lightmap::Backend::CUDA;
        if(argc==6)
        {
            if(std::string(argv[4])!="--backend" || (std::string(argv[5])!="cuda" && std::string(argv[5])!="cpu"))
                throw std::runtime_error("Expected --backend cuda|cpu");
            if(std::string(argv[5])=="cpu") backend=Arvo::Lightmap::Backend::CPU;
        }
        std::cout<<"Offline backend: "<<(backend==Arvo::Lightmap::Backend::CUDA?"CUDA":"CPU")<<'\n'<<std::flush;
        ArvoLightmap::Job job; ArvoLightmap::TransferJob(std::filesystem::u8path(argv[2]),job,true);
        std::cout<<"Arvo diffuse lightmap: "<<job.Options.Resolution<<" pixels, "<<job.Options.Samples<<" samples, "<<job.Options.Bounces<<" bounces\n"<<std::flush;
        auto result=Arvo::Lightmap::Bake(job,[](auto completed,auto total) {
            if(completed==total || completed%16==0) std::cout<<"Charts "<<completed<<'/'<<total<<'\n'<<std::flush;
        },backend);
        ArvoLightmap::TransferResult(std::filesystem::u8path(argv[3]),result,false);
        std::cout<<"Lightmap complete\n"; return 0;
    }
    catch(const std::exception& error) { std::cerr<<"ArvoLightmap: "<<error.what()<<'\n'; return 1; }
}
