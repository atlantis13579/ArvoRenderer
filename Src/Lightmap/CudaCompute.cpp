#include "Lightmap/Compute.h"
#include <iostream>
#ifdef ARVO_LIGHTMAP_CUDA
#include "LightmapPtx.h"
#include <cuda.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#else
#include <dlfcn.h>
#endif
#endif

namespace Arvo::Lightmap
{
#ifdef ARVO_LIGHTMAP_CUDA
namespace
{
using namespace Transport;
struct Driver
{
#ifdef _WIN32
    HMODULE Library=LoadLibraryW(L"nvcuda.dll");
    void* Symbol(const char* name) { return reinterpret_cast<void*>(GetProcAddress(Library,name)); }
    ~Driver() { if(Library) FreeLibrary(Library); }
#else
    void* Library=dlopen("libcuda.so.1",RTLD_NOW|RTLD_LOCAL);
    void* Symbol(const char* name) { return dlsym(Library,name); }
    ~Driver() { if(Library) dlclose(Library); }
#endif
#define LM_CUDA_APIS(X) \
    X(cuInit) X(cuDeviceGetCount) X(cuDeviceGet) X(cuDeviceGetName) \
    X(cuDevicePrimaryCtxRetain) X(cuDevicePrimaryCtxRelease) X(cuCtxPushCurrent) X(cuCtxPopCurrent) \
    X(cuModuleLoadData) X(cuModuleGetFunction) X(cuModuleUnload) X(cuMemAlloc) X(cuMemFree) \
    X(cuMemcpyHtoD) X(cuMemcpyDtoH) X(cuMemsetD8) X(cuLaunchKernel) X(cuCtxSynchronize) \
    X(cuGetErrorName) X(cuGetErrorString)
#define LM_DECLARE(name) decltype(&name) name=nullptr;
    LM_CUDA_APIS(LM_DECLARE)
#undef LM_DECLARE
#define LM_STRING_IMPL(name) #name
#define LM_STRING(name) LM_STRING_IMPL(name)
    Driver()
    {
        if(!Library) throw std::runtime_error("CUDA driver is unavailable. Install an NVIDIA driver or explicitly select the CPU backend.");
        try
        {
#define LM_LOAD(name) name=reinterpret_cast<decltype(name)>(Symbol(LM_STRING(name))); if(!name) throw std::runtime_error("Missing CUDA driver entry: " LM_STRING(name));
            LM_CUDA_APIS(LM_LOAD)
#undef LM_LOAD
        }
        catch(...)
        {
#ifdef _WIN32
            FreeLibrary(Library);
#else
            dlclose(Library);
#endif
            Library=nullptr; throw;
        }
    }
#undef LM_STRING
#undef LM_STRING_IMPL
#undef LM_CUDA_APIS
    void Check(CUresult result,const char* operation)
    {
        if(result==CUDA_SUCCESS) return;
        const char* name=nullptr; const char* detail=nullptr;
        cuGetErrorName(result,&name); cuGetErrorString(result,&detail);
        throw std::runtime_error(std::string(operation)+": "+(name?name:"CUDA error")+" ("+(detail?detail:"unknown")+")");
    }
};
class Compute final:public CudaCompute
{
    Driver Api;
    CUdevice Device=0;
    CUcontext Context=nullptr;
    CUmodule Module=nullptr;
    CUfunction Kernel=nullptr;
    bool Pushed=false;
    Scene Gpu;
    std::vector<CUdeviceptr> Allocations;
    CUdeviceptr Tasks=0,Estimates=0;
    size_t Capacity=0;
    CUdeviceptr Allocate(size_t bytes)
    {
        CUdeviceptr result=0; if(!bytes) return result;
        Api.Check(Api.cuMemAlloc(&result,bytes),"CUDA allocate");
        Allocations.push_back(result); return result;
    }
    template<class T> Span<T> Upload(Span<T> source)
    {
        const auto bytes=size_t(source.Count)*sizeof(T); const auto ptr=Allocate(bytes);
        if(bytes) Api.Check(Api.cuMemcpyHtoD(ptr,source.Data,bytes),"CUDA upload");
        return {reinterpret_cast<const T*>(ptr),source.Count};
    }
    void Release() noexcept
    {
        if(Pushed)
        {
            for(auto p:Allocations) Api.cuMemFree(p);
            if(Module) Api.cuModuleUnload(Module);
            CUcontext previous=nullptr; Api.cuCtxPopCurrent(&previous); Pushed=false;
        }
        if(Context) { Api.cuDevicePrimaryCtxRelease(Device); Context=nullptr; }
    }
public:
    explicit Compute(const Scene& scene):Gpu(scene)
    {
        try
        {
            Api.Check(Api.cuInit(0),"CUDA initialization");
            int count=0; Api.Check(Api.cuDeviceGetCount(&count),"CUDA device enumeration");
            if(!count) throw std::runtime_error("No CUDA device is available; select CPU explicitly to bake without CUDA.");
            Api.Check(Api.cuDeviceGet(&Device,0),"CUDA select device");
            Api.Check(Api.cuDevicePrimaryCtxRetain(&Context,Device),"CUDA create compute context");
            Api.Check(Api.cuCtxPushCurrent(Context),"CUDA activate compute context"); Pushed=true;
            Api.Check(Api.cuModuleLoadData(&Module,LightmapPtx),"CUDA load lightmap kernel");
            Api.Check(Api.cuModuleGetFunction(&Kernel,Module,"BakeFootprints"),"CUDA find lightmap kernel");
            char name[256]={}; Api.Check(Api.cuDeviceGetName(name,sizeof(name),Device),"CUDA device name");
            std::cout<<"Offline CUDA compute: "<<name<<" (no graphics API)\n"<<std::flush;
            Gpu.Materials=Upload(scene.Materials); Gpu.Triangles=Upload(scene.Triangles);
            Gpu.EnvironmentFaces=Upload(scene.EnvironmentFaces); Gpu.Lights=Upload(scene.Lights);
            Gpu.Receivers=Upload(scene.Receivers); Gpu.Nodes=Upload(scene.Nodes); Gpu.Order=Upload(scene.Order);
            std::vector<Image> images;
            for(const auto& image:scene.Textures)
            {
                const auto size=uint64_t(image.Width)*image.Height*4;
                const auto ptr=Allocate(size_t(size));
                Api.Check(Api.cuMemcpyHtoD(ptr,image.Pixels,size_t(size)),"CUDA upload texture");
                images.push_back({image.Width,image.Height,reinterpret_cast<const uint8_t*>(ptr)});
            }
            Gpu.Textures=Upload(Span<Image>{images.data(),uint32_t(images.size())});
            Capacity=8192; Tasks=Allocate(Capacity*sizeof(Footprint)); Estimates=Allocate(Capacity*sizeof(Estimate));
        }
        catch(...) { Release(); throw; }
    }
    ~Compute() override { Release(); }
    void Run(const std::vector<Footprint>& tasks,std::vector<Estimate>& estimates) override
    {
        if(tasks.size()>Capacity || estimates.size()!=tasks.size()) throw std::runtime_error("Invalid CUDA lightmap batch");
        Api.Check(Api.cuMemcpyHtoD(Tasks,tasks.data(),tasks.size()*sizeof(Footprint)),"CUDA upload footprints");
        Api.Check(Api.cuMemsetD8(Estimates,0,tasks.size()*sizeof(Estimate)),"CUDA clear estimates");
        uint32_t count=uint32_t(tasks.size());
        // Bound each dispatch to keep cancellation responsive on display GPUs.
        for(uint32_t first=0;first<Gpu.Options.Samples;first+=32)
        {
            uint32_t end=std::min(first+32,Gpu.Options.Samples);
            void* arguments[]={&Gpu,&Tasks,&Estimates,&count,&first,&end};
            Api.Check(Api.cuLaunchKernel(Kernel,(count+63)/64,1,1,64,1,1,0,nullptr,arguments,nullptr),"CUDA trace lightmap paths");
            Api.Check(Api.cuCtxSynchronize(),"CUDA finish lightmap paths");
        }
        Api.Check(Api.cuMemcpyDtoH(estimates.data(),Estimates,tasks.size()*sizeof(Estimate)),"CUDA read lightmap estimates");
    }
};
}
#endif
std::unique_ptr<CudaCompute> CreateCudaCompute(const Transport::Scene& scene)
{
#ifdef ARVO_LIGHTMAP_CUDA
    return std::make_unique<Compute>(scene);
#else
    (void)scene;
    throw std::runtime_error("ArvoLightmap was built without CUDA. Rebuild with the CUDA Toolkit or explicitly select --backend cpu.");
#endif
}
}
