#include "Renderer/DX11MicropolygonDicing.h"
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include "DX11MicropolygonSource.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <thread>

namespace Arvo::Renderer
{
using Microsoft::WRL::ComPtr;
namespace M = RayTracing::MicropolygonData;
struct DX11MicropolygonDicer::Impl
{
    ComPtr<ID3D11Device> Device;
    ComPtr<ID3D11DeviceContext> Context;
    ComPtr<ID3D11ComputeShader> Shader;
    bool Check(HRESULT status, const char* operation, std::string& error)
    {
        if (SUCCEEDED(status))
            return true;
        error = std::string(operation) + " failed (HRESULT " + std::to_string(static_cast<uint32_t>(status)) +
                ")";
        return false;
    }
    bool EnsureShader(std::string& error)
    {
        if (Shader)
            return true;
        ComPtr<ID3DBlob> bytecode, diagnostics;
        const auto status = D3DCompile(
            DX11MicropolygonSource, sizeof(DX11MicropolygonSource) - 1, "DX11MicropolygonDicing.hlsl",
            nullptr, nullptr, "DicePreview", "cs_5_0",
            D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_IEEE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3, 0,
            &bytecode, &diagnostics);
        if (FAILED(status))
        {
            error = diagnostics ? std::string(static_cast<const char*>(diagnostics->GetBufferPointer()),
                                              diagnostics->GetBufferSize())
                                : "DX11 micropolygon compilation failed";
            return false;
        }
        return Check(Device->CreateComputeShader(bytecode->GetBufferPointer(), bytecode->GetBufferSize(),
                                                 nullptr, &Shader),
                     "Creating DX11 dicing shader", error);
    }
    bool Wait(ID3D11Query* query, const RayTracing::CancellationToken& cancellation, std::string& error)
    {
        constexpr auto PollInterval = std::chrono::milliseconds(1),
                       BatchDeadline = std::chrono::milliseconds(30000);
        const auto deadline = std::chrono::steady_clock::now() + BatchDeadline;
        Context->End(query);
        Context->Flush();
        for (;;)
        {
            if (cancellation.IsCancellationRequested())
                return false;
            const auto status = Context->GetData(query, nullptr, 0, D3D11_ASYNC_GETDATA_DONOTFLUSH);
            if (status == S_OK)
                return true;
            if (FAILED(status))
                return Check(status, "Waiting for DX11 dicing", error);
            if (std::chrono::steady_clock::now() >= deadline)
            {
                error = "DX11 dicing batch did not retire before its deadline";
                return false;
            }
            std::this_thread::sleep_for(PollInterval);
        }
    }
    bool Pass(M::Controls controls, const std::vector<M::Patch>& patches, const std::vector<M::Range>& ranges,
              std::vector<uint32_t>& words, const RayTracing::CancellationToken& cancellation,
              std::string& error)
    {
        if (!EnsureShader(error))
            return false;
        constexpr uint32_t InputCount = 2;
        const M::Range empty{};
        const void* sources[]{patches.data(), ranges.empty() ? &empty : ranges.data()};
        const uint32_t counts[]{uint32_t(patches.size()), uint32_t(std::max(size_t(1), ranges.size()))};
        const uint32_t strides[]{sizeof(M::Patch), sizeof(M::Range)};
        ComPtr<ID3D11Buffer> inputs[InputCount], output, constants, readback;
        ComPtr<ID3D11ShaderResourceView> views[InputCount];
        ComPtr<ID3D11UnorderedAccessView> writable;
        for (uint32_t i = 0; i < InputCount; ++i)
        {
            D3D11_BUFFER_DESC desc{};
            desc.ByteWidth = counts[i] * strides[i];
            desc.Usage = D3D11_USAGE_IMMUTABLE;
            desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
            desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
            desc.StructureByteStride = strides[i];
            const D3D11_SUBRESOURCE_DATA data{sources[i], 0, 0};
            if (!Check(Device->CreateBuffer(&desc, &data, &inputs[i]), "Uploading DX11 dicing input", error))
                return false;
            D3D11_SHADER_RESOURCE_VIEW_DESC view{};
            view.Format = DXGI_FORMAT_UNKNOWN;
            view.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
            view.Buffer.NumElements = counts[i];
            if (!Check(Device->CreateShaderResourceView(inputs[i].Get(), &view, &views[i]),
                       "Creating DX11 dicing SRV", error))
                return false;
        }
        D3D11_BUFFER_DESC desc{};
        desc.ByteWidth = UINT(words.size() * sizeof(uint32_t));
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = D3D11_BIND_UNORDERED_ACCESS;
        desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        desc.StructureByteStride = sizeof(uint32_t);
        if (!Check(Device->CreateBuffer(&desc, nullptr, &output), "Allocating DX11 dicing output", error))
            return false;
        D3D11_UNORDERED_ACCESS_VIEW_DESC uav{};
        uav.Format = DXGI_FORMAT_UNKNOWN;
        uav.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
        uav.Buffer.NumElements = UINT(words.size());
        if (!Check(Device->CreateUnorderedAccessView(output.Get(), &uav, &writable),
                   "Creating DX11 dicing UAV", error))
            return false;
        desc.BindFlags = desc.MiscFlags = desc.StructureByteStride = 0;
        desc.Usage = D3D11_USAGE_STAGING;
        desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        if (!Check(Device->CreateBuffer(&desc, nullptr, &readback), "Allocating DX11 dicing readback", error))
            return false;
        desc = {};
        desc.ByteWidth = sizeof(controls);
        desc.Usage = D3D11_USAGE_DEFAULT;
        desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        if (!Check(Device->CreateBuffer(&desc, nullptr, &constants), "Allocating DX11 dicing constants",
                   error))
            return false;
        const D3D11_QUERY_DESC queryDescription{D3D11_QUERY_EVENT, 0};
        ComPtr<ID3D11Query> query;
        if (!Check(Device->CreateQuery(&queryDescription, &query), "Creating DX11 dicing fence", error))
            return false;
        struct Unbind
        {
            ID3D11DeviceContext* Context;
            ~Unbind()
            {
                ID3D11ShaderResourceView* views[InputCount]{};
                ID3D11UnorderedAccessView* uav = nullptr;
                ID3D11Buffer* constants = nullptr;
                Context->CSSetShaderResources(0, InputCount, views);
                Context->CSSetUnorderedAccessViews(0, 1, &uav, nullptr);
                Context->CSSetConstantBuffers(0, 1, &constants);
                Context->CSSetShader(nullptr, nullptr, 0);
            }
        } unbind{Context.Get()};
        const UINT clear[4]{};
        Context->ClearUnorderedAccessViewUint(writable.Get(), clear);
        ID3D11ShaderResourceView* resources[]{views[0].Get(), views[1].Get()};
        ID3D11UnorderedAccessView* target = writable.Get();
        ID3D11Buffer* packet = constants.Get();
        Context->CSSetShader(Shader.Get(), nullptr, 0);
        Context->CSSetShaderResources(0, InputCount, resources);
        Context->CSSetUnorderedAccessViews(0, 1, &target, nullptr);
        Context->CSSetConstantBuffers(0, 1, &packet);
        constexpr uint32_t BatchItems = 16384, Threads = 64;
        const uint32_t count = controls.Dispatch.Y;
        for (controls.Dispatch.X = 0; controls.Dispatch.X < count; controls.Dispatch.X += BatchItems)
        {
            if (cancellation.IsCancellationRequested())
                return false;
            controls.Dispatch.Y = std::min(BatchItems, count - controls.Dispatch.X);
            Context->UpdateSubresource(constants.Get(), 0, nullptr, &controls, 0, 0);
            Context->Dispatch((controls.Dispatch.Y + Threads - 1) / Threads, 1, 1);
            if (!Wait(query.Get(), cancellation, error))
                return false;
        }
        target = nullptr;
        Context->CSSetUnorderedAccessViews(0, 1, &target, nullptr);
        Context->CopyResource(readback.Get(), output.Get());
        if (!Wait(query.Get(), cancellation, error))
            return false;
        D3D11_MAPPED_SUBRESOURCE mapped{};
        if (!Check(Context->Map(readback.Get(), 0, D3D11_MAP_READ, 0, &mapped),
                   "Mapping DX11 dicing geometry", error))
            return false;
        std::memcpy(words.data(), mapped.pData, words.size() * sizeof(uint32_t));
        Context->Unmap(readback.Get(), 0);
        return !cancellation.IsCancellationRequested();
    }
};
DX11MicropolygonDicer::DX11MicropolygonDicer(ID3D11Device* device, ID3D11DeviceContext* context)
    : Impl_(std::make_unique<Impl>())
{
    Impl_->Device = device;
    Impl_->Context = context;
}
DX11MicropolygonDicer::~DX11MicropolygonDicer() = default;
RayTracing::MicropolygonResult DX11MicropolygonDicer::Prepare(const RayTracing::Scene& scene,
                                                              const RayTracing::Camera& camera,
                                                              const RayTracing::RenderSettings& settings,
                                                              const RayTracing::CancellationToken& token)
{
    return RayTracing::Detail::PrepareMicropolygons(
        scene, camera, settings, [&] { return token.IsCancellationRequested(); },
        [&](const auto& controls, const auto& patches, const auto& ranges, auto& words, auto& error) {
            return Impl_->Pass(controls, patches, ranges, words, token, error);
        });
}
} // namespace Arvo::Renderer
#endif
