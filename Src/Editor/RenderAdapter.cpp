#include "Editor/RenderAdapter.hpp"
#include "Renderer/RayTracing.h"

#include <stdexcept>
#include <utility>

namespace Arvo::Editor
{

CancellationToken::CancellationToken(std::shared_ptr<std::atomic<bool>> state)
    : State_(std::move(state))
{
}

bool CancellationToken::IsCancellationRequested() const noexcept
{
    return State_ != nullptr && State_->load(std::memory_order_relaxed);
}

CancellationSource::CancellationSource()
    : State_(std::make_shared<std::atomic<bool>>(false))
{
}

CancellationToken CancellationSource::Token() const
{
    return CancellationToken(State_);
}

void CancellationSource::Cancel() const noexcept
{
    State_->store(true, std::memory_order_relaxed);
}

CallbackRenderAdapter::CallbackRenderAdapter(
    BackendDescriptor descriptor,
    RenderCallback callback)
    : Descriptor_(std::move(descriptor)), Callback_(std::move(callback))
{
    if (Descriptor_.Id.empty())
    {
        throw std::invalid_argument("A render adapter must have a stable identifier");
    }
    if (!Callback_)
    {
        throw std::invalid_argument("A callback render adapter needs a callback");
    }
}

BackendDescriptor CallbackRenderAdapter::Descriptor() const
{
    return Descriptor_;
}

RenderResult CallbackRenderAdapter::Render(
    const RenderRequest& request,
    const CancellationToken& cancellation)
{
    return Callback_(request, cancellation);
}

const std::vector<BackendDescriptor>& StandardBackendCatalog()
{
    static const std::vector<BackendDescriptor> backends = {
        {"renderer.dx11", "DX11 raster preview", RenderDomain::RealtimeRenderer, BackendPlatform::Windows,
         std::nullopt, BackendUsage::Preview},
        {"renderer.dxr", "DXR ray-traced preview", RenderDomain::RealtimeRenderer, BackendPlatform::Windows,
         RayTracing::BackendType::DXR, BackendUsage::PreviewAndOffline},
        {"integrator.cpu", "CPU offline", RenderDomain::RayTracing, BackendPlatform::Any,
         RayTracing::BackendType::CPU, BackendUsage::Offline},
        {"integrator.cuda", "CUDA offline", RenderDomain::RayTracing, BackendPlatform::Windows,
         RayTracing::BackendType::CUDA, BackendUsage::Offline},
        {"integrator.compute", "Compute offline", RenderDomain::RayTracing, BackendPlatform::Windows,
         RayTracing::BackendType::ComputeShader, BackendUsage::Offline},
        {"integrator.metal", "Metal offline", RenderDomain::RayTracing, BackendPlatform::Apple,
         RayTracing::BackendType::Metal, BackendUsage::Offline}};
    return backends;
}

HostPlatform CurrentHostPlatform() noexcept
{
#if defined(_WIN32)
    return HostPlatform::Windows;
#elif defined(__APPLE__)
    return HostPlatform::Apple;
#else
    return HostPlatform::Other;
#endif
}

bool IsBackendSupportedOnPlatform(
    const BackendDescriptor& descriptor,
    HostPlatform platform) noexcept
{
    return descriptor.Platform == BackendPlatform::Any ||
        (descriptor.Platform == BackendPlatform::Windows &&
            platform == HostPlatform::Windows) ||
        (descriptor.Platform == BackendPlatform::Apple &&
            platform == HostPlatform::Apple);
}

} // namespace Arvo::Editor
