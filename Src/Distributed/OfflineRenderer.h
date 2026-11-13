#pragma once
#include "Distributed/Distributed.h"
#include "Renderer/TemporalSession.h"

namespace Arvo::Distributed
{
inline constexpr uint32_t MaximumTemporalBatchFrames = 256;
struct TemporalRenderOptions
{
    bool Enabled = false;
    // Each fresh frame receives the complete configured transport sample budget.
    uint32_t FramesPerRender = 4;
    RayTracing::TemporalOptions Filter;
};
void ValidateTemporalRendering(const RenderConfig&, const TemporalRenderOptions&);

// Offline PNG transaction: assemble fresh films, reconstruct on an actual native
// producer, then atomically save. A failed batch cannot advance accepted history.
// This schedules native implementations; it contains no denoising arithmetic.
class OfflineRenderClient
{
public:
    OfflineRenderClient();
    ~OfflineRenderClient();
    OfflineRenderClient(const OfflineRenderClient&) = delete;
    OfflineRenderClient& operator=(const OfflineRenderClient&) = delete;
    void Start(RenderConfig, TemporalRenderOptions = {});
    void Cancel();
    void Wait();
    bool IsRunning() const;
    Progress Snapshot() const;
    // Idle only. Resetting history does not delete any published PNG.
    void ResetHistory();
    std::shared_ptr<const RayTracing::TemporalHistory> History() const;

private:
    struct Impl;
    std::unique_ptr<Impl> Impl_;
};
} // namespace Arvo::Distributed
