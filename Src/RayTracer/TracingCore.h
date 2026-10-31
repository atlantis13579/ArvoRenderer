#pragma once

#include "Renderer/RayTracingBackendCommon.h"

#include <cstdint>
#include <functional>
#include <memory>

namespace Arvo
{
namespace RayTracing
{
namespace Detail
{
    using UnitFunction = std::function<void(uint32_t)>;

    // A backend-owned persistent pool. No worker is created while rendering a frame.
    class PersistentExecutor
    {
    public:
        explicit PersistentExecutor(uint32_t workerCount);
        ~PersistentExecutor();

        PersistentExecutor(const PersistentExecutor&) = delete;
        PersistentExecutor& operator=(const PersistentExecutor&) = delete;

        uint32_t GetWorkerCount() const;
        void Execute(
            uint32_t unitCount,
            uint32_t requestedWorkers,
            const UnitFunction& function,
            const CancellationProbe& cancellation);

    private:
        struct Impl;
        std::unique_ptr<Impl> m_Impl;
    };

    RenderResult RenderCpuTiles(
        const Scene& scene,
        const Camera& camera,
        const RenderSettings& settings,
        PersistentExecutor& executor,
        const CancellationProbe& cancellation);

    bool ValidateRenderInput(
        const Scene& scene,
        const Camera& camera,
        const RenderSettings& settings,
        std::string& error);
}
}
}
