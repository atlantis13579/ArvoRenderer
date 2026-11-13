#include "RayTracer/Backends/CPU/SpectralScene.h"
#include "RayTracer/TracingCore.h"
#include <stdexcept>

namespace Arvo::RayTracing::Cpu
{
RenderStatus PrepareSpectralScene(const Scene& source, const SpectralSceneData::Cancellation& cancel,
                                  PreparedSpectralScene& output, std::string& error,
                                  Detail::PersistentExecutor* executor)
{
    // A preparation transaction fits each palette entry once, then publishes
    // the immutable scene. Failed or cancelled fits never replace a usable cache.
    PreparedSpectralScene candidate;
    const auto status = BuildSpectralSceneInputs(source, cancel, candidate.Inputs, error);
    if (status != RenderStatus::Completed)
        return status;
    std::atomic<bool> cancelledOnce{false};
    const auto stopped = [&] {
        if (cancelledOnce.load(std::memory_order_relaxed))
            return true;
        if (cancel && cancel())
        {
            cancelledOnce.store(true, std::memory_order_relaxed);
            return true;
        }
        return false;
    };
    try
    {
        RgbFitTable response;
        for (unsigned i = 0; i < response.size(); ++i)
            response[i] = MakeRgbFitResponse(i);
        candidate.Spectra.resize(candidate.Inputs.Requests.size());
        // Independent fits write disjoint elements. The persistent backend pool
        // joins every fit before either publishing the cache or propagating failure.
        const auto fitEntry = [&](uint32_t index) {
            const auto fit = FitSceneSpectrum(candidate.Inputs.Requests[index], response);
            if (fit.Status != SpectrumFitConverged)
                throw std::runtime_error("CPU spectral palette fit failed at entry " + std::to_string(index));
            candidate.Spectra[index] = fit;
        };
        const uint32_t count = uint32_t(candidate.Spectra.size());
        if (executor)
            executor->Execute(count, executor->GetWorkerCount(), fitEntry, stopped);
        else
            for (uint32_t index = 0; index < count && !stopped(); ++index)
                fitEntry(index);
        if (stopped())
        {
            error = "CPU spectral coefficient preparation cancelled.";
            return RenderStatus::Cancelled;
        }
        uint32_t first = 0;
        const auto indexGeometry = [&](const SceneGeometry& geometry) {
            for (const auto& triangle : geometry.Triangles)
                candidate.TriangleIndices.emplace(&triangle, first++);
        };
        indexGeometry(source);
        for (const auto& geometry : source.Geometries)
            indexGeometry(*geometry);
        if (stopped())
        {
            error = "CPU spectral cache publication cancelled.";
            return RenderStatus::Cancelled;
        }
        output = std::move(candidate);
        error.clear();
        return RenderStatus::Completed;
    }
    catch (const std::exception& failure)
    {
        error = failure.what();
        return RenderStatus::Failed;
    }
}
} // namespace Arvo::RayTracing::Cpu
