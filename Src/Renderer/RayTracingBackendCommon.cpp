#include "Renderer/RayTracingBackendCommon.h"

#include "Maths/ColorSpace.h"
#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <exception>
#include <future>
#include <limits>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>

namespace Arvo
{
namespace RayTracing
{
namespace Detail
{
    // Texture filtering is performed in linear light; environment colors are linear.
    static float RepeatCoordinate(float value) noexcept
    {
        if (!std::isfinite(value))
        {
            return 0.0f;
        }
        return value - std::floor(value);
    }

    static uint32_t WrapTexel(int64_t coordinate, uint32_t size) noexcept
    {
        const int64_t signedSize = static_cast<int64_t>(size);
        int64_t wrapped = coordinate % signedSize;
        if (wrapped < 0)
        {
            wrapped += signedSize;
        }
        return static_cast<uint32_t>(wrapped);
    }

    Float3 SampleDiffuseTexture(
        const Scene& scene,
        uint32_t textureIndex,
        const Float2& texCoord) noexcept
    {
        if (textureIndex >= scene.Textures.size() ||
            !scene.Textures[textureIndex].IsValid())
        {
            return Float3(1.0f, 1.0f, 1.0f);
        }

        const RayTexture& texture = scene.Textures[textureIndex];
        const float x = RepeatCoordinate(texCoord.X) *
                            static_cast<float>(texture.Width) -
                        0.5f;
        const float y = RepeatCoordinate(1.0f - texCoord.Y) *
                            static_cast<float>(texture.Height) -
                        0.5f;
        const float floorX = std::floor(x);
        const float floorY = std::floor(y);
        const int64_t x0 = static_cast<int64_t>(floorX);
        const int64_t y0 = static_cast<int64_t>(floorY);
        const uint32_t sampleX[2] = {
            WrapTexel(x0, texture.Width),
            WrapTexel(x0 + 1, texture.Width)};
        const uint32_t sampleY[2] = {
            WrapTexel(y0, texture.Height),
            WrapTexel(y0 + 1, texture.Height)};
        const float blendX = x - floorX;
        const float blendY = y - floorY;
        const std::vector<uint8_t>& pixels = *texture.PixelsRGBA8;
        const auto load = [&](uint32_t pixelX, uint32_t pixelY)
        {
            const size_t byte =
                (static_cast<size_t>(pixelY) * texture.Width + pixelX) * 4U;
            return Float3(
                Maths::SRGBToLinear(
                    static_cast<float>(pixels[byte]) / 255.0f),
                Maths::SRGBToLinear(
                    static_cast<float>(pixels[byte + 1U]) / 255.0f),
                Maths::SRGBToLinear(
                    static_cast<float>(pixels[byte + 2U]) / 255.0f));
        };
        const auto lerp = [](
            const Float3& left,
            const Float3& right,
            float amount)
        {
            return left * (1.0f - amount) + right * amount;
        };
        const Float3 top = lerp(
            load(sampleX[0], sampleY[0]),
            load(sampleX[1], sampleY[0]),
            blendX);
        const Float3 bottom = lerp(
            load(sampleX[0], sampleY[1]),
            load(sampleX[1], sampleY[1]),
            blendX);
        return lerp(top, bottom, blendY);
    }

    bool ValidateMaterialTransport(const Scene& scene, BackendType backend,
        Algorithm algorithm, std::string& error)
    {
        const auto inRange = [](float value, float minimum, float maximum)
        { return std::isfinite(value) && value >= minimum && value <= maximum; };
        const auto vectorInRange = [&](Float3 value, float minimum, float maximum)
        { return inRange(value.X, minimum, maximum) && inRange(value.Y, minimum, maximum)
            && inRange(value.Z, minimum, maximum); };
        for (const Material& material : scene.Materials)
        {
            if (algorithm == Algorithm::PointBasedGlobalIllumination && !material.SampledEmitter &&
                material.Model == MaterialModel::Legacy &&
                (material.Reflectivity != 0 || material.SpecularColor.X != 0 || material.SpecularColor.Y != 0 || material.SpecularColor.Z != 0))
            {
                error = "Point GI requires diffuse scattering; select Diffuse for authored highlight/mirror materials.";
                return false;
            }
            if (material.Model >= MaterialModel::Count)
            {
                error = "Unknown material scattering model.";
                return false;
            }
            if (material.Model == MaterialModel::Legacy) continue;
            if (!ValidateMaterialSupport(material, backend, algorithm, error)) return false;
            if (!vectorInRange(material.Albedo, 0, 1))
            {
                error = "Physical materials require albedo in [0,1].";
                return false;
            }
        }
        return true;
    }

    bool BuildRayTextureAtlas(
        const Scene& scene,
        PackedRayTextureAtlas& atlas,
        std::string& error)
    {
        if (scene.Textures.size() >
            static_cast<size_t>(std::numeric_limits<uint32_t>::max()))
        {
            error = "Ray texture table is too large.";
            return false;
        }

        uint64_t texelCount = 0;
        for (const RayTexture& texture : scene.Textures)
        {
            if (!texture.IsValid())
            {
                continue;
            }
            texelCount += static_cast<uint64_t>(texture.Width) * texture.Height;
            if (texelCount > std::numeric_limits<uint32_t>::max())
            {
                error = "Ray texture atlas is too large.";
                return false;
            }
        }

        PackedRayTextureAtlas result;
        result.Descriptors.resize(scene.Textures.size());
        result.Texels.reserve(static_cast<size_t>(texelCount));
        for (size_t textureIndex = 0;
             textureIndex < scene.Textures.size();
             ++textureIndex)
        {
            const RayTexture& texture = scene.Textures[textureIndex];
            if (!texture.IsValid())
            {
                continue;
            }

            PackedRayTextureDescriptor& descriptor =
                result.Descriptors[textureIndex];
            descriptor.OffsetTexels =
                static_cast<uint32_t>(result.Texels.size());
            descriptor.Width = texture.Width;
            descriptor.Height = texture.Height;

            const std::vector<uint8_t>& source = *texture.PixelsRGBA8;
            const size_t sourceTexelCount =
                static_cast<size_t>(texture.Width) * texture.Height;
            for (size_t texel = 0; texel < sourceTexelCount; ++texel)
            {
                const size_t byte = texel * 4U;
                result.Texels.push_back(
                    static_cast<uint32_t>(source[byte]) |
                    (static_cast<uint32_t>(source[byte + 1U]) << 8U) |
                    (static_cast<uint32_t>(source[byte + 2U]) << 16U) |
                    (static_cast<uint32_t>(source[byte + 3U]) << 24U));
            }
        }
        atlas = std::move(result);
        error.clear();
        return true;
    }

    RenderResult MakeFailedResult(std::string error)
    {
        RenderResult result;
        result.Status = RenderStatus::Failed;
        result.Error = std::move(error);
        return result;
    }

    RenderResult MakeCancelledResult()
    {
        RenderResult result;
        result.Status = RenderStatus::Cancelled;
        return result;
    }

    uint32_t GetBoundedWorkerCount(uint32_t maximum)
    {
        const uint32_t hardware = std::max(1u, std::thread::hardware_concurrency());
        return std::max(1u, std::min(hardware, maximum));
    }

    CancellationProbe CombineCancellation(
        const CancellationToken& queued,
        const CancellationToken& external)
    {
        return [queued, external]() {
            return queued.IsCancellationRequested() || external.IsCancellationRequested();
        };
    }

    struct PersistentRenderQueue::Impl
    {
        struct QueueItem
        {
            Task TaskFunction;
            CancellationSource Cancellation;
            std::shared_ptr<std::promise<RenderResult>> Promise;
        };

        Impl()
            : Thread(&Impl::ThreadMain, this)
        {
        }

        ~Impl()
        {
            Shutdown();
        }

        RenderJob Submit(Task task) const
        {
            CancellationSource cancellation;
            auto promise = std::make_shared<std::promise<RenderResult>>();
            std::future<RenderResult> future = promise->get_future();
            if (!task)
            {
                promise->set_value(MakeFailedResult("A ray tracing queue task must not be empty."));
                return RenderJob(cancellation, std::move(future));
            }

            {
                std::lock_guard<std::mutex> lock(Mutex);
                if (StopRequested)
                {
                    cancellation.Cancel();
                    promise->set_value(MakeCancelledResult());
                    return RenderJob(cancellation, std::move(future));
                }
                QueueItem item;
                item.TaskFunction = std::move(task);
                item.Cancellation = cancellation;
                item.Promise = std::move(promise);
                Queue.push_back(std::move(item));
            }
            WorkCondition.notify_one();
            return RenderJob(cancellation, std::move(future));
        }

        void WaitIdle() const
        {
            std::unique_lock<std::mutex> lock(Mutex);
            IdleCondition.wait(lock, [this]() { return Queue.empty() && !Active; });
        }

        void Shutdown()
        {
            {
                std::lock_guard<std::mutex> lock(Mutex);
                if (!StopRequested)
                {
                    StopRequested = true;
                    if (CurrentCancellation) CurrentCancellation->Cancel();
                    for (const QueueItem& item : Queue) item.Cancellation.Cancel();
                }
            }
            WorkCondition.notify_all();
            if (Thread.joinable()) Thread.join();
        }

        void ThreadMain()
        {
            while (true)
            {
                QueueItem item;
                {
                    std::unique_lock<std::mutex> lock(Mutex);
                    WorkCondition.wait(lock, [this]() {
                        return StopRequested || !Queue.empty();
                    });
                    if (StopRequested && Queue.empty()) break;
                    item = std::move(Queue.front());
                    Queue.pop_front();
                    CurrentCancellation = item.Cancellation;
                    Active = true;
                }

                RenderResult result;
                try
                {
                    result = item.TaskFunction(item.Cancellation.GetToken());
                }
                catch (const std::exception& exception)
                {
                    result = MakeFailedResult(
                        std::string("Ray tracing queue failure: ") + exception.what());
                }
                catch (...)
                {
                    result = MakeFailedResult(
                        "Ray tracing queue failed with an unknown exception.");
                }
                try
                {
                    item.Promise->set_value(std::move(result));
                }
                catch (...)
                {
                    // An abandoned promise must not terminate the persistent backend thread.
                }

                {
                    std::lock_guard<std::mutex> lock(Mutex);
                    CurrentCancellation.reset();
                    Active = false;
                }
                IdleCondition.notify_all();
            }

            {
                std::lock_guard<std::mutex> lock(Mutex);
                Active = false;
                CurrentCancellation.reset();
            }
            IdleCondition.notify_all();
        }

        mutable std::mutex Mutex;
        mutable std::condition_variable WorkCondition;
        mutable std::condition_variable IdleCondition;
        mutable std::deque<QueueItem> Queue;
        mutable std::optional<CancellationSource> CurrentCancellation;
        mutable bool Active = false;
        bool StopRequested = false;
        std::thread Thread;
    };

    PersistentRenderQueue::PersistentRenderQueue()
        : m_Impl(std::make_unique<Impl>())
    {
    }

    PersistentRenderQueue::~PersistentRenderQueue() = default;

    RenderJob PersistentRenderQueue::Submit(Task task) const
    {
        return m_Impl->Submit(std::move(task));
    }

    void PersistentRenderQueue::WaitIdle() const
    {
        m_Impl->WaitIdle();
    }
}
}
}
