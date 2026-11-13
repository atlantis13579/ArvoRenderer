#include "Renderer/RayTracing.h"

#include <chrono>
#include <cmath>
#include <fstream>
#include <utility>

namespace Arvo
{
namespace RayTracing
{
    namespace
    {
        RenderResult FailedResult(const std::string& error)
        {
            RenderResult result;
            result.Status = RenderStatus::Failed;
            result.Error = error;
            return result;
        }
    }

    float Dot(const Float3& lhs, const Float3& rhs)
    {
        return lhs.X * rhs.X + lhs.Y * rhs.Y + lhs.Z * rhs.Z;
    }

    bool ValidateRayQueries(const std::vector<Ray>& rays, std::string& error)
    {
        constexpr size_t MaximumQueryBatch = 1u << 20;
        if (rays.size() > MaximumQueryBatch)
        {
            error = "A native ray-query batch may contain at most 1048576 rays.";
            return false;
        }
        const auto finite = [](const Float3& value)
        {
            return std::isfinite(value.X) && std::isfinite(value.Y) && std::isfinite(value.Z);
        };
        for (const Ray& ray : rays)
        {
            if (!finite(ray.Origin) || !finite(ray.Direction)
                || (ray.Direction.X == 0 && ray.Direction.Y == 0 && ray.Direction.Z == 0)
                || !std::isfinite(ray.MinDistance) || !std::isfinite(ray.MaxDistance)
                || ray.MinDistance < 0 || ray.MaxDistance < ray.MinDistance)
            {
                error = "Ray queries require finite origins, nonzero finite directions, and finite 0 <= min <= max intervals.";
                return false;
            }
        }
        return true;
    }

    Float3 Cross(const Float3& lhs, const Float3& rhs)
    {
        return Float3(
            lhs.Y * rhs.Z - lhs.Z * rhs.Y,
            lhs.Z * rhs.X - lhs.X * rhs.Z,
            lhs.X * rhs.Y - lhs.Y * rhs.X);
    }

    Float3 Normalize(const Float3& value)
    {
        const float lengthSquared = Dot(value, value);
        if (lengthSquared <= 1.0e-20f) return Float3();
        return value / std::sqrt(lengthSquared);
    }

    Image::Image(uint32_t width, uint32_t height)
        : m_Width(width)
        , m_Height(height)
        , m_Pixels(static_cast<size_t>(width) * height * 4, 0)
    {
        for (size_t alpha = 3; alpha < m_Pixels.size(); alpha += 4)
        {
            m_Pixels[alpha] = 255;
        }
    }

    const uint8_t* Image::GetPixel(uint32_t x, uint32_t y) const
    {
        if (x >= m_Width || y >= m_Height) return nullptr;
        return &m_Pixels[(static_cast<size_t>(y) * m_Width + x) * 4];
    }

    bool Image::SavePPM(const std::string& path, std::string* error) const
    {
        if (IsEmpty())
        {
            if (error) *error = "Cannot save an empty image.";
            return false;
        }
        std::ofstream stream(path, std::ios::binary);
        if (!stream)
        {
            if (error) *error = "Could not open the output path.";
            return false;
        }
        stream << "P6\n" << m_Width << ' ' << m_Height << "\n255\n";
        for (size_t pixel = 0;
             pixel < static_cast<size_t>(m_Width) * m_Height;
             ++pixel)
        {
            stream.write(
                reinterpret_cast<const char*>(&m_Pixels[pixel * 4]),
                3);
        }
        if (!stream)
        {
            if (error) *error = "Writing the image failed.";
            return false;
        }
        return true;
    }

    bool CancellationToken::IsCancellationRequested() const
    {
        return m_State && m_State->load(std::memory_order_relaxed);
    }

    CancellationSource::CancellationSource()
        : m_State(std::make_shared<std::atomic<bool>>(false))
    {
    }

    CancellationToken CancellationSource::GetToken() const
    {
        return CancellationToken(m_State);
    }

    void CancellationSource::Cancel() const
    {
        if (m_State) m_State->store(true, std::memory_order_relaxed);
    }

    bool CancellationSource::IsCancellationRequested() const
    {
        return m_State && m_State->load(std::memory_order_relaxed);
    }

    RenderJob::RenderJob(
        CancellationSource cancellation,
        std::future<RenderResult> future)
        : m_Cancellation(std::move(cancellation))
        , m_Future(std::move(future))
    {
    }

    RenderJob::RenderJob(RenderJob&&) noexcept = default;

    RenderJob& RenderJob::operator=(RenderJob&& other) noexcept
    {
        if (this != &other)
        {
            Cancel();
            m_Cancellation = std::move(other.m_Cancellation);
            m_Future = std::move(other.m_Future);
        }
        return *this;
    }

    RenderJob::~RenderJob()
    {
        Cancel();
    }

    void RenderJob::Cancel() const
    {
        m_Cancellation.Cancel();
    }

    bool RenderJob::IsReady() const
    {
        return !m_Future.valid()
            || m_Future.wait_for(std::chrono::milliseconds(0))
                == std::future_status::ready;
    }

    void RenderJob::Wait() const
    {
        if (m_Future.valid()) m_Future.wait();
    }

    RenderResult RenderJob::GetResult()
    {
        if (!m_Future.valid())
        {
            return FailedResult("This render job has no pending result.");
        }
        return m_Future.get();
    }
}
}
