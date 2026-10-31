#include "TracingCore.h"

#include "Maths/ColorSpace.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <exception>
#include <limits>
#include <mutex>
#include <thread>
#include <utility>

namespace Arvo
{
namespace RayTracing
{
namespace Detail
{
    struct PersistentExecutor::Impl
    {
        struct Batch
        {
            uint32_t UnitCount = 0;
            uint32_t WorkerLimit = 1;
            UnitFunction Function;
            CancellationProbe Cancellation;
            std::atomic<uint32_t> NextUnit{0};
            std::atomic<uint32_t> RemainingWorkers{0};
            std::atomic<bool> Failed{false};
            std::mutex CompletionMutex;
            std::condition_variable CompletionCondition;
            bool Completed = false;
            std::exception_ptr Exception;
        };

        explicit Impl(uint32_t workerCount)
        {
            const uint32_t count = std::max(1u, workerCount);
            Workers.reserve(count);
            for (uint32_t index = 0; index < count; ++index)
            {
                Workers.emplace_back([this, index]() { WorkerMain(index); });
            }
        }

        ~Impl()
        {
            {
                std::lock_guard<std::mutex> lock(Mutex);
                StopRequested = true;
            }
            WorkCondition.notify_all();
            for (std::thread& worker : Workers)
            {
                if (worker.joinable()) worker.join();
            }
        }

        void WorkerMain(uint32_t workerIndex)
        {
            uint64_t observedGeneration = 0;
            while (true)
            {
                std::shared_ptr<Batch> batch;
                {
                    std::unique_lock<std::mutex> lock(Mutex);
                    WorkCondition.wait(lock, [&]() {
                        return StopRequested || Generation != observedGeneration;
                    });
                    if (StopRequested) return;
                    observedGeneration = Generation;
                    batch = CurrentBatch;
                }
                if (!batch || workerIndex >= batch->WorkerLimit) continue;

                try
                {
                    while (!batch->Failed.load(std::memory_order_relaxed) &&
                           !(batch->Cancellation && batch->Cancellation()))
                    {
                        const uint32_t unit = batch->NextUnit.fetch_add(1, std::memory_order_relaxed);
                        if (unit >= batch->UnitCount) break;
                        batch->Function(unit);
                    }
                }
                catch (...)
                {
                    bool expected = false;
                    if (batch->Failed.compare_exchange_strong(expected, true))
                    {
                        std::lock_guard<std::mutex> lock(batch->CompletionMutex);
                        batch->Exception = std::current_exception();
                    }
                }

                if (batch->RemainingWorkers.fetch_sub(1, std::memory_order_acq_rel) == 1)
                {
                    {
                        std::lock_guard<std::mutex> lock(batch->CompletionMutex);
                        batch->Completed = true;
                    }
                    batch->CompletionCondition.notify_one();
                }
            }
        }

        std::vector<std::thread> Workers;
        std::mutex Mutex;
        std::condition_variable WorkCondition;
        std::shared_ptr<Batch> CurrentBatch;
        uint64_t Generation = 0;
        bool StopRequested = false;
    };

    namespace
    {
        constexpr float Pi = 3.14159265358979323846f;
        constexpr float RayEpsilon = 1.0e-4f;

        Float3 Multiply(const Float3& lhs, const Float3& rhs)
        {
            return Float3(lhs.X * rhs.X, lhs.Y * rhs.Y, lhs.Z * rhs.Z);
        }

        float RepeatCoordinate(float value) noexcept
        {
            if (!std::isfinite(value))
            {
                return 0.0f;
            }
            return value - std::floor(value);
        }

        uint32_t WrapTexel(int64_t coordinate, uint32_t size) noexcept
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

        Float3 Clamp(const Float3& value, float minimum, float maximum)
        {
            return Float3(
                std::max(minimum, std::min(maximum, value.X)),
                std::max(minimum, std::min(maximum, value.Y)),
                std::max(minimum, std::min(maximum, value.Z)));
        }

        Float3 Reflect(const Float3& direction, const Float3& normal)
        {
            return direction - normal * (2.0f * Dot(direction, normal));
        }

        float LengthSquared(const Float3& value)
        {
            return Dot(value, value);
        }

        bool IsFinite(const Float3& value)
        {
            return std::isfinite(value.X) && std::isfinite(value.Y) && std::isfinite(value.Z);
        }

        bool IsFinite(const Float2& value)
        {
            return std::isfinite(value.X) && std::isfinite(value.Y);
        }

        float Component(const Float3& value, uint32_t axis)
        {
            if (axis == 0) return value.X;
            if (axis == 1) return value.Y;
            return value.Z;
        }

        struct Bounds
        {
            Float3 Minimum = Float3(
                std::numeric_limits<float>::max(),
                std::numeric_limits<float>::max(),
                std::numeric_limits<float>::max());
            Float3 Maximum = Float3(
                -std::numeric_limits<float>::max(),
                -std::numeric_limits<float>::max(),
                -std::numeric_limits<float>::max());

            void Expand(const Float3& point)
            {
                Minimum.X = std::min(Minimum.X, point.X);
                Minimum.Y = std::min(Minimum.Y, point.Y);
                Minimum.Z = std::min(Minimum.Z, point.Z);
                Maximum.X = std::max(Maximum.X, point.X);
                Maximum.Y = std::max(Maximum.Y, point.Y);
                Maximum.Z = std::max(Maximum.Z, point.Z);
            }

            void Expand(const Bounds& bounds)
            {
                Expand(bounds.Minimum);
                Expand(bounds.Maximum);
            }

            Float3 Center() const { return (Minimum + Maximum) * 0.5f; }
        };

        enum class PrimitiveType : uint8_t
        {
            Triangle,
            Sphere
        };

        struct PrimitiveReference
        {
            Bounds BoundingBox;
            Float3 Centroid;
            uint32_t Index = 0;
            PrimitiveType Type = PrimitiveType::Triangle;
        };

        struct BVHNode
        {
            Bounds BoundingBox;
            uint32_t FirstPrimitive = 0;
            uint32_t PrimitiveCount = 0;
            uint32_t LeftChild = 0;
            uint32_t RightChild = 0;

            bool IsLeaf() const { return PrimitiveCount != 0; }
        };

        class BVH
        {
        public:
            BVH(const Scene& scene, CancellationProbe cancellation)
                : m_Scene(scene)
                , m_Cancellation(std::move(cancellation))
            {
                m_Primitives.reserve(scene.Triangles.size() + scene.Spheres.size());
                for (uint32_t index = 0; index < scene.Triangles.size(); ++index)
                {
                    if ((index & 4095u) == 0 && IsCancelled())
                    {
                        m_WasCancelled = true;
                        return;
                    }
                    const Triangle& triangle = scene.Triangles[index];
                    PrimitiveReference reference;
                    reference.BoundingBox.Expand(triangle.Position0);
                    reference.BoundingBox.Expand(triangle.Position1);
                    reference.BoundingBox.Expand(triangle.Position2);
                    reference.Centroid = reference.BoundingBox.Center();
                    reference.Index = index;
                    reference.Type = PrimitiveType::Triangle;
                    m_Primitives.push_back(reference);
                }
                for (uint32_t index = 0; index < scene.Spheres.size(); ++index)
                {
                    if ((index & 4095u) == 0 && IsCancelled())
                    {
                        m_WasCancelled = true;
                        return;
                    }
                    const Sphere& sphere = scene.Spheres[index];
                    PrimitiveReference reference;
                    const Float3 radius(sphere.Radius, sphere.Radius, sphere.Radius);
                    reference.BoundingBox.Expand(sphere.Center - radius);
                    reference.BoundingBox.Expand(sphere.Center + radius);
                    reference.Centroid = sphere.Center;
                    reference.Index = index;
                    reference.Type = PrimitiveType::Sphere;
                    m_Primitives.push_back(reference);
                }
                if (!m_Primitives.empty())
                {
                    m_Nodes.reserve(m_Primitives.size() * 2);
                    BuildNode(0, static_cast<uint32_t>(m_Primitives.size()));
                }
            }

            bool WasCancelled() const { return m_WasCancelled; }

            struct Hit
            {
                float Distance = std::numeric_limits<float>::max();
                Float3 Normal;
                Float3 Color = Float3(1.0f, 1.0f, 1.0f);
                Float2 TexCoord;
                uint32_t MaterialIndex = 0;
                bool HasTexCoord = false;
            };

            bool Intersect(
                const Ray& ray,
                Hit& closestHit,
                uint64_t& primitiveTests,
                bool anyHit = false) const
            {
                if (m_Nodes.empty()) return false;

                bool found = false;
                std::array<uint32_t, 64> localStack{};
                std::vector<uint32_t> overflowStack;
                uint32_t stackSize = 1;
                localStack[0] = 0;

                const auto push = [&localStack, &overflowStack, &stackSize](uint32_t value)
                {
                    if (stackSize < localStack.size()) localStack[stackSize] = value;
                    else overflowStack.push_back(value);
                    ++stackSize;
                };
                const auto pop = [&localStack, &overflowStack, &stackSize]()
                {
                    --stackSize;
                    if (stackSize < localStack.size()) return localStack[stackSize];
                    const uint32_t value = overflowStack.back();
                    overflowStack.pop_back();
                    return value;
                };

                while (stackSize != 0)
                {
                    const BVHNode& node = m_Nodes[pop()];
                    if (!IntersectBounds(node.BoundingBox, ray, closestHit.Distance)) continue;
                    if (!node.IsLeaf())
                    {
                        push(node.RightChild);
                        push(node.LeftChild);
                        continue;
                    }

                    for (uint32_t offset = 0; offset < node.PrimitiveCount; ++offset)
                    {
                        ++primitiveTests;
                        const PrimitiveReference& primitive = m_Primitives[node.FirstPrimitive + offset];
                        Hit candidate;
                        candidate.Distance = closestHit.Distance;
                        bool primitiveHit = false;
                        if (primitive.Type == PrimitiveType::Triangle)
                        {
                            primitiveHit = IntersectTriangle(m_Scene.Triangles[primitive.Index], ray, candidate);
                        }
                        else
                        {
                            primitiveHit = IntersectSphere(m_Scene.Spheres[primitive.Index], ray, candidate);
                        }
                        if (primitiveHit)
                        {
                            // Shadow rays only need an occluder inside their finite interval.
                            if (anyHit) return true;
                            closestHit = candidate;
                            found = true;
                        }
                    }
                }
                return found;
            }

            bool Occluded(const Ray& ray, uint64_t& primitiveTests) const
            {
                Hit hit;
                hit.Distance = ray.MaxDistance;
                return Intersect(ray, hit, primitiveTests, true);
            }

        private:
            uint32_t BuildNode(uint32_t begin, uint32_t end)
            {
                const uint32_t nodeIndex = static_cast<uint32_t>(m_Nodes.size());
                m_Nodes.emplace_back();
                if (IsCancelled())
                {
                    m_WasCancelled = true;
                    m_Nodes[nodeIndex].FirstPrimitive = begin;
                    m_Nodes[nodeIndex].PrimitiveCount = end - begin;
                    return nodeIndex;
                }
                Bounds primitiveBounds;
                Bounds centroidBounds;
                for (uint32_t index = begin; index < end; ++index)
                {
                    primitiveBounds.Expand(m_Primitives[index].BoundingBox);
                    centroidBounds.Expand(m_Primitives[index].Centroid);
                }
                m_Nodes[nodeIndex].BoundingBox = primitiveBounds;

                const uint32_t count = end - begin;
                const Float3 extent = centroidBounds.Maximum - centroidBounds.Minimum;
                uint32_t axis = extent.Y > extent.X ? 1u : 0u;
                if (Component(extent, 2) > Component(extent, axis)) axis = 2;
                if (count <= 4 || Component(extent, axis) < 1.0e-6f)
                {
                    m_Nodes[nodeIndex].FirstPrimitive = begin;
                    m_Nodes[nodeIndex].PrimitiveCount = count;
                    return nodeIndex;
                }

                const uint32_t middle = begin + count / 2;
                std::nth_element(
                    m_Primitives.begin() + begin,
                    m_Primitives.begin() + middle,
                    m_Primitives.begin() + end,
                    [axis](const PrimitiveReference& lhs, const PrimitiveReference& rhs)
                    {
                        return Component(lhs.Centroid, axis) < Component(rhs.Centroid, axis);
                    });
                const uint32_t left = BuildNode(begin, middle);
                const uint32_t right = BuildNode(middle, end);
                m_Nodes[nodeIndex].LeftChild = left;
                m_Nodes[nodeIndex].RightChild = right;
                return nodeIndex;
            }

            bool IsCancelled() const
            {
                return m_Cancellation && m_Cancellation();
            }

            static bool IntersectBounds(const Bounds& bounds, const Ray& ray, float closestDistance)
            {
                float nearDistance = ray.MinDistance;
                float farDistance = std::min(ray.MaxDistance, closestDistance);
                for (uint32_t axis = 0; axis < 3; ++axis)
                {
                    const float origin = Component(ray.Origin, axis);
                    const float direction = Component(ray.Direction, axis);
                    const float minimum = Component(bounds.Minimum, axis);
                    const float maximum = Component(bounds.Maximum, axis);
                    if (std::abs(direction) < 1.0e-12f)
                    {
                        if (origin < minimum || origin > maximum) return false;
                        continue;
                    }
                    const float inverseDirection = 1.0f / direction;
                    float distance0 = (minimum - origin) * inverseDirection;
                    float distance1 = (maximum - origin) * inverseDirection;
                    if (distance0 > distance1) std::swap(distance0, distance1);
                    nearDistance = std::max(nearDistance, distance0);
                    farDistance = std::min(farDistance, distance1);
                    if (nearDistance > farDistance) return false;
                }
                return true;
            }

            static bool IntersectTriangle(const Triangle& triangle, const Ray& ray, Hit& hit)
            {
                const Float3 edge1 = triangle.Position1 - triangle.Position0;
                const Float3 edge2 = triangle.Position2 - triangle.Position0;
                const Float3 p = Cross(ray.Direction, edge2);
                const float determinant = Dot(edge1, p);
                if (std::abs(determinant) < 1.0e-8f) return false;
                const float inverseDeterminant = 1.0f / determinant;
                const Float3 t = ray.Origin - triangle.Position0;
                const float barycentric1 = Dot(t, p) * inverseDeterminant;
                if (barycentric1 < 0.0f || barycentric1 > 1.0f) return false;
                const Float3 q = Cross(t, edge1);
                const float barycentric2 = Dot(ray.Direction, q) * inverseDeterminant;
                if (barycentric2 < 0.0f || barycentric1 + barycentric2 > 1.0f) return false;
                const float distance = Dot(edge2, q) * inverseDeterminant;
                if (distance < ray.MinDistance || distance > ray.MaxDistance || distance >= hit.Distance) return false;

                Float3 normal = Normalize(Cross(edge1, edge2));
                if (triangle.SmoothNormals)
                {
                    const float barycentric0 = 1.0f - barycentric1 - barycentric2;
                    const Float3 smoothNormal = triangle.Normal0 * barycentric0
                        + triangle.Normal1 * barycentric1 + triangle.Normal2 * barycentric2;
                    if (LengthSquared(smoothNormal) > 1.0e-12f) normal = Normalize(smoothNormal);
                }
                const float barycentric0 = 1.0f - barycentric1 - barycentric2;
                if (Dot(normal, ray.Direction) > 0.0f) normal = -normal;
                hit.Distance = distance;
                hit.Normal = normal;
                hit.Color = triangle.Color0 * barycentric0
                    + triangle.Color1 * barycentric1 + triangle.Color2 * barycentric2;
                hit.TexCoord = {
                    triangle.TexCoord0.X * barycentric0 +
                        triangle.TexCoord1.X * barycentric1 +
                        triangle.TexCoord2.X * barycentric2,
                    triangle.TexCoord0.Y * barycentric0 +
                        triangle.TexCoord1.Y * barycentric1 +
                        triangle.TexCoord2.Y * barycentric2};
                hit.MaterialIndex = triangle.MaterialIndex;
                hit.HasTexCoord = true;
                return true;
            }

            static bool IntersectSphere(const Sphere& sphere, const Ray& ray, Hit& hit)
            {
                const Float3 offset = ray.Origin - sphere.Center;
                const float halfB = Dot(offset, ray.Direction);
                const float c = Dot(offset, offset) - sphere.Radius * sphere.Radius;
                const float discriminant = halfB * halfB - c;
                if (discriminant < 0.0f) return false;
                const float root = std::sqrt(discriminant);
                float distance = -halfB - root;
                if (distance < ray.MinDistance) distance = -halfB + root;
                if (distance < ray.MinDistance || distance > ray.MaxDistance || distance >= hit.Distance) return false;
                const Float3 position = ray.Origin + ray.Direction * distance;
                Float3 normal = Normalize(position - sphere.Center);
                if (Dot(normal, ray.Direction) > 0.0f) normal = -normal;
                hit.Distance = distance;
                hit.Normal = normal;
                hit.MaterialIndex = sphere.MaterialIndex;
                return true;
            }

            const Scene& m_Scene;
            CancellationProbe m_Cancellation;
            std::vector<PrimitiveReference> m_Primitives;
            std::vector<BVHNode> m_Nodes;
            bool m_WasCancelled = false;
        };

        struct AtomicStatistics
        {
            std::atomic<uint64_t> PrimaryRays{0};
            std::atomic<uint64_t> ShadowRays{0};
            std::atomic<uint64_t> ReflectionRays{0};
            std::atomic<uint64_t> PrimitiveTests{0};
            std::atomic<uint64_t> CompletedPixels{0};

            RenderStatistics Resolve(double buildMilliseconds, double renderMilliseconds) const
            {
                RenderStatistics statistics;
                statistics.PrimaryRays = PrimaryRays.load(std::memory_order_relaxed);
                statistics.ShadowRays = ShadowRays.load(std::memory_order_relaxed);
                statistics.ReflectionRays = ReflectionRays.load(std::memory_order_relaxed);
                statistics.PrimitiveTests = PrimitiveTests.load(std::memory_order_relaxed);
                statistics.CompletedPixels = CompletedPixels.load(std::memory_order_relaxed);
                statistics.BuildMilliseconds = buildMilliseconds;
                statistics.RenderMilliseconds = renderMilliseconds;
                return statistics;
            }
        };

        struct CameraFrame
        {
            Float3 Position;
            Float3 Forward;
            Float3 Right;
            Float3 Up;
            float HalfWidth = 1.0f;
            float HalfHeight = 1.0f;
        };

        uint32_t Hash(uint32_t value)
        {
            value ^= value >> 16;
            value *= 0x7feb352du;
            value ^= value >> 15;
            value *= 0x846ca68bu;
            value ^= value >> 16;
            return value;
        }

        float Random01(uint32_t seed)
        {
            return static_cast<float>(Hash(seed) & 0x00ffffffu) / static_cast<float>(0x01000000u);
        }

        Ray MakeCameraRay(
            const CameraFrame& camera,
            const RenderSettings& settings,
            uint32_t x,
            uint32_t y,
            uint32_t sample)
        {
            float jitterX = 0.5f;
            float jitterY = 0.5f;
            if (settings.SamplesPerPixel > 1)
            {
                const uint32_t seed = ((y + settings.TileY) * FullImageWidth(settings)
                    + x + settings.TileX) * 9781u + sample * 6271u;
                jitterX = Random01(seed);
                jitterY = Random01(seed ^ 0x68bc21ebu);
            }
            const float u = (static_cast<float>(x + settings.TileX) + jitterX) / static_cast<float>(FullImageWidth(settings));
            const float v = (static_cast<float>(y + settings.TileY) + jitterY) / static_cast<float>(FullImageHeight(settings));
            const float screenX = (2.0f * u - 1.0f) * camera.HalfWidth;
            const float screenY = (1.0f - 2.0f * v) * camera.HalfHeight;
            Ray ray;
            ray.Origin = camera.Position;
            ray.Direction = Normalize(camera.Forward + camera.Right * screenX + camera.Up * screenY);
            ray.MinDistance = RayEpsilon;
            return ray;
        }

        Float3 Background(const Scene& scene, const Ray& ray)
        {
            const float blend = std::max(0.0f, std::min(1.0f, ray.Direction.Y * 0.5f + 0.5f));
            return scene.BackgroundBottom * (1.0f - blend) + scene.BackgroundTop * blend;
        }

        Float3 TraceRay(
            const Scene& scene,
            const BVH& bvh,
            Ray ray,
            const RenderSettings& settings,
            AtomicStatistics& statistics,
            const CancellationProbe& cancellation,
            uint32_t pathSeed)
        {
            const bool pathTracing = settings.Integrator == Algorithm::PathTracing;
            bool previousDelta = true;
            Float3 radiance;
            Float3 throughput(1.0f, 1.0f, 1.0f);
            const float occludedVisibility = 1.0f - std::clamp(
                settings.ShadowStrength,
                0.0f,
                1.0f);
            for (uint32_t bounce = 0; bounce <= settings.MaxReflectionBounces; ++bounce)
            {
                if (cancellation && cancellation()) break;
                BVH::Hit hit;
                uint64_t primitiveTests = 0;
                if (!bvh.Intersect(ray, hit, primitiveTests))
                {
                    statistics.PrimitiveTests.fetch_add(primitiveTests, std::memory_order_relaxed);
                    radiance = radiance + Multiply(throughput, Background(scene, ray));
                    break;
                }
                statistics.PrimitiveTests.fetch_add(primitiveTests, std::memory_order_relaxed);
                const Material& material = scene.Materials[hit.MaterialIndex];
                Float3 surface = hit.Color;
                if (hit.HasTexCoord)
                {
                    surface = Multiply(
                        surface,
                        SampleDiffuseTexture(
                            scene,
                            material.DiffuseTextureIndex,
                            hit.TexCoord));
                }
                const Float3 diffuse = Multiply(surface, material.Albedo);
                const Float3 position = ray.Origin + ray.Direction * hit.Distance;
                const Float3 viewDirection = Normalize(-ray.Direction);
                Float3 direct = material.Emission
                    + Multiply(surface, material.AmbientColor)
                        * (pathTracing ? 0.0f : std::max(0.0f, scene.AmbientIntensity));
                const auto accumulateLightSample = [&](
                    const Float3& lightDirection,
                    const Float3& lightRadiance,
                    float visibility)
                {
                    const float directionLengthSquared = LengthSquared(lightDirection);
                    if (!std::isfinite(directionLengthSquared)
                        || directionLengthSquared <= 1.0e-12f)
                    {
                        return;
                    }
                    const Float3 direction =
                        lightDirection / std::sqrt(directionLengthSquared);
                    const float normalToLight =
                        std::max(0.0f, Dot(hit.Normal, direction));
                    if (normalToLight <= 0.0f)
                    {
                        return;
                    }
                    const Float3 visibleRadiance =
                        lightRadiance * std::clamp(visibility, 0.0f, 1.0f);
                    direct = direct
                        + Multiply(diffuse, visibleRadiance) * normalToLight
                            * (pathTracing ? 1.0f / Pi : 1.0f);
                    const Float3 halfVector =
                        Normalize(direction + viewDirection);
                    const float specularAmount = std::pow(
                        std::max(0.0f, Dot(hit.Normal, halfVector)),
                        std::max(material.Shininess, 1.0f));
                    direct = direct
                        + Multiply(material.SpecularColor, visibleRadiance)
                            * (pathTracing ? 0.0f : specularAmount);
                };
                const auto accumulateFiniteLight = [&](
                    const Float3& lightPosition,
                    const Float3& lightColor,
                    float intensity,
                    bool castShadows,
                    const Float3* emissionDirection,
                    float sampleWeight)
                {
                    if (intensity <= 0.0f || sampleWeight <= 0.0f)
                    {
                        return;
                    }
                    const Float3 toLight = lightPosition - position;
                    const float distanceSquared = LengthSquared(toLight);
                    if (!std::isfinite(distanceSquared)
                        || distanceSquared <= RayEpsilon * RayEpsilon)
                    {
                        return;
                    }
                    const float distance = std::sqrt(distanceSquared);
                    const Float3 directionToLight = toLight / distance;
                    const float surfaceCosine =
                        std::max(0.0f, Dot(hit.Normal, directionToLight));
                    if (surfaceCosine <= 0.0f)
                    {
                        return;
                    }
                    float emitterCosine = 1.0f;
                    if (emissionDirection != nullptr)
                    {
                        // A rectangular emitter radiates only through its front face.
                        emitterCosine = std::max(
                            0.0f,
                            Dot(*emissionDirection, -directionToLight));
                        if (emitterCosine <= 0.0f)
                        {
                            return;
                        }
                    }

                    float visibility = 1.0f;
                    if (settings.ShadowsEnabled &&
                        settings.ShadowStrength > 0.0f && castShadows)
                    {
                        Ray shadowRay;
                        shadowRay.Origin = position + hit.Normal * RayEpsilon;
                        shadowRay.Direction = directionToLight;
                        shadowRay.MinDistance = RayEpsilon;
                        // Finite lights must not be shadowed by geometry behind
                        // the emitter or by the emitter endpoint itself.
                        shadowRay.MaxDistance = distance - RayEpsilon;
                        if (shadowRay.MaxDistance > shadowRay.MinDistance)
                        {
                            uint64_t shadowTests = 0;
                            statistics.ShadowRays.fetch_add(
                                1,
                                std::memory_order_relaxed);
                            if (bvh.Occluded(shadowRay, shadowTests))
                            {
                                visibility = occludedVisibility;
                            }
                            statistics.PrimitiveTests.fetch_add(
                                shadowTests,
                                std::memory_order_relaxed);
                        }
                    }
                    const float attenuation =
                        intensity * emitterCosine * sampleWeight / distanceSquared;
                    accumulateLightSample(
                        directionToLight,
                        lightColor * attenuation,
                        visibility);
                };
                for (const DirectionalLight& light : scene.DirectionalLights)
                {
                    const Float3 directionToLight = Normalize(light.DirectionToLight);
                    const float cosine = std::max(0.0f, Dot(hit.Normal, directionToLight));
                    if (cosine <= 0.0f || light.Intensity <= 0.0f) continue;
                    float visibility = 1.0f;
                    if (settings.ShadowsEnabled &&
                        settings.ShadowStrength > 0.0f && light.CastShadows)
                    {
                        Ray shadowRay;
                        const Float3 shadowNormal =
                            Dot(hit.Normal, directionToLight) >= 0.0f
                                ? hit.Normal
                                : -hit.Normal;
                        shadowRay.Origin =
                            position + shadowNormal * RayEpsilon;
                        shadowRay.Direction = directionToLight;
                        shadowRay.MinDistance = RayEpsilon;
                        uint64_t shadowTests = 0;
                        statistics.ShadowRays.fetch_add(1, std::memory_order_relaxed);
                        if (bvh.Occluded(shadowRay, shadowTests))
                        {
                            visibility = occludedVisibility;
                        }
                        statistics.PrimitiveTests.fetch_add(shadowTests, std::memory_order_relaxed);
                    }
                    accumulateLightSample(
                        directionToLight,
                        light.Color * light.Intensity,
                        visibility);
                }
                for (const PointLight& light : scene.PointLights)
                {
                    accumulateFiniteLight(
                        light.Position,
                        light.Color,
                        light.Intensity,
                        light.CastShadows,
                        nullptr,
                        1.0f);
                }
                uint32_t areaLightIndex = static_cast<uint32_t>(scene.DirectionalLights.size() + scene.PointLights.size());
                for (const RectAreaLight& light : scene.RectAreaLights)
                {
                    const Float3 emissionDirection = Normalize(light.Direction);
                    const Float3 helper = std::fabs(emissionDirection.Y) < 0.999f
                        ? Float3(0.0f, 1.0f, 0.0f)
                        : Float3(1.0f, 0.0f, 0.0f);
                    const Float3 tangent =
                        Normalize(Cross(helper, emissionDirection));
                    const Float3 bitangent =
                        Normalize(Cross(emissionDirection, tangent));
                    for (uint32_t sampleIndex = 0; sampleIndex < 4; ++sampleIndex)
                    {
                        if (cancellation && cancellation()) break;
                        float offsetX = (sampleIndex & 1u) != 0 ? 0.25f : -0.25f;
                        float offsetY = (sampleIndex & 2u) != 0 ? 0.25f : -0.25f;
                        if (pathTracing)
                        {
                            const uint32_t lightSeed = pathSeed ^ ((bounce + 1u) * 0x9e3779b9u)
                                ^ ((areaLightIndex + 1u) * 0x85ebca6bu) ^ ((sampleIndex + 1u) * 0xc2b2ae35u);
                            offsetX = (static_cast<float>(sampleIndex & 1u) + Random01(lightSeed)) * 0.5f - 0.5f;
                            offsetY = (static_cast<float>(sampleIndex >> 1u) + Random01(lightSeed ^ 0x68bc21ebu)) * 0.5f - 0.5f;
                        }
                        const Float3 samplePosition = light.Position
                            + tangent * (offsetX * light.Size.X)
                            + bitangent * (offsetY * light.Size.Y);
                        accumulateFiniteLight(samplePosition, light.Color, light.Intensity,
                            light.CastShadows, &emissionDirection, 0.25f);
                    }
                    ++areaLightIndex;
                }

                const float reflectivity = std::max(0.0f, std::min(1.0f, material.Reflectivity));
                if (pathTracing)
                {
                    // Cosine-weighted Lambert sampling cancels the BRDF/PDF ratio.
                    const Float3 visibleEmission = previousDelta || !material.SampledEmitter
                        ? material.Emission : Float3();
                    radiance = radiance + Multiply(throughput,
                        visibleEmission + (direct - material.Emission) * (1.0f - reflectivity));
                    if (bounce == settings.MaxReflectionBounces) break;
                    const uint32_t seed = pathSeed ^ ((bounce + 1u) * 0x9e3779b9u);
                    Float3 outgoing;
                    previousDelta = Random01(seed ^ 0xa511e9b3u) < reflectivity;
                    if (previousDelta)
                    {
                        outgoing = Normalize(Reflect(ray.Direction, hit.Normal));
                    }
                    else
                    {
                        const float radius = std::sqrt(Random01(seed));
                        const float phi = 2.0f * Pi * Random01(seed ^ 0x68bc21ebu);
                        const Float3 helper = std::fabs(hit.Normal.Y) < 0.999f
                            ? Float3(0.0f, 1.0f, 0.0f) : Float3(1.0f, 0.0f, 0.0f);
                        const Float3 tangent = Normalize(Cross(helper, hit.Normal));
                        const Float3 bitangent = Cross(hit.Normal, tangent);
                        outgoing = Normalize(tangent * (radius * std::cos(phi))
                            + bitangent * (radius * std::sin(phi))
                            + hit.Normal * std::sqrt(std::max(0.0f, 1.0f - radius * radius)));
                        throughput = Multiply(throughput, Float3(
                            std::clamp(diffuse.X, 0.0f, 1.0f),
                            std::clamp(diffuse.Y, 0.0f, 1.0f),
                            std::clamp(diffuse.Z, 0.0f, 1.0f)));
                    }
                    ray.Origin = position + hit.Normal * RayEpsilon;
                    ray.Direction = outgoing;
                    ray.MinDistance = RayEpsilon;
                    ray.MaxDistance = 1.0e30f;
                    statistics.ReflectionRays.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }
                radiance = radiance + Multiply(throughput, direct) * (1.0f - reflectivity);
                if (reflectivity <= 0.0f || bounce == settings.MaxReflectionBounces) break;
                throughput = throughput * reflectivity;
                const Float3 reflectedDirection =
                    Normalize(Reflect(ray.Direction, hit.Normal));
                const Float3 reflectionNormal =
                    Dot(hit.Normal, reflectedDirection) >= 0.0f
                        ? hit.Normal
                        : -hit.Normal;
                ray.Origin = position + reflectionNormal * RayEpsilon;
                ray.Direction = reflectedDirection;
                ray.MinDistance = RayEpsilon;
                ray.MaxDistance = 1.0e30f;
                statistics.ReflectionRays.fetch_add(1, std::memory_order_relaxed);
            }
            return radiance;
        }

        uint8_t EncodeChannel(float value, bool gammaCorrection)
        {
            return gammaCorrection
                ? Maths::LinearToSRGB8(value)
                : Maths::LinearToUNorm8(value);
        }

        bool ValidateInput(
            const Scene& scene,
            const Camera& camera,
            const RenderSettings& settings,
            std::string& error)
        {
            if (!ValidateTileSettings(settings, error)) return false;
            if (settings.Width == 0 || settings.Height == 0)
            {
                error = "Image dimensions must be non-zero.";
                return false;
            }
            if (settings.Width > 16384 || settings.Height > 16384)
            {
                error = "Image dimensions exceed the 16384 pixel safety limit.";
                return false;
            }
            if (settings.SamplesPerPixel == 0 || settings.SamplesPerPixel > 4096)
            {
                error = "SamplesPerPixel must be in the range [1, 4096].";
                return false;
            }
            if (settings.MaxReflectionBounces > 32)
            {
                error = "MaxReflectionBounces must not exceed 32.";
                return false;
            }
            if (!std::isfinite(settings.ShadowStrength)
                || settings.ShadowStrength < 0.0f
                || settings.ShadowStrength > 1.0f)
            {
                error = "ShadowStrength must be finite and in the range [0, 1].";
                return false;
            }
            if (!IsFinite(camera.Position) || !IsFinite(camera.Target) || !IsFinite(camera.Up)
                || !std::isfinite(camera.VerticalFieldOfViewDegrees))
            {
                error = "Camera values must be finite.";
                return false;
            }
            if (camera.VerticalFieldOfViewDegrees <= 1.0f || camera.VerticalFieldOfViewDegrees >= 179.0f)
            {
                error = "The camera vertical field of view must be between 1 and 179 degrees.";
                return false;
            }
            const float cameraDirectionLengthSquared = LengthSquared(camera.Target - camera.Position);
            const float cameraUpLengthSquared = LengthSquared(camera.Up);
            if (!std::isfinite(cameraDirectionLengthSquared)
                || !std::isfinite(cameraUpLengthSquared)
                || cameraDirectionLengthSquared < 1.0e-10f
                || cameraUpLengthSquared < 1.0e-10f)
            {
                error = "The camera position, target, and up vector do not form a valid frame.";
                return false;
            }
            const Float3 forward = Normalize(camera.Target - camera.Position);
            const float cameraFrameArea = LengthSquared(Cross(forward, Normalize(camera.Up)));
            if (!std::isfinite(cameraFrameArea) || cameraFrameArea < 1.0e-10f)
            {
                error = "The camera up vector is parallel to its view direction.";
                return false;
            }
            if ((!scene.Triangles.empty() || !scene.Spheres.empty()) && scene.Materials.empty())
            {
                error = "A scene containing geometry must contain at least one material.";
                return false;
            }
            if (!IsFinite(scene.BackgroundTop) || !IsFinite(scene.BackgroundBottom)
                || !std::isfinite(scene.AmbientIntensity) || scene.AmbientIntensity < 0.0f)
            {
                error = "Scene background and ambient values must be finite and non-negative.";
                return false;
            }
            for (const Material& material : scene.Materials)
            {
                if (!IsFinite(material.Albedo)
                    || !IsFinite(material.AmbientColor)
                    || !IsFinite(material.SpecularColor)
                    || !IsFinite(material.Emission)
                    || !std::isfinite(material.Shininess)
                    || !std::isfinite(material.Reflectivity))
                {
                    error = "Material values must be finite.";
                    return false;
                }
                if (material.AmbientColor.X < 0.0f
                    || material.AmbientColor.Y < 0.0f
                    || material.AmbientColor.Z < 0.0f
                    || material.SpecularColor.X < 0.0f
                    || material.SpecularColor.Y < 0.0f
                    || material.SpecularColor.Z < 0.0f
                    || material.Shininess < 0.0f)
                {
                    error =
                        "Material ambient, specular, and shininess values must be "
                        "non-negative.";
                    return false;
                }
            }
            for (const Triangle& triangle : scene.Triangles)
            {
                if (triangle.MaterialIndex >= scene.Materials.size())
                {
                    error = "A triangle has an invalid material index.";
                    return false;
                }
                if (!IsFinite(triangle.Position0) || !IsFinite(triangle.Position1)
                    || !IsFinite(triangle.Position2) || !IsFinite(triangle.Normal0)
                    || !IsFinite(triangle.Normal1) || !IsFinite(triangle.Normal2)
                    || !IsFinite(triangle.Color0) || !IsFinite(triangle.Color1)
                    || !IsFinite(triangle.Color2) || !IsFinite(triangle.TexCoord0)
                    || !IsFinite(triangle.TexCoord1) || !IsFinite(triangle.TexCoord2))
                {
                    error =
                        "Triangle positions, normals, colors, and texture "
                        "coordinates must be finite.";
                    return false;
                }
                const Float3 edge1 = triangle.Position1 - triangle.Position0;
                const Float3 edge2 = triangle.Position2 - triangle.Position0;
                const float edge1LengthSquared = LengthSquared(edge1);
                const float edge2LengthSquared = LengthSquared(edge2);
                const float triangleAreaSquared = LengthSquared(Cross(edge1, edge2));
                if (!std::isfinite(edge1LengthSquared) || !std::isfinite(edge2LengthSquared)
                    || !std::isfinite(triangleAreaSquared)
                    || triangleAreaSquared < 1.0e-16f)
                {
                    error = "Degenerate or numerically unsafe triangles are not accepted.";
                    return false;
                }
            }
            for (const Sphere& sphere : scene.Spheres)
            {
                if (sphere.MaterialIndex >= scene.Materials.size())
                {
                    error = "A sphere has an invalid material index.";
                    return false;
                }
                if (!IsFinite(sphere.Center) || !std::isfinite(sphere.Radius)
                    || sphere.Radius <= 0.0f)
                {
                    error = "Sphere radii must be positive.";
                    return false;
                }
                const Float3 radius(sphere.Radius, sphere.Radius, sphere.Radius);
                if (!IsFinite(sphere.Center - radius) || !IsFinite(sphere.Center + radius))
                {
                    error = "Sphere bounds must remain finite.";
                    return false;
                }
            }
            for (const DirectionalLight& light : scene.DirectionalLights)
            {
                const float lightDirectionLengthSquared = LengthSquared(light.DirectionToLight);
                if (!IsFinite(light.DirectionToLight) || !IsFinite(light.Color)
                    || !std::isfinite(light.Intensity) || light.Intensity < 0.0f
                    || !std::isfinite(lightDirectionLengthSquared)
                    || lightDirectionLengthSquared < 1.0e-12f)
                {
                    error = "Directional light values must be finite with a non-zero direction.";
                    return false;
                }
            }
            for (const PointLight& light : scene.PointLights)
            {
                if (!IsFinite(light.Position) || !IsFinite(light.Color)
                    || !std::isfinite(light.Intensity) || light.Intensity < 0.0f)
                {
                    error =
                        "Point light values must be finite with non-negative intensity.";
                    return false;
                }
            }
            for (const RectAreaLight& light : scene.RectAreaLights)
            {
                const float directionLengthSquared = LengthSquared(light.Direction);
                if (!IsFinite(light.Position) || !IsFinite(light.Direction)
                    || !IsFinite(light.Color) || !IsFinite(light.Size)
                    || !std::isfinite(light.Intensity) || light.Intensity < 0.0f
                    || !std::isfinite(directionLengthSquared)
                    || directionLengthSquared < 1.0e-12f
                    || light.Size.X <= 0.0f || light.Size.Y <= 0.0f)
                {
                    error =
                        "Rect area light values must be finite with a non-zero "
                        "direction, positive size, and non-negative intensity.";
                    return false;
                }
                const Float3 direction = Normalize(light.Direction);
                const Float3 helper = std::fabs(direction.Y) < 0.999f
                    ? Float3(0.0f, 1.0f, 0.0f)
                    : Float3(1.0f, 0.0f, 0.0f);
                const Float3 tangent = Normalize(Cross(helper, direction));
                const Float3 bitangent = Normalize(Cross(direction, tangent));
                const Float3 halfTangent = tangent * (light.Size.X * 0.5f);
                const Float3 halfBitangent = bitangent * (light.Size.Y * 0.5f);
                if (!IsFinite(light.Position + halfTangent + halfBitangent)
                    || !IsFinite(light.Position + halfTangent - halfBitangent)
                    || !IsFinite(light.Position - halfTangent + halfBitangent)
                    || !IsFinite(light.Position - halfTangent - halfBitangent))
                {
                    error = "Rect area light bounds must remain finite.";
                    return false;
                }
            }
            return true;
        }

        CameraFrame BuildCameraFrame(const Camera& camera, const RenderSettings& settings)
        {
            CameraFrame frame;
            frame.Position = camera.Position;
            frame.Forward = Normalize(camera.Target - camera.Position);
            frame.Right = Normalize(Cross(frame.Forward, camera.Up));
            frame.Up = Normalize(Cross(frame.Right, frame.Forward));
            frame.HalfHeight = std::tan(camera.VerticalFieldOfViewDegrees * Pi / 360.0f);
            frame.HalfWidth = frame.HalfHeight
                * (static_cast<float>(FullImageWidth(settings)) / static_cast<float>(FullImageHeight(settings)));
            return frame;
        }

        using PixelFunction = std::function<void(uint32_t, uint32_t)>;
        using PixelDispatcher = std::function<void(const PixelFunction&)>;

        RenderResult RenderCommon(
            const Scene& scene,
            const Camera& camera,
            const RenderSettings& settings,
            const CancellationProbe& cancellation,
            const PixelDispatcher& dispatch)
        {
            RenderResult result;
            if (cancellation && cancellation())
            {
                result.Status = RenderStatus::Cancelled;
                return result;
            }
            if (!ValidateInput(scene, camera, settings, result.Error)) return result;

            result.Output = Image(settings.Width, settings.Height);

            const auto buildBegin = std::chrono::steady_clock::now();
            const BVH bvh(scene, cancellation);
            const auto buildEnd = std::chrono::steady_clock::now();
            if (bvh.WasCancelled() || (cancellation && cancellation()))
            {
                result.Status = RenderStatus::Cancelled;
                return result;
            }

            const CameraFrame cameraFrame = BuildCameraFrame(camera, settings);
            AtomicStatistics statistics;
            std::vector<uint8_t>& pixels = result.Output.GetPixelsRGBA8();
            const auto renderBegin = std::chrono::steady_clock::now();
            const PixelFunction renderPixel = [&](uint32_t x, uint32_t y)
            {
                Float3 color;
                for (uint32_t sample = 0; sample < settings.SamplesPerPixel; ++sample)
                {
                    if (cancellation && cancellation()) return;
                    const Ray ray = MakeCameraRay(cameraFrame, settings, x, y, sample);
                    statistics.PrimaryRays.fetch_add(1, std::memory_order_relaxed);
                    const uint32_t pathSeed = ((y + settings.TileY) * FullImageWidth(settings)
                        + x + settings.TileX) * 9781u + sample * 6271u;
                    color = color + TraceRay(scene, bvh, ray, settings, statistics, cancellation, pathSeed);
                }
                color = color / static_cast<float>(settings.SamplesPerPixel);
                color.X = std::max(color.X, 0.0f);
                color.Y = std::max(color.Y, 0.0f);
                color.Z = std::max(color.Z, 0.0f);
                const float displayScale = Maths::DisplayLinearScale(
                    color.X,
                    color.Y,
                    color.Z);
                color = Clamp(color * displayScale, 0.0f, 1.0f);
                const size_t pixelOffset = (static_cast<size_t>(y) * settings.Width + x) * 4;
                pixels[pixelOffset] = EncodeChannel(color.X, settings.GammaCorrection);
                pixels[pixelOffset + 1] = EncodeChannel(color.Y, settings.GammaCorrection);
                pixels[pixelOffset + 2] = EncodeChannel(color.Z, settings.GammaCorrection);
                pixels[pixelOffset + 3] = 255;
                statistics.CompletedPixels.fetch_add(1, std::memory_order_relaxed);
            };

            dispatch(renderPixel);
            const auto renderEnd = std::chrono::steady_clock::now();
            result.Statistics = statistics.Resolve(
                std::chrono::duration<double, std::milli>(buildEnd - buildBegin).count(),
                std::chrono::duration<double, std::milli>(renderEnd - renderBegin).count());
            result.Status = cancellation && cancellation()
                ? RenderStatus::Cancelled
                : RenderStatus::Completed;
            return result;
        }
    }

    PersistentExecutor::PersistentExecutor(uint32_t workerCount)
        : m_Impl(std::make_unique<Impl>(workerCount))
    {
    }

    PersistentExecutor::~PersistentExecutor() = default;

    uint32_t PersistentExecutor::GetWorkerCount() const
    {
        return static_cast<uint32_t>(m_Impl->Workers.size());
    }

    void PersistentExecutor::Execute(
        uint32_t unitCount,
        uint32_t requestedWorkers,
        const UnitFunction& function,
        const CancellationProbe& cancellation)
    {
        if (unitCount == 0) return;
        auto batch = std::make_shared<Impl::Batch>();
        batch->UnitCount = unitCount;
        batch->WorkerLimit = requestedWorkers == 0
            ? GetWorkerCount()
            : std::min(requestedWorkers, GetWorkerCount());
        batch->WorkerLimit = std::max(1u, std::min(batch->WorkerLimit, unitCount));
        batch->RemainingWorkers.store(batch->WorkerLimit, std::memory_order_relaxed);
        batch->Function = function;
        batch->Cancellation = cancellation;
        {
            std::lock_guard<std::mutex> lock(m_Impl->Mutex);
            m_Impl->CurrentBatch = batch;
            ++m_Impl->Generation;
        }
        m_Impl->WorkCondition.notify_all();
        {
            std::unique_lock<std::mutex> lock(batch->CompletionMutex);
            batch->CompletionCondition.wait(lock, [&]() { return batch->Completed; });
        }
        {
            std::lock_guard<std::mutex> lock(m_Impl->Mutex);
            m_Impl->CurrentBatch.reset();
        }
        if (batch->Exception) std::rethrow_exception(batch->Exception);
    }

    RenderResult RenderCpuTiles(
        const Scene& scene,
        const Camera& camera,
        const RenderSettings& settings,
        PersistentExecutor& executor,
        const CancellationProbe& cancellation)
    {
        return RenderCommon(scene, camera, settings, cancellation, [&](const PixelFunction& pixel)
        {
            constexpr uint32_t TileSize = 16;
            const uint32_t tilesX = (settings.Width + TileSize - 1) / TileSize;
            const uint32_t tilesY = (settings.Height + TileSize - 1) / TileSize;
            executor.Execute(tilesX * tilesY, settings.WorkerCount, [&](uint32_t tile)
            {
                const uint32_t firstX = (tile % tilesX) * TileSize;
                const uint32_t firstY = (tile / tilesX) * TileSize;
                for (uint32_t y = firstY; y < std::min(firstY + TileSize, settings.Height); ++y)
                {
                    for (uint32_t x = firstX; x < std::min(firstX + TileSize, settings.Width); ++x)
                    {
                        pixel(x, y);
                    }
                }
            }, cancellation);
        });
    }

    bool ValidateRenderInput(
        const Scene& scene,
        const Camera& camera,
        const RenderSettings& settings,
        std::string& error)
    {
        return ValidateInput(scene, camera, settings, error);
    }
}
}
}
