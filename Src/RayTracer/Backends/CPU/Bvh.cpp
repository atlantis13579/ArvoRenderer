#include "RayTracer/Backends/CPU/CpuTransport.h"
#include <algorithm>
#include <array>
#include <cmath>

namespace Arvo::RayTracing::Cpu
{
// Median-split bounding volume hierarchy. Centroid partitioning bounds tree
// depth; overflow storage keeps traversal correct even beyond the local stack.
// Slab intervals are clipped to the ray and the nearest accepted intersection.
constexpr uint32_t LocalStackCapacity = 64;
constexpr uint32_t CancellationPollInterval = 4096;
constexpr uint32_t MaximumLeafPrimitives = 4;
constexpr float MinimumCentroidExtent = 1.0e-6f;
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

struct BVH::Impl
{
public:
    Impl(const SceneGeometry& scene, CancellationProbe cancellation)
        : m_Scene(scene)
        , m_Cancellation(std::move(cancellation))
    {
        m_Primitives.reserve(scene.Triangles.size() + scene.Spheres.size());
        for (uint32_t index = 0; index < scene.Triangles.size(); ++index)
        {
            if (index % CancellationPollInterval == 0 && IsCancelled())
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
            if (index % CancellationPollInterval == 0 && IsCancelled())
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

    bool Intersect(
        const Ray& ray,
        Hit& closestHit,
        uint64_t& primitiveTests,
        bool anyHit = false) const
    {
        if (m_Nodes.empty()) return false;

        bool found = false;
        std::array<uint32_t, LocalStackCapacity> localStack{};
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
                candidate.PrimitiveIndex = closestHit.PrimitiveIndex;
                bool primitiveHit = false;
                if (primitive.Type == PrimitiveType::Triangle)
                {
                    primitiveHit = IntersectTriangle(m_Scene.Triangles[primitive.Index], primitive.Index, ray, candidate);
                }
                else
                {
                    primitiveHit = IntersectSphere(m_Scene.Spheres[primitive.Index], static_cast<uint32_t>(m_Scene.Triangles.size()) + primitive.Index, ray, candidate);
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
        if (count <= MaximumLeafPrimitives || Component(extent, axis) < MinimumCentroidExtent)
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
            if (direction == 0.0f)
            {
                if (origin < minimum || origin > maximum) return false;
                continue;
            }
            float distance0 = (minimum - origin) / direction;
            float distance1 = (maximum - origin) / direction;
            if (distance0 > distance1) std::swap(distance0, distance1);
            nearDistance = std::max(nearDistance, distance0);
            farDistance = std::min(farDistance, distance1);
            if (nearDistance > farDistance) return false;
        }
        return true;
    }

    const SceneGeometry& m_Scene;
    CancellationProbe m_Cancellation;
    std::vector<PrimitiveReference> m_Primitives;
    std::vector<BVHNode> m_Nodes;
    bool m_WasCancelled = false;
};
}

namespace Arvo::RayTracing::Cpu
{
BVH::BVH(const SceneGeometry& scene, CancellationProbe cancellation)
    : Implementation(std::make_unique<Impl>(scene, std::move(cancellation))) {}
BVH::~BVH() = default;
bool BVH::WasCancelled() const { return Implementation->WasCancelled(); }
bool BVH::Intersect(const Ray& ray, Hit& hit, uint64_t& tests, bool anyHit) const
{ return Implementation->Intersect(ray, hit, tests, anyHit); }
bool BVH::Occluded(const Ray& ray, uint64_t& tests) const
{ return Implementation->Occluded(ray, tests); }
}
