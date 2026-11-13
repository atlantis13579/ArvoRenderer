#include "KdTreeBuilder.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>

// Independently authored DXR spatial kd-tree builder. A plane partitions space;
// a primitive crossing it belongs to both children. Candidate-plane SAH stops
// when leaf work is cheaper. Depth and total reference budgets bound worst cases.
// Pharr, Jakob, Humphreys, PBRT 3e, "Kd-Tree Accelerator":
// https://pbr-book.org/3ed-2018/Primitives_and_Intersection_Acceleration/Kd-Tree_Accelerator
namespace Arvo::RayTracing::Dxr
{
    namespace
    {
        constexpr uint32_t LeafPrimitiveLimit = 8;
        constexpr uint32_t CandidatePlaneCount = 16;
        constexpr uint64_t MaximumReferencesPerPrimitive = 8;
        constexpr uint64_t MaximumNodesPerPrimitive = 4;
        constexpr uint64_t MaximumByteAddressedNodes =
            uint64_t(std::numeric_limits<uint32_t>::max()) / sizeof(KdNode);
        constexpr size_t CancellationCheckPeriod = 256;
        constexpr double TraversalCost = 1.0;

        float Component(const Float3& value, uint32_t axis)
        {
            return axis == 0 ? value.X : axis == 1 ? value.Y : value.Z;
        }

        void SetComponent(Float3& value, uint32_t axis, float component)
        {
            if (axis == 0) value.X = component;
            else if (axis == 1) value.Y = component;
            else value.Z = component;
        }

        struct Bounds
        {
            Float3 Minimum = Float3(std::numeric_limits<float>::max(),
                std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
            Float3 Maximum = Float3(std::numeric_limits<float>::lowest(),
                std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest());

            void Include(const Float3& point)
            {
                Minimum.X = std::min(Minimum.X, point.X);
                Minimum.Y = std::min(Minimum.Y, point.Y);
                Minimum.Z = std::min(Minimum.Z, point.Z);
                Maximum.X = std::max(Maximum.X, point.X);
                Maximum.Y = std::max(Maximum.Y, point.Y);
                Maximum.Z = std::max(Maximum.Z, point.Z);
            }

            double Area() const
            {
                const double x = double(Maximum.X) - Minimum.X;
                const double y = double(Maximum.Y) - Minimum.Y;
                const double z = double(Maximum.Z) - Minimum.Z;
                return 2 * (x * y + x * z + y * z);
            }

            void ExpandOutwards()
            {
                for (uint32_t axis = 0; axis < 3; ++axis)
                {
                    SetComponent(Minimum, axis, std::nextafter(Component(Minimum, axis),
                        std::numeric_limits<float>::lowest()));
                    SetComponent(Maximum, axis, std::nextafter(Component(Maximum, axis),
                        std::numeric_limits<float>::max()));
                }
            }
        };

        class Builder
        {
        public:
            Builder(const SceneGeometry& scene, const std::function<bool()>& cancelled,
                KdTreeData& tree, std::string& error)
                : m_Scene(scene), m_Cancelled(cancelled), m_Tree(tree), m_Error(error) {}

            bool Run()
            {
                m_Tree = {};
                const uint64_t count = m_Scene.Triangles.size() + m_Scene.Spheres.size();
                if (count >= InvalidRayPrimitiveIndex)
                {
                    m_Error = "DXR kd-tree exceeds 32-bit primitive indexing.";
                    return false;
                }
                if (Cancelled()) return false;
                if (count == 0) return true;
                m_ReferenceLimit = std::min(uint64_t(InvalidRayPrimitiveIndex),
                    count * MaximumReferencesPerPrimitive);
                // Raw HLSL loads use 32-bit byte offsets, even though the
                // root descriptor itself contains a 64-bit GPU address.
                m_NodeLimit = std::min(MaximumByteAddressedNodes,
                    count * MaximumNodesPerPrimitive + 1);
                m_ReferenceCount = count;
                m_PrimitiveBounds.reserve(static_cast<size_t>(count));
                Bounds world;
                for (const Triangle& triangle : m_Scene.Triangles)
                {
                    if (PeriodicCancellation()) return false;
                    Bounds bounds;
                    bounds.Include(triangle.Position0);
                    bounds.Include(triangle.Position1);
                    bounds.Include(triangle.Position2);
                    m_PrimitiveBounds.push_back(bounds);
                    world.Include(bounds.Minimum);
                    world.Include(bounds.Maximum);
                }
                for (const Sphere& sphere : m_Scene.Spheres)
                {
                    if (PeriodicCancellation()) return false;
                    const Float3 radius(sphere.Radius, sphere.Radius, sphere.Radius);
                    Bounds bounds;
                    bounds.Include(sphere.Center - radius);
                    bounds.Include(sphere.Center + radius);
                    bounds.ExpandOutwards();
                    m_PrimitiveBounds.push_back(bounds);
                    world.Include(bounds.Minimum);
                    world.Include(bounds.Maximum);
                }
                world.ExpandOutwards();
                m_Tree.Minimum = world.Minimum;
                m_Tree.Maximum = world.Maximum;
                std::vector<uint32_t> indices(static_cast<size_t>(count));
                std::iota(indices.begin(), indices.end(), 0u);
                m_Tree.Nodes.emplace_back();
                return BuildNode(0, world, indices, 0);
            }

        private:
            bool Cancelled()
            {
                if (!m_Cancelled || !m_Cancelled()) return false;
                m_Error = "DXR kd-tree construction was cancelled.";
                return true;
            }

            bool PeriodicCancellation()
            {
                return (++m_Work % CancellationCheckPeriod) == 0 && Cancelled();
            }

            bool MakeLeaf(uint32_t nodeIndex, const std::vector<uint32_t>& indices)
            {
                KdNode& node = m_Tree.Nodes[nodeIndex];
                node.Axis = KdLeafAxis;
                node.First = static_cast<uint32_t>(m_Tree.PrimitiveIndices.size());
                node.Count = static_cast<uint32_t>(indices.size());
                for (uint32_t index : indices)
                {
                    if (PeriodicCancellation()) return false;
                    m_Tree.PrimitiveIndices.push_back(index);
                }
                return true;
            }

            bool BuildNode(uint32_t nodeIndex, const Bounds& cell,
                const std::vector<uint32_t>& indices, uint32_t depth)
            {
                if (Cancelled()) return false;
                const double parentArea = cell.Area();
                if (indices.size() <= LeafPrimitiveLimit || depth == KdMaximumDepth
                    || !(parentArea > 0) || m_Tree.Nodes.size() + 2 > m_NodeLimit)
                    return MakeLeaf(nodeIndex, indices);

                double bestCost = static_cast<double>(indices.size());
                uint32_t bestAxis = KdLeafAxis;
                float bestSplit = 0;
                size_t bestLowCount = 0, bestHighCount = 0;
                for (uint32_t axis = 0; axis < 3; ++axis)
                {
                    const double lower = Component(cell.Minimum, axis);
                    const double extent = double(Component(cell.Maximum, axis)) - lower;
                    for (uint32_t candidate = 1; candidate <= CandidatePlaneCount; ++candidate)
                    {
                        const float plane = static_cast<float>(lower + extent * candidate / (CandidatePlaneCount + 1));
                        if (!(plane > Component(cell.Minimum, axis) && plane < Component(cell.Maximum, axis)))
                            continue;
                        size_t lowCount = 0, highCount = 0;
                        for (uint32_t index : indices)
                        {
                            if (PeriodicCancellation()) return false;
                            const Bounds& bounds = m_PrimitiveBounds[index];
                            lowCount += Component(bounds.Minimum, axis) <= plane;
                            highCount += Component(bounds.Maximum, axis) >= plane;
                        }
                        const uint64_t addedReferences = lowCount + highCount - indices.size();
                        if (m_ReferenceCount + addedReferences > m_ReferenceLimit) continue;
                        Bounds low = cell, high = cell;
                        SetComponent(low.Maximum, axis, plane);
                        SetComponent(high.Minimum, axis, plane);
                        const double cost = TraversalCost
                            + (low.Area() * lowCount + high.Area() * highCount) / parentArea;
                        if (cost < bestCost)
                        {
                            bestCost = cost;
                            bestAxis = axis;
                            bestSplit = plane;
                            bestLowCount = lowCount;
                            bestHighCount = highCount;
                        }
                    }
                }
                if (bestAxis == KdLeafAxis) return MakeLeaf(nodeIndex, indices);

                std::vector<uint32_t> low, high;
                low.reserve(bestLowCount);
                high.reserve(bestHighCount);
                for (uint32_t index : indices)
                {
                    if (PeriodicCancellation()) return false;
                    const Bounds& bounds = m_PrimitiveBounds[index];
                    if (Component(bounds.Minimum, bestAxis) <= bestSplit) low.push_back(index);
                    if (Component(bounds.Maximum, bestAxis) >= bestSplit) high.push_back(index);
                }
                m_ReferenceCount += low.size() + high.size() - indices.size();
                const uint32_t firstChild = static_cast<uint32_t>(m_Tree.Nodes.size());
                m_Tree.Nodes[nodeIndex] = {bestSplit, firstChild, 0, bestAxis};
                m_Tree.Nodes.resize(m_Tree.Nodes.size() + 2);
                Bounds lowCell = cell, highCell = cell;
                SetComponent(lowCell.Maximum, bestAxis, bestSplit);
                SetComponent(highCell.Minimum, bestAxis, bestSplit);
                return BuildNode(firstChild, lowCell, low, depth + 1)
                    && BuildNode(firstChild + 1, highCell, high, depth + 1);
            }

            const SceneGeometry& m_Scene;
            const std::function<bool()>& m_Cancelled;
            KdTreeData& m_Tree;
            std::string& m_Error;
            std::vector<Bounds> m_PrimitiveBounds;
            uint64_t m_ReferenceCount = 0;
            uint64_t m_ReferenceLimit = 0;
            uint64_t m_NodeLimit = 0;
            size_t m_Work = 0;
        };
    }

    bool BuildKdTree(const SceneGeometry& scene, const std::function<bool()>& cancelled,
        KdTreeData& tree, std::string& error)
    {
        return Builder(scene, cancelled, tree, error).Run();
    }
}
