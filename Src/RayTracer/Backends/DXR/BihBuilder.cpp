#include "BihBuilder.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

// Wachter and Keller (2006), "Instant Ray Tracing: The Bounding Interval
// Hierarchy". Object partitioning gives two independent child clipping planes;
// they can overlap or leave a gap. Every primitive belongs to exactly one leaf.
// https://doi.org/10.2312/EGWR/EGSR06/139-149
// This DXR-owned implementation uses midpoint partitions and a deterministic
// centroid median when a midpoint produces an empty child. It stores explicit
// fields instead of the paper's compact pointer tagging.
namespace Arvo::RayTracing::Dxr
{
    namespace
    {
        constexpr uint32_t BihLeafPrimitiveLimit = 8;
        constexpr uint32_t BihCancellationPeriod = 256;
        constexpr uint64_t MaximumByteAddressedNodes =
            uint64_t(std::numeric_limits<uint32_t>::max()) / sizeof(BihNode);

        float Coordinate(const Float3& point, uint32_t axis)
        {
            switch (axis)
            {
            case 0: return point.X;
            case 1: return point.Y;
            default: return point.Z;
            }
        }

        void SetCoordinate(Float3& point, uint32_t axis, float coordinate)
        {
            switch (axis)
            {
            case 0: point.X = coordinate; break;
            case 1: point.Y = coordinate; break;
            default: point.Z = coordinate; break;
            }
        }

        struct Box
        {
            Float3 Low{std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
                std::numeric_limits<float>::max()};
            Float3 High{std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(),
                std::numeric_limits<float>::lowest()};

            void Add(const Float3& point)
            {
                Low = {std::min(Low.X, point.X), std::min(Low.Y, point.Y), std::min(Low.Z, point.Z)};
                High = {std::max(High.X, point.X), std::max(High.Y, point.Y), std::max(High.Z, point.Z)};
            }

            void Add(const Box& other)
            {
                Add(other.Low);
                Add(other.High);
            }

            double Center(uint32_t axis) const
            {
                return (double(Coordinate(Low, axis)) + Coordinate(High, axis)) * 0.5;
            }

            uint32_t LongestAxis() const
            {
                uint32_t selected = 0;
                double largest = double(High.X) - Low.X;
                for (uint32_t axis = 1; axis < 3; ++axis)
                {
                    const double extent = double(Coordinate(High, axis)) - Coordinate(Low, axis);
                    if (extent > largest) { selected = axis; largest = extent; }
                }
                return selected;
            }

            void RoundOutwards()
            {
                for (uint32_t axis = 0; axis < 3; ++axis)
                {
                    SetCoordinate(Low, axis, std::nextafter(Coordinate(Low, axis),
                        std::numeric_limits<float>::lowest()));
                    SetCoordinate(High, axis, std::nextafter(Coordinate(High, axis),
                        std::numeric_limits<float>::max()));
                }
            }
        };

        struct CancelledBuild {};

        class Construction
        {
        public:
            Construction(const SceneGeometry& scene, const std::function<bool()>& cancelled, BihData& hierarchy)
                : m_Source(scene), m_Cancelled(cancelled), m_Result(hierarchy) {}

            void Build()
            {
                CheckCancellation();
                m_Result = {};
                const size_t count = m_Source.Triangles.size() + m_Source.Spheres.size();
                m_Boxes.reserve(count);
                Box root;
                for (const Triangle& triangle : m_Source.Triangles)
                {
                    PollCancellation();
                    Box object;
                    object.Add(triangle.Position0);
                    object.Add(triangle.Position1);
                    object.Add(triangle.Position2);
                    m_Boxes.push_back(object);
                    root.Add(object);
                }
                for (const Sphere& sphere : m_Source.Spheres)
                {
                    PollCancellation();
                    const Float3 radius(sphere.Radius, sphere.Radius, sphere.Radius);
                    Box object;
                    object.Add(sphere.Center - radius);
                    object.Add(sphere.Center + radius);
                    object.RoundOutwards();
                    m_Boxes.push_back(object);
                    root.Add(object);
                }
                if (count == 0) return;
                root.RoundOutwards();
                m_Result.Minimum = root.Low;
                m_Result.Maximum = root.High;
                m_Result.PrimitiveIndices.resize(count);
                std::iota(m_Result.PrimitiveIndices.begin(), m_Result.PrimitiveIndices.end(), 0u);
                // A binary object hierarchy with nonempty children needs at most
                // 2*N-1 nodes. Raw HLSL loads also need 32-bit byte offsets;
                // retaining a leaf at that budget preserves all primitives.
                m_NodeLimit = std::min(uint64_t(count) * 2 - 1, MaximumByteAddressedNodes);
                m_Result.Nodes.emplace_back();
                Subdivide(0, 0, static_cast<uint32_t>(count), root, 0);
            }

        private:
            void CheckCancellation() const
            {
                if (m_Cancelled && m_Cancelled()) throw CancelledBuild{};
            }

            void PollCancellation()
            {
                if (++m_Work % BihCancellationPeriod == 0) CheckCancellation();
            }

            uint32_t Partition(uint32_t begin, uint32_t end, uint32_t axis, double middle)
            {
                uint32_t low = begin, high = end;
                while (low < high)
                {
                    PollCancellation();
                    if (m_Boxes[m_Result.PrimitiveIndices[low]].Center(axis) < middle) ++low;
                    else std::swap(m_Result.PrimitiveIndices[low], m_Result.PrimitiveIndices[--high]);
                }
                if (low != begin && low != end) return low;

                const uint32_t median = begin + (end - begin) / 2;
                auto first = m_Result.PrimitiveIndices.begin();
                std::nth_element(first + begin, first + median, first + end,
                    [this, axis](uint32_t a, uint32_t b)
                    {
                        PollCancellation();
                        const double ca = m_Boxes[a].Center(axis), cb = m_Boxes[b].Center(axis);
                        return ca < cb || (ca == cb && a < b);
                    });
                return median;
            }

            Box RangeBounds(uint32_t begin, uint32_t end)
            {
                Box bounds;
                for (uint32_t index = begin; index < end; ++index)
                {
                    PollCancellation();
                    bounds.Add(m_Boxes[m_Result.PrimitiveIndices[index]]);
                }
                return bounds;
            }

            void Subdivide(uint32_t nodeIndex, uint32_t begin, uint32_t end, const Box& cell, uint32_t depth)
            {
                CheckCancellation();
                const uint32_t count = end - begin;
                if (count <= BihLeafPrimitiveLimit || depth == BihMaximumDepth
                    || m_Result.Nodes.size() + 2 > m_NodeLimit)
                {
                    BihNode& leaf = m_Result.Nodes[nodeIndex];
                    leaf.First = begin;
                    leaf.Count = count;
                    return;
                }

                const uint32_t axis = cell.LongestAxis();
                const uint32_t middle = Partition(begin, end, axis, cell.Center(axis));
                const Box leftObjects = RangeBounds(begin, middle);
                const Box rightObjects = RangeBounds(middle, end);
                const float leftMaximum = std::min(Coordinate(cell.High, axis),
                    std::nextafter(Coordinate(leftObjects.High, axis), std::numeric_limits<float>::max()));
                const float rightMinimum = std::max(Coordinate(cell.Low, axis),
                    std::nextafter(Coordinate(rightObjects.Low, axis), std::numeric_limits<float>::lowest()));
                const uint32_t firstChild = static_cast<uint32_t>(m_Result.Nodes.size());
                m_Result.Nodes.resize(m_Result.Nodes.size() + 2);
                BihNode& node = m_Result.Nodes[nodeIndex];
                node.LeftMaximum = leftMaximum;
                node.RightMinimum = rightMinimum;
                node.First = firstChild;
                node.Axis = axis;

                Box leftCell = cell, rightCell = cell;
                SetCoordinate(leftCell.High, axis, leftMaximum);
                SetCoordinate(rightCell.Low, axis, rightMinimum);
                Subdivide(firstChild, begin, middle, leftCell, depth + 1);
                Subdivide(firstChild + 1, middle, end, rightCell, depth + 1);
            }

            const SceneGeometry& m_Source;
            const std::function<bool()>& m_Cancelled;
            BihData& m_Result;
            std::vector<Box> m_Boxes;
            uint64_t m_NodeLimit = 0;
            uint32_t m_Work = 0;
        };
    }

    bool BuildBih(const SceneGeometry& scene, const std::function<bool()>& cancelled,
        BihData& hierarchy, std::string& error)
    {
        if (scene.Triangles.size() >= InvalidRayPrimitiveIndex
            || scene.Spheres.size() >= InvalidRayPrimitiveIndex - scene.Triangles.size())
        {
            error = "DXR BIH exceeds 32-bit primitive indexing.";
            return false;
        }
        try
        {
            Construction(scene, cancelled, hierarchy).Build();
            return true;
        }
        catch (const CancelledBuild&)
        {
            hierarchy = {};
            error = "DXR BIH construction was cancelled.";
            return false;
        }
    }
}
