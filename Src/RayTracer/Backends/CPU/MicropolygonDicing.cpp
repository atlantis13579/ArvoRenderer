#include "RayTracer/Backends/CPU/MicropolygonDicing.h"
#include "Renderer/RayTracing.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace Arvo::RayTracing::Cpu
{
// Cook, Carpenter and Catmull, "The Reyes Image Rendering Architecture" (1987).
// This renderer borrows dicing, then traces the displaced triangles with its
// selected integrator; it does not replace transport with Reyes visibility.
// A connected component uses one power-of-two grid level. This deliberately
// trades some over-tessellation for matching edges without transition patches.
// The full-film projection estimates density; limits are reported in the plan.
// https://graphics.pixar.com/library/Reyes/
namespace
{
namespace M = MicropolygonData;
constexpr float Pi = 3.14159265358979323846f;
constexpr float ProjectionNearDistance = 1.e-4f;
constexpr float SegmentsPerWave = 8;
Float3 Vector(M::Float4 value)
{
    return {value.X, value.Y, value.Z};
}
M::Float4 Pack(Float3 value)
{
    return {value.X, value.Y, value.Z, 0};
}
Float3 Unit(Float3 value)
{
    const float square = Dot(value, value);
    return square > 0 ? value / std::sqrt(square) : Float3(0, 0, 1);
}
uint32_t Plan(const M::Patch& patch, const M::Controls& controls)
{
    Float3 low = Vector(patch.Corners[0].Position), high = low;
    for (const auto& vertex : patch.Corners)
    {
        const auto p = Vector(vertex.Position);
        low = {std::min(low.X, p.X), std::min(low.Y, p.Y), std::min(low.Z, p.Z)};
        high = {std::max(high.X, p.X), std::max(high.Y, p.Y), std::max(high.Z, p.Z)};
    }
    const bool displaced = (patch.Metadata.Y & M::Displace) != 0;
    const float amplitude = displaced ? std::abs(controls.CameraUpAmplitude.W) : 0;
    low.Y -= amplitude;
    high.Y += amplitude;
    Float2 screenLow{std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()};
    Float2 screenHigh{-screenLow.X, -screenLow.Y};
    float nearest = std::numeric_limits<float>::infinity(), farthest = -nearest;
    for (uint32_t corner = 0; corner < (1u << 3); ++corner)
    {
        const Float3 position{corner & 1 ? high.X : low.X, corner & 2 ? high.Y : low.Y,
                              corner & 4 ? high.Z : low.Z};
        const Float3 offset = position - Vector(controls.CameraPositionTarget);
        const float depth = Dot(offset, Vector(controls.CameraForwardHalfHeight));
        nearest = std::min(nearest, depth);
        farthest = std::max(farthest, depth);
        if (depth > ProjectionNearDistance)
        {
            const float x = Dot(offset, Vector(controls.CameraRightHalfWidth)) /
                            (depth * controls.CameraRightHalfWidth.W);
            const float y = Dot(offset, Vector(controls.CameraUpAmplitude)) /
                            (depth * controls.CameraForwardHalfHeight.W);
            screenLow = {std::min(screenLow.X, x), std::min(screenLow.Y, y)};
            screenHigh = {std::max(screenHigh.X, x), std::max(screenHigh.Y, y)};
        }
    }
    float pixels = 0;
    // Clipping the footprint to the viewport would under-dice a large patch
    // whose visible portion occupies only a small part of its parameter domain.
    if (farthest > ProjectionNearDistance && nearest <= ProjectionNearDistance)
        return controls.Extent.Z | M::LevelClamped;
    if (farthest > ProjectionNearDistance)
    {
        Float2 extent{float(controls.Extent.X), float(controls.Extent.Y)};
        if (nearest > ProjectionNearDistance)
        {
            extent.X *= (screenHigh.X - screenLow.X) / 2;
            extent.Y *= (screenHigh.Y - screenLow.Y) / 2;
        }
        pixels = std::sqrt(extent.X * extent.X + extent.Y * extent.Y);
    }
    float required = std::max(1.f, pixels / controls.CameraPositionTarget.W);
    if (displaced && amplitude > 0)
        required = std::max(required, SegmentsPerWave * (high.X - low.X) / controls.Wave.X);
    uint32_t level = 0;
    while (level < controls.Extent.Z && float(1u << level) < required)
        ++level;
    return level | (float(1u << level) < required ? M::LevelClamped : 0);
}
M::Float4 Blend(M::Float4 a, M::Float4 b, M::Float4 c, Float3 weights)
{
    return {a.X * weights.X + b.X * weights.Y + c.X * weights.Z,
            a.Y * weights.X + b.Y * weights.Y + c.Y * weights.Z,
            a.Z * weights.X + b.Z * weights.Y + c.Z * weights.Z,
            a.W * weights.X + b.W * weights.Y + c.W * weights.Z};
}
M::Vertex Vertex(const M::Patch& patch, uint32_t i, uint32_t j, uint32_t segments,
                 const M::Controls& controls)
{
    const uint32_t weights[]{segments - i - j, i, j};
    const Float3 barycentric{float(weights[0]) / segments, float(weights[1]) / segments,
                             float(weights[2]) / segments};
    const uint32_t ids[]{patch.CornerIds.X, patch.CornerIds.Y, patch.CornerIds.Z};
    Float3 position;
    bool edge = false;
    for (uint32_t corner = 0; corner < std::size(weights); ++corner)
        if (weights[corner] == 0)
        {
            uint32_t first = (corner + 1) % 3, second = (corner + 2) % 3;
            if (ids[first] > ids[second])
                std::swap(first, second);
            const auto a = Vector(patch.Corners[first].Position), b = Vector(patch.Corners[second].Position);
            // Endpoints retain their exact bits; both adjacent patches use the
            // same ordered interpolation for all other shared-edge samples.
            position = weights[first] == 0    ? b
                       : weights[second] == 0 ? a
                                              : a + (b - a) * (float(weights[second]) / segments);
            edge = true;
            break;
        }
    if (!edge)
    {
        const auto a = Vector(patch.Corners[0].Position);
        position = a + (Vector(patch.Corners[1].Position) - a) * barycentric.Y +
                   (Vector(patch.Corners[2].Position) - a) * barycentric.Z;
    }
    Float3 normal = patch.Metadata.Y & M::SmoothNormals
                        ? Vector(Blend(patch.Corners[0].Normal, patch.Corners[1].Normal,
                                       patch.Corners[2].Normal, barycentric))
                        : Cross(Vector(patch.Corners[1].Position) - Vector(patch.Corners[0].Position),
                                Vector(patch.Corners[2].Position) - Vector(patch.Corners[0].Position));
    if ((patch.Metadata.Y & M::Displace) != 0 && controls.CameraUpAmplitude.W != 0)
    {
        // An analytic Y shear is continuous across material/UV seams and has
        // determinant one. The inverse-transpose Jacobian transforms normals.
        const float cycles = position.X / controls.Wave.X + controls.Wave.Y;
        const float angle = (cycles - std::floor(cycles)) * (2 * Pi);
        const float slope = controls.CameraUpAmplitude.W * (2 * Pi / controls.Wave.X) * std::cos(angle);
        position.Y += controls.CameraUpAmplitude.W * std::sin(angle);
        normal.X -= slope * normal.Y;
    }
    return {
        Pack(position), Pack(Unit(normal)),
        Blend(patch.Corners[0].Color, patch.Corners[1].Color, patch.Corners[2].Color, barycentric),
        Blend(patch.Corners[0].TexCoord, patch.Corners[1].TexCoord, patch.Corners[2].TexCoord, barycentric)};
}
} // namespace
bool MicropolygonPass(const M::Controls& controls, const std::vector<M::Patch>& patches,
                      const std::vector<M::Range>& ranges, std::vector<uint32_t>& output,
                      const std::function<bool()>& cancelled)
{
    const uint32_t end = controls.Dispatch.X + controls.Dispatch.Y;
    if (controls.Populations.W == M::PlanPass)
    {
        for (uint32_t patch = controls.Dispatch.X; patch < end; ++patch)
        {
            if (cancelled && cancelled())
                return false;
            auto& word = output[patches[patch].Metadata.Z];
            word = std::max(word, Plan(patches[patch], controls));
        }
        return true;
    }
    size_t patch =
        std::upper_bound(ranges.begin(), ranges.end(), controls.Dispatch.X,
                         [](uint32_t triangle, const M::Range& range) { return triangle < range.End; }) -
        ranges.begin();
    for (uint32_t triangle = controls.Dispatch.X; triangle < end; ++triangle)
    {
        if (cancelled && cancelled())
            return false;
        while (triangle >= ranges[patch].End)
            ++patch;
        const auto& range = ranges[patch];
        const uint32_t local = triangle - range.First, n = range.Segments;
        uint32_t row = 0, remaining = local;
        while (remaining >= 2 * (n - row) - 1)
        {
            remaining -= 2 * (n - row) - 1;
            ++row;
        }
        const uint32_t column = remaining / 2;
        M::OutputTriangle result;
        if ((remaining & 1) == 0)
            result.Corners = {Vertex(patches[patch], row, column, n, controls),
                              Vertex(patches[patch], row + 1, column, n, controls),
                              Vertex(patches[patch], row, column + 1, n, controls)};
        else
            result.Corners = {Vertex(patches[patch], row + 1, column, n, controls),
                              Vertex(patches[patch], row + 1, column + 1, n, controls),
                              Vertex(patches[patch], row, column + 1, n, controls)};
        std::memcpy(output.data() + size_t(triangle) * M::OutputTriangleWords, &result, sizeof(result));
    }
    return true;
}
} // namespace Arvo::RayTracing::Cpu
