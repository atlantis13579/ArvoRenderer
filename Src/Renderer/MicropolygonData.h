#pragma once
#include "Renderer/VectorTypes.h"
#include <array>
#include <cstddef>
#include <cstdint>

namespace Arvo::RayTracing::MicropolygonData
{
// Wire/layout contract only. Each execution backend owns its projection,
// triangle-grid enumeration, interpolation and displacement implementation.
// Positions are canonical reference-pose world coordinates. Shared corners
// carry identical position bits and IDs even across normal/UV/material seams.
inline constexpr uint32_t SmoothNormals = 1, Displace = 2;
inline constexpr uint32_t PlanPass = 0, EmitPass = 1;
inline constexpr uint32_t LevelMask = 0xff, LevelClamped = 1u << 8;
inline constexpr uint32_t MaximumLevel = 8;
inline constexpr uint32_t MaximumOutputTriangles = 1u << 21;
struct Float4
{
    float X = 0, Y = 0, Z = 0, W = 0;
};
struct UInt4
{
    uint32_t X = 0, Y = 0, Z = 0, W = 0;
};
struct Vertex
{
    Float4 Position, Normal, Color, TexCoord;
};
struct Patch
{
    std::array<Vertex, 3> Corners;
    UInt4 Metadata;  // Material index, flags, connected component, source group.
    UInt4 CornerIds; // Canonical position IDs; W is reserved.
};
struct Range
{
    uint32_t First = 0, End = 0, Segments = 1, PlanWord = 0;
};
struct OutputTriangle
{
    std::array<Vertex, 3> Corners;
};
struct Controls
{
    Float4 CameraPositionTarget; // XYZ camera position, W target edge pixels.
    Float4 CameraForwardHalfHeight;
    Float4 CameraRightHalfWidth;
    Float4 CameraUpAmplitude;
    Float4 Wave;       // Wavelength, phase in cycles, reserved, reserved.
    UInt4 Populations; // Patches, components, output triangles, pass.
    UInt4 Extent;      // Full-film width, height, maximum subdivision level, reserved.
    UInt4 Dispatch;    // Begin, count, reserved, reserved.
};
static_assert(sizeof(Vertex) == 64 && sizeof(Patch) == 224);
static_assert(offsetof(Patch, Metadata) == 192 && offsetof(Patch, CornerIds) == 208);
static_assert(sizeof(Range) == 16 && sizeof(OutputTriangle) == 192 && sizeof(Controls) == 128);
inline constexpr uint32_t OutputTriangleWords = sizeof(OutputTriangle) / sizeof(uint32_t);
} // namespace Arvo::RayTracing::MicropolygonData
