// Independent Metal geometry guides. World geometry, orientation and background
// directions support later visibility-aware temporal reconstruction; a central
// camera probe alone cannot represent the footprint of a finite aperture.
#include <metal_stdlib>
using namespace metal;
struct GeometryGuideControls
{
    uint Width, Height, FullWidth, FullHeight;
    uint TileX, TileY, Begin, Count;
    packed_float3 Eye;
    float Fov;
    packed_float3 Target;
    float Minimum;
    packed_float3 Up;
    float Maximum;
    uint Reserved[4];
};
struct GeometryGuideRay
{
    float4 OriginMin, DirectionMax;
};
struct GeometryGuideHit
{
    packed_float3 Normal;
    float Distance;
    packed_float3 Color;
    uint Primitive;
    float2 Uv;
    uint Material, Flags;
    packed_float3 GeometricNormal;
    uint Reserved;
};
struct GeometryGuidePixel
{
    packed_float3 Position;
    float Depth;
    packed_float3 Normal;
    float Distance;
    packed_float3 Direction;
    uint Flags;
    uint Primitive, Material, Reserved[2];
};
static_assert(sizeof(GeometryGuideControls) == 96 && sizeof(GeometryGuideRay) == 32 &&
                  sizeof(GeometryGuideHit) == 64 && sizeof(GeometryGuidePixel) == 64,
              "Metal geometry guide ABI");
constant uint GuideMaximumExtent = 32768, GuideBadWindow = 1, GuideBadValue = 2;
constant uint GuideSurface = 1, GuideBackground = 2, GuideFront = 4, GuideNoPrimitive = 0xffffffffu;
constant float GuidePi = 3.14159265358979323846f, GuideHalfTurn = 180, GuidePixelCenter = 0.5f;
float3 GeometryGuideUnit(float3 v)
{
    const float peak = max(max(abs(v.x), abs(v.y)), abs(v.z));
    if (!(peak > 0))
        return float3(0);
    v /= peak;
    return v / sqrt(dot(v, v));
}
bool GeometryGuideWindow(constant GeometryGuideControls& c)
{
    return c.Width && c.Height && c.FullWidth && c.FullHeight && c.FullWidth <= GuideMaximumExtent &&
           c.FullHeight <= GuideMaximumExtent && c.TileX <= c.FullWidth && c.TileY <= c.FullHeight &&
           c.Width <= c.FullWidth - c.TileX && c.Height <= c.FullHeight - c.TileY &&
           c.Begin <= c.Width * c.Height && c.Count <= c.Width * c.Height - c.Begin;
}
kernel void ArvoGenerateGeometryGuideRays(device GeometryGuideRay* rays [[buffer(0)]],
                                          device atomic_uint* errors [[buffer(1)]],
                                          constant GeometryGuideControls& c [[buffer(2)]],
                                          uint lane [[thread_position_in_grid]])
{
    if (lane >= c.Count)
        return;
    if (!GeometryGuideWindow(c))
    {
        atomic_fetch_or_explicit(errors, GuideBadWindow, memory_order_relaxed);
        return;
    }
    const uint i = c.Begin + lane;
    const float3 forward = GeometryGuideUnit(float3(c.Target) - float3(c.Eye));
    const float3 right = GeometryGuideUnit(cross(forward, float3(c.Up)));
    const float3 up = cross(right, forward);
    const float extentY = tan(c.Fov * GuidePi / (2 * GuideHalfTurn));
    const float extentX = extentY * float(c.FullWidth) / float(c.FullHeight);
    const float filmX = float(i % c.Width + c.TileX) + GuidePixelCenter;
    const float filmY = float(i / c.Width + c.TileY) + GuidePixelCenter;
    const float3 direction = GeometryGuideUnit(forward + right * ((2 * filmX / c.FullWidth - 1) * extentX) +
                                               up * ((1 - 2 * filmY / c.FullHeight) * extentY));
    if (!all(isfinite(direction)) || !(dot(direction, direction) > 0))
    {
        atomic_fetch_or_explicit(errors, GuideBadValue, memory_order_relaxed);
        return;
    }
    GeometryGuideRay ray;
    ray.OriginMin = float4(float3(c.Eye), c.Minimum);
    ray.DirectionMax = float4(direction, c.Maximum);
    rays[i] = ray;
}
kernel void ArvoResolveGeometryGuideHits(device const GeometryGuideRay* rays [[buffer(0)]],
                                         device const GeometryGuideHit* hits [[buffer(1)]],
                                         device GeometryGuidePixel* guides [[buffer(2)]],
                                         device atomic_uint* errors [[buffer(3)]],
                                         constant GeometryGuideControls& c [[buffer(4)]],
                                         uint lane [[thread_position_in_grid]])
{
    if (lane >= c.Count)
        return;
    if (!GeometryGuideWindow(c))
    {
        atomic_fetch_or_explicit(errors, GuideBadWindow, memory_order_relaxed);
        return;
    }
    const uint i = c.Begin + lane;
    const GeometryGuideRay ray = rays[i];
    const GeometryGuideHit hit = hits[i];
    GeometryGuidePixel guide{};
    guide.Direction = packed_float3(ray.DirectionMax.xyz);
    guide.Flags = GuideBackground;
    guide.Primitive = GuideNoPrimitive;
    if (hit.Flags & GuideSurface)
    {
        // FMA preserves translated small surfaces without an intermediate
        // rounding of distance * direction. Metal has no native binary64 type.
        const float3 point = fma(ray.DirectionMax.xyz, float3(hit.Distance), ray.OriginMin.xyz);
        const float3 normal = GeometryGuideUnit(float3(hit.GeometricNormal));
        const float3 forward = GeometryGuideUnit(float3(c.Target) - float3(c.Eye));
        guide.Position = packed_float3(point);
        guide.Normal = packed_float3(normal);
        guide.Depth = hit.Distance * dot(ray.DirectionMax.xyz, forward);
        guide.Distance = hit.Distance;
        guide.Primitive = hit.Primitive;
        guide.Material = hit.Material;
        guide.Flags = GuideSurface | (hit.Flags & GuideFront);
        if (!all(isfinite(point)) || !all(isfinite(normal)) || !(dot(normal, normal) > 0) ||
            !isfinite(guide.Depth) || !(guide.Depth > 0) || !isfinite(hit.Distance) ||
            hit.Distance <= ray.OriginMin.w || hit.Distance >= ray.DirectionMax.w ||
            hit.Primitive == GuideNoPrimitive)
        {
            atomic_fetch_or_explicit(errors, GuideBadValue, memory_order_relaxed);
            return;
        }
    }
    guides[i] = guide;
}
