// Independent CUDA geometric correspondence probes. No transport estimator is
// reused: generation owns central camera rays, resolve consumes native hits.
// Schied et al., "Spatiotemporal Variance-Guided Filtering" (HPG 2017), motivates
// the geometric guides; these kernels alone do not implement temporal filtering.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

namespace
{
constexpr float Pi = 3.14159265358979323846f, HalfTurnDegrees = 180, PixelMidpoint = 0.5f;
constexpr unsigned MaximumExtent = 16384, InvalidConfiguration = 1, InvalidGeometry = 2;
constexpr unsigned Surface = 1, Background = 2, Front = 4, InvalidPrimitive = 0xffffffffu;
struct alignas(16) GuideControls
{
    unsigned Width, Height, FullWidth, FullHeight, TileX, TileY, Begin, Count;
    float Px, Py, Pz, Fov, Tx, Ty, Tz, Minimum, Ux, Uy, Uz, Maximum;
    unsigned Reserved[4];
};
struct GuideRay
{
    float4 OriginMin, DirectionMax;
};
struct GuideHit
{
    float3 Normal;
    float Distance;
    float3 Color;
    unsigned Primitive;
    float2 Uv;
    unsigned Material, Flags;
    float3 GeometryNormal;
    unsigned Reserved;
};
struct GuidePixel
{
    float3 Position;
    float Depth;
    float3 Normal;
    float Distance;
    float3 Direction;
    unsigned Flags;
    unsigned Primitive, Material, Reserved[2];
};
static_assert(sizeof(GuideControls) == 96 && sizeof(GuideRay) == 32 && sizeof(GuideHit) == 64 &&
              sizeof(GuidePixel) == 64);
__device__ float3 Unit(float3 v)
{
    const float scale = fmaxf(fabsf(v.x), fmaxf(fabsf(v.y), fabsf(v.z)));
    if (!(scale > 0))
        return make_float3(0, 0, 0);
    v = make_float3(v.x / scale, v.y / scale, v.z / scale);
    const float inverse = 1 / sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    return make_float3(v.x * inverse, v.y * inverse, v.z * inverse);
}
__device__ float3 CrossGuide(float3 a, float3 b)
{
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
__device__ float3 ForwardGuide(GuideControls c)
{
    return Unit(make_float3(c.Tx - c.Px, c.Ty - c.Py, c.Tz - c.Pz));
}
__device__ bool GoodVector(float3 v)
{
    return isfinite(v.x) && isfinite(v.y) && isfinite(v.z);
}
__device__ bool GoodDispatch(GuideControls c)
{
    return c.Width && c.Height && c.FullWidth && c.FullHeight && c.FullWidth <= MaximumExtent &&
           c.FullHeight <= MaximumExtent && c.TileX <= c.FullWidth && c.Width <= c.FullWidth - c.TileX &&
           c.TileY <= c.FullHeight && c.Height <= c.FullHeight - c.TileY && c.Begin <= c.Width * c.Height &&
           c.Count <= c.Width * c.Height - c.Begin;
}
} // namespace
extern "C" __global__ void ArvoGenerateGeometryGuideRays(GuideRay* rays, unsigned* errors,
                                                         GuideControls controls)
{
    const unsigned lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= controls.Count)
        return;
    if (!GoodDispatch(controls))
    {
        atomicOr(errors, InvalidConfiguration);
        return;
    }
    const unsigned index = controls.Begin + lane;
    const float3 forward = ForwardGuide(controls);
    const float3 right = Unit(CrossGuide(forward, make_float3(controls.Ux, controls.Uy, controls.Uz)));
    const float3 up = CrossGuide(right, forward);
    const float halfHeight = tanf(controls.Fov * Pi / (2 * HalfTurnDegrees));
    const float halfWidth = halfHeight * float(controls.FullWidth) / float(controls.FullHeight);
    const float x = float(index % controls.Width + controls.TileX) + PixelMidpoint;
    const float y = float(index / controls.Width + controls.TileY) + PixelMidpoint;
    const float sx = (2 * x / controls.FullWidth - 1) * halfWidth;
    const float sy = (1 - 2 * y / controls.FullHeight) * halfHeight;
    const float3 direction =
        Unit(make_float3(forward.x + sx * right.x + sy * up.x, forward.y + sx * right.y + sy * up.y,
                         forward.z + sx * right.z + sy * up.z));
    if (!GoodVector(direction) ||
        !(direction.x * direction.x + direction.y * direction.y + direction.z * direction.z > 0))
    {
        atomicOr(errors, InvalidGeometry);
        return;
    }
    rays[index] = {make_float4(controls.Px, controls.Py, controls.Pz, controls.Minimum),
                   make_float4(direction.x, direction.y, direction.z, controls.Maximum)};
}
extern "C" __global__ void ArvoResolveGeometryGuideHits(const GuideRay* rays, const GuideHit* hits,
                                                        GuidePixel* guides, unsigned* errors,
                                                        GuideControls controls)
{
    const unsigned lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= controls.Count)
        return;
    if (!GoodDispatch(controls))
    {
        atomicOr(errors, InvalidConfiguration);
        return;
    }
    const unsigned index = controls.Begin + lane;
    const GuideRay ray = rays[index];
    const GuideHit hit = hits[index];
    GuidePixel pixel{};
    pixel.Direction = make_float3(ray.DirectionMax.x, ray.DirectionMax.y, ray.DirectionMax.z);
    pixel.Flags = Background;
    pixel.Primitive = InvalidPrimitive;
    if (hit.Flags & Surface)
    {
        const float3 forward = ForwardGuide(controls);
        pixel.Position =
            make_float3(float(double(ray.OriginMin.x) + double(ray.DirectionMax.x) * hit.Distance),
                        float(double(ray.OriginMin.y) + double(ray.DirectionMax.y) * hit.Distance),
                        float(double(ray.OriginMin.z) + double(ray.DirectionMax.z) * hit.Distance));
        pixel.Normal = Unit(hit.GeometryNormal);
        pixel.Depth = hit.Distance * (pixel.Direction.x * forward.x + pixel.Direction.y * forward.y +
                                      pixel.Direction.z * forward.z);
        pixel.Distance = hit.Distance;
        pixel.Primitive = hit.Primitive;
        pixel.Material = hit.Material;
        pixel.Flags = Surface | (hit.Flags & Front);
        if (!GoodVector(pixel.Position) || !GoodVector(pixel.Normal) ||
            !(pixel.Normal.x * pixel.Normal.x + pixel.Normal.y * pixel.Normal.y +
                  pixel.Normal.z * pixel.Normal.z >
              0) ||
            !isfinite(pixel.Depth) || !(pixel.Depth > 0) || !isfinite(hit.Distance) ||
            hit.Distance <= ray.OriginMin.w || hit.Distance >= ray.DirectionMax.w ||
            hit.Primitive == InvalidPrimitive)
        {
            atomicOr(errors, InvalidGeometry);
            return;
        }
    }
    guides[index] = pixel;
}
