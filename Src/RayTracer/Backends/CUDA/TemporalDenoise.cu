// Independent CUDA temporal radiance reconstruction. Geometry-validated
// reprojection and neighborhood clipping bound stale history; frame age is
// neither a transport sample count nor an unbiased variance estimate.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
namespace
{
struct alignas(16) TemporalControls
{
    unsigned Width, Height, Begin, Count, UseHistory, MaximumAge, Gamma, Reserved;
    float CurrentWeight, DepthAbsolute, DepthRelative, NormalCosine;
    float MinimumWeight, Expansion, ReservedScale[2];
    float EyeX, EyeY, EyeZ, Fov, TargetX, TargetY, TargetZ, ReservedTarget;
    float UpX, UpY, UpZ, ReservedUp;
};
struct TemporalColor
{
    float3 Rgb;
    unsigned Age;
};
struct TemporalGuide
{
    float3 Point;
    float Depth;
    float3 Normal;
    float Distance;
    float3 Direction;
    unsigned Flags;
    unsigned Primitive, Material, Reserved[2];
};
static_assert(sizeof(TemporalControls) == 112 && sizeof(TemporalColor) == 16 && sizeof(TemporalGuide) == 64);
constexpr unsigned Surface = 1, Background = 2, BadWindow = 1, BadColor = 2, MaximumExtent = 16384,
                   MaximumAge = 256;
constexpr unsigned TapCount = 4, BilinearSide = 2;
constexpr int NeighborhoodRadius = 1;
constexpr float Pi = 3.14159265358979323846f, HalfTurn = 180, Center = 0.5f;
__device__ float3 Plus(float3 a, float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ float3 Minus(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ float3 Times(float3 a, float s)
{
    return make_float3(a.x * s, a.y * s, a.z * s);
}
__device__ float DotTemporal(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ float3 CrossTemporal(float3 a, float3 b)
{
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
__device__ float Largest(float3 v)
{
    return fmaxf(v.x, fmaxf(v.y, v.z));
}
__device__ float Magnitude(float3 v)
{
    return fmaxf(fabsf(v.x), fmaxf(fabsf(v.y), fabsf(v.z)));
}
__device__ float ClampSigned(float value)
{
    return fminf(1.0f, fmaxf(-1.0f, value));
}
__device__ float3 UnitTemporal(float3 a)
{
    const float peak = fmaxf(fabsf(a.x), fmaxf(fabsf(a.y), fabsf(a.z)));
    if (!(peak > 0))
        return make_float3(0, 0, 0);
    a = make_float3(a.x / peak, a.y / peak, a.z / peak);
    return Times(a, 1 / sqrtf(DotTemporal(a, a)));
}
__device__ bool FiniteTemporal(float3 a)
{
    return isfinite(a.x) && isfinite(a.y) && isfinite(a.z);
}
__device__ float ClampUnit(float x)
{
    return fminf(1, fmaxf(0, x));
}
__device__ bool SameSurface(TemporalGuide a, TemporalGuide b, float depth, TemporalControls c)
{
    if (a.Flags & Background)
        return (b.Flags & Background) != 0;
    if (!(b.Flags & Surface) || a.Flags != b.Flags || a.Primitive != b.Primitive ||
        a.Material != b.Material || DotTemporal(a.Normal, b.Normal) < c.NormalCosine)
        return false;
    const float tolerance = c.DepthAbsolute + c.DepthRelative * fmaxf(depth, b.Depth);
    const float3 difference = Minus(b.Point, a.Point);
    const float plane =
        fmaxf(fabsf(DotTemporal(difference, a.Normal)), fabsf(DotTemporal(difference, b.Normal)));
    return isfinite(plane) && fabsf(b.Depth - depth) <= tolerance && plane <= tolerance;
}
__device__ unsigned EncodeTemporal(float3 value, unsigned gamma)
{
    constexpr float SrgbKnee = 0.0031308f, LinearSlope = 12.92f, PowerScale = 1.055f, Offset = 0.055f;
    constexpr float InverseGamma = 1.0f / 2.4f, ByteMaximum = 255, RoundToNearest = 0.5f;
    constexpr unsigned GreenShift = 8, BlueShift = 16, Opaque = 0xff000000u;
    const float scale = fmaxf(1, Largest(value));
    float channels[3] = {value.x / scale, value.y / scale, value.z / scale};
    unsigned bytes[3];
    for (unsigned channel = 0; channel < 3; ++channel)
    {
        float v = ClampUnit(channels[channel]);
        if (gamma)
            v = v <= SrgbKnee ? LinearSlope * v : PowerScale * powf(v, InverseGamma) - Offset;
        bytes[channel] = unsigned(v * ByteMaximum + RoundToNearest);
    }
    return bytes[0] | (bytes[1] << GreenShift) | (bytes[2] << BlueShift) | Opaque;
}
} // namespace
extern "C" __global__ void ArvoTemporalDenoise(const TemporalColor* current, const TemporalGuide* geometry,
                                               const TemporalColor* previous,
                                               const TemporalGuide* oldGeometry, TemporalColor* output,
                                               unsigned* display, unsigned* errors, TemporalControls c)
{
    const unsigned lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= c.Count)
        return;
    if (!c.Width || !c.Height || c.Width > MaximumExtent || c.Height > MaximumExtent ||
        c.Begin > c.Width * c.Height || c.Count > c.Width * c.Height - c.Begin || !c.MaximumAge ||
        c.MaximumAge > MaximumAge)
    {
        atomicOr(errors, BadWindow);
        return;
    }
    const unsigned index = c.Begin + lane, x = index % c.Width, y = index / c.Width;
    const TemporalGuide guide = geometry[index];
    const float3 fresh = current[index].Rgb;
    TemporalColor result{fresh, 1};
    const float3 forward =
        UnitTemporal(make_float3(c.TargetX - c.EyeX, c.TargetY - c.EyeY, c.TargetZ - c.EyeZ));
    const float3 right = UnitTemporal(CrossTemporal(forward, make_float3(c.UpX, c.UpY, c.UpZ)));
    const float3 up = CrossTemporal(right, forward);
    const float3 relative =
        guide.Flags & Background ? guide.Direction : Minus(guide.Point, make_float3(c.EyeX, c.EyeY, c.EyeZ));
    const float z = DotTemporal(relative, forward);
    if (c.UseHistory && z > 0 && isfinite(z))
    {
        const float halfY = tanf(c.Fov * Pi / (2 * HalfTurn)), halfX = halfY * c.Width / c.Height;
        const float px = (DotTemporal(relative, right) / z / halfX + 1) * c.Width / 2 - Center;
        const float py = (1 - DotTemporal(relative, up) / z / halfY) * c.Height / 2 - Center;
        if (isfinite(px) && isfinite(py) && px > -1 && px < c.Width && py > -1 && py < c.Height)
        {
            const int left = int(floorf(px)), top = int(floorf(py));
            const float fx = px - left, fy = py - top;
            float weights[TapCount]{};
            float3 colors[TapCount]{};
            float weightSum = 0, historyScale = 0;
            unsigned youngest = c.MaximumAge;
            for (unsigned tap = 0; tap < TapCount; ++tap)
            {
                const unsigned dx = tap % BilinearSide, dy = tap / BilinearSide;
                const int sx = left + dx, sy = top + dy;
                const float w = (dx ? fx : 1 - fx) * (dy ? fy : 1 - fy);
                if (!(w > 0) || sx < 0 || sy < 0 || sx >= int(c.Width) || sy >= int(c.Height))
                    continue;
                const unsigned source = unsigned(sy) * c.Width + unsigned(sx);
                if (!SameSurface(guide, oldGeometry[source], z, c))
                    continue;
                weights[tap] = w;
                colors[tap] = previous[source].Rgb;
                historyScale = fmaxf(historyScale, Magnitude(colors[tap]));
                weightSum += w;
                youngest = youngest < previous[source].Age ? youngest : previous[source].Age;
            }
            if (weightSum >= c.MinimumWeight)
            {
                float3 history = make_float3(0, 0, 0);
                if (historyScale > 0)
                {
                    for (unsigned tap = 0; tap < TapCount; ++tap)
                    {
                        const float3 unit =
                            make_float3(colors[tap].x / historyScale, colors[tap].y / historyScale,
                                        colors[tap].z / historyScale);
                        history = Plus(history, Times(unit, weights[tap]));
                    }
                    history = make_float3(ClampSigned(history.x / weightSum) * historyScale,
                                          ClampSigned(history.y / weightSum) * historyScale,
                                          ClampSigned(history.z / weightSum) * historyScale);
                }
                float3 low = fresh, high = fresh;
                for (int dy = -NeighborhoodRadius; dy <= NeighborhoodRadius; ++dy)
                    for (int dx = -NeighborhoodRadius; dx <= NeighborhoodRadius; ++dx)
                    {
                        const int sx = int(x) + dx, sy = int(y) + dy;
                        if (sx < 0 || sy < 0 || sx >= int(c.Width) || sy >= int(c.Height))
                            continue;
                        const unsigned source = unsigned(sy) * c.Width + unsigned(sx);
                        if (!SameSurface(guide, geometry[source], guide.Depth, c))
                            continue;
                        const float3 value = current[source].Rgb;
                        low =
                            make_float3(fminf(low.x, value.x), fminf(low.y, value.y), fminf(low.z, value.z));
                        high = make_float3(fmaxf(high.x, value.x), fmaxf(high.y, value.y),
                                           fmaxf(high.z, value.z));
                    }
                result.Age = youngest < c.MaximumAge ? youngest + 1 : c.MaximumAge;
                const float alpha = fmaxf(c.CurrentWeight, 1.0f / result.Age);
                // A signed unit preserves spectral RGB coordinates without
                // overflowing interpolation between finite HDR endpoints.
                const float scale = fmaxf(Magnitude(low), fmaxf(Magnitude(high), Magnitude(history)));
                if (scale > 0)
                {
                    float h[3] = {history.x / scale, history.y / scale, history.z / scale};
                    const float lo[3] = {low.x / scale, low.y / scale, low.z / scale};
                    const float hi[3] = {high.x / scale, high.y / scale, high.z / scale};
                    const float now[3] = {fresh.x / scale, fresh.y / scale, fresh.z / scale};
                    for (unsigned channel = 0; channel < 3; ++channel)
                    {
                        const float spread = (hi[channel] - lo[channel]) * c.Expansion;
                        h[channel] = fminf(ClampSigned(hi[channel] + spread),
                                           fmaxf(ClampSigned(lo[channel] - spread), h[channel]));
                        h[channel] = ClampSigned(h[channel] * (1 - alpha) + now[channel] * alpha) * scale;
                    }
                    result.Rgb = make_float3(h[0], h[1], h[2]);
                }
            }
        }
    }
    if (!FiniteTemporal(result.Rgb))
    {
        atomicOr(errors, BadColor);
        return;
    }
    output[index] = result;
    display[index] = EncodeTemporal(result.Rgb, c.Gamma);
}
