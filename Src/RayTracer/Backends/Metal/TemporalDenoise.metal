// Independent Metal temporal radiance reconstruction. Bilinear reprojection
// validates geometry per history tap; clipping uses compatible current pixels.
// RGB normalization avoids overflow at finite HDR endpoints without binary64.
#include <metal_stdlib>
using namespace metal;
struct TemporalControls
{
    uint Width, Height, Begin, Count;
    uint HistoryAvailable, MaximumAge, GammaCorrection, Reserved;
    float CurrentWeight, AbsoluteDepth, RelativeDepth, NormalCosine;
    float MinimumSupport, Expansion, ReservedScale[2];
    packed_float3 Eye;
    float FieldOfView;
    packed_float3 Target;
    float ReservedTarget;
    packed_float3 Up;
    float ReservedUp;
};
struct TemporalPixel
{
    packed_float3 Color;
    uint Age;
};
struct TemporalGuide
{
    packed_float3 Position;
    float Depth;
    packed_float3 Normal;
    float Distance;
    packed_float3 Direction;
    uint Flags;
    uint Primitive, Material, Reserved[2];
};
static_assert(sizeof(TemporalControls) == 112 && sizeof(TemporalPixel) == 16 && sizeof(TemporalGuide) == 64,
              "Metal temporal upload ABI");
constant uint TemporalSurface = 1, TemporalSky = 2, TemporalBadWindow = 1, TemporalBadColor = 2;
constant uint TemporalExtentLimit = 16384, TemporalHistoryLimit = 256, TemporalTapCount = 4;
constant int TemporalRadius = 1, TemporalBilinearSide = 2;
constant float TemporalPi = 3.14159265358979323846f, TemporalHalfTurn = 180, TemporalPixelCenter = 0.5f;
float3 TemporalNormalize(float3 v)
{
    const float magnitude = max(max(abs(v.x), abs(v.y)), abs(v.z));
    if (!(magnitude > 0))
        return float3(0);
    v /= magnitude;
    return v / sqrt(dot(v, v));
}
float TemporalPeak(float3 v)
{
    return max(v.x, max(v.y, v.z));
}
bool TemporalCompatible(TemporalGuide a, TemporalGuide b, float expectedDepth, constant TemporalControls& c)
{
    if (a.Flags & TemporalSky)
        return (b.Flags & TemporalSky) != 0;
    if (!(b.Flags & TemporalSurface) || a.Flags != b.Flags || a.Primitive != b.Primitive ||
        a.Material != b.Material || dot(float3(a.Normal), float3(b.Normal)) < c.NormalCosine)
        return false;
    const float tolerance = c.AbsoluteDepth + c.RelativeDepth * max(expectedDepth, b.Depth);
    const float3 offset = float3(b.Position) - float3(a.Position);
    const float plane = max(abs(dot(offset, float3(a.Normal))), abs(dot(offset, float3(b.Normal))));
    return isfinite(plane) && plane <= tolerance && abs(expectedDepth - b.Depth) <= tolerance;
}
uint TemporalDisplay(float3 color, uint gamma)
{
    constexpr float LinearKnee = 0.0031308f, LinearScale = 12.92f, PowerScale = 1.055f, Offset = 0.055f;
    constexpr float InverseGamma = 1.0f / 2.4f, ByteMaximum = 255, Nearest = 0.5f;
    constexpr uint GreenShift = 8, BlueShift = 16, Alpha = 0xff000000u;
    color = clamp(color / max(1.0f, TemporalPeak(color)), 0.0f, 1.0f);
    for (uint channel = 0; channel < 3; ++channel)
        if (gamma)
            color[channel] = color[channel] <= LinearKnee
                                 ? color[channel] * LinearScale
                                 : PowerScale * pow(color[channel], InverseGamma) - Offset;
    const uint red = uint(color.x * ByteMaximum + Nearest), green = uint(color.y * ByteMaximum + Nearest),
               blue = uint(color.z * ByteMaximum + Nearest);
    return red | (green << GreenShift) | (blue << BlueShift) | Alpha;
}
kernel void ArvoTemporalDenoise(device const TemporalPixel* current [[buffer(0)]],
                                device const TemporalGuide* guides [[buffer(1)]],
                                device const TemporalPixel* previous [[buffer(2)]],
                                device const TemporalGuide* oldGuides [[buffer(3)]],
                                device TemporalPixel* output [[buffer(4)]],
                                device uint* display [[buffer(5)]], device atomic_uint* errors [[buffer(6)]],
                                constant TemporalControls& c [[buffer(7)]],
                                uint lane [[thread_position_in_grid]])
{
    if (lane >= c.Count)
        return;
    if (!c.Width || !c.Height || c.Width > TemporalExtentLimit || c.Height > TemporalExtentLimit ||
        c.Begin > c.Width * c.Height || c.Count > c.Width * c.Height - c.Begin || !c.MaximumAge ||
        c.MaximumAge > TemporalHistoryLimit)
    {
        atomic_fetch_or_explicit(errors, TemporalBadWindow, memory_order_relaxed);
        return;
    }
    const uint index = c.Begin + lane;
    const TemporalGuide center = guides[index];
    const float3 color = float3(current[index].Color);
    TemporalPixel result{packed_float3(color), 1};
    const float3 look = TemporalNormalize(float3(c.Target) - float3(c.Eye));
    const float3 side = TemporalNormalize(cross(look, float3(c.Up))), up = cross(side, look);
    const float3 relative =
        (center.Flags & TemporalSky) ? float3(center.Direction) : float3(center.Position) - float3(c.Eye);
    const float depth = dot(relative, look);
    if (c.HistoryAvailable && depth > 0 && isfinite(depth))
    {
        const float halfY = tan(c.FieldOfView * TemporalPi / (2 * TemporalHalfTurn));
        const float halfX = halfY * float(c.Width) / float(c.Height);
        const float rasterX = (dot(relative, side) / depth / halfX + 1) * c.Width / 2 - TemporalPixelCenter;
        const float rasterY = (1 - dot(relative, up) / depth / halfY) * c.Height / 2 - TemporalPixelCenter;
        if (isfinite(rasterX) && isfinite(rasterY) && rasterX > -1 && rasterY > -1 && rasterX < c.Width &&
            rasterY < c.Height)
        {
            const int left = int(floor(rasterX)), top = int(floor(rasterY));
            const float blendX = rasterX - left, blendY = rasterY - top;
            float weights[TemporalTapCount] = {0, 0, 0, 0};
            uint sources[TemporalTapCount] = {0, 0, 0, 0};
            float total = 0, unit = 0;
            uint youngest = c.MaximumAge;
            for (int dy = 0; dy < TemporalBilinearSide; ++dy)
                for (int dx = 0; dx < TemporalBilinearSide; ++dx)
                {
                    const int x = left + dx, y = top + dy;
                    const float weight = (dx ? blendX : 1 - blendX) * (dy ? blendY : 1 - blendY);
                    if (!(weight > 0) || x < 0 || y < 0 || x >= int(c.Width) || y >= int(c.Height))
                        continue;
                    const uint source = uint(y) * c.Width + uint(x),
                               tap = uint(dy * TemporalBilinearSide + dx);
                    if (!TemporalCompatible(center, oldGuides[source], depth, c))
                        continue;
                    weights[tap] = weight;
                    sources[tap] = source;
                    total += weight;
                    unit = max(unit, TemporalPeak(abs(float3(previous[source].Color))));
                    youngest = min(youngest, previous[source].Age);
                }
            if (total >= c.MinimumSupport)
            {
                float3 history = float3(0);
                if (unit > 0)
                {
                    for (uint tap = 0; tap < TemporalTapCount; ++tap)
                        if (weights[tap] > 0)
                            history += (float3(previous[sources[tap]].Color) / unit) * weights[tap];
                    history = clamp(history / total, -1.0f, 1.0f) * unit;
                }
                float3 minimum = color, maximum = color;
                const int cx = int(index % c.Width), cy = int(index / c.Width);
                for (int y = cy - TemporalRadius; y <= cy + TemporalRadius; ++y)
                    for (int x = cx - TemporalRadius; x <= cx + TemporalRadius; ++x)
                    {
                        if (x < 0 || y < 0 || x >= int(c.Width) || y >= int(c.Height))
                            continue;
                        const uint neighbor = uint(y) * c.Width + uint(x);
                        if (!TemporalCompatible(center, guides[neighbor], center.Depth, c))
                            continue;
                        const float3 value = float3(current[neighbor].Color);
                        minimum = min(minimum, value);
                        maximum = max(maximum, value);
                    }
                result.Age = min(c.MaximumAge, youngest + 1);
                const float fresh = max(c.CurrentWeight, 1.0f / result.Age);
                // Keep signed linear color through history reconstruction.
                const float scale = max(TemporalPeak(abs(minimum)),
                                        max(TemporalPeak(abs(maximum)), TemporalPeak(abs(history))));
                if (scale > 0)
                {
                    minimum /= scale;
                    maximum /= scale;
                    const float3 expansion = (maximum - minimum) * c.Expansion;
                    const float3 clipped = clamp(history / scale, clamp(minimum - expansion, -1.0f, 1.0f),
                                                 clamp(maximum + expansion, -1.0f, 1.0f));
                    result.Color = packed_float3(
                        clamp(clipped * (1 - fresh) + (color / scale) * fresh, -1.0f, 1.0f) * scale);
                }
            }
        }
    }
    if (!all(isfinite(float3(result.Color))))
    {
        atomic_fetch_or_explicit(errors, TemporalBadColor, memory_order_relaxed);
        return;
    }
    output[index] = result;
    display[index] = TemporalDisplay(float3(result.Color), c.GammaCorrection);
}
