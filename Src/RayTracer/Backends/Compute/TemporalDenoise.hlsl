// Independent Compute temporal filter. Previous-frame bilinear support is
// visibility-tested per tap before neighborhood clipping and bounded blending.
// It reconstructs linear radiance; raw Monte Carlo moments remain untouched.
struct TemporalPixel
{
    float3 Radiance;
    uint Age;
};
struct GeometryPixel
{
    float3 Position;
    float Depth;
    float3 Normal;
    float Distance;
    float3 Direction;
    uint Flags;
    uint Primitive, Material;
    uint2 Reserved;
};
StructuredBuffer<TemporalPixel> Current : register(t0);
StructuredBuffer<GeometryPixel> Geometry : register(t1);
StructuredBuffer<TemporalPixel> History : register(t2);
StructuredBuffer<GeometryPixel> HistoryGeometry : register(t3);
RWStructuredBuffer<TemporalPixel> Filtered : register(u0);
RWStructuredBuffer<uint> Display : register(u1);
RWStructuredBuffer<uint> Errors : register(u2);
cbuffer TemporalControls : register(b0)
{
    uint Width, Height, Begin, Count;
    uint HasHistory, MaximumHistory, GammaCorrection, Reserved;
    float MinimumCurrentWeight, AbsoluteDepthTolerance, RelativeDepthTolerance, MinimumNormalCosine;
    float MinimumHistoryWeight, NeighborhoodExpansion;
    float2 ReservedScales;
    float3 Eye;
    float Fov;
    float3 Target;
    float ReservedTarget;
    float3 Up;
    float ReservedUp;
};
static const uint Threads = 64, ExtentLimit = 16384, HistoryLimit = 256;
static const uint Surface = 1, Background = 2, ConfigurationError = 1, NumericalError = 2;
static const uint BilinearTaps = 4, BilinearSide = 2;
static const int NeighborhoodRadius = 1;
static const float Pi = 3.14159265358979323846f, HalfTurnDegrees = 180, PixelCenter = 0.5f;
float3 TemporalUnit(float3 v)
{
    float scale = max(abs(v.x), max(abs(v.y), abs(v.z)));
    if (!(scale > 0))
        return 0;
    v /= scale;
    return v / sqrt(dot(v, v));
}
float MaxChannel(float3 color)
{
    return max(color.x, max(color.y, color.z));
}
bool CompatibleGeometry(GeometryPixel current, GeometryPixel tap, float depth)
{
    if (current.Flags & Background)
        return (tap.Flags & Background) != 0;
    if (!(tap.Flags & Surface) || current.Flags != tap.Flags || current.Primitive != tap.Primitive ||
        current.Material != tap.Material || dot(current.Normal, tap.Normal) < MinimumNormalCosine)
        return false;
    float tolerance = AbsoluteDepthTolerance + RelativeDepthTolerance * max(depth, tap.Depth);
    float3 offset = tap.Position - current.Position;
    float plane = max(abs(dot(offset, current.Normal)), abs(dot(offset, tap.Normal)));
    return isfinite(plane) && abs(tap.Depth - depth) <= tolerance && plane <= tolerance;
}
uint TemporalDisplay(float3 color)
{
    static const float LinearKnee = 0.0031308f, LinearSlope = 12.92f, GammaScale = 1.055f,
                       GammaOffset = 0.055f;
    static const float GammaPower = 1.0f / 2.4f, MaximumByte = 255, Nearest = 0.5f;
    static const uint GreenShift = 8, BlueShift = 16, Opaque = 0xff000000u;
    color = saturate(color / max(1.0f, MaxChannel(color)));
    if (GammaCorrection)
        for (uint channel = 0; channel < 3; ++channel)
            color[channel] = color[channel] <= LinearKnee
                                 ? LinearSlope * color[channel]
                                 : GammaScale * pow(color[channel], GammaPower) - GammaOffset;
    uint3 bytes = uint3(color * MaximumByte + Nearest);
    return bytes.x | (bytes.y << GreenShift) | (bytes.z << BlueShift) | Opaque;
}
[numthreads(Threads, 1, 1)] void TemporalDenoise(uint3 dispatch : SV_DispatchThreadID)
{
    if (dispatch.x >= Count)
        return;
    uint population, stride;
    Current.GetDimensions(population, stride);
    if (!Width || !Height || Width > ExtentLimit || Height > ExtentLimit || Width * Height != population ||
        Begin > population || Count > population - Begin || !MaximumHistory ||
        MaximumHistory > HistoryLimit || HasHistory > 1)
    {
        InterlockedOr(Errors[0], ConfigurationError);
        return;
    }
    const uint index = Begin + dispatch.x;
    const GeometryPixel guide = Geometry[index];
    const float3 fresh = Current[index].Radiance;
    TemporalPixel result;
    result.Radiance = fresh;
    result.Age = 1;
    float3 forward = TemporalUnit(Target - Eye);
    float3 right = TemporalUnit(cross(forward, Up)), vertical = cross(right, forward);
    float3 offset = (guide.Flags & Background) ? guide.Direction : guide.Position - Eye;
    float projectedDepth = dot(offset, forward);
    if (HasHistory && projectedDepth > 0 && isfinite(projectedDepth))
    {
        float halfHeight = tan(Fov * Pi / (2 * HalfTurnDegrees));
        float halfWidth = halfHeight * float(Width) / float(Height);
        float projectedX = (dot(offset, right) / projectedDepth / halfWidth + 1) * Width / 2 - PixelCenter;
        float projectedY =
            (1 - dot(offset, vertical) / projectedDepth / halfHeight) * Height / 2 - PixelCenter;
        if (isfinite(projectedX) && isfinite(projectedY) && projectedX > -1 && projectedY > -1 &&
            projectedX < Width && projectedY < Height)
        {
            int left = int(floor(projectedX)), top = int(floor(projectedY));
            float fractionX = projectedX - left, fractionY = projectedY - top;
            float weights[BilinearTaps] = {0, 0, 0, 0};
            uint sources[BilinearTaps] = {0, 0, 0, 0};
            float total = 0, peak = 0;
            uint youngest = MaximumHistory;
            for (uint tap = 0; tap < BilinearTaps; ++tap)
            {
                uint dx = tap % BilinearSide, dy = tap / BilinearSide;
                int x = left + int(dx), y = top + int(dy);
                float weight = (dx ? fractionX : 1 - fractionX) * (dy ? fractionY : 1 - fractionY);
                if (!(weight > 0) || x < 0 || y < 0 || x >= int(Width) || y >= int(Height))
                    continue;
                uint source = uint(y) * Width + uint(x);
                if (!CompatibleGeometry(guide, HistoryGeometry[source], projectedDepth))
                    continue;
                sources[tap] = source;
                weights[tap] = weight;
                total += weight;
                peak = max(peak, MaxChannel(abs(History[source].Radiance)));
                youngest = min(youngest, History[source].Age);
            }
            if (total >= MinimumHistoryWeight)
            {
                float3 history = 0;
                if (peak > 0)
                {
                    for (uint tap = 0; tap < BilinearTaps; ++tap)
                        if (weights[tap] > 0)
                            history += (History[sources[tap]].Radiance / peak) * weights[tap];
                    history = clamp(history / total, -1.0f, 1.0f) * peak;
                }
                float3 low = fresh, high = fresh;
                int centerX = int(index % Width), centerY = int(index / Width);
                for (int dy = -NeighborhoodRadius; dy <= NeighborhoodRadius; ++dy)
                    for (int dx = -NeighborhoodRadius; dx <= NeighborhoodRadius; ++dx)
                    {
                        int x = centerX + dx, y = centerY + dy;
                        if (x < 0 || y < 0 || x >= int(Width) || y >= int(Height))
                            continue;
                        uint tap = uint(y) * Width + uint(x);
                        if (!CompatibleGeometry(guide, Geometry[tap], guide.Depth))
                            continue;
                        low = min(low, Current[tap].Radiance);
                        high = max(high, Current[tap].Radiance);
                    }
                result.Age = min(MaximumHistory, youngest + 1);
                float currentWeight = max(MinimumCurrentWeight, 1.0f / result.Age);
                // Signed spectral observations must survive both interpolation
                // and clipping; only the final display conversion is saturated.
                float scale = max(MaxChannel(abs(low)), max(MaxChannel(abs(high)), MaxChannel(abs(history))));
                if (scale > 0)
                {
                    low /= scale;
                    high /= scale;
                    float3 spread = (high - low) * NeighborhoodExpansion;
                    float3 clipped = clamp(history / scale, clamp(low - spread, -1.0f, 1.0f),
                                           clamp(high + spread, -1.0f, 1.0f));
                    result.Radiance =
                        clamp(clipped * (1 - currentWeight) + (fresh / scale) * currentWeight, -1.0f, 1.0f) * scale;
                }
            }
        }
    }
    if (any(!isfinite(result.Radiance)))
    {
        InterlockedOr(Errors[0], NumericalError);
        return;
    }
    Filtered[index] = result;
    Display[index] = TemporalDisplay(result.Radiance);
}
