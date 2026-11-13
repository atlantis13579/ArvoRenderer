// Independent DXR offline temporal reconstruction; preview uses stationary averaging.
// A ray-generation dispatch resolves film history without tracing transport.
// Surface checks use oriented geometry, not shading normals or motion guessed
// from noisy radiance. Background correspondence ignores camera translation.
struct HistoryPixel
{
    float3 Color;
    uint Age;
};
struct GuidePixel
{
    float3 Point;
    float CameraDepth;
    float3 Normal;
    float RayDistance;
    float3 Direction;
    uint Flags;
    uint Primitive, Material;
    uint2 Reserved;
};
StructuredBuffer<HistoryPixel> Radiance : register(t0);
StructuredBuffer<GuidePixel> Guides : register(t1);
StructuredBuffer<HistoryPixel> Prior : register(t2);
StructuredBuffer<GuidePixel> PriorGuides : register(t3);
RWStructuredBuffer<HistoryPixel> Result : register(u0);
RWStructuredBuffer<uint> Display : register(u1);
RWStructuredBuffer<uint> Failure : register(u2);
cbuffer TemporalControls : register(b0)
{
    uint FilmWidth, FilmHeight, First, Population;
    uint UseHistory, HistoryCap, EncodeGamma, Reserved;
    float MinimumFresh, DepthAbsolute, DepthRelative, NormalThreshold;
    float MinimumSupport, ClipExpansion;
    float2 ReservedScale;
    float3 PreviousEye;
    float FieldOfView;
    float3 PreviousTarget;
    float ReservedTarget;
    float3 PreviousUp;
    float ReservedUp;
};
static const uint SurfaceFlag = 1, SkyFlag = 2, InvalidConfiguration = 1, InvalidRadiance = 2;
static const uint MaximumExtent = 16384, MaximumFrames = 256, BilinearTapCount = 4;
static const int NeighborhoodRadius = 1;
static const float Pi = 3.14159265358979323846f, DegreesPerHalfTurn = 180, PixelMidpoint = 0.5f;
float3 NormalizeTemporal(float3 value)
{
    float unit = max(max(abs(value.x), abs(value.y)), abs(value.z));
    if (unit == 0)
        return 0;
    value /= unit;
    return value / sqrt(value.x * value.x + value.y * value.y + value.z * value.z);
}
float PeakTemporal(float3 v)
{
    return max(max(v.x, v.y), v.z);
}
bool VisibilityCompatible(GuidePixel center, GuidePixel neighbor, float expectedDepth)
{
    if (center.Flags & SkyFlag)
        return (neighbor.Flags & SkyFlag) != 0;
    if (!(neighbor.Flags & SurfaceFlag) || neighbor.Flags != center.Flags ||
        neighbor.Material != center.Material || neighbor.Primitive != center.Primitive ||
        dot(center.Normal, neighbor.Normal) < NormalThreshold)
        return false;
    float limit = DepthAbsolute + DepthRelative * max(expectedDepth, neighbor.CameraDepth);
    float3 difference = neighbor.Point - center.Point;
    float normalDistance = max(abs(dot(difference, center.Normal)), abs(dot(difference, neighbor.Normal)));
    return isfinite(normalDistance) && normalDistance <= limit &&
           abs(neighbor.CameraDepth - expectedDepth) <= limit;
}
bool PreviousRaster(GuidePixel pixel, out float x, out float y, out float depth)
{
    float3 look = NormalizeTemporal(PreviousTarget - PreviousEye);
    float3 side = NormalizeTemporal(cross(look, PreviousUp));
    float3 up = cross(side, look);
    float3 relative = (pixel.Flags & SkyFlag) ? pixel.Direction : pixel.Point - PreviousEye;
    depth = dot(relative, look);
    x = 0;
    y = 0;
    if (!isfinite(depth) || depth <= 0)
        return false;
    float vertical = tan(FieldOfView * Pi / (2 * DegreesPerHalfTurn));
    float horizontal = vertical * float(FilmWidth) / float(FilmHeight);
    x = (dot(relative, side) / depth / horizontal + 1) * FilmWidth / 2 - PixelMidpoint;
    y = (1 - dot(relative, up) / depth / vertical) * FilmHeight / 2 - PixelMidpoint;
    return isfinite(x) && isfinite(y) && x > -1 && y > -1 && x < FilmWidth && y < FilmHeight;
}
uint EncodeTemporalDisplay(float3 rgb)
{
    static const float Knee = 0.0031308f, Slope = 12.92f, Scale = 1.055f, Offset = 0.055f,
                       Power = 1.0f / 2.4f;
    static const float ByteMaximum = 255, Nearest = 0.5f;
    static const uint GreenShift = 8, BlueShift = 16, Alpha = 0xff000000u;
    rgb = saturate(rgb / max(1.0f, PeakTemporal(rgb)));
    for (uint channel = 0; EncodeGamma && channel < 3; ++channel)
        rgb[channel] =
            rgb[channel] <= Knee ? rgb[channel] * Slope : Scale * pow(rgb[channel], Power) - Offset;
    uint3 bytes = uint3(rgb * ByteMaximum + Nearest);
    return bytes.x | (bytes.y << GreenShift) | (bytes.z << BlueShift) | Alpha;
}
[shader("raygeneration")] void TemporalDenoiseRayGen()
{
    uint lane = DispatchRaysIndex().x;
    if (lane >= Population)
        return;
    if (!FilmWidth || !FilmHeight || FilmWidth > MaximumExtent || FilmHeight > MaximumExtent ||
        First > FilmWidth * FilmHeight || Population > FilmWidth * FilmHeight - First || !HistoryCap ||
        HistoryCap > MaximumFrames)
    {
        InterlockedOr(Failure[0], InvalidConfiguration);
        return;
    }
    uint index = First + lane;
    GuidePixel guide = Guides[index];
    HistoryPixel value;
    value.Color = Radiance[index].Color;
    value.Age = 1;
    float rasterX, rasterY, depth;
    if (UseHistory && PreviousRaster(guide, rasterX, rasterY, depth))
    {
        int x0 = int(floor(rasterX)), y0 = int(floor(rasterY));
        float blendX = rasterX - x0, blendY = rasterY - y0;
        float weights[BilinearTapCount] = {0, 0, 0, 0};
        uint sources[BilinearTapCount] = {0, 0, 0, 0};
        float support = 0, colorUnit = 0;
        uint age = HistoryCap;
        for (uint tap = 0; tap < BilinearTapCount; ++tap)
        {
            static const uint RowWidth = 2;
            uint dx = tap % RowWidth, dy = tap / RowWidth;
            int x = x0 + int(dx), y = y0 + int(dy);
            float weight = (dx ? blendX : 1 - blendX) * (dy ? blendY : 1 - blendY);
            if (weight <= 0 || x < 0 || y < 0 || x >= int(FilmWidth) || y >= int(FilmHeight))
                continue;
            uint from = uint(y) * FilmWidth + uint(x);
            if (!VisibilityCompatible(guide, PriorGuides[from], depth))
                continue;
            weights[tap] = weight;
            sources[tap] = from;
            support += weight;
            colorUnit = max(colorUnit, PeakTemporal(abs(Prior[from].Color)));
            age = min(age, Prior[from].Age);
        }
        if (support >= MinimumSupport)
        {
            float3 history = 0;
            if (colorUnit > 0)
            {
                for (uint tap = 0; tap < BilinearTapCount; ++tap)
                    if (weights[tap] > 0)
                        history += (Prior[sources[tap]].Color / colorUnit) * weights[tap];
                history = clamp(history / support, -1.0f, 1.0f) * colorUnit;
            }
            float3 lower = value.Color, upper = value.Color;
            int x = int(index % FilmWidth), y = int(index / FilmWidth);
            for (int oy = -NeighborhoodRadius; oy <= NeighborhoodRadius; ++oy)
                for (int ox = -NeighborhoodRadius; ox <= NeighborhoodRadius; ++ox)
                {
                    int nx = x + ox, ny = y + oy;
                    if (nx < 0 || ny < 0 || nx >= int(FilmWidth) || ny >= int(FilmHeight))
                        continue;
                    uint neighbor = uint(ny) * FilmWidth + uint(nx);
                    if (!VisibilityCompatible(guide, Guides[neighbor], guide.CameraDepth))
                        continue;
                    lower = min(lower, Radiance[neighbor].Color);
                    upper = max(upper, Radiance[neighbor].Color);
                }
            value.Age = min(HistoryCap, age + 1);
            float fresh = max(MinimumFresh, 1.0f / value.Age);
            // Normalize by magnitude so negative RGB basis coefficients are
            // retained while finite HDR interpolation remains bounded.
            float unit = max(PeakTemporal(abs(lower)), max(PeakTemporal(abs(upper)), PeakTemporal(abs(history))));
            if (unit > 0)
            {
                lower /= unit;
                upper /= unit;
                float3 radius = (upper - lower) * ClipExpansion;
                history = clamp(history / unit, clamp(lower - radius, -1.0f, 1.0f),
                                clamp(upper + radius, -1.0f, 1.0f));
                value.Color = clamp(history * (1 - fresh) + (value.Color / unit) * fresh, -1.0f, 1.0f) * unit;
            }
        }
    }
    if (any(!isfinite(value.Color)))
    {
        InterlockedOr(Failure[0], InvalidRadiance);
        return;
    }
    Result[index] = value;
    Display[index] = EncodeTemporalDisplay(value.Color);
}
