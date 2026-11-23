// Independent Metal color/dispersion-guided a-trous reconstruction.
// Dammertz et al. (2010), https://doi.org/10.2312/EGGH/HPG10/067-075.
// Pair-normalized RGB avoids HDR overflow without requiring native doubles.
// Optional static pinhole guides stop mixing across normal/plane/material edges.
// Original film and sampling statistics stay read-only; there is no history.
#include <metal_stdlib>
using namespace metal;
struct DenoisePixel
{
    packed_float3 Radiance;
    float M2;
    uint Samples, GuideFlags, GuideMaterial, Reserved;
    packed_float3 GuidePosition;
    float GuideDepth;
    packed_float3 GuideNormal;
    float ReservedNormal;
    packed_float3 DiffuseAlbedo;
    float ReservedAlbedo;
};
struct DenoiseControls
{
    uint Width, Height, Level, Begin, Count, Flags;
    float NoiseScale, RelativeScale, AbsoluteScale;
    uint GammaCorrection;
    float NormalScale, RelativePlaneScale;
};
static_assert(sizeof(DenoisePixel) == 80 && sizeof(DenoiseControls) == 48, "Metal denoising upload ABI");
constant uint DenoiseMomentBit = 1, DenoiseGeometryBit = 2, DenoiseMaximumLevels = 5;
constant uint DenoiseConfigurationError = 1, DenoiseNumericalError = 2;
constant int DenoiseRadius = 2;
constant float DenoiseRed = 0.2126f, DenoiseGreen = 0.7152f, DenoiseBlue = 0.0722f;
float DenoiseLuma(float3 value)
{
    return value.x * DenoiseRed + value.y * DenoiseGreen + value.z * DenoiseBlue;
}
float DenoiseKernel(int displacement)
{
    switch (abs(displacement))
    {
    case 0:
        return 6;
    case 1:
        return 4;
    default:
        return 1;
    }
}
float DenoiseNoise(DenoisePixel pixel, uint flags, float scale)
{
    if (!(flags & DenoiseMomentBit) || pixel.Samples < 2)
        return 0;
    // Taking the square root before rescaling retains large/small HDR ranges.
    return sqrt(pixel.M2) / sqrt(float(pixel.Samples)) / sqrt(float(pixel.Samples - 1)) / scale;
}
float DenoiseHypot(float a, float b)
{
    const float scale = max(a, b);
    if (scale == 0)
        return 0;
    a /= scale;
    b /= scale;
    return scale * sqrt(a * a + b * b);
}
float DenoiseWeight(float3 a, float3 b, DenoisePixel rawA, DenoisePixel rawB,
                    constant DenoiseControls& controls)
{
    const float3 magnitude = max(abs(a), abs(b));
    const float unit = max(controls.AbsoluteScale, max(magnitude.x, max(magnitude.y, magnitude.z)));
    a /= unit;
    b /= unit;
    const float sigma =
        controls.AbsoluteScale / unit + controls.RelativeScale * max(abs(DenoiseLuma(a)), abs(DenoiseLuma(b))) +
        controls.NoiseScale *
            DenoiseHypot(DenoiseNoise(rawA, controls.Flags, unit), DenoiseNoise(rawB, controls.Flags, unit));
    const float3 delta = a - b;
    if (sigma == 0)
        return all(delta == float3(0)) ? 1.0f : 0.0f;
    const float3 normalized = delta / sigma;
    return exp(-dot(normalized, normalized) / 2);
}
float DenoiseGeometry(DenoisePixel a, DenoisePixel b, constant DenoiseControls& controls)
{
    if (!(controls.Flags & DenoiseGeometryBit)) return 1;
    constexpr uint SurfaceFlag = 1;
    const bool firstSurface = (a.GuideFlags & SurfaceFlag) != 0;
    const bool secondSurface = (b.GuideFlags & SurfaceFlag) != 0;
    if (firstSurface != secondSurface) return 0;
    if (!firstSurface) return 1;
    if (a.GuideMaterial != b.GuideMaterial) return 0;
    const float3 firstNormal = float3(a.GuideNormal), secondNormal = float3(b.GuideNormal);
    const float3 dn = (firstNormal - secondNormal) / controls.NormalScale;
    const float3 p = float3(a.GuidePosition), q = float3(b.GuidePosition);
    const float depth = max(a.GuideDepth, b.GuideDepth);
    // Normalize positions before subtraction: opposite finite endpoints must
    // not overflow. Plane/depth ratios are invariant under this rescaling.
    const float scale = max(depth, max(max(abs(p.x), max(abs(p.y), abs(p.z))),
                                       max(abs(q.x), max(abs(q.y), abs(q.z)))));
    const float3 displacement = q / scale - p / scale;
    const float plane = max(abs(dot(displacement, firstNormal)), abs(dot(displacement, secondNormal)));
    const float scaledDepth = depth / scale;
    if (plane > 0 && !(scaledDepth > 0)) return 0;
    const float normalizedPlane = plane > 0 ? (plane / scaledDepth) / controls.RelativePlaneScale : 0;
    return exp(-(dot(dn, dn) + normalizedPlane * normalizedPlane) / 2);
}
uint DenoiseDisplay(float3 color, uint gamma)
{
    constexpr float Threshold = 0.0031308f, LinearSlope = 12.92f, PowerScale = 1.055f;
    constexpr float Offset = 0.055f, Exponent = 1.0f / 2.4f, ByteMaximum = 255;
    constexpr uint BitsPerChannel = 8, Opaque = 0xff000000u;
    const float range = max(1.0f, max(color.x, max(color.y, color.z)));
    color /= range;
    uint encoded = Opaque;
    for (uint channel = 0; channel < 3; ++channel)
    {
        float value = clamp(color[channel], 0.0f, 1.0f);
        if (gamma)
            value = value <= Threshold ? value * LinearSlope : PowerScale * pow(value, Exponent) - Offset;
        encoded |= uint(value * ByteMaximum + 0.5f) << (channel * BitsPerChannel);
    }
    return encoded;
}
kernel void ArvoSpatialDenoise(device const DenoisePixel* raw [[buffer(0)]],
                               device const float4* previous [[buffer(1)]], device float4* next [[buffer(2)]],
                               device atomic_uint* errors [[buffer(3)]],
                               constant DenoiseControls& controls [[buffer(4)]],
                               device uint* display [[buffer(5)]], uint lane [[thread_position_in_grid]])
{
    if (lane >= controls.Count)
        return;
    constexpr uint MaximumExtent = 16384;
    if (!controls.Width || !controls.Height || controls.Width > MaximumExtent ||
        controls.Height > MaximumExtent || controls.Level >= DenoiseMaximumLevels ||
        controls.Begin > controls.Width * controls.Height ||
        controls.Count > controls.Width * controls.Height - controls.Begin ||
        (controls.Flags & ~(DenoiseMomentBit | DenoiseGeometryBit)) || !(controls.AbsoluteScale > 0) ||
        ((controls.Flags & DenoiseGeometryBit) && (!(controls.NormalScale > 0) || !(controls.RelativePlaneScale > 0))))
    {
        atomic_fetch_or_explicit(errors, DenoiseConfigurationError, memory_order_relaxed);
        return;
    }
    const uint pixel = controls.Begin + lane;
    const int x = int(pixel % controls.Width), y = int(pixel / controls.Width), step = 1 << controls.Level;
    const float3 center = controls.Level ? previous[pixel].xyz : float3(raw[pixel].Radiance);
    float3 mean = float3(0);
    const bool diffuse = raw[pixel].DiffuseAlbedo.x >= 0;
    float3 albedoSum = 0;
    float accumulated = 0;
    for (int dy = -DenoiseRadius; dy <= DenoiseRadius; ++dy)
    {
        const int yy = y + dy * step;
        if (yy < 0 || yy >= int(controls.Height))
            continue;
        for (int dx = -DenoiseRadius; dx <= DenoiseRadius; ++dx)
        {
            const int xx = x + dx * step;
            if (xx < 0 || xx >= int(controls.Width))
                continue;
            const uint address = uint(yy) * controls.Width + uint(xx);
            if (diffuse != (raw[address].DiffuseAlbedo.x >= 0))
                continue;
            const float3 color = controls.Level ? previous[address].xyz : float3(raw[address].Radiance);
            const float weight = DenoiseKernel(dx) * DenoiseKernel(dy) *
                                 DenoiseWeight(center, color, raw[pixel], raw[address], controls) *
                                 DenoiseGeometry(raw[pixel], raw[address], controls);
            if (!(weight > 0))
                continue;
            if (diffuse)
                albedoSum += weight * float3(raw[address].DiffuseAlbedo);
            const float previousWeight = accumulated;
            accumulated += weight;
            const float retained = previousWeight / accumulated, incoming = weight / accumulated;
            for (uint channel = 0; channel < 3; ++channel)
            {
                // Opposite HDR signs require a convex sum to avoid overflowing
                // their difference. Otherwise anchor at the heavier endpoint
                // and retain the small weight without a 1 - blend subtraction.
                if ((mean[channel] < 0) != (color[channel] < 0))
                    mean[channel] = mean[channel] * retained + color[channel] * incoming;
                else if (weight > previousWeight)
                    mean[channel] = color[channel] + (mean[channel] - color[channel]) * retained;
                else
                    mean[channel] += (color[channel] - mean[channel]) * incoming;
            }
        }
    }
    if (diffuse && accumulated > 0)
    {
        constexpr float AlbedoFloor = 0.001;
        const float3 ratio = (float3(raw[pixel].DiffuseAlbedo) + AlbedoFloor) /
                              (albedoSum / accumulated + AlbedoFloor);
        const float3 adjusted = mean * ratio;
        for (uint channel = 0; channel < 3; ++channel)
            if (isfinite(float(adjusted[channel])))
                mean[channel] = adjusted[channel];
    }
    if (!(accumulated > 0) || !all(isfinite(mean)))
    {
        atomic_fetch_or_explicit(errors, DenoiseNumericalError, memory_order_relaxed);
        return;
    }
    next[pixel] = float4(mean, 0);
    display[pixel] = DenoiseDisplay(mean, controls.GammaCorrection);
}
