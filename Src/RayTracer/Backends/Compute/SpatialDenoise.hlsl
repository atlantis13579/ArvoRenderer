// Independent Compute color/dispersion-guided a-trous filter.
// Dammertz et al., HPG 2010, https://doi.org/10.2312/EGGH/HPG10/067-075.
// Raw mean/M2/counts are immutable. The output is biased reconstructed RGB,
// Primary normal/plane/material guides are optional; there is no temporal history
// or fabricated film moment. One pinhole probe cannot represent lens/time/media.
struct FilmInput
{
    float3 Color;
    float M2;
    uint Samples;
    uint GuideFlags, GuideMaterial, Reserved;
    float3 GuidePosition;
    float GuideDepth;
    float3 GuideNormal;
    float ReservedNormal;
};
StructuredBuffer<FilmInput> Raw : register(t0);
StructuredBuffer<float4> Previous : register(t1);
RWStructuredBuffer<float4> Next : register(u0);
RWStructuredBuffer<uint> Errors : register(u1);
RWStructuredBuffer<uint> DisplayPixels : register(u2);
cbuffer Controls : register(b0)
{
    uint Width, Height, Pass, Begin;
    uint Count, Flags;
    float NoiseScale, RelativeColorScale;
    float AbsoluteColorScale;
    uint GammaCorrection;
    float NormalScale, RelativePlaneScale;
};
static const uint Threads = 64, MaximumPasses = 5, MomentGuidance = 1, GeometryGuidance = 2;
static const uint ConfigurationError = 1, NumericalError = 2;
static const int Radius = 2, StencilWidth = 5;
static const double Stencil[StencilWidth] = {1, 4, 6, 4, 1};
static const double3 Luminance = double3(0.2126, 0.7152, 0.0722);
double3 ReadColor(uint index)
{
    return Pass == 0 ? double3(Raw[index].Color) : double3(Previous[index].xyz);
}
double NoiseVariance(uint index)
{
    const FilmInput pixel = Raw[index];
    return (Flags & MomentGuidance) && pixel.Samples > 1
               ? double(pixel.M2) / double(pixel.Samples) / double(pixel.Samples - 1)
               : 0;
}
double Luma(double3 value)
{
    return value.x * Luminance.x + value.y * Luminance.y + value.z * Luminance.z;
}
double NoiseRoot(double value)
{
    if (value == 0) return 0;
    // Positive film variances stay in the normal binary64 range. DXIL has no
    // double sqrt intrinsic, so seed from the exponent and refine quadratically.
    static const uint ExponentBias = 1023, HighMantissaBits = 20, ExponentMask = 0x7ff;
    static const uint NewtonSteps = 6;
    uint low, high; asuint(value, low, high);
    const uint biasedExponent = (high >> HighMantissaBits) & ExponentMask;
    double root = asdouble(0u, ((biasedExponent + ExponentBias) / 2) << HighMantissaBits);
    for (uint iteration = 0; iteration < NewtonSteps; ++iteration) root = (root + value / root) / 2;
    return root;
}
double RangeWeight(double3 center, double3 value, double variance)
{
    const double unit = max(double(AbsoluteColorScale),
                            max(max(abs(center.x), max(abs(center.y), abs(center.z))),
                                max(abs(value.x), max(abs(value.y), abs(value.z)))));
    center /= unit;
    value /= unit;
    const double3 delta = center - value;
    const double sigma = double(AbsoluteColorScale) / unit +
                         double(RelativeColorScale) * max(abs(Luma(center)), abs(Luma(value))) +
                         double(NoiseScale) * NoiseRoot(variance) / unit;
    // DXIL does not define a double-vector dot intrinsic; keep scalar products.
    return double(
        exp(float(-(delta.x * delta.x + delta.y * delta.y + delta.z * delta.z) / (2 * sigma * sigma))));
}
double GeometryWeight(uint centerIndex, uint neighborIndex)
{
    if (!(Flags & GeometryGuidance)) return 1;
    static const uint SurfaceGuide = 1;
    const FilmInput a = Raw[centerIndex], b = Raw[neighborIndex];
    const uint kindA = a.GuideFlags & SurfaceGuide, kindB = b.GuideFlags & SurfaceGuide;
    if (kindA != kindB) return 0;
    if (!kindA) return 1;
    if (a.GuideMaterial != b.GuideMaterial) return 0;
    const double3 na = double3(a.GuideNormal), nb = double3(b.GuideNormal);
    const double3 delta = double3(b.GuidePosition) - double3(a.GuidePosition), dn = na - nb;
    const double distanceA = delta.x * na.x + delta.y * na.y + delta.z * na.z;
    const double distanceB = delta.x * nb.x + delta.y * nb.y + delta.z * nb.z;
    const double distance = max(abs(distanceA), abs(distanceB));
    const double width = double(RelativePlaneScale) * max(a.GuideDepth, b.GuideDepth);
    const double plane = distance / width;
    const double angular = (dn.x * dn.x + dn.y * dn.y + dn.z * dn.z) /
                           (double(NormalScale) * NormalScale);
    return double(exp(float(-(angular + plane * plane) / 2)));
}
uint DisplayColor(float3 color)
{
    static const float Knee = 0.0031308, LinearScale = 12.92, PowerScale = 1.055, Offset = 0.055;
    static const float InverseGamma = 1.0 / 2.4, ByteMaximum = 255;
    static const uint GreenShift = 8, BlueShift = 16, OpaqueAlpha = 0xff000000u;
    color = saturate(color / max(1.0, max(color.x, max(color.y, color.z))));
    if (GammaCorrection)
        for (uint c = 0; c < 3; ++c)
            color[c] =
                color[c] <= Knee ? color[c] * LinearScale : PowerScale * pow(color[c], InverseGamma) - Offset;
    const uint3 bytes = uint3(color * ByteMaximum + 0.5);
    return bytes.x | (bytes.y << GreenShift) | (bytes.z << BlueShift) | OpaqueAlpha;
}
[numthreads(Threads, 1, 1)] void SpatialDenoise(uint3 dispatch : SV_DispatchThreadID) {
    if (dispatch.x >= Count)
        return;
    uint rawCount, stride;
    Raw.GetDimensions(rawCount, stride);
    if (!Width || !Height || Pass >= MaximumPasses || Width > rawCount / Height ||
        Width * Height != rawCount || Begin > rawCount || Count > rawCount - Begin ||
        (Flags & ~(MomentGuidance | GeometryGuidance)) || !(AbsoluteColorScale > 0) ||
        ((Flags & GeometryGuidance) && (!(NormalScale > 0) || !(RelativePlaneScale > 0))))
    {
        InterlockedOr(Errors[0], ConfigurationError);
        return;
    }
    const uint index = Begin + dispatch.x, step = 1u << Pass;
    const int2 centerPixel = int2(index % Width, index / Width);
    const double3 center = ReadColor(index);
    const double variance = NoiseVariance(index);
    double3 mean = 0;
    double totalWeight = 0;
    for (uint tap = 0; tap < StencilWidth * StencilWidth; ++tap)
    {
        const uint column = tap % StencilWidth, row = tap / StencilWidth;
        const int2 pixel = centerPixel + (int2(column, row) - Radius) * int(step);
        if (any(pixel < 0) || pixel.x >= int(Width) || pixel.y >= int(Height))
            continue;
        const uint neighbor = uint(pixel.y) * Width + uint(pixel.x);
        const double3 color = ReadColor(neighbor);
        const double weight =
            Stencil[column] * Stencil[row] * RangeWeight(center, color, variance + NoiseVariance(neighbor)) *
            GeometryWeight(index, neighbor);
        if (weight <= 0)
            continue;
        const double previousWeight = totalWeight;
        totalWeight += weight;
        // Anchor at the heavier endpoint to retain tiny contributions when a
        // dominant black tap would otherwise subtract almost the entire mean.
        if (weight > previousWeight)
            mean = color + (mean - color) * (previousWeight / totalWeight);
        else
            mean += (color - mean) * (weight / totalWeight);
    }
    const float3 result = float3(mean);
    if (!(totalWeight > 0) || any(!isfinite(result)))
    {
        InterlockedOr(Errors[0], NumericalError);
        return;
    }
    Next[index] = float4(result, 0);
    DisplayPixels[index] = DisplayColor(result);
}
