// Independent CUDA color/dispersion-guided a-trous reconstruction.
// Dammertz et al. (2010), "Edge-Avoiding A-Trous Wavelet Transform for fast
// Global Illumination Filtering", https://doi.org/10.2312/EGGH/HPG10/067-075.
// Optional primary guides preserve normal, tangent-plane and material edges.
// The guides describe static pinhole surfaces, not optical or volume paths.
// Raw film stays unchanged. Compile this handwritten CUDA unit directly.
#include <cuda_runtime.h>
#include <cmath>
#include <cfloat>
#include <cstdint>

namespace
{
constexpr unsigned StencilRadius = 2, MaximumPasses = 5, MomentGuidance = 1, GeometryGuidance = 2;
constexpr unsigned InvalidConfiguration = 1, InvalidRadiance = 2;
constexpr double LuminanceRed = 0.2126, LuminanceGreen = 0.7152, LuminanceBlue = 0.0722;
struct alignas(16) FilmInput
{
    float4 RadianceMoment;
    unsigned Samples, GuideFlags, GuideMaterial, Reserved;
    float3 GuidePosition;
    float GuideDepth;
    float3 GuideNormal;
    float ReservedNormal;
    float3 DiffuseAlbedo;
    float ReservedAlbedo;
};
struct alignas(16) Controls
{
    unsigned Width, Height, Pass, Begin, Count, Flags;
    float NoiseScale, RelativeColorScale, AbsoluteColorScale;
    unsigned GammaCorrection;
    float NormalScale, RelativePlaneScale;
};
static_assert(sizeof(FilmInput) == 80 && sizeof(Controls) == 48);
__device__ double3 ReadColor(const FilmInput* raw, const float4* previous, unsigned index, unsigned pass)
{
    const float4 value = pass ? previous[index] : raw[index].RadianceMoment;
    return make_double3(value.x, value.y, value.z);
}
__device__ double Noise(const FilmInput& raw, unsigned flags)
{
    if (!(flags & MomentGuidance) || raw.Samples < 2)
        return 0;
    return sqrt(double(raw.RadianceMoment.w) / double(raw.Samples) / double(raw.Samples - 1));
}
__device__ double B3(int offset)
{
    const int distance = offset < 0 ? -offset : offset;
    // Cubic B-spline stencil; the common 1/256 cancels during normalization.
    return distance == 0 ? 6 : distance == 1 ? 4 : 1;
}
__device__ double EdgeWeight(double3 a, double3 b, double sa, double sb, Controls c)
{
    const double scale =
        fmax(double(c.AbsoluteColorScale), fmax(fmax(fabs(a.x), fmax(fabs(a.y), fabs(a.z))),
                                              fmax(fabs(b.x), fmax(fabs(b.y), fabs(b.z)))));
    a = make_double3(a.x / scale, a.y / scale, a.z / scale);
    b = make_double3(b.x / scale, b.y / scale, b.z / scale);
    const double la = LuminanceRed * a.x + LuminanceGreen * a.y + LuminanceBlue * a.z;
    const double lb = LuminanceRed * b.x + LuminanceGreen * b.y + LuminanceBlue * b.z;
    sa /= scale;
    sb /= scale;
    const double sigma = c.AbsoluteColorScale / scale + c.RelativeColorScale * fmax(fabs(la), fabs(lb)) +
                         c.NoiseScale * sqrt(sa * sa + sb * sb);
    const double x = a.x - b.x, y = a.y - b.y, z = a.z - b.z;
    return exp(-(x * x + y * y + z * z) / (2 * sigma * sigma));
}
__device__ double GeometrySimilarity(const FilmInput& a, const FilmInput& b, const Controls& controls)
{
    if (!(controls.Flags & GeometryGuidance)) return 1;
    constexpr unsigned SurfaceGuide = 1;
    if ((a.GuideFlags & SurfaceGuide) != (b.GuideFlags & SurfaceGuide)) return 0;
    if (!(a.GuideFlags & SurfaceGuide)) return 1;
    if (a.GuideMaterial != b.GuideMaterial) return 0;
    const double3 normalDelta = make_double3(double(a.GuideNormal.x) - b.GuideNormal.x,
        double(a.GuideNormal.y) - b.GuideNormal.y, double(a.GuideNormal.z) - b.GuideNormal.z);
    const double3 displacement = make_double3(double(b.GuidePosition.x) - a.GuidePosition.x,
        double(b.GuidePosition.y) - a.GuidePosition.y, double(b.GuidePosition.z) - a.GuidePosition.z);
    const double first = displacement.x * a.GuideNormal.x + displacement.y * a.GuideNormal.y +
                         displacement.z * a.GuideNormal.z;
    const double second = displacement.x * b.GuideNormal.x + displacement.y * b.GuideNormal.y +
                          displacement.z * b.GuideNormal.z;
    const double planeWidth = double(controls.RelativePlaneScale) * fmax(a.GuideDepth, b.GuideDepth);
    const double planeDistance = fmax(fabs(first), fabs(second)) / planeWidth;
    const double angular = (normalDelta.x * normalDelta.x + normalDelta.y * normalDelta.y +
                            normalDelta.z * normalDelta.z) / (double(controls.NormalScale) * controls.NormalScale);
    return exp(-(angular + planeDistance * planeDistance) / 2);
}
__device__ double RestoreDiffuse(double mean, float centerAlbedo, double averageAlbedo)
{
    constexpr double AlbedoFloor = 0.001;
    const double value = mean * ((centerAlbedo + AlbedoFloor) / (averageAlbedo + AlbedoFloor));
    return fabs(value) <= FLT_MAX ? value : mean;
}
__device__ unsigned Display(double3 color, unsigned gamma)
{
    constexpr double SrgbKnee = 0.0031308, SrgbLinear = 12.92, SrgbScale = 1.055;
    constexpr double SrgbOffset = 0.055, SrgbPower = 1.0 / 2.4, ByteMaximum = 255;
    constexpr unsigned GreenShift = 8, BlueShift = 16, Alpha = 0xff000000u;
    const double scale = 1 / fmax(1.0, fmax(color.x, fmax(color.y, color.z)));
    double channels[3]{color.x * scale, color.y * scale, color.z * scale};
    unsigned encoded[3];
    for (unsigned c = 0; c < 3; ++c)
    {
        double v = fmin(1.0, fmax(0.0, channels[c]));
        if (gamma)
            v = v <= SrgbKnee ? SrgbLinear * v : SrgbScale * pow(v, SrgbPower) - SrgbOffset;
        encoded[c] = unsigned(v * ByteMaximum + 0.5);
    }
    return encoded[0] | (encoded[1] << GreenShift) | (encoded[2] << BlueShift) | Alpha;
}
} // namespace
extern "C" __global__ void ArvoSpatialDenoise(const FilmInput* raw, const float4* previous, float4* next,
                                              unsigned* display, unsigned* errors, Controls controls)
{
    const unsigned lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= controls.Count)
        return;
    const uint64_t population = uint64_t(controls.Width) * controls.Height;
    if (!controls.Width || !controls.Height || controls.Pass >= MaximumPasses ||
        controls.Begin > population || controls.Count > population - controls.Begin ||
        (controls.Flags & ~(MomentGuidance | GeometryGuidance)) || !(controls.AbsoluteColorScale > 0) ||
        ((controls.Flags & GeometryGuidance) && (!(controls.NormalScale > 0) || !(controls.RelativePlaneScale > 0))))
    {
        atomicOr(errors, InvalidConfiguration);
        return;
    }
    const unsigned index = controls.Begin + lane, x = index % controls.Width, y = index / controls.Width;
    const int64_t spacing = int64_t(1) << controls.Pass;
    const double3 center = ReadColor(raw, previous, index, controls.Pass);
    const double centerNoise = Noise(raw[index], controls.Flags);
    double3 mean = make_double3(0, 0, 0);
    const bool diffuse = raw[index].DiffuseAlbedo.x >= 0;
    double3 albedoSum = make_double3(0, 0, 0);
    double sum = 0;
    for (int row = -int(StencilRadius); row <= int(StencilRadius); ++row)
    {
        const int64_t sampleY = int64_t(y) + spacing * row;
        if (sampleY < 0 || sampleY >= controls.Height)
            continue;
        for (int column = -int(StencilRadius); column <= int(StencilRadius); ++column)
        {
            const int64_t sampleX = int64_t(x) + spacing * column;
            if (sampleX < 0 || sampleX >= controls.Width)
                continue;
            const unsigned neighborIndex = unsigned(sampleY) * controls.Width + unsigned(sampleX);
            if (diffuse != (raw[neighborIndex].DiffuseAlbedo.x >= 0))
                continue;
            const double3 value = ReadColor(raw, previous, neighborIndex, controls.Pass);
            const double weight =
                B3(row) * B3(column) *
                EdgeWeight(center, value, centerNoise, Noise(raw[neighborIndex], controls.Flags), controls) *
                GeometrySimilarity(raw[index], raw[neighborIndex], controls);
            if (!(weight > 0))
                continue;
            if (diffuse)
            {
                albedoSum.x += weight * raw[neighborIndex].DiffuseAlbedo.x;
                albedoSum.y += weight * raw[neighborIndex].DiffuseAlbedo.y;
                albedoSum.z += weight * raw[neighborIndex].DiffuseAlbedo.z;
            }
            const double previous = sum;
            sum += weight;
            // Blend toward the lighter endpoint from the dominant endpoint;
            // a near-unit subtraction would erase a small but finite signal.
            if (weight > previous)
            {
                const double retained = previous / sum;
                mean.x = value.x + (mean.x - value.x) * retained;
                mean.y = value.y + (mean.y - value.y) * retained;
                mean.z = value.z + (mean.z - value.z) * retained;
            }
            else
            {
                const double incoming = weight / sum;
                mean.x += (value.x - mean.x) * incoming;
                mean.y += (value.y - mean.y) * incoming;
                mean.z += (value.z - mean.z) * incoming;
            }
        }
    }
    if (diffuse && sum > 0)
    {
        mean.x = RestoreDiffuse(mean.x, raw[index].DiffuseAlbedo.x, albedoSum.x / sum);
        mean.y = RestoreDiffuse(mean.y, raw[index].DiffuseAlbedo.y, albedoSum.y / sum);
        mean.z = RestoreDiffuse(mean.z, raw[index].DiffuseAlbedo.z, albedoSum.z / sum);
    }
    if (!(sum > 0) || !isfinite(mean.x) || !isfinite(mean.y) || !isfinite(mean.z))
    {
        atomicOr(errors, InvalidRadiance);
        return;
    }
    next[index] = make_float4(float(mean.x), float(mean.y), float(mean.z), 0);
    display[index] = Display(mean, controls.GammaCorrection);
}
