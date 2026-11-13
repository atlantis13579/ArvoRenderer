#include "SpatialDenoise.h"
#include <algorithm>
#include <array>
#include <cmath>
#include "Maths/ColorSpace.h"

namespace Arvo::RayTracing::Cpu
{
namespace
{
// Independent CPU color/dispersion/geometry-guided a-trous reconstruction.
// Dammertz et al., "Edge-Avoiding A-Trous Wavelet Transform for fast Global
// Illumination Filtering" (2010), https://doi.org/10.2312/EGGH/HPG10/067-075.
// Optional primary guides gate normals, tangent planes and material boundaries.
// These center-ray guides are restricted to static pinhole surfaces.
// Raw film statistics remain intact;
// moments supply a noise heuristic, never new sample counts or confidence bounds.
constexpr int Radius = 2;
constexpr std::array<double, 5> B3Weights{1, 4, 6, 4, 1};
constexpr double RedLuminance = 0.2126, GreenLuminance = 0.7152, BlueLuminance = 0.0722;
using Color = std::array<double, 3>;
Color Wide(Float3 color)
{
    return {color.X, color.Y, color.Z};
}
double Luminance(Color color)
{
    return RedLuminance * color[0] + GreenLuminance * color[1] + BlueLuminance * color[2];
}
double Dispersion(const FilmPixel& pixel, bool enabled)
{
    if (!enabled || pixel.SampleCount < 2)
        return 0;
    return std::sqrt(double(pixel.LuminanceM2) / double(pixel.SampleCount) / (pixel.SampleCount - 1));
}
double Affinity(Color center, Color neighbor, double centerNoise, double neighborNoise,
                const ReconstructionOptions& options)
{
    // Pairwise normalization avoids overflow at HDR endpoints. A Gaussian in
    // RGB retains chromatic edges even when the two colors have equal luminance.
    const double scale = std::max({double(options.AbsoluteColorScale), std::abs(center[0]),
                                   std::abs(center[1]), std::abs(center[2]), std::abs(neighbor[0]),
                                   std::abs(neighbor[1]), std::abs(neighbor[2])});
    double distanceSquared = 0;
    for (uint32_t channel = 0; channel < center.size(); ++channel)
    {
        center[channel] /= scale;
        neighbor[channel] /= scale;
        const double difference = center[channel] - neighbor[channel];
        distanceSquared += difference * difference;
    }
    const double width = double(options.AbsoluteColorScale) / scale +
                         options.RelativeColorScale * std::max(std::abs(Luminance(center)), std::abs(Luminance(neighbor))) +
                         options.NoiseScale * std::hypot(centerNoise / scale, neighborNoise / scale);
    return std::exp(-distanceSquared / (2 * width * width));
}
double GeometryAffinity(const GeometryGuide &center, const GeometryGuide &neighbor,
                        const ReconstructionOptions &options)
{
    const bool surface = (center.Flags & GeometryGuide::Surface) != 0;
    if (surface != ((neighbor.Flags & GeometryGuide::Surface) != 0))
        return 0;
    if (!surface)
        return 1;
    if (center.MaterialIndex != neighbor.MaterialIndex)
        return 0;
    const auto a = Wide(center.GeometricNormal), b = Wide(neighbor.GeometricNormal);
    const auto p = Wide(center.Position), q = Wide(neighbor.Position);
    double normalDistance = 0, planeA = 0, planeB = 0;
    for (size_t axis = 0; axis < a.size(); ++axis)
    {
        const double normalDelta = a[axis] - b[axis];
        normalDistance += normalDelta * normalDelta;
        const double displacement = q[axis] - p[axis];
        planeA += displacement * a[axis];
        planeB += displacement * b[axis];
    }
    // Symmetric plane distances preserve slanted coplanar surfaces. Axial
    // depth sets a world-scale tolerance without a scene-specific constant.
    const double planeWidth =
        double(options.RelativePlaneScale) * std::max(center.CameraDepth, neighbor.CameraDepth);
    const double plane = std::max(std::abs(planeA), std::abs(planeB)) / planeWidth;
    const double angular = normalDistance / (double(options.NormalScale) * options.NormalScale);
    return std::exp(-(angular + plane * plane) / 2);
}
} // namespace
bool SpatialDenoise(const LinearFilm &film, const ReconstructionOptions &options, bool useMoments,
                    bool gammaCorrection, const Detail::CancellationProbe &cancelled,
                    ReconstructedImage &result, std::string &error, const GeometryGuideImage *guides)
{
    if (!ValidateReconstructionInput(film, options, error) ||
        !ValidateReconstructionGuides(film, options, guides, error))
        return false;
    ReconstructedImage output{film.Width, film.Height, {}, {}};
    output.Pixels.reserve(film.Pixels.size());
    std::vector<double> noise;
    noise.reserve(film.Pixels.size());
    for (size_t index = 0; index < film.Pixels.size(); ++index)
    {
        if (cancelled && cancelled())
        {
            error = "CPU reconstruction cancelled";
            return false;
        }
        output.Pixels.push_back(film.Pixels[index].MeanRadiance);
        noise.push_back(Dispersion(film.Pixels[index], useMoments));
    }
    std::vector<Float3> next(output.Pixels.size());
    const uint32_t passes = options.Filter == ReconstructionFilter::None ? 0 : options.Passes;
    for (uint32_t pass = 0; pass < passes; ++pass)
    {
        const int64_t step = int64_t(1) << pass;
        for (uint32_t y = 0; y < film.Height; ++y)
        {
            if (cancelled && cancelled())
            {
                error = "CPU reconstruction cancelled";
                return false;
            }
            for (uint32_t x = 0; x < film.Width; ++x)
            {
                const size_t index = size_t(y) * film.Width + x;
                const Color center = Wide(output.Pixels[index]);
                Color mean{};
                double totalWeight = 0;
                for (int dy = -Radius; dy <= Radius; ++dy)
                {
                    const int64_t sampleY = int64_t(y) + dy * step;
                    if (sampleY < 0 || sampleY >= film.Height)
                        continue;
                    for (int dx = -Radius; dx <= Radius; ++dx)
                    {
                        const int64_t sampleX = int64_t(x) + dx * step;
                        if (sampleX < 0 || sampleX >= film.Width)
                            continue;
                        const size_t neighborIndex = size_t(sampleY) * film.Width + size_t(sampleX);
                        const Color neighbor = Wide(output.Pixels[neighborIndex]);
                        const double weight =
                            B3Weights[size_t(dx + Radius)] * B3Weights[size_t(dy + Radius)] *
                            Affinity(center, neighbor, noise[index], noise[neighborIndex], options) *
                            (options.GeometryGuidance
                                 ? GeometryAffinity(guides->Pixels[index], guides->Pixels[neighborIndex],
                                                    options)
                                 : 1);
                        if (!(weight > 0))
                            continue;
                        const double previousWeight = totalWeight;
                        totalWeight += weight;
                        // Update about the heavier endpoint. Subtracting almost
                        // the whole mean erases faint light beside a black tap.
                        const bool incomingDominates = weight > previousWeight;
                        const double fraction = (incomingDominates ? previousWeight : weight) / totalWeight;
                        for (uint32_t channel = 0; channel < mean.size(); ++channel)
                            mean[channel] = incomingDominates
                                ? neighbor[channel] + (mean[channel] - neighbor[channel]) * fraction
                                : mean[channel] + (neighbor[channel] - mean[channel]) * fraction;
                    }
                }
                if (!(totalWeight > 0) || !std::isfinite(mean[0]) || !std::isfinite(mean[1]) ||
                    !std::isfinite(mean[2]))
                {
                    error = "CPU spatial denoising produced invalid radiance";
                    return false;
                }
                next[index] = {float(mean[0]), float(mean[1]), float(mean[2])};
            }
        }
        output.Pixels.swap(next);
    }
    output.Display = Image(film.Width, film.Height);
    auto& rgba = output.Display.GetPixelsRGBA8();
    constexpr uint32_t Channels = 4, Opaque = 255;
    for (size_t index = 0; index < output.Pixels.size(); ++index)
    {
        if (cancelled && cancelled())
        {
            error = "CPU reconstruction cancelled";
            return false;
        }
        const auto pixel = output.Pixels[index];
        const float scale = Maths::DisplayLinearScale(pixel.X, pixel.Y, pixel.Z);
        const auto encode = [&](float value) {
            return gammaCorrection ? Maths::LinearToSRGB8(value * scale)
                                   : Maths::LinearToUNorm8(value * scale);
        };
        rgba[index * Channels] = encode(pixel.X);
        rgba[index * Channels + 1] = encode(pixel.Y);
        rgba[index * Channels + 2] = encode(pixel.Z);
        rgba[index * Channels + 3] = Opaque;
    }
    result = std::move(output);
    error.clear();
    return true;
}
} // namespace Arvo::RayTracing::Cpu
