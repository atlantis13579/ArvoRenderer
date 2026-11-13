#include "RayTracer/Backends/CPU/Spectral.h"
#include <algorithm>
#include <cmath>

namespace Arvo::RayTracing::Cpu
{
namespace
{
// Jakob and Hanika, "A Low-Dimensional Function Space for Efficient Spectral
// Upsampling" (2019): sigmoid of a quadratic, bounded by construction.
// https://rgl.epfl.ch/publications/Jakob2019Spectral
// This is a numerical preparation solver, not their precomputed lookup table.
// Cache its coefficients per material/texel; never run Newton at every bounce.
// RGB defines a metamer, not a measured spectrum. Inputs are linear sRGB.
constexpr double WavelengthCenter = (VisibleMinimumNm + VisibleMaximumNm) / 2;
constexpr double WavelengthRadius = (VisibleMaximumNm - VisibleMinimumNm) / 2;
constexpr double YIntegral = 106.94617150055822;
constexpr unsigned MaximumNewtonSteps = 48, MaximumBacktracks = 16;
constexpr double FitTolerance = 0.00001, RelativeSingularity = 0.0000001;
constexpr double MaximumCoefficient = 1000000;
constexpr float UnavailableError = -1;

double Sigmoid(double p)
{
    // The reciprocal form retains small positive reflectances near black.
    const double root = std::sqrt(1 + p * p);
    const double tail = 1 / (2 * root * (root + std::abs(p)));
    return p < 0 ? tail : 1 - tail;
}
struct Integral
{
    double Value[3]{}, Jacobian[3][3]{};
};
Integral Integrate(const double coefficients[3], const RgbFitTable& table)
{
    Integral result;
    for (const auto& sample : table)
    {
        const double x = sample.Coordinate, basis[3]{x * x, x, 1};
        const double p = coefficients[0] * basis[0] + coefficients[1] * x + coefficients[2];
        const double root = std::sqrt(1 + p * p), derivative = 1 / (2 * root * root * root);
        const double weights[3]{sample.Weight.X, sample.Weight.Y, sample.Weight.Z};
        for (unsigned row = 0; row < 3; ++row)
        {
            result.Value[row] += weights[row] * Sigmoid(p);
            for (unsigned column = 0; column < 3; ++column)
                result.Jacobian[row][column] += weights[row] * derivative * basis[column];
        }
    }
    return result;
}
bool NewtonStep(Integral integral, const double target[3], double step[3])
{
    // Partial pivoting keeps near-neutral and saturated fits well conditioned.
    double augmented[3][4]{};
    double largest = 0;
    for (unsigned row = 0; row < 3; ++row)
    {
        for (unsigned column = 0; column < 3; ++column)
        {
            augmented[row][column] = integral.Jacobian[row][column];
            largest = std::max(largest, std::abs(augmented[row][column]));
        }
        augmented[row][3] = integral.Value[row] - target[row];
    }
    for (unsigned column = 0; column < 3; ++column)
    {
        unsigned pivot = column;
        for (unsigned row = column + 1; row < 3; ++row)
            if (std::abs(augmented[row][column]) > std::abs(augmented[pivot][column]))
                pivot = row;
        if (std::abs(augmented[pivot][column]) <= largest * RelativeSingularity)
            return false;
        for (unsigned entry = column; entry < 4; ++entry)
            std::swap(augmented[pivot][entry], augmented[column][entry]);
        for (unsigned row = column + 1; row < 3; ++row)
        {
            const double factor = augmented[row][column] / augmented[column][column];
            for (unsigned entry = column; entry < 4; ++entry)
                augmented[row][entry] -= factor * augmented[column][entry];
        }
    }
    for (int row = 2; row >= 0; --row)
    {
        double value = augmented[row][3];
        for (unsigned column = unsigned(row + 1); column < 3; ++column)
            value -= augmented[row][column] * step[column];
        step[row] = value / augmented[row][row];
    }
    return true;
}
double Loss(const Integral& integral, const double target[3])
{
    double loss = 0;
    for (unsigned channel = 0; channel < 3; ++channel)
    {
        const double residual = integral.Value[channel] - target[channel];
        loss += residual * residual;
    }
    return loss;
}
} // namespace
float RgbReferenceIlluminant(float wavelength)
{
    if (!std::isfinite(wavelength) || wavelength < VisibleMinimumNm || wavelength > VisibleMaximumNm)
        return 0;
    // Positive exp-quadratic calibrated against our analytic CIE observer and
    // rounded sRGB matrix: integral maps to RGB(1,1,1). This is a synthetic white
    // with D65 chromaticity, NOT the measured D65 spectral power distribution.
    constexpr double WhiteQuadratic = -0.6592809648409116, WhiteLinear = -0.6601308563300832,
                     WhiteOffset = -0.06589824578190885;
    const double x = (wavelength - WavelengthCenter) / WavelengthRadius;
    return float(std::exp((WhiteQuadratic * x + WhiteLinear) * x + WhiteOffset));
}
RgbFitResponse MakeRgbFitResponse(unsigned index)
{
    const float step = (VisibleMaximumNm - VisibleMinimumNm) / RgbFitSampleCount;
    const float wavelength = VisibleMinimumNm + (float(index) + 0.5f) * step;
    return {XYZToLinearSRGB(CieMatching(wavelength)) *
                float(RgbReferenceIlluminant(wavelength) * step / YIntegral),
            float((wavelength - WavelengthCenter) / WavelengthRadius)};
}
RgbSpectrum FitRgbReflectance(Float3 rgb, const RgbFitTable& table)
{
    RgbSpectrum result{0, 0, 0, 0, 0, UnavailableError, SpectrumFitInvalid, 0};
    if (!std::isfinite(rgb.X) || !std::isfinite(rgb.Y) || !std::isfinite(rgb.Z) || rgb.X < 0 || rgb.Y < 0 ||
        rgb.Z < 0 || rgb.X > 1 || rgb.Y > 1 || rgb.Z > 1)
        return result;
    if (rgb.X == rgb.Y && rgb.Y == rgb.Z)
    {
        result.Constant = rgb.X;
        result.Error = 0;
        result.Status = SpectrumFitConverged;
        return result;
    }
    // Remove the achromatic floor and normalize color contrast. Solving the
    // remaining RGB-cube edge avoids the nearly singular dark/near-white basins
    // of an unscaled polynomial. Adding the floor back preserves [minRGB,maxRGB].
    const float floor = std::min({rgb.X, rgb.Y, rgb.Z});
    const float contrast = std::max({rgb.X, rgb.Y, rgb.Z}) - floor;
    const double target[3]{(rgb.X - floor) / contrast, (rgb.Y - floor) / contrast,
                           (rgb.Z - floor) / contrast};
    double coefficients[3]{};
    result.Status = SpectrumFitUnconverged;
    for (unsigned iteration = 0; iteration <= MaximumNewtonSteps; ++iteration)
    {
        const auto integral = Integrate(coefficients, table);
        double error = 0;
        for (unsigned channel = 0; channel < 3; ++channel)
            error = std::max(error, std::abs(integral.Value[channel] - target[channel]));
        result = {float(coefficients[0]),
                  float(coefficients[1]),
                  float(coefficients[2]),
                  contrast,
                  floor,
                  float(error),
                  error <= FitTolerance ? SpectrumFitConverged : SpectrumFitUnconverged,
                  iteration};
        if (error <= FitTolerance || iteration == MaximumNewtonSteps)
            break;
        double step[3]{};
        if (!NewtonStep(integral, target, step))
            break;
        double fraction = 1;
        bool accepted = false;
        for (unsigned search = 0; search < MaximumBacktracks; ++search, fraction /= 2)
        {
            double trial[3]{};
            bool finite = true;
            for (unsigned axis = 0; axis < 3; ++axis)
            {
                trial[axis] = coefficients[axis] - fraction * step[axis];
                finite &= std::isfinite(trial[axis]) && std::abs(trial[axis]) <= MaximumCoefficient;
            }
            if (finite && Loss(Integrate(trial, table), target) < Loss(integral, target))
            {
                std::copy(trial, trial + 3, coefficients);
                accepted = true;
                break;
            }
        }
        if (!accepted)
            break;
    }
    return result;
}
RgbSpectrum FitRgbIlluminant(Float3 rgb, const RgbFitTable& table)
{
    if (!std::isfinite(rgb.X) || !std::isfinite(rgb.Y) || !std::isfinite(rgb.Z) || rgb.X < 0 || rgb.Y < 0 ||
        rgb.Z < 0)
        return {0, 0, 0, 0, 0, UnavailableError, SpectrumFitInvalid, 0};
    const float scale = std::max({rgb.X, rgb.Y, rgb.Z});
    auto result = FitRgbReflectance(scale > 0 ? rgb / scale : Float3{}, table);
    result.Scale *= scale;
    result.Constant *= scale;
    // Error belongs to the normalized chromatic component, before floor/exposure.
    return result;
}
float EvaluateRgbReflectance(RgbSpectrum spectrum, float wavelength)
{
    if (spectrum.Status == SpectrumFitInvalid || !std::isfinite(wavelength) ||
        wavelength < VisibleMinimumNm || wavelength > VisibleMaximumNm)
        return 0;
    const double x = (wavelength - WavelengthCenter) / WavelengthRadius;
    const double shape = Sigmoid((spectrum.Quadratic * x + spectrum.Linear) * x + spectrum.Offset);
    return float(spectrum.Constant + shape * spectrum.Scale);
}
float EvaluateRgbIlluminant(RgbSpectrum spectrum, float wavelength)
{
    return EvaluateRgbReflectance(spectrum, wavelength) * RgbReferenceIlluminant(wavelength);
}
RgbSpectrum FitSceneSpectrum(const SpectralSceneData::Request& request, const RgbFitTable& response)
{
    switch (request.Kind)
    {
    case SpectralSceneData::Role::Reflectance:
        return FitRgbReflectance(request.Color, response);
    case SpectralSceneData::Role::Illuminant:
    case SpectralSceneData::Role::Nonnegative:
        return FitRgbIlluminant(request.Color, response);
    }
    return {0, 0, 0, 0, 0, UnavailableError, SpectrumFitInvalid, 0};
}
} // namespace Arvo::RayTracing::Cpu
