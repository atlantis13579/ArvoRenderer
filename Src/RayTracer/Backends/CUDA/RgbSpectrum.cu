#include "SpectralData.cuh"
// CUDA preparation solver, independently authored for sigmoid-quadratic spectra.
// Jakob/Hanika, "A Low-Dimensional Function Space for Efficient Spectral
// Upsampling" (2019), https://rgl.epfl.ch/publications/Jakob2019Spectral
// Store fitted coefficients with scene data. RGB is a metameric constraint, not
// a measurement; per-ray evaluation must not repeat this nonlinear solve.
constexpr double CudaRgbCenter = (VisibleMinimumNm + VisibleMaximumNm) / 2;
constexpr double CudaRgbRadius = (VisibleMaximumNm - VisibleMinimumNm) / 2;
constexpr unsigned CudaRgbMaxSteps = 48, CudaRgbLineSearch = 16;
constexpr double CudaRgbAccuracy = 0.00001, CudaRgbRelativeDeterminant = 0.0000001;
constexpr double CudaRgbMaxCoefficient = 1000000;
constexpr float CudaRgbNoError = -1;
__device__ double CudaRgbSigmoid(double value)
{
    const double radius = sqrt(1 + value * value);
    const double tail = 0.5 / (radius * (radius + fabs(value)));
    return value < 0 ? tail : 1 - tail;
}
__device__ float RgbReferenceIlluminant(float wavelength)
{
    if (!isfinite(wavelength) || wavelength < VisibleMinimumNm || wavelength > VisibleMaximumNm)
        return 0;
    // Positive synthetic white calibrated with our CIE fit and sRGB matrix.
    // D65 chromaticity only: these are not measured D65 emission samples.
    constexpr double WhiteCurvature = -0.6592809648409116, WhiteSlope = -0.6601308563300832;
    constexpr double WhiteConstant = -0.06589824578190885;
    const double coordinate = (wavelength - CudaRgbCenter) / CudaRgbRadius;
    return float(exp(coordinate * (WhiteCurvature * coordinate + WhiteSlope) + WhiteConstant));
}
__device__ float4 MakeRgbFitResponse(unsigned index)
{
    constexpr double ColorNormalization = 106.94617150055822;
    constexpr float step = (VisibleMaximumNm - VisibleMinimumNm) / RgbFitSampleCount;
    const float wavelength = VisibleMinimumNm + (float(index) + 0.5f) * step;
    const float3 color = XYZToLinearSRGB(CieMatching(wavelength));
    const double weight = RgbReferenceIlluminant(wavelength) * step / ColorNormalization;
    return make_float4(float(color.x * weight), float(color.y * weight), float(color.z * weight),
                       float((wavelength - CudaRgbCenter) / CudaRgbRadius));
}
struct CudaRgbMoments
{
    double Color[3], Derivative[3][3];
};
__device__ CudaRgbMoments CudaRgbIntegrate(const double* coefficients, const float4* response)
{
    CudaRgbMoments moments{};
    for (unsigned index = 0; index < RgbFitSampleCount; ++index)
    {
        const float4 point = response[index];
        const double x = point.w;
        const double basis[]{x * x, x, 1};
        const double weights[]{point.x, point.y, point.z};
        const double value = coefficients[0] * x * x + coefficients[1] * x + coefficients[2];
        const double radius = sqrt(1 + value * value);
        const double slope = 0.5 / (radius * radius * radius);
        for (unsigned channel = 0; channel < 3; ++channel)
        {
            moments.Color[channel] += weights[channel] * CudaRgbSigmoid(value);
            for (unsigned axis = 0; axis < 3; ++axis)
                moments.Derivative[channel][axis] += weights[channel] * slope * basis[axis];
        }
    }
    return moments;
}
__device__ double CudaRgbDeterminant(const double m[3][3])
{
    return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
           m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
           m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
}
__device__ double CudaRgbLoss(CudaRgbMoments moments, const double* target)
{
    double sum = 0;
    for (unsigned channel = 0; channel < 3; ++channel)
    {
        const double delta = moments.Color[channel] - target[channel];
        sum += delta * delta;
    }
    return sum;
}
__device__ RgbSpectrum FitRgbReflectance(float3 rgb, const float4* response)
{
    RgbSpectrum fit{0, 0, 0, 0, 0, CudaRgbNoError, SpectrumFitInvalid, 0};
    if (!isfinite(rgb.x) || !isfinite(rgb.y) || !isfinite(rgb.z) || rgb.x < 0 || rgb.y < 0 || rgb.z < 0 ||
        rgb.x > 1 || rgb.y > 1 || rgb.z > 1)
        return fit;
    if (rgb.x == rgb.y && rgb.y == rgb.z)
    {
        fit.Constant = rgb.x;
        fit.Error = 0;
        fit.Status = SpectrumFitConverged;
        return fit;
    }
    // Separate the grey floor and contrast. Newton then fits an RGB-cube edge
    // instead of entering ill-conditioned dark/near-white polynomial basins.
    // Restoring this decomposition keeps reflectance in [minRGB,maxRGB].
    const float floor = fminf(rgb.x, fminf(rgb.y, rgb.z));
    const float contrast = fmaxf(rgb.x, fmaxf(rgb.y, rgb.z)) - floor;
    const double target[]{(rgb.x - floor) / contrast, (rgb.y - floor) / contrast, (rgb.z - floor) / contrast};
    double coefficients[3]{};
    for (unsigned iteration = 0; iteration <= CudaRgbMaxSteps; ++iteration)
    {
        const CudaRgbMoments integral = CudaRgbIntegrate(coefficients, response);
        double error = 0, residual[3]{};
        for (unsigned channel = 0; channel < 3; ++channel)
        {
            residual[channel] = integral.Color[channel] - target[channel];
            error = fmax(error, fabs(residual[channel]));
        }
        fit = {float(coefficients[0]),
               float(coefficients[1]),
               float(coefficients[2]),
               contrast,
               floor,
               float(error),
               error <= CudaRgbAccuracy ? SpectrumFitConverged : SpectrumFitUnconverged,
               iteration};
        if (error <= CudaRgbAccuracy || iteration == CudaRgbMaxSteps)
            break;
        const double determinant = CudaRgbDeterminant(integral.Derivative);
        double determinantScale = 1;
        for (unsigned axis = 0; axis < 3; ++axis)
        {
            double columnScale = 0;
            for (unsigned row = 0; row < 3; ++row)
                columnScale = fmax(columnScale, fabs(integral.Derivative[row][axis]));
            determinantScale *= columnScale;
        }
        if (fabs(determinant) <= determinantScale * CudaRgbRelativeDeterminant)
            break;
        double step[3]{};
        for (unsigned axis = 0; axis < 3; ++axis)
        {
            double substituted[3][3]{};
            for (unsigned row = 0; row < 3; ++row)
                for (unsigned column = 0; column < 3; ++column)
                    substituted[row][column] =
                        column == axis ? residual[row] : integral.Derivative[row][column];
            step[axis] = CudaRgbDeterminant(substituted) / determinant;
        }
        bool improved = false;
        double damping = 1;
        for (unsigned attempt = 0; attempt < CudaRgbLineSearch; ++attempt, damping *= 0.5)
        {
            double candidate[3]{};
            bool admissible = true;
            for (unsigned axis = 0; axis < 3; ++axis)
            {
                candidate[axis] = coefficients[axis] - step[axis] * damping;
                admissible &= isfinite(candidate[axis]) && fabs(candidate[axis]) <= CudaRgbMaxCoefficient;
            }
            if (admissible &&
                CudaRgbLoss(CudaRgbIntegrate(candidate, response), target) < CudaRgbLoss(integral, target))
            {
                for (unsigned axis = 0; axis < 3; ++axis)
                    coefficients[axis] = candidate[axis];
                improved = true;
                break;
            }
        }
        if (!improved)
            break;
    }
    return fit;
}
__device__ RgbSpectrum FitRgbIlluminant(float3 rgb, const float4* response)
{
    if (!isfinite(rgb.x) || !isfinite(rgb.y) || !isfinite(rgb.z) || rgb.x < 0 || rgb.y < 0 || rgb.z < 0)
        return {0, 0, 0, 0, 0, CudaRgbNoError, SpectrumFitInvalid, 0};
    const float scale = fmaxf(rgb.x, fmaxf(rgb.y, rgb.z));
    const float3 normalized =
        scale > 0 ? make_float3(rgb.x / scale, rgb.y / scale, rgb.z / scale) : make_float3(0, 0, 0);
    auto fit = FitRgbReflectance(normalized, response);
    fit.Scale *= scale;
    fit.Constant *= scale; // Error is for normalized chromatic contrast, before floor/exposure.
    return fit;
}
__device__ float EvaluateRgbReflectance(RgbSpectrum fit, float wavelength)
{
    if (fit.Status == SpectrumFitInvalid || !isfinite(wavelength) || wavelength < VisibleMinimumNm ||
        wavelength > VisibleMaximumNm)
        return 0;
    const double x = (wavelength - CudaRgbCenter) / CudaRgbRadius;
    const double shape = CudaRgbSigmoid(double(fit.Quadratic) * x * x + fit.Linear * x + fit.Offset);
    return float(fit.Constant + shape * fit.Scale);
}
__device__ float EvaluateRgbIlluminant(RgbSpectrum fit, float wavelength)
{
    return EvaluateRgbReflectance(fit, wavelength) * RgbReferenceIlluminant(wavelength);
}
__device__ RgbSpectrum FitSceneSpectrum(SpectralSource source, const float4* response)
{
    const float3 rgb = make_float3(source.Red, source.Green, source.Blue);
    if (source.Kind == SpectrumRoleReflectance)
        return FitRgbReflectance(rgb, response);
    if (source.Kind == SpectrumRoleIlluminant || source.Kind == SpectrumRoleNonnegative)
        return FitRgbIlluminant(rgb, response);
    return {0, 0, 0, 0, 0, CudaRgbNoError, SpectrumFitInvalid, 0};
}
