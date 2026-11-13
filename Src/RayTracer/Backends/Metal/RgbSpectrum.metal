#include "SpectralData.h"
// Independent Metal RGB-to-spectrum preparation. The sigmoid-quadratic model is
// from Jakob/Hanika (2019), "A Low-Dimensional Function Space for Efficient
// Spectral Upsampling", https://rgl.epfl.ch/publications/Jakob2019Spectral
// A linear-sRGB color specifies a metamer, never a measured material spectrum.
// Cache coefficients after preparation; shading evaluates without iteration.
constant float MetalRgbCenter = (VisibleMinimumNm + VisibleMaximumNm) / 2;
constant float MetalRgbRadius = (VisibleMaximumNm - VisibleMinimumNm) / 2;
constant uint MetalRgbIterationLimit = 48, MetalRgbDampingLimit = 16;
constant float MetalRgbTolerance = 0.00001f, MetalRgbSingularRatio = 0.0000001f;
constant float MetalRgbCoefficientLimit = 1000000;
constant float MetalRgbUnknownError = -1;
float MetalRgbSigmoid(float polynomial)
{
    const float hypotenuse = sqrt(1 + polynomial * polynomial);
    const float tail = 1 / (2 * hypotenuse * (hypotenuse + abs(polynomial)));
    return polynomial < 0 ? tail : 1 - tail;
}
float RgbReferenceIlluminant(float wavelength)
{
    if (!isfinite(wavelength) || wavelength < VisibleMinimumNm || wavelength > VisibleMaximumNm)
        return 0;
    // Analytic positive white matched to our CIE approximation and sRGB matrix;
    // D65 chromaticity is matched, its measured spectral distribution is not.
    const float WhiteA = -0.6592809648409116f, WhiteB = -0.6601308563300832f;
    const float WhiteC = -0.06589824578190885f;
    const float x = (wavelength - MetalRgbCenter) / MetalRgbRadius;
    return exp(WhiteA * x * x + WhiteB * x + WhiteC);
}
float4 MakeRgbFitResponse(uint index)
{
    const float FittedYIntegral = 106.94617150055822f;
    const float cellWidth = (VisibleMaximumNm - VisibleMinimumNm) / RgbFitSampleCount;
    const float wavelength = VisibleMinimumNm + (float(index) + 0.5f) * cellWidth;
    const float3 weight = XYZToLinearSRGB(CieMatching(wavelength)) *
                          (RgbReferenceIlluminant(wavelength) * cellWidth / FittedYIntegral);
    return float4(weight, (wavelength - MetalRgbCenter) / MetalRgbRadius);
}
struct MetalRgbIntegral
{
    float3 Color, CurvatureDerivative, SlopeDerivative, ConstantDerivative;
};
MetalRgbIntegral MetalRgbIntegrate(float3 polynomial, device const float4* response)
{
    MetalRgbIntegral result;
    result.Color = result.CurvatureDerivative = result.SlopeDerivative = result.ConstantDerivative =
        float3(0, 0, 0);
    for (uint cell = 0; cell < RgbFitSampleCount; cell++)
    {
        const float4 point = response[cell];
        const float x = point.w;
        const float p = polynomial.x * x * x + polynomial.y * x + polynomial.z;
        const float root = sqrt(1 + p * p);
        const float3 derivative = point.xyz / (2 * root * root * root);
        result.Color += point.xyz * MetalRgbSigmoid(p);
        result.CurvatureDerivative += derivative * x * x;
        result.SlopeDerivative += derivative * x;
        result.ConstantDerivative += derivative;
    }
    return result;
}
float MetalRgbMaxAbs(float3 value)
{
    return max(abs(value.x), max(abs(value.y), abs(value.z)));
}
RgbSpectrum FitRgbReflectance(float3 target, device const float4* response)
{
    RgbSpectrum fit{0, 0, 0, 0, 0, MetalRgbUnknownError, SpectrumFitInvalid, 0};
    if (!isfinite(target.x) || !isfinite(target.y) || !isfinite(target.z) ||
        min(target.x, min(target.y, target.z)) < 0 || max(target.x, max(target.y, target.z)) > 1)
        return fit;
    if (target.x == target.y && target.y == target.z)
    {
        fit.Constant = target.x;
        fit.Error = 0;
        fit.Status = SpectrumFitConverged;
        return fit;
    }
    // Fit normalized chromatic contrast on an RGB-cube edge, then add the grey
    // floor back. Dark/near-white inputs no longer force singular polynomial
    // basins; the spectrum stays between the original RGB channel extrema.
    fit.Constant = min(target.x, min(target.y, target.z));
    fit.Scale = max(target.x, max(target.y, target.z)) - fit.Constant;
    target = (target - float3(fit.Constant, fit.Constant, fit.Constant)) / fit.Scale;
    float3 coefficients = float3(0, 0, 0);
    fit.Status = SpectrumFitUnconverged;
    for (uint iteration = 0; iteration <= MetalRgbIterationLimit; iteration++)
    {
        const MetalRgbIntegral integral = MetalRgbIntegrate(coefficients, response);
        const float3 residual = integral.Color - target;
        fit.Quadratic = coefficients.x;
        fit.Linear = coefficients.y;
        fit.Offset = coefficients.z;
        fit.Iterations = iteration;
        fit.Error = MetalRgbMaxAbs(residual);
        if (fit.Error <= MetalRgbTolerance)
        {
            fit.Status = SpectrumFitConverged;
            break;
        }
        if (iteration == MetalRgbIterationLimit)
            break;
        const float3 a = integral.CurvatureDerivative, b = integral.SlopeDerivative,
                     c = integral.ConstantDerivative;
        const float3 cofactorA = cross(b, c), cofactorB = cross(c, a), cofactorC = cross(a, b);
        const float volume = dot(a, cofactorA);
        const float magnitude = MetalRgbMaxAbs(a) * MetalRgbMaxAbs(b) * MetalRgbMaxAbs(c);
        if (abs(volume) <= MetalRgbSingularRatio * magnitude)
            break;
        const float3 correction =
            float3(dot(cofactorA, residual), dot(cofactorB, residual), dot(cofactorC, residual)) / volume;
        bool accepted = false;
        float scale = 1;
        for (uint trialIndex = 0; trialIndex < MetalRgbDampingLimit; trialIndex++, scale /= 2)
        {
            const float3 trial = coefficients - correction * scale;
            if (!isfinite(trial.x) || !isfinite(trial.y) || !isfinite(trial.z) ||
                MetalRgbMaxAbs(trial) > MetalRgbCoefficientLimit)
                continue;
            const float3 trialResidual = MetalRgbIntegrate(trial, response).Color - target;
            if (dot(trialResidual, trialResidual) < dot(residual, residual))
            {
                coefficients = trial;
                accepted = true;
                break;
            }
        }
        if (!accepted)
            break;
    }
    return fit;
}
RgbSpectrum FitRgbIlluminant(float3 color, device const float4* response)
{
    if (!isfinite(color.x) || !isfinite(color.y) || !isfinite(color.z) ||
        min(color.x, min(color.y, color.z)) < 0)
        return {0, 0, 0, 0, 0, MetalRgbUnknownError, SpectrumFitInvalid, 0};
    const float intensity = max(color.x, max(color.y, color.z));
    const float3 chromaticity = intensity > 0 ? color / intensity : float3(0, 0, 0);
    RgbSpectrum fit = FitRgbReflectance(chromaticity, response);
    fit.Scale *= intensity;
    fit.Constant *= intensity; // Error is for normalized chromatic contrast, before floor/exposure.
    return fit;
}
float EvaluateRgbReflectance(RgbSpectrum fit, float wavelength)
{
    if (fit.Status == SpectrumFitInvalid || !isfinite(wavelength) || wavelength < VisibleMinimumNm ||
        wavelength > VisibleMaximumNm)
        return 0;
    const float coordinate = (wavelength - MetalRgbCenter) / MetalRgbRadius;
    const float polynomial = (fit.Quadratic * coordinate + fit.Linear) * coordinate + fit.Offset;
    return fit.Constant + MetalRgbSigmoid(polynomial) * fit.Scale;
}
float EvaluateRgbIlluminant(RgbSpectrum fit, float wavelength)
{
    return EvaluateRgbReflectance(fit, wavelength) * RgbReferenceIlluminant(wavelength);
}
RgbSpectrum FitSceneSpectrum(SpectralSource source, device const float4* response)
{
    const float3 rgb = float3(source.Red, source.Green, source.Blue);
    if (source.Kind == SpectrumRoleReflectance)
        return FitRgbReflectance(rgb, response);
    if (source.Kind == SpectrumRoleIlluminant || source.Kind == SpectrumRoleNonnegative)
        return FitRgbIlluminant(rgb, response);
    return {0, 0, 0, 0, 0, MetalRgbUnknownError, SpectrumFitInvalid, 0};
}
