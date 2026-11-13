#include "SpectralData.hlsli"
// Independent compute preparation of bounded sigmoid-quadratic RGB metamers.
// Jakob/Hanika (2019), "A Low-Dimensional Function Space for Efficient Spectral
// Upsampling", https://rgl.epfl.ch/publications/Jakob2019Spectral
// Fit once per material/texel. Shading evaluates the stored polynomial only.
// No fitted spectrum is claimed to be measured; inputs must be linear sRGB.
static const float RgbCenterNm = (VisibleMinimumNm + VisibleMaximumNm) / 2;
static const float RgbRadiusNm = (VisibleMaximumNm - VisibleMinimumNm) / 2;
static const uint RgbNewtonLimit = 48, RgbBacktrackLimit = 16;
static const float RgbFitTolerance = 0.00001f, RgbDeterminantTolerance = 0.0000001f;
static const float RgbCoefficientLimit = 1000000;
static const float RgbUnknownError = -1;
float ComputeRgbSigmoid(float p)
{
    float root = sqrt(1 + p * p);
    float tail = 1 / (2 * root * (root + abs(p)));
    return p < 0 ? tail : 1 - tail;
}
float RgbReferenceIlluminant(float wavelength)
{
    if (!isfinite(wavelength) || wavelength < VisibleMinimumNm || wavelength > VisibleMaximumNm)
        return 0;
    // Analytic exp-quadratic white calibrated to our CIE fit and sRGB matrix.
    // It has D65 chromaticity; it is NOT the measured D65 spectral distribution.
    const float WhiteQuadratic = -0.6592809648409116f, WhiteLinear = -0.6601308563300832f;
    const float WhiteOffset = -0.06589824578190885f;
    float x = (wavelength - RgbCenterNm) / RgbRadiusNm;
    return exp((WhiteQuadratic * x + WhiteLinear) * x + WhiteOffset);
}
float4 MakeRgbFitResponse(uint index)
{
    const float FittedYIntegral = 106.94617150055822f;
    const float step = (VisibleMaximumNm - VisibleMinimumNm) / RgbFitSampleCount;
    float wavelength = VisibleMinimumNm + (index + 0.5f) * step;
    float3 rgb = XYZToLinearSRGB(CieMatching(wavelength));
    return float4(rgb * (RgbReferenceIlluminant(wavelength) * step / FittedYIntegral),
                  (wavelength - RgbCenterNm) / RgbRadiusNm);
}
struct ComputeRgbIntegral
{
    float3 Value, Quadratic, Linear, Offset;
};
ComputeRgbIntegral ComputeRgbIntegrate(float3 polynomial, StructuredBuffer<float4> response)
{
    ComputeRgbIntegral result;
    result.Value = result.Quadratic = result.Linear = result.Offset = float3(0, 0, 0);
    for (uint sampleIndex = 0; sampleIndex < RgbFitSampleCount; ++sampleIndex)
    {
        float4 sample = response[sampleIndex];
        float x = sample.w;
        float p = (polynomial.x * x + polynomial.y) * x + polynomial.z;
        float root = sqrt(1 + p * p);
        float3 derivative = sample.xyz / (2 * root * root * root);
        result.Value += sample.xyz * ComputeRgbSigmoid(p);
        result.Quadratic += derivative * (x * x);
        result.Linear += derivative * x;
        result.Offset += derivative;
    }
    return result;
}
float ComputeRgbLargest(float3 value)
{
    return max(abs(value.x), max(abs(value.y), abs(value.z)));
}
RgbSpectrum FitRgbReflectance(float3 rgb, StructuredBuffer<float4> response)
{
    RgbSpectrum fit;
    fit.Quadratic = fit.Linear = fit.Offset = 0;
    fit.Scale = 0;
    fit.Constant = 0;
    fit.Error = RgbUnknownError;
    fit.Status = SpectrumFitInvalid;
    fit.Iterations = 0;
    if (!isfinite(rgb.x) || !isfinite(rgb.y) || !isfinite(rgb.z) || min(rgb.x, min(rgb.y, rgb.z)) < 0 ||
        max(rgb.x, max(rgb.y, rgb.z)) > 1)
        return fit;
    if (rgb.x == rgb.y && rgb.y == rgb.z)
    {
        fit.Constant = rgb.x;
        fit.Error = 0;
        fit.Status = SpectrumFitConverged;
        return fit;
    }
    // Fit chromatic contrast on an RGB-cube edge, then restore the grey floor.
    // This keeps dark/near-white inputs out of singular basins and bounds every
    // wavelength by the original minimum and maximum RGB channels.
    fit.Constant = min(rgb.x, min(rgb.y, rgb.z));
    fit.Scale = max(rgb.x, max(rgb.y, rgb.z)) - fit.Constant;
    float3 chromatic = (rgb - float3(fit.Constant, fit.Constant, fit.Constant)) / fit.Scale;
    float3 coefficients = float3(0, 0, 0);
    fit.Status = SpectrumFitUnconverged;
    for (uint iteration = 0; iteration <= RgbNewtonLimit; ++iteration)
    {
        ComputeRgbIntegral integral = ComputeRgbIntegrate(coefficients, response);
        float3 residual = integral.Value - chromatic;
        fit.Quadratic = coefficients.x;
        fit.Linear = coefficients.y;
        fit.Offset = coefficients.z;
        fit.Error = ComputeRgbLargest(residual);
        fit.Iterations = iteration;
        if (fit.Error <= RgbFitTolerance)
        {
            fit.Status = SpectrumFitConverged;
            break;
        }
        if (iteration == RgbNewtonLimit)
            break;
        // Solve the three-column Jacobian with relative determinant guarding.
        float3 yz = cross(integral.Linear, integral.Offset);
        float determinant = dot(integral.Quadratic, yz);
        float scale = ComputeRgbLargest(integral.Quadratic) * ComputeRgbLargest(integral.Linear) *
                      ComputeRgbLargest(integral.Offset);
        if (abs(determinant) <= RgbDeterminantTolerance * scale)
            break;
        float3 step = float3(dot(residual, yz), dot(integral.Quadratic, cross(residual, integral.Offset)),
                             dot(integral.Quadratic, cross(integral.Linear, residual))) /
                      determinant;
        float fraction = 1;
        bool accepted = false;
        for (uint search = 0; search < RgbBacktrackLimit; ++search)
        {
            float3 trial = coefficients - step * fraction;
            if (isfinite(trial.x) && isfinite(trial.y) && isfinite(trial.z) &&
                ComputeRgbLargest(trial) <= RgbCoefficientLimit)
            {
                float3 error = ComputeRgbIntegrate(trial, response).Value - chromatic;
                if (dot(error, error) < dot(residual, residual))
                {
                    coefficients = trial;
                    accepted = true;
                    break;
                }
            }
            fraction /= 2;
        }
        if (!accepted)
            break;
    }
    return fit;
}
RgbSpectrum FitRgbIlluminant(float3 rgb, StructuredBuffer<float4> response)
{
    bool valid = isfinite(rgb.x) && isfinite(rgb.y) && isfinite(rgb.z) && min(rgb.x, min(rgb.y, rgb.z)) >= 0;
    float scale = valid ? max(rgb.x, max(rgb.y, rgb.z)) : 0;
    float3 normalized = scale > 0 ? rgb / scale : float3(0, 0, 0);
    RgbSpectrum fit = FitRgbReflectance(normalized, response);
    fit.Scale *= scale;
    fit.Constant *= scale;
    if (!valid)
    {
        fit.Status = SpectrumFitInvalid;
        fit.Error = RgbUnknownError;
    }
    // The residual is for normalized chromatic contrast, before floor/exposure.
    return fit;
}
float EvaluateRgbReflectance(RgbSpectrum fit, float wavelength)
{
    if (fit.Status == SpectrumFitInvalid || !isfinite(wavelength) || wavelength < VisibleMinimumNm ||
        wavelength > VisibleMaximumNm)
        return 0;
    float x = (wavelength - RgbCenterNm) / RgbRadiusNm;
    float p = (fit.Quadratic * x + fit.Linear) * x + fit.Offset;
    return fit.Constant + fit.Scale * ComputeRgbSigmoid(p);
}
float EvaluateRgbIlluminant(RgbSpectrum fit, float wavelength)
{
    return RgbReferenceIlluminant(wavelength) * EvaluateRgbReflectance(fit, wavelength);
}
RgbSpectrum FitSceneSpectrum(SpectralSource source, StructuredBuffer<float4> response)
{
    const float3 rgb = float3(source.Red, source.Green, source.Blue);
    if (source.Kind == SpectrumRoleReflectance)
        return FitRgbReflectance(rgb, response);
    if (source.Kind == SpectrumRoleIlluminant || source.Kind == SpectrumRoleNonnegative)
        return FitRgbIlluminant(rgb, response);
    RgbSpectrum invalid;
    invalid.Quadratic = invalid.Linear = invalid.Offset = invalid.Scale = invalid.Constant = 0;
    invalid.Error = RgbUnknownError;
    invalid.Status = SpectrumFitInvalid;
    invalid.Iterations = 0;
    return invalid;
}
