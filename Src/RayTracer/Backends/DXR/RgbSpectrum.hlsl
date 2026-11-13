#include "SpectralData.hlsli"
// DXR-owned RGB upsampling for spectral offline jobs; preview retains RGB.
// Bounded sigmoid polynomial: Jakob & Hanika (2019), "A Low-Dimensional Function
// Space for Efficient Spectral Upsampling", https://rgl.epfl.ch/publications/Jakob2019Spectral
// Preparation computes coefficients; ray hits only evaluate the resulting metamer.
static const float DxrRgbMidpoint = (VisibleMinimumNm + VisibleMaximumNm) / 2;
static const float DxrRgbHalfRange = (VisibleMaximumNm - VisibleMinimumNm) / 2;
static const uint DxrRgbIterations = 48, DxrRgbSearchSteps = 16;
static const float DxrRgbTolerance = 0.00001f, DxrRgbPivotTolerance = 0.0000001f;
static const float DxrRgbCoefficientBound = 1000000;
static const float DxrRgbErrorUnavailable = -1;
float DxrRgbBoundedValue(float polynomial)
{
    const float magnitude = abs(polynomial);
    const float hypotenuse = sqrt(1 + magnitude * magnitude);
    const float smallSide = 0.5f / (hypotenuse * (hypotenuse + magnitude));
    return polynomial >= 0 ? 1 - smallSide : smallSide;
}
float RgbReferenceIlluminant(float wavelength)
{
    if (!isfinite(wavelength) || wavelength < VisibleMinimumNm || wavelength > VisibleMaximumNm)
        return 0;
    // Synthetic positive white matched to our analytic CIE observer and rounded
    // sRGB matrix. D65 chromaticity does not make this a measured D65 spectrum.
    const float Quadratic = -0.6592809648409116f, Linear = -0.6601308563300832f;
    const float Intercept = -0.06589824578190885f;
    float coordinate = (wavelength - DxrRgbMidpoint) / DxrRgbHalfRange;
    return exp(Quadratic * coordinate * coordinate + Linear * coordinate + Intercept);
}
float4 MakeRgbFitResponse(uint index)
{
    const float YNormalization = 106.94617150055822f;
    const float interval = (VisibleMaximumNm - VisibleMinimumNm) / RgbFitSampleCount;
    const float wavelength = VisibleMinimumNm + interval * (index + 0.5f);
    return float4(XYZToLinearSRGB(CieMatching(wavelength)) *
                      (interval * RgbReferenceIlluminant(wavelength) / YNormalization),
                  (wavelength - DxrRgbMidpoint) / DxrRgbHalfRange);
}
struct DxrRgbEvaluation
{
    float3 Color;
    float3 RedDerivative, GreenDerivative, BlueDerivative;
};
DxrRgbEvaluation DxrRgbEvaluate(float3 coefficients, StructuredBuffer<float4> table)
{
    DxrRgbEvaluation value;
    value.Color = value.RedDerivative = value.GreenDerivative = value.BlueDerivative = float3(0, 0, 0);
    for (uint index = 0; index != RgbFitSampleCount; ++index)
    {
        const float4 quadrature = table[index];
        const float3 basis = float3(quadrature.w * quadrature.w, quadrature.w, 1);
        const float p = dot(coefficients, basis);
        const float root = sqrt(1 + p * p);
        const float3 gradient = basis / (2 * root * root * root);
        value.Color += quadrature.xyz * DxrRgbBoundedValue(p);
        value.RedDerivative += gradient * quadrature.x;
        value.GreenDerivative += gradient * quadrature.y;
        value.BlueDerivative += gradient * quadrature.z;
    }
    return value;
}
float DxrRgbMaxAbs(float3 v)
{
    return max(abs(v.x), max(abs(v.y), abs(v.z)));
}
struct DxrRgbStep
{
    float3 Value;
    bool Valid;
};
DxrRgbStep DxrRgbSolve(DxrRgbEvaluation evaluation, float3 residual)
{
    float rows[3][4];
    rows[0][0] = evaluation.RedDerivative.x;
    rows[0][1] = evaluation.RedDerivative.y;
    rows[0][2] = evaluation.RedDerivative.z;
    rows[0][3] = residual.x;
    rows[1][0] = evaluation.GreenDerivative.x;
    rows[1][1] = evaluation.GreenDerivative.y;
    rows[1][2] = evaluation.GreenDerivative.z;
    rows[1][3] = residual.y;
    rows[2][0] = evaluation.BlueDerivative.x;
    rows[2][1] = evaluation.BlueDerivative.y;
    rows[2][2] = evaluation.BlueDerivative.z;
    rows[2][3] = residual.z;
    const float magnitude =
        max(DxrRgbMaxAbs(evaluation.RedDerivative),
            max(DxrRgbMaxAbs(evaluation.GreenDerivative), DxrRgbMaxAbs(evaluation.BlueDerivative)));
    DxrRgbStep step;
    step.Value = float3(0, 0, 0);
    step.Valid = false;
    for (uint column = 0; column < 3; ++column)
    {
        uint pivot = column;
        for (uint row = column + 1; row < 3; ++row)
            if (abs(rows[row][column]) > abs(rows[pivot][column]))
                pivot = row;
        if (abs(rows[pivot][column]) <= magnitude * DxrRgbPivotTolerance)
            return step;
        for (uint field = column; field < 4; ++field)
        {
            float swap = rows[pivot][field];
            rows[pivot][field] = rows[column][field];
            rows[column][field] = swap;
        }
        for (uint other = column + 1; other < 3; ++other)
        {
            float ratio = rows[other][column] / rows[column][column];
            for (uint field = column; field < 4; ++field)
                rows[other][field] -= ratio * rows[column][field];
        }
    }
    step.Value.z = rows[2][3] / rows[2][2];
    step.Value.y = (rows[1][3] - rows[1][2] * step.Value.z) / rows[1][1];
    step.Value.x = (rows[0][3] - rows[0][2] * step.Value.z - rows[0][1] * step.Value.y) / rows[0][0];
    step.Valid = isfinite(step.Value.x) && isfinite(step.Value.y) && isfinite(step.Value.z);
    return step;
}
RgbSpectrum FitRgbReflectance(float3 target, StructuredBuffer<float4> table)
{
    RgbSpectrum fit;
    fit.Quadratic = fit.Linear = fit.Offset = 0;
    fit.Scale = 0;
    fit.Constant = 0;
    fit.Error = DxrRgbErrorUnavailable;
    fit.Iterations = 0;
    fit.Status = SpectrumFitInvalid;
    if (!isfinite(target.x) || !isfinite(target.y) || !isfinite(target.z) || target.x < 0 || target.y < 0 ||
        target.z < 0 || target.x > 1 || target.y > 1 || target.z > 1)
        return fit;
    if (target.x == target.y && target.y == target.z)
    {
        fit.Constant = target.x;
        fit.Error = 0;
        fit.Status = SpectrumFitConverged;
        return fit;
    }
    // Extract achromatic floor and contrast before Newton. The remaining color
    // lies on a cube edge; this avoids singular dark and almost-white fits.
    // Recombination is bounded by the source color's minimum/maximum channels.
    fit.Constant = min(target.x, min(target.y, target.z));
    fit.Scale = max(target.x, max(target.y, target.z)) - fit.Constant;
    target = (target - float3(fit.Constant, fit.Constant, fit.Constant)) / fit.Scale;
    float3 coefficients = float3(0, 0, 0);
    fit.Status = SpectrumFitUnconverged;
    for (uint iteration = 0; iteration <= DxrRgbIterations; iteration++)
    {
        const DxrRgbEvaluation evaluation = DxrRgbEvaluate(coefficients, table);
        const float3 difference = evaluation.Color - target;
        fit.Quadratic = coefficients.x;
        fit.Linear = coefficients.y;
        fit.Offset = coefficients.z;
        fit.Error = DxrRgbMaxAbs(difference);
        fit.Iterations = iteration;
        if (fit.Error <= DxrRgbTolerance)
        {
            fit.Status = SpectrumFitConverged;
            break;
        }
        if (iteration == DxrRgbIterations)
            break;
        const DxrRgbStep step = DxrRgbSolve(evaluation, difference);
        if (!step.Valid)
            break;
        bool moved = false;
        float damping = 1;
        for (uint search = 0; search < DxrRgbSearchSteps; ++search, damping *= 0.5f)
        {
            const float3 candidate = coefficients - damping * step.Value;
            if (DxrRgbMaxAbs(candidate) > DxrRgbCoefficientBound)
                continue;
            const float3 candidateError = DxrRgbEvaluate(candidate, table).Color - target;
            if (dot(candidateError, candidateError) < dot(difference, difference))
            {
                coefficients = candidate;
                moved = true;
                break;
            }
        }
        if (!moved)
            break;
    }
    return fit;
}
RgbSpectrum FitRgbIlluminant(float3 rgb, StructuredBuffer<float4> table)
{
    const bool valid =
        isfinite(rgb.x) && isfinite(rgb.y) && isfinite(rgb.z) && rgb.x >= 0 && rgb.y >= 0 && rgb.z >= 0;
    const float exposure = valid ? max(rgb.x, max(rgb.y, rgb.z)) : 0;
    RgbSpectrum fit = FitRgbReflectance(exposure > 0 ? rgb / exposure : float3(0, 0, 0), table);
    fit.Scale *= exposure;
    fit.Constant *= exposure;
    if (!valid)
    {
        fit.Status = SpectrumFitInvalid;
        fit.Error = DxrRgbErrorUnavailable;
    }
    return fit; // Error describes normalized chromatic contrast, before floor/exposure.
}
float EvaluateRgbReflectance(RgbSpectrum fit, float wavelength)
{
    if (fit.Status == SpectrumFitInvalid || !isfinite(wavelength) || wavelength < VisibleMinimumNm ||
        wavelength > VisibleMaximumNm)
        return 0;
    float coordinate = (wavelength - DxrRgbMidpoint) / DxrRgbHalfRange;
    float polynomial = fit.Quadratic * coordinate * coordinate + fit.Linear * coordinate + fit.Offset;
    return fit.Constant + fit.Scale * DxrRgbBoundedValue(polynomial);
}
float EvaluateRgbIlluminant(RgbSpectrum fit, float wavelength)
{
    return EvaluateRgbReflectance(fit, wavelength) * RgbReferenceIlluminant(wavelength);
}
RgbSpectrum FitSceneSpectrum(SpectralSource input, StructuredBuffer<float4> table)
{
    const float3 color = float3(input.Red, input.Green, input.Blue);
    if (input.Kind == SpectrumRoleReflectance)
        return FitRgbReflectance(color, table);
    if (input.Kind == SpectrumRoleIlluminant || input.Kind == SpectrumRoleNonnegative)
        return FitRgbIlluminant(color, table);
    RgbSpectrum fit;
    fit.Quadratic = fit.Linear = fit.Offset = fit.Scale = fit.Constant = 0;
    fit.Error = DxrRgbErrorUnavailable;
    fit.Status = SpectrumFitInvalid;
    fit.Iterations = 0;
    return fit;
}
