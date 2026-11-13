// Native DXR two-stage budget selection. Pilot paths are independent MC and
// never contribute to the production film: Kirk and Arvo (1991), "Unbiased
// Sampling Techniques for Image Synthesis". The error target is a heuristic,
// not a confidence guarantee for QMC or rare light paths.
// https://authors.library.caltech.edu/records/8sntk-73q87
static const uint AdaptivePilotScramble = 0x7f4a7c15u;
static const uint DenoisePilotAPhase = 2u, DenoisePilotBPhase = 3u, PreparedBudgetPhase = 4u;
static const uint DenoisePilotAScramble = 0x68bc21ebu, DenoisePilotBScramble = 0x02e5be93u;

uint AdaptiveProductionCount(LinearFilmPixel pilot)
{
    static const float3 LuminanceWeights = float3(0.2126f, 0.7152f, 0.0722f);
    if (pilot.SampleCount < 2u || !isfinite(pilot.LuminanceM2)
        || !all(isfinite(pilot.MeanRadiance))) return SamplesPerPixel;
    const float luminance = dot(pilot.MeanRadiance, LuminanceWeights);
    const float allowedError = AdaptiveAbsoluteError + abs(luminance) * AdaptiveRelativeError;
    const float sampleVariance = max(pilot.LuminanceM2, 0.0f) / float(pilot.SampleCount - 1u);
    const float desiredCount = sampleVariance / (allowedError * allowedError);
    if (!isfinite(desiredCount)) return SamplesPerPixel;
    uint result = uint(ceil(clamp(desiredCount, float(AdaptiveMinimumSamples), float(SamplesPerPixel))));
    if (SamplingSequence == SobolSequence || SamplingSequence == BlueNoiseSequence)
    {
        uint completeNet = 1u;
        [loop] while (completeNet < result) completeNet <<= 1u;
        result = min(completeNet, SamplesPerPixel);
    }
    return result;
}

SampleContext AdaptiveSamples(uint2 pixel, uint sample, uint count, bool pilot)
{
    SampleContext context = MakeSampleContext(pixel.x + TileX, pixel.y + TileY,
        FullWidth, sample, count, pilot ? IndependentSequence : SamplingSequence, SamplingSeed);
    if (pilot) context.PixelSeed = Hash(context.PixelSeed ^ AdaptivePilotScramble);
    else if (AdaptivePhase == DenoisePilotAPhase) context.PixelSeed = Hash(context.PixelSeed ^ DenoisePilotAScramble);
    else if (AdaptivePhase == DenoisePilotBPhase) context.PixelSeed = Hash(context.PixelSeed ^ DenoisePilotBScramble);
    return context;
}
