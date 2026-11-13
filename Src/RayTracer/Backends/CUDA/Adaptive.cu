// Kirk and Arvo (1991), "Unbiased Sampling Techniques for Image Synthesis".
// An independent pilot decides the production budget. Its samples never enter
// the output film, avoiding the bias of stopping on the image's own samples.
constexpr unsigned int AdaptivePilotPhase = 1u;
constexpr unsigned int AdaptivePilotScramble = 0x7f4a7c15u;
constexpr unsigned int DenoisePilotAPhase = 2u, DenoisePilotBPhase = 3u;
constexpr unsigned int DenoisePilotAScramble = 0x68bc21ebu, DenoisePilotBScramble = 0x02e5be93u;

__device__ unsigned int AdaptiveSampleBudget(Vec4 moments, unsigned int pilotCount,
    const FrameConstants& constants)
{
    if (pilotCount < 2u || !isfinite(moments.X) || !isfinite(moments.Y)
        || !isfinite(moments.Z) || !isfinite(moments.W)) return constants.TotalSamples;
    constexpr float LuminanceRed = 0.2126f;
    constexpr float LuminanceGreen = 0.7152f;
    constexpr float LuminanceBlue = 0.0722f;
    const float mean = moments.X * LuminanceRed + moments.Y * LuminanceGreen + moments.Z * LuminanceBlue;
    const float tolerance = constants.AdaptiveAbsoluteError + constants.AdaptiveRelativeError * fabsf(mean);
    const float variance = fmaxf(0.0f, moments.W) / static_cast<float>(pilotCount - 1u);
    const float required = variance / (tolerance * tolerance);
    unsigned int target = constants.TotalSamples;
    if (isfinite(required))
        target = static_cast<unsigned int>(ceilf(fminf(static_cast<float>(constants.TotalSamples),
            fmaxf(static_cast<float>(constants.AdaptiveMinimumSamples), required))));
    // Complete digital nets when the cap permits. Variable neighboring budgets
    // do not retain blue noise's fixed-spp screen-space net guarantee.
    if (constants.SamplingSequence == SobolSequence || constants.SamplingSequence == BlueNoiseSequence)
    {
        unsigned int netSize = 1u;
        while (netSize < target) netSize <<= 1u;
        target = netSize < constants.TotalSamples ? netSize : constants.TotalSamples;
    }
    return target;
}

__device__ SampleContext AdaptiveSampleContext(unsigned int x, unsigned int y,
    unsigned int sample, unsigned int target, const FrameConstants& constants)
{
    const bool pilot = constants.AdaptiveEnabled != 0u && constants.AdaptivePhase == AdaptivePilotPhase;
    SampleContext context = MakeSampleContext(x, y, constants.FullWidth, sample,
        pilot ? constants.AdaptivePilotSamples : target,
        pilot ? IndependentSequence : constants.SamplingSequence, constants.SamplingSeed);
    if (pilot) context.PixelSeed = Hash(context.PixelSeed ^ AdaptivePilotScramble);
    else if (constants.AdaptivePhase == DenoisePilotAPhase)
        context.PixelSeed = Hash(context.PixelSeed ^ DenoisePilotAScramble);
    else if (constants.AdaptivePhase == DenoisePilotBPhase)
        context.PixelSeed = Hash(context.PixelSeed ^ DenoisePilotBScramble);
    return context;
}
