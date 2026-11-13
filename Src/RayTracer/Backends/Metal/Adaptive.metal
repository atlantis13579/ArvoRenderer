// Native MSL two-stage allocation: Kirk and Arvo, "Unbiased Sampling Techniques
// for Image Synthesis" (1991). Fixed independent pilot samples select the budget;
// a fresh production sequence supplies every sample retained in the film.
constant uint AdaptivePilotPhase = 1u;
constant uint AdaptivePilotScramble = 0x7f4a7c15u;
constant uint DenoisePilotAPhase = 2u, DenoisePilotBPhase = 3u;
constant uint DenoisePilotAScramble = 0x68bc21ebu, DenoisePilotBScramble = 0x02e5be93u;

uint AdaptiveSampleBudget(float4 moments, uint pilotCount, constant FrameConstants& constants)
{
    if (pilotCount < 2u || !all(isfinite(moments))) return constants.TotalSamples;
    const float3 LuminanceWeights = float3(0.2126f, 0.7152f, 0.0722f);
    const float mean = dot(moments.xyz, LuminanceWeights);
    const float threshold = constants.AdaptiveAbsoluteError + constants.AdaptiveRelativeError * abs(mean);
    const float variance = max(moments.w, 0.0f) / float(pilotCount - 1u);
    const float required = variance / (threshold * threshold);
    uint target = constants.TotalSamples;
    if (isfinite(required))
        target = uint(ceil(clamp(required, float(constants.AdaptiveMinimumSamples), float(constants.TotalSamples))));
    // Favor complete digital nets. Per-pixel budgets weaken the fixed-spp
    // neighboring-pixel blue-noise guarantee; this is an allocation heuristic.
    if (constants.SamplingSequence == SobolSequence || constants.SamplingSequence == BlueNoiseSequence)
    {
        uint netSize = 1u;
        while (netSize < target) netSize <<= 1u;
        target = min(netSize, constants.TotalSamples);
    }
    return target;
}

SampleContext AdaptiveSampleContext(uint x, uint y, uint sample, uint target,
    constant FrameConstants& constants)
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
