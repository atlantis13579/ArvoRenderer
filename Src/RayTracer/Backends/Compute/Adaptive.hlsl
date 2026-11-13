// Kirk and Arvo (1991), "Unbiased Sampling Techniques for Image Synthesis".
// A fixed independent pilot estimates cost; a fresh production set supplies the
// image. Reusing pilot values in that mean would introduce stopping bias.
// https://authors.library.caltech.edu/records/8sntk-73q87
static const uint AdaptivePilotPhase = 1u;
static const uint AdaptivePilotScramble = 0x7f4a7c15u;
static const uint DenoisePilotAPhase = 2u, DenoisePilotBPhase = 3u;
static const uint DenoisePilotAScramble = 0x68bc21ebu, DenoisePilotBScramble = 0x02e5be93u;

uint AdaptiveSampleBudget(float4 moments, uint count)
{
    static const float3 LinearSrgbLuminance = float3(0.2126f, 0.7152f, 0.0722f);
    if (count < 2u || !all(isfinite(moments))) return TotalSamples;
    const float mean = dot(moments.xyz, LinearSrgbLuminance);
    const float tolerance = AdaptiveAbsoluteError + AdaptiveRelativeError * abs(mean);
    const float variance = max(0.0f, moments.w) / float(count - 1u);
    const float estimate = variance / (tolerance * tolerance);
    if (!isfinite(estimate)) return TotalSamples;
    uint budget = uint(ceil(clamp(estimate, float(AdaptiveMinimumSamples), float(TotalSamples))));
    // Complete QMC blocks are preferable; the user's maximum remains absolute.
    // Variable pixel budgets weaken the fixed-spp blue-noise neighborhood net.
    if (SamplingSequence == SobolSequence || SamplingSequence == BlueNoiseSequence)
    {
        uint block = 1u;
        [loop] while (block < budget) block <<= 1u;
        budget = min(block, TotalSamples);
    }
    return budget;
}

SampleContext AdaptiveSamples(uint x, uint y, uint sample, uint count, bool pilot)
{
    SampleContext context = MakeSampleContext(x, y, FullWidth, sample, count,
        pilot ? IndependentSequence : SamplingSequence, SamplingSeed);
    if (pilot) context.PixelSeed = Hash(context.PixelSeed ^ AdaptivePilotScramble);
    else if (AdaptivePhase == DenoisePilotAPhase) context.PixelSeed = Hash(context.PixelSeed ^ DenoisePilotAScramble);
    else if (AdaptivePhase == DenoisePilotBPhase) context.PixelSeed = Hash(context.PixelSeed ^ DenoisePilotBScramble);
    return context;
}
