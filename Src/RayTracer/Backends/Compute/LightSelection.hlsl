// Independent compute uniform discrete selection. The catalog contains each
// analytic source once and excludes its SampledEmitter geometry proxy.
// PBRT 4e, "Light Sampling". Multiply endpoint PDFs by this PMF for MIS and
// divide emitted flux by it; area / solid-angle PDFs remain separate measures.
static const uint MaximumUniformEmitterCount = 1u << 24; // Binary32 significand bits.
static const uint InvalidSelectedEmitter = 0xffffffffu;
struct EmitterSelection { uint Index; float Probability; };
EmitterSelection SelectEmitter(uint count,float variate)
{
    EmitterSelection result;
    result.Index=InvalidSelectedEmitter; result.Probability=0;
    if (count == 0 || count > MaximumUniformEmitterCount || !(variate >= 0 && variate < 1)) return result;
    result.Index=min(uint(variate*float(count)),count-1);
    result.Probability=1.0f/float(count);
    return result;
}
float EmitterProbability(uint count,uint index)
{
    return count <= MaximumUniformEmitterCount && index < count ? 1.0f/float(count) : 0;
}
