// Independent Metal uniform source selection. PBRT 4e, "Light Sampling".
// This is a discrete PMF, additional to an endpoint's area/solid-angle PDF.
// Emission throughput divides by it; endpoint path densities multiply by it.
constant uint MaximumUniformSources=1u<<24; // Binary32 significand bits.
constant uint InvalidSourceSelection=0xffffffffu;
struct EmitterSelection { uint Index; float Probability; };
EmitterSelection SelectEmitter(uint count,float variate)
{
    EmitterSelection selection{InvalidSourceSelection,0};
    if (count == 0 || count > MaximumUniformSources || !(variate >= 0 && variate < 1)) return selection;
    selection.Index=min(uint(variate*float(count)),count-1);
    selection.Probability=1.0f/float(count);
    return selection;
}
float EmitterProbability(uint count,uint index)
{
    if (count > MaximumUniformSources || index >= count) return 0;
    return 1.0f/float(count);
}
