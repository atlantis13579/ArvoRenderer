// DXR's independent uniform discrete light selection. PBRT 4e, "Light Sampling".
// The return probability belongs to source identity, not the chosen endpoint's
// area or direction. Retain it in both initial throughput and path-density MIS.
static const uint UniformEmitterLimit=1u<<24; // Binary32 significand bits.
static const uint InvalidEmitterSelection=0xffffffffu;
struct EmitterSelection { uint Index; float Probability; };
float EmitterProbability(uint count,uint index)
{
    if (index >= count || count > UniformEmitterLimit) return 0;
    return rcp(float(count));
}
EmitterSelection SelectEmitter(uint count,float u)
{
    EmitterSelection selected;
    selected.Index=InvalidEmitterSelection; selected.Probability=0;
    if (count > 0 && count <= UniformEmitterLimit && u >= 0 && u < 1)
    {
        uint candidate=uint(u*float(count));
        selected.Index=min(candidate,count-1);
        selected.Probability=1.0f/float(count);
    }
    return selected;
}
