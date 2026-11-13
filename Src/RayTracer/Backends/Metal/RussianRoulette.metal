// Native MSL roulette policy. Surviving paths receive throughput/p in the
// integrator; the lower bound limits weights and the upper bound permits exit.
float RouletteProbability(float3 throughput)
{
    const float MinimumSurvival = 0.05f;
    const float MaximumSurvival = 0.95f;
    const float energy = max(throughput.x, max(throughput.y, throughput.z));
    return energy > 0.0f ? clamp(energy, MinimumSurvival, MaximumSurvival) : 0.0f;
}
