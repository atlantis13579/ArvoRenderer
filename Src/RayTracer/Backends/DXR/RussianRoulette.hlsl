// Russian roulette replaces deterministic low-energy truncation. The caller
// divides surviving path throughput by this probability to preserve expectation.
float RouletteProbability(float3 throughput)
{
    const float MinimumSurvival = 0.05f;
    const float MaximumSurvival = 0.95f;
    const float energy = max(throughput.x, max(throughput.y, throughput.z));
    if (energy <= 0.0f) return 0.0f;
    return clamp(energy, MinimumSurvival, MaximumSurvival);
}
