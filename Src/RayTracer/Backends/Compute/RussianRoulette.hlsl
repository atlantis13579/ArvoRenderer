// Terminate low-energy paths without bias: surviving throughput is divided by p.
// The upper bound guarantees occasional termination even for perfect mirrors.
float RouletteProbability(float3 throughput)
{
    const float MinimumSurvival = 0.05f;
    const float MaximumSurvival = 0.95f;
    const float energy = max(throughput.x, max(throughput.y, throughput.z));
    return energy > 0.0f ? clamp(energy, MinimumSurvival, MaximumSurvival) : 0.0f;
}
