// The largest RGB component bounds survival; f/p compensation is performed
// by the native CUDA integrator after a path survives its independent draw.
__device__ float RouletteProbability(Vec3 throughput)
{
    constexpr float MinimumSurvival = 0.05f;
    constexpr float MaximumSurvival = 0.95f;
    const float energy = fmaxf(throughput.X, fmaxf(throughput.Y, throughput.Z));
    return energy > 0.0f ? fminf(MaximumSurvival, fmaxf(MinimumSurvival, energy)) : 0.0f;
}
