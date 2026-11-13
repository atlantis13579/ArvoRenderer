// CUDA's film stores a running RGB mean and a central luminance moment.
// Welford (1962) avoids catastrophic cancellation in E[L^2]-E[L]^2.
__device__ void AccumulateFilmSample(Vec4& moments, unsigned int& count, Vec3 radiance)
{
    constexpr float LuminanceRed = 0.2126f;
    constexpr float LuminanceGreen = 0.7152f;
    constexpr float LuminanceBlue = 0.0722f;
    if (count == 0u)
    {
        moments = {radiance.X, radiance.Y, radiance.Z, 0.0f};
        count = 1u;
        return;
    }
    const Vec3 delta{radiance.X - moments.X, radiance.Y - moments.Y, radiance.Z - moments.Z};
    const float luminanceDelta = delta.X * LuminanceRed + delta.Y * LuminanceGreen + delta.Z * LuminanceBlue;
    const float oldCount = static_cast<float>(count);
    const float newCount = static_cast<float>(++count);
    moments.X += delta.X / newCount;
    moments.Y += delta.Y / newCount;
    moments.Z += delta.Z / newCount;
    moments.W += luminanceDelta * luminanceDelta * (oldCount / newCount);
}
