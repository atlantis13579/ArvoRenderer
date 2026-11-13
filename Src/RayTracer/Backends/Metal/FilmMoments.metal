// Native MSL running moments, following Welford (1962). The fourth lane is
// luminance M2; it is not alpha and is never modified by display encoding.
inline void AccumulateFilmSample(thread float4& moments, thread uint& count, float3 radiance)
{
    const float3 LinearSrgbLuminance = float3(0.2126f, 0.7152f, 0.0722f);
    if (count == 0u)
    {
        moments = float4(radiance, 0.0f);
        count = 1u;
        return;
    }
    const float3 delta = radiance - moments.xyz;
    const float deviation = dot(delta, LinearSrgbLuminance);
    const float oldCount = float(count);
    ++count;
    moments.xyz += delta / float(count);
    moments.w += deviation * deviation * (oldCount / float(count));
}
