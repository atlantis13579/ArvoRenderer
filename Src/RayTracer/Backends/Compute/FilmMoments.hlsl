// Native compute Welford accumulation. xyz is mean RGB; w is luminance M2.
// Welford (1962), "Note on a Method for Calculating Corrected Sums of Squares
// and Products". Resolve applies the display transform only after accumulation.
void AccumulateFilmSample(inout float4 moments, inout uint count, float3 radiance)
{
    static const float3 LinearSrgbLuminance = float3(0.2126f, 0.7152f, 0.0722f);
    if (count == 0u)
    {
        moments = float4(radiance, 0.0f);
        count = 1u;
        return;
    }
    const float3 delta = radiance - moments.xyz;
    const float luminanceDelta = dot(delta, LinearSrgbLuminance);
    const float oldCount = float(count);
    ++count;
    moments.xyz += delta / float(count);
    moments.w += luminanceDelta * luminanceDelta * (oldCount / float(count));
}
