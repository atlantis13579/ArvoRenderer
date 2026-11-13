// Independent Metal in-scattering estimator. Analytic connections include
// volume transmittance even when surface shadow tests are disabled. Rectangle
// and phase proposals use n*p solid-angle MIS, not surface cosine densities.
float3 VolumeConnection(DirectLightContext context, float3 point, float3 direction, float distance,
                        float3 radiance, bool shadows, thread uint2& shadowRays, thread uint2& primitiveTests)
{
    float visible = 1;
    if (shadows && context.ShadowStrength > 0)
    {
        RayData ray;
        ray.Origin = point;
        ray.Direction = direction;
        ray.MinimumDistance = RayEpsilon;
        ray.MaximumDistance = distance - RayEpsilon;
        HitData hit{};
        hit.Distance = ray.MaximumDistance;
        IncrementCounter(shadowRays);
        if (IntersectScene(context.Geometry, ray, true, hit, primitiveTests))
            visible -= clamp(context.ShadowStrength, 0.0f, 1.0f);
    }
    return radiance * visible * MediumTransmission(context.Medium, point, direction, 0, distance);
}
float3 VolumeDirectLighting(DirectLightContext context, float3 point, float3 incoming, SampleContext samples,
                            uint bounce, thread uint2& shadowRays, thread uint2& primitiveTests)
{
    float3 result = float3(0);
    const bool nee = (context.TransportFlags & NeeBit) != 0;
    const bool importance = (context.TransportFlags & ImportanceBit) != 0;
    const bool mis = ((context.TransportFlags >> MisShift) & MisMask) != NoMis;
    const uint PhaseDirectScramble = 0x6c8e9cf5u;
    for (uint index = 0; index < context.LightCount; ++index)
    {
        const PackedLight light = context.Lights[index];
        if (light.Intensity <= 0)
            continue;
        const float3 center = float3(light.Position), normal = float3(light.Direction);
        float3 color = float3(light.Color);
#if defined(ARVO_SPECTRAL_TRANSPORT)
        color = SpectralLight(context.Spectral, color, index);
#endif
        const bool shadows = light.CastShadows != 0;
        if (light.Type == LightTypeDirectional)
        {
            result += VolumeConnection(context, point, normal, MaximumRayDistance, color * light.Intensity,
                                       shadows, shadowRays, primitiveTests) *
                      MediumPhase(dot(incoming, normal), context.Medium.Anisotropy);
            continue;
        }
        if (light.Type == LightTypePoint)
        {
            const float3 offset = center - point;
            const float squared = dot(offset, offset);
            if (squared <= RayEpsilon * RayEpsilon)
                continue;
            const float distance = sqrt(squared);
            const float3 direction = offset / distance;
            result +=
                VolumeConnection(context, point, direction, distance, color * (light.Intensity / squared),
                                 shadows, shadowRays, primitiveTests) *
                MediumPhase(dot(incoming, direction), context.Medium.Anisotropy);
            continue;
        }
        const float area = light.Width * light.Height;
        if (light.Type != LightTypeRectArea || area <= 0)
            continue;
        const float3 reference = abs(normal.y) < BasisParallelThreshold ? float3(0, 1, 0) : float3(1, 0, 0);
        const float3 tangent = normalize(cross(reference, normal)), bitangent = cross(normal, tangent);
        const float3 emission = color * (light.Intensity / area);
        const uint domain = (bounce + 1) * BounceSeedMultiplier ^ (index + 1) * LightSeedMultiplier;
        for (uint sample = 0; nee && sample < AreaLightSamples; ++sample)
        {
            const float2 random = Sample2D(samples, domain ^ (sample + 1) * LightSampleSeedMultiplier);
            const float2 strata =
                float2(float(sample % AreaLightStrataPerAxis), float(sample / AreaLightStrataPerAxis));
            const float2 uv = (strata + random) / float(AreaLightStrataPerAxis) - 0.5f;
            const float3 endpoint =
                center + tangent * (uv.x * light.Width) + bitangent * (uv.y * light.Height);
            const float3 offset = endpoint - point;
            const float squared = dot(offset, offset);
            if (squared <= RayEpsilon * RayEpsilon)
                continue;
            const float distance = sqrt(squared);
            const float3 direction = offset / distance;
            const float facing = -dot(direction, normal);
            if (facing <= 0)
                continue;
            const float density = squared / (area * facing);
            const float phase = MediumPhase(dot(incoming, direction), context.Medium.Anisotropy);
            const float competitor = importance ? phase : 1 / (4 * Pi);
            const float weight = MisWeight(AreaLightSamples * density, competitor, context.TransportFlags) *
                                 phase / (AreaLightSamples * density);
            result += VolumeConnection(context, point, direction, distance, emission, shadows, shadowRays,
                                       primitiveTests) *
                      weight;
        }
        if (!nee || mis)
        {
            float density;
            const float3 direction =
                MediumSampleDirection(incoming, context.Medium.Anisotropy,
                                      Sample2D(samples, domain ^ PhaseDirectScramble), importance, density);
            const float facing = -dot(direction, normal);
            if (facing <= 0 || density <= 0)
                continue;
            const float distance = dot(point - center, normal) / facing;
            if (distance <= RayEpsilon)
                continue;
            const float3 offset = point + direction * distance - center;
            if (abs(dot(offset, tangent)) > light.Width / 2 || abs(dot(offset, bitangent)) > light.Height / 2)
                continue;
            const float lightDensity = distance * distance / (area * facing);
            const float weight =
                (nee ? MisWeight(density, AreaLightSamples * lightDensity, context.TransportFlags) : 1) *
                MediumPhase(dot(incoming, direction), context.Medium.Anisotropy) / density;
            result += VolumeConnection(context, point, direction, distance, emission, shadows, shadowRays,
                                       primitiveTests) *
                      weight;
        }
    }
    return result;
}
