// Independent compute volume direct estimator. Phase values replace f*cos;
// every connection includes analytic Beer attenuation and native visibility.
float3 VolumeIncident(MediumParameters medium, float3 position, float3 direction, float distance, float3 radiance, bool shadows,
                      inout uint2 shadowRays, inout uint2 primitiveTests, float wavelength = 0, float sceneTime = 0)
{
    float visible = 1;
    if (shadows && ShadowStrength > 0)
    {
        RayData ray;
        ray.Origin = position;
        ray.Direction = direction;
        ray.MinimumDistance = RayEpsilon;
        ray.MaximumDistance = distance - RayEpsilon;
        HitData hit = (HitData)0;
        hit.Distance = ray.MaximumDistance;
        hit.PrimitiveIndex = InvalidPrimitiveIndex;
        IncrementCounter(shadowRays);
        if (IntersectScene(ray, true, hit, primitiveTests, sceneTime))
            visible -= saturate(ShadowStrength);
    }
    return radiance * visible * MediumTransmission(medium, position, direction, 0, distance);
}
float3 VolumeDirectLighting(MediumParameters medium, float3 position, float3 incoming, SampleContext samples, uint bounce,
                            inout uint2 shadowRays, inout uint2 primitiveTests, float wavelength = 0, float sceneTime = 0)
{
    float3 direct = float3(0, 0, 0);
    const bool nee = (TransportFlags & NeeBit) != 0;
    const bool importance = (TransportFlags & ImportanceBit) != 0;
    const uint PhaseDirectScramble = 0x6c8e9cf5u;
    for (uint index = 0; index < LightCount; ++index)
    {
        PackedLight light = Lights[index];
#if defined(ARVO_SPECTRAL_TRANSPORT)
        light.Color = SpectralLight(light.Color, index, wavelength);
#endif
        if (light.Intensity <= 0)
            continue;
        if (light.Type == DirectionalLightType || light.Type == PointLightType)
        {
            float3 direction = light.Direction;
            float distance = MaximumRayDistance;
            float scale = light.Intensity;
            if (light.Type == PointLightType)
            {
                const float3 delta = light.Position - position;
                const float squared = dot(delta, delta);
                if (squared <= RayEpsilon * RayEpsilon)
                    continue;
                distance = sqrt(squared);
                direction = delta / distance;
                scale /= squared;
            }
            direct += VolumeIncident(medium, position, direction, distance, light.Color * scale,
                                     light.CastShadows != 0, shadowRays, primitiveTests, wavelength, sceneTime) *
                      MediumPhase(dot(incoming, direction), medium.Anisotropy);
            continue;
        }
        const float area = light.Width * light.Height;
        if (light.Type != RectAreaLightType || area <= 0)
            continue;
        const float3 helper =
            abs(light.Direction.y) < BasisParallelThreshold ? float3(0, 1, 0) : float3(1, 0, 0);
        const float3 tangent = normalize(cross(helper, light.Direction)),
                     bitangent = cross(light.Direction, tangent);
        const float3 emitted = light.Color * (light.Intensity / area);
        const uint domain = (bounce + 1) * BounceSeedMultiplier ^ (index + 1) * LightSeedMultiplier;
        for (uint sample = 0; nee && sample < AreaLightSamples; ++sample)
        {
            const float2 u = Sample2D(samples, domain ^ (sample + 1) * LightSampleSeedMultiplier);
            const float2 cell = float2(sample % AreaLightStrataPerAxis, sample / AreaLightStrataPerAxis);
            const float2 uv = (cell + u) / AreaLightStrataPerAxis - 0.5f;
            const float3 endpoint =
                light.Position + tangent * (uv.x * light.Width) + bitangent * (uv.y * light.Height);
            const float3 offset = endpoint - position;
            const float squared = dot(offset, offset);
            if (squared <= RayEpsilon * RayEpsilon)
                continue;
            const float distance = sqrt(squared);
            const float3 direction = offset / distance;
            const float facing = dot(light.Direction, -direction);
            if (facing <= 0)
                continue;
            const float density = squared / (area * facing);
            const float phase = MediumPhase(dot(incoming, direction), medium.Anisotropy);
            const float competitor = importance ? phase : 1 / (4 * Pi);
            const float weight =
                MisWeight(AreaLightSamples * density, competitor) * phase / (AreaLightSamples * density);
            direct += VolumeIncident(medium, position, direction, distance, emitted, light.CastShadows != 0,
                                     shadowRays, primitiveTests, wavelength, sceneTime) *
                      weight;
        }
        if (!nee || ((TransportFlags >> MisShift) & MisMask) != NoMis)
        {
            float pdf;
            const float3 direction =
                MediumSampleDirection(incoming, medium.Anisotropy,
                                      Sample2D(samples, domain ^ PhaseDirectScramble), importance, pdf);
            const float facing = dot(light.Direction, -direction);
            if (facing <= 0 || pdf <= 0)
                continue;
            const float distance = dot(position - light.Position, light.Direction) / facing;
            if (distance <= RayEpsilon)
                continue;
            const float3 offset = position + direction * distance - light.Position;
            if (abs(dot(offset, tangent)) > light.Width / 2 || abs(dot(offset, bitangent)) > light.Height / 2)
                continue;
            const float density = distance * distance / (area * facing);
            const float weight = (nee ? MisWeight(pdf, AreaLightSamples * density) : 1) *
                                 MediumPhase(dot(incoming, direction), medium.Anisotropy) / pdf;
            direct += VolumeIncident(medium, position, direction, distance, emitted, light.CastShadows != 0,
                                     shadowRays, primitiveTests, wavelength, sceneTime) *
                      weight;
        }
    }
    return direct;
}
