// Independent DXR phase/light estimator, shared by offline and live preview.
// Convert area proposals to solid angle before Veach MIS. Separate phase
// samples own direct emitter paths; continuation excludes duplicate hits.
float3 VolumeConnection(MediumParameters medium, float3 position, float3 direction, float distance, float3 emission, bool shadows,
                        inout uint2 shadowRays, float wavelength = 0)
{
    float visibility = 1;
    if (shadows && ShadowStrength > 0)
    {
        IncrementLocalCounter(shadowRays);
        const Payload blocker = CastRay(position, direction, RayEpsilon, distance - RayEpsilon);
        if ((blocker.Flags & SurfaceHitFlag) != 0)
            visibility -= saturate(ShadowStrength);
    }
    return emission * visibility * MediumTransmission(medium, position, direction, 0, distance);
}
float3 VolumeDirectLighting(MediumParameters medium, float3 position, float3 incoming, SampleContext samples, uint bounce,
                            inout uint2 shadowRays, float wavelength = 0)
{
    float3 estimate = float3(0, 0, 0);
    const bool nee = (TransportFlags & NeeBit) != 0;
    const bool importance = (TransportFlags & ImportanceBit) != 0;
    const bool combine = ((TransportFlags >> MisShift) & MisMask) != NoMis;
    const uint PhaseDirectScramble = 0x6c8e9cf5u;
    for (uint index = 0; index < LightCount; ++index)
    {
        LightData light = Lights[index];
#if defined(ARVO_SPECTRAL_TRANSPORT)
        light.Color = EvaluateSpectralLight(light.Color, index, wavelength);
#endif
        if (light.Intensity <= 0)
            continue;
        if (light.Type == DirectionalLightType)
        {
            const float phase = MediumPhase(dot(incoming, light.Direction), medium.Anisotropy);
            estimate += VolumeConnection(medium, position, light.Direction, MaximumRayDistance,
                                         light.Color * light.Intensity, light.CastShadows != 0, shadowRays, wavelength) *
                        phase;
            continue;
        }
        if (light.Type == PointLightType)
        {
            const float3 delta = light.Position - position;
            const float squaredDistance = dot(delta, delta);
            if (squaredDistance <= RayEpsilon * RayEpsilon)
                continue;
            const float distance = sqrt(squaredDistance);
            const float3 direction = delta / distance;
            estimate += VolumeConnection(medium, position, direction, distance,
                                         light.Color * (light.Intensity / squaredDistance),
                                         light.CastShadows != 0, shadowRays, wavelength) *
                        MediumPhase(dot(incoming, direction), medium.Anisotropy);
            continue;
        }
        const float area = light.Width * light.Height;
        if (light.Type != RectAreaLightType || area <= 0)
            continue;
        const float3 axis =
            abs(light.Direction.y) < BasisParallelThreshold ? float3(0, 1, 0) : float3(1, 0, 0);
        const float3 x = normalize(cross(axis, light.Direction)), y = cross(light.Direction, x);
        const uint domain = (bounce + 1) * BounceSeedMultiplier ^ (index + 1) * LightSeedMultiplier;
        const float3 emitted = light.Color * (light.Intensity / area);
        for (uint sample = 0; nee && sample < AreaLightSamples; ++sample)
        {
            const float2 jitter = Sample2D(samples, domain ^ (sample + 1) * LightSampleSeedMultiplier);
            const float2 cell = float2(sample % AreaLightStrataPerAxis, sample / AreaLightStrataPerAxis);
            const float2 uv = (cell + jitter) / AreaLightStrataPerAxis - 0.5f;
            const float3 endpoint = light.Position + x * (uv.x * light.Width) + y * (uv.y * light.Height);
            const float3 offset = endpoint - position;
            const float squared = dot(offset, offset);
            if (squared <= RayEpsilon * RayEpsilon)
                continue;
            const float distance = sqrt(squared);
            const float3 direction = offset / distance;
            const float facing = -dot(direction, light.Direction);
            if (facing <= 0)
                continue;
            const float lightDensity = squared / (area * facing);
            const float phase = MediumPhase(dot(incoming, direction), medium.Anisotropy);
            const float phaseDensity = importance ? phase : 1 / (4 * Pi);
            estimate +=
                VolumeConnection(medium, position, direction, distance, emitted, light.CastShadows != 0, shadowRays, wavelength) *
                (MisWeight(AreaLightSamples * lightDensity, phaseDensity) * phase /
                 (AreaLightSamples * lightDensity));
        }
        if (!nee || combine)
        {
            float phaseDensity;
            const float3 direction = MediumSampleDirection(incoming, medium.Anisotropy,
                                                           Sample2D(samples, domain ^ PhaseDirectScramble),
                                                           importance, phaseDensity);
            const float facing = -dot(direction, light.Direction);
            if (facing <= 0 || phaseDensity <= 0)
                continue;
            const float distance = dot(position - light.Position, light.Direction) / facing;
            if (distance <= RayEpsilon)
                continue;
            const float3 offset = position + direction * distance - light.Position;
            if (abs(dot(offset, x)) > light.Width / 2 || abs(dot(offset, y)) > light.Height / 2)
                continue;
            const float lightDensity = distance * distance / (area * facing);
            const float mis = nee ? MisWeight(phaseDensity, AreaLightSamples * lightDensity) : 1;
            estimate +=
                VolumeConnection(medium, position, direction, distance, emitted, light.CastShadows != 0, shadowRays, wavelength) *
                (mis * MediumPhase(dot(incoming, direction), medium.Anisotropy) / phaseDensity);
        }
    }
    return estimate;
}
