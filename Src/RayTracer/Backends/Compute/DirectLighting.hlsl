// Native compute direct estimator: area-to-solid-angle conversion, independent
// light/BSDF samples, and Veach balance/power MIS with sample-count densities.
// Ideal BSDF events have discrete masses, so their rectangle connections use
// weight one and never compete with a continuous light proposal. Continuation
// must then exclude registered emitter-mesh hits, including after delta events.
float PhysicalVisibility(SurfaceBsdf surface, float3 position, float3 direction,
    float3 endpoint, bool finiteLight, bool castShadows, inout uint2 shadowRays, inout uint2 primitiveTests, float sceneTime = 0)
{
    if (!castShadows || ShadowStrength <= 0) return 1;
    RayData shadow;
    shadow.Origin = SurfaceOffset(surface, position, direction);
    shadow.Direction = direction;
    shadow.MinimumDistance = RayEpsilon;
    shadow.MaximumDistance = MaximumRayDistance;
    if (finiteLight)
    {
        const float3 segment = endpoint - shadow.Origin;
        const float distance = length(segment);
        shadow.MaximumDistance = distance - RayEpsilon;
        if (shadow.MaximumDistance <= shadow.MinimumDistance) return 1;
        shadow.Direction = segment / distance;
    }
    HitData hit = (HitData)0;
    hit.Distance = shadow.MaximumDistance;
    hit.PrimitiveIndex = InvalidPrimitiveIndex;
    IncrementCounter(shadowRays);
    return IntersectScene(shadow, true, hit, primitiveTests, sceneTime) ? 1 - saturate(ShadowStrength) : 1;
}
#if defined(ARVO_PARTICIPATING_MEDIA)
float3 PhysicalConnectionBeam(SurfaceBsdf surface, float3 position, float3 direction, float distance,
                               float wavelength, float currentInterior)
{
    const float inside = surface.Material.Optical.w;
    const float meanFreePath = inside > 0 ? (dot(surface.GeometricNormal, direction) < 0 ? inside : 0) : currentInterior;
    if (meanFreePath > 0) return exp(-distance / meanFreePath).xxx;
    MediumParameters medium = ParticipatingMedium;
#if defined(ARVO_SPECTRAL_TRANSPORT)
    medium = SpectralMedium(medium, wavelength);
#endif
    return MediumTransmission(medium, position, direction, 0, distance);
}
#endif
float3 PhysicalLightSample(SurfaceBsdf surface, float3 position, float3 direction, float3 radiance,
    float3 endpoint, bool finiteLight, bool shadows, float weight,
    inout uint2 shadowRays, inout uint2 primitiveTests, float wavelength = 0, float sceneTime = 0, float interiorMeanFreePath = 0)
{
    const SurfaceEvaluation scattering = EvaluateSurface(surface, direction);
    if (all(scattering.Value == 0)) return 0.0f.xxx;
    const float visible = PhysicalVisibility(surface, position, direction, endpoint, finiteLight,
                                             shadows, shadowRays, primitiveTests, sceneTime);
#if defined(ARVO_PARTICIPATING_MEDIA)
    radiance *= PhysicalConnectionBeam(surface, position, direction,
        finiteLight ? length(endpoint - position) : MaximumRayDistance, wavelength, interiorMeanFreePath);
#endif
    return scattering.Value * radiance * (abs(dot(surface.Normal, direction)) * weight * visible);
}
float3 PhysicalDirectLighting(SurfaceBsdf surface, float3 position, SampleContext samples, uint bounce,
    inout uint2 shadowRays, inout uint2 primitiveTests, float wavelength = 0, float sceneTime = 0, float interiorMeanFreePath = 0)
{
    float3 direct = 0.0f.xxx;
    const bool delta = SurfaceDelta(surface);
    const bool nee = (TransportFlags & NeeBit) != 0u;
    const uint heuristic = (TransportFlags >> MisShift) & MisMask;
    [loop]
    for (uint lightIndex = 0; lightIndex < LightCount; ++lightIndex)
    {
        PackedLight light = Lights[lightIndex];
#if defined(ARVO_SPECTRAL_TRANSPORT)
        light.Color = SpectralLight(light.Color, lightIndex, wavelength);
#endif
        if (light.Intensity <= 0) continue;
        if (light.Type == DirectionalLightType)
        {
            direct += PhysicalLightSample(surface, position, light.Direction, light.Color * light.Intensity,
                0.0f.xxx, false, light.CastShadows != 0u, 1, shadowRays, primitiveTests, wavelength, sceneTime, interiorMeanFreePath);
            continue;
        }
        if (light.Type == PointLightType)
        {
            const float3 offset = light.Position - position;
            const float squaredDistance = dot(offset, offset);
            if (squaredDistance > RayEpsilon * RayEpsilon)
                direct += PhysicalLightSample(surface, position, offset / sqrt(squaredDistance),
                    light.Color * (light.Intensity / squaredDistance), light.Position, true,
                    light.CastShadows != 0u, 1, shadowRays, primitiveTests, wavelength, sceneTime, interiorMeanFreePath);
            continue;
        }
        if (light.Type != RectAreaLightType || light.Width <= 0 || light.Height <= 0) continue;
        const float area = light.Width * light.Height;
        const float3 helper = abs(light.Direction.y) < BasisParallelThreshold
            ? float3(0, 1, 0) : float3(1, 0, 0);
        const float3 tangent = normalize(cross(helper, light.Direction));
        const float3 bitangent = cross(light.Direction, tangent);
        const float3 radiance = light.Color * (light.Intensity / area);
        const uint dimension = ((bounce + 1u) * BounceSeedMultiplier)
            ^ ((lightIndex + 1u) * LightSeedMultiplier);
        [loop]
        for (uint index = 0; nee && !delta && index < AreaLightSamples; ++index)
        {
            const float2 variates = Sample2D(samples, dimension ^ ((index + 1u) * LightSampleSeedMultiplier));
            const float2 uv = (float2(index & 1u, index >> 1u) + variates) / AreaLightStrataPerAxis - 0.5f;
            const float3 endpoint = light.Position + tangent * (uv.x * light.Width)
                + bitangent * (uv.y * light.Height);
            const float3 offset = endpoint - position;
            const float squaredDistance = dot(offset, offset);
            if (squaredDistance <= RayEpsilon * RayEpsilon) continue;
            const float3 direction = offset / sqrt(squaredDistance);
            const float emitterCosine = dot(light.Direction, -direction);
            if (emitterCosine <= 0) continue;
            const float lightPdf = squaredDistance / (area * emitterCosine);
            const float bsdfPdf = EvaluateSurface(surface, direction).Pdf;
            const float weight = MisWeight(AreaLightSamples * lightPdf, bsdfPdf) / (AreaLightSamples * lightPdf);
            direct += PhysicalLightSample(surface, position, direction, radiance, endpoint, true,
                light.CastShadows != 0u, weight, shadowRays, primitiveTests, wavelength, sceneTime, interiorMeanFreePath);
        }
        if (delta || !nee || heuristic != NoMis)
        {
            const uint bsdfDimension = dimension ^ DirectBsdfScramble;
            const SurfaceSample sample = SampleSurface(surface, Sample2D(samples, bsdfDimension),
                Sample2D(samples, bsdfDimension ^ LobeSelectionScramble).x);
            if (sample.Event == 0u || sample.Pdf <= 0) continue;
            const float emitterCosine = dot(light.Direction, -sample.Direction);
            if (emitterCosine <= 0) continue;
            const float distance = dot(position - light.Position, light.Direction) / emitterCosine;
            if (distance <= RayEpsilon) continue;
            const float3 endpoint = position + sample.Direction * distance;
            const float3 offset = endpoint - light.Position;
            if (abs(dot(offset, tangent)) > light.Width * 0.5f
                || abs(dot(offset, bitangent)) > light.Height * 0.5f) continue;
            const float lightPdf = distance * distance / (area * emitterCosine);
            const float weight = (sample.Event & ScatteringDelta) != 0u || !nee ? 1
                : MisWeight(sample.Pdf, AreaLightSamples * lightPdf);
            float3 connectionRadiance = radiance;
#if defined(ARVO_PARTICIPATING_MEDIA)
            connectionRadiance *= PhysicalConnectionBeam(surface, position, sample.Direction, distance,
                                                          wavelength, interiorMeanFreePath);
#endif
            direct += sample.Weight * connectionRadiance * weight
                * PhysicalVisibility(surface, position, sample.Direction, endpoint, true,
                                     light.CastShadows != 0u, shadowRays, primitiveTests, sceneTime);
        }
    }
    return direct;
}
