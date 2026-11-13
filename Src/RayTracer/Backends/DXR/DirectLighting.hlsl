// DXR physical direct estimator: four stratified rectangle samples and one
// independent BSDF sample. MIS compares n*p in solid angle; delta events carry
// probability masses and have no competing area proposal. Veach (1997), ch. 9.
// This estimator owns registered emitters after *all* physical continuations,
// including ideal events, so analytic rectangles also work without emitter meshes.
static const uint AreaLightStrataPerAxis = 2;
#if defined(ARVO_PARTICIPATING_MEDIA)
float3 PhysicalConnectionBeam(SurfaceBsdf surface, float3 origin, float3 outgoing, float distance,
                               float wavelength, float interiorMeanFreePath)
{
    float meanFreePath = interiorMeanFreePath;
    if (surface.Material.Optical.w > 0)
    {
        // DXR's BSDF geometry normal faces the incident side, not always out
        // of the solid. Transmission toggles the explicit interior path state.
        const bool transmitted = dot(outgoing, surface.GeometricNormal) < 0;
        const bool inside = (interiorMeanFreePath > 0) != transmitted;
        meanFreePath = inside ? surface.Material.Optical.w : 0;
    }
    if (meanFreePath > 0)
    {
        const float survival = exp(-distance / meanFreePath);
        return float3(survival, survival, survival);
    }
    MediumParameters medium = ParticipatingMedium;
#if defined(ARVO_SPECTRAL_TRANSPORT)
    medium = EvaluateSpectralMedium(medium, wavelength);
#endif
    return MediumTransmission(medium, origin, outgoing, 0, distance);
}
#endif
float PhysicalVisibility(SurfaceBsdf surface, float3 position, float3 direction,
                         float3 endpoint, bool finiteLight, bool shadows, inout uint2 shadowRays)
{
    if (!shadows || ShadowStrength <= 0) return 1;
    const float3 origin = SurfaceOffset(surface, position, direction);
    float maximum = MaximumRayDistance;
    if (finiteLight)
    {
        const float3 segment = endpoint - origin;
        const float distance = length(segment);
        maximum = distance - RayEpsilon;
        if (maximum <= RayEpsilon) return 1;
        direction = segment / distance;
    }
    IncrementLocalCounter(shadowRays);
    const Payload blocker = CastRay(origin, direction, RayEpsilon, maximum);
    return (blocker.Flags & SurfaceHitFlag) != 0u ? 1 - saturate(ShadowStrength) : 1;
}
float3 PhysicalLightConnection(SurfaceBsdf surface, float3 position, float3 direction,
    float3 endpoint, float3 radiance, bool finiteLight, bool shadows, inout uint2 shadowRays, float wavelength = 0,
    float interiorMeanFreePath = 0)
{
    const SurfaceEvaluation scattering = EvaluateSurface(surface, direction);
    if (all(scattering.Value == 0)) return 0.0.xxx;
    const float visibility = PhysicalVisibility(surface, position, direction, endpoint, finiteLight,
                                                shadows, shadowRays);
#if defined(ARVO_PARTICIPATING_MEDIA)
    radiance *= PhysicalConnectionBeam(surface, position, direction,
        finiteLight ? length(endpoint - position) : MaximumRayDistance, wavelength, interiorMeanFreePath);
#endif
    return scattering.Value * radiance * (abs(dot(surface.Normal, direction)) * visibility);
}
float3 PhysicalDirectLighting(SurfaceBsdf surface, float3 position, SampleContext samples,
                              uint bounce, inout uint2 shadowRays, float wavelength = 0, float interiorMeanFreePath = 0)
{
    float3 direct = 0.0.xxx;
    const bool delta = SurfaceDelta(surface);
    const bool nee = (TransportFlags & NeeBit) != 0u;
    const bool combine = ((TransportFlags >> MisShift) & MisMask) != NoMis;
    [loop]
    for (uint index = 0; index < LightCount; ++index)
    {
        LightData light = Lights[index];
#if defined(ARVO_SPECTRAL_TRANSPORT)
        light.Color = EvaluateSpectralLight(light.Color, index, wavelength);
#endif
        if (light.Intensity <= 0) continue;
        const bool shadows = light.CastShadows != 0u;
        if (light.Type == DirectionalLightType)
        {
            direct += PhysicalLightConnection(surface, position, light.Direction, 0.0.xxx,
                light.Color * light.Intensity, false, shadows, shadowRays, wavelength, interiorMeanFreePath);
            continue;
        }
        if (light.Type == PointLightType)
        {
            const float3 offset = light.Position - position;
            const float distanceSquared = dot(offset, offset);
            if (distanceSquared <= RayEpsilon * RayEpsilon) continue;
            direct += PhysicalLightConnection(surface, position, offset / sqrt(distanceSquared), light.Position,
                light.Color * (light.Intensity / distanceSquared), true, shadows, shadowRays, wavelength, interiorMeanFreePath);
            continue;
        }
        if (light.Type != RectAreaLightType || light.Width <= 0 || light.Height <= 0) continue;
        const float area = light.Width * light.Height;
        const float3 emitted = light.Color * (light.Intensity / area);
        const float3 helper = abs(light.Direction.y) < BasisParallelThreshold
            ? float3(0, 1, 0) : float3(1, 0, 0);
        const float3 tangent = normalize(cross(helper, light.Direction));
        const float3 bitangent = cross(light.Direction, tangent);
        const uint dimension = ((bounce + 1u) * BounceSeedMultiplier) ^ ((index + 1u) * LightSeedMultiplier);
        if (nee && !delta)
        {
            [loop]
            for (uint sampleIndex = 0; sampleIndex < AreaLightSamples; ++sampleIndex)
            {
                const float2 random = Sample2D(samples, dimension ^ ((sampleIndex + 1u) * LightSampleSeedMultiplier));
                const float2 cell = float2(sampleIndex % AreaLightStrataPerAxis, sampleIndex / AreaLightStrataPerAxis);
                const float2 uv = (cell + random) / AreaLightStrataPerAxis - 0.5;
                const float3 endpoint = light.Position + tangent * (uv.x * light.Width)
                    + bitangent * (uv.y * light.Height);
                const float3 offset = endpoint - position;
                const float distanceSquared = dot(offset, offset);
                if (distanceSquared <= RayEpsilon * RayEpsilon) continue;
                const float3 direction = offset / sqrt(distanceSquared);
                const float facing = -dot(direction, light.Direction);
                if (facing <= 0) continue;
                const SurfaceEvaluation scattering = EvaluateSurface(surface, direction);
                if (all(scattering.Value == 0)) continue;
                const float lightPdf = distanceSquared / (area * facing);
                const float mis = combine ? MisWeight(AreaLightSamples * lightPdf, scattering.Pdf) : 1;
                const float visible = PhysicalVisibility(surface, position, direction, endpoint, true, shadows, shadowRays);
                float3 connectionRadiance = emitted;
#if defined(ARVO_PARTICIPATING_MEDIA)
                connectionRadiance *= PhysicalConnectionBeam(surface, position, direction,
                    sqrt(distanceSquared), wavelength, interiorMeanFreePath);
#endif
                direct += scattering.Value * connectionRadiance *
                    (abs(dot(surface.Normal, direction)) * visible * mis / (AreaLightSamples * lightPdf));
            }
        }
        if (delta || !nee || combine)
        {
            const uint bsdfDimension = dimension ^ DirectBsdfScramble;
            const SurfaceSample event = SampleSurface(surface, Sample2D(samples, bsdfDimension),
                Sample2D(samples, bsdfDimension ^ LobeSelectionScramble).x);
            if (event.Event == 0u || event.Pdf <= 0 || all(event.Weight == 0)) continue;
            const float facing = -dot(event.Direction, light.Direction);
            if (facing <= 0) continue;
            const float distance = dot(position - light.Position, light.Direction) / facing;
            if (distance <= RayEpsilon) continue;
            const float3 endpoint = position + event.Direction * distance;
            const float3 offset = endpoint - light.Position;
            if (abs(dot(offset, tangent)) > light.Width * 0.5 || abs(dot(offset, bitangent)) > light.Height * 0.5) continue;
            const float mis = !delta && nee ? MisWeight(event.Pdf, AreaLightSamples * distance * distance / (area * facing)) : 1;
            const float visible = PhysicalVisibility(surface, position, event.Direction, endpoint, true, shadows, shadowRays);
            float3 connectionRadiance = emitted;
#if defined(ARVO_PARTICIPATING_MEDIA)
            connectionRadiance *= PhysicalConnectionBeam(surface, position, event.Direction,
                distance, wavelength, interiorMeanFreePath);
#endif
            direct += event.Weight * connectionRadiance * (mis * visible);
        }
    }
    return direct;
}
