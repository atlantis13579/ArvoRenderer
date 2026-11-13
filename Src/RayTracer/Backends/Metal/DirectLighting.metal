// Native Metal direct lighting: stratified rectangle quadrature and an
// independent BSDF proposal. Veach (1997), chapter 9: compare n*p in solid angle.
// Discrete ideal events cannot compete with continuous area samples. Registered
// emitter meshes belong to this estimator after every physical continuation.
constant uint AreaLightStrataPerAxis = 2;
float3 PhysicalDirectLighting(DirectLightContext context, SurfaceBsdf surface, float3 position,
                              SampleContext samples, uint bounce, thread uint2& shadowRays,
                              thread uint2& primitiveTests, float interiorMeanFreePath = 0)
{
    float3 result = float3(0);
    const bool delta = SurfaceDelta(surface);
    const bool nee = (context.TransportFlags & NeeBit) != 0u;
    const bool mis = ((context.TransportFlags >> MisShift) & MisMask) != NoMis;
    for (uint index = 0; index < context.LightCount; ++index)
    {
        const PackedLight light = context.Lights[index];
        if (light.Intensity <= 0) continue;
        const bool shadows = light.CastShadows != 0u;
        const float3 lightPosition = float3(light.Position);
        const float3 lightDirection = float3(light.Direction);
        float3 color = float3(light.Color);
#if defined(ARVO_SPECTRAL_TRANSPORT)
        color = SpectralLight(context.Spectral, color, index);
#endif
        if (light.Type == LightTypeDirectional)
        {
            result += PhysicalDeltaLight(context, surface, position, lightDirection, float3(0),
                color * light.Intensity, false, shadows, shadowRays, primitiveTests, interiorMeanFreePath);
            continue;
        }
        if (light.Type == LightTypePoint)
        {
            const float3 segment = lightPosition - position;
            const float distanceSquared = dot(segment, segment);
            if (distanceSquared <= RayEpsilon * RayEpsilon) continue;
            result += PhysicalDeltaLight(context, surface, position, segment / sqrt(distanceSquared), lightPosition,
                color * (light.Intensity / distanceSquared), true, shadows, shadowRays, primitiveTests, interiorMeanFreePath);
            continue;
        }
        if (light.Type != LightTypeRectArea || light.Width <= 0 || light.Height <= 0) continue;
        const float area = light.Width * light.Height;
        const float3 emitted = color * (light.Intensity / area);
        const float3 helper = abs(lightDirection.y) < BasisParallelThreshold ? float3(0, 1, 0) : float3(1, 0, 0);
        const float3 tangent = normalize(cross(helper, lightDirection)), bitangent = cross(lightDirection, tangent);
        const uint domain = ((bounce + 1u) * BounceSeedMultiplier) ^ ((index + 1u) * LightSeedMultiplier);
        if (nee && !delta)
        {
            for (uint sample = 0; sample < AreaLightSamples; ++sample)
            {
                const float2 variates = Sample2D(samples, domain ^ ((sample + 1u) * LightSampleSeedMultiplier));
                const float2 cell = float2(float(sample % AreaLightStrataPerAxis), float(sample / AreaLightStrataPerAxis));
                const float2 uv = (cell + variates) / float(AreaLightStrataPerAxis) - 0.5f;
                const float3 endpoint = lightPosition + tangent * (uv.x * light.Width) + bitangent * (uv.y * light.Height);
                const float3 offset = endpoint - position;
                const float distanceSquared = dot(offset, offset);
                if (distanceSquared <= RayEpsilon * RayEpsilon) continue;
                const float3 incoming = offset / sqrt(distanceSquared);
                const float facing = -dot(lightDirection, incoming);
                if (facing <= 0) continue;
                const SurfaceEvaluation evaluated = EvaluateSurface(surface, incoming);
                if (all(evaluated.Value == float3(0))) continue;
                const float lightPdf = distanceSquared / (area * facing);
                const float weight = mis ? MisWeight(AreaLightSamples * lightPdf, evaluated.Pdf, context.TransportFlags) : 1;
                const float visible = PhysicalVisibility(context, surface, position, incoming, endpoint, true,
                                                         shadows, shadowRays, primitiveTests);
                float3 connectionRadiance = emitted;
#if defined(ARVO_PARTICIPATING_MEDIA)
                connectionRadiance *= PhysicalConnectionBeam(context, surface, position, incoming,
                    sqrt(distanceSquared), interiorMeanFreePath);
#endif
                result += evaluated.Value * connectionRadiance * (abs(dot(surface.Normal, incoming)) * visible * weight /
                                                       (AreaLightSamples * lightPdf));
            }
        }
        if (delta || !nee || mis)
        {
            const uint domainBsdf = domain ^ DirectBsdfScramble;
            const SurfaceSample event = SampleSurface(surface, Sample2D(samples, domainBsdf),
                Sample2D(samples, domainBsdf ^ LobeSelectionScramble).x);
            if (event.Event == 0u || event.Pdf <= 0 || all(event.Weight == float3(0))) continue;
            const float facing = -dot(event.Direction, lightDirection);
            if (facing <= 0) continue;
            const float distance = dot(position - lightPosition, lightDirection) / facing;
            if (distance <= RayEpsilon) continue;
            const float3 endpoint = position + event.Direction * distance, offset = endpoint - lightPosition;
            if (abs(dot(offset, tangent)) > light.Width * 0.5f || abs(dot(offset, bitangent)) > light.Height * 0.5f) continue;
            const float weight = delta || !nee ? 1 : MisWeight(event.Pdf,
                AreaLightSamples * distance * distance / (area * facing), context.TransportFlags);
            const float visible = PhysicalVisibility(context, surface, position, event.Direction, endpoint, true,
                                                     shadows, shadowRays, primitiveTests);
            float3 connectionRadiance = emitted;
#if defined(ARVO_PARTICIPATING_MEDIA)
            connectionRadiance *= PhysicalConnectionBeam(context, surface, position, event.Direction,
                distance, interiorMeanFreePath);
#endif
            result += event.Weight * connectionRadiance * (weight * visible);
        }
    }
    return result;
}
