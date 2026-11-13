// Native Metal finite and infinite light connections. Finite shadow segments
// are re-aimed after the geometric-side offset, keeping the emitter endpoint
// outside the occlusion interval. This leaf does not choose sampling strategies.
struct DirectLightContext
{
    SceneGeometryBuffers Geometry;
    device const PackedLight* Lights;
    uint LightCount, TransportFlags;
    float ShadowStrength;
#if defined(ARVO_PARTICIPATING_MEDIA)
    MediumParameters Medium;
#endif
#if defined(ARVO_SPECTRAL_TRANSPORT)
    SpectralPathContext Spectral;
#endif
};
float PhysicalVisibility(DirectLightContext context, SurfaceBsdf surface, float3 position,
    float3 direction, float3 endpoint, bool finiteLight, bool shadows,
    thread uint2& shadowRays, thread uint2& primitiveTests)
{
    if (!shadows || context.ShadowStrength <= 0) return 1;
    RayData ray;
    ray.Origin = SurfaceOffset(surface, position, direction);
    ray.Direction = direction;
    ray.MinimumDistance = RayEpsilon;
    ray.MaximumDistance = MaximumRayDistance;
    if (finiteLight)
    {
        const float3 segment = endpoint - ray.Origin;
        const float distance = length(segment);
        ray.MaximumDistance = distance - RayEpsilon;
        if (ray.MaximumDistance <= ray.MinimumDistance) return 1;
        ray.Direction = segment / distance;
    }
    HitData hit{};
    hit.Distance = ray.MaximumDistance;
    IncrementCounter(shadowRays);
    return IntersectScene(context.Geometry, ray, true, hit, primitiveTests)
        ? 1 - clamp(context.ShadowStrength, 0.0f, 1.0f) : 1;
}
#if defined(ARVO_PARTICIPATING_MEDIA)
float3 PhysicalConnectionBeam(DirectLightContext context, SurfaceBsdf surface, float3 point,
                               float3 direction, float distance, float currentInterior)
{
    float interior = currentInterior;
    if (surface.Material.Optical.w > 0)
    {
        // This BSDF normal is face-forward. A transmitted connection changes
        // the incident medium; a reflected one stays on the same side.
        const bool transmitted = dot(surface.GeometricNormal, direction) < 0;
        const bool inside = (currentInterior > 0) != transmitted;
        interior = inside ? surface.Material.Optical.w : 0;
    }
    if (interior > 0) return float3(exp(-distance / interior));
    return MediumTransmission(context.Medium, point, direction, 0, distance);
}
#endif
float3 PhysicalDeltaLight(DirectLightContext context, SurfaceBsdf surface, float3 position,
    float3 direction, float3 endpoint, float3 radiance, bool finiteLight, bool shadows,
    thread uint2& shadowRays, thread uint2& primitiveTests, float interiorMeanFreePath = 0)
{
    const SurfaceEvaluation evaluated = EvaluateSurface(surface, direction);
    if (all(evaluated.Value == float3(0))) return float3(0);
    const float visibility = PhysicalVisibility(context, surface, position, direction, endpoint, finiteLight,
                                                 shadows, shadowRays, primitiveTests);
#if defined(ARVO_PARTICIPATING_MEDIA)
    radiance *= PhysicalConnectionBeam(context, surface, position, direction,
        finiteLight ? length(endpoint - position) : MaximumRayDistance, interiorMeanFreePath);
#endif
    return evaluated.Value * radiance * (abs(dot(surface.Normal, direction)) * visibility);
}
