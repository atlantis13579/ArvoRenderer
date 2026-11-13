// Veach (1997), "Robust Monte Carlo Methods for Light Transport Simulation", ch. 9.
// Each rectangle gets four light samples and one independent BSDF sample.
// MIS compares sample-count times solid-angle density. Discrete specular events
// have no continuous competitor; their connections also support mesh-free lights.
// Registered emitter hits after physical continuation belong to this estimator,
// including delta paths, and must not be added again by PathTracing.
struct PhysicalScene
{
    const PackedAccelerationNode* Nodes;
    const PackedPrimitiveRef* References;
    const PackedTriangle* Triangles;
    const PackedSphere* Spheres;
    const FrameConstants* Frame;
#if defined(ARVO_PARTICIPATING_MEDIA)
    MediumParameters Medium;
#endif
#if defined(ARVO_SPECTRAL_TRANSPORT)
    PathSpectra Spectral;
#endif
    float SceneTime = 0;
};
__device__ Vec3 PhysicalMediumBeam(const PhysicalScene& scene, const PhysicalSurface& surface,
                                   Vec3 position, Vec3 direction, float distance, float currentInterior)
{
#if defined(ARVO_PARTICIPATING_MEDIA)
    const float boundaryMeanFreePath = surface.Material.Optical.W;
    const float meanFreePath = boundaryMeanFreePath > 0
        ? (Dot(surface.Geometry, direction) < 0 ? boundaryMeanFreePath : 0) : currentInterior;
    if (meanFreePath > 0)
    {
        const float attenuation = expf(-distance / meanFreePath);
        return {attenuation, attenuation, attenuation};
    }
    const float3 value = MediumTransmission(scene.Medium, make_float3(position.X, position.Y, position.Z),
        make_float3(direction.X, direction.Y, direction.Z), 0, distance);
    return {value.x, value.y, value.z};
#else
    return {1, 1, 1};
#endif
}
constexpr unsigned AreaLightStrataPerAxis = 2;
static_assert(AreaLightStrataPerAxis * AreaLightStrataPerAxis == AreaLightSamples,
              "Area-light strata must cover the sample budget.");

__device__ float PhysicalVisibility(const PhysicalScene& scene, const PhysicalSurface& surface, Vec3 position,
                                    Vec3 direction, const Vec3* endpoint, bool castShadows,
                                    unsigned long long& shadowRays, unsigned long long& primitiveTests)
{
    if (!castShadows || scene.Frame->ShadowStrength <= 0)
        return 1;
    Ray shadow{PhysicalRayOrigin(surface, position, direction), direction, RayEpsilon, MaximumRayDistance};
    if (endpoint)
    {
        const Vec3 segment = Subtract(*endpoint, shadow.Origin);
        const float length = sqrtf(Dot(segment, segment));
        shadow.MaximumDistance = length - RayEpsilon;
        if (shadow.MaximumDistance <= shadow.MinimumDistance)
            return 1;
        shadow.Direction = Multiply(segment, 1 / length);
    }
    Hit obstruction{};
    obstruction.Distance = shadow.MaximumDistance;
    ++shadowRays;
    const bool blocked = IntersectScene(scene.Nodes, scene.References, scene.Triangles, scene.Spheres,
                                        *scene.Frame, shadow, true, obstruction, primitiveTests, scene.SceneTime);
    return blocked ? 1 - fminf(1, fmaxf(0, scene.Frame->ShadowStrength)) : 1;
}
__device__ Vec3 PhysicalDirectLighting(const PhysicalScene& scene, const PackedLight* lights,
                                       const PhysicalSurface& surface, Vec3 position,
                                       const SampleContext& samples, unsigned bounce,
                                       unsigned long long& shadowRays, unsigned long long& primitiveTests,
                                       float interiorMeanFreePath = 0)
{
    const unsigned flags = scene.Frame->TransportFlags;
    const bool importance = (flags & ImportanceBit) != 0;
    const bool nee = (flags & NeeBit) != 0;
    const bool delta = PhysicalDelta(surface);
    Vec3 result{};
    for (unsigned index = 0; index < scene.Frame->LightCount; ++index)
    {
        PackedLight light = lights[index];
#if defined(ARVO_SPECTRAL_TRANSPORT)
        light.Color = PathLight(light.Color, index, scene.Spectral);
#endif
        if (light.Intensity <= 0)
            continue;
        if (light.Type == DirectionalLightType || light.Type == PointLightType)
        {
            Vec3 direction = light.Direction;
            float intensity = light.Intensity;
            const Vec3* endpoint = nullptr;
            if (light.Type == PointLightType)
            {
                const Vec3 segment = Subtract(light.Position, position);
                const float distanceSquared = Dot(segment, segment);
                if (distanceSquared <= RayEpsilon * RayEpsilon)
                    continue;
                direction = Multiply(segment, rsqrtf(distanceSquared));
                intensity /= distanceSquared;
                endpoint = &light.Position;
            }
            const PhysicalEvaluation evaluation = EvaluatePhysical(surface, direction, importance);
            if (evaluation.Value.X == 0 && evaluation.Value.Y == 0 && evaluation.Value.Z == 0)
                continue;
            const float visibility = PhysicalVisibility(scene, surface, position, direction, endpoint,
                                                        light.CastShadows != 0, shadowRays, primitiveTests);
            const float weight = intensity * fabsf(Dot(surface.Normal, direction)) * visibility;
            const float distance = endpoint ? sqrtf(Dot(Subtract(*endpoint, position), Subtract(*endpoint, position))) : MaximumRayDistance;
            result = Add(result, Multiply(Multiply(Multiply(evaluation.Value, light.Color),
                PhysicalMediumBeam(scene, surface, position, direction, distance, interiorMeanFreePath)), weight));
            continue;
        }
        const float area = light.Width * light.Height;
        if (light.Type != RectAreaLightType || area <= 0)
            continue;
        const Vec3 normal = light.Direction;
        const Vec3 helper = fabsf(normal.Y) < BasisParallelThreshold ? Vec3{0, 1, 0} : Vec3{1, 0, 0};
        const Vec3 tangent = Normalize(Cross(helper, normal));
        const Vec3 bitangent = Cross(normal, tangent);
        const Vec3 emitted = Multiply(light.Color, light.Intensity / area);
        const unsigned dimension = (bounce + 1) * BounceSeedMultiplier ^ (index + 1) * LightSeedMultiplier;
        for (unsigned sampleIndex = 0; nee && !delta && sampleIndex < AreaLightSamples; ++sampleIndex)
        {
            const float2 random =
                Sample2D(samples, dimension ^ (sampleIndex + 1) * LightSampleSeedMultiplier);
            const float u = (sampleIndex % AreaLightStrataPerAxis + random.x) / AreaLightStrataPerAxis - 0.5f;
            const float v = (sampleIndex / AreaLightStrataPerAxis + random.y) / AreaLightStrataPerAxis - 0.5f;
            const Vec3 endpoint = Add(light.Position, Add(Multiply(tangent, light.Width * u),
                                                          Multiply(bitangent, light.Height * v)));
            const Vec3 segment = Subtract(endpoint, position);
            const float distanceSquared = Dot(segment, segment);
            if (distanceSquared <= RayEpsilon * RayEpsilon)
                continue;
            const Vec3 direction = Multiply(segment, rsqrtf(distanceSquared));
            const float emitterCosine = -Dot(normal, direction);
            if (emitterCosine <= 0)
                continue;
            const PhysicalEvaluation evaluation = EvaluatePhysical(surface, direction, importance);
            if (evaluation.Value.X == 0 && evaluation.Value.Y == 0 && evaluation.Value.Z == 0)
                continue;
            const float lightPdf = distanceSquared / (area * emitterCosine);
            const float countedPdf = AreaLightSamples * lightPdf;
            const float mis = MisWeight(countedPdf, evaluation.Pdf, flags);
            const float visibility = PhysicalVisibility(scene, surface, position, direction, &endpoint,
                                                        light.CastShadows != 0, shadowRays, primitiveTests);
            const float weight = fabsf(Dot(surface.Normal, direction)) * mis * visibility / countedPdf;
            result = Add(result, Multiply(Multiply(Multiply(evaluation.Value, emitted),
                PhysicalMediumBeam(scene, surface, position, direction, sqrtf(distanceSquared), interiorMeanFreePath)), weight));
        }
        if (delta || !nee || ((flags >> MisShift) & MisMask) != NoMis)
        {
            const unsigned bsdfDimension = dimension ^ DirectBsdfScramble;
            const PhysicalSample sample =
                SamplePhysical(surface, Sample2D(samples, bsdfDimension),
                               Sample2D(samples, bsdfDimension ^ LobeSelectionScramble).x, importance);
            if (sample.Event == 0 || sample.Pdf <= 0)
                continue;
            const float emitterCosine = -Dot(normal, sample.Direction);
            if (emitterCosine <= 0)
                continue;
            const float distance = Dot(Subtract(position, light.Position), normal) / emitterCosine;
            if (distance <= RayEpsilon)
                continue;
            const Vec3 endpoint = Add(position, Multiply(sample.Direction, distance));
            const Vec3 offset = Subtract(endpoint, light.Position);
            if (fabsf(Dot(offset, tangent)) > light.Width * 0.5f ||
                fabsf(Dot(offset, bitangent)) > light.Height * 0.5f)
                continue;
            float mis = 1;
            if ((sample.Event & ArvoCuda::ScatteringDelta) == 0 && nee)
            {
                const float lightPdf = distance * distance / (area * emitterCosine);
                mis = MisWeight(sample.Pdf, AreaLightSamples * lightPdf, flags);
            }
            const float visibility = PhysicalVisibility(scene, surface, position, sample.Direction, &endpoint,
                                                        light.CastShadows != 0, shadowRays, primitiveTests);
            result = Add(result, Multiply(Multiply(Multiply(sample.Weight, emitted),
                PhysicalMediumBeam(scene, surface, position, sample.Direction, distance, interiorMeanFreePath)), mis * visibility));
        }
    }
    return result;
}
