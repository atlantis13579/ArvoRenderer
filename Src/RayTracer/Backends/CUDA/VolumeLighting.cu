// Native CUDA volume direct lighting: analytic deltas, stratified rectangle
// NEE and an independent HG/uniform direction proposal with solid-angle MIS.
__device__ Vec3 VolumeBeam(MediumParameters medium, Vec3 point, Vec3 direction, float distance)
{
    const float3 beam = MediumTransmission(medium, make_float3(point.X, point.Y, point.Z),
                                           make_float3(direction.X, direction.Y, direction.Z), 0, distance);
    return {beam.x, beam.y, beam.z};
}
__device__ Vec3 VolumeIncident(const PhysicalScene& scene, Vec3 position, Vec3 direction, float distance,
                               Vec3 emitted, bool shadows, unsigned long long& shadowRays,
                               unsigned long long& primitiveTests)
{
    float visibility = 1;
    if (shadows && scene.Frame->ShadowStrength > 0)
    {
        Ray connection{position, direction, RayEpsilon, distance - RayEpsilon};
        Hit hit{};
        hit.Distance = connection.MaximumDistance;
        ++shadowRays;
        if (IntersectScene(scene.Nodes, scene.References, scene.Triangles, scene.Spheres, *scene.Frame,
                           connection, true, hit, primitiveTests, scene.SceneTime))
            visibility -= fminf(1, fmaxf(0, scene.Frame->ShadowStrength));
    }
    return Multiply(Multiply(emitted, VolumeBeam(scene.Medium, position, direction, distance)), visibility);
}
__device__ Vec3 VolumeDirectLighting(const PhysicalScene& scene, const PackedLight* lights, Vec3 position,
                                     Vec3 incoming, const SampleContext& samples, unsigned bounce,
                                     unsigned long long& shadowRays, unsigned long long& primitiveTests)
{
    Vec3 result{};
    const unsigned flags = scene.Frame->TransportFlags;
    const bool nee = (flags & NeeBit) != 0, importance = (flags & ImportanceBit) != 0;
    constexpr unsigned PhaseDirectScramble = 0x6c8e9cf5u;
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
            float distance = MaximumRayDistance, intensity = light.Intensity;
            if (light.Type == PointLightType)
            {
                const Vec3 offset = Subtract(light.Position, position);
                const float squared = Dot(offset, offset);
                if (squared <= RayEpsilon * RayEpsilon)
                    continue;
                distance = sqrtf(squared);
                direction = Multiply(offset, 1 / distance);
                intensity /= squared;
            }
            const Vec3 energy =
                VolumeIncident(scene, position, direction, distance, Multiply(light.Color, intensity),
                               light.CastShadows != 0, shadowRays, primitiveTests);
            result =
                Add(result, Multiply(energy, MediumPhase(Dot(incoming, direction), scene.Medium.Anisotropy)));
            continue;
        }
        const float area = light.Width * light.Height;
        if (light.Type != RectAreaLightType || area <= 0)
            continue;
        const Vec3 helper = fabsf(light.Direction.Y) < BasisParallelThreshold ? Vec3{0, 1, 0} : Vec3{1, 0, 0};
        const Vec3 tangent = Normalize(Cross(helper, light.Direction)),
                   bitangent = Cross(light.Direction, tangent);
        const Vec3 emitted = Multiply(light.Color, light.Intensity / area);
        const unsigned domain = (bounce + 1) * BounceSeedMultiplier ^ (index + 1) * LightSeedMultiplier;
        for (unsigned i = 0; nee && i < AreaLightSamples; ++i)
        {
            const float2 u = Sample2D(samples, domain ^ (i + 1) * LightSampleSeedMultiplier);
            const float x = (float(i % AreaLightStrataPerAxis) + u.x) / AreaLightStrataPerAxis - 0.5f;
            const float y = (float(i / AreaLightStrataPerAxis) + u.y) / AreaLightStrataPerAxis - 0.5f;
            const Vec3 endpoint = Add(light.Position, Add(Multiply(tangent, x * light.Width),
                                                          Multiply(bitangent, y * light.Height)));
            const Vec3 offset = Subtract(endpoint, position);
            const float squared = Dot(offset, offset);
            if (squared <= RayEpsilon * RayEpsilon)
                continue;
            const float distance = sqrtf(squared);
            const Vec3 direction = Multiply(offset, 1 / distance);
            const float facing = -Dot(light.Direction, direction);
            if (facing <= 0)
                continue;
            const float density = squared / (area * facing);
            const float phase = MediumPhase(Dot(incoming, direction), scene.Medium.Anisotropy);
            const float competitor = importance ? phase : 1 / (4 * Pi);
            const float weight = MisWeight(AreaLightSamples * density, competitor, flags) * phase /
                                 (AreaLightSamples * density);
            result = Add(result, Multiply(VolumeIncident(scene, position, direction, distance, emitted,
                                                         light.CastShadows != 0, shadowRays, primitiveTests),
                                          weight));
        }
        if (!nee || ((flags >> MisShift) & MisMask) != NoMis)
        {
            float density;
            const float3 sample = MediumSampleDirection(
                make_float3(incoming.X, incoming.Y, incoming.Z), scene.Medium.Anisotropy,
                Sample2D(samples, domain ^ PhaseDirectScramble), importance, density);
            const Vec3 direction{sample.x, sample.y, sample.z};
            const float facing = -Dot(light.Direction, direction);
            if (facing <= 0 || density <= 0)
                continue;
            const float distance = Dot(Subtract(position, light.Position), light.Direction) / facing;
            if (distance <= RayEpsilon)
                continue;
            const Vec3 offset = Subtract(Add(position, Multiply(direction, distance)), light.Position);
            if (fabsf(Dot(offset, tangent)) > light.Width / 2 ||
                fabsf(Dot(offset, bitangent)) > light.Height / 2)
                continue;
            const float competitor = distance * distance / (area * facing);
            const float mis = nee ? MisWeight(density, AreaLightSamples * competitor, flags) : 1;
            const float weight =
                mis * MediumPhase(Dot(incoming, direction), scene.Medium.Anisotropy) / density;
            result = Add(result, Multiply(VolumeIncident(scene, position, direction, distance, emitted,
                                                         light.CastShadows != 0, shadowRays, primitiveTests),
                                          weight));
        }
    }
    return result;
}
