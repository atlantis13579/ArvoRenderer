// The driver owns this acceleration structure. Software kd-tree/BIH/spatial
// traversal has a separate implementation. Ray intervals remain open.
// https://microsoft.github.io/DirectX-Specs/d3d/Raytracing.html#ray-extents
RaytracingAccelerationStructure SceneAcceleration : register(t0);

Payload CastHardwareRay(EncodedRay input)
{
    Payload payload = MissEncodedPayload(input.Maximum);
    if (PrimitiveCount == 0 || input.Minimum >= input.Maximum) return payload;
    RayDesc ray;
    ray.Origin = asfloat(input.Origin);
    ray.Direction = asfloat(input.Direction);
    ray.TMin = asfloat(input.Minimum);
    ray.TMax = asfloat(input.Maximum);
    double parameterFactor = 1.0L;
    if (InstanceCount != 0)
    {
        double3 wideDirection = WideRayVector(input.Direction);
        parameterFactor = DirectionFactor(wideDirection);
        ray.Direction = float3(wideDirection / parameterFactor);
        ray.TMin = asfloat(RayParameterBits(WideRayBits(input.Minimum) * parameterFactor));
        ray.TMax = asfloat(RayParameterBits(WideRayBits(input.Maximum) * parameterFactor));
        EncodedRay conditioned = input;
        conditioned.Direction = asuint(ray.Direction);
        conditioned.Minimum = asuint(ray.TMin);
        conditioned.Maximum = asuint(ray.TMax);
        bool valid;
        DispatchSoftwareTraversal(conditioned, true, valid);
        if (!valid) return payload;
        payload.DistanceBits = conditioned.Maximum;
    }
    // Each BLAS contains one geometry type. The TLAS contribution selects its
    // triangle/sphere hit group, while InstanceID addresses shared metadata.
    TraceRay(SceneAcceleration, RAY_FLAG_FORCE_OPAQUE, 0xff, 0, 1, 0, ray, payload);
    if (InstanceCount != 0 && (payload.Flags & SurfaceHitFlag) != 0u)
    {
        // Visit the closest representable layer again without accepting any hit.
        // This preserves deterministic lower primitive IDs for coincident hits.
        uint encoded = payload.DistanceBits;
        ray.TMin = max(ray.TMin, asfloat(encoded - 1u));
        ray.TMax = min(ray.TMax, asfloat(encoded + 1u));
        payload.Flags |= TieSearchFlag;
        TraceRay(SceneAcceleration, RAY_FLAG_FORCE_NON_OPAQUE | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, 0xff, 0, 1,
                 0, ray, payload);
        payload.Flags &= ~TieSearchFlag;
        payload.DistanceBits = RayParameterBits(WideRayBits(payload.DistanceBits) / parameterFactor);
        if (payload.DistanceBits <= input.Minimum || payload.DistanceBits >= input.Maximum)
            return MissEncodedPayload(input.Maximum);
    }
    if ((payload.Flags & SurfaceHitFlag) == 0u) return MissEncodedPayload(input.Maximum);
    return payload;
}

[shader("miss")]
void Miss(inout Payload payload)
{
    if ((payload.Flags & TieSearchFlag) == 0u) payload.Flags = 0;
}
