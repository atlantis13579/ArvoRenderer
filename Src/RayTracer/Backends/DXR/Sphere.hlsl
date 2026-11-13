// Analytic sphere intersection inside hardware-tested procedural AABBs.
// DXR Functional Spec, "Intersection shaders" and "ReportHit":
// https://microsoft.github.io/DirectX-Specs/d3d/Raytracing.html#intersection-shaders
// Condition D by its largest component before squaring, then restore the
// original ray parameter. The hardware ray itself is never normalized.
struct SphereAttributes
{
    float3 Normal;
};

bool IntersectSphere(uint sphereIndex, float3 origin, float3 originalDirection, float minimumDistance,
                     float maximumDistance, out float distance, out float3 normal)
{
    distance = maximumDistance;
    normal = 0.0;
    SphereData sphere = Spheres[sphereIndex];
    float directionScale =
        max(abs(originalDirection.x), max(abs(originalDirection.y), abs(originalDirection.z)));
    float3 direction = originalDirection / directionScale;
    float3 offset = origin - sphere.Center;
    float a = dot(direction, direction);
    float halfB = dot(offset, direction);
    float c = dot(offset, offset) - sphere.Radius * sphere.Radius;
    float discriminant = halfB * halfB - a * c;
    if (discriminant < 0.0) return false;
    float rootDiscriminant = sqrt(discriminant);
    // The q form avoids subtracting nearly equal values for the nearer root.
    float q = -halfB - (halfB >= 0.0 ? rootDiscriminant : -rootDiscriminant);
    float first = q == 0.0 ? -halfB / a : q / a;
    float second = q == 0.0 ? first : c / q;
    float nearDistance = min(first, second) / directionScale;
    float farDistance = max(first, second) / directionScale;
    // Procedural DXR permits closed endpoints; use the shared open interval
    // instead, matching fixed-function triangles and all renderer backends.
    distance = nearDistance > minimumDistance ? nearDistance : farDistance;
    if (!isfinite(distance) || distance <= minimumDistance || distance >= maximumDistance) return false;
    normal = (offset + distance * originalDirection) / sphere.Radius;
    return true;
}

[shader("intersection")]
void SphereIntersection()
{
    float distance;
    SphereAttributes attributes;
    uint sphere = PrimitiveIndex();
    double3 origin = double3(ObjectRayOrigin()), direction = double3(ObjectRayDirection());
    if (InstanceCount != 0)
    {
        const InstanceRecord instance = InstanceRecords[InstanceID()];
        sphere += instance.Geometry.SphereOffset;
        if (instance.Motion.Enabled != 0)
        {
            const MotionPose pose = InterpolateMotion(instance.Motion.Transform, InstanceSceneTime());
            InverseInstanceMotion(pose, origin, direction);
        }
    }
    if (IntersectSphere(sphere, float3(origin), float3(direction), RayTMin(), RayTCurrent(), distance,
                        attributes.Normal))
        ReportHit(distance, 0u, attributes);
}

Payload SphereSurface(uint sphereIndex, float distance, float3 outwardNormal, float3 direction)
{
    Payload payload;
    float3 normal = normalize(outwardNormal);
    if (dot(normal, direction) > 0.0) normal = -normal;
    payload.DistanceBits = asuint(distance);
    payload.Normal = normal;
    payload.GeometricNormal = normalize(outwardNormal);
    const float directionScale = max(abs(direction.x), max(abs(direction.y), abs(direction.z)));
    const bool frontFace = dot(payload.GeometricNormal, direction / directionScale) < 0.0;
    payload.Reserved = 0u;
#if defined(ARVO_SPECTRAL_TRANSPORT)
    payload.VertexSpectrumBase = InvalidPrimitiveIndex;
    payload.Barycentric = 0.0f.xx;
#endif
    payload.Color = 1.0;
    payload.PrimitiveIndex = TriangleCount + sphereIndex;
    payload.MaterialIndex = Spheres[sphereIndex].MaterialIndex;
    payload.TexCoord = 0.0;
    payload.Flags = SurfaceHitFlag
        | (frontFace ? SurfaceFrontFaceFlag : 0u);
    return payload;
}

[shader("closesthit")]
void SphereClosestHit(inout Payload payload, in SphereAttributes attributes)
{
    if (RayTCurrent() >= asfloat(payload.DistanceBits)) return;
    uint sphere = PrimitiveIndex();
    if (InstanceCount == 0)
        payload = SphereSurface(sphere, RayTCurrent(), attributes.Normal, WorldRayDirection());
    else
    {
        InstanceRecord instance = InstanceRecords[InstanceID()];
        payload = SphereSurface(instance.Geometry.SphereOffset + sphere, RayTCurrent(), attributes.Normal,
                                InstanceRestDirection(instance, ObjectRayDirection()));
        payload = InstanceSurface(payload, instance, instance.Geometry.TriangleCount + sphere,
                                  asuint(WorldRayDirection()));
    }
}

[shader("anyhit")]
void SphereTieHit(inout Payload payload, in SphereAttributes attributes)
{
    InstanceRecord instance = InstanceRecords[InstanceID()];
    uint local = instance.Geometry.TriangleCount + PrimitiveIndex();
    uint identifier = instance.Geometry.FirstPrimitive + local;
    if (RayTCurrent() < asfloat(payload.DistanceBits) ||
        (RayTCurrent() == asfloat(payload.DistanceBits) && identifier < payload.PrimitiveIndex))
    {
        payload = SphereSurface(instance.Geometry.SphereOffset + PrimitiveIndex(), RayTCurrent(),
                                attributes.Normal, InstanceRestDirection(instance, ObjectRayDirection()));
        payload = InstanceSurface(payload, instance, local, asuint(WorldRayDirection()));
        payload.Flags |= TieSearchFlag;
    }
    IgnoreHit();
}
