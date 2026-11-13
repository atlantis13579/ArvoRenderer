// Attribute reconstruction is native to this DXR backend and shared by its
// fixed-function hits and software kd-tree triangle tests.
Payload TriangleSurface(uint primitive, float2 barycentrics, float distance, float3 direction)
{
    Payload payload;
    TriangleData triangleData = Triangles[primitive];
    float barycentric0 = 1.0 - barycentrics.x - barycentrics.y;
    uint firstVertex = primitive * 3;
    float3 edge1 = Vertices[firstVertex + 1].xyz - Vertices[firstVertex].xyz;
    float3 edge2 = Vertices[firstVertex + 2].xyz - Vertices[firstVertex].xyz;
    float3 geometricNormal = cross(edge1, edge2);
    float3 normal = geometricNormal;
    if (triangleData.SmoothNormals != 0)
    {
        normal = triangleData.Normal0.xyz * barycentric0 + triangleData.Normal1.xyz * barycentrics.x +
                 triangleData.Normal2.xyz * barycentrics.y;
    }
    if (dot(normal, normal) < MinimumNormalLengthSquared) normal = geometricNormal;
    normal = normalize(normal);
    if (dot(normal, direction) > 0.0) normal = -normal;
    payload.DistanceBits = asuint(distance);
    payload.Normal = normal;
    payload.GeometricNormal = normalize(geometricNormal);
    // Sidedness must not depend on triangle area or the ray's parameter scale.
    const float directionScale = max(abs(direction.x), max(abs(direction.y), abs(direction.z)));
    const bool frontFace = dot(payload.GeometricNormal, direction / directionScale) < 0.0;
    payload.Reserved = 0u;
#if defined(ARVO_SPECTRAL_TRANSPORT)
    payload.VertexSpectrumBase = triangleData.Padding;
    payload.Barycentric = barycentrics;
#endif
    payload.Color = triangleData.Color0.xyz * barycentric0 + triangleData.Color1.xyz * barycentrics.x +
                    triangleData.Color2.xyz * barycentrics.y;
    payload.PrimitiveIndex = primitive;
    payload.MaterialIndex = triangleData.MaterialIndex;
    payload.TexCoord = triangleData.TexCoord01.xy * barycentric0 +
                       triangleData.TexCoord01.zw * barycentrics.x +
                       triangleData.TexCoord2.xy * barycentrics.y;
    payload.Flags = SurfaceHitFlag | (triangleData.HasTexCoord != 0 ? SurfaceTexCoordFlag : 0u)
        | (frontFace ? SurfaceFrontFaceFlag : 0u);
    return payload;
}

[shader("closesthit")]
void ClosestHit(inout Payload payload, in BuiltInTriangleIntersectionAttributes attributes)
{
    if (RayTCurrent() >= asfloat(payload.DistanceBits)) return;
    uint primitive = PrimitiveIndex();
    if (InstanceCount == 0)
        payload = TriangleSurface(primitive, attributes.barycentrics, RayTCurrent(), WorldRayDirection());
    else
    {
        InstanceRecord instance = InstanceRecords[InstanceID()];
        payload = TriangleSurface(instance.Geometry.TriangleOffset + primitive, attributes.barycentrics,
                                  RayTCurrent(), InstanceRestDirection(instance, ObjectRayDirection()));
        payload = InstanceSurface(payload, instance, primitive, asuint(WorldRayDirection()));
    }
}

// A narrow second hardware query resolves exact-distance ties. IgnoreHit keeps
// the narrow interval open for every candidate; the miss shader preserves the
// payload. This gives stable IDs without replacing hardware triangle tests.
[shader("anyhit")]
void TriangleTieHit(inout Payload payload, in BuiltInTriangleIntersectionAttributes attributes)
{
    InstanceRecord instance = InstanceRecords[InstanceID()];
    uint local = PrimitiveIndex();
    uint identifier = instance.Geometry.FirstPrimitive + local;
    if (RayTCurrent() < asfloat(payload.DistanceBits) ||
        (RayTCurrent() == asfloat(payload.DistanceBits) && identifier < payload.PrimitiveIndex))
    {
        payload = TriangleSurface(instance.Geometry.TriangleOffset + local, attributes.barycentrics,
                                  RayTCurrent(), InstanceRestDirection(instance, ObjectRayDirection()));
        payload = InstanceSurface(payload, instance, local, asuint(WorldRayDirection()));
        payload.Flags |= TieSearchFlag;
    }
    IgnoreHit();
}
