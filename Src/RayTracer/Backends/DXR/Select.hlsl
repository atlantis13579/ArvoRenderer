// Selection only; hardware BVH and each software traversal have their own implementation.
#include "Storage.hlsl"
#include "Hardware.hlsl"
#include "KdTree.hlsl"
#include "Bih.hlsl"
#include "SpatialBvh.hlsl"

static const uint KdTreeAcceleration = 1u;
static const uint BihAcceleration = 2u;
static const uint SpatialSplitPolicy = 1u;

Payload CastSoftwareObject(float3 origin, float3 direction, float minimumDistance, float maximumDistance,
                           InstanceGeometry geometry)
{
    if (Acceleration == KdTreeAcceleration)
        return CastKdRay(origin, direction, minimumDistance, maximumDistance, geometry);
    if (Acceleration == BihAcceleration)
        return CastBihRay(origin, direction, minimumDistance, maximumDistance, geometry);
    if (BvhPolicy == SpatialSplitPolicy)
        return CastSpatialBvhRay(origin, direction, minimumDistance, maximumDistance, geometry);
    return MissPayload(maximumDistance);
}

// Callable shaders keep software traversal out of every light/bounce call site.
// TraceRay is not permitted in a callable shader: hardware tracing remains in
// ray generation; this stage only validates hardware rays or traces software BLASes.
// DXR Functional Spec, "Callable shader" and "Intrinsics".
// https://microsoft.github.io/DirectX-Specs/d3d/Raytracing.html#callable-shader
Payload DispatchSoftwareTraversal(EncodedRay ray, bool validateHardware, out bool valid)
{
    TraversalRequest request;
    request.Ray = ray;
    request.Hit = MissEncodedPayload(ray.Maximum);
    request.ValidateHardware = validateHardware ? 1u : 0u;
    request.Valid = 1u;
    static const uint TraversalCallableIndex = 0u;
    CallShader(TraversalCallableIndex, request);
    valid = request.Valid != 0u;
    return request.Hit;
}

[shader("callable")]
void SoftwareTraversal(inout TraversalRequest request)
{
    if (InstanceCount != 0)
    {
        bool valid;
        request.Hit = TraverseInstances(request.Ray, request.ValidateHardware != 0u, valid);
        request.Valid = valid ? 1u : 0u;
        return;
    }
    if (request.ValidateHardware != 0u) return;
    InstanceGeometry geometry = (InstanceGeometry)0;
    geometry.TriangleCount = TriangleCount;
    geometry.SphereCount = SphereCount;
    geometry.NodeCount = SoftwareNodeCount;
    geometry.Minimum = SoftwareBoundsMinimum;
    geometry.Maximum = SoftwareBoundsMaximum;
    request.Hit = CastSoftwareObject(asfloat(request.Ray.Origin), asfloat(request.Ray.Direction),
                                    asfloat(request.Ray.Minimum), asfloat(request.Ray.Maximum), geometry);
}

Payload CastEncodedRay(EncodedRay ray)
{
    // Both signed zero encodings describe the same nonnegative endpoint.
    ray.Minimum &= ~FloatSignBit;
    ray.Maximum &= ~FloatSignBit;
    if (Acceleration == 0u && BvhPolicy != SpatialSplitPolicy) return CastHardwareRay(ray);
    bool valid;
    return DispatchSoftwareTraversal(ray, false, valid);
}

Payload CastRay(float3 origin, float3 direction, float minimumDistance, float maximumDistance)
{
    EncodedRay ray;
    ray.Origin = asuint(origin);
    ray.Direction = asuint(direction);
    ray.Minimum = asuint(minimumDistance);
    ray.Maximum = asuint(maximumDistance);
    return CastEncodedRay(ray);
}
