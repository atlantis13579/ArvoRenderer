#include "MotionInterpolation.hlsl"
#include "MotionInstancing.hlsl"
#include "Bvh.hlsl"
#include "KdTree.hlsl"
#include "Bih.hlsl"

// The host validates the selection and uploads the corresponding native tree.
// Production integrators and direct ray diagnostics share this dispatch only.
bool IntersectGeometry(RayData ray, GeometryRange geometry, bool anyHit, inout HitData hit, inout uint2 primitiveTests, float sceneTime = 0)
{
    static const uint BvhAcceleration = 0u, KdTreeAcceleration = 1u, BihAcceleration = 2u;
    if (Acceleration == BihAcceleration)
        return IntersectBih(ray, geometry, anyHit, hit, primitiveTests);
    if (Acceleration == KdTreeAcceleration)
        return IntersectKdTree(ray, geometry, anyHit, hit, primitiveTests);
    if (Acceleration == BvhAcceleration)
        return IntersectBvh(ray, geometry, anyHit, hit, primitiveTests);
    return false;
}

#include "Instancing.hlsl"
bool IntersectScene(RayData ray, bool anyHit, inout HitData hit, inout uint2 primitiveTests, float sceneTime = 0)
{
    if (InstanceNodeCount != 0u) return IntersectInstances(ray, anyHit, hit, primitiveTests, sceneTime);
    GeometryRange geometry = (GeometryRange)0;
    geometry.NodeCount = NodeCount;
    geometry.TriangleCount = TriangleCount;
    return IntersectGeometry(ray, geometry, anyHit, hit, primitiveTests, sceneTime);
}
