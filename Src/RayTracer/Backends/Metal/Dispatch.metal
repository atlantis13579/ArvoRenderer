#include "MotionInterpolation.metal"
#include "MotionInstancing.metal"
// This leaf selects Metal's independent object accelerators and instance TLAS.
constant uint BvhAcceleration = 0u;
constant uint KdTreeAcceleration = 1u;
constant uint BihAcceleration = 2u;

bool IntersectObject(ObjectGeometry object, RayData ray, bool anyHit, thread HitData &hit,
                     thread uint2 &primitiveTests)
{
#if defined(ARVO_OBSERVE_PATH_TIME)
    ARVO_OBSERVE_PATH_TIME(object.SceneTime, anyHit, false);
#endif
    if (object.Acceleration == BvhAcceleration)
        return IntersectBvh(object, ray, anyHit, hit, primitiveTests);
    if (object.Acceleration == KdTreeAcceleration)
        return IntersectKdTree(object, ray, anyHit, hit, primitiveTests);
    if (object.Acceleration == BihAcceleration)
        return IntersectBih(object, ray, anyHit, hit, primitiveTests);
    hit.PrimitiveIndex = InvalidPrimitiveIndex;
    return false;
}

#include "Instancing.metal"

bool IntersectEncodedScene(SceneGeometryBuffers scene, EncodedRay ray, bool anyHit, thread HitData &hit,
                           thread uint &distanceBits, thread uint2 &primitiveTests)
{
#if defined(ARVO_OBSERVE_PATH_TIME)
    ARVO_OBSERVE_PATH_TIME(scene.Object.SceneTime, anyHit, true);
#endif
    hit.PrimitiveIndex = InvalidPrimitiveIndex;
    distanceBits = ray.DirectionMaximum.w & ~WideMath::SignMask;
    if ((ray.DirectionMaximum.w & WideMath::SignMask) != 0u)
        return false;
    if (scene.NodeCount)
        return IntersectInstances(scene, ray, anyHit, hit, distanceBits, primitiveTests);
    bool valid = true;
    const bool found = IntersectConditionedObject(scene.Object, DecodeRay(ray), ray, anyHit, hit,
                                                  distanceBits, primitiveTests, valid);
    if (!valid)
        atomic_store_explicit(scene.Error, InstanceRangeError, memory_order_relaxed);
    return found;
}

bool IntersectScene(SceneGeometryBuffers scene, RayData ray, bool anyHit, thread HitData &hit,
                    thread uint2 &primitiveTests)
{
    const EncodedRay encoded = {as_type<uint4>(float4(ray.Origin, ray.MinimumDistance)),
                                as_type<uint4>(float4(ray.Direction, ray.MaximumDistance))};
    uint distanceBits = as_type<uint>(hit.Distance);
    return IntersectEncodedScene(scene, encoded, anyHit, hit, distanceBits, primitiveTests);
}
