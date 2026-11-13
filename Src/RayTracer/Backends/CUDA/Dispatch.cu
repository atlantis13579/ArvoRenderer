#include "MotionInterpolation.cu"
#include "MotionInstancing.cu"
    // Mechanical selection only; each accelerator owns its traversal algorithm.
    constexpr unsigned int BvhAcceleration = 0u;
    constexpr unsigned int KdTreeAcceleration = 1u;
    constexpr unsigned int BihAcceleration = 2u;

    __device__ bool IntersectGeometry(
        const PackedAccelerationNode* nodes,
        const PackedPrimitiveRef* primitiveRefs,
        const PackedTriangle* triangles,
        const PackedSphere* spheres,
        const FrameConstants& constants,
        const Ray& ray,
        bool anyHit,
        Hit& hit,
        unsigned long long& primitiveTests, float sceneTime = 0)
    {
#if defined(ARVO_OBSERVE_PATH_TIME)
        ARVO_OBSERVE_PATH_TIME(sceneTime, anyHit, false);
#endif
        if (constants.Acceleration == BvhAcceleration)
            return IntersectBvh(nodes, primitiveRefs, triangles, spheres, constants,
                ray, anyHit, hit, primitiveTests);
        if (constants.Acceleration == KdTreeAcceleration)
            return IntersectKdTree(nodes, primitiveRefs, triangles, spheres, constants,
                ray, anyHit, hit, primitiveTests);
        if (constants.Acceleration == BihAcceleration)
            return IntersectBih(nodes, primitiveRefs, triangles, spheres, constants,
                ray, anyHit, hit, primitiveTests);
        hit.PrimitiveIndex = InvalidPrimitiveIndex;
        return false;
    }

#include "Instancing.cu"

    __device__ bool IntersectScene(
        const PackedAccelerationNode* nodes, const PackedPrimitiveRef* references,
        const PackedTriangle* triangles, const PackedSphere* spheres, const FrameConstants& constants,
        const Ray& ray, bool anyHit, Hit& hit, unsigned long long& tests, float sceneTime = 0)
    {
#if defined(ARVO_OBSERVE_PATH_TIME)
        ARVO_OBSERVE_PATH_TIME(sceneTime, anyHit, true);
#endif
        if (constants.InstanceNodeCount != 0)
            return IntersectInstances(nodes, references, triangles, spheres, constants, ray, anyHit, hit, tests, sceneTime);
        return IntersectGeometry(nodes, references, triangles, spheres, constants, ray, anyHit, hit, tests, sceneTime);
    }
