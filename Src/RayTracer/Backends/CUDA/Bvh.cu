    // Stackless depth-first BVH traversal. Interior nodes carry escape links;
    // leaf references preserve original scene indices after builder sorting.
    __device__ bool IntersectBounds(
        const PackedAccelerationNode& node,
        const Ray& ray,
        float closestDistance)
    {
        float nearDistance = ray.MinimumDistance;
        float farDistance = fminf(ray.MaximumDistance, closestDistance);
        const Vec3 minimum = {node.MinX, node.MinY, node.MinZ};
        const Vec3 maximum = {node.MaxX, node.MaxY, node.MaxZ};
        for (unsigned int axis = 0; axis < 3; ++axis)
        {
            const float origin = Component(ray.Origin, axis);
            const float direction = Component(ray.Direction, axis);
            const float minimumValue = Component(minimum, axis);
            const float maximumValue = Component(maximum, axis);
            if (direction == 0.0f)
            {
                if (origin < minimumValue || origin > maximumValue) return false;
                continue;
            }
            // Divide directly: 0 * an overflowing reciprocal would become NaN.
            float distance0 = (minimumValue - origin) / direction;
            float distance1 = (maximumValue - origin) / direction;
            if (distance0 > distance1)
            {
                const float temporary = distance0;
                distance0 = distance1;
                distance1 = temporary;
            }
            nearDistance = fmaxf(nearDistance, distance0);
            farDistance = fminf(farDistance, distance1);
            if (nearDistance > farDistance) return false;
        }
        return true;
    }

    __device__ bool IntersectBvh(
        const PackedAccelerationNode* nodes,
        const PackedPrimitiveRef* primitiveRefs,
        const PackedTriangle* triangles,
        const PackedSphere* spheres,
        const FrameConstants& constants,
        const Ray& ray,
        bool anyHit,
        Hit& hit,
        unsigned long long& primitiveTests)
    {
        hit.PrimitiveIndex = InvalidPrimitiveIndex;
        if (constants.NodeCount == 0) return false;
        bool found = false;
        unsigned int nodeIndex = 0;
        while (nodeIndex < constants.NodeCount)
        {
            const PackedAccelerationNode node = nodes[nodeIndex];
            if (!IntersectBounds(node, ray, hit.Distance))
            {
                nodeIndex = node.PrimitiveCount == 0 ? node.FirstOrEscape : nodeIndex + 1;
                continue;
            }
            if (node.PrimitiveCount == 0)
            {
                ++nodeIndex;
                continue;
            }
            for (unsigned int offset = 0; offset < node.PrimitiveCount; ++offset)
            {
                const unsigned int encoded = primitiveRefs[node.FirstOrEscape + offset].TypeAndIndex;
                ++primitiveTests;
                Hit candidate = hit;
                const unsigned int primitiveIndex = (encoded & SphereBit) != 0u
                    ? constants.TriangleCount + (encoded & ~SphereBit) : encoded;
                bool primitiveHit = false;
                if ((encoded & SphereBit) != 0)
                {
                    primitiveHit = IntersectSphere(spheres[encoded & ~SphereBit], ray, candidate);
                }
                else
                {
                    primitiveHit = IntersectTriangle(triangles[encoded], ray, candidate);
                }
                if (primitiveHit)
                {
                    if (candidate.Distance > hit.Distance
                        || (candidate.Distance == hit.Distance && primitiveIndex >= hit.PrimitiveIndex)) continue;
                    candidate.PrimitiveIndex = primitiveIndex;
                    hit = candidate;
                    found = true;
                    if (anyHit) return true;
                }
            }
            ++nodeIndex;
        }
        return found;
    }
