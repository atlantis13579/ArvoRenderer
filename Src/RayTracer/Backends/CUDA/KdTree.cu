    // Spatial kd-tree traversal carries the ray interval for each cell. Only
    // split planes choose children; duplicated references handle straddlers.
    // See PBRT 3rd ed., "Kd-Tree Accelerator", section 4.4.3.
    constexpr unsigned int KdLeafAxis = 3u;
    constexpr unsigned int KdStackCapacity = 33u; // Host builder depth limit + root.

    struct KdPendingCell
    {
        unsigned int NodeIndex;
        float NearDistance;
        float FarDistance;
    };

    __device__ bool IntersectKdTree(
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
        if (constants.NodeCount == 0u) return false;
        float nearDistance = ray.MinimumDistance;
        float farDistance = fminf(ray.MaximumDistance, hit.Distance);
        const PackedAccelerationNode root = nodes[0];
        const Vec3 lower = {root.MinX, root.MinY, root.MinZ};
        const Vec3 upper = {root.MaxX, root.MaxY, root.MaxZ};
        for (unsigned int axis = 0u; axis < KdLeafAxis; ++axis)
        {
            const float origin = Component(ray.Origin, axis);
            const float direction = Component(ray.Direction, axis);
            if (direction == 0.0f)
            {
                if (origin < Component(lower, axis) || origin > Component(upper, axis)) return false;
                continue;
            }
            const float first = (Component(lower, axis) - origin) / direction;
            const float second = (Component(upper, axis) - origin) / direction;
            nearDistance = fmaxf(nearDistance, fminf(first, second));
            farDistance = fminf(farDistance, fmaxf(first, second));
            if (nearDistance > farDistance) return false;
        }
        KdPendingCell pending[KdStackCapacity];
        unsigned int pendingCount = 0u;
        unsigned int nodeIndex = 0u;
        bool found = false;
        for (;;)
        {
            if (nearDistance <= hit.Distance)
            {
                const PackedAccelerationNode node = nodes[nodeIndex];
                if (node.SplitAxis != KdLeafAxis)
                {
                    const unsigned int below = nodeIndex + 1u;
                    const unsigned int above = node.AboveChild;
                    const float origin = Component(ray.Origin, node.SplitAxis);
                    const float direction = Component(ray.Direction, node.SplitAxis);
                    if (direction == 0.0f)
                    {
                        nodeIndex = origin <= node.SplitPosition ? below : above;
                        if (origin == node.SplitPosition)
                            pending[pendingCount++] = {above, nearDistance, farDistance};
                        continue;
                    }
                    const unsigned int firstChild = direction > 0.0f ? below : above;
                    const unsigned int secondChild = direction > 0.0f ? above : below;
                    const float planeDistance = (node.SplitPosition - origin) / direction;
                    if (planeDistance <= nearDistance)
                        nodeIndex = secondChild;
                    else if (planeDistance >= farDistance)
                        nodeIndex = firstChild;
                    else
                    {
                        pending[pendingCount++] = {secondChild, planeDistance, farDistance};
                        nodeIndex = firstChild;
                        farDistance = planeDistance;
                    }
                    continue;
                }
                for (unsigned int offset = 0u; offset < node.PrimitiveCount; ++offset)
                {
                    const unsigned int encoded = primitiveRefs[node.FirstOrEscape + offset].TypeAndIndex;
                    ++primitiveTests;
                    Hit candidate = hit;
                    const bool sphere = (encoded & SphereBit) != 0u;
                    const unsigned int originalIndex = sphere
                        ? constants.TriangleCount + (encoded & ~SphereBit) : encoded;
                    const bool accepted = sphere
                        ? IntersectSphere(spheres[encoded & ~SphereBit], ray, candidate)
                        : IntersectTriangle(triangles[encoded], ray, candidate);
                    if (accepted && (candidate.Distance < hit.Distance
                        || (candidate.Distance == hit.Distance && originalIndex < hit.PrimitiveIndex)))
                    {
                        candidate.PrimitiveIndex = originalIndex;
                        hit = candidate;
                        found = true;
                        if (anyHit) return true;
                    }
                }
            }
            if (pendingCount == 0u) break;
            const KdPendingCell next = pending[--pendingCount];
            nodeIndex = next.NodeIndex;
            nearDistance = next.NearDistance;
            farDistance = fminf(next.FarDistance, hit.Distance);
        }
        return found;
    }
