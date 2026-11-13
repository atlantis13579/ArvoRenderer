    // Wachter/Keller (2006), "Instant Ray Tracing: The Bounding Interval Hierarchy".
    // The inherited cell is closed. Primitive tests still use the original open
    // ray interval; clipping planes only decide which object groups to visit.
    constexpr unsigned int BihLeafAxis = 3u;
    constexpr unsigned int BihStackCapacity = 33u; // Host depth limit plus root.

    struct BihCell
    {
        unsigned int NodeIndex;
        float NearDistance;
        float FarDistance;
    };

    __device__ bool ClipBihHalfSpace(
        const Ray& ray, unsigned int axis, float plane, bool below, BihCell& cell)
    {
        const float origin = Component(ray.Origin, axis);
        const float direction = Component(ray.Direction, axis);
        if (direction == 0.0f) return below ? origin <= plane : origin >= plane;
        const float planeDistance = (plane - origin) / direction;
        if (below == (direction > 0.0f))
            cell.FarDistance = fminf(cell.FarDistance, planeDistance);
        else
            cell.NearDistance = fmaxf(cell.NearDistance, planeDistance);
        return cell.NearDistance <= cell.FarDistance;
    }

    __device__ bool IntersectBih(
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
        BihCell root = {0u, ray.MinimumDistance, fminf(ray.MaximumDistance, hit.Distance)};
        const PackedAccelerationNode rootNode = nodes[0];
        const Vec3 lower = {rootNode.MinX, rootNode.MinY, rootNode.MinZ};
        const Vec3 upper = {rootNode.MaxX, rootNode.MaxY, rootNode.MaxZ};
        for (unsigned int axis = 0u; axis < BihLeafAxis; ++axis)
        {
            if (!ClipBihHalfSpace(ray, axis, Component(lower, axis), false, root)
                || !ClipBihHalfSpace(ray, axis, Component(upper, axis), true, root)) return false;
        }

        BihCell pending[BihStackCapacity];
        unsigned int pendingCount = 1u;
        pending[0] = root;
        bool found = false;
        while (pendingCount != 0u)
        {
            BihCell cell = pending[--pendingCount];
            if (cell.NearDistance > hit.Distance) continue;
            cell.FarDistance = fminf(cell.FarDistance, hit.Distance);
            const PackedAccelerationNode node = nodes[cell.NodeIndex];
            if (node.SplitAxis != BihLeafAxis)
            {
                BihCell left = {cell.NodeIndex + 1u, cell.NearDistance, cell.FarDistance};
                BihCell right = {node.AboveChild, cell.NearDistance, cell.FarDistance};
                const float leftMaximum = node.SplitPosition;
                const float rightMinimum = node.SecondPlane;
                const bool visitLeft = ClipBihHalfSpace(ray, node.SplitAxis, leftMaximum, true, left);
                const bool visitRight = ClipBihHalfSpace(ray, node.SplitAxis, rightMinimum, false, right);
                if (visitLeft && visitRight)
                {
                    // LIFO ordering visits the smaller entry parameter first.
                    const bool leftFirst = left.NearDistance <= right.NearDistance;
                    pending[pendingCount++] = leftFirst ? right : left;
                    pending[pendingCount++] = leftFirst ? left : right;
                }
                else if (visitLeft) pending[pendingCount++] = left;
                else if (visitRight) pending[pendingCount++] = right;
                continue;
            }
            for (unsigned int offset = 0u; offset < node.PrimitiveCount; ++offset)
            {
                const unsigned int reference = primitiveRefs[node.FirstOrEscape + offset].TypeAndIndex;
                const bool sphere = (reference & SphereBit) != 0u;
                const unsigned int originalIndex = sphere
                    ? constants.TriangleCount + (reference & ~SphereBit) : reference;
                Hit candidate = hit;
                ++primitiveTests;
                const bool accepted = sphere
                    ? IntersectSphere(spheres[reference & ~SphereBit], ray, candidate)
                    : IntersectTriangle(triangles[reference], ray, candidate);
                if (!accepted || candidate.Distance > hit.Distance
                    || (candidate.Distance == hit.Distance && originalIndex >= hit.PrimitiveIndex)) continue;
                candidate.PrimitiveIndex = originalIndex;
                hit = candidate;
                found = true;
                if (anyHit) return true;
            }
        }
        return found;
    }
