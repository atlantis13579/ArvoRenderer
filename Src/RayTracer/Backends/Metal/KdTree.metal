// PBRT 3rd ed., "Kd-Tree Accelerator": retain a finite parameter range for
// every traversed spatial cell and visit the side nearest the ray first.
constant uint KdLeafAxis = 3u;
constant uint KdPendingCapacity = 33u; // DepthLimit in the native host builder + root.

struct KdTraversalEntry
{
    uint Index;
    float Begin;
    float End;
};

bool IntersectKdTree(
    ObjectGeometry object,
    RayData ray,
    bool anyHit,
    thread HitData& hit,
    thread uint2& primitiveTests)
{
    hit.PrimitiveIndex = InvalidPrimitiveIndex;
    if (object.NodeCount == 0u) return false;
    const float3 lower = float3(object.Nodes[0].Minimum);
    const float3 upper = float3(object.Nodes[0].Maximum);
    float entry = ray.MinimumDistance;
    float exit = min(ray.MaximumDistance, hit.Distance);
    for (uint axis = 0u; axis < KdLeafAxis; ++axis)
    {
        if (ray.Direction[axis] == 0.0f)
        {
            if (ray.Origin[axis] < lower[axis] || ray.Origin[axis] > upper[axis]) return false;
        }
        else
        {
            const float a = (lower[axis] - ray.Origin[axis]) / ray.Direction[axis];
            const float b = (upper[axis] - ray.Origin[axis]) / ray.Direction[axis];
            entry = max(entry, min(a, b));
            exit = min(exit, max(a, b));
            if (entry > exit) return false;
        }
    }

    KdTraversalEntry deferred[KdPendingCapacity];
    uint count = 1u;
    deferred[0] = {0u, entry, exit};
    bool found = false;
    while (count != 0u)
    {
        KdTraversalEntry current = deferred[--count];
        if (current.Begin > hit.Distance) continue;
        current.End = min(current.End, hit.Distance);
        PackedAccelerationNode node = object.Nodes[current.Index];
        while (node.SplitAxis != KdLeafAxis)
        {
            const uint lowerChild = current.Index + 1u;
            const uint upperChild = node.AboveChild;
            const float direction = ray.Direction[node.SplitAxis];
            const float origin = ray.Origin[node.SplitAxis];
            if (direction == 0.0f)
            {
                if (origin == node.SplitPosition)
                    deferred[count++] = {upperChild, current.Begin, current.End};
                current.Index = origin <= node.SplitPosition ? lowerChild : upperChild;
            }
            else
            {
                const float crossing = (node.SplitPosition - origin) / direction;
                const uint enteringChild = direction < 0.0f ? upperChild : lowerChild;
                const uint leavingChild = direction < 0.0f ? lowerChild : upperChild;
                if (crossing < current.Begin)
                    current.Index = leavingChild;
                else if (crossing > current.End)
                    current.Index = enteringChild;
                else
                {
                    deferred[count++] = {leavingChild, crossing, current.End};
                    current.Index = enteringChild;
                    current.End = crossing;
                }
            }
            node = object.Nodes[current.Index];
        }
        for (uint referenceOffset = 0u; referenceOffset < node.PrimitiveCount; ++referenceOffset)
        {
            const uint reference = object.References[node.FirstOrEscape + referenceOffset];
            const uint primitiveIndex = (reference & SphereBit) == 0u
                ? reference : object.TriangleCount + (reference & ~SphereBit);
            HitData proposed = hit;
            IncrementCounter(primitiveTests);
            bool intersects;
            if ((reference & SphereBit) == 0u)
                intersects = IntersectTriangle(object.Triangles[reference], ray, proposed);
            else
                intersects = IntersectSphere(object.Spheres[reference & ~SphereBit], ray, proposed);
            if (!intersects) continue;
            if (proposed.Distance > hit.Distance
                || (proposed.Distance == hit.Distance && primitiveIndex >= hit.PrimitiveIndex)) continue;
            proposed.PrimitiveIndex = primitiveIndex;
            hit = proposed;
            found = true;
            if (anyHit) return true;
        }
    }
    return found;
}
