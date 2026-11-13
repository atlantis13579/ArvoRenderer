// Stackless escape-link traversal over the backend's packed BVH. Primitive
// indices are stable even though the builder reorders leaf references.
bool IntersectBounds(PackedAccelerationNode node, RayData ray, float closestDistance)
{
    const float3 minimum = float3(node.Minimum);
    const float3 maximum = float3(node.Maximum);
    float nearDistance = ray.MinimumDistance;
    float farDistance = min(ray.MaximumDistance, closestDistance);
    for (uint axis = 0u; axis < 3u; ++axis)
    {
        const float origin = ray.Origin[axis];
        const float direction = ray.Direction[axis];
        if (direction == 0.0f)
        {
            if (origin < minimum[axis] || origin > maximum[axis])
            {
                return false;
            }
            continue;
        }
        const float distance0Unsorted = (minimum[axis] - origin) / direction;
        const float distance1Unsorted = (maximum[axis] - origin) / direction;
        float distance0 = distance0Unsorted;
        float distance1 = distance1Unsorted;
        if (distance0 > distance1)
        {
            const float temporary = distance0;
            distance0 = distance1;
            distance1 = temporary;
        }
        nearDistance = max(nearDistance, distance0);
        farDistance = min(farDistance, distance1);
        if (nearDistance > farDistance)
        {
            return false;
        }
    }
    return true;
}

bool IntersectBvh(
    ObjectGeometry object,
    RayData ray,
    bool anyHit,
    thread HitData& hit,
    thread uint2& primitiveTests)
{
    hit.PrimitiveIndex = InvalidPrimitiveIndex;
    if (object.NodeCount == 0u)
    {
        return false;
    }
    bool found = false;
    uint nodeIndex = 0u;
    while (nodeIndex < object.NodeCount)
    {
        const PackedAccelerationNode node = object.Nodes[nodeIndex];
        if (!IntersectBounds(node, ray, hit.Distance))
        {
            nodeIndex = node.PrimitiveCount == 0u
                ? node.FirstOrEscape
                : nodeIndex + 1u;
            continue;
        }
        if (node.PrimitiveCount == 0u)
        {
            ++nodeIndex;
            continue;
        }
        for (uint offset = 0u; offset < node.PrimitiveCount; ++offset)
        {
            const uint encoded = object.References[node.FirstOrEscape + offset];
            IncrementCounter(primitiveTests);
            HitData candidate = hit;
            const uint primitiveIndex = (encoded & SphereBit) != 0u
                ? object.TriangleCount + (encoded & ~SphereBit) : encoded;
            bool primitiveHit = false;
            if ((encoded & SphereBit) != 0u)
            {
                primitiveHit = IntersectSphere(
                    object.Spheres[encoded & ~SphereBit],
                    ray,
                    candidate);
            }
            else
            {
                primitiveHit = IntersectTriangle(
                    object.Triangles[encoded],
                    ray,
                    candidate);
            }
            if (primitiveHit)
            {
                if (candidate.Distance > hit.Distance
                    || (candidate.Distance == hit.Distance && primitiveIndex >= hit.PrimitiveIndex)) continue;
                candidate.PrimitiveIndex = primitiveIndex;
                hit = candidate;
                found = true;
                if (anyHit)
                {
                    return true;
                }
            }
        }
        ++nodeIndex;
    }
    return found;
}
