// Two-plane object hierarchy, following Wachter and Keller's "Instant Ray
// Tracing: The Bounding Interval Hierarchy" (2006). Children may overlap or
// leave an empty gap. Closed cell intervals retain points on clipping planes.
constant uint BihLeafAxis = 3u;
constant uint BihDeferredCapacity = 33u; // MaximumDepth in BihBuilder.cpp plus root.

struct BihVisit
{
    uint Node;
    float Entry;
    float Exit;
};

bool IntersectBih(
    ObjectGeometry object,
    RayData ray,
    bool anyHit,
    thread HitData& hit,
    thread uint2& primitiveTests)
{
    hit.PrimitiveIndex = InvalidPrimitiveIndex;
    if (object.NodeCount == 0u) return false;
    BihVisit current = {0u, ray.MinimumDistance, min(ray.MaximumDistance, hit.Distance)};
    const float3 lower = float3(object.Nodes[0].Minimum), upper = float3(object.Nodes[0].Maximum);
    for (uint axis = 0u; axis < BihLeafAxis; ++axis)
    {
        const float direction = ray.Direction[axis];
        const float origin = ray.Origin[axis];
        if (direction == 0.0f)
        {
            if (origin < lower[axis] || origin > upper[axis]) return false;
        }
        else
        {
            const float lowerTime = (lower[axis] - origin) / direction;
            const float upperTime = (upper[axis] - origin) / direction;
            current.Entry = max(current.Entry, min(lowerTime, upperTime));
            current.Exit = min(current.Exit, max(lowerTime, upperTime));
            if (current.Entry > current.Exit) return false;
        }
    }

    BihVisit deferred[BihDeferredCapacity];
    uint count = 0u;
    bool found = false;
    for (;;)
    {
        if (current.Entry <= hit.Distance)
        {
            current.Exit = min(current.Exit, hit.Distance);
            const PackedAccelerationNode node = object.Nodes[current.Node];
            if (node.SplitAxis < BihLeafAxis)
            {
                const float origin = ray.Origin[node.SplitAxis];
                const float direction = ray.Direction[node.SplitAxis];
                const float leftMaximum = node.SplitPosition;
                const float rightMinimum = node.SecondPlane;
                BihVisit left = {current.Node + 1u, current.Entry, current.Exit};
                BihVisit right = {node.AboveChild, current.Entry, current.Exit};
                bool leftVisible, rightVisible;
                if (direction == 0.0f)
                {
                    leftVisible = origin <= leftMaximum;
                    rightVisible = origin >= rightMinimum;
                }
                else
                {
                    const float leftBoundary = (leftMaximum - origin) / direction;
                    const float rightBoundary = (rightMinimum - origin) / direction;
                    if (direction > 0.0f)
                    {
                        left.Exit = min(left.Exit, leftBoundary);
                        right.Entry = max(right.Entry, rightBoundary);
                    }
                    else
                    {
                        left.Entry = max(left.Entry, leftBoundary);
                        right.Exit = min(right.Exit, rightBoundary);
                    }
                    leftVisible = left.Entry <= left.Exit;
                    rightVisible = right.Entry <= right.Exit;
                }
                if (leftVisible || rightVisible)
                {
                    if (leftVisible && rightVisible)
                    {
                        if (left.Entry <= right.Entry)
                        {
                            deferred[count++] = right;
                            current = left;
                        }
                        else
                        {
                            deferred[count++] = left;
                            current = right;
                        }
                    }
                    else current = leftVisible ? left : right;
                    continue;
                }
            }
            else
            {
                for (uint offset = 0u; offset < node.PrimitiveCount; ++offset)
                {
                    const uint encoded = object.References[node.FirstOrEscape + offset];
                    HitData proposed = hit;
                    IncrementCounter(primitiveTests);
                    uint originalIndex = encoded;
                    bool accepted;
                    if ((encoded & SphereBit) != 0u)
                    {
                        originalIndex = object.TriangleCount + (encoded & ~SphereBit);
                        accepted = IntersectSphere(object.Spheres[encoded & ~SphereBit], ray, proposed);
                    }
                    else accepted = IntersectTriangle(object.Triangles[encoded], ray, proposed);
                    if (!accepted || proposed.Distance > hit.Distance
                        || (proposed.Distance == hit.Distance && originalIndex >= hit.PrimitiveIndex)) continue;
                    proposed.PrimitiveIndex = originalIndex;
                    hit = proposed;
                    found = true;
                    if (anyHit) return true;
                }
            }
        }
        if (count == 0u) break;
        current = deferred[--count];
    }
    return found;
}
