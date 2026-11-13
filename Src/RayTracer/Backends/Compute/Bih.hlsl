// BIH traversal clips each child against its own plane, so overlapping children
// retain both intervals and separated children leave an empty gap. References
// belong to one leaf only; primitive tests still use the original open ray range.
// Wachter and Keller (2006), "Instant Ray Tracing: The Bounding Interval Hierarchy":
// https://doi.org/10.2312/EGWR/EGSR06/139-149
static const uint BihLeafAxis = 3u;
static const uint BihMaximumDepth = 48u;
static const uint BihBoundsBytes = 32u;
static const uint BihNodeBytes = 32u;

struct BihNode
{
    float LeftMaximum;
    float RightMinimum;
    uint Axis;
    uint RightChildOrFirstReference;
    uint PrimitiveCount;
};

BihNode LoadBihNode(uint index, uint byteOffset)
{
    static const uint PrimitiveCountOffset = 16u;
    const uint address = byteOffset + BihBoundsBytes + index * BihNodeBytes;
    const uint4 planesAndChildren = Nodes.Load4(address);
    BihNode node;
    node.LeftMaximum = asfloat(planesAndChildren.x);
    node.RightMinimum = asfloat(planesAndChildren.y);
    node.Axis = planesAndChildren.z;
    node.RightChildOrFirstReference = planesAndChildren.w;
    node.PrimitiveCount = Nodes.Load(address + PrimitiveCountOffset);
    return node;
}

bool BihClipMaximum(float origin, float direction, float maximum, inout float nearT, inout float farT)
{
    if (direction == 0.0f) return origin <= maximum;
    const float planeT = (maximum - origin) / direction;
    if (direction > 0.0f) farT = min(farT, planeT);
    else nearT = max(nearT, planeT);
    return nearT <= farT;
}

bool BihClipMinimum(float origin, float direction, float minimum, inout float nearT, inout float farT)
{
    if (direction == 0.0f) return origin >= minimum;
    const float planeT = (minimum - origin) / direction;
    if (direction > 0.0f) nearT = max(nearT, planeT);
    else farT = min(farT, planeT);
    return nearT <= farT;
}

bool BihInitialInterval(RayData ray, uint byteOffset, out float nearT, out float farT)
{
    static const uint MaximumBoundsOffset = 16u;
    const float3 minimum = asfloat(Nodes.Load3(byteOffset));
    const float3 maximum = asfloat(Nodes.Load3(byteOffset + MaximumBoundsOffset));
    nearT = ray.MinimumDistance;
    farT = ray.MaximumDistance;
    [unroll] for (uint axis = 0u; axis < 3u; ++axis)
    {
        if (!BihClipMinimum(ray.Origin[axis], ray.Direction[axis], minimum[axis], nearT, farT)
            || !BihClipMaximum(ray.Origin[axis], ray.Direction[axis], maximum[axis], nearT, farT)) return false;
    }
    return true;
}

struct BihVisit
{
    uint Index;
    float NearT;
    float FarT;
};

bool IntersectBih(RayData ray, GeometryRange geometry, bool anyHit, inout HitData hit, inout uint2 primitiveTests)
{
    if (geometry.NodeCount == 0u) return false;
    BihVisit current;
    current.Index = 0u;
    if (!BihInitialInterval(ray, geometry.NodeByteOffset, current.NearT, current.FarT)) return false;
    // Reuse invocation-private scratch across sequential ray queries. Only
    // pendingCount entries are live; each is written before it can be popped.
    static BihVisit pending[BihMaximumDepth + 1u];
    uint pendingCount = 0u;
    bool found = false;
    [loop] while (true)
    {
        if (current.NearT <= hit.Distance)
        {
            const BihNode node = LoadBihNode(current.Index, geometry.NodeByteOffset);
            if (node.Axis != BihLeafAxis)
            {
                BihVisit left = current, right = current;
                left.Index = current.Index + 1u;
                right.Index = node.RightChildOrFirstReference;
                const float origin = ray.Origin[node.Axis], direction = ray.Direction[node.Axis];
                const bool visitLeft = BihClipMaximum(origin, direction, node.LeftMaximum, left.NearT, left.FarT)
                    && left.NearT <= hit.Distance;
                const bool visitRight = BihClipMinimum(origin, direction, node.RightMinimum, right.NearT, right.FarT)
                    && right.NearT <= hit.Distance;
                if (visitLeft && visitRight)
                {
                    // Sort the actual clipped intervals, including overlaps and
                    // parallel rays, rather than assuming disjoint split cells.
                    if (right.NearT < left.NearT)
                    {
                        pending[pendingCount++] = left;
                        current = right;
                    }
                    else
                    {
                        pending[pendingCount++] = right;
                        current = left;
                    }
                    continue;
                }
                if (visitLeft) { current = left; continue; }
                if (visitRight) { current = right; continue; }
            }
            else
            {
                [loop] for (uint offset = 0u; offset < node.PrimitiveCount; ++offset)
                {
                    const uint encoded = PrimitiveRefs[geometry.ReferenceOffset + node.RightChildOrFirstReference + offset];
                    IncrementCounter(primitiveTests);
                    bool primitiveHit;
                    if ((encoded & SphereBit) == 0u)
                        primitiveHit = IntersectTriangle(Triangles[geometry.TriangleOffset + encoded], encoded, ray, hit);
                    else
                    {
                        const uint sphere = encoded & ~SphereBit;
                        primitiveHit = IntersectSphere(Spheres[geometry.SphereOffset + sphere], geometry.TriangleCount + sphere, ray, hit);
                    }
                    if (primitiveHit)
                    {
                        found = true;
                        if (anyHit) return true;
                    }
                }
            }
        }
        if (pendingCount == 0u) break;
        current = pending[--pendingCount];
    }
    return found;
}
