// Native split-plane traversal, independent of the BVH traversal. The initial
// slab interval and each deferred child retain the caller's unnormalized ray t.
// See PBRT 3rd ed., section 4.4.3, "Traversal":
// https://pbr-book.org/3ed-2018/Primitives_and_Intersection_Acceleration/Kd-Tree_Accelerator#Traversal
static const uint KdLeafAxis = 3u;
static const uint KdMaximumDepth = 48u; // Matches KdTreeBuilder.cpp.
static const uint KdBoundsBytes = 32u;
static const uint KdNodeBytes = 16u;

struct KdNode
{
    float SplitPosition;
    uint SplitAxis;
    uint AboveChildOrFirstReference;
    uint PrimitiveCount;
};

KdNode LoadKdNode(uint index, uint byteOffset)
{
    const uint4 words = Nodes.Load4(byteOffset + KdBoundsBytes + index * KdNodeBytes);
    KdNode node;
    node.SplitPosition = asfloat(words.x);
    node.SplitAxis = words.y;
    node.AboveChildOrFirstReference = words.z;
    node.PrimitiveCount = words.w;
    return node;
}

bool KdRootInterval(RayData ray, uint byteOffset, out float nearDistance, out float farDistance)
{
    static const uint MaximumBoundsOffset = 16u;
    const float3 minimum = asfloat(Nodes.Load3(byteOffset));
    const float3 maximum = asfloat(Nodes.Load3(byteOffset + MaximumBoundsOffset));
    nearDistance = ray.MinimumDistance;
    farDistance = ray.MaximumDistance;
    [unroll] for (uint axis = 0u; axis < 3u; ++axis)
    {
        const float direction = ray.Direction[axis];
        const float origin = ray.Origin[axis];
        if (direction == 0.0f)
        {
            if (origin < minimum[axis] || origin > maximum[axis]) return false;
            continue;
        }
        const float first = (minimum[axis] - origin) / direction;
        const float second = (maximum[axis] - origin) / direction;
        nearDistance = max(nearDistance, min(first, second));
        farDistance = min(farDistance, max(first, second));
        if (nearDistance > farDistance) return false;
    }
    return true;
}

struct KdDeferredNode
{
    uint Index;
    float NearDistance;
    float FarDistance;
};

bool IntersectKdTree(RayData ray, GeometryRange geometry, bool anyHit, inout HitData hit, inout uint2 primitiveTests)
{
    if (geometry.NodeCount == 0u) return false;
    float nearDistance, farDistance;
    if (!KdRootInterval(ray, geometry.NodeByteOffset, nearDistance, farDistance)) return false;
    // HLSL static storage is private to a shader invocation. Traversals are
    // sequential, so reuse one stack instead of one array per inlined ray call.
    static KdDeferredNode deferred[KdMaximumDepth + 1u];
    uint deferredCount = 0u;
    uint nodeIndex = 0u;
    bool found = false;
    [loop] while (true)
    {
        if (nearDistance <= hit.Distance)
        {
            const KdNode node = LoadKdNode(nodeIndex, geometry.NodeByteOffset);
            if (node.SplitAxis != KdLeafAxis)
            {
                const uint below = nodeIndex + 1u;
                const uint above = node.AboveChildOrFirstReference;
                const float origin = ray.Origin[node.SplitAxis];
                const float direction = ray.Direction[node.SplitAxis];
                if (direction == 0.0f)
                {
                    if (origin < node.SplitPosition) { nodeIndex = below; continue; }
                    if (origin > node.SplitPosition) { nodeIndex = above; continue; }
                    // A ray lying on the plane can hit geometry in either cell.
                    deferred[deferredCount].Index = above;
                    deferred[deferredCount].NearDistance = nearDistance;
                    deferred[deferredCount].FarDistance = farDistance;
                    ++deferredCount;
                    nodeIndex = below;
                    continue;
                }
                const float planeDistance = (node.SplitPosition - origin) / direction;
                const uint first = direction > 0.0f ? below : above;
                const uint second = direction > 0.0f ? above : below;
                if (planeDistance < nearDistance) { nodeIndex = second; continue; }
                if (planeDistance > farDistance) { nodeIndex = first; continue; }
                deferred[deferredCount].Index = second;
                deferred[deferredCount].NearDistance = planeDistance;
                deferred[deferredCount].FarDistance = farDistance;
                ++deferredCount;
                nodeIndex = first;
                farDistance = planeDistance;
                continue;
            }
            [loop] for (uint offset = 0u; offset < node.PrimitiveCount; ++offset)
            {
                const uint encoded = PrimitiveRefs[geometry.ReferenceOffset + node.AboveChildOrFirstReference + offset];
                IncrementCounter(primitiveTests);
                bool primitiveHit;
                if ((encoded & SphereBit) != 0u)
                {
                    const uint sphere = encoded & ~SphereBit;
                    primitiveHit = IntersectSphere(Spheres[geometry.SphereOffset + sphere], geometry.TriangleCount + sphere, ray, hit);
                }
                else primitiveHit = IntersectTriangle(Triangles[geometry.TriangleOffset + encoded], encoded, ray, hit);
                if (primitiveHit)
                {
                    found = true;
                    if (anyHit) return true;
                }
            }
        }
        if (deferredCount == 0u) break;
        --deferredCount;
        nodeIndex = deferred[deferredCount].Index;
        nearDistance = deferred[deferredCount].NearDistance;
        farDistance = deferred[deferredCount].FarDistance;
    }
    return found;
}
