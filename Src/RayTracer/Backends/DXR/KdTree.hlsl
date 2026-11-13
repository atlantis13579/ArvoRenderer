// Native DXR software kd traversal. Nodes partition spatial cells, so crossing
// a split can visit both children; straddling primitives are duplicated by the
// independently owned DXR builder. This does not invoke hardware TraceRay.
// PBRT 3e, "Kd-Tree Accelerator", section "Traversal":
// https://pbr-book.org/3ed-2018/Primitives_and_Intersection_Acceleration/Kd-Tree_Accelerator
static const uint KdLeafAxis = 3u;
static const uint KdNodeByteStride = 16u;
// Host construction stops at depth48; at most one deferred sibling per level.
static const uint KdStackCapacity = 64u;

struct KdNode
{
    float Split;
    uint First;
    uint Count;
    uint Axis;
};

struct KdVisit
{
    uint Node;
    float Near;
    float Far;
};

KdNode LoadKdNode(uint index, uint nodeBytes)
{
    uint4 fields = SoftwareNodes.Load4(nodeBytes + index * KdNodeByteStride);
    KdNode node;
    node.Split = asfloat(fields.x);
    node.First = fields.y;
    node.Count = fields.z;
    node.Axis = fields.w;
    return node;
}

bool IntersectKdBounds(float3 origin, float3 direction, float minimumDistance, float maximumDistance,
                       out float nearDistance, out float farDistance, InstanceGeometry geometry)
{
    nearDistance = minimumDistance;
    farDistance = maximumDistance;
    [unroll] for (uint axis = 0; axis < 3; ++axis)
    {
        if (direction[axis] == 0.0)
        {
            if (origin[axis] < geometry.Minimum[axis] || origin[axis] > geometry.Maximum[axis]) return false;
        }
        else
        {
            float first = (geometry.Minimum[axis] - origin[axis]) / direction[axis];
            float second = (geometry.Maximum[axis] - origin[axis]) / direction[axis];
            nearDistance = max(nearDistance, min(first, second));
            farDistance = min(farDistance, max(first, second));
            // Bounds are closed even though primitive ray intervals are open.
            if (nearDistance > farDistance) return false;
        }
    }
    return true;
}

Payload CastKdRay(float3 origin, float3 direction, float minimumDistance, float maximumDistance,
                  InstanceGeometry geometry)
{
    Payload payload = MissPayload(maximumDistance);
    float rootNear, rootFar;
    if (geometry.NodeCount == 0 || minimumDistance >= maximumDistance ||
        !IntersectKdBounds(origin, direction, minimumDistance, maximumDistance, rootNear, rootFar, geometry))
        return payload;
    KdVisit stack[KdStackCapacity];
    uint stackCount = 1;
    stack[0].Node = 0;
    stack[0].Near = rootNear;
    stack[0].Far = rootFar;
    while (stackCount != 0)
    {
        KdVisit visit = stack[--stackCount];
        if (visit.Near > asfloat(payload.DistanceBits)) continue;
        KdNode node = LoadKdNode(visit.Node, geometry.NodeBytes);
        if (node.Axis == KdLeafAxis)
        {
            for (uint entry = 0; entry < node.Count; ++entry)
            {
                uint primitive = SoftwarePrimitiveIndices[geometry.ReferenceOffset + node.First + entry];
                float distance;
                if (primitive < geometry.TriangleCount)
                {
                    float2 barycentrics;
                    if (IntersectTriangle(geometry.TriangleOffset + primitive, origin, direction,
                                          minimumDistance, maximumDistance, distance, barycentrics) &&
                        (distance < asfloat(payload.DistanceBits) ||
                         (distance == asfloat(payload.DistanceBits) && primitive < payload.PrimitiveIndex)))
                    {
                        payload = TriangleSurface(geometry.TriangleOffset + primitive, barycentrics, distance,
                                                  direction);
                        payload.PrimitiveIndex = primitive;
                    }
                }
                else
                {
                    uint sphere = primitive - geometry.TriangleCount;
                    float3 normal;
                    if (IntersectSphere(geometry.SphereOffset + sphere, origin, direction, minimumDistance,
                                        maximumDistance, distance, normal) &&
                        (distance < asfloat(payload.DistanceBits) ||
                         (distance == asfloat(payload.DistanceBits) && primitive < payload.PrimitiveIndex)))
                    {
                        payload = SphereSurface(geometry.SphereOffset + sphere, distance, normal, direction);
                        payload.PrimitiveIndex = primitive;
                    }
                }
            }
            continue;
        }
        float axisDirection = direction[node.Axis];
        uint low = node.First;
        uint high = low + 1;
        if (axisDirection == 0.0)
        {
            // A ray in the plane may choose either child: all touching bounds
            // are assigned to both, including planar and straddling primitives.
            visit.Node = origin[node.Axis] <= node.Split ? low : high;
            stack[stackCount++] = visit;
            continue;
        }
        uint nearChild = axisDirection > 0.0 ? low : high;
        uint farChild = axisDirection > 0.0 ? high : low;
        float planeDistance = (node.Split - origin[node.Axis]) / axisDirection;
        if (planeDistance < visit.Near)
        {
            visit.Node = farChild;
            stack[stackCount++] = visit;
        }
        else if (planeDistance > visit.Far)
        {
            visit.Node = nearChild;
            stack[stackCount++] = visit;
        }
        else
        {
            KdVisit deferred;
            deferred.Node = farChild;
            deferred.Near = max(planeDistance, visit.Near);
            deferred.Far = visit.Far;
            stack[stackCount++] = deferred;
            visit.Node = nearChild;
            visit.Far = min(planeDistance, visit.Far);
            stack[stackCount++] = visit;
        }
    }
    return payload;
}
