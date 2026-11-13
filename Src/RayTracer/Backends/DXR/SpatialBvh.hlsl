// Native DXR software SBVH traversal. Spatially clipped references can repeat
// primitive IDs; final surface tests always use the original open ray interval.
// Depth-first escape links skip rejected subtrees without a per-ray stack.
// Keeping traversal compact also limits driver compilation work when this leaf
// is inlined into direct-light and multi-bounce transport shaders.
static const uint SpatialNodeBytes = 32u;
static const uint SpatialNodeMaximumOffset = 16u;
struct SpatialNode
{
    float3 Minimum;
    uint FirstOrEscape;
    float3 Maximum;
    uint Count;
};
SpatialNode LoadSpatialNode(uint index, uint nodeBytes)
{
    uint address = nodeBytes + index * SpatialNodeBytes;
    uint4 lower = SoftwareNodes.Load4(address);
    uint4 upper = SoftwareNodes.Load4(address + SpatialNodeMaximumOffset);
    SpatialNode node;
    node.Minimum = asfloat(lower.xyz);
    node.FirstOrEscape = lower.w;
    node.Maximum = asfloat(upper.xyz);
    node.Count = upper.w;
    return node;
}
bool SpatialBoxInterval(SpatialNode node, float3 origin, float3 direction, float minimumDistance,
                        float maximumDistance, out float entry)
{
    entry = minimumDistance;
    float exit = maximumDistance;
    for (uint axis = 0; axis < 3; ++axis)
    {
        if (direction[axis] == 0.0)
        {
            if (origin[axis] < node.Minimum[axis] || origin[axis] > node.Maximum[axis]) return false;
        }
        else
        {
            float first = (node.Minimum[axis] - origin[axis]) / direction[axis];
            float last = (node.Maximum[axis] - origin[axis]) / direction[axis];
            entry = max(entry, min(first, last));
            exit = min(exit, max(first, last));
            if (entry > exit) return false;
        }
    }
    return true;
}
Payload CastSpatialBvhRay(float3 origin, float3 direction, float minimumDistance, float maximumDistance,
                          InstanceGeometry geometry)
{
    Payload closest = MissPayload(maximumDistance);
    if (geometry.NodeCount == 0 || minimumDistance >= maximumDistance) return closest;
    uint nodeIndex = 0;
    while (nodeIndex < geometry.NodeCount)
    {
        SpatialNode node = LoadSpatialNode(nodeIndex, geometry.NodeBytes);
        float entry;
        if (!SpatialBoxInterval(node, origin, direction, minimumDistance, asfloat(closest.DistanceBits),
                                entry))
        {
            nodeIndex = node.Count == 0 ? node.FirstOrEscape : nodeIndex + 1;
            continue;
        }
        if (node.Count > 0)
        {
            for (uint index = 0; index < node.Count; ++index)
            {
                uint primitive =
                    SoftwarePrimitiveIndices[geometry.ReferenceOffset + node.FirstOrEscape + index];
                float distance;
                if (primitive < geometry.TriangleCount)
                {
                    float2 barycentrics;
                    if (IntersectTriangle(geometry.TriangleOffset + primitive, origin, direction,
                                          minimumDistance, maximumDistance, distance, barycentrics) &&
                        (distance < asfloat(closest.DistanceBits) ||
                         (distance == asfloat(closest.DistanceBits) && primitive < closest.PrimitiveIndex)))
                    {
                        closest = TriangleSurface(geometry.TriangleOffset + primitive, barycentrics, distance,
                                                  direction);
                        closest.PrimitiveIndex = primitive;
                    }
                }
                else
                {
                    uint sphere = primitive - geometry.TriangleCount;
                    float3 normal;
                    if (IntersectSphere(geometry.SphereOffset + sphere, origin, direction, minimumDistance,
                                        maximumDistance, distance, normal) &&
                        (distance < asfloat(closest.DistanceBits) ||
                         (distance == asfloat(closest.DistanceBits) && primitive < closest.PrimitiveIndex)))
                    {
                        closest = SphereSurface(geometry.SphereOffset + sphere, distance, normal, direction);
                        closest.PrimitiveIndex = primitive;
                    }
                }
            }
        }
        ++nodeIndex;
    }
    return closest;
}
