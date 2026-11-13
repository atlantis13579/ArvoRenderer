// Wachter and Keller (2006), "Instant Ray Tracing: The Bounding Interval
// Hierarchy", https://doi.org/10.2312/EGWR/EGSR06/139-149 .
// Two independent halfspaces clip each inherited closed cell interval. Unlike
// a single split, they may overlap or leave a gap. Primitive intervals remain
// open and retain the original ray parameter. No kd or hardware trace is called.
// SoftwareNodes (t9), SoftwarePrimitiveIndices (t10), and software root constants
// are declared by the same DXR backend's storage interface.
static const uint BihLeafAxis = 3u;
static const uint BihNodeByteStride = 32u;
static const uint BihAxisByteOffset = 16u;
// BihBuilder.h permits depth 48: one pending sibling per level plus the root.
static const uint BihStackCapacity = 49u;

struct BihNode
{
    float LeftMaximum;
    float RightMinimum;
    uint First;
    uint Count;
    uint Axis;
};

BihNode LoadBihNode(uint index, uint nodeBytes)
{
    uint4 fields = SoftwareNodes.Load4(nodeBytes + index * BihNodeByteStride);
    BihNode node;
    node.LeftMaximum = asfloat(fields.x);
    node.RightMinimum = asfloat(fields.y);
    node.First = fields.z;
    node.Count = fields.w;
    node.Axis = SoftwareNodes.Load(nodeBytes + index * BihNodeByteStride + BihAxisByteOffset);
    return node;
}

struct BihVisit
{
    uint Node;
    float Near;
    float Far;
};

bool BihRootInterval(float3 origin, float3 direction, inout float nearDistance, inout float farDistance,
                     InstanceGeometry geometry)
{
    [unroll] for (uint axis = 0; axis < 3; ++axis)
    {
        if (direction[axis] == 0.0)
        {
            if (origin[axis] < geometry.Minimum[axis] || origin[axis] > geometry.Maximum[axis]) return false;
            continue;
        }
        float low = (geometry.Minimum[axis] - origin[axis]) / direction[axis];
        float high = (geometry.Maximum[axis] - origin[axis]) / direction[axis];
        nearDistance = max(nearDistance, min(low, high));
        farDistance = min(farDistance, max(low, high));
        if (nearDistance > farDistance) return false;
    }
    return true;
}

Payload CastBihRay(float3 origin, float3 direction, float minimumDistance, float maximumDistance,
                   InstanceGeometry geometry)
{
    Payload payload = MissPayload(maximumDistance);
    BihVisit root;
    root.Node = 0;
    root.Near = minimumDistance;
    root.Far = maximumDistance;
    if (geometry.NodeCount == 0 || minimumDistance >= maximumDistance ||
        !BihRootInterval(origin, direction, root.Near, root.Far, geometry))
        return payload;
    BihVisit pending[BihStackCapacity];
    uint pendingCount = 1;
    pending[0] = root;
    while (pendingCount != 0)
    {
        BihVisit visit = pending[--pendingCount];
        if (visit.Near > asfloat(payload.DistanceBits) || visit.Near > visit.Far) continue;
        BihNode node = LoadBihNode(visit.Node, geometry.NodeBytes);
        if (node.Axis == BihLeafAxis)
        {
            for (uint index = 0; index < node.Count; ++index)
            {
                uint primitive = SoftwarePrimitiveIndices[geometry.ReferenceOffset + node.First + index];
                float distance;
                if (primitive < geometry.TriangleCount)
                {
                    float2 uv;
                    if (IntersectTriangle(geometry.TriangleOffset + primitive, origin, direction,
                                          minimumDistance, maximumDistance, distance, uv) &&
                        (distance < asfloat(payload.DistanceBits) ||
                         (distance == asfloat(payload.DistanceBits) && primitive < payload.PrimitiveIndex)))
                    {
                        payload =
                            TriangleSurface(geometry.TriangleOffset + primitive, uv, distance, direction);
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

        BihVisit left = visit, right = visit;
        left.Node = node.First;
        right.Node = node.First + 1;
        bool leftValid = true, rightValid = true;
        float component = direction[node.Axis];
        float position = origin[node.Axis];
        if (component == 0.0)
        {
            // This also handles negative zero without division. A parallel ray
            // can belong to both overlap cells, neither gap cell, or one side.
            leftValid = position <= node.LeftMaximum;
            rightValid = position >= node.RightMinimum;
        }
        else
        {
            float leftPlane = (node.LeftMaximum - position) / component;
            float rightPlane = (node.RightMinimum - position) / component;
            if (component > 0.0)
            {
                left.Far = min(left.Far, leftPlane);
                right.Near = max(right.Near, rightPlane);
            }
            else
            {
                left.Near = max(left.Near, leftPlane);
                right.Far = min(right.Far, rightPlane);
            }
        }
        leftValid = leftValid && left.Near <= left.Far && left.Near <= asfloat(payload.DistanceBits);
        rightValid = rightValid && right.Near <= right.Far && right.Near <= asfloat(payload.DistanceBits);
        if (leftValid && rightValid)
        {
            // Order by actual clipped entry distance, including overlapping cells.
            if (left.Near <= right.Near)
            {
                pending[pendingCount++] = right;
                pending[pendingCount++] = left;
            }
            else
            {
                pending[pendingCount++] = left;
                pending[pendingCount++] = right;
            }
        }
        else if (leftValid)
            pending[pendingCount++] = left;
        else if (rightValid)
            pending[pendingCount++] = right;
    }
    return payload;
}
