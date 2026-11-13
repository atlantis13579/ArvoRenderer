// Flattened depth-first BVH traversal. Interior nodes carry an escape index;
// leaves reference a contiguous range of source-primitive identifiers.
// Equal-distance closest hits use the lower source index regardless of BVH order.
PackedBvhNode LoadBvhNode(uint index, uint byteOffset)
{
    static const uint BvhNodeBytes = 32u;
    static const uint VectorBytes = 16u;
    const uint4 lower = Nodes.Load4(byteOffset + index * BvhNodeBytes);
    const uint4 upper = Nodes.Load4(byteOffset + index * BvhNodeBytes + VectorBytes);
    PackedBvhNode node;
    node.Minimum = asfloat(lower.xyz);
    node.FirstOrEscape = lower.w;
    node.Maximum = asfloat(upper.xyz);
    node.PrimitiveCount = upper.w;
    return node;
}

bool IntersectBvh(RayData ray, GeometryRange geometry, bool anyHit, inout HitData hit, inout uint2 primitiveTests)
{
    if (geometry.NodeCount == 0) return false;
    bool found = false;
    uint nodeIndex = 0;
    [loop] while (nodeIndex < geometry.NodeCount)
    {
        const PackedBvhNode node = LoadBvhNode(nodeIndex, geometry.NodeByteOffset);
        if (!IntersectBounds(node, ray, hit.Distance))
        {
            nodeIndex = node.PrimitiveCount == 0 ? node.FirstOrEscape : nodeIndex + 1;
            continue;
        }
        if (node.PrimitiveCount == 0)
        {
            ++nodeIndex;
            continue;
        }
        [loop] for (uint offset = 0; offset < node.PrimitiveCount; ++offset)
        {
            const uint encoded = PrimitiveRefs[geometry.ReferenceOffset + node.FirstOrEscape + offset];
            IncrementCounter(primitiveTests);
            bool primitiveHit;
            if ((encoded & SphereBit) != 0u)
            {
                const uint sphere = encoded & ~SphereBit;
                primitiveHit = IntersectSphere(Spheres[geometry.SphereOffset + sphere], geometry.TriangleCount + sphere, ray, hit);
            }
            else
                primitiveHit = IntersectTriangle(Triangles[geometry.TriangleOffset + encoded], encoded, ray, hit);
            if (primitiveHit)
            {
                found = true;
                if (anyHit) return true;
            }
        }
        ++nodeIndex;
    }
    return found;
}
