// Production-module diagnostics: the same BVH/primitive functions service both
// camera paths and supplied rays. Packed vectors keep the host ABI explicit.
struct QueryHit
{
    packed_float3 Normal;
    uint DistanceBits;
    packed_float3 Color;
    uint PrimitiveIndex;
    float2 TexCoord;
    uint MaterialIndex, Flags;
    packed_float3 GeometricNormal;
    uint Reserved;
};

kernel void ArvoQueryRays(
    device const PackedAccelerationNode *nodes [[buffer(0)]], device const uint *primitiveRefs [[buffer(1)]],
    device const PackedTriangle *triangles [[buffer(2)]], device const PackedSphere *spheres [[buffer(3)]],
    constant FrameConstants &constants [[buffer(4)]], device const EncodedRay *rays [[buffer(5)]],
    device QueryHit *hits [[buffer(6)]], device const InstanceRecord *instances [[buffer(13)]],
    device const InstanceNode *instanceNodes [[buffer(14)]], device atomic_uint *instanceError [[buffer(15)]],
    uint gridIndex [[thread_position_in_grid]])
{
    constexpr uint HitFlag = 1u, TexCoordFlag = 2u, FrontFaceFlag = 4u;
    if (gridIndex >= constants.PixelCount)
        return;
    const uint index = constants.PixelBegin + gridIndex;
    const SceneGeometryBuffers geometry = {{nodes, primitiveRefs, triangles, spheres, constants.NodeCount,
                                            constants.TriangleCount, constants.Acceleration},
                                           instances,
                                           instanceNodes,
                                           instanceError,
                                           constants.InstanceNodeCount};
    HitData hit{};
    hit.Color = float3(1.0f);
    uint2 primitiveTests = uint2(0u);
    uint distanceBits = 0u;
    const bool found = IntersectEncodedScene(geometry, rays[index], false, hit, distanceBits, primitiveTests);
    QueryHit output{};
    output.Normal = packed_float3(hit.Normal);
    output.GeometricNormal = packed_float3(hit.GeometricNormal);
    output.DistanceBits = distanceBits;
    output.Color = packed_float3(hit.Color);
    output.PrimitiveIndex = hit.PrimitiveIndex;
    output.TexCoord = hit.TexCoord;
    output.MaterialIndex = hit.MaterialIndex;
    output.Flags = found ? HitFlag | (hit.HasTexCoord ? TexCoordFlag : 0u)
        | (hit.FrontFace ? FrontFaceFlag : 0u) : 0u;
    hits[index] = output;
}
