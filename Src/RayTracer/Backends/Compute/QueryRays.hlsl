// Direct native traversal probes share the exact primitive tests and BVH used
// above by production transport. No camera, sampling, materials or film resolve.
struct QueryHit
{
    float3 Normal;
    float Distance;
    float3 Color;
    uint PrimitiveIndex;
    float2 TexCoord;
    uint MaterialIndex;
    uint Flags;
    float3 GeometricNormal;
    uint Reserved;
};
StructuredBuffer<RayData> QueryInputs : register(t8);
RWStructuredBuffer<QueryHit> QueryOutputs : register(u5);

[numthreads(256, 1, 1)]
void QueryRaysKernel(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    if (dispatchThreadId.x >= PixelCount) return;
    const uint index = PixelBegin + dispatchThreadId.x;
    const RayData ray = QueryInputs[index];
    HitData hit = (HitData)0;
    hit.Distance = ray.MaximumDistance;
    hit.Color = 1.0f.xxx;
    hit.PrimitiveIndex = InvalidPrimitiveIndex;
    uint2 primitiveTests = uint2(0u, 0u);
    const bool found = IntersectScene(ray, false, hit, primitiveTests);
    QueryHit result;
    result.Normal = hit.Normal;
    result.GeometricNormal = hit.GeometricNormal;
    result.Reserved = 0u;
    result.Distance = hit.Distance;
    result.Color = hit.Color;
    result.PrimitiveIndex = hit.PrimitiveIndex;
    result.TexCoord = hit.TexCoord;
    result.MaterialIndex = hit.MaterialIndex;
    static const uint HitFlag = 1u, TexCoordFlag = 2u, FrontFaceFlag = 4u;
    result.Flags = found ? HitFlag | (hit.HasTexCoord != 0u ? TexCoordFlag : 0u)
        | (hit.FrontFace != 0u ? FrontFaceFlag : 0u) : 0u;
    QueryOutputs[index] = result;
}
