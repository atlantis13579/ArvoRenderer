// This entry point lives in each production module and calls its real traversal.
// It exposes intersection data without camera sampling or shading.
struct QueryRay
{
    Vec4 OriginMinimum;
    Vec4 DirectionMaximum;
};

struct QueryHit
{
    Vec3 Normal;
    float Distance;
    Vec3 Color;
    unsigned int PrimitiveIndex;
    float TexCoordX, TexCoordY;
    unsigned int MaterialIndex, Flags;
    Vec3 GeometricNormal;
    unsigned int Reserved;
};
static_assert(sizeof(QueryRay) == 32 && sizeof(QueryHit) == 64, "CUDA query ABI mismatch.");

extern "C" __global__ void QueryRaysKernel(
    const PackedAccelerationNode* nodes,
    const PackedPrimitiveRef* primitiveRefs,
    const PackedTriangle* triangles,
    const PackedSphere* spheres,
    const FrameConstants* constantsPointer,
    const QueryRay* rays,
    QueryHit* hits)
{
    constexpr unsigned int HitFlag = 1u, TexCoordFlag = 2u, FrontFaceFlag = 4u;
    const FrameConstants constants = *constantsPointer;
    const unsigned int localIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (localIndex >= constants.PixelCount) return;
    const unsigned int index = constants.PixelBegin + localIndex;
    const QueryRay input = rays[index];
    const Ray ray{ToVec3(input.OriginMinimum), ToVec3(input.DirectionMaximum), input.OriginMinimum.W, input.DirectionMaximum.W};
    Hit hit{};
    hit.Distance = ray.MaximumDistance;
    hit.Color = {1.0f, 1.0f, 1.0f};
    unsigned long long primitiveTests = 0;
    const bool found = IntersectScene(nodes, primitiveRefs, triangles, spheres, constants, ray, false, hit, primitiveTests);
    QueryHit output{};
    output.Normal = hit.Normal;
    output.GeometricNormal = hit.GeometricNormal;
    output.Distance = hit.Distance;
    output.Color = hit.Color;
    output.PrimitiveIndex = hit.PrimitiveIndex;
    output.TexCoordX = hit.TexCoordX;
    output.TexCoordY = hit.TexCoordY;
    output.MaterialIndex = hit.MaterialIndex;
    output.Flags = found ? HitFlag | (hit.HasTexCoord ? TexCoordFlag : 0u)
        | (hit.FrontFace ? FrontFaceFlag : 0u) : 0u;
    hits[index] = output;
}
