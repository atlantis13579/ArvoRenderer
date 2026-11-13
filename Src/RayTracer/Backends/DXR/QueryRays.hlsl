// Native closest-hit diagnostics use production traversal without shading.
// EncodedRay and Payload retain the public 32/64-byte float ABI bit for bit.
#include "Globals.hlsl"
#include "Surface.hlsl"
#include "InstancePrograms.hlsli"
#include "Triangle.hlsl"
#include "Sphere.hlsl"
#include "TriangleIntersection.hlsl"
#include "MotionTriangle.hlsl"
#include "Select.hlsl"

StructuredBuffer<EncodedRay> InputRays : register(t8);
RWStructuredBuffer<Payload> QueryHits : register(u3);

[shader("raygeneration")]
void RayGen()
{
    uint index = DispatchRaysIndex().x;
    QueryHits[index] = CastEncodedRay(InputRays[index]);
}
