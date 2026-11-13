#ifndef ARVO_COMPUTE_GEOMETRY_TYPES
#define ARVO_COMPUTE_GEOMETRY_TYPES

static const uint SphereBit = 0x80000000u;
static const uint InvalidPrimitiveIndex = 0xffffffffu;

struct GeometryRange
{
    uint NodeByteOffset, NodeCount, ReferenceOffset, TriangleOffset;
    uint SphereOffset, TriangleCount, FirstPrimitive, Reserved;
};

struct PackedBvhNode
{
    float3 Minimum;
    uint FirstOrEscape;
    float3 Maximum;
    uint PrimitiveCount;
};

struct PackedTriangle
{
    float4 Position0;
    float4 Edge1;
    float4 Edge2;
    float4 Normal0;
    float4 Normal1;
    float4 Normal2;
    float4 Color0;
    float4 Color1;
    float4 Color2;
    float4 TexCoord01;
    float4 TexCoord2;
    uint MaterialIndex;
    uint Flags;
    uint2 Padding;
};

struct PackedSphere
{
    float4 CenterRadius;
    uint MaterialIndex;
    uint3 Padding;
};

struct RayData
{
    float3 Origin;
    float3 Direction;
    float MinimumDistance;
    float MaximumDistance;
};

struct HitData
{
    float Distance;
    float3 Normal;
    float3 GeometricNormal;
    uint FrontFace;
    float3 Color;
    uint MaterialIndex;
    float2 TexCoord;
    uint HasTexCoord;
    uint PrimitiveIndex;
#if defined(ARVO_SPECTRAL_TRANSPORT)
    uint VertexSpectrumBase;
    float2 Barycentric;
#endif
};
#endif
