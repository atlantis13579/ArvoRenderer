// Data-only primitive and hit layouts shared by native Metal translation units.
constant uint SphereBit = 0x80000000u;
constant uint InvalidPrimitiveIndex = 0xffffffffu;
constant float MinimumDirectionLengthSquared = 1.0e-12f;
constant float MinimumTriangleDeterminant = 1.0e-8f;

struct PackedAccelerationNode
{
    packed_float3 Minimum;
    uint FirstOrEscape;
    packed_float3 Maximum;
    uint PrimitiveCount;
    float SplitPosition;
    uint SplitAxis;
    uint AboveChild;
    float SecondPlane;
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
    uint Padding0;
    uint Padding1;
};

struct PackedSphere
{
    float4 CenterRadius;
    uint MaterialIndex;
    uint Padding0;
    uint Padding1;
    uint Padding2;
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
