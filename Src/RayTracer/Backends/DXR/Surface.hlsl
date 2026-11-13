// Native DXR surface ABI. The base 64 bytes also serve the diagnostic output;
// transport integrators only consume attributes and never own intersection code.
// Both procedural and fixed-function hits use the open ray interval (min, max).
static const uint SurfaceHitFlag = 1u;
static const uint SurfaceTexCoordFlag = 2u;
static const uint SurfaceFrontFaceFlag = 4u;
static const uint InvalidPrimitiveIndex = 0xffffffffu;
static const float MinimumNormalLengthSquared = 1.0e-12;

struct TriangleData
{
    float4 Normal0;
    float4 Normal1;
    float4 Normal2;
    float4 Color0;
    float4 Color1;
    float4 Color2;
    float4 TexCoord01;
    float4 TexCoord2;
    uint MaterialIndex;
    uint SmoothNormals;
    uint HasTexCoord;
    uint Padding;
};

struct SphereData
{
    float3 Center;
    float Radius;
    uint MaterialIndex;
    uint3 Padding;
};

struct Payload
{
    float3 Normal;
    uint DistanceBits;
    float3 Color;
    uint PrimitiveIndex;
    float2 TexCoord;
    uint MaterialIndex;
    uint Flags;
    float3 GeometricNormal;
    uint Reserved;
#if defined(ARVO_SPECTRAL_TRANSPORT)
    // Only the path library needs these attributes; public queries stay 64 bytes.
    uint VertexSpectrumBase;
    float2 Barycentric;
#endif
};

StructuredBuffer<float4> Vertices : register(t1);
StructuredBuffer<TriangleData> Triangles : register(t2);
StructuredBuffer<SphereData> Spheres : register(t7);

Payload MissPayload(float maximumDistance)
{
    Payload payload;
    payload.Normal = 0.0;
    payload.DistanceBits = asuint(maximumDistance);
    payload.Color = 1.0;
    payload.PrimitiveIndex = InvalidPrimitiveIndex;
    payload.TexCoord = 0.0;
    payload.MaterialIndex = 0;
    payload.Flags = 0;
    payload.GeometricNormal = 0.0;
    payload.Reserved = 0u;
#if defined(ARVO_SPECTRAL_TRANSPORT)
    payload.VertexSpectrumBase = InvalidPrimitiveIndex;
    payload.Barycentric = 0.0f.xx;
#endif
    return payload;
}

// The bit layout still matches the public float ray/hit ABI. Keep transported
// parameters as integers until arithmetic is needed, including across TraceRay.
struct EncodedRay
{
    uint3 Origin;
    uint Minimum;
    uint3 Direction;
    uint Maximum;
};
Payload MissEncodedPayload(uint maximum)
{
    Payload payload = MissPayload(0.0);
    payload.DistanceBits = maximum;
    return payload;
}

struct TraversalRequest
{
    EncodedRay Ray;
    Payload Hit;
    uint ValidateHardware;
    uint Valid;
};
Payload DispatchSoftwareTraversal(EncodedRay ray, bool validateHardware, out bool valid);
