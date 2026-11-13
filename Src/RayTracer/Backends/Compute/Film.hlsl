// Film clear/resolve kernels. Keep linear accumulation until display encoding.
// Generate each entry point with dxc -T cs_6_0 -O3 -Ges.

static const uint SphereBit = 0x80000000u;
static const uint DirectionalLightType = 0u;
static const uint PointLightType = 1u;
static const uint RectAreaLightType = 2u;
static const float RayEpsilon = 1.0e-4f;

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

#include "MaterialTypes.hlsl"

struct PackedTextureDescriptor
{
    uint OffsetTexels;
    uint Width;
    uint Height;
    uint Padding;
};

struct PackedLight
{
    float3 Position;
    uint Type;
    float3 Direction;
    float Intensity;
    float3 Color;
    uint CastShadows;
    float Width;
    float Height;
    uint Padding0;
    uint Padding1;
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
    float3 Color;
    uint MaterialIndex;
    float2 TexCoord;
    uint HasTexCoord;
};

ByteAddressBuffer Nodes : register(t0);
StructuredBuffer<uint> PrimitiveRefs : register(t1);
StructuredBuffer<PackedTriangle> Triangles : register(t2);
StructuredBuffer<PackedSphere> Spheres : register(t3);
StructuredBuffer<PackedMaterial> Materials : register(t4);
StructuredBuffer<PackedLight> Lights : register(t5);
StructuredBuffer<PackedTextureDescriptor> Textures : register(t6);
StructuredBuffer<uint> TextureTexels : register(t7);
RWStructuredBuffer<float4> Accumulation : register(u0);
RWStructuredBuffer<uint> SampleCounts : register(u1);
RWStructuredBuffer<uint> OutputPixels : register(u2);
RWByteAddressBuffer Statistics : register(u3);
RWStructuredBuffer<uint> SampleTargets : register(u4);

cbuffer Globals : register(b0)
{
    float3 CameraPosition;
    uint Width;
    float3 CameraForward;
    uint Height;
    float3 CameraRight;
    uint TotalSamples;
    float3 CameraUp;
    uint MaxReflectionBounces;
    float HalfWidth;
    float HalfHeight;
    uint MaterialCount;
    uint LightCount;
    float3 BackgroundTop;
    float AmbientIntensity;
    float3 BackgroundBottom;
    uint NodeCount;
    uint PrimitiveRefCount;
    uint TriangleCount;
    uint SphereCount;
    float ShadowStrength;
    uint PixelBegin;
    uint PixelCount;
    uint SampleBegin;
    uint SampleCount;
    uint GammaCorrection;
    uint GroupCount;
    uint TextureCount;
    uint TextureTexelCount;
    uint FullWidth;
    uint FullHeight;
    uint TileX;
    uint TileY;
    uint Integrator;
    uint TransportFlags;
    uint RouletteStartDepth;
    uint SamplingSequence;
    uint AdaptiveEnabled;
    uint AdaptivePilotSamples;
    uint AdaptiveMinimumSamples;
    uint AdaptivePhase;
    float AdaptiveRelativeError;
    float AdaptiveAbsoluteError;
    uint Acceleration;
    uint InstanceNodeCount;
    float LensRadius;
    float FocusDistance;
    uint SamplingSeed;
};

float3 LinearToSRGB(float3 color)
{
    color = saturate(color);
    const float3 low = color * 12.92f;
    const float3 high = 1.055f * pow(color, 1.0f / 2.4f) - 0.055f;
    return float3(
        color.r <= 0.0031308f ? low.r : high.r,
        color.g <= 0.0031308f ? low.g : high.g,
        color.b <= 0.0031308f ? low.b : high.b);
}

[numthreads(256, 1, 1)]
void ClearPixelsKernel(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    if (dispatchThreadId.x >= PixelCount) return;
    const uint pixelIndex = PixelBegin + dispatchThreadId.x;
    Accumulation[pixelIndex] = 0.0f.xxxx;
    SampleCounts[pixelIndex] = 0;
    SampleTargets[pixelIndex] = TotalSamples;
    OutputPixels[pixelIndex] = 0xff000000u;
}

[numthreads(1, 1, 1)]
void ClearStatisticsKernel(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    Statistics.Store4(0, uint4(0, 0, 0, 0));
    Statistics.Store4(16, uint4(0, 0, 0, 0));
}

[numthreads(256, 1, 1)]
void ResolvePixelsKernel(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    if (dispatchThreadId.x >= PixelCount) return;
    const uint pixelIndex = PixelBegin + dispatchThreadId.x;
    const uint sampleCount = SampleCounts[pixelIndex];
    float3 color = 0.0f.xxx;
    if (sampleCount != 0)
    {
        color = max(Accumulation[pixelIndex].xyz, 0.0f);
        color /= max(1.0f, max(color.r, max(color.g, color.b)));
        if (GammaCorrection != 0) color = LinearToSRGB(color);
    }
    const uint3 packed = uint3(color * 255.0f + 0.5f);
    OutputPixels[pixelIndex] = packed.x | (packed.y << 8) | (packed.z << 16) | 0xff000000u;
}
