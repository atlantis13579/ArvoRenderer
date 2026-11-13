// Native Metal film resolve; display encoding follows linear accumulation.
#include <metal_stdlib>
using namespace metal;
#include "SurfaceTypes.metal"

constant uint LightTypeDirectional = 0u;
constant uint LightTypePoint = 1u;
constant uint LightTypeRectArea = 2u;
constant float RayEpsilon = 1.0e-4f;

struct PackedMaterial
{
    float4 AlbedoReflectivity;
    float4 AmbientShininess;
    float4 Specular;
    float4 Emission;
    uint DiffuseTextureIndex;
    uint SampledEmitter;
    uint Padding1;
    uint Padding2;
};

struct PackedTextureDescriptor
{
    uint OffsetTexels;
    uint Width;
    uint Height;
    uint Padding;
};

struct PackedLight
{
    packed_float3 Position;
    uint Type;
    packed_float3 Direction;
    float Intensity;
    packed_float3 Color;
    uint CastShadows;
    float Width;
    float Height;
    uint Padding0;
    uint Padding1;
};

struct FrameConstants
{
    packed_float3 CameraPosition;
    uint Width;
    packed_float3 CameraForward;
    uint Height;
    packed_float3 CameraRight;
    uint TotalSamples;
    packed_float3 CameraUp;
    uint MaxReflectionBounces;
    float HalfWidth;
    float HalfHeight;
    uint MaterialCount;
    uint LightCount;
    packed_float3 BackgroundTop;
    float AmbientIntensity;
    packed_float3 BackgroundBottom;
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
    uint LensPadding1;
};

struct Counter64Parts
{
    uint Low;
    uint High;
};

struct GroupStatistics
{
    Counter64Parts PrimaryRays;
    Counter64Parts ShadowRays;
    Counter64Parts ReflectionRays;
    Counter64Parts PrimitiveTests;
};

float3 LinearToSRGB(float3 color)
{
    color = clamp(color, float3(0.0f), float3(1.0f));
    const float3 low = color * 12.92f;
    const float3 high = 1.055f * pow(color, float3(1.0f / 2.4f)) - 0.055f;
    return select(high, low, color <= float3(0.0031308f));
}

kernel void ArvoResolvePixels(
    device const float4* accumulation [[buffer(0)]],
    device const uint* sampleCounts [[buffer(1)]],
    device uint* outputPixels [[buffer(2)]],
    constant FrameConstants& constants [[buffer(3)]],
    uint gridIndex [[thread_position_in_grid]])
{
    if (gridIndex >= constants.PixelCount)
    {
        return;
    }
    const uint pixelIndex = constants.PixelBegin + gridIndex;
    const uint sampleCount = sampleCounts[pixelIndex];
    float3 color = float3(0.0f);
    if (sampleCount != 0u)
    {
        color = max(
            accumulation[pixelIndex].xyz,
            float3(0.0f));
        color /= max(1.0f, max(color.r, max(color.g, color.b)));
        if (constants.GammaCorrection != 0u)
        {
            color = LinearToSRGB(color);
        }
    }
    const uint3 packed = uint3(color * 255.0f + 0.5f);
    outputPixels[pixelIndex] =
        packed.x | (packed.y << 8u) | (packed.z << 16u) | 0xff000000u;
}
