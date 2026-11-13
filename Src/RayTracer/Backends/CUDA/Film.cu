// Native CUDA film resolve: linear accumulation is encoded only once.
#include <cuda_runtime.h>
#include <cfloat>
#include "InstanceLayout.h"

namespace
{

    struct Vec3
    {
        float X;
        float Y;
        float Z;
    };

    struct __align__(16) Vec4
    {
        float X;
        float Y;
        float Z;
        float W;
    };

    struct __align__(16) PackedAccelerationNode
    {
        float MinX;
        float MinY;
        float MinZ;
        unsigned int FirstOrEscape;
        float MaxX;
        float MaxY;
        float MaxZ;
        unsigned int PrimitiveCount;
        float SplitPosition;
        unsigned int SplitAxis;
        unsigned int AboveChild;
        float SecondPlane;
    };

    struct PackedPrimitiveRef
    {
        unsigned int TypeAndIndex;
    };

    struct __align__(16) PackedTriangle
    {
        Vec4 Position0;
        Vec4 Edge1;
        Vec4 Edge2;
        Vec4 Normal0;
        Vec4 Normal1;
        Vec4 Normal2;
        Vec4 Color0;
        Vec4 Color1;
        Vec4 Color2;
        Vec4 TexCoord01;
        Vec4 TexCoord2;
        unsigned int MaterialIndex;
        unsigned int Flags;
        unsigned int Padding0;
        unsigned int Padding1;
    };

    struct __align__(16) PackedSphere
    {
        Vec4 CenterRadius;
        unsigned int MaterialIndex;
        unsigned int Padding0;
        unsigned int Padding1;
        unsigned int Padding2;
    };

#include "MaterialTypes.cuh"

    struct __align__(16) PackedTexture
    {
        unsigned int OffsetTexels;
        unsigned int Width;
        unsigned int Height;
        unsigned int Padding;
    };

    struct __align__(16) PackedLight
    {
        Vec3 Position;
        unsigned int Type;
        Vec3 Direction;
        float Intensity;
        Vec3 Color;
        unsigned int CastShadows;
        float Width;
        float Height;
        unsigned int Padding0;
        unsigned int Padding1;
    };

    static_assert(sizeof(PackedLight) == 64, "CUDA light ABI mismatch.");

    struct __align__(16) FrameConstants
    {
        Vec3 CameraPosition;
        unsigned int Width;
        Vec3 CameraForward;
        unsigned int Height;
        Vec3 CameraRight;
        unsigned int TotalSamples;
        Vec3 CameraUp;
        unsigned int MaxReflectionBounces;
        float HalfWidth;
        float HalfHeight;
        unsigned int MaterialCount;
        unsigned int LightCount;
        Vec3 BackgroundTop;
        float AmbientIntensity;
        Vec3 BackgroundBottom;
        unsigned int NodeCount;
        unsigned int PrimitiveRefCount;
        unsigned int TriangleCount;
        unsigned int SphereCount;
        float ShadowStrength;
        unsigned int PixelBegin;
        unsigned int PixelCount;
        unsigned int SampleBegin;
        unsigned int SampleCount;
        unsigned int GammaCorrection;
        unsigned int GroupCount;
        unsigned int TextureCount;
        unsigned int TextureTexelCount;
        unsigned int FullWidth;
        unsigned int FullHeight;
        unsigned int TileX;
        unsigned int TileY;
        unsigned int Integrator;
        unsigned int TransportFlags;
        unsigned int RouletteStartDepth;
        unsigned int SamplingSequence;
        unsigned int AdaptiveEnabled;
        unsigned int AdaptivePilotSamples;
        unsigned int AdaptiveMinimumSamples;
        unsigned int AdaptivePhase;
        float AdaptiveRelativeError;
        float AdaptiveAbsoluteError;
        unsigned int Acceleration;
        unsigned int InstanceNodeCount;
        const Arvo::RayTracing::CudaInstances::Record* InstanceRecords;
        const Arvo::RayTracing::CudaInstances::Node* InstanceNodes;
        unsigned int* InstanceError;
        float LensRadius;
        float FocusDistance;
        unsigned int SamplingSeed;
        unsigned int SamplingPadding[3];
    };

    static_assert(sizeof(FrameConstants) == 272 && offsetof(FrameConstants, InstanceRecords) == 224
        && offsetof(FrameConstants, InstanceError) == 240, "CUDA instance constant ABI mismatch.");

    struct __align__(16) GroupStatistics
    {
        unsigned long long PrimaryRays;
        unsigned long long ShadowRays;
        unsigned long long ReflectionRays;
        unsigned long long PrimitiveTests;
    };

    struct Ray
    {
        Vec3 Origin;
        Vec3 Direction;
        float MinimumDistance;
        float MaximumDistance;
    };

    struct Hit
    {
        float Distance;
        Vec3 Normal;
        Vec3 GeometricNormal;
        unsigned int FrontFace;
        Vec3 Color;
        unsigned int MaterialIndex;
        float TexCoordX;
        float TexCoordY;
        unsigned int HasTexCoord;
    };

    static_assert(sizeof(PackedTriangle) == 192, "CUDA triangle ABI mismatch.");
    static_assert(sizeof(PackedMaterial) == 128, "CUDA material ABI mismatch.");
    static_assert(sizeof(PackedTexture) == 16, "CUDA texture ABI mismatch.");

    __device__ float LinearToSRGB(float color)
    {
        color = fminf(1.0f, fmaxf(0.0f, color));
        return color <= 0.0031308f
            ? color * 12.92f
            : 1.055f * powf(color, 1.0f / 2.4f) - 0.055f;
    }
}

extern "C" __global__ void ResolvePixelsKernel(
    const FrameConstants* constantsPointer,
    const Vec4* accumulation,
    const unsigned int* sampleCounts,
    unsigned int* output)
{
    const FrameConstants constants = *constantsPointer;
    const unsigned int localIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (localIndex >= constants.PixelCount) return;
    const unsigned int pixelIndex = constants.PixelBegin + localIndex;
    const unsigned int sampleCount = sampleCounts[pixelIndex];
    Vec3 color = {0.0f, 0.0f, 0.0f};
    if (sampleCount != 0)
    {
        const Vec4 value = accumulation[pixelIndex];
        color = {
            fmaxf(0.0f, value.X),
            fmaxf(0.0f, value.Y),
            fmaxf(0.0f, value.Z)};
        const float displayScale = 1.0f /
            fmaxf(1.0f, fmaxf(color.X, fmaxf(color.Y, color.Z)));
        color = {
            color.X * displayScale,
            color.Y * displayScale,
            color.Z * displayScale};
        if (constants.GammaCorrection != 0)
        {
            color = {
                LinearToSRGB(color.X),
                LinearToSRGB(color.Y),
                LinearToSRGB(color.Z)};
        }
    }
    const unsigned int red = static_cast<unsigned int>(color.X * 255.0f + 0.5f);
    const unsigned int green = static_cast<unsigned int>(color.Y * 255.0f + 0.5f);
    const unsigned int blue = static_cast<unsigned int>(color.Z * 255.0f + 0.5f);
    output[pixelIndex] = red | (green << 8) | (blue << 16) | 0xff000000u;
}
