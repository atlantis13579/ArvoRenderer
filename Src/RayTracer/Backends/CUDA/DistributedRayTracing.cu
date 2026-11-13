// Cook, Porter and Carpenter (1984), "Distributed Ray Tracing".
// Stochastic area-light quadrature and normalized Phong glossy continuation.
// f=rho*(n+2)/(2*pi)*cos(alpha)^n and p=(n+1)/(2*pi)*cos(alpha)^n;
// the continuation weight is f*cos(theta)/p. Below-surface samples are zero.
// Physical materials use this backend's GGX/Fresnel scattering.
// Diffuse indirect continuation is absent. This is not network distribution.
// Dielectric interiors sample free flights and HG phase events before the next
// boundary; the surface estimator still omits diffuse indirect continuation.
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cfloat>
#include "InstanceLayout.h"
#include "BsdfTypes.cuh"
#include "Fresnel.cu"
#include "Ggx.cu"
#include "Diffuse.cu"
#include "Conductor.cu"
#include "Dielectric.cu"
#include "HomogeneousMedium.cu"
#include "SubsurfaceScattering.cu"
#include "ShutterSampling.cu"
#include "MotionInterpolation.cu"
#include "CameraMotion.cu"
#define ARVO_PARTICIPATING_MEDIA

namespace
{
    constexpr unsigned int SphereBit = 0x80000000u;
    constexpr unsigned int InvalidPrimitiveIndex = 0xffffffffu;
    constexpr unsigned int DirectionalLightType = 0u;
    constexpr unsigned int PointLightType = 1u;
    constexpr unsigned int RectAreaLightType = 2u;
    constexpr float RayEpsilon = 1.0e-4f;


    constexpr float MaximumRayDistance = 1.0e30f;
    constexpr float BasisParallelThreshold = 0.999f;
    constexpr unsigned int ThreadsPerBlock = 256;
    constexpr unsigned int AreaLightSamples = 4;
    constexpr unsigned int PixelSeedMultiplier = 9781u;
    constexpr unsigned int SampleSeedMultiplier = 6271u;
    constexpr unsigned int SecondDimensionScramble = 0x68bc21ebu;
    constexpr unsigned int AvalancheMultiplierA = 0x7feb352du;
    constexpr unsigned int AvalancheMultiplierB = 0x846ca68bu;
    constexpr unsigned int RandomMantissaMask = 0x00ffffffu;
    constexpr float RandomMantissaRange = 16777216.0f;
    constexpr float MinimumDirectionLengthSquared = 1.0e-12f;
    constexpr float MinimumTriangleDeterminant = 1.0e-8f;

    constexpr float Pi = 3.14159265358979323846f;
    constexpr unsigned int BounceSeedMultiplier = 0x9e3779b9u;
    constexpr unsigned int LightSeedMultiplier = 0x85ebca6bu;
    constexpr unsigned int LightSampleSeedMultiplier = 0xc2b2ae35u;

    constexpr unsigned int LobeSelectionScramble = 0xa511e9b3u;
    constexpr unsigned int DirectBsdfScramble = 0xd1b54a35u;
    constexpr unsigned int ImportanceBit = 1u << 0;
    constexpr unsigned int NeeBit = 1u << 1;
    constexpr unsigned int MisShift = 3, MisMask = 3, NoMis = 0, PowerMis = 2;

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
        unsigned int PrimitiveIndex;
    };

    static_assert(sizeof(PackedTriangle) == 192, "CUDA triangle ABI mismatch.");
    static_assert(sizeof(PackedMaterial) == 128, "CUDA material ABI mismatch.");
    static_assert(sizeof(PackedTexture) == 16, "CUDA texture ABI mismatch.");

    __device__ Vec3 ToVec3(const Vec4& value)
    {
        return {value.X, value.Y, value.Z};
    }

    __device__ Vec3 Add(const Vec3& left, const Vec3& right)
    {
        return {left.X + right.X, left.Y + right.Y, left.Z + right.Z};
    }

    __device__ Vec3 Subtract(const Vec3& left, const Vec3& right)
    {
        return {left.X - right.X, left.Y - right.Y, left.Z - right.Z};
    }

    __device__ Vec3 Multiply(const Vec3& value, float scalar)
    {
        return {value.X * scalar, value.Y * scalar, value.Z * scalar};
    }

    __device__ Vec3 Multiply(const Vec3& left, const Vec3& right)
    {
        return {left.X * right.X, left.Y * right.Y, left.Z * right.Z};
    }

    __device__ float SRGBToLinear(float value)
    {
        return value <= 0.04045f
            ? value / 12.92f
            : powf((value + 0.055f) / 1.055f, 2.4f);
    }

    __device__ unsigned int WrapIndex(long long value, unsigned int size)
    {
        const long long signedSize = static_cast<long long>(size);
        long long wrapped = value % signedSize;
        return static_cast<unsigned int>(
            wrapped < 0 ? wrapped + signedSize : wrapped);
    }

    __device__ Vec3 DecodeDiffuseTexel(unsigned int packed)
    {
        constexpr float InverseByte = 1.0f / 255.0f;
        return {
            SRGBToLinear(static_cast<float>(packed & 0xffu) * InverseByte),
            SRGBToLinear(static_cast<float>((packed >> 8u) & 0xffu) * InverseByte),
            SRGBToLinear(static_cast<float>((packed >> 16u) & 0xffu) * InverseByte)};
    }

    __device__ Vec3 SampleDiffuseTexture(
        const PackedTexture* textures,
        const unsigned int* textureTexels,
        const FrameConstants& constants,
        unsigned int textureIndex,
        float textureU,
        float textureV)
    {
        if (textureIndex >= constants.TextureCount
            || !isfinite(textureU) || !isfinite(textureV))
        {
            return {1.0f, 1.0f, 1.0f};
        }
        const PackedTexture descriptor = textures[textureIndex];
        const unsigned long long texelCount =
            static_cast<unsigned long long>(descriptor.Width) * descriptor.Height;
        if (descriptor.Width == 0u || descriptor.Height == 0u
            || descriptor.OffsetTexels >= constants.TextureTexelCount
            || texelCount > static_cast<unsigned long long>(
                constants.TextureTexelCount - descriptor.OffsetTexels))
        {
            return {1.0f, 1.0f, 1.0f};
        }

        const float repeatU = textureU - floorf(textureU);
        const float flippedV = 1.0f - textureV;
        const float repeatV = flippedV - floorf(flippedV);
        const float x = repeatU * static_cast<float>(descriptor.Width) - 0.5f;
        const float y = repeatV * static_cast<float>(descriptor.Height) - 0.5f;
        const float floorX = floorf(x);
        const float floorY = floorf(y);
        const long long x0 = static_cast<long long>(floorX);
        const long long y0 = static_cast<long long>(floorY);
        const float blendX = x - floorX;
        const float blendY = y - floorY;
        const unsigned int ix0 = WrapIndex(x0, descriptor.Width);
        const unsigned int ix1 = WrapIndex(x0 + 1, descriptor.Width);
        const unsigned int iy0 = WrapIndex(y0, descriptor.Height);
        const unsigned int iy1 = WrapIndex(y0 + 1, descriptor.Height);
        const unsigned int row0 = descriptor.OffsetTexels + iy0 * descriptor.Width;
        const unsigned int row1 = descriptor.OffsetTexels + iy1 * descriptor.Width;
        const Vec3 color00 = DecodeDiffuseTexel(textureTexels[row0 + ix0]);
        const Vec3 color10 = DecodeDiffuseTexel(textureTexels[row0 + ix1]);
        const Vec3 color01 = DecodeDiffuseTexel(textureTexels[row1 + ix0]);
        const Vec3 color11 = DecodeDiffuseTexel(textureTexels[row1 + ix1]);
        const Vec3 top = Add(
            Multiply(color00, 1.0f - blendX),
            Multiply(color10, blendX));
        const Vec3 bottom = Add(
            Multiply(color01, 1.0f - blendX),
            Multiply(color11, blendX));
        return Add(
            Multiply(top, 1.0f - blendY),
            Multiply(bottom, blendY));
    }

    __device__ float Dot(const Vec3& left, const Vec3& right)
    {
        return left.X * right.X + left.Y * right.Y + left.Z * right.Z;
    }

    __device__ Vec3 Cross(const Vec3& left, const Vec3& right)
    {
        return {
            left.Y * right.Z - left.Z * right.Y,
            left.Z * right.X - left.X * right.Z,
            left.X * right.Y - left.Y * right.X};
    }

    __device__ Vec3 Normalize(const Vec3& value)
    {
        const float lengthSquared = Dot(value, value);
        if (lengthSquared <= 1.0e-20f) return {0.0f, 0.0f, 0.0f};
        return Multiply(value, rsqrtf(lengthSquared));
    }

    __device__ Vec3 Reflect(const Vec3& direction, const Vec3& normal)
    {
        return Subtract(direction, Multiply(normal, 2.0f * Dot(direction, normal)));
    }

    __device__ float Component(const Vec3& value, unsigned int axis)
    {
        return axis == 0 ? value.X : (axis == 1 ? value.Y : value.Z);
    }

    __device__ unsigned int Hash(unsigned int value)
    {
        value ^= value >> 16;
        value *= AvalancheMultiplierA;
        value ^= value >> 15;
        value *= AvalancheMultiplierB;
        value ^= value >> 16;
        return value;
    }

    __device__ float Random01(unsigned int seed)
    {
        return static_cast<float>(Hash(seed) & RandomMantissaMask) / RandomMantissaRange;
    }

#include "Sequence.cu"
#include "FilmMoments.cu"
#include "Adaptive.cu"
#include "ThinLens.cu"

#include "Triangle.cu"
#include "Sphere.cu"
#include "Bvh.cu"
#include "KdTree.cu"
#include "Bih.cu"
#include "Dispatch.cu"
#include "MultipleImportance.cu"
#include "SurfaceScattering.cu"
#include "DirectLighting.cu"
#include "VolumeLighting.cu"

    __device__ MediumParameters SubsurfaceInterior(const PackedMaterial& material)
    {
        MediumParameters medium{};
        medium.Enabled = 1;
        medium.Minimum = make_float3(-MaximumRayDistance, -MaximumRayDistance, -MaximumRayDistance);
        medium.Maximum = make_float3(MaximumRayDistance, MaximumRayDistance, MaximumRayDistance);
        medium.Anisotropy = material.ConductorIndex.W;
        const float scale = material.ConductorExtinction.W;
        const float rate = 1 / material.Optical.W;
        medium.Scattering = make_float3(material.AlbedoReflectivity.X * scale * rate,
            material.AlbedoReflectivity.Y * scale * rate, material.AlbedoReflectivity.Z * scale * rate);
        medium.Absorption = make_float3((1 - material.AlbedoReflectivity.X * scale) * rate,
            (1 - material.AlbedoReflectivity.Y * scale) * rate, (1 - material.AlbedoReflectivity.Z * scale) * rate);
        return medium;
    }

    __device__ Vec3 Background(const FrameConstants& constants, const Vec3& direction)
    {
        const float blend = fminf(1.0f, fmaxf(0.0f, direction.Y * 0.5f + 0.5f));
        return Add(
            Multiply(constants.BackgroundBottom, 1.0f - blend),
            Multiply(constants.BackgroundTop, blend));
    }

    __device__ void AccumulateLightSample(
        const Vec3& lightDirection,
        const Vec3& radiance,
        float visibility,
        const Vec3& surfaceNormal,
        const Vec3& viewDirection,
        const Vec3& diffuse,
        const Vec3& specular,
        float shininess,
        Vec3& direct)
    {
        const float directionLengthSquared = Dot(lightDirection, lightDirection);
        if (!isfinite(directionLengthSquared) || directionLengthSquared <= MinimumDirectionLengthSquared)
        {
            return;
        }
        const Vec3 direction = Multiply(lightDirection, rsqrtf(directionLengthSquared));
        const float normalToLight = fmaxf(Dot(surfaceNormal, direction), 0.0f);
        if (normalToLight <= 0.0f)
        {
            return;
        }
        const Vec3 visibleRadiance = Multiply(
            radiance,
            fminf(1.0f, fmaxf(0.0f, visibility)));
        direct = Add(
            direct,
            Multiply(Multiply(diffuse, visibleRadiance), normalToLight));
        const Vec3 halfVector = Normalize(Add(direction, viewDirection));
        const float specularAmount = powf(
            fmaxf(Dot(surfaceNormal, halfVector), 0.0f),
            fmaxf(shininess, 1.0f));
        direct = Add(
            direct,
            Multiply(Multiply(specular, visibleRadiance), specularAmount));
    }

    __device__ void AccumulateFiniteLight(
        const PackedAccelerationNode* nodes,
        const PackedPrimitiveRef* primitiveRefs,
        const PackedTriangle* triangles,
        const PackedSphere* spheres,
        const FrameConstants& constants,
        const PackedLight& light,
        const Vec3& lightPosition,
        float emitterWeight,
        bool areaEmitter,
        const Vec3& position,
        const Vec3& surfaceNormal,
        const Vec3& viewDirection,
        const Vec3& diffuse,
        const Vec3& specular,
        float shininess,
        Vec3& direct,
        unsigned long long& shadowRays,
        unsigned long long& primitiveTests, float sceneTime)
    {
        if (light.Intensity <= 0.0f || emitterWeight <= 0.0f)
        {
            return;
        }
        const Vec3 toLight = Subtract(lightPosition, position);
        const float distanceSquared = Dot(toLight, toLight);
        if (!isfinite(distanceSquared)
            || distanceSquared <= RayEpsilon * RayEpsilon)
        {
            return;
        }
        const float distance = sqrtf(distanceSquared);
        const Vec3 directionToLight = Multiply(toLight, 1.0f / distance);
        const float surfaceCosine =
            fmaxf(Dot(surfaceNormal, directionToLight), 0.0f);
        if (surfaceCosine <= 0.0f)
        {
            return;
        }
        float emitterCosine = 1.0f;
        if (areaEmitter)
        {
            emitterCosine = fmaxf(
                Dot(light.Direction, Multiply(directionToLight, -1.0f)),
                0.0f);
            if (emitterCosine <= 0.0f)
            {
                return;
            }
        }

        float visibility = 1.0f;
        if (constants.ShadowStrength > 0.0f && light.CastShadows != 0u)
        {
            Ray shadowRay;
            shadowRay.Origin =
                Add(position, Multiply(surfaceNormal, RayEpsilon));
            // Aim from the displaced origin; the original segment can hit its emitter.
            const Vec3 shadowOffset = Subtract(lightPosition, shadowRay.Origin);
            const float shadowDistance = sqrtf(Dot(shadowOffset, shadowOffset));
            shadowRay.MinimumDistance = RayEpsilon;
            shadowRay.MaximumDistance = shadowDistance - RayEpsilon;
            if (shadowRay.MaximumDistance > shadowRay.MinimumDistance)
            {
                shadowRay.Direction = Multiply(shadowOffset, 1.0f / shadowDistance);
                ++shadowRays;
                Hit shadowHit;
                shadowHit.Distance = shadowRay.MaximumDistance;
                shadowHit.Normal = {0.0f, 0.0f, 0.0f};
                shadowHit.Color = {1.0f, 1.0f, 1.0f};
                shadowHit.MaterialIndex = 0u;
                shadowHit.TexCoordX = 0.0f;
                shadowHit.TexCoordY = 0.0f;
                shadowHit.HasTexCoord = 0u;
                if (IntersectScene(
                        nodes,
                        primitiveRefs,
                        triangles,
                        spheres,
                        constants,
                        shadowRay,
                        true,
                        shadowHit,
                        primitiveTests, sceneTime))
                {
                    visibility = 1.0f - constants.ShadowStrength;
                }
            }
        }
        const float attenuation =
            light.Intensity * emitterCosine * emitterWeight / distanceSquared;
        AccumulateLightSample(
            directionToLight,
            Multiply(light.Color, attenuation),
            visibility,
            surfaceNormal,
            viewDirection,
            diffuse,
            specular,
            shininess,
            direct);
    }
}

extern "C" __global__ void TracePixelsKernel(
    const PackedAccelerationNode* nodes,
    const PackedPrimitiveRef* primitiveRefs,
    const PackedTriangle* triangles,
    const PackedSphere* spheres,
    const PackedMaterial* materials,
    const PackedLight* lights,
    const PackedTexture* textures,
    const unsigned int* textureTexels,
    const FrameConstants* constantsPointer,
    Vec4* accumulation,
    unsigned int* sampleCounts,
    GroupStatistics* groupStatistics,
    unsigned int* sampleTargets, MotionExposure cameraExposure)
{
    const FrameConstants constants = *constantsPointer;
    const unsigned int localIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long primaryRays = 0;
    unsigned long long shadowRays = 0;
    unsigned long long reflectionRays = 0;
    unsigned long long primitiveTests = 0;
    if (localIndex < constants.PixelCount)
    {
        const unsigned int pixelIndex = constants.PixelBegin + localIndex;
        const unsigned int x = pixelIndex % constants.Width;
        const unsigned int y = pixelIndex / constants.Width;
        Vec4 moments = accumulation[pixelIndex];
        unsigned int completedSamples = sampleCounts[pixelIndex];
        const bool pilot = constants.AdaptiveEnabled != 0u && constants.AdaptivePhase == AdaptivePilotPhase;
        if (constants.AdaptiveEnabled == 0u && constants.SampleBegin == 0u)
            sampleTargets[pixelIndex] = constants.TotalSamples;
        const unsigned int target = pilot ? constants.AdaptivePilotSamples : sampleTargets[pixelIndex];
        for (unsigned int sampleOffset = 0; sampleOffset < constants.SampleCount; ++sampleOffset)
        {
            const unsigned int sample = constants.SampleBegin + sampleOffset;
            if (sample >= target) break;
            ++primaryRays;
            const SampleContext samples = AdaptiveSampleContext(x + constants.TileX, y + constants.TileY, sample, target, constants);
            float jitterX = 0.5f;
            float jitterY = 0.5f;
            if (samples.SampleCount > 1u || samples.Sequence != IndependentSequence || samples.SamplingSeed != 0)
            {
                const float2 pixelSample = Sample2D(samples, 0u);
                jitterX = pixelSample.x;
                jitterY = pixelSample.y;
            }
            const float u = (static_cast<float>(x + constants.TileX) + jitterX) / constants.FullWidth;
            const float v = (static_cast<float>(y + constants.TileY) + jitterY) / constants.FullHeight;
            const float screenX = (2.0f * u - 1.0f) * constants.HalfWidth;
            const float screenY = (1.0f - 2.0f * v) * constants.HalfHeight;
            Ray ray;
            ray.Origin = constants.CameraPosition;
            ray.Direction = Normalize(Add(
                Add(constants.CameraForward, Multiply(constants.CameraRight, screenX)),
                Multiply(constants.CameraUp, screenY)));
            if (constants.LensRadius > 0)
                ApplyThinLens(ray, constants, Sample2D(samples, LensSampleDimension));
            float sceneTime = 0;
            if (cameraExposure.Enabled || cameraExposure.SceneEnabled)
            {
                const float shutterSample = cameraExposure.Open == cameraExposure.Close
                    ? 0 : Sample2D(samples, TimeSampleDimension).x;
                const auto exposureRay = SampleCameraMotion(
                    make_float3(ray.Origin.X, ray.Origin.Y, ray.Origin.Z),
                    make_float3(ray.Direction.X, ray.Direction.Y, ray.Direction.Z),
                    make_float3(constants.CameraPosition.X, constants.CameraPosition.Y, constants.CameraPosition.Z),
                    cameraExposure, shutterSample);
                ray.Origin = {exposureRay.Origin.x, exposureRay.Origin.y, exposureRay.Origin.z};
                ray.Direction = {exposureRay.Direction.x, exposureRay.Direction.y, exposureRay.Direction.z};
                sceneTime = exposureRay.SceneTime;
            }
            ray.MinimumDistance = RayEpsilon;
            ray.MaximumDistance = MaximumRayDistance;
            Vec3 radiance = {0.0f, 0.0f, 0.0f};
            Vec3 throughput = {1.0f, 1.0f, 1.0f};
            bool includeRegisteredEmission = true;
            unsigned interiorMaterial = InvalidPrimitiveIndex;
            for (unsigned int bounce = 0; bounce <= constants.MaxReflectionBounces; ++bounce)
            {
                Hit hit;
                hit.Distance = ray.MaximumDistance;
                hit.Normal = {0.0f, 0.0f, 0.0f};
                hit.Color = {1.0f, 1.0f, 1.0f};
                hit.MaterialIndex = 0;
                hit.TexCoordX = 0.0f;
                hit.TexCoordY = 0.0f;
                hit.HasTexCoord = 0u;
                const bool found = IntersectScene(
                    nodes, primitiveRefs, triangles, spheres, constants,
                    ray, false, hit, primitiveTests, sceneTime);
                if (bounce == 0 && found && !hit.FrontFace && materials[hit.MaterialIndex].Optical.W > 0)
                    interiorMaterial = hit.MaterialIndex;
                const PhysicalScene scene{nodes, primitiveRefs, triangles, spheres, &constants, {}, sceneTime};
                if (interiorMaterial != InvalidPrimitiveIndex)
                {
                    constexpr unsigned FreeFlightScramble = 0x7a4d2e91u, PhaseScramble = 0x3c6ef372u;
                    const unsigned domain = (bounce + 1) * BounceSeedMultiplier;
                    const PackedMaterial interior = materials[interiorMaterial];
                    const MediumParameters medium = SubsurfaceInterior(interior);
                    const float scale = interior.ConductorExtinction.W;
                    const SubsurfaceFlight flight = SubsurfaceFreeFlight(interior.Optical.W,
                        make_float3(interior.AlbedoReflectivity.X * scale,
                            interior.AlbedoReflectivity.Y * scale, interior.AlbedoReflectivity.Z * scale),
                        make_float3(ray.Direction.X, ray.Direction.Y, ray.Direction.Z),
                        found ? hit.Distance : ray.MaximumDistance,
                        Sample2D(samples, domain ^ FreeFlightScramble).x);
                    throughput = Multiply(throughput, Vec3{flight.Weight.x, flight.Weight.y, flight.Weight.z});
                    if (throughput.X == 0 && throughput.Y == 0 && throughput.Z == 0) break;
                    if (flight.Scattered)
                    {
                        const Vec3 position = Add(ray.Origin, Multiply(ray.Direction, flight.Distance));
                        PhysicalScene volumeScene = scene;
                        volumeScene.Medium = medium;
                        radiance = Add(radiance, Multiply(throughput, VolumeDirectLighting(volumeScene, lights,
                            position, ray.Direction, samples, bounce, shadowRays, primitiveTests)));
                        if (bounce == constants.MaxReflectionBounces) break;
                        const SubsurfaceDirection phase = SubsurfaceSampleDirection(
                            make_float3(ray.Direction.X, ray.Direction.Y, ray.Direction.Z), medium.Anisotropy,
                            Sample2D(samples, domain ^ PhaseScramble), (constants.TransportFlags & ImportanceBit) != 0);
                        throughput = Multiply(throughput, phase.Weight);
                        ray = {position, {phase.Direction.x, phase.Direction.y, phase.Direction.z},
                            RayEpsilon, MaximumRayDistance};
                        includeRegisteredEmission = false;
                        ++reflectionRays;
                        continue;
                    }
                }
                if (!found)
                {
                    radiance = Add(radiance, Multiply(throughput, Background(constants, ray.Direction)));
                    break;
                }
                const PackedMaterial material = materials[
                    hit.MaterialIndex < constants.MaterialCount ? hit.MaterialIndex : 0];
                Vec3 textureColor = {1.0f, 1.0f, 1.0f};
                if (hit.HasTexCoord != 0u)
                {
                    textureColor = SampleDiffuseTexture(
                        textures,
                        textureTexels,
                        constants,
                        material.DiffuseTextureIndex,
                        hit.TexCoordX,
                        hit.TexCoordY);
                }
                const Vec3 surface = Multiply(hit.Color, textureColor);
                const Vec3 diffuse = Multiply(
                    surface,
                    Multiply(ToVec3(material.AlbedoReflectivity),
                        1.0f));
                const Vec3 position = Add(ray.Origin, Multiply(ray.Direction, hit.Distance));
                const Vec3 viewDirection = Normalize(Multiply(ray.Direction, -1.0f));
                if (material.Model != LegacyMaterial)
                {
                    const PhysicalSurface bsdf = MakePhysicalSurface(material, hit, viewDirection, surface);
                    const Vec3 emission = includeRegisteredEmission || material.SampledEmitter == 0
                        ? ToVec3(material.Emission) : Vec3{};
                    const Vec3 lighting = PhysicalDirectLighting(scene, lights, bsdf, position,
                        samples, bounce, shadowRays, primitiveTests,
                        interiorMaterial == InvalidPrimitiveIndex ? 0 : materials[interiorMaterial].Optical.W);
                    radiance = Add(radiance, Multiply(throughput, Add(emission, lighting)));
                    // Distributed tracing stops at diffuse surfaces; it estimates
                    // glossy/specular transport without adding diffuse GI or roulette.
                    if (material.Model == DiffuseMaterial || bounce == constants.MaxReflectionBounces) break;
                    const unsigned dimension = (bounce + 1) * BounceSeedMultiplier;
                    const PhysicalSample continuation = SamplePhysical(bsdf, Sample2D(samples, dimension),
                        Sample2D(samples, dimension ^ LobeSelectionScramble).x, true);
                    if (continuation.Event == 0 || continuation.Pdf <= 0) break;
                    throughput = Multiply(throughput, continuation.Weight);
                    if (throughput.X == 0 && throughput.Y == 0 && throughput.Z == 0) break;
                    ray.Origin = PhysicalRayOrigin(bsdf, position, continuation.Direction);
                    ray.Direction = continuation.Direction;
                    ray.MinimumDistance = RayEpsilon;
                    ray.MaximumDistance = MaximumRayDistance;
                    if (material.Optical.W > 0 && (continuation.Event & ArvoCuda::ScatteringTransmission) != 0)
                        interiorMaterial = hit.FrontFace ? hit.MaterialIndex : InvalidPrimitiveIndex;
                    includeRegisteredEmission = false;
                    ++reflectionRays;
                    continue;
                }
                Vec3 direct = Add(
                    ToVec3(material.Emission),
                    Multiply(
                        Multiply(surface, ToVec3(material.AmbientShininess)),
                        fmaxf(0.0f, constants.AmbientIntensity)));
                for (unsigned int lightIndex = 0; lightIndex < constants.LightCount; ++lightIndex)
                {
                    const PackedLight light = lights[lightIndex];
                    if (light.Type == DirectionalLightType)
                    {
                        const Vec3 directionToLight = light.Direction;
                        const float cosine =
                            fmaxf(0.0f, Dot(hit.Normal, directionToLight));
                        if (cosine <= 0.0f || light.Intensity <= 0.0f)
                        {
                            continue;
                        }
                        float visibility = 1.0f;
                        if (constants.ShadowStrength > 0.0f
                            && light.CastShadows != 0u)
                        {
                            ++shadowRays;
                            Ray shadowRay;
                            const Vec3 shadowNormal =
                                Dot(hit.Normal, directionToLight) >= 0.0f
                                    ? hit.Normal
                                    : Multiply(hit.Normal, -1.0f);
                            shadowRay.Origin =
                                Add(position, Multiply(shadowNormal, RayEpsilon));
                            shadowRay.Direction = directionToLight;
                            shadowRay.MinimumDistance = RayEpsilon;
                            shadowRay.MaximumDistance = MaximumRayDistance;
                            Hit shadowHit;
                            shadowHit.Distance = shadowRay.MaximumDistance;
                            shadowHit.Normal = {0.0f, 0.0f, 0.0f};
                            shadowHit.Color = {1.0f, 1.0f, 1.0f};
                            shadowHit.MaterialIndex = 0u;
                            shadowHit.TexCoordX = 0.0f;
                            shadowHit.TexCoordY = 0.0f;
                            shadowHit.HasTexCoord = 0u;
                            if (IntersectScene(
                                    nodes,
                                    primitiveRefs,
                                    triangles,
                                    spheres,
                                    constants,
                                    shadowRay,
                                    true,
                                    shadowHit,
                                    primitiveTests, sceneTime))
                            {
                                visibility = 1.0f - constants.ShadowStrength;
                            }
                        }
                        AccumulateLightSample(
                            directionToLight,
                            Multiply(light.Color, light.Intensity),
                            visibility,
                            hit.Normal,
                            viewDirection,
                            diffuse,
                            ToVec3(material.Specular),
                            material.AmbientShininess.W,
                            direct);
                    }
                    else if (light.Type == PointLightType)
                    {
                        AccumulateFiniteLight(
                            nodes,
                            primitiveRefs,
                            triangles,
                            spheres,
                            constants,
                            light,
                            light.Position,
                            1.0f,
                            false,
                            position,
                            hit.Normal,
                            viewDirection,
                            diffuse,
                            ToVec3(material.Specular),
                            material.AmbientShininess.W,
                            direct,
                            shadowRays,
                            primitiveTests, sceneTime);
                    }
                    else if (light.Type == RectAreaLightType)
                    {
                        const Vec3 emissionDirection = light.Direction;
                        const Vec3 helper =
                            fabsf(emissionDirection.Y) < BasisParallelThreshold
                            ? Vec3{0.0f, 1.0f, 0.0f}
                            : Vec3{1.0f, 0.0f, 0.0f};
                        const Vec3 tangent =
                            Normalize(Cross(helper, emissionDirection));
                        const Vec3 bitangent =
                            Normalize(Cross(emissionDirection, tangent));
#pragma unroll
                        for (unsigned int sampleIndex = 0u;
                             sampleIndex < AreaLightSamples;
                             ++sampleIndex)
                        {
                            const unsigned int lightDimension = ((bounce + 1u) * BounceSeedMultiplier)
                                ^ ((lightIndex + 1u) * LightSeedMultiplier) ^ ((sampleIndex + 1u) * LightSampleSeedMultiplier);
                            const float2 lightSample = Sample2D(samples, lightDimension);
                            const float offsetX = (static_cast<float>(sampleIndex & 1u) + lightSample.x) * 0.5f - 0.5f;
                            const float offsetY = (static_cast<float>(sampleIndex >> 1u) + lightSample.y) * 0.5f - 0.5f;

                            const Vec3 samplePosition = Add(
                                Add(
                                    light.Position,
                                    Multiply(
                                        tangent,
                                        offsetX * light.Width)),
                                Multiply(
                                    bitangent,
                                    offsetY * light.Height));
                            AccumulateFiniteLight(
                                nodes,
                                primitiveRefs,
                                triangles,
                                spheres,
                                constants,
                                light,
                                samplePosition,
                                (1.0f / AreaLightSamples),
                                true,
                                position,
                                hit.Normal,
                                viewDirection,
                                diffuse,
                                ToVec3(material.Specular),
                                material.AmbientShininess.W,
                                direct,
                                shadowRays,
                                primitiveTests, sceneTime);
                        }
                    }
                }
                if (!includeRegisteredEmission && material.SampledEmitter != 0)
                    direct = Subtract(direct, ToVec3(material.Emission));
                const float reflectivity = fminf(1.0f, fmaxf(0.0f, material.AlbedoReflectivity.W));

                radiance = Add(radiance, Multiply(Multiply(throughput, direct), 1.0f - reflectivity));
                if (reflectivity <= 0.0f || bounce == constants.MaxReflectionBounces) break;
                const float exponent = fmaxf(material.AmbientShininess.W, 1.0f);
                const unsigned int dimension = ((bounce + 1u) * BounceSeedMultiplier);
                const float2 reflectionSample = Sample2D(samples, dimension);
                const float lobeCosine = powf(reflectionSample.x, 1.0f / (exponent + 1.0f));
                const float lobeSine = sqrtf(fmaxf(0.0f, 1.0f - lobeCosine * lobeCosine));
                const float azimuth = 2.0f * Pi * reflectionSample.y;
                const Vec3 reflection = Normalize(Reflect(ray.Direction, hit.Normal));
                const Vec3 helper = fabsf(reflection.Y) < BasisParallelThreshold
                    ? Vec3{0.0f, 1.0f, 0.0f} : Vec3{1.0f, 0.0f, 0.0f};
                const Vec3 tangent = Normalize(Cross(helper, reflection));
                const Vec3 bitangent = Cross(reflection, tangent);
                const Vec3 outgoing = Normalize(Add(Add(Multiply(tangent, lobeSine * cosf(azimuth)),
                    Multiply(bitangent, lobeSine * sinf(azimuth))), Multiply(reflection, lobeCosine)));
                const float surfaceCosine = fmaxf(0.0f, Dot(hit.Normal, outgoing));
                if (surfaceCosine <= 0.0f) break;
                throughput = Multiply(throughput, reflectivity * (exponent + 2.0f) / (exponent + 1.0f) * surfaceCosine);
                ray.Origin = Add(position, Multiply(hit.Normal, RayEpsilon));
                ray.Direction = outgoing;
                includeRegisteredEmission = true;
                ++reflectionRays;
            }
            AccumulateFilmSample(moments, completedSamples, radiance);
        }
        if (pilot && completedSamples == constants.AdaptivePilotSamples)
        {
            sampleTargets[pixelIndex] = AdaptiveSampleBudget(moments, completedSamples, constants);
            moments = {0.0f, 0.0f, 0.0f, 0.0f};
            completedSamples = 0u;
        }
        accumulation[pixelIndex] = moments;
        sampleCounts[pixelIndex] = completedSamples;
    }

    __shared__ unsigned long long primaryShared[ThreadsPerBlock];
    __shared__ unsigned long long shadowShared[ThreadsPerBlock];
    __shared__ unsigned long long reflectionShared[ThreadsPerBlock];
    __shared__ unsigned long long primitiveShared[ThreadsPerBlock];
    primaryShared[threadIdx.x] = primaryRays;
    shadowShared[threadIdx.x] = shadowRays;
    reflectionShared[threadIdx.x] = reflectionRays;
    primitiveShared[threadIdx.x] = primitiveTests;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride != 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            primaryShared[threadIdx.x] += primaryShared[threadIdx.x + stride];
            shadowShared[threadIdx.x] += shadowShared[threadIdx.x + stride];
            reflectionShared[threadIdx.x] += reflectionShared[threadIdx.x + stride];
            primitiveShared[threadIdx.x] += primitiveShared[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        GroupStatistics statistics;
        statistics.PrimaryRays = primaryShared[0];
        statistics.ShadowRays = shadowShared[0];
        statistics.ReflectionRays = reflectionShared[0];
        statistics.PrimitiveTests = primitiveShared[0];
        groupStatistics[blockIdx.x] = statistics;
    }
}


#include "QueryRays.cu"
