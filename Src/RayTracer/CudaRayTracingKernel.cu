// Canonical source for the PTX embedded by CudaRayTracingRHI.
// Generate with: nvcc --ptx --std=c++17 -O3 --gpu-architecture=compute_75.
// The embedded copy declares PTX 9.1 because this kernel uses a 9.1-compatible subset.

#include <cuda_runtime.h>

namespace
{
    constexpr unsigned int SphereBit = 0x80000000u;
    constexpr unsigned int DirectionalLightType = 0u;
    constexpr unsigned int PointLightType = 1u;
    constexpr unsigned int RectAreaLightType = 2u;
    constexpr float RayEpsilon = 1.0e-4f;

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

    struct __align__(16) PackedBvhNode
    {
        float MinX;
        float MinY;
        float MinZ;
        unsigned int FirstOrEscape;
        float MaxX;
        float MaxY;
        float MaxZ;
        unsigned int PrimitiveCount;
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

    struct __align__(16) PackedMaterial
    {
        Vec4 AlbedoReflectivity;
        Vec4 AmbientShininess;
        Vec4 Specular;
        Vec4 Emission;
        unsigned int DiffuseTextureIndex;
        unsigned int SampledEmitter;
        unsigned int Padding1;
        unsigned int Padding2;
    };

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
        unsigned int IntegratorPadding0;
        unsigned int IntegratorPadding1;
        unsigned int IntegratorPadding2;
    };

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
        Vec3 Color;
        unsigned int MaterialIndex;
        float TexCoordX;
        float TexCoordY;
        unsigned int HasTexCoord;
    };

    static_assert(sizeof(PackedTriangle) == 192, "CUDA triangle ABI mismatch.");
    static_assert(sizeof(PackedMaterial) == 80, "CUDA material ABI mismatch.");
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
        value *= 0x7feb352du;
        value ^= value >> 15;
        value *= 0x846ca68bu;
        value ^= value >> 16;
        return value;
    }

    __device__ float Random01(unsigned int seed)
    {
        return static_cast<float>(Hash(seed) & 0x00ffffffu) / 16777216.0f;
    }

    __device__ bool IntersectBounds(
        const PackedBvhNode& node,
        const Ray& ray,
        float closestDistance)
    {
        float nearDistance = ray.MinimumDistance;
        float farDistance = fminf(ray.MaximumDistance, closestDistance);
        const Vec3 minimum = {node.MinX, node.MinY, node.MinZ};
        const Vec3 maximum = {node.MaxX, node.MaxY, node.MaxZ};
        for (unsigned int axis = 0; axis < 3; ++axis)
        {
            const float origin = Component(ray.Origin, axis);
            const float direction = Component(ray.Direction, axis);
            const float minimumValue = Component(minimum, axis);
            const float maximumValue = Component(maximum, axis);
            if (fabsf(direction) < 1.0e-12f)
            {
                if (origin < minimumValue || origin > maximumValue) return false;
                continue;
            }
            const float inverseDirection = 1.0f / direction;
            float distance0 = (minimumValue - origin) * inverseDirection;
            float distance1 = (maximumValue - origin) * inverseDirection;
            if (distance0 > distance1)
            {
                const float temporary = distance0;
                distance0 = distance1;
                distance1 = temporary;
            }
            nearDistance = fmaxf(nearDistance, distance0);
            farDistance = fminf(farDistance, distance1);
            if (nearDistance > farDistance) return false;
        }
        return true;
    }

    __device__ bool IntersectTriangle(
        const PackedTriangle& triangle,
        const Ray& ray,
        Hit& hit)
    {
        const Vec3 position0 = ToVec3(triangle.Position0);
        const Vec3 edge1 = ToVec3(triangle.Edge1);
        const Vec3 edge2 = ToVec3(triangle.Edge2);
        const Vec3 determinantVector = Cross(ray.Direction, edge2);
        const float determinant = Dot(edge1, determinantVector);
        if (fabsf(determinant) < 1.0e-8f) return false;
        const float inverseDeterminant = 1.0f / determinant;
        const Vec3 originOffset = Subtract(ray.Origin, position0);
        const float barycentric1 = Dot(originOffset, determinantVector) * inverseDeterminant;
        if (barycentric1 < 0.0f || barycentric1 > 1.0f) return false;
        const Vec3 crossValue = Cross(originOffset, edge1);
        const float barycentric2 = Dot(ray.Direction, crossValue) * inverseDeterminant;
        if (barycentric2 < 0.0f || barycentric1 + barycentric2 > 1.0f) return false;
        const float distance = Dot(edge2, crossValue) * inverseDeterminant;
        if (distance < ray.MinimumDistance || distance > ray.MaximumDistance || distance >= hit.Distance)
        {
            return false;
        }

        const float barycentric0 = 1.0f - barycentric1 - barycentric2;
        Vec3 normal = Normalize(Cross(edge1, edge2));
        if ((triangle.Flags & 1u) != 0)
        {
            const Vec3 smoothNormal = Add(
                Add(Multiply(ToVec3(triangle.Normal0), barycentric0),
                    Multiply(ToVec3(triangle.Normal1), barycentric1)),
                Multiply(ToVec3(triangle.Normal2), barycentric2));
            if (Dot(smoothNormal, smoothNormal) > 1.0e-12f) normal = Normalize(smoothNormal);
        }
        if (Dot(normal, ray.Direction) > 0.0f) normal = Multiply(normal, -1.0f);
        hit.Distance = distance;
        hit.Normal = normal;
        hit.Color = Add(
            Add(Multiply(ToVec3(triangle.Color0), barycentric0),
                Multiply(ToVec3(triangle.Color1), barycentric1)),
            Multiply(ToVec3(triangle.Color2), barycentric2));
        hit.MaterialIndex = triangle.MaterialIndex;
        hit.TexCoordX = triangle.TexCoord01.X * barycentric0
            + triangle.TexCoord01.Z * barycentric1
            + triangle.TexCoord2.X * barycentric2;
        hit.TexCoordY = triangle.TexCoord01.Y * barycentric0
            + triangle.TexCoord01.W * barycentric1
            + triangle.TexCoord2.Y * barycentric2;
        hit.HasTexCoord = 1u;
        return true;
    }

    __device__ bool IntersectSphere(const PackedSphere& sphere, const Ray& ray, Hit& hit)
    {
        const Vec3 center = ToVec3(sphere.CenterRadius);
        const float radius = sphere.CenterRadius.W;
        const Vec3 offset = Subtract(ray.Origin, center);
        const float halfB = Dot(offset, ray.Direction);
        const float c = Dot(offset, offset) - radius * radius;
        const float discriminant = halfB * halfB - c;
        if (discriminant < 0.0f) return false;
        const float root = sqrtf(discriminant);
        float distance = -halfB - root;
        if (distance < ray.MinimumDistance) distance = -halfB + root;
        if (distance < ray.MinimumDistance || distance > ray.MaximumDistance || distance >= hit.Distance)
        {
            return false;
        }
        const Vec3 position = Add(ray.Origin, Multiply(ray.Direction, distance));
        Vec3 normal = Normalize(Subtract(position, center));
        if (Dot(normal, ray.Direction) > 0.0f) normal = Multiply(normal, -1.0f);
        hit.Distance = distance;
        hit.Normal = normal;
        hit.Color = {1.0f, 1.0f, 1.0f};
        hit.MaterialIndex = sphere.MaterialIndex;
        hit.TexCoordX = 0.0f;
        hit.TexCoordY = 0.0f;
        hit.HasTexCoord = 0u;
        return true;
    }

    __device__ bool IntersectScene(
        const PackedBvhNode* nodes,
        const PackedPrimitiveRef* primitiveRefs,
        const PackedTriangle* triangles,
        const PackedSphere* spheres,
        const FrameConstants& constants,
        const Ray& ray,
        bool anyHit,
        Hit& hit,
        unsigned long long& primitiveTests)
    {
        if (constants.NodeCount == 0) return false;
        bool found = false;
        unsigned int nodeIndex = 0;
        while (nodeIndex < constants.NodeCount)
        {
            const PackedBvhNode node = nodes[nodeIndex];
            if (!IntersectBounds(node, ray, hit.Distance))
            {
                nodeIndex = node.PrimitiveCount == 0 ? node.FirstOrEscape : nodeIndex + 1;
                continue;
            }
            if (node.PrimitiveCount == 0)
            {
                ++nodeIndex;
                continue;
            }
            for (unsigned int offset = 0; offset < node.PrimitiveCount; ++offset)
            {
                const unsigned int encoded = primitiveRefs[node.FirstOrEscape + offset].TypeAndIndex;
                ++primitiveTests;
                bool primitiveHit = false;
                if ((encoded & SphereBit) != 0)
                {
                    primitiveHit = IntersectSphere(spheres[encoded & ~SphereBit], ray, hit);
                }
                else
                {
                    primitiveHit = IntersectTriangle(triangles[encoded], ray, hit);
                }
                if (primitiveHit)
                {
                    found = true;
                    if (anyHit) return true;
                }
            }
            ++nodeIndex;
        }
        return found;
    }

    __device__ Vec3 Background(const FrameConstants& constants, const Vec3& direction)
    {
        const float blend = fminf(1.0f, fmaxf(0.0f, direction.Y * 0.5f + 0.5f));
        return Add(
            Multiply(constants.BackgroundBottom, 1.0f - blend),
            Multiply(constants.BackgroundTop, blend));
    }

    __device__ float LinearToSRGB(float color)
    {
        color = fminf(1.0f, fmaxf(0.0f, color));
        return color <= 0.0031308f
            ? color * 12.92f
            : 1.055f * powf(color, 1.0f / 2.4f) - 0.055f;
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
        if (!isfinite(directionLengthSquared) || directionLengthSquared <= 1.0e-12f)
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
        const PackedBvhNode* nodes,
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
        unsigned long long& primitiveTests)
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
            shadowRay.Direction = directionToLight;
            shadowRay.MinimumDistance = RayEpsilon;
            shadowRay.MaximumDistance = distance - RayEpsilon;
            if (shadowRay.MaximumDistance > shadowRay.MinimumDistance)
            {
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
                        primitiveTests))
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
    const PackedBvhNode* nodes,
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
    GroupStatistics* groupStatistics)
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
        Vec3 accumulated = {0.0f, 0.0f, 0.0f};
        for (unsigned int sampleOffset = 0; sampleOffset < constants.SampleCount; ++sampleOffset)
        {
            const unsigned int sample = constants.SampleBegin + sampleOffset;
            ++primaryRays;
            const unsigned int pathSeed = ((y + constants.TileY) * constants.FullWidth + x + constants.TileX) * 9781u + sample * 6271u;
            float jitterX = 0.5f;
            float jitterY = 0.5f;
            if (constants.TotalSamples > 1)
            {
                const unsigned int seed = pathSeed;
                jitterX = Random01(seed);
                jitterY = Random01(seed ^ 0x68bc21ebu);
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
            ray.MinimumDistance = RayEpsilon;
            ray.MaximumDistance = 1.0e30f;
            Vec3 radiance = {0.0f, 0.0f, 0.0f};
            Vec3 throughput = {1.0f, 1.0f, 1.0f};
            bool previousDelta = true;
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
                if (!IntersectScene(
                    nodes, primitiveRefs, triangles, spheres, constants,
                    ray, false, hit, primitiveTests))
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
                        constants.Integrator != 0u ? 0.31830988618f : 1.0f));
                const Vec3 position = Add(ray.Origin, Multiply(ray.Direction, hit.Distance));
                const Vec3 viewDirection = Normalize(Multiply(ray.Direction, -1.0f));
                Vec3 direct = Add(
                    ToVec3(material.Emission),
                    Multiply(
                        Multiply(surface, ToVec3(material.AmbientShininess)),
                        (constants.Integrator != 0u ? 0.0f : fmaxf(0.0f, constants.AmbientIntensity))));
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
                            shadowRay.MaximumDistance = 1.0e30f;
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
                                    primitiveTests))
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
                            (constants.Integrator != 0u ? Vec3{0.0f, 0.0f, 0.0f} : ToVec3(material.Specular)),
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
                            (constants.Integrator != 0u ? Vec3{0.0f, 0.0f, 0.0f} : ToVec3(material.Specular)),
                            material.AmbientShininess.W,
                            direct,
                            shadowRays,
                            primitiveTests);
                    }
                    else if (light.Type == RectAreaLightType)
                    {
                        const Vec3 emissionDirection = light.Direction;
                        const Vec3 helper =
                            fabsf(emissionDirection.Y) < 0.999f
                            ? Vec3{0.0f, 1.0f, 0.0f}
                            : Vec3{1.0f, 0.0f, 0.0f};
                        const Vec3 tangent =
                            Normalize(Cross(helper, emissionDirection));
                        const Vec3 bitangent =
                            Normalize(Cross(emissionDirection, tangent));
#pragma unroll
                        for (unsigned int sampleIndex = 0u;
                             sampleIndex < 4u;
                             ++sampleIndex)
                        {
                            float offsetX = (sampleIndex & 1u) != 0u ? 0.25f : -0.25f;
                            float offsetY = (sampleIndex & 2u) != 0u ? 0.25f : -0.25f;
                            if (constants.Integrator != 0u)
                            {
                                const unsigned int lightSeed = pathSeed ^ ((bounce + 1u) * 0x9e3779b9u)
                                    ^ ((lightIndex + 1u) * 0x85ebca6bu) ^ ((sampleIndex + 1u) * 0xc2b2ae35u);
                                offsetX = (static_cast<float>(sampleIndex & 1u) + Random01(lightSeed)) * 0.5f - 0.5f;
                                offsetY = (static_cast<float>(sampleIndex >> 1u) + Random01(lightSeed ^ 0x68bc21ebu)) * 0.5f - 0.5f;
                            }
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
                                0.25f,
                                true,
                                position,
                                hit.Normal,
                                viewDirection,
                                diffuse,
                                (constants.Integrator != 0u ? Vec3{0.0f, 0.0f, 0.0f} : ToVec3(material.Specular)),
                                material.AmbientShininess.W,
                                direct,
                                shadowRays,
                                primitiveTests);
                        }
                    }
                }
                const float reflectivity = fminf(1.0f, fmaxf(0.0f, material.AlbedoReflectivity.W));
                if (constants.Integrator != 0u)
                {
                    const Vec3 visibleEmission = previousDelta || material.SampledEmitter == 0u
                        ? ToVec3(material.Emission) : Vec3{0.0f, 0.0f, 0.0f};
                    radiance = Add(radiance, Multiply(throughput,
                        Add(visibleEmission, Multiply(
                            Subtract(direct, ToVec3(material.Emission)), 1.0f - reflectivity))));
                    if (bounce == constants.MaxReflectionBounces) break;
                    const unsigned int seed = pathSeed ^ ((bounce + 1u) * 0x9e3779b9u);
                    Vec3 outgoing;
                    previousDelta = Random01(seed ^ 0xa511e9b3u) < reflectivity;
                    if (previousDelta)
                    {
                        outgoing = Normalize(Reflect(ray.Direction, hit.Normal));
                    }
                    else
                    {
                        const float radius = sqrtf(Random01(seed));
                        const float phi = 6.28318530718f * Random01(seed ^ 0x68bc21ebu);
                        const Vec3 helper = fabsf(hit.Normal.Y) < 0.999f
                            ? Vec3{0.0f, 1.0f, 0.0f} : Vec3{1.0f, 0.0f, 0.0f};
                        const Vec3 tangent = Normalize(Cross(helper, hit.Normal));
                        const Vec3 bitangent = Cross(hit.Normal, tangent);
                        outgoing = Normalize(Add(Add(
                            Multiply(tangent, radius * cosf(phi)),
                            Multiply(bitangent, radius * sinf(phi))),
                            Multiply(hit.Normal, sqrtf(fmaxf(0.0f, 1.0f - radius * radius)))));
                        const Vec3 albedo = Multiply(surface, ToVec3(material.AlbedoReflectivity));
                        throughput = Multiply(throughput, Vec3{
                            fminf(1.0f, fmaxf(0.0f, albedo.X)),
                            fminf(1.0f, fmaxf(0.0f, albedo.Y)),
                            fminf(1.0f, fmaxf(0.0f, albedo.Z))});
                    }
                    ray.Origin = Add(position, Multiply(hit.Normal, RayEpsilon));
                    ray.Direction = outgoing;
                    ++reflectionRays;
                    continue;
                }
                radiance = Add(radiance, Multiply(Multiply(throughput, direct), 1.0f - reflectivity));
                if (reflectivity <= 0.0f || bounce == constants.MaxReflectionBounces) break;
                throughput = Multiply(throughput, reflectivity);
                const Vec3 reflectedDirection =
                    Normalize(Reflect(ray.Direction, hit.Normal));
                const Vec3 reflectionNormal =
                    Dot(hit.Normal, reflectedDirection) >= 0.0f
                        ? hit.Normal
                        : Multiply(hit.Normal, -1.0f);
                ray.Origin =
                    Add(position, Multiply(reflectionNormal, RayEpsilon));
                ray.Direction = reflectedDirection;
                ++reflectionRays;
            }
            accumulated = Add(accumulated, radiance);
        }
        Vec4 value = accumulation[pixelIndex];
        value.X += accumulated.X;
        value.Y += accumulated.Y;
        value.Z += accumulated.Z;
        accumulation[pixelIndex] = value;
        sampleCounts[pixelIndex] += constants.SampleCount;
    }

    __shared__ unsigned long long primaryShared[256];
    __shared__ unsigned long long shadowShared[256];
    __shared__ unsigned long long reflectionShared[256];
    __shared__ unsigned long long primitiveShared[256];
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
        const float inverseSamples = 1.0f / sampleCount;
        color = {
            fmaxf(0.0f, value.X * inverseSamples),
            fmaxf(0.0f, value.Y * inverseSamples),
            fmaxf(0.0f, value.Z * inverseSamples)};
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
