// Canonical source for the DXIL embedded by ComputeShaderRayTracingRHI.
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

struct PackedMaterial
{
    float4 AlbedoReflectivity;
    float4 AmbientShininess;
    float4 Specular;
    float4 Emission;
    uint DiffuseTextureIndex;
    uint SampledEmitter;
    uint2 Padding;
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

StructuredBuffer<PackedBvhNode> Nodes : register(t0);
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
    uint IntegratorPadding0;
    uint IntegratorPadding1;
    uint IntegratorPadding2;
};

uint Hash(uint value)
{
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    value ^= value >> 16;
    return value;
}

float Random01(uint seed)
{
    return float(Hash(seed) & 0x00ffffffu) / 16777216.0f;
}

float SRGBToLinear(float value)
{
    return value <= 0.04045f
        ? value / 12.92f
        : pow((value + 0.055f) / 1.055f, 2.4f);
}

int WrapTextureCoordinate(int value, uint size)
{
    const int signedSize = int(size);
    const int wrapped = value % signedSize;
    return wrapped < 0 ? wrapped + signedSize : wrapped;
}

float3 LoadDiffuseTexel(PackedTextureDescriptor descriptor, int x, int y)
{
    const uint offset = descriptor.OffsetTexels
        + uint(WrapTextureCoordinate(y, descriptor.Height)) * descriptor.Width
        + uint(WrapTextureCoordinate(x, descriptor.Width));
    const uint packed = TextureTexels[offset];
    return float3(
        SRGBToLinear(float(packed & 0xffu) / 255.0f),
        SRGBToLinear(float((packed >> 8u) & 0xffu) / 255.0f),
        SRGBToLinear(float((packed >> 16u) & 0xffu) / 255.0f));
}

float3 SampleDiffuse(uint textureIndex, float2 texCoord, uint hasTexCoord)
{
    if (hasTexCoord == 0u || textureIndex >= TextureCount)
    {
        return 1.0f.xxx;
    }
    const PackedTextureDescriptor descriptor = Textures[textureIndex];
    if (descriptor.Width == 0u || descriptor.Height == 0u)
    {
        return 1.0f.xxx;
    }
    if (descriptor.Height > TextureTexelCount / descriptor.Width)
    {
        return 1.0f.xxx;
    }
    const uint pixelCount = descriptor.Width * descriptor.Height;
    if (descriptor.OffsetTexels > TextureTexelCount
        || pixelCount > TextureTexelCount - descriptor.OffsetTexels)
    {
        return 1.0f.xxx;
    }
    const float x = frac(texCoord.x) * float(descriptor.Width) - 0.5f;
    const float y = frac(1.0f - texCoord.y) * float(descriptor.Height) - 0.5f;
    const int x0 = int(floor(x));
    const int y0 = int(floor(y));
    const float amountX = x - floor(x);
    const float amountY = y - floor(y);
    const float3 top = lerp(
        LoadDiffuseTexel(descriptor, x0, y0),
        LoadDiffuseTexel(descriptor, x0 + 1, y0),
        amountX);
    const float3 bottom = lerp(
        LoadDiffuseTexel(descriptor, x0, y0 + 1),
        LoadDiffuseTexel(descriptor, x0 + 1, y0 + 1),
        amountX);
    return lerp(top, bottom, amountY);
}

void IncrementCounter(inout uint2 counter)
{
    const uint previousLow = counter.x;
    counter.x += 1u;
    if (counter.x < previousLow) counter.y += 1u;
}

uint2 AddCounter(uint2 left, uint2 right)
{
    const uint low = left.x + right.x;
    return uint2(low, left.y + right.y + (low < left.x ? 1u : 0u));
}

void AtomicAddCounter(uint byteOffset, uint2 value)
{
    uint previousLow;
    Statistics.InterlockedAdd(byteOffset, value.x, previousLow);
    const uint carry = previousLow + value.x < previousLow ? 1u : 0u;
    uint ignored;
    Statistics.InterlockedAdd(byteOffset + 4u, value.y + carry, ignored);
}

bool IntersectBounds(PackedBvhNode node, RayData ray, float closestDistance)
{
    float nearDistance = ray.MinimumDistance;
    float farDistance = min(ray.MaximumDistance, closestDistance);
    [unroll]
    for (uint axis = 0; axis < 3; ++axis)
    {
        const float origin = ray.Origin[axis];
        const float direction = ray.Direction[axis];
        if (abs(direction) < 1.0e-12f)
        {
            if (origin < node.Minimum[axis] || origin > node.Maximum[axis]) return false;
            continue;
        }
        const float inverseDirection = 1.0f / direction;
        float distance0 = (node.Minimum[axis] - origin) * inverseDirection;
        float distance1 = (node.Maximum[axis] - origin) * inverseDirection;
        if (distance0 > distance1)
        {
            const float temporary = distance0;
            distance0 = distance1;
            distance1 = temporary;
        }
        nearDistance = max(nearDistance, distance0);
        farDistance = min(farDistance, distance1);
        if (nearDistance > farDistance) return false;
    }
    return true;
}

bool IntersectTriangle(PackedTriangle primitive, RayData ray, inout HitData hit)
{
    const float3 position0 = primitive.Position0.xyz;
    const float3 edge1 = primitive.Edge1.xyz;
    const float3 edge2 = primitive.Edge2.xyz;
    const float3 determinantVector = cross(ray.Direction, edge2);
    const float determinant = dot(edge1, determinantVector);
    if (abs(determinant) < 1.0e-8f) return false;
    const float inverseDeterminant = 1.0f / determinant;
    const float3 originOffset = ray.Origin - position0;
    const float barycentric1 = dot(originOffset, determinantVector) * inverseDeterminant;
    if (barycentric1 < 0.0f || barycentric1 > 1.0f) return false;
    const float3 crossValue = cross(originOffset, edge1);
    const float barycentric2 = dot(ray.Direction, crossValue) * inverseDeterminant;
    if (barycentric2 < 0.0f || barycentric1 + barycentric2 > 1.0f) return false;
    const float distance = dot(edge2, crossValue) * inverseDeterminant;
    if (distance < ray.MinimumDistance || distance > ray.MaximumDistance || distance >= hit.Distance)
    {
        return false;
    }

    const float barycentric0 = 1.0f - barycentric1 - barycentric2;
    float3 normal = normalize(cross(edge1, edge2));
    if ((primitive.Flags & 1u) != 0)
    {
        const float3 smoothNormal = primitive.Normal0.xyz * barycentric0
            + primitive.Normal1.xyz * barycentric1 + primitive.Normal2.xyz * barycentric2;
        if (dot(smoothNormal, smoothNormal) > 1.0e-12f) normal = normalize(smoothNormal);
    }
    if (dot(normal, ray.Direction) > 0.0f) normal = -normal;
    hit.Distance = distance;
    hit.Normal = normal;
    hit.Color = primitive.Color0.xyz * barycentric0
        + primitive.Color1.xyz * barycentric1
        + primitive.Color2.xyz * barycentric2;
    hit.MaterialIndex = primitive.MaterialIndex;
    hit.TexCoord = primitive.TexCoord01.xy * barycentric0
        + primitive.TexCoord01.zw * barycentric1
        + primitive.TexCoord2.xy * barycentric2;
    hit.HasTexCoord = 1u;
    return true;
}

bool IntersectSphere(PackedSphere sphere, RayData ray, inout HitData hit)
{
    const float3 offset = ray.Origin - sphere.CenterRadius.xyz;
    const float halfB = dot(offset, ray.Direction);
    const float c = dot(offset, offset) - sphere.CenterRadius.w * sphere.CenterRadius.w;
    const float discriminant = halfB * halfB - c;
    if (discriminant < 0.0f) return false;
    const float root = sqrt(discriminant);
    float distance = -halfB - root;
    if (distance < ray.MinimumDistance) distance = -halfB + root;
    if (distance < ray.MinimumDistance || distance > ray.MaximumDistance || distance >= hit.Distance)
    {
        return false;
    }
    const float3 position = ray.Origin + ray.Direction * distance;
    float3 normal = normalize(position - sphere.CenterRadius.xyz);
    if (dot(normal, ray.Direction) > 0.0f) normal = -normal;
    hit.Distance = distance;
    hit.Normal = normal;
    hit.Color = 1.0f.xxx;
    hit.MaterialIndex = sphere.MaterialIndex;
    hit.TexCoord = 0.0f.xx;
    hit.HasTexCoord = 0u;
    return true;
}

bool IntersectScene(RayData ray, bool anyHit, inout HitData hit, inout uint2 primitiveTests)
{
    if (NodeCount == 0) return false;
    bool found = false;
    uint nodeIndex = 0;
    [loop]
    while (nodeIndex < NodeCount)
    {
        const PackedBvhNode node = Nodes[nodeIndex];
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
        [loop]
        for (uint offset = 0; offset < node.PrimitiveCount; ++offset)
        {
            const uint encoded = PrimitiveRefs[node.FirstOrEscape + offset];
            IncrementCounter(primitiveTests);
            bool primitiveHit = false;
            if ((encoded & SphereBit) != 0)
            {
                primitiveHit = IntersectSphere(Spheres[encoded & ~SphereBit], ray, hit);
            }
            else
            {
                primitiveHit = IntersectTriangle(Triangles[encoded], ray, hit);
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

float3 Background(float3 direction)
{
    const float blend = saturate(direction.y * 0.5f + 0.5f);
    return lerp(BackgroundBottom, BackgroundTop, blend);
}

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

void AccumulateLightSample(
    float3 lightDirection,
    float3 radiance,
    float visibility,
    float3 surfaceNormal,
    float3 viewDirection,
    float3 diffuse,
    float3 specular,
    float shininess,
    inout float3 direct)
{
    const float directionLengthSquared = dot(lightDirection, lightDirection);
    if (!isfinite(directionLengthSquared) || directionLengthSquared <= 1.0e-12f)
    {
        return;
    }
    const float3 direction = lightDirection * rsqrt(directionLengthSquared);
    const float normalToLight = max(dot(surfaceNormal, direction), 0.0f);
    if (normalToLight <= 0.0f)
    {
        return;
    }
    const float3 visibleRadiance = radiance * saturate(visibility);
    direct += diffuse * visibleRadiance * normalToLight
        * (Integrator != 0u ? 0.31830988618f : 1.0f);
    const float3 halfVector = normalize(direction + viewDirection);
    const float specularAmount = pow(
        max(dot(surfaceNormal, halfVector), 0.0f),
        max(shininess, 1.0f));
    direct += specular * visibleRadiance * specularAmount * (Integrator != 0u ? 0.0f : 1.0f);
}

void AccumulateFiniteLight(
    PackedLight light,
    float3 lightPosition,
    float emitterWeight,
    bool areaEmitter,
    float3 position,
    float3 surfaceNormal,
    float3 viewDirection,
    float3 diffuse,
    float3 specular,
    float shininess,
    inout float3 direct,
    inout uint2 shadowRays,
    inout uint2 primitiveTests)
{
    if (light.Intensity <= 0.0f || emitterWeight <= 0.0f)
    {
        return;
    }
    const float3 toLight = lightPosition - position;
    const float distanceSquared = dot(toLight, toLight);
    if (!isfinite(distanceSquared)
        || distanceSquared <= RayEpsilon * RayEpsilon)
    {
        return;
    }
    const float distance = sqrt(distanceSquared);
    const float3 directionToLight = toLight / distance;
    const float surfaceCosine =
        max(dot(surfaceNormal, directionToLight), 0.0f);
    if (surfaceCosine <= 0.0f)
    {
        return;
    }
    float emitterCosine = 1.0f;
    if (areaEmitter)
    {
        emitterCosine =
            max(dot(light.Direction, -directionToLight), 0.0f);
        if (emitterCosine <= 0.0f)
        {
            return;
        }
    }

    float visibility = 1.0f;
    if (ShadowStrength > 0.0f && light.CastShadows != 0u)
    {
        RayData shadowRay;
        shadowRay.Origin = position + surfaceNormal * RayEpsilon;
        shadowRay.Direction = directionToLight;
        shadowRay.MinimumDistance = RayEpsilon;
        shadowRay.MaximumDistance = distance - RayEpsilon;
        if (shadowRay.MaximumDistance > shadowRay.MinimumDistance)
        {
            IncrementCounter(shadowRays);
            HitData shadowHit;
            shadowHit.Distance = shadowRay.MaximumDistance;
            shadowHit.Normal = 0.0f.xxx;
            shadowHit.Color = 1.0f.xxx;
            shadowHit.MaterialIndex = 0u;
            if (IntersectScene(
                    shadowRay,
                    true,
                    shadowHit,
                    primitiveTests))
            {
                visibility = 1.0f - ShadowStrength;
            }
        }
    }
    const float attenuation =
        light.Intensity * emitterCosine * emitterWeight / distanceSquared;
    AccumulateLightSample(
        directionToLight,
        light.Color * attenuation,
        visibility,
        surfaceNormal,
        viewDirection,
        diffuse,
        specular,
        shininess,
        direct);
}

[numthreads(256, 1, 1)]
void ClearPixelsKernel(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    if (dispatchThreadId.x >= PixelCount) return;
    const uint pixelIndex = PixelBegin + dispatchThreadId.x;
    Accumulation[pixelIndex] = 0.0f.xxxx;
    SampleCounts[pixelIndex] = 0;
    OutputPixels[pixelIndex] = 0xff000000u;
}

[numthreads(1, 1, 1)]
void ClearStatisticsKernel(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    Statistics.Store4(0, uint4(0, 0, 0, 0));
    Statistics.Store4(16, uint4(0, 0, 0, 0));
}

groupshared uint2 PrimaryShared[256];
groupshared uint2 ShadowShared[256];
groupshared uint2 ReflectionShared[256];
groupshared uint2 PrimitiveShared[256];

[numthreads(256, 1, 1)]
void TracePixelsKernel(
    uint3 dispatchThreadId : SV_DispatchThreadID,
    uint groupIndex : SV_GroupIndex)
{
    uint2 primaryRays = uint2(0, 0);
    uint2 shadowRays = uint2(0, 0);
    uint2 reflectionRays = uint2(0, 0);
    uint2 primitiveTests = uint2(0, 0);
    if (dispatchThreadId.x < PixelCount)
    {
        const uint pixelIndex = PixelBegin + dispatchThreadId.x;
        const uint x = pixelIndex % Width;
        const uint y = pixelIndex / Width;
        float3 accumulated = 0.0f.xxx;
        [loop]
        for (uint sampleOffset = 0; sampleOffset < SampleCount; ++sampleOffset)
        {
            const uint sample = SampleBegin + sampleOffset;
            IncrementCounter(primaryRays);
            const uint pathSeed = ((y + TileY) * FullWidth + x + TileX) * 9781u + sample * 6271u;
            float jitterX = 0.5f;
            float jitterY = 0.5f;
            if (TotalSamples > 1)
            {
                const uint seed = pathSeed;
                jitterX = Random01(seed);
                jitterY = Random01(seed ^ 0x68bc21ebu);
            }
            const float u = (float(x + TileX) + jitterX) / float(FullWidth);
            const float v = (float(y + TileY) + jitterY) / float(FullHeight);
            const float screenX = (2.0f * u - 1.0f) * HalfWidth;
            const float screenY = (1.0f - 2.0f * v) * HalfHeight;
            RayData ray;
            ray.Origin = CameraPosition;
            ray.Direction = normalize(CameraForward + CameraRight * screenX + CameraUp * screenY);
            ray.MinimumDistance = RayEpsilon;
            ray.MaximumDistance = 1.0e30f;
            float3 radiance = 0.0f.xxx;
            float3 throughput = 1.0f.xxx;
            bool previousDelta = true;
            [loop]
            for (uint bounce = 0; bounce <= MaxReflectionBounces; ++bounce)
            {
                HitData hit;
                hit.Distance = ray.MaximumDistance;
                hit.Normal = 0.0f.xxx;
                hit.Color = 1.0f.xxx;
                hit.MaterialIndex = 0;
                hit.TexCoord = 0.0f.xx;
                hit.HasTexCoord = 0u;
                if (!IntersectScene(ray, false, hit, primitiveTests))
                {
                    radiance += throughput * Background(ray.Direction);
                    break;
                }
                const PackedMaterial material = Materials[
                    hit.MaterialIndex < MaterialCount ? hit.MaterialIndex : 0];
                const float3 surface = hit.Color * SampleDiffuse(
                        material.DiffuseTextureIndex,
                        hit.TexCoord,
                        hit.HasTexCoord);
                const float3 diffuse = surface * material.AlbedoReflectivity.xyz;
                const float3 position = ray.Origin + ray.Direction * hit.Distance;
                const float3 viewDirection = normalize(-ray.Direction);
                float3 direct = material.Emission.xyz
                    + surface * material.AmbientShininess.xyz
                        * (Integrator != 0u ? 0.0f : max(0.0f, AmbientIntensity));
                [loop]
                for (uint lightIndex = 0; lightIndex < LightCount; ++lightIndex)
                {
                    const PackedLight light = Lights[lightIndex];
                    if (light.Type == DirectionalLightType)
                    {
                        const float3 directionToLight = light.Direction;
                        const float cosine =
                            max(0.0f, dot(hit.Normal, directionToLight));
                        if (cosine <= 0.0f || light.Intensity <= 0.0f)
                        {
                            continue;
                        }
                        float visibility = 1.0f;
                        if (ShadowStrength > 0.0f && light.CastShadows != 0u)
                        {
                            IncrementCounter(shadowRays);
                            RayData shadowRay;
                            const float3 shadowNormal =
                                dot(hit.Normal, directionToLight) >= 0.0f
                                    ? hit.Normal
                                    : -hit.Normal;
                            shadowRay.Origin =
                                position + shadowNormal * RayEpsilon;
                            shadowRay.Direction = directionToLight;
                            shadowRay.MinimumDistance = RayEpsilon;
                            shadowRay.MaximumDistance = 1.0e30f;
                            HitData shadowHit;
                            shadowHit.Distance = shadowRay.MaximumDistance;
                            shadowHit.Normal = 0.0f.xxx;
                            shadowHit.Color = 1.0f.xxx;
                            shadowHit.MaterialIndex = 0u;
                            if (IntersectScene(
                                    shadowRay,
                                    true,
                                    shadowHit,
                                    primitiveTests))
                            {
                                visibility = 1.0f - ShadowStrength;
                            }
                        }
                        AccumulateLightSample(
                            directionToLight,
                            light.Color * light.Intensity,
                            visibility,
                            hit.Normal,
                            viewDirection,
                            diffuse,
                            material.Specular.xyz,
                            material.AmbientShininess.w,
                            direct);
                    }
                    else if (light.Type == PointLightType)
                    {
                        AccumulateFiniteLight(
                            light,
                            light.Position,
                            1.0f,
                            false,
                            position,
                            hit.Normal,
                            viewDirection,
                            diffuse,
                            material.Specular.xyz,
                            material.AmbientShininess.w,
                            direct,
                            shadowRays,
                            primitiveTests);
                    }
                    else if (light.Type == RectAreaLightType)
                    {
                        const float3 emissionDirection = light.Direction;
                        const float3 helper =
                            abs(emissionDirection.y) < 0.999f
                            ? float3(0.0f, 1.0f, 0.0f)
                            : float3(1.0f, 0.0f, 0.0f);
                        const float3 tangent =
                            normalize(cross(helper, emissionDirection));
                        const float3 bitangent =
                            normalize(cross(emissionDirection, tangent));
                        [unroll]
                        for (uint sampleIndex = 0u; sampleIndex < 4u; ++sampleIndex)
                        {
                            float offsetX = (sampleIndex & 1u) != 0u ? 0.25f : -0.25f;
                            float offsetY = (sampleIndex & 2u) != 0u ? 0.25f : -0.25f;
                            if (Integrator != 0u)
                            {
                                const uint lightSeed = pathSeed ^ ((bounce + 1u) * 0x9e3779b9u)
                                    ^ ((lightIndex + 1u) * 0x85ebca6bu) ^ ((sampleIndex + 1u) * 0xc2b2ae35u);
                                offsetX = (float(sampleIndex & 1u) + Random01(lightSeed)) * 0.5f - 0.5f;
                                offsetY = (float(sampleIndex >> 1u) + Random01(lightSeed ^ 0x68bc21ebu)) * 0.5f - 0.5f;
                            }
                            const float3 samplePosition =
                                light.Position
                                + tangent * (offsetX * light.Width)
                                + bitangent * (offsetY * light.Height);
                            AccumulateFiniteLight(
                                light,
                                samplePosition,
                                0.25f,
                                true,
                                position,
                                hit.Normal,
                                viewDirection,
                                diffuse,
                                material.Specular.xyz,
                                material.AmbientShininess.w,
                                direct,
                                shadowRays,
                                primitiveTests);
                        }
                    }
                }
                const float reflectivity = saturate(material.AlbedoReflectivity.w);
                if (Integrator != 0u)
                {
                    radiance += throughput * ((previousDelta || material.SampledEmitter == 0u
                        ? material.Emission.xyz : float3(0.0f, 0.0f, 0.0f))
                        + (direct - material.Emission.xyz) * (1.0f - reflectivity));
                    if (bounce == MaxReflectionBounces) break;
                    const uint seed = pathSeed ^ ((bounce + 1u) * 0x9e3779b9u);
                    float3 outgoing;
                    previousDelta = Random01(seed ^ 0xa511e9b3u) < reflectivity;
                    if (previousDelta)
                    {
                        outgoing = normalize(reflect(ray.Direction, hit.Normal));
                    }
                    else
                    {
                        const float radius = sqrt(Random01(seed));
                        const float phi = 6.28318530718f * Random01(seed ^ 0x68bc21ebu);
                        const float3 helper = abs(hit.Normal.y) < 0.999f
                            ? float3(0.0f, 1.0f, 0.0f) : float3(1.0f, 0.0f, 0.0f);
                        const float3 tangent = normalize(cross(helper, hit.Normal));
                        const float3 bitangent = cross(hit.Normal, tangent);
                        outgoing = normalize(tangent * (radius * cos(phi))
                            + bitangent * (radius * sin(phi))
                            + hit.Normal * sqrt(max(0.0f, 1.0f - radius * radius)));
                        throughput *= clamp(diffuse, 0.0f, 1.0f);
                    }
                    ray.Origin = position + hit.Normal * RayEpsilon;
                    ray.Direction = outgoing;
                    IncrementCounter(reflectionRays);
                    continue;
                }
                radiance += throughput * direct * (1.0f - reflectivity);
                if (reflectivity <= 0.0f || bounce == MaxReflectionBounces) break;
                throughput *= reflectivity;
                const float3 reflectedDirection =
                    normalize(reflect(ray.Direction, hit.Normal));
                const float3 reflectionNormal =
                    dot(hit.Normal, reflectedDirection) >= 0.0f
                        ? hit.Normal
                        : -hit.Normal;
                ray.Origin = position + reflectionNormal * RayEpsilon;
                ray.Direction = reflectedDirection;
                IncrementCounter(reflectionRays);
            }
            accumulated += radiance;
        }
        Accumulation[pixelIndex].xyz += accumulated;
        SampleCounts[pixelIndex] += SampleCount;
    }

    PrimaryShared[groupIndex] = primaryRays;
    ShadowShared[groupIndex] = shadowRays;
    ReflectionShared[groupIndex] = reflectionRays;
    PrimitiveShared[groupIndex] = primitiveTests;
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (uint stride = 128; stride != 0; stride >>= 1)
    {
        if (groupIndex < stride)
        {
            PrimaryShared[groupIndex] = AddCounter(
                PrimaryShared[groupIndex], PrimaryShared[groupIndex + stride]);
            ShadowShared[groupIndex] = AddCounter(
                ShadowShared[groupIndex], ShadowShared[groupIndex + stride]);
            ReflectionShared[groupIndex] = AddCounter(
                ReflectionShared[groupIndex], ReflectionShared[groupIndex + stride]);
            PrimitiveShared[groupIndex] = AddCounter(
                PrimitiveShared[groupIndex], PrimitiveShared[groupIndex + stride]);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (groupIndex == 0)
    {
        AtomicAddCounter(0, PrimaryShared[0]);
        AtomicAddCounter(8, ShadowShared[0]);
        AtomicAddCounter(16, ReflectionShared[0]);
        AtomicAddCounter(24, PrimitiveShared[0]);
    }
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
        color = max(Accumulation[pixelIndex].xyz / float(sampleCount), 0.0f);
        color /= max(1.0f, max(color.r, max(color.g, color.b)));
        if (GammaCorrection != 0) color = LinearToSRGB(color);
    }
    const uint3 packed = uint3(color * 255.0f + 0.5f);
    OutputPixels[pixelIndex] = packed.x | (packed.y << 8) | (packed.z << 16) | 0xff000000u;
}
