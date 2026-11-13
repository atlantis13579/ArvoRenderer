#include "Renderer/DX11Shading.h"
#include "Renderer/RealtimeShader.h"
#include "Renderer/RealtimeShaderContract.h"
#include "DX11PhysicalSource.h"
#include "DX11MediumSource.h"
#include "DX11SubsurfaceSource.h"

namespace Arvo::Renderer
{
namespace
{
constexpr const char* ShaderSourceTemplate = R"(
struct LightConstants
{
    float4 PositionType;
    float4 DirectionIntensity;
    float4 ColorCastShadows;
    float4 AreaSizePadding;
};

cbuffer DrawConstants : register(b{{DRAWCONSTANTS_BINDING}})
{
    row_major float4x4 World;
    row_major float4x4 NormalWorld;
    row_major float4x4 ViewProjection;
    row_major float4x4 LightViewProjection;
    float4 PipelineColor;
    float4 MaterialDiffuseColor;
    float4 MaterialAmbientColor;
    float4 MaterialSpecularColor;
    float4 MaterialEmissiveColor;
    float4 MaterialParameters;
    float4 AmbientColorStrength;
    float4 CameraLightCount;
    float4 RenderParameters;
    uint4 TextureFlags;
    LightConstants Lights[{{MAX_LIGHTS}}];
};

cbuffer PhysicalConstants : register(b1)
{
    float4 PhysicalScattering;
    float4 PhysicalConductorIndex;
    float4 PhysicalConductorExtinction;
    float4 PhysicalEnvironmentTop;
    float4 PhysicalEnvironmentBottom;
    float4 PhysicalSubsurface;
};
bool RasterSubsurfaceEnabled()
{
    return (uint)PhysicalScattering.x == RasterDielectric && PhysicalSubsurface.x > 0;
}

cbuffer MediumConstants : register(b2)
{
    float4 MediumMinimumEnabled;
    float4 MediumMaximumAnisotropy;
    float4 MediumAbsorptionBackground;
    float4 MediumScattering;
};
RasterMedium CurrentRasterMedium()
{
    RasterMedium medium;
    medium.Minimum = MediumMinimumEnabled.xyz;
    medium.Enabled = MediumMinimumEnabled.w > 0 ? 1 : 0;
    medium.Maximum = MediumMaximumAnisotropy.xyz;
    medium.Anisotropy = MediumMaximumAnisotropy.w;
    medium.Absorption = MediumAbsorptionBackground.xyz;
    medium.Scattering = MediumScattering.xyz;
    return medium;
}

Texture2D<float> ShadowMap : register(t{{SHADOWMAP_BINDING}});
Texture2D<float4> DiffuseMap : register(t{{DIFFUSEMAP_BINDING}});
Texture2D<float4> NormalMap : register(t{{NORMALMAP_BINDING}});
Texture2D<float4> OpacityMap : register(t{{OPACITYMAP_BINDING}});
SamplerState SurfaceSampler : register(s{{SURFACESAMPLER_BINDING}});

float3 SRGBToLinear(float3 color)
{
    float3 low = color / 12.92f;
    float3 high = pow((color + 0.055f) / 1.055f, 2.4f);
    return float3(
        color.r <= 0.04045f ? low.r : high.r,
        color.g <= 0.04045f ? low.g : high.g,
        color.b <= 0.04045f ? low.b : high.b);
}

float3 PrepareDisplayLinear(float3 color)
{
    color = max(color, 0.0f);
    color /= max(1.0f, max(color.r, max(color.g, color.b)));
    return saturate(color);
}

float3 NormalizeSafe(float3 value, float3 fallback)
{
    float lengthSquared = dot(value, value);
    return lengthSquared > 1.0e-12f ? value * rsqrt(lengthSquared) : fallback;
}

struct VSInput
{
    float3 Position : POSITION;
    float3 Normal : NORMAL;
    float4 Color : COLOR0;
    float2 TexCoord : TEXCOORD0;
};

struct VSOutput
{
    float4 Position : SV_POSITION;
    float3 WorldPosition : TEXCOORD0;
    float3 Normal : TEXCOORD1;
    float4 Color : COLOR0;
    float4 LightPosition : TEXCOORD2;
    float2 TexCoord : TEXCOORD3;
};

VSOutput {{VERTEX_ENTRY}}(VSInput input)
{
    VSOutput output;
    if (MediumAbsorptionBackground.w > 0.5f)
    {
        // Background draws supply inverse view-projection as World. Mid-depth
        // avoids numerical cancellation at a distant far clip plane.
        const float ClipReferenceDepth = 0.5f;
        const float4 world = mul(World, float4(input.Position.xy, ClipReferenceDepth, 1));
        output.Position = float4(input.Position.xy, 0, 1);
        output.WorldPosition = world.xyz / world.w;
        output.Normal = float3(0, 0, 1);
        output.Color = float4(1, 1, 1, 1);
        output.LightPosition = float4(0, 0, 0, 1);
        output.TexCoord = float2(0, 0);
        return output;
    }
    float4 worldPosition = mul(World, float4(input.Position, 1.0f));
    float3 worldNormal = NormalizeSafe(
        mul((float3x3)NormalWorld, input.Normal),
        float3(0.0f, 1.0f, 0.0f));
    output.Position = mul(ViewProjection, worldPosition);
    output.WorldPosition = worldPosition.xyz / max(abs(worldPosition.w), 1.0e-7f);
    output.Normal = worldNormal;
    output.Color = float4(SRGBToLinear(saturate(input.Color.rgb)), input.Color.a);
    output.LightPosition = mul(LightViewProjection, worldPosition);
    output.TexCoord = float2(input.TexCoord.x, 1.0f - input.TexCoord.y);
    return output;
}

float4 SampleDiffuse(float2 texCoord)
{
    return TextureFlags.x != 0
        ? DiffuseMap.Sample(SurfaceSampler, texCoord)
        : float4(1.0f, 1.0f, 1.0f, 1.0f);
}

float SampleOpacity(float2 texCoord)
{
    return TextureFlags.z != 0
        ? OpacityMap.Sample(SurfaceSampler, texCoord).r
        : 1.0f;
}

float SurfaceAlpha(VSOutput input, float4 diffuseSample)
{
    return saturate(
        input.Color.a *
        MaterialParameters.y *
        diffuseSample.a *
        SampleOpacity(input.TexCoord));
}

void ApplyAlphaClip(float alpha)
{
    if (MaterialParameters.z > 0.0f)
    {
        clip(alpha - MaterialParameters.z);
    }
}

void ApplyShadowAlpha(float alpha, float2 pixelPosition)
{
    if (MaterialParameters.z > 0.0f)
    {
        clip(alpha - MaterialParameters.z);
        return;
    }
    if (RenderParameters.w > 0.5f)
    {
        if (alpha <= 0.0f)
        {
            clip(-1.0f);
        }
        if (alpha >= 1.0f)
        {
            return;
        }
        // Stable screen-space coverage approximates transmission in a depth-only
        // shadow map while guaranteeing fully transparent texels cast no shadow.
        float threshold = frac(
            52.9829189f *
            frac(dot(floor(pixelPosition), float2(0.06711056f, 0.00583715f))));
        clip(alpha - threshold);
    }
}

float SampleShadow(
    float4 lightPosition,
    float3 surfaceNormal,
    float3 directionToLight)
{
    if (RenderParameters.x < 0.5f || abs(lightPosition.w) < 1.0e-7f)
    {
        return 1.0f;
    }
    float3 projected = lightPosition.xyz / lightPosition.w;
    float2 uv = float2(projected.x * 0.5f + 0.5f, 0.5f - projected.y * 0.5f);
    if (any(uv < 0.0f) || any(uv > 1.0f) || projected.z < 0.0f || projected.z > 1.0f)
    {
        return 1.0f;
    }
    uint width;
    uint height;
    ShadowMap.GetDimensions(width, height);
    if (width == 0 || height == 0)
    {
        return 1.0f;
    }
    float lightDirectionLengthSquared =
        dot(directionToLight, directionToLight);
    float3 normalizedLightDirection =
        lightDirectionLengthSquared > 1.0e-12f
        ? directionToLight * rsqrt(lightDirectionLengthSquared)
        : surfaceNormal;
    float normalToLight = saturate(
        dot(surfaceNormal, normalizedLightDirection));
    float sineToLight = sqrt(max(
        1.0f - normalToLight * normalToLight,
        0.0f));
    float slope = sineToLight / max(normalToLight, 0.15f);
    float receiverBias = min(
        0.0045f,
        0.00035f + 0.00060f * slope);

    float2 texelPosition =
        uv * float2(width, height) - 0.5f;
    float2 interpolation = frac(texelPosition);
    int2 basePixel = int2(floor(texelPosition));
    float occlusion = 0.0f;
    [unroll]
    for (int y = {{SHADOW_MIN_OFFSET}}; y <= {{SHADOW_MAX_OFFSET}}; ++y)
    {
        [unroll]
        for (int x = {{SHADOW_MIN_OFFSET}}; x <= {{SHADOW_MAX_OFFSET}}; ++x)
        {
            int2 pixel00 = clamp(
                basePixel + int2(x, y),
                int2(0, 0),
                int2(width - 1, height - 1));
            int2 pixel10 = clamp(
                pixel00 + int2(1, 0),
                int2(0, 0),
                int2(width - 1, height - 1));
            int2 pixel01 = clamp(
                pixel00 + int2(0, 1),
                int2(0, 0),
                int2(width - 1, height - 1));
            int2 pixel11 = clamp(
                pixel00 + int2(1, 1),
                int2(0, 0),
                int2(width - 1, height - 1));
            float comparisonDepth = projected.z - receiverBias;
            float comparison00 = comparisonDepth >
                ShadowMap.Load(int3(pixel00, 0)) ? 1.0f : 0.0f;
            float comparison10 = comparisonDepth >
                ShadowMap.Load(int3(pixel10, 0)) ? 1.0f : 0.0f;
            float comparison01 = comparisonDepth >
                ShadowMap.Load(int3(pixel01, 0)) ? 1.0f : 0.0f;
            float comparison11 = comparisonDepth >
                ShadowMap.Load(int3(pixel11, 0)) ? 1.0f : 0.0f;
            float comparisonTop = lerp(
                comparison00,
                comparison10,
                interpolation.x);
            float comparisonBottom = lerp(
                comparison01,
                comparison11,
                interpolation.x);
            occlusion += lerp(
                comparisonTop,
                comparisonBottom,
                interpolation.y);
        }
    }
    return 1.0f - {{SHADOW_STRENGTH}}f * occlusion / {{SHADOW_SAMPLE_COUNT}}.0f;
}

float3 ResolveSurfaceNormal(VSOutput input, bool isFrontFace)
{
    float3 normal = NormalizeSafe(input.Normal, float3(0.0f, 1.0f, 0.0f));
    if (!isFrontFace)
    {
        normal = -normal;
    }
    if (TextureFlags.y == 0)
    {
        return normal;
    }

    float3 positionDx = ddx(input.WorldPosition);
    float3 positionDy = ddy(input.WorldPosition);
    float2 texCoordDx = ddx(input.TexCoord);
    float2 texCoordDy = ddy(input.TexCoord);
    float determinant =
        texCoordDx.x * texCoordDy.y -
        texCoordDx.y * texCoordDy.x;
    if (abs(determinant) < 1.0e-10f)
    {
        return normal;
    }

    float inverseDeterminant = rcp(determinant);
    float3 tangent = (positionDx * texCoordDy.y -
        positionDy * texCoordDx.y) * inverseDeterminant;
    float3 sourceBitangent = (-positionDx * texCoordDy.x +
        positionDy * texCoordDx.x) * inverseDeterminant;
    tangent = tangent - normal * dot(normal, tangent);
    float3 tangentHelper = abs(normal.y) < 0.999f
        ? float3(0.0f, 1.0f, 0.0f)
        : float3(1.0f, 0.0f, 0.0f);
    float3 tangentFallback = NormalizeSafe(
        cross(tangentHelper, normal),
        float3(1.0f, 0.0f, 0.0f));
    tangent = NormalizeSafe(tangent, tangentFallback);
    float handedness = dot(cross(normal, tangent), sourceBitangent) < 0.0f
        ? -1.0f
        : 1.0f;
    float3 bitangent = NormalizeSafe(
        cross(normal, tangent) * handedness,
        float3(0.0f, 0.0f, 1.0f));
    float3 tangentNormal =
        NormalMap.Sample(SurfaceSampler, input.TexCoord).xyz * 2.0f - 1.0f;
    return NormalizeSafe(
        tangent * tangentNormal.x +
        bitangent * tangentNormal.y +
        normal * tangentNormal.z,
        normal);
}

)";
constexpr const char* ShaderLightingTemplate = R"(
RasterMediumLight RasterGetMediumLight(uint index)
{
    LightConstants packed = Lights[index];
    RasterMediumLight light;
    light.Position = packed.PositionType.xyz;
    light.Type = (uint)(packed.PositionType.w + 0.5f);
    light.Direction = packed.DirectionIntensity.xyz;
    light.Intensity = packed.DirectionIntensity.w;
    light.Color = packed.ColorCastShadows.xyz;
    light.Size = packed.AreaSizePadding.xy;
    return light;
}
float3 RasterMediumSky(float3 direction)
{
    return RasterEnvironment(direction, max(PhysicalEnvironmentTop.xyz, 0), max(PhysicalEnvironmentBottom.xyz, 0));
}
float RasterMediumVisibility(uint index, float3 position)
{
    if (RenderParameters.x < 0.5f || int(index) != int(RenderParameters.z) || Lights[index].ColorCastShadows.w < 0.5f)
        return 1;
    const float4 lightPosition = mul(LightViewProjection, float4(position, 1));
    const float ProjectionTolerance = 1.0e-7f;
    if (abs(lightPosition.w) <= ProjectionTolerance) return 1;
    const float3 projected = lightPosition.xyz / lightPosition.w;
    const float2 uv = float2(projected.x * 0.5f + 0.5f, 0.5f - projected.y * 0.5f);
    if (any(uv < 0) || any(uv > 1) || projected.z < 0 || projected.z > 1) return 1;
    uint width, height;
    ShadowMap.GetDimensions(width, height);
    if (width == 0 || height == 0) return 1;
    const float2 texel = uv * float2(width, height) - 0.5f;
    const int2 base = int2(floor(texel));
    const float2 blend = frac(texel);
    // Volume receivers have no normal/slope. Four bilinear comparison taps
    // bound the cost of every ray sample; only the selected shadow light is mapped.
    const float ReceiverBias = 0.00035f;
    float occlusion = 0;
    for (int y = 0; y < 2; ++y)
        for (int x = 0; x < 2; ++x)
        {
            const int2 pixel = clamp(base + int2(x, y), int2(0, 0), int2(width - 1, height - 1));
            const float weight = (x == 0 ? 1 - blend.x : blend.x) * (y == 0 ? 1 - blend.y : blend.y);
            if (projected.z - ReceiverBias > ShadowMap.Load(int3(pixel, 0))) occlusion += weight;
        }
    return 1 - {{SHADOW_STRENGTH}}f * occlusion;
}

RasterMaterial ResolveRasterMaterial(float3 surfaceBase, bool isFrontFace)
{
    RasterMaterial material;
    material.Model = (uint)PhysicalScattering.x;
    material.Slope = PhysicalScattering.y;
    material.IncidentIndex = isFrontFace ? PhysicalScattering.z : PhysicalScattering.w;
    material.TransmittedIndex = isFrontFace ? PhysicalScattering.w : PhysicalScattering.z;
    material.ConductorIndex = PhysicalConductorIndex.xyz;
    material.ConductorExtinction = PhysicalConductorExtinction.xyz;
    material.Reflectance = saturate(surfaceBase * MaterialDiffuseColor.rgb);
    return material;
}

// An ideal surface sees a finite rectangle as emitted radiance, not irradiance
// samples of a zero-width specular lobe. Shadow-map visibility remains approximate.
float3 DeltaRectangleRadiance(float3 origin, float3 direction, LightConstants light,
                              float3 emissionDirection, float3 tangent, float3 bitangent)
{
    const float weight = RasterRectangleRadianceWeight(origin, direction, light.PositionType.xyz,
        emissionDirection, tangent, bitangent, light.AreaSizePadding.xy);
    if (weight <= 0) return float3(0, 0, 0);
    const float distance = dot(light.PositionType.xyz - origin, emissionDirection) / dot(direction, emissionDirection);
    return max(light.ColorCastShadows.rgb, 0) * max(light.DirectionIntensity.w, 0) * weight *
        RasterMediumTransmission(CurrentRasterMedium(), origin, direction, distance);
}

void AccumulateLightSample(
    float3 lightDirection,
    float3 radiance,
    float visibility,
    float3 normal,
    float3 viewDirection,
    RasterMaterial material,
    inout float3 diffuseLight,
    inout float3 specularLight)
{
    float directionLengthSquared = dot(lightDirection, lightDirection);
    if (directionLengthSquared <= 1.0e-12f)
    {
        return;
    }
    float3 direction = lightDirection * rsqrt(directionLengthSquared);
    if (material.Model != RasterLegacy)
    {
        specularLight += radiance * visibility * abs(dot(normal, direction)) *
                         RasterEvaluateBsdf(material, normal, viewDirection, direction, RasterSubsurfaceEnabled());
        if (RasterSubsurfaceEnabled())
            specularLight += radiance * visibility * max(dot(normal, direction), 0) *
                RasterSubsurfaceBsdf(material, normal, viewDirection, direction,
                    PhysicalSubsurface.y, PhysicalSubsurface.z);
        return;
    }
    float normalToLight = max(dot(normal, direction), 0.0f);
    if (normalToLight <= 0.0f)
    {
        return;
    }
    float3 visibleRadiance = radiance * visibility;
    diffuseLight += visibleRadiance * normalToLight;
    float3 halfVector = NormalizeSafe(
        direction + viewDirection,
        normal);
    float specularAmount = pow(
        max(dot(normal, halfVector), 0.0f),
        max(MaterialParameters.x, 1.0f));
    specularLight += visibleRadiance * specularAmount;
}

void AccumulateSceneLight(
    uint lightIndex,
    VSOutput input,
    float3 normal,
    float3 viewDirection,
    RasterMaterial material,
    inout float3 diffuseLight,
    inout float3 specularLight)
{
    LightConstants light = Lights[lightIndex];
    uint type = (uint)(light.PositionType.w + 0.5f);
    float3 shadowDirection = type == 0
        ? light.DirectionIntensity.xyz
        : light.PositionType.xyz - input.WorldPosition;
    bool isShadowLight =
        int(lightIndex) == int(RenderParameters.z) &&
        light.ColorCastShadows.w > 0.5f;
    float visibility = isShadowLight
        ? SampleShadow(
            input.LightPosition,
            normal,
            shadowDirection)
        : 1.0f;
    float3 radiance =
        max(light.ColorCastShadows.rgb, 0.0f) *
        max(light.DirectionIntensity.w, 0.0f);

    if (type == 0)
    {
        radiance *= RasterMediumTransmission(CurrentRasterMedium(), input.WorldPosition,
            normalize(light.DirectionIntensity.xyz), RasterMediumRayLimit);
        AccumulateLightSample(
            light.DirectionIntensity.xyz,
            radiance,
            visibility,
            normal,
            viewDirection,
            material,
            diffuseLight,
            specularLight);
        return;
    }

    if (type == 1)
    {
        float3 toLight = light.PositionType.xyz - input.WorldPosition;
        float distanceSquared = max(dot(toLight, toLight), 1.0e-6f);
        radiance *= RasterMediumTransmission(CurrentRasterMedium(), input.WorldPosition,
            toLight * rsqrt(distanceSquared), sqrt(distanceSquared));
        AccumulateLightSample(
            toLight,
            radiance / distanceSquared,
            visibility,
            normal,
            viewDirection,
            material,
            diffuseLight,
            specularLight);
        return;
    }

    float3 emissionDirection = NormalizeSafe(
        light.DirectionIntensity.xyz,
        float3(0.0f, -1.0f, 0.0f));
    float3 helper = abs(emissionDirection.y) < 0.999f
        ? float3(0.0f, 1.0f, 0.0f)
        : float3(1.0f, 0.0f, 0.0f);
    float3 tangent = NormalizeSafe(
        cross(helper, emissionDirection),
        float3(1.0f, 0.0f, 0.0f));
    float3 bitangent = NormalizeSafe(
        cross(emissionDirection, tangent),
        float3(0.0f, 0.0f, 1.0f));
    bool matched = material.Model == RasterDielectric && material.IncidentIndex == material.TransmittedIndex;
    if ((material.Model == RasterConductor || material.Model == RasterDielectric) &&
        (material.Slope == 0 || matched))
    {
        float viewCosine = max(dot(normal, viewDirection), 0);
        float3 reflected = normal * (2 * viewCosine) - viewDirection;
        float3 fresnel = RasterFresnel(material, viewCosine);
        specularLight += fresnel * visibility * DeltaRectangleRadiance(input.WorldPosition, reflected,
            light, emissionDirection, tangent, bitangent);
        if (material.Model == RasterDielectric && !RasterSubsurfaceEnabled())
        {
            float ratio = material.IncidentIndex / material.TransmittedIndex;
            float3 transmitted = RasterRefract(viewDirection, normal, ratio);
            specularLight += (1 - fresnel) * (ratio * ratio * visibility) *
                DeltaRectangleRadiance(input.WorldPosition, transmitted, light, emissionDirection, tangent, bitangent);
        }
        if (!RasterSubsurfaceEnabled()) return;
    }
    static const int AreaLightStrata = 4;
    static const float AreaLightSampleWeight = 1.0f / (AreaLightStrata * AreaLightStrata);
    [loop]
    for (int sampleY = 0; sampleY < AreaLightStrata; ++sampleY)
    {
        [loop]
        for (int sampleX = 0; sampleX < AreaLightStrata; ++sampleX)
        {
            float offsetX = (sampleX + 0.5f) / AreaLightStrata - 0.5f;
            float offsetY = (sampleY + 0.5f) / AreaLightStrata - 0.5f;
            float3 samplePosition =
                light.PositionType.xyz +
                tangent * (offsetX * light.AreaSizePadding.x) +
                bitangent * (offsetY * light.AreaSizePadding.y);
            float3 toLight = samplePosition - input.WorldPosition;
            float distanceSquared = max(dot(toLight, toLight), 1.0e-6f);
            float3 direction = toLight * rsqrt(distanceSquared);
            float emitterCosine = max(
                dot(emissionDirection, -direction),
                0.0f);
            const float3 beam = RasterMediumTransmission(CurrentRasterMedium(), input.WorldPosition,
                direction, sqrt(distanceSquared));
            AccumulateLightSample(
                direction,
                radiance * beam * (AreaLightSampleWeight * emitterCosine / distanceSquared),
                visibility,
                normal,
                viewDirection,
                material,
                diffuseLight,
                specularLight);
        }
    }
}

float4 {{FRAGMENT_ENTRY}}(
    VSOutput input,
    bool isFrontFace : SV_IsFrontFace) : SV_TARGET
{
    if (MediumAbsorptionBackground.w > 0.5f)
    {
        const float3 direction = normalize(input.WorldPosition - CameraLightCount.xyz);
        const float3 sky = RasterIntegrateMedium(CurrentRasterMedium(), CameraLightCount.xyz, direction,
            RasterMediumRayLimit, RasterMediumSky(direction), min((uint)CameraLightCount.w, {{MAX_LIGHTS}}u));
        return float4(PrepareDisplayLinear(sky), 1);
    }
    float4 diffuseSample = SampleDiffuse(input.TexCoord);
    float alpha = SurfaceAlpha(input, diffuseSample);
    ApplyAlphaClip(alpha);
    if (RenderParameters.y > 0.5f)
    {
        return float4(
            SRGBToLinear(saturate(PipelineColor.rgb)),
            alpha);
    }

    float3 surfaceBase =
        input.Color.rgb *
        max(PipelineColor.rgb, 0.0f) *
        max(diffuseSample.rgb, 0.0f);
    float3 normal = ResolveSurfaceNormal(input, isFrontFace);
    RasterMaterial material = ResolveRasterMaterial(surfaceBase, isFrontFace);
    float3 viewDirection = NormalizeSafe(
        CameraLightCount.xyz - input.WorldPosition,
        normal);
    float3 diffuseLight = 0.0f;
    float3 specularLight = 0.0f;
    uint lightCount = min((uint)(CameraLightCount.w + 0.5f), {{MAX_LIGHTS}}u);
    [loop]
    for (uint lightIndex = 0; lightIndex < lightCount; ++lightIndex)
    {
        AccumulateSceneLight(
            lightIndex,
            input,
            normal,
            viewDirection,
            material,
            diffuseLight,
            specularLight);
    }

    float3 color =
        surfaceBase *
            max(MaterialAmbientColor.rgb, 0.0f) *
                max(AmbientColorStrength.rgb, 0.0f) *
                max(AmbientColorStrength.w, 0.0f) +
        surfaceBase *
            max(MaterialDiffuseColor.rgb, 0.0f) *
            diffuseLight +
        max(MaterialSpecularColor.rgb, 0.0f) * specularLight +
        max(MaterialEmissiveColor.rgb, 0.0f);
    if (material.Model != RasterLegacy)
        color = specularLight + max(MaterialEmissiveColor.rgb, 0) +
                RasterEnvironmentLighting(material, normal, viewDirection,
                    max(PhysicalEnvironmentTop.rgb, 0), max(PhysicalEnvironmentBottom.rgb, 0),
                    CurrentRasterMedium(), input.WorldPosition, RasterSubsurfaceEnabled());
    if (RasterSubsurfaceEnabled())
        color += RasterSubsurfaceEnvironment(material, normal, viewDirection,
            max(PhysicalEnvironmentTop.rgb, 0), max(PhysicalEnvironmentBottom.rgb, 0),
            CurrentRasterMedium(), input.WorldPosition, PhysicalSubsurface.y, PhysicalSubsurface.z);
    color = RasterIntegrateMedium(CurrentRasterMedium(), CameraLightCount.xyz, -viewDirection,
        length(input.WorldPosition - CameraLightCount.xyz), color, lightCount);
    return float4(PrepareDisplayLinear(color), alpha);
}

void {{SHADOW_ENTRY}}(VSOutput input)
{
    float4 diffuseSample = SampleDiffuse(input.TexCoord);
    ApplyShadowAlpha(SurfaceAlpha(input, diffuseSample), input.Position.xy);
}

float4 {{OUTLINE_ENTRY}}(VSOutput input) : SV_TARGET
{
    float4 diffuseSample = SampleDiffuse(input.TexCoord);
    ApplyAlphaClip(SurfaceAlpha(input, diffuseSample));
    uint radius = (uint)round(clamp(
        MaterialParameters.w,
        0.0f,
        255.0f));
    uint opacity = (uint)round(
        saturate(PipelineColor.a) * 255.0f);
    uint packedOutline = (radius << 8) | opacity;
    return float4(
        saturate(PipelineColor.rgb),
        (float)packedOutline / 65535.0f);
}
)";

} // namespace

DX11PhysicalConstants BuildDX11PhysicalConstants(const PipelineState& pipeline)
{
    const auto& material = pipeline.Scattering;
    const auto& lighting = pipeline.Lighting;
    return {
        {float(material.Model), material.MicrofacetSlope, material.ExteriorIndex, material.InteriorIndex},
        {material.ConductorIndex.X, material.ConductorIndex.Y, material.ConductorIndex.Z, 0},
        {material.ConductorExtinction.X, material.ConductorExtinction.Y, material.ConductorExtinction.Z, 0},
        {lighting.EnvironmentTop.X, lighting.EnvironmentTop.Y, lighting.EnvironmentTop.Z, 0},
        {lighting.EnvironmentBottom.X, lighting.EnvironmentBottom.Y, lighting.EnvironmentBottom.Z, 0},
        {material.SubsurfaceMeanFreePath, material.SubsurfaceAlbedoScale, material.SubsurfaceAnisotropy, 0}};
}

DX11MediumConstants BuildDX11MediumConstants(const PipelineState& pipeline)
{
    const auto& medium = pipeline.Medium;
    return {{medium.Minimum.X, medium.Minimum.Y, medium.Minimum.Z, medium.Enabled ? 1.0f : 0.0f},
            {medium.Maximum.X, medium.Maximum.Y, medium.Maximum.Z, medium.Anisotropy},
            {medium.Absorption.X, medium.Absorption.Y, medium.Absorption.Z, pipeline.MediumBackground ? 1.0f : 0.0f},
            {medium.Scattering.X, medium.Scattering.Y, medium.Scattering.Z, 0}};
}

bool BuildDX11ShaderSource(std::string& source, std::string& error)
{
    // Only embed this backend's own HLSL. Runtime use needs no source-file assets.
    const std::string combined =
        std::string(DX11MediumShaderSource) + DX11PhysicalShaderSource + DX11SubsurfaceShaderSource +
        ShaderSourceTemplate + ShaderLightingTemplate;
    return PrepareBuiltinRealtimeShader(RealtimeShaderLanguage::HLSL, combined,
                                        {RealtimeShaderStage::Vertex, RealtimeShaderStage::Fragment,
                                         RealtimeShaderStage::Shadow, RealtimeShaderStage::Outline},
                                        sizeof(RealtimeDrawConstants), MaximumSceneLights, source, error);
}
} // namespace Arvo::Renderer
