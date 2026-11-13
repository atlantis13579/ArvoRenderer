// Metal-only shading buffer contracts; host records live in ScenePacking.h.
// PackedMaterial is 128 bytes. Optical = (slope alpha, exterior IOR, interior IOR, padding).
constant uint MaterialLegacy = 0u, MaterialDiffuse = 1u;
constant uint MaterialConductor = 2u, MaterialDielectric = 3u;
struct PackedMaterial
{
    float4 AlbedoReflectivity;
    float4 AmbientShininess;
    float4 Specular;
    float4 Emission;
    uint DiffuseTextureIndex;
    uint SampledEmitter;
    uint Model;
    uint Padding;
    float4 Optical;
    float4 ConductorIndex;
    float4 ConductorExtinction;
};
struct PackedTextureDescriptor
{
    uint OffsetTexels, Width, Height, Padding;
};
struct PackedLight
{
    packed_float3 Position;
    uint Type;
    packed_float3 Direction;
    float Intensity;
    packed_float3 Color;
    uint CastShadows;
    float Width, Height;
    uint Padding0, Padding1;
};
