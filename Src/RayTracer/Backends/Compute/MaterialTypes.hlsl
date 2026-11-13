#ifndef ARVO_COMPUTE_MATERIAL_TYPES
#define ARVO_COMPUTE_MATERIAL_TYPES
// Data-only upload contract. All compute entry points use the same 128-byte stride.
static const uint MaterialLegacy = 0u, MaterialDiffuse = 1u;
static const uint MaterialConductor = 2u, MaterialDielectric = 3u;
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
    float4 Optical; // GGX slope, exterior index, interior index, reserved.
    float4 ConductorIndex;
    float4 ConductorExtinction;
};
#endif
