#ifndef ARVO_DXR_MATERIAL_TYPES
#define ARVO_DXR_MATERIAL_TYPES
// DXR upload contract: 128 bytes, including RGB complex indices for conductors.
// Optical stores slope alpha, exterior index, interior index and padding.
static const uint MaterialLegacy = 0u, MaterialDiffuse = 1u;
static const uint MaterialConductor = 2u, MaterialDielectric = 3u;
struct MaterialData
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
#endif
