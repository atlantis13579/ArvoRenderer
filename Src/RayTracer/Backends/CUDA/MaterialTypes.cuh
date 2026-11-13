#pragma once

// CUDA upload ABI only. Vec4 is the caller's aligned four-float storage type.
constexpr unsigned LegacyMaterial = 0, DiffuseMaterial = 1, ConductorMaterial = 2, DielectricMaterial = 3;
struct alignas(16) PackedMaterial
{
    Vec4 AlbedoReflectivity;
    Vec4 AmbientShininess;
    Vec4 Specular;
    Vec4 Emission;
    unsigned DiffuseTextureIndex;
    unsigned SampledEmitter;
    unsigned Model;
    unsigned Padding;
    Vec4 Optical; // GGX slope, exterior index, interior index, reserved.
    Vec4 ConductorIndex;
    Vec4 ConductorExtinction;
};
static_assert(sizeof(PackedMaterial) == 128 && offsetof(PackedMaterial, Optical) == 80 &&
                  offsetof(PackedMaterial, ConductorIndex) == 96 &&
                  offsetof(PackedMaterial, ConductorExtinction) == 112,
              "CUDA material ABI mismatch.");
