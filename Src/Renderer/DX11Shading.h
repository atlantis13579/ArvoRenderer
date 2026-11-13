#pragma once
#include "Renderer/Types.h"

namespace Arvo::Renderer
{
// DX11-only extension at b1; the common raster b0 ABI stays unchanged.
struct DX11PhysicalConstants
{
    Float4 Scattering; // Model, GGX slope, exterior IOR, interior IOR.
    Float4 ConductorIndex;
    Float4 ConductorExtinction;
    Float4 EnvironmentTop;
    Float4 EnvironmentBottom;
    Float4 Subsurface; // Mean free path, scattering albedo scale, HG anisotropy.
};
inline constexpr unsigned DX11PhysicalConstantSlot = 1;
static_assert(sizeof(DX11PhysicalConstants) == 96);
DX11PhysicalConstants BuildDX11PhysicalConstants(const PipelineState& pipeline);
struct DX11MediumConstants
{
    Float4 MinimumEnabled;
    Float4 MaximumAnisotropy;
    Float4 AbsorptionBackground;
    Float4 Scattering;
};
inline constexpr unsigned DX11MediumConstantSlot = 2;
static_assert(sizeof(DX11MediumConstants) == 64);
DX11MediumConstants BuildDX11MediumConstants(const PipelineState& pipeline);
bool BuildDX11ShaderSource(std::string& source, std::string& error);
} // namespace Arvo::Renderer
