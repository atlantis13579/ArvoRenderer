#pragma once
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace Arvo::RayTracing
{
struct MicropolygonOptions
{
    bool Enabled = false;
    float TargetEdgePixels = 8;
    uint32_t MaximumLevel = 5;
    uint32_t TriangleBudget = 1u << 20;
    // A material selects entire edge-connected components, including seams.
    // The sentinel enables dicing without displacement. Texture maps require
    // authored height/bound data and remain an asset-dependent TODO.
    uint32_t DisplacementMaterial = std::numeric_limits<uint32_t>::max();
    float Amplitude = 0;
    float Wavelength = 1;
    float PhaseCycles = 0;
    bool operator==(const MicropolygonOptions& b) const
    {
        return Enabled == b.Enabled && TargetEdgePixels == b.TargetEdgePixels &&
               MaximumLevel == b.MaximumLevel && TriangleBudget == b.TriangleBudget &&
               DisplacementMaterial == b.DisplacementMaterial && Amplitude == b.Amplitude &&
               Wavelength == b.Wavelength && PhaseCycles == b.PhaseCycles;
    }
    bool operator!=(const MicropolygonOptions& b) const
    {
        return !(*this == b);
    }
};
// A coordinator shares one immutable plan across tiles/backends. Fingerprint
// identifies packed input geometry, full-film camera and dicing controls;
// it deliberately excludes crop bounds and the executing API.
struct MicropolygonPlan
{
    uint64_t Fingerprint = 0;
    std::vector<uint32_t> Components;
};
bool ValidateMicropolygonOptions(const MicropolygonOptions&, std::string& error);
} // namespace Arvo::RayTracing
