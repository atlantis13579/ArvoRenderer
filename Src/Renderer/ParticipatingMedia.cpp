#include "Renderer/ParticipatingMedia.h"
#include <cmath>

namespace Arvo::RayTracing
{
// PBRT v3, sections 11.3 and 15.2 ("Sampling Volume Scattering"):
// https://pbr-book.org/3ed-2018/Light_Transport_II_Volume_Rendering/Sampling_Volume_Scattering
// A channel-mixture exponential samples either a volume event or the endpoint.
// Its RGB weight divides by the mixture density, not the selected channel PDF.
// Analytic Beer attenuation also belongs on every direct-light connection.
// Surface and volume events share the path depth budget and roulette policy.
//
// Path tracing is the first medium-aware integrator. Extending BDPT/MLT/VCM and
// photon methods requires volume vertices, distance measures and consistent MIS;
// applying a display fog overlay would not implement those transport methods.
// Temporal surface correspondence cannot currently validate volume paths.
// Existing meshes/lights suffice; heterogeneous grids require a separate model.
bool ValidateMediumOptions(Algorithm algorithm, const MediumOptions& medium, std::string& error)
{
    constexpr float MaximumAnisotropy = 0.99f;
    const auto finite = [](Float3 value) {
        constexpr float MaximumParameter = 1000000;
        return std::isfinite(value.X) && std::isfinite(value.Y) && std::isfinite(value.Z) &&
               std::abs(value.X) <= MaximumParameter && std::abs(value.Y) <= MaximumParameter &&
               std::abs(value.Z) <= MaximumParameter;
    };
    const auto positive = [](Float3 value) { return value.X >= 0 && value.Y >= 0 && value.Z >= 0; };
    if (!finite(medium.Minimum) || !finite(medium.Maximum) || medium.Minimum.X >= medium.Maximum.X ||
        medium.Minimum.Y >= medium.Maximum.Y || medium.Minimum.Z >= medium.Maximum.Z)
        error = "Medium bounds must be within +/-1000000 with positive extent on every axis";
    else if (!finite(medium.Absorption) || !finite(medium.Scattering) || !positive(medium.Absorption) ||
             !positive(medium.Scattering) || !finite(medium.Absorption + medium.Scattering))
        error = "Medium absorption, scattering and extinction must be in [0, 1000000] inverse world units";
    else if (!std::isfinite(medium.Anisotropy) || std::abs(medium.Anisotropy) > MaximumAnisotropy)
        error = "Medium anisotropy must be in [-0.99, 0.99]";
    else if (medium.Enabled && algorithm != Algorithm::PathTracing)
        error = "Participating media currently require the path tracing integrator";
    else
        return true;
    return false;
}
MediumData PackMedium(const MediumOptions& medium)
{
    return {medium.Minimum,    medium.Enabled ? 1u : 0u,
            medium.Maximum,    medium.Anisotropy,
            medium.Absorption, 0,
            medium.Scattering, 0};
}
} // namespace Arvo::RayTracing
