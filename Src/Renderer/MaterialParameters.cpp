#include "Renderer/RayTracing.h"
#include <cmath>

namespace Arvo::RayTracing
{
const char* MaterialModelKey(MaterialModel model)
{
    switch (model)
    {
    case MaterialModel::Legacy:
        return "legacy";
    case MaterialModel::Diffuse:
        return "diffuse";
    case MaterialModel::Conductor:
        return "conductor";
    case MaterialModel::Dielectric:
        return "dielectric";
    default:
        return nullptr;
    }
}
bool ParseMaterialModel(std::string_view key, MaterialModel& model)
{
    for (auto candidate :
         {MaterialModel::Legacy, MaterialModel::Diffuse, MaterialModel::Conductor, MaterialModel::Dielectric})
        if (key == MaterialModelKey(candidate))
        {
            model = candidate;
            return true;
        }
    return false;
}
// Configuration validation is shared; no backend's numerical BSDF is called here.
// Keeping these predicates independent of device creation lets editors and workers
// reject incompatible combinations before starting a render or changing caches.
bool ValidateMaterialScattering(const MaterialScattering& material, std::string& error)
{
    constexpr float MinimumOpticalIndex = 0.001f, MaximumOpticalConstant = 100;
    const auto inRange = [](float value, float minimum, float maximum) {
        return std::isfinite(value) && value >= minimum && value <= maximum;
    };
    const auto vectorInRange = [&](Float3 value, float minimum, float maximum) {
        return inRange(value.X, minimum, maximum) && inRange(value.Y, minimum, maximum) &&
               inRange(value.Z, minimum, maximum);
    };
    if (material.Model >= MaterialModel::Count)
        error = "Unknown material scattering model.";
    else if (!inRange(material.SubsurfaceMeanFreePath, 0, MaximumSubsurfaceMeanFreePath) ||
             (material.SubsurfaceMeanFreePath != 0 &&
              material.SubsurfaceMeanFreePath < MinimumSubsurfaceMeanFreePath) ||
             !inRange(material.SubsurfaceAlbedoScale, 0, 1) ||
             !inRange(material.SubsurfaceAnisotropy, -MaximumSubsurfaceAnisotropy, MaximumSubsurfaceAnisotropy))
        error = "Subsurface scattering requires mean free path zero or [0.001,10000], "
                "albedo scale in [0,1], and anisotropy in [-0.95,0.95].";
    else if (material.SubsurfaceMeanFreePath > 0 && material.Model != MaterialModel::Dielectric)
        error = "Subsurface scattering requires a dielectric boundary.";
    else if (!inRange(material.MicrofacetSlope, 0, 1) ||
             (material.MicrofacetSlope != 0 && material.MicrofacetSlope < MinimumMicrofacetSlope) ||
             !inRange(material.ExteriorIndex, MinimumOpticalIndex, MaximumOpticalConstant) ||
             !inRange(material.InteriorIndex, MinimumOpticalIndex, MaximumOpticalConstant) ||
             !vectorInRange(material.ConductorIndex, MinimumOpticalIndex, MaximumOpticalConstant) ||
             !vectorInRange(material.ConductorExtinction, 0, MaximumOpticalConstant))
        error = "Material scattering requires slope zero or [0.001,1], indices in [0.001,100], "
                "and extinction in [0,100].";
    else
        return true;
    return false;
}
bool ValidateMaterialSupport(const MaterialScattering& material, BackendType backend, Algorithm algorithm,
                             std::string& error)
{
    const bool subsurface = material.SubsurfaceMeanFreePath != 0;
    const bool pointGi = algorithm == Algorithm::PointBasedGlobalIllumination;
    if ((subsurface || pointGi || material.Model != MaterialModel::Legacy) &&
        !ValidateMaterialScattering(material, error))
        return false;

    const bool supported = SupportsIntegrator(backend, algorithm);
    if (subsurface && (!supported || algorithm == Algorithm::RayTracing))
    {
        error = algorithm == Algorithm::RayTracing
            ? "Whitted tracing has no interior scattering events in its direct/specular recursion."
            : "Subsurface random walks are unavailable for this backend and integrator.";
        return false;
    }
    if (pointGi)
    {
        // Legacy highlight/mirror fields are checked with the complete material.
        if (supported && (material.Model == MaterialModel::Diffuse || material.Model == MaterialModel::Legacy ||
                          (material.Model == MaterialModel::Dielectric && subsurface)))
            return true;
        error = "Point GI's view-independent surfel cache cannot represent this directional material.";
        return false;
    }
    if (material.Model == MaterialModel::Legacy)
        return true;
    if (!supported)
    {
        error = "Physical materials require CPU, CUDA, Compute Shader, DXR or Metal "
                "Whitted/distributed/path tracing, BDPT, MLT, photon mapping, progressive photon mapping, or "
                "VCM.";
        return false;
    }
    const bool matched =
        material.Model == MaterialModel::Dielectric && material.ExteriorIndex == material.InteriorIndex;
    if (algorithm == Algorithm::RayTracing && material.Model != MaterialModel::Diffuse &&
        material.MicrofacetSlope != 0 && !matched)
    {
        error = "Whitted requires diffuse or ideal specular materials; use a stochastic integrator for rough "
                "surfaces.";
        return false;
    }
    return true;
}
} // namespace Arvo::RayTracing
