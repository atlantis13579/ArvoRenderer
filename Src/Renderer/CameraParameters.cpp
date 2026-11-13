#include "Renderer/RayTracing.h"
#include <cmath>

namespace Arvo::RayTracing
{
bool ValidateCameraLens(const CameraLens& lens, std::string& error)
{
    constexpr float MaximumCameraDistance = 1000000;
    if (!std::isfinite(lens.Radius) || !std::isfinite(lens.FocusDistance) || lens.Radius < 0 ||
        lens.Radius > MaximumCameraDistance || lens.FocusDistance < MinimumFocusDistance ||
        lens.FocusDistance > MaximumCameraDistance)
    {
        error = "Lens radius must be in [0,1000000] and focus distance in [0.001,1000000] world units.";
        return false;
    }
    return true;
}
} // namespace Arvo::RayTracing
