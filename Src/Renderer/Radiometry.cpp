#include "Renderer/RayTracing.h"

namespace Arvo::RayTracing
{
const char* RadiometryKey(RadiometryMode mode)
{
    switch (mode)
    {
    case RadiometryMode::RGB:
        return "rgb";
    case RadiometryMode::Spectral:
        return "spectral";
    default:
        return nullptr;
    }
}
bool ParseRadiometry(const std::string& key, RadiometryMode& mode)
{
    if (key == "rgb")
        mode = RadiometryMode::RGB;
    else if (key == "spectral")
        mode = RadiometryMode::Spectral;
    else
        return false;
    return true;
}
bool ValidateRadiometry(Algorithm integrator, RadiometryMode mode, std::string& error)
{
    if (mode != RadiometryMode::RGB && mode != RadiometryMode::Spectral)
    {
        error = "Unknown radiometry mode.";
        return false;
    }
    // Connected walks, photon caches and MLT primary samples must agree on
    // wavelength measure. They cannot inherit a camera path's spectral toggle.
    if (mode == RadiometryMode::Spectral && integrator != Algorithm::PathTracing)
    {
        error = "Spectral transport currently requires Path Tracing.";
        return false;
    }
    return true;
}
} // namespace Arvo::RayTracing
