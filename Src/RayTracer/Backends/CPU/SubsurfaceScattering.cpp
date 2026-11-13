#include "RayTracer/Backends/CPU/SubsurfaceScattering.h"
#include <algorithm>
#include <cmath>

namespace Arvo::RayTracing::Cpu
{
// Chiang et al., "Practical and Controllable Subsurface Scattering for Production
// Path Tracing" (2016): follow actual interior collisions rather than projecting
// a diffusion lobe onto the entry surface. This implementation exposes physical
// mean free path / single-scattering albedo; it does not claim their artist remap.
// https://disneyanimation.com/publications/practical-and-controllable-subsurface-scattering-for-production-path-tracing/
// A scalar extinction proposal exactly cancels Beer attenuation, including the
// probability of reaching the boundary. A collision retains only scattering albedo.
namespace
{
constexpr double SubsurfacePi = 3.14159265358979323846;
constexpr float OpenUniformMaximum = 1.0f - 1.0f / float(1u << 24);
constexpr float BasisParallelCosine = 0.999f;
} // namespace
SubsurfaceFlight SampleSubsurfaceFlight(float meanFreePath, Float3 albedo, Float3 direction, float endpoint,
                                        float variate)
{
    SubsurfaceFlight result;
    result.Distance = endpoint;
    const double speed = std::sqrt(double(direction.X) * direction.X + double(direction.Y) * direction.Y +
                                   double(direction.Z) * direction.Z);
    const double distance =
        -std::log1p(-double(std::clamp(variate, 0.0f, OpenUniformMaximum))) * meanFreePath;
    if (distance < double(endpoint) * speed)
    {
        result.Distance = float(distance / speed);
        result.Weight = albedo;
        result.Scattered = true;
    }
    return result;
}
float SubsurfacePhase(float propagationCosine, float anisotropy)
{
    const double g = anisotropy, cosine = std::clamp(double(propagationCosine), -1.0, 1.0);
    const double denominator = 1 + g * g - 2 * g * cosine;
    return float((1 - g * g) / (4 * SubsurfacePi * denominator * std::sqrt(denominator)));
}
SubsurfaceDirection SampleSubsurfaceDirection(Float3 incoming, float anisotropy, Float2 variates,
                                              bool importance)
{
    const double uniformCosine = 2 * double(variates.X) - 1, g = anisotropy;
    double cosine = uniformCosine;
    if (importance)
    {
        // Algebraic HG inversion with its removable g=0 singularity cancelled.
        // This avoids a small-g threshold that would silently change the phase PDF.
        const double denominator = 1 + g * uniformCosine;
        const double square = uniformCosine * uniformCosine;
        cosine = ((1 + g * g) * uniformCosine + g / 2 * (square + 3 + g * g * (square - 1))) /
                 (denominator * denominator);
    }
    cosine = std::clamp(cosine, -1.0, 1.0);
    const double sine = std::sqrt(std::max(0.0, 1 - cosine * cosine));
    const double azimuth = 2 * SubsurfacePi * variates.Y;
    const Float3 axis = Normalize(incoming);
    const Float3 helper = std::abs(axis.Y) < BasisParallelCosine ? Float3(0, 1, 0) : Float3(1, 0, 0);
    const Float3 tangent = Normalize(Cross(helper, axis)), bitangent = Cross(axis, tangent);
    SubsurfaceDirection result;
    result.Direction = Normalize(axis * float(cosine) + tangent * float(sine * std::cos(azimuth)) +
                                 bitangent * float(sine * std::sin(azimuth)));
    const float phase = SubsurfacePhase(Dot(axis, result.Direction), anisotropy);
    result.Pdf = importance ? phase : float(1 / (4 * SubsurfacePi));
    result.Weight = phase / result.Pdf;
    return result;
}
} // namespace Arvo::RayTracing::Cpu
