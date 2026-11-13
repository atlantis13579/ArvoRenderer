#include "RayTracer/Backends/CPU/CameraEndpoint.h"
#include "RayTracer/Backends/CPU/CpuTransport.h"
#include <algorithm>
#include <cmath>

namespace Arvo::RayTracing::Cpu
{
// PBRT 3e, "The Path-Space Measurement Equation", section 16.1.1:
// https://pbr-book.org/3ed-2018/Light_Transport_III_Bidirectional_Methods/The_Path-Space_Measurement_Equation
// Full-film uniform sampling gives p(w)=1/(A*cos^3), A=4*halfWidth*halfHeight
// on the unit-depth image plane. Normalized importance is p(lens)*p(w)/cos;
// hence camera-path throughput is one, including a finite thin-lens aperture.
// Pixel-conditioned PDFs multiply p(w) by pixel count; light-tracing splats
// need the corresponding film normalization. Neither belongs in this leaf.
namespace
{
constexpr float CameraPi = 3.14159265358979323846f;
bool UnitSquare(Float2 v)
{
    return v.X >= 0 && v.X < 1 && v.Y >= 0 && v.Y < 1;
}
float LensDensity(const CameraFrame& frame)
{
    if (!(frame.HalfWidth > 0 && frame.HalfHeight > 0 && frame.Lens.Radius >= 0))
        return 0;
    if (frame.Lens.Radius == 0)
        return 1;
    if (!(frame.Lens.FocusDistance > 0))
        return 0;
    float pdf = 1 / (CameraPi * frame.Lens.Radius * frame.Lens.Radius);
    return pdf > 0 && std::isfinite(pdf) ? pdf : 0;
}
float Norm(Float3 v)
{
    float peak = std::max(std::abs(v.X), std::max(std::abs(v.Y), std::abs(v.Z)));
    if (!(peak > 0) || !std::isfinite(peak))
        return 0;
    Float3 scaled = v / peak;
    return peak * std::sqrt(Dot(scaled, scaled));
}
Float2 LensPoint(const CameraFrame& camera, Float2 u)
{
    float x = 2 * u.X - 1, y = 2 * u.Y - 1;
    if (camera.Lens.Radius == 0 || (x == 0 && y == 0))
        return {};
    bool horizontal = std::abs(x) > std::abs(y);
    float radius = horizontal ? x : y;
    float angle = horizontal ? (CameraPi / 4) * (y / x) : CameraPi / 2 - (CameraPi / 4) * (x / y);
    return {camera.Lens.Radius * radius * std::cos(angle), camera.Lens.Radius * radius * std::sin(angle)};
}
CameraEndpointEvaluation Response(const CameraFrame& camera, Float2 raster, float cosine)
{
    CameraEndpointEvaluation result;
    float position = LensDensity(camera);
    if (!(position > 0 && cosine > 0))
        return result;
    float imageArea = 4 * camera.HalfWidth * camera.HalfHeight;
    float direction = 1 / (imageArea * cosine * cosine * cosine),
          importance = (position * direction) / cosine;
    if (!(direction > 0 && importance > 0) || !std::isfinite(direction) || !std::isfinite(importance))
        return result;
    result.Raster = raster;
    result.Importance = importance;
    result.DirectionPdf = direction;
    result.PositionPdf = camera.Lens.Radius == 0 ? 0 : position;
    result.Flags = CameraEndpointValid | (camera.Lens.Radius == 0 ? CameraDeltaPosition : 0);
    return result;
}
} // namespace
Float2 CameraPixelRaster(uint32_t x, uint32_t y, uint32_t width, uint32_t height, Float2 jitter)
{
    if (x >= width || y >= height || !UnitSquare(jitter)) return {-1, -1};
    // Conversion of a valid final-pixel sample can round to 1. Keep generated
    // coordinates inside [0,1), while external endpoint queries stay strict.
    const float upper = std::nextafter(1.0f, 0.0f);
    return {std::min(float((double(x) + jitter.X) / width), upper),
            std::min(float((double(y) + jitter.Y) / height), upper)};
}
CameraEndpointEvaluation EvaluateCameraEndpoint(const CameraFrame& camera, Float2 lens, Float3 direction)
{
    float cosine = Dot(camera.Forward, direction);
    if (!(cosine > 0) || !(LensDensity(camera) > 0))
        return {};
    float focus = camera.Lens.Radius == 0 ? 1 : camera.Lens.FocusDistance;
    float horizontal = lens.X / focus + Dot(direction, camera.Right) / cosine;
    float vertical = lens.Y / focus + Dot(direction, camera.Up) / cosine;
    Float2 raster{0.5f + horizontal / (2 * camera.HalfWidth), 0.5f - vertical / (2 * camera.HalfHeight)};
    if (!UnitSquare(raster))
        return {};
    return Response(camera, raster, cosine);
}
CameraEndpointSample SampleCameraEndpoint(const CameraFrame& camera, Float2 raster, Float2 variates)
{
    if (!UnitSquare(raster) || !UnitSquare(variates) || !(LensDensity(camera) > 0))
        return {};
    Float2 lens = LensPoint(camera, variates);
    Float3 offset = camera.Right * lens.X + camera.Up * lens.Y;
    Float3 axis = camera.Forward + camera.Right * ((2 * raster.X - 1) * camera.HalfWidth) +
                  camera.Up * ((1 - 2 * raster.Y) * camera.HalfHeight);
    Float3 towardScene = camera.Lens.Radius == 0 ? axis : axis * camera.Lens.FocusDistance - offset;
    float length = Norm(towardScene);
    if (!(length > 0) || !std::isfinite(length))
        return {};
    Float3 direction = towardScene / length;
    auto response = Response(camera, raster, Dot(camera.Forward, direction));
    if (!response.Flags)
        return {};
    // Retain the input raster at exact image edges instead of round-tripping
    // through a world-space ray and potentially clipping it by one float ULP.
    return {camera.Position + offset,
            direction,
            lens,
            raster,
            response.Importance,
            camera.Lens.Radius == 0 ? 1 : response.PositionPdf,
            response.DirectionPdf,
            1,
            response.Flags};
}
CameraConnectionSample SampleCameraConnection(const CameraFrame& camera, Float3 reference, Float2 variates)
{
    if (!UnitSquare(variates) || !(LensDensity(camera) > 0))
        return {};
    Float2 lens = LensPoint(camera, variates);
    Float3 offset = camera.Right * lens.X + camera.Up * lens.Y;
    Float3 segment = (reference - camera.Position) - offset;
    float distance = Norm(segment);
    if (!(distance > 0) || !std::isfinite(distance))
        return {};
    Float3 outward = segment / distance;
    auto response = EvaluateCameraEndpoint(camera, lens, outward);
    if (!response.Flags)
        return {};
    bool pinhole = camera.Lens.Radius == 0;
    float cosine = Dot(camera.Forward, outward);
    float directionPdf = pinhole ? 1 : (response.PositionPdf * distance) * (distance / cosine);
    // For finite apertures this equals We/p(wi). For a pinhole, include the
    // lens geometry factor explicitly and keep the sampled delta mass at one.
    float weight = (response.DirectionPdf / distance) / distance;
    if (!(directionPdf > 0 && weight > 0) || !std::isfinite(directionPdf) || !std::isfinite(weight))
        return {};
    return {camera.Position + offset,
            -outward,
            lens,
            response.Raster,
            distance,
            weight,
            pinhole ? 1 : response.PositionPdf,
            directionPdf,
            response.Flags | (pinhole ? CameraDeltaDirection : 0)};
}
} // namespace Arvo::RayTracing::Cpu
