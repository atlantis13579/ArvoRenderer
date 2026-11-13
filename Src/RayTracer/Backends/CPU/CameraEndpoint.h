#pragma once
#include "Renderer/RayTracing.h"

namespace Arvo::RayTracing::Cpu
{
struct CameraFrame;
enum CameraEndpointFlag : uint32_t
{
    CameraEndpointValid = 1,
    CameraDeltaPosition = 2,
    CameraDeltaDirection = 4
};
struct CameraEndpointEvaluation
{
    Float2 Raster;
    float Importance = 0, PositionPdf = 0, DirectionPdf = 0;
    uint32_t Flags = 0;
};
struct CameraEndpointSample
{
    Float3 Origin, Direction;
    Float2 LensPoint, Raster;
    float Importance = 0, PositionPdf = 0, DirectionPdf = 0, Weight = 0;
    uint32_t Flags = 0;
};
struct CameraConnectionSample
{
    Float3 Position, Direction; // Direction from the scene reference toward the lens.
    Float2 LensPoint, Raster;
    float Distance = 0, Weight = 0, PositionPdf = 0, DirectionPdf = 0;
    uint32_t Flags = 0;
};
// Frame position is finite and axes are orthonormal. Raster covers [0,1)^2
// over the FULL image, not a tile/pixel. Importance preserves aperture averaging.
// Densities/response must be finite and representable as float.
Float2 CameraPixelRaster(uint32_t x, uint32_t y, uint32_t width, uint32_t height, Float2 jitter);
CameraEndpointSample SampleCameraEndpoint(const CameraFrame&, Float2 raster, Float2 lensVariates);
// The 2-D lens point is in world units along camera Right/Up and is already on
// aperture support (zero for a pinhole). Direction is unit, FROM camera to scene.
// Queries report zero continuous position PDF for a pinhole; samples store mass 1.
CameraEndpointEvaluation EvaluateCameraEndpoint(const CameraFrame&, Float2 lensPoint, Float3 direction);
// Visibility and the scene vertex's BSDF/cosine are external. Weight includes
// camera response, lens cosine and inverse squared distance / sampled lens PDF.
// A finite lens has a solid-angle DirectionPdf; a pinhole sample has mass 1 and
// CameraDeltaDirection. Do not compare these measures without their flags.
CameraConnectionSample SampleCameraConnection(const CameraFrame&, Float3 reference, Float2 lensVariates);
} // namespace Arvo::RayTracing::Cpu
