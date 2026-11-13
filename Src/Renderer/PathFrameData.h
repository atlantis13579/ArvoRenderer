#pragma once
#include "Renderer/ParticipatingMedia.h"

namespace Arvo::RayTracing
{
// Upload layout only. Medium and exposure share one CBV; the unchanged clock
// consumed by procedural DXR stages lives in a separate per-lane UAV.
struct alignas(16) PathFrameData
{
    MediumData Medium;
    MotionData::FrameExposure CameraExposure;
};
static_assert(sizeof(PathFrameData) == 144 && offsetof(PathFrameData, CameraExposure) == 64);
} // namespace Arvo::RayTracing
