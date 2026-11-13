#pragma once
#include "Renderer/MotionData.h"
#include "Renderer/RayTracing.h"

namespace Arvo::RayTracing::Cpu
{
Ray SampleCameraMotion(const Ray&, Float3 pivot, const MotionData::Exposure&, float variate,
                       float* sceneTime = nullptr, bool sceneMotion = false);
bool ValidShutter(MotionData::Shutter);
MotionData::TimeSample SampleShutter(MotionData::Shutter, float variate);
bool ValidMotionTrack(const MotionData::Track&);
Float3 InterpolateMotionPoint(Float3 start, Float3 end, float time);
MotionData::Pose InterpolateMotion(const MotionData::Track&, float time);
Float3 TransformMotionPoint(const MotionData::Pose&, Float3);
Float3 TransformMotionVector(const MotionData::Pose&, Float3);
Float3 InverseMotionPoint(const MotionData::Pose&, Float3);
Float3 InverseMotionVector(const MotionData::Pose&, Float3);
MotionData::Bounds LinearMotionBounds(const MotionData::Bounds&, const MotionData::Bounds&,
                                      MotionData::Shutter);
MotionData::Bounds RigidMotionBounds(const MotionData::Bounds&, const MotionData::Track&,
                                     MotionData::Shutter);
} // namespace Arvo::RayTracing::Cpu
