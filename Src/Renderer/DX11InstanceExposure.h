#pragma once
#include "Maths/Matrix4.h"
#include "Renderer/MotionData.h"

namespace Arvo::Renderer
{
Maths::Matrix4 InstanceAtExposureMidpoint(const Maths::Matrix4& fixed,
                                          const RayTracing::MotionData::RigidMotion&,
                                          RayTracing::MotionData::Shutter);
}
