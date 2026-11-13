#pragma once
#include "Renderer/MotionData.h"
#include <array>

namespace Arvo::RayTracing::MetalInstances
{
bool BoundMotion(const MotionData::RigidMotion&, const std::array<double, 3>& minimum,
                 const std::array<double, 3>& maximum, std::array<double, 3>& sweptMinimum,
                 std::array<double, 3>& sweptMaximum);
}
