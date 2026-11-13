#pragma once
#include "Renderer/MotionData.h"
#include <array>

namespace Arvo::RayTracing::CudaInstances
{
bool MotionBounds(const MotionData::RigidMotion&, const std::array<double, 3>& lower,
                  const std::array<double, 3>& upper, std::array<double, 3>& sweptLower,
                  std::array<double, 3>& sweptUpper);
}
