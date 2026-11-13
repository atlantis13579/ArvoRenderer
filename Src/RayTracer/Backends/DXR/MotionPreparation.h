#pragma once
#include "Renderer/MotionData.h"
#include <array>

namespace Arvo::RayTracing::DxrInstances
{
bool SweptLocalBounds(const MotionData::RigidMotion&, const std::array<double, 3>& low,
                      const std::array<double, 3>& high, std::array<double, 3>& resultLow,
                      std::array<double, 3>& resultHigh);
}
