#pragma once
#include "Renderer/MotionData.h"
#include <array>

namespace Arvo::RayTracing::Cpu
{
void InverseInstanceMotion(const MotionData::Pose&, std::array<double, 3>& origin,
                            std::array<double, 3>& direction);
}
