#pragma once
#include "Renderer/MotionData.h"
#include <array>
namespace Arvo::RayTracing::ComputeInstances
{
bool SweptBounds(const MotionData::RigidMotion&, const std::array<double, 3>& minimum,
                 const std::array<double, 3>& maximum, std::array<double, 3>& resultMinimum,
                 std::array<double, 3>& resultMaximum);
}
