#include "MotionData.h"
// MSL has no binary64. Rotate the scaled expansions directly so a transient
// object-space direction outside float range survives until ray conditioning.
void InverseMotionCoordinates(MotionPose pose, thread WideMath::Number* value)
{
    const WideMath::Number q[3] = {WideMath::FromFloat(-pose.Rotation.X),
        WideMath::FromFloat(-pose.Rotation.Y), WideMath::FromFloat(-pose.Rotation.Z)};
    WideMath::Number tangent[3];
    const auto twice = WideMath::FromFloat(2);
    const auto w = WideMath::FromFloat(pose.Rotation.W);
    for (uint component = 0; component < 3; ++component)
    {
        const uint next = (component + 1) % 3, last = (component + 2) % 3;
        tangent[component] = WideMath::Multiply(twice, WideMath::Subtract(
            WideMath::Multiply(q[next], value[last]), WideMath::Multiply(q[last], value[next])));
    }
    for (uint component = 0; component < 3; ++component)
    {
        const uint next = (component + 1) % 3, last = (component + 2) % 3;
        value[component] = WideMath::Add(value[component], WideMath::Add(
            WideMath::Multiply(w, tangent[component]), WideMath::Subtract(
                WideMath::Multiply(q[next], tangent[last]), WideMath::Multiply(q[last], tangent[next]))));
    }
}
void InverseInstanceMotion(MotionPose pose, thread WideMath::Number* origin, thread WideMath::Number* direction)
{
    const float3 translation = float3(pose.Translation);
    for (uint coordinate = 0; coordinate < 3; ++coordinate)
        origin[coordinate] = WideMath::Subtract(origin[coordinate], WideMath::FromFloat(translation[coordinate]));
    InverseMotionCoordinates(pose, origin);
    InverseMotionCoordinates(pose, direction);
}
