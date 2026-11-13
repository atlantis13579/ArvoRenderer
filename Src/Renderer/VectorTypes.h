#pragma once

namespace Arvo::RayTracing
{
struct Float2
{
    float X = 0.0f;
    float Y = 0.0f;

    Float2() = default;
    Float2(float x, float y) : X(x), Y(y) {}
};

struct Float3
{
    float X = 0.0f;
    float Y = 0.0f;
    float Z = 0.0f;

    Float3() = default;
    Float3(float x, float y, float z) : X(x), Y(y), Z(z) {}

    Float3 operator-() const { return Float3(-X, -Y, -Z); }
    Float3 operator+(const Float3& rhs) const { return Float3(X + rhs.X, Y + rhs.Y, Z + rhs.Z); }
    Float3 operator-(const Float3& rhs) const { return Float3(X - rhs.X, Y - rhs.Y, Z - rhs.Z); }
    Float3 operator*(float scalar) const { return Float3(X * scalar, Y * scalar, Z * scalar); }
    Float3 operator/(float scalar) const { return Float3(X / scalar, Y / scalar, Z / scalar); }
};

inline Float3 operator*(float scalar, const Float3& value)
{
    return value * scalar;
}
} // namespace Arvo::RayTracing
