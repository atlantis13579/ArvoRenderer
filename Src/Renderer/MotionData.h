#pragma once
#include "VectorTypes.h"
#include <cstdint>

namespace Arvo::RayTracing::MotionData
{
// Data contracts only. Keyframes use normalized frame time [0,1]; a shutter
// selects a subinterval. A collapsed shutter is a discrete observation.
// Nonzero shutter endpoints/width and quaternion peak must be normal floats:
// this avoids CPU/GPU disagreement when shader arithmetic flushes denormals.
inline constexpr uint32_t DeltaTime = 1, InvalidSample = 2;
struct Shutter
{
    float Open = 0, Close = 0;
};
struct TimeSample
{
    float Time = 0, Density = 0;
    uint32_t Flags = InvalidSample, Reserved = 0;
};
struct Quaternion
{
    float X = 0, Y = 0, Z = 0, W = 1;
};
struct Pose
{
    Quaternion Rotation;
    Float3 Translation;
    float Reserved = 0;
};
struct Track
{
    Pose Start, End;
};
struct RigidMotion
{
    // Instance-local rigid motion precedes the fixed ObjectToWorld affine map.
    // Rotation is about the object origin; translation is in object units.
    Track Transform;
    uint32_t Enabled = 0;
    uint32_t Reserved[3]{};
};
struct Exposure
{
    // Camera pose deltas use world axes, rotating about the base camera origin.
    // Translation moves that origin; the whole lens frame rotates rigidly.
    Track Motion;
    Shutter Interval;
    uint32_t Enabled = 0, Reserved = 0;
    bool operator==(const Exposure& other) const
    {
        const auto samePose = [](const Pose& a, const Pose& b) {
            return a.Rotation.X == b.Rotation.X && a.Rotation.Y == b.Rotation.Y &&
                   a.Rotation.Z == b.Rotation.Z && a.Rotation.W == b.Rotation.W &&
                   a.Translation.X == b.Translation.X && a.Translation.Y == b.Translation.Y &&
                   a.Translation.Z == b.Translation.Z;
        };
        return Enabled == other.Enabled && Interval.Open == other.Interval.Open &&
               Interval.Close == other.Interval.Close && samePose(Motion.Start, other.Motion.Start) &&
               samePose(Motion.End, other.Motion.End);
    }
    bool operator!=(const Exposure& other) const { return !(*this == other); }
};
struct Bounds
{
    Float3 Minimum;
    uint32_t Valid = 1;
    Float3 Maximum;
    uint32_t Reserved = 0;
};
// Upload packet only. SceneEnabled is derived from the instance graph, never
// serialized as a user camera option. A stationary camera can expose motion.
struct FrameExposure
{
    Track Motion;
    Shutter Interval;
    uint32_t Enabled = 0, SceneEnabled = 0;
    FrameExposure() = default;
    FrameExposure(const Exposure& camera, bool sceneEnabled = false)
        : Motion(camera.Motion), Interval(camera.Interval), Enabled(camera.Enabled),
          SceneEnabled(sceneEnabled ? 1u : 0u) {}
};
static_assert(sizeof(TimeSample) == 16 && sizeof(Pose) == 32 && sizeof(Track) == 64 && sizeof(Bounds) == 32);
static_assert(sizeof(Exposure) == 80);
static_assert(sizeof(RigidMotion) == 80);
static_assert(sizeof(FrameExposure) == sizeof(Exposure));
} // namespace Arvo::RayTracing::MotionData
