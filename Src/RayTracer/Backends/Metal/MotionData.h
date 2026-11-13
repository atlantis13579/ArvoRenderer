#ifndef ARVO_METAL_MOTION_DATA
#define ARVO_METAL_MOTION_DATA
#include <metal_stdlib>
using namespace metal;
constant uint MotionDeltaTime = 1, MotionInvalidSample = 2;
constant float MotionFloatEpsilon = 1.1920928955078125e-7f;
constant float MotionMinimumNormal = 1.1754943508222875e-38f;
struct MotionShutter
{
    float Open, Close;
};
struct MotionTimeSample
{
    float Time, Density;
    uint Flags, Reserved;
};
struct MotionQuaternion
{
    float X, Y, Z, W;
};
struct MotionPose
{
    MotionQuaternion Rotation;
    packed_float3 Translation;
    float Reserved;
};
struct MotionTrack
{
    MotionPose Start, End;
};
struct MotionRigid
{
    MotionTrack Transform;
    uint Enabled;
    uint Reserved[3];
};
struct MotionExposure
{
    MotionTrack Motion;
    float Open, Close;
    uint Enabled, SceneEnabled;
};
static_assert(sizeof(MotionExposure) == 80, "Camera exposure layout");
struct MotionBounds
{
    packed_float3 Minimum;
    uint Valid;
    packed_float3 Maximum;
    uint Reserved;
};
static_assert(sizeof(MotionPose) == 32 && sizeof(MotionTrack) == 64 && sizeof(MotionBounds) == 32,
              "Motion layout");
#endif
