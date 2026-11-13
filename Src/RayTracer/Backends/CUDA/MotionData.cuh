#ifndef ARVO_CUDA_MOTION_DATA
#define ARVO_CUDA_MOTION_DATA
#include <cuda_runtime.h>
#include <float.h>
static constexpr unsigned MotionDeltaTime = 1, MotionInvalidSample = 2;
struct MotionShutter
{
    float Open, Close;
};
struct MotionTimeSample
{
    float Time, Density;
    unsigned Flags, Reserved;
};
struct MotionQuaternion
{
    float X, Y, Z, W;
};
struct MotionPose
{
    MotionQuaternion Rotation;
    float3 Translation;
    float Reserved;
};
struct MotionTrack
{
    MotionPose Start, End;
};
struct MotionExposure
{
    MotionTrack Motion;
    float Open, Close;
    unsigned Enabled, SceneEnabled;
};
static_assert(sizeof(MotionExposure) == 80 && offsetof(MotionExposure, Enabled) == 72,
              "Camera exposure layout");
struct MotionBounds
{
    float3 Minimum;
    unsigned Valid;
    float3 Maximum;
    unsigned Reserved;
};
static_assert(sizeof(MotionPose) == 32 && sizeof(MotionTrack) == 64 && sizeof(MotionBounds) == 32,
              "Motion layout");
#endif
