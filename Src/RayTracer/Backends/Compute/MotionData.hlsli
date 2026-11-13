#ifndef ARVO_COMPUTE_MOTION_DATA
#define ARVO_COMPUTE_MOTION_DATA
static const uint MotionDeltaTime = 1, MotionInvalidSample = 2;
static const float MotionFloatEpsilon = 1.1920928955078125e-7f;
static const float MotionMinimumNormal = 1.1754943508222875e-38f;
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
    float3 Translation;
    float Reserved;
};
struct MotionTrack
{
    MotionPose Start, End;
};
// Flat shutter fields preserve the same 80-byte layout in CBVs and SRVs.
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
struct MotionBounds
{
    float3 Minimum;
    uint Valid;
    float3 Maximum;
    uint Reserved;
};
#endif
