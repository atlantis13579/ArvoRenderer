// Composition only: native interpolation, rigid ray mapping and two-level traversal.
#include "MotionInterpolation.hlsl"
#include "MotionInstancing.hlsl"
float InstanceSceneTime()
{
#if defined(ARVO_DXR_PATH_TIME)
    return ReadPathTime();
#else
    return 0;
#endif
}
#include "Instancing.hlsl"
