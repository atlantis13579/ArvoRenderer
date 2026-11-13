#ifndef ARVO_DXR_PATH_TIME
#define ARVO_DXR_PATH_TIME
// Intersection shaders cannot read a ray payload. One lane owns its clock for
// the complete camera walk, including all visibility and callable traversals.
// Shader stages may run on different cores: globallycoherent plus a device
// barrier publishes the value before TraceRay/CallShader can consume it.
// DXR Functional Spec, "Execution and memory ordering":
// https://microsoft.github.io/DirectX-Specs/d3d/Raytracing.html#execution-and-memory-ordering
globallycoherent RWStructuredBuffer<float> PathTimes : register(u8);
uint PathTimeLane()
{
    const uint3 index = DispatchRaysIndex(), dimensions = DispatchRaysDimensions();
    return index.x + dimensions.x * (index.y + dimensions.y * index.z);
}
void BeginPathTime(float sceneTime)
{
    PathTimes[PathTimeLane()] = sceneTime;
    DeviceMemoryBarrier();
}
float ReadPathTime()
{
    return PathTimes[PathTimeLane()];
}
#endif
