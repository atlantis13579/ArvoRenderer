#pragma once
// packed_float3 is essential: ordinary MSL float3 would not fit beside the
// scalar in each 16-byte lane. This is the Renderer/EmitterData.h upload ABI.
constant uint PackedEmitterDirectional = 0, PackedEmitterPoint = 1, PackedEmitterRectangle = 2;
constant uint PackedEmitterEnvironment = 3, PackedEmitterTriangle = 4, PackedEmitterSphere = 5;
struct PackedEmitter
{
    packed_float3 Origin;
    uint Kind;
    packed_float3 Direction;
    float Width;
    packed_float3 Strength;
    float Height;
    packed_float3 BackgroundTop;
    uint Primitive;
    packed_float3 BackgroundBottom;
    uint SourceIndex;
    packed_float3 AxisX;
    uint Instance;
    packed_float3 AxisY;
    uint Flags;
    packed_float3 AxisZ;
    uint Reserved;
};
