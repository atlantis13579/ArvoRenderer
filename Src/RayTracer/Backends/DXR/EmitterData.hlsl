#pragma once
// DXR source upload schema. Catalog slot, scene source index and virtual
// primitive ID are distinct. See Renderer/EmitterData.h for the host layout.
static const uint PackedEmitterDirectional = 0, PackedEmitterPoint = 1, PackedEmitterRectangle = 2;
static const uint PackedEmitterEnvironment = 3, PackedEmitterTriangle = 4, PackedEmitterSphere = 5;
struct PackedEmitter
{
    float3 Origin;
    uint Kind;
    float3 Direction;
    float Width;
    float3 Strength;
    float Height;
    float3 BackgroundTop;
    uint Primitive;
    float3 BackgroundBottom;
    uint SourceIndex;
    float3 AxisX;
    uint Instance;
    float3 AxisY;
    uint Flags;
    float3 AxisZ;
    uint Reserved;
};
